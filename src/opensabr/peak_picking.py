"""Methods for picking and labeling peaks in the ABR waveform

The basic idea is to trace peaks over adjacent levels using a "ridge-tracing"
algorithm on the heatmap. A ridge is a peak that is traced over sound levels.
Once traced, these ridges are labeled with canonical wave names based on their
latency and slope.
The ridge-tracing algorithm will likely work well on other datasets, but the
parameters of the labeling algorithm are tightly tuned to our data and would
have to be adjusted for another dataset.

User functions:
trace_ridges - Finds ridges in a labeled heatmap
label_ridges - Labels ridges with canonical wave names

Helper functions:
_find_peaks_max_prominence - Wrapper around find_peaks for prominence filter
_masked_and_padded_hungarian - Wrapper around linear_sum_assignment with masking
"""

import pandas
import numpy as np
import scipy

# Each traced ridge will traverse peaks of at least this prominence (uV)
# Lower is more sensitive but risks bringing in noise
# Any more than 0.15 and we cannot detect the smallest W4
# With the standard find_peaks function on min prominence, this had to be
# quite small to catch W4 (eg 0.15) at which point it also tended to pick up
# a lot of noise. With the adjusted max_prominence method it can be much higher
# and still detect W1 and W4 (e.g., 0.5)
min_prominence = 0.4

# How far to search for "bases" when calculating prominence. This is the 
# total window size in samples.
# This should be no more than 1.5x of an ABR period. 1x should barely capture
# the adjacent troughs, but round up a little in case of frequency skew.
# The default will search forever, which will greatly overestimate prominence
# for a narrowband signal like this one, leading to noise peaks especially
# before t=0.
find_peaks_wlen = 17

# Ridge tracing will start at this level
# Choices are 'middle', 'loudest', and an integer with 0 being softest
start_level = 'loudest'

# This is how far a peak can shift across one level and still be in the 
# same traced ridge. Units: samples
# Higher might cause jumping; lower might cause legitimate ridges to
# terminate early
# Any less than 5 and we can't reliably detect W4
max_shift = 5

# This is how apart peaks in the `start_level` can be
# Lower pulls out more noise ridges; higher might skip over real ridges
# that are too close together
min_seed_spacing = 5

# The maximum cost of a wave assignment
# Lower is stricter (more unlabeled waves); higher might allow false labels
max_cost_wave_assign = 0.15

# Assign priority waves first - Can be helpful to ensure we get W1 and W4
priority_wave_list = [] #['W1', 'W4']


def trace_ridges(abr_2d, min_prominence=min_prominence, start_level=start_level,
    max_shift=max_shift, min_seed_spacing=min_seed_spacing):
    """Trace ridges across levels from seed peaks at `start level`.

    Arguments
    - abr_2d : DataFrame. ABR data over levels and time.
        index: sound level (dB), sorted ascending
        columns: timepoint
        values: ABR voltage (uV)
    - min_prominence : min peak prominence (uV) to be included in a ridge
    - start_level : str in {'middle', 'loudest'} or integer
        Which level to seed from. If 'middle', take the middle row; if
        'loudest', take the last row; otherwise this is an integer row index.
        Ridges can only be created at the start level,
        so this should be a level where all ridges of interest are clearly
        visible and distinguishable. While ridges are most visible at the 
        loudest level, they can also become less distinguishable (ie, become
        a shoulder peak below `min_prominence`) at that level.
    - max_shift : int
        The max temporal shift between levels (in samples)
        Set this to be around half an ABR period
    - min_seed_spacing : Int
        Initial seeds must be at least this far apart
        This avoids creating noise ridges that compete for peaks
    
    Workflow
    - Seeds one ridge per peak at the start level. This sets the total number
      of riges -- no new ridges can be created later.
    - Iterate upwards and then downwards from the start level, one level
      at a time.
    - At each step, find peaks, and assign them to existing ridges by
      minimizing the total latency shift across ridges using the Hungarian
      algorithm. No shift can exceed `max_shift`. 
    - A ridge terminates when it is not assigned a peak.

    Unexpected failures often occur in the assignment step, where ridges
    compete for peaks. For instance, when there are no low-cost choices, the
    Hungarian algorithm may assign a bad peak that is needed by another ridge,
    causing cascading mis-assignments. This scenario can sometimes be avoided
    by increasing `min_prominence` to avoid creating noise ridges.

    Returns: DataFrame
        index: MultiIndex, n_ridge * row_index
            'n_ridge': integer label for this ridge
            'row_idx': integer index into rows of `abr_2d`
        columns: 
            'level': abr_2d.index[row_idx]
            'col_idx': integer index int columns of `abr_2d`
            'timepoint': abr_2d.columns[col_idx]
            'amplitude': amplitude (prominence) of the peak
    """
    # Sort ascending
    if not abr_2d.index.is_monotonic_increasing:
        # Warn because row_idx will be unexpected
        print('warning: ABR data was not sorted by level, resorting')
        abr_2d = abr_2d.sort_index()

    # Choose the start_row_idx
    if start_level == 'middle':
        start_row_idx = len(abr_2d) // 2
    elif start_level == 'loudest':
        start_row_idx = len(abr_2d) - 1
    else:
        start_row_idx = start_level
    
    # Error check
    if start_row_idx not in range(len(abr_2d)):
        raise ValueError(
            f'could not form start_row_idx from start_level: {start_level}')

    # Find the seed peaks at `start_level` that initalize each ridge
    # The distance criterion avoids noise ridges competing for peaks
    sig = abr_2d.iloc[start_row_idx].values
    #~ seed_peak_idxs1, seed_peak_data1 = scipy.signal.find_peaks(
        #~ sig, 
        #~ prominence=min_prominence, 
        #~ distance=min_seed_spacing,
        #~ )
    seed_peak_idxs, seed_peak_data = _find_peaks_max_prominence(
        sig, 
        prominence=min_prominence,
        wlen=find_peaks_wlen,
        distance=min_seed_spacing,
        )
    
    # Store ridges points in `ridge_points_l` as they are discovered
    # Begin by storing the peak that seeds each ridge
    ridge_points_l = []
    for n_seed_peak, seed_peak_idx in enumerate(seed_peak_idxs):
        
        # Get prom for this ridge
        prom = seed_peak_data['prominences'][n_seed_peak]
        
        # Form a dict for this ridge
        ridge_dict = {
            'n_ridge': n_seed_peak,
            'row_idx': start_row_idx,
            'col_idx': seed_peak_idx,
            'size': prom,
            }
        
        # Store
        ridge_points_l.append(ridge_dict)
    
    # The number of start ridges is the maximum number 
    # (Ridges cannot be created later in this version)
    n_ridges = len(ridge_points_l)

    # Trace outward from start level: louder (step +1) and then softer (step -1)
    for step in (1, -1):

        # This Series maintains a memory of all currently active ridges
        # It's initalized to all ridges found at the start level, and ridges
        # drop out of this series as they terminate
        # Index: ridge number. Values: current location (column index) of ridge.
        active_ridges = pandas.Series(seed_peak_idxs)
        active_ridges.index.name = 'n_ridge'
        active_ridges.name = 'prev_peak_col'
        
        # This iteration variable keeps track of current row in while loop
        row_idx = start_row_idx

        # Step one level at a time, breaking when we reach either extreme
        while True:
            
            # Iterate and break when extreme is reached
            row_idx += step
            if not 0 <= row_idx < len(abr_2d):
                break
            
            # Find all peaks at this level
            # Unlike the seed (start) level, here we allow peaks to be
            # arbritrarily close together. Some will be noise. 
            sig = abr_2d.iloc[row_idx].values
            #~ level_peak_idxs, level_peak_data = scipy.signal.find_peaks(
                #~ sig, prominence=min_prominence)
            level_peak_idxs, level_peak_data = _find_peaks_max_prominence(
                sig, prominence=min_prominence, wlen=find_peaks_wlen)

            # Break if there are no peaks or no active ridges
            if len(level_peak_idxs) == 0 or len(active_ridges) == 0:
                break
            
            # Compute distance from each peak to the nearest active ridge
            cost = np.abs(
                active_ridges.values[:, None] - level_peak_idxs[None, :])

            # Pre-filter step: Try to make the cost matrix more well-behaved
            # by including only peaks near ridges and ridges near peaks
            # The risk is that a noise ridge has a high cost for all peaks,
            # but just takes one anyway, thus shifting everybody else off
            # A one pass pre-filter isn't guaranteed to converge
            peak_mask = cost.min(axis=0) <= max_shift
            ridge_mask = cost.min(axis=1) <= max_shift
            level_peak_idxs = level_peak_idxs[peak_mask]
            level_proms = level_peak_data['prominences'][peak_mask]
            active_ridges = active_ridges.loc[ridge_mask]

            # Break if there are no peaks or no active ridges
            if len(level_peak_idxs) == 0 or len(active_ridges) == 0:
                break

            # Define the assignment cost as the |shift| between each active 
            # ridge and each newly discovered peak
            # The rows are the previous peaks and the cols are the current peaks
            cost = np.abs(
                active_ridges.values[:, None] - level_peak_idxs[None, :])
            
            # Inflate costs above max shift to try to prevent them being 
            # selected by the Hungarian
            cost[cost > max_shift] = (
                cost[cost > max_shift] + 
                (cost[cost > max_shift] - max_shift) ** 2
                )
            
            # Assign ridges to peaks, one-to-one, minimizing total shift
            # These are indices within `active_ridges` and `level_peak_idxs`
            assigned_ridges, assigned_peaks = (
                scipy.optimize.linear_sum_assignment(cost))

            # Apply each assignment, keeping track of which ridges are active
            still_active = {}
            for assigned_ridge, assigned_peak in zip(
                    assigned_ridges, assigned_peaks):

                # Skip assignments that jump too far
                # This is the tricky and fragile bit. Other assignments might 
                # differ if the Hungarian knew this one would be dropped. 
                # The pre-filtering and cost-scaling above makes this less 
                # likely but not impossible.
                if cost[assigned_ridge, assigned_peak] > max_shift:
                    continue

                # Store
                ridge_points_l.append({
                    'row_idx': row_idx,
                    'n_ridge': active_ridges.index[assigned_ridge],
                    'col_idx': level_peak_idxs[assigned_peak],
                    'size': level_proms[assigned_peak],
                    })

                # Keep track of ridges that are still active
                still_active[
                    active_ridges.index[assigned_ridge]
                    ] = level_peak_idxs[assigned_peak]

            # Update memory of active ridges
            active_ridges = pandas.Series(still_active)

    # Data Frame
    res = pandas.DataFrame(ridge_points_l)
    
    # Use row_idx and col_idx to index into abr_2d
    if len(res) > 0:
        res['timepoint'] = abr_2d.columns[res['col_idx']]
        res['level'] = abr_2d.index[res['row_idx']]
        res = res.set_index(['n_ridge', 'level']).sort_index()

    # Return
    return res


def label_ridges(recording_coefs, wave_centroids, 
    max_cost=max_cost_wave_assign, slope_downweighting=30):
    """Assign each ridge to a wave centroid in (slope, intercept) space.

    Arguments
    - recording_coefs : DataFrame, one row per ridge
        index: n_ridge
        columns: ['slope_us_per_db', 'latency_ms_at_ref_level']
    - centroids : DataFrame, one row per labeled wave
        index: wave_name
        columns: same as recording_coefs
    - max_cost : reject assignments above this distance (normalized units).
    - slope_downweighting : downweight slope's cost by this much

    Work flow
    - Standardize the weighting of slope_us_per_db by dividing by 10 and
      of latency_ms_at_ref_level by dividing by 4. This leaves all costs as
      order ~1.
    - Further downweight slope by 100 because it is so noisy
    - Define distance between all ridges and all centroids (waves)
    - Drop ridges that are too far (above `max_cost`) from any wave
    - Assign the "priority" waves W1 and W4 first, because these tend to be
      the most consistently pulled out.
    - Assign all remaining waves to all remaining ridges next

    This two-stage assignment process hopefully stops a mediocre ridge 
    taking the W1 or W4 label.

    Returns: DataFrame, one row per assigned ridge.
        columns: ['wave_name', 'n_ridge', 'cost',
            'slope_us_per_db', 'latency_ms_at_ref_level']
    """

    # Error check
    assert wave_centroids.columns.equals(recording_coefs.columns)

    # Copy before mutating; keep originals for the final join
    orig_recording_coefs = recording_coefs.copy()
    recording_coefs = recording_coefs.copy()
    wave_centroids = wave_centroids.copy()

    # Normalize to approx unit scale
    recording_coefs['slope_us_per_db'] /= 10
    recording_coefs['latency_ms_at_ref_level'] /= 4
    wave_centroids['slope_us_per_db'] /= 10
    wave_centroids['latency_ms_at_ref_level'] /= 4

    # Further downweight slope because it is noisy
    recording_coefs['slope_us_per_db'] /= slope_downweighting
    wave_centroids['slope_us_per_db'] /= slope_downweighting

    # Define the cost as the distance between each ridge and each centroid
    cost = np.linalg.norm(
        recording_coefs.values[:, None, :] - wave_centroids.values[None, :, :],
        axis=2)

    # Drop ridges with no centroid within max_cost
    # This prevents them from contributing impossible constraints to the 
    # linear sum assignment
    ridge_keep = cost.min(axis=1) <= max_cost
    cost = cost[ridge_keep]
    recording_coefs = recording_coefs.loc[ridge_keep]
    
    # Stage 1: Assign priority waves W1/W4
    # Use only priority waves, but allow all ridges
    priority_wave_mask = wave_centroids.index.isin(priority_wave_list)
    assignments1 = _masked_and_padded_hungarian(
        cost=cost, 
        row_mask=np.ones(cost.shape[0], dtype=bool), 
        col_mask=priority_wave_mask, 
        max_cost=max_cost)

    # Stage 2: Assign all other waves to all remaining ridges
    used_ridges = [tup[0] for tup in assignments1]
    free_ridges_mask = [r not in used_ridges for r in range(cost.shape[0])]
    assignments2 = _masked_and_padded_hungarian(
        cost=cost, 
        row_mask=free_ridges_mask,
        col_mask=~priority_wave_mask, 
        max_cost=max_cost)
    
    # Concat assignments from both steps
    assignments = assignments1 + assignments2
    
    # DataFrame it
    res = pandas.DataFrame(
        assignments, columns=['rc_idx', 'centroids_idx', 'cost'])

    if len(res) > 0:
        # Convert indexing into recording_coefs and wave_centroids into 
        # n_ridge and wave_name
        res['wave_name'] = wave_centroids.index[res['centroids_idx']]
        res['n_ridge'] = recording_coefs.index[res['rc_idx']]
        res = res.drop(['centroids_idx', 'rc_idx'], axis=1)
        
        # Join on original recording coefs
        res = res.join(orig_recording_coefs, on='n_ridge')
    
    # Return
    return res

def _find_peaks_max_prominence(sig, prominence, initial_prom=0, **kwargs):
    """Edit find_peaks output to filter on max prominence instead of min
    
    By default, find_peaks calculates prominence as the minimum of left
    and right side, a conservative estimate that underestimates the prominence
    of shoulder peaks like W4. This function instead calculates prominence
    of the maximum of left and right peaks. It works by calling find_peaks
    with prominence=0, overwriting 'prominence' as the max prominence, and
    then applying a filter for `prominence`.
    
    One issue with this choice is that a trivially small dip (e.g., W0n)
    can have a large "prominence" simply because the following peak is high
    (e.g., W1p). The old method (min of both) was resilient to this. An 
    alternative that might make sense for ABR is to always use left prominence,
    so that only peaks with sharp ascents can be picked. I think for now it
    is okay to leave it like this - it helps pick out the subsequent troughs
    in case we want to compute peak-to-peak.
    
    sig : signal
    initial_prom : sent to find_peaks
    prominence : filters found peaks by this value
    kwargs : sent to find_peaks
    """
    # Run with first_pass_prominence
    peak_idxs, peak_data = scipy.signal.find_peaks(
        sig, 
        prominence=initial_prom, 
        **kwargs,
        )
    
    # Extract max prominence and overwrite 'prominences'
    left_prom = (
        sig[peak_idxs] - sig[peak_data['left_bases']])
    right_prom = (
        sig[peak_idxs] - sig[peak_data['right_bases']])
    peak_data['prominences'] = np.max([left_prom, right_prom], axis=0)
    
    # Filter
    keep_mask = peak_data['prominences'] > prominence
    peak_idxs = peak_idxs[keep_mask]
    peak_data['prominences'] = peak_data['prominences'][keep_mask]
    peak_data['left_bases'] = peak_data['left_bases'][keep_mask]
    peak_data['right_bases'] = peak_data['right_bases'][keep_mask]
    
    # Return
    return peak_idxs, peak_data

def _masked_and_padded_hungarian(cost, row_mask, col_mask, max_cost):
    """Linear sum assign a subset of waves to ridges with padding.

    Arguments
    - cost : cost matrix (ridges on rows, waves on cols)
    - row_mask : boolean array indicating which rows to keep
    - col_mask : boolean array indicating which cols to keep
    - max_cost : do not assign if cost is greater than this

    Workflow
    - Slice the cost matrix by row_mask and col_mask
    - Pad the bottom with `n_wave` dummy ridges, where `n_wave` is the 
      number of waves remaining after slicing
    - Hungarian assign
    - Drop assignments to dummy ridges or if they cost more than `max_cost`
    
    Returns: list of tuples (r, c, cost), one per assignment
    - r is the row index into the original cost matrix
    - c is the col index into the original cost matrix
    - cost is the cost of that assignment
    """
    # Slice out allowed ridges and waves
    sub = cost[np.ix_(row_mask, col_mask)]
    n_wave = sub.shape[1]

    # Pad with `n_wave` dummy ridges (one for each wave; new rows at the 
    # bottom) so that unmatched waves have a fallback at max_cost
    padded = np.vstack([sub, np.full((n_wave, n_wave), max_cost)])
    
    # Hungarian
    rows, cols = scipy.optimize.linear_sum_assignment(padded)

    # Store
    out = []
    for r, c in zip(rows, cols):
        # If r is a dummy row, or if the cost exceeds `max_cost`, then skip
        if r >= np.sum(row_mask) or padded[r, c] > max_cost:
            continue
        
        # Otherwise store the assignment
        # Undo the effect of masking to make these indices into the original
        # cost matrix
        out.append((
            np.arange(cost.shape[0])[row_mask][r], 
            np.arange(cost.shape[1])[col_mask][c], 
            padded[r, c]))
    
    # Return
    return out

