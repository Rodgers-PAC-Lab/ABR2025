## Peak picking plots

import os
import json
import datetime
import matplotlib
import scipy.signal
import numpy as np
import pandas
import my.plot
import matplotlib.pyplot as plt
import seaborn


## Plotting defaults
my.plot.manuscript_defaults()
my.plot.font_embed()
MU = chr(956)


## Paths
# Load the required file filepaths.json (see README)
with open('filepaths.json') as fi:
    paths = json.load(fi)

# Parse into paths to raw data and output directory
raw_data_directory = paths['raw_data_directory']
output_directory = paths['output_directory']


## Params
sampling_rate = 16000

# Define a set of priors on where waves typically are, which are used as 
# starting points for labeling waves
# Each tuple is (slope_us_per_db, latency_ms)
wave_centroids_pos = pandas.DataFrame.from_dict({
    'W0p': (-4.3, 0.6),
    'W1p': (-5.1, 1.36),
    'W2p': (-8.7, 2.3),
    'W3p': (-8.2, 3.3),
    'W4p': (-12.9, 4.2),
    'W5p': (-13.5, 5.2),
    'W6p': (-8, 6.1),
    'W7p': (-9, 7.0),
    }, 
    orient='index', 
    columns=['slope_us_per_db', 'latency_ms_at_ref_level'])
wave_centroids_pos.index.name = 'wave_name'

wave_centroids_neg = pandas.DataFrame.from_dict({
    'W0n': (-3.7, 0.9),
    'W1n': (-6.4, 1.8),
    'W2n': (-9, 2.7),
    'W3n': (-9, 3.7),
    'W4n': (-15, 4.7),
    'W5n': (-12, 5.7),
    'W6n': (-9, 6.5),
    'W7n': (-10, 7.4),
    }, 
    orient='index', 
    columns=['slope_us_per_db', 'latency_ms_at_ref_level'])
wave_centroids_neg.index.name = 'wave_name'

all_wave_centroids = pandas.concat([wave_centroids_pos, wave_centroids_neg]).sort_index()

# Wave colors for plotting
cmap = plt.get_cmap('tab20')
wave_colors = {}
for i in range(8):
    wave_colors[f'W{i}p'] = cmap(2 * i)      # dark
    wave_colors[f'W{i}n'] = cmap(2 * i + 1)  # light

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

# The minimum length of a ridge in levels. Shorter ridges are discarded
# Lower is more sensitive for weak responses but might cause false labels
minimum_ridge_length = 5

# The maximum cost of a wave assignment
# Lower is stricter (more unlabeled waves); higher might allow false labels
max_cost_wave_assign = 0.15

# Assign priority waves first - Can be helpful to ensure we get W1 and W4
priority_wave_list = [] #['W1', 'W4']


## Helper functions
def find_peaks_max_prominence(sig, prominence, initial_prom=0, **kwargs):
    """Edit find_peaks output to filter on max prominence instead of min
    
    By default, find_peaks calculates prominence as the minimum of left
    and right side, a conservative estimate that underestimates the prominence
    of shoulder peaks like W4. This function instead calculates prominence
    of the maximum of left and right peaks. It works by calling find_peaks
    with prominence=0, overwriting 'prominence' as the max prominence, and
    then applying a filter for `prominence`.
    
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
    seed_peak_idxs, seed_peak_data = find_peaks_max_prominence(
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
            level_peak_idxs, level_peak_data = find_peaks_max_prominence(
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

def masked_and_padded_hungarian(cost, row_mask, col_mask, max_cost):
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
    assignments1 = masked_and_padded_hungarian(
        cost=cost, 
        row_mask=np.ones(cost.shape[0], dtype=bool), 
        col_mask=priority_wave_mask, 
        max_cost=max_cost)

    # Stage 2: Assign all other waves to all remaining ridges
    used_ridges = [tup[0] for tup in assignments1]
    free_ridges_mask = [r not in used_ridges for r in range(cost.shape[0])]
    assignments2 = masked_and_padded_hungarian(
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


## Load metadata
mouse_metadata = pandas.read_csv(
    os.path.join(raw_data_directory, 'metadata', 'mouse_metadata.csv'))
experiment_metadata = pandas.read_csv(
    os.path.join(raw_data_directory, 'metadata', 'experiment_metadata.csv'))
recording_metadata = pandas.read_csv(
    os.path.join(raw_data_directory, 'metadata', 'recording_metadata.csv'))

# Coerce
recording_metadata['date'] = recording_metadata['date'].apply(
    lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())
experiment_metadata['date'] = experiment_metadata['date'].apply(
    lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())
mouse_metadata['DOB'] = mouse_metadata['DOB'].apply(
    lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())

# Coerce: special case this one because it can be null
mouse_metadata['HL_date'] = mouse_metadata['HL_date'].apply(
    lambda x: None if pandas.isnull(x) else 
    datetime.datetime.strptime(x, '%Y-%m-%d').date())

# Index
recording_metadata = recording_metadata.set_index(
    ['date', 'mouse', 'recording']).sort_index()
    

## Load previous results
# Load results of Step2b_avg
big_abrs = pandas.read_pickle(
    os.path.join(output_directory, 'big_abrs'))
averaged_abrs_by_mouse = pandas.read_pickle(
    os.path.join(output_directory, 'averaged_abrs_by_mouse'))
averaged_abrs_by_date = pandas.read_pickle(
    os.path.join(output_directory, 'averaged_abrs_by_date'))
trial_counts = pandas.read_pickle(
    os.path.join(output_directory, 'trial_counts'))

# Loudest dB
loudest_db = big_abrs.index.get_level_values('label').max()
    

## Keep only after_HL == False
big_abrs = big_abrs.xs(False, level='after_HL').droplevel('HL_type')
averaged_abrs_by_mouse = averaged_abrs_by_mouse.xs(False, level='after_HL').droplevel('HL_type')
averaged_abrs_by_date = averaged_abrs_by_date.xs(False, level='after_HL').droplevel('HL_type')


## Slice before extracting peaks
# Convert to uV
df = averaged_abrs_by_mouse * 1e6  # V -> uV

# VL / VR only
df = df[df.index.get_level_values('channel').isin(['VL', 'VR'])]
df = df.sort_index()


## Trace ridges for all recordings
# Iterate over recordings
group_levels = ['mouse', 'channel', 'speaker_side']
ridges_l = []
ridges_keys_l = []

for this_recording_keys, this_recording in df.groupby(group_levels):
    
    # Drop level
    this_recording = this_recording.droplevel(group_levels).copy()
    
    # Drop timepoints before t = 0 to avoid noise ridges
    # There is occasionally a ghost ridge at t=-1 or so, which could be a 
    # filter ring from the summating potential
    #~ this_recording = this_recording.loc[:, 0:]
    
    # Ridges for this config
    this_ridges_pos = trace_ridges(this_recording)
    this_ridges_neg = trace_ridges(-this_recording)
    
    # Concat
    this_ridges = pandas.concat(
        [this_ridges_pos, this_ridges_neg], 
        keys=['pos', 'neg'], names=['sign'])
    
    # Skip if no ridges found (otherwise next line will fail)
    if len(this_ridges) == 0:
        continue
    
    # Drop short ridges
    ridge_len = this_ridges.groupby(['sign', 'n_ridge']).size()
    drop_ridges = ridge_len.index[ridge_len.values < minimum_ridge_length]
    this_ridges = this_ridges.drop(drop_ridges)
    
    # Store
    ridges_l.append(this_ridges)
    ridges_keys_l.append(this_recording_keys)

# Concat
big_ridges = pandas.concat(ridges_l, keys=ridges_keys_l, names=group_levels)


## Fit each ridge and label by wave
# This is the reference level used to compute the latency-vs-level line
ref_level = df.index.get_level_values('label').max()

# Compute the slope and intercept of each ridges latency-vs-level
labeled_waves_l = []
labeled_waves_keys_l = []
group_levels = ['mouse', 'channel', 'speaker_side']
for recording_keys, recording_ridges in big_ridges.groupby(group_levels):

    #~ if recording_keys != ('Cat_227', 'VR', 'L'):
        #~ continue

    # Iterate over ridges
    coef_l = []
    for (sign, n_ridge), ridge in recording_ridges.groupby(['sign', 'n_ridge']):
        
        # Extract the regressors
        # Reference levels to `ref_level` (so the "intercept" becomes the
        # latency at this reference level)
        levels_db = ridge.index.get_level_values('level').values - ref_level
        assert len(levels_db) == len(np.unique(levels_db))
        latency_us = ridge['timepoint'].values / sampling_rate * 1e6

        # Fit a line
        slope, intercept = np.polyfit(levels_db, latency_us, deg=1)
        coef_l.append({
            'sign': sign,
            'n_ridge': n_ridge,
            'slope_us_per_db': slope, 
            'latency_ms_at_ref_level': intercept / 1e3,
            })

    # DataFrame
    recording_ridge_coefs = pandas.DataFrame(coef_l).set_index(['sign', 'n_ridge'])
    
    # Label the ridges - pos and neg separately
    labeled_pos = label_ridges(recording_ridge_coefs.loc['pos'], wave_centroids_pos)
    labeled_neg = label_ridges(recording_ridge_coefs.loc['neg'], wave_centroids_neg)

    # Concat
    labeled = pandas.concat(
        [labeled_pos.set_index('n_ridge'), labeled_neg.set_index('n_ridge')], 
        keys=['pos', 'neg'], names=['sign'])

    # Store
    labeled_waves_l.append(labeled)
    labeled_waves_keys_l.append(recording_keys)

# Concat
big_labeled_waves = pandas.concat(
    labeled_waves_l, keys=labeled_waves_keys_l, names=group_levels).sort_index()

# Join wave_name on big_ridges
big_ridges = big_ridges.join(big_labeled_waves['wave_name'])


## Unlabel waves that are out of order
# First compute order of the waves on big_ridges
big_ridges_with_order = big_ridges.dropna(subset='wave_name').copy()
big_ridges_with_order['wave_num'] = (
    big_ridges_with_order['wave_name'].str.extract(r'W(\d)').astype(int))
big_ridges_with_order['order'] = big_ridges_with_order['wave_num'] * 2 + (
    big_ridges_with_order['wave_name'].str[-1] == 'n').astype(int)

# Find out of order waves
group_levels = ['mouse', 'channel', 'speaker_side', 'level']
bad_l = []
for keys, subdf in big_ridges_with_order.groupby(group_levels):
    
    # Sort in order
    subdf = subdf.sort_values('order')
    
    # Identify out-of-order rows
    badmask = subdf['timepoint'].diff() < 0
    badmask = badmask | badmask.shift(-1)
    assert subdf.drop(
        badmask.index[badmask.values])['timepoint'].is_monotonic_increasing
    
    # Store the offending waves
    badidx = subdf.reset_index().loc[:, 
        group_levels + ['sign', 'n_ridge', 'wave_name']
        ][badmask.values]
    
    # Store
    if len(badidx) > 0:
        bad_l.append(badidx)

# Concat all out of order
out_of_order = pandas.concat(bad_l, ignore_index=True)

# Construct a MultiIndex to unlabel big_labeled_waves and big_ridges
midx = pandas.MultiIndex.from_frame(
    out_of_order[
    ['mouse', 'channel', 'speaker_side', 'sign', 'n_ridge', 'wave_name']
    ].drop_duplicates())

# Unlabel - this two-step process avoids ChainedAssignmentError
if len(midx) > 0:
    print(f'warning: unlabeling {len(midx)} waves')
    print(midx)
midx = midx.droplevel('wave_name')
big_ridges.loc[
    my.misc.slice_df_by_some_levels(big_ridges, midx).index, 
    'wave_name'] = np.nan
big_labeled_waves = big_labeled_waves.drop(
    my.misc.slice_df_by_some_levels(big_labeled_waves, midx).index, 
    )


## Print out highest cost assignments
print(big_labeled_waves.sort_values('cost').iloc[-30:])


## Plot
PLOT_RIDGES = True
PLOT_COEFS = True
HISTOGRAM_WAVE_LATENCIES = True
SWARM_PLOT_LATENCIES = True
PLOT_PROPORTION_OF_WAVES_DETECTED = True


## Plot ridges
if PLOT_RIDGES:
    """Plot all ABR heatmaps with detected ridges overlaid
    
    4 rows (channel * speaker_side) x N cols (mouse) of ABR heatmaps.


    Grayscale heatmap per stack, with each traced ridge drawn in a cycled
    color at (time, level).
    
    df : DataFrame
        index: MultiIndex (mouse, channel, speaker_side, label)
        columns: timepoint
        values: ABR in microvolts
    
    big_ridges : DataFrame, one row per detected ridge
        index: MultiIndex (mouse, channel, speaker_side, n_ridge, level)
        columns: row_idx, col_idx, size, timepoint
    
    big_labeled_waves : DataFrame, one row per labeled wave
        index: MultiIndex (mouse, channel, speaker_side, unlabeled)
        columns; wave_name, n_ridge, cost, slope_us_per_db, latency_ms_at_ref_level
    """
    
    ## Set up layout
    # Layout
    mouse_l = sorted(df.index.get_level_values('mouse').unique())
    config_l = [('VL', 'L'), ('VL', 'R'), ('VR', 'L'), ('VR', 'R')]

    # Set t
    t = df.columns.values / sampling_rate * 1000

    # Color cycle for ridges
    cmap = plt.get_cmap('tab10')

    
    ## Figure handles
    f, axa = plt.subplots(
        len(config_l), len(mouse_l), sharex=True, sharey=True,
        figsize=(0.9 * len(mouse_l), 3.5))
    f.subplots_adjust(
        left=.04, right=.98, wspace=0.1, hspace=0.1, bottom=.05, top=.9)

    
    ## Iterate over configs
    for config in config_l:
        
        # Parse config
        channel, speaker_side = config

        # Each mouse
        for mouse in mouse_l:
            
            # Get ax
            ax = axa[config_l.index(config), mouse_l.index(mouse)]

            # Get ABR for this mouse
            try:
                abr_heatmap = df.loc[mouse].loc[channel].loc[speaker_side]
            except KeyError:
                continue

            # Imshow heatmap
            im = my.plot.imshow(
                abr_heatmap, 
                ax=ax, 
                cmap='gray', 
                x=abr_heatmap.columns.values / sampling_rate * 1000,
                y=abr_heatmap.index.values,
                alpha=.9,
                origin='lower',
                )

            # Slice this stack's ridges and wave labels
            try:
                this_ridges = big_ridges.loc[mouse].loc[channel].loc[speaker_side]
            except KeyError:
                this_ridges = None

            # Plot if ridges exist
            if this_ridges is not None:
                
                # Iterate over ridges
                for (sign, n_ridge), ridge in this_ridges.groupby(['sign', 'n_ridge']):

                    # Undo multi-indexing
                    ridge = ridge.reset_index()

                    # Get wave name
                    wave_name = ridge['wave_name'].unique().item()
                    
                    # Get color by wave name
                    if pandas.isnull(wave_name):
                        color = 'k'
                    else:
                        color = wave_colors[wave_name]

                    # Plot
                    ridge_t = ridge['timepoint'].values / sampling_rate * 1000
                    ax.plot(
                        ridge_t, 
                        ridge['level'].values, 
                        ls='-', color=color, lw=.8)

            # Pretty
            im.set_clim((-3, 3))
            ax.set_xlim((-2, 7))
            ax.set_xticks([])
            ax.set_yticks([])

            # Column titles on top row
            if ax in axa[0]:
                ax.set_title(mouse, ha='center', va='bottom', size='xx-small')

            # Row labels on left
            if ax in axa[:, 0]:
                ax.set_ylabel(f'{channel} {speaker_side}', rotation=0,
                    ha='right', va='center')

    # Savefig
    f.savefig('figures/PLOT_RIDGES.png', dpi=300)
    f.savefig('figures/PLOT_RIDGES.svg')    

## Plot labeled ridges in coef space to refine centroids
if PLOT_COEFS:
    f, ax = plt.subplots(figsize=(5, 7))

    # Color per wave
    wave_names = list(wave_colors.keys())

    # Each wave's ridges
    for n_wave, wave_name in enumerate(wave_names):

        # Get color
        color = wave_colors[wave_name]

        # This wave's assigned ridges
        this_wave = big_labeled_waves[big_labeled_waves['wave_name'] == wave_name]
        ax.plot(
            this_wave['slope_us_per_db'], 
            this_wave['latency_ms_at_ref_level'], 
            'o',
            color=color,
            ms=3, mew=0, alpha=0.5, label=wave_name)

        # Centroid
        slope, intercept = all_wave_centroids.loc[wave_name]
        ax.plot(slope, intercept, 'x', color=color, ms=10, mew=2)

    # Pretty
    ax.set_xlabel('slope (us/dB)')
    ax.set_ylabel('intercept at loudest (ms)')
    ax.legend(loc='upper right', fontsize='small')

    # Savefig
    f.savefig('figures/PLOT_COEFS.png', dpi=300)
    f.savefig('figures/PLOT_COEFS.svg')
    
## Histogram the wave times
if HISTOGRAM_WAVE_LATENCIES:
    # Levels to plot
    levels = sorted(big_ridges.index.get_level_values('level').unique())[::-1]

    # Bins
    bins = np.arange(0, big_ridges['timepoint'].max() + 2) / sampling_rate * 1000
    
    # One ax per level
    f, axa = plt.subplots(
        len(levels), 1, sharex=True, sharey=True, figsize=(6, 0.5 * len(levels)))

    # PLot each
    for ax, level in zip(axa, levels):
        # Slice
        sub = big_ridges.xs(level, level='level')
        
        # Hist each
        for wave_name, grp in sub.groupby('wave_name', dropna=False):
            color = 'k' if pandas.isnull(wave_name) else wave_colors[wave_name]
            ax.hist(grp['timepoint'] / sampling_rate * 1000,
                bins=bins, color=color, alpha=0.5)
        
        # ylabel
        ax.set_ylabel(f'{level}', rotation=0, ha='right', va='center')
        
        # Despine
        my.plot.despine(ax)
        ax.set_yticks([])

    # xlabel
    axa[-1].set_xlabel('latency (ms)')

    # Savefig
    f.savefig('figures/HISTOGRAM_WAVE_LATENCIES.png', dpi=300)
    f.savefig('figures/HISTOGRAM_WAVE_LATENCIES.svg')

## Swarm plot
if SWARM_PLOT_LATENCIES:
    # Assign latency
    big_ridges = big_ridges.copy()
    big_ridges['latency_ms'] = big_ridges['timepoint'] / sampling_rate * 1000

    # Fillna with unlabeled
    sub = big_ridges.reset_index()
    sub['wave_name'] = sub['wave_name'].fillna('unlabeled')

    # Define a palette to use with unlabeled as black
    palette = dict(wave_colors, unlabeled='k')

    # Swarm or strip plot the latency
    f, ax = plt.subplots(figsize=(6, 8))
    f.subplots_adjust(right=.8)
    hue_order = sorted(sub['wave_name'].unique())
    seaborn.stripplot(
        data=sub, x='latency_ms', y='level', hue='wave_name',
        hue_order=hue_order, order=sorted(sub['level'].unique(), reverse=True),
        orient='h', palette=palette, size=2, jitter=0.3, ax=ax)

    ax.set_xlabel('latency (ms)')
    ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1.02, 1))    
    
    # Pretty
    ax.set_xlabel('latency (ms)')

    # Savefig
    f.savefig('figures/SWARM_PLOT_LATENCIES.png', dpi=300)
    f.savefig('figures/SWARM_PLOT_LATENCIES.svg')

## Proportion of waves detected
if PLOT_PROPORTION_OF_WAVES_DETECTED:
    # Count waves detected
    counts = big_ridges.copy()
    counts['wave_name'] = counts['wave_name'].fillna('unlabeled')
    counts = counts.groupby(
        ['channel', 'speaker_side', 'wave_name', 'level']).size()

    # Reindex to fill absent (wave, level) combos with 0
    levels = sorted(big_ridges.index.get_level_values('level').unique())
    full_idx = pandas.MultiIndex.from_product([
        counts.index.get_level_values('channel').unique(),
        counts.index.get_level_values('speaker_side').unique(),
        counts.index.get_level_values('wave_name').unique(),
        levels],
        names=['channel', 'speaker_side', 'wave_name', 'level'])
    counts = counts.reindex(full_idx, fill_value=0)

    # Get a color palette
    palette = dict(wave_colors, unlabeled='k')
    
    # Configs to plot
    configs = [('VL', 'L'), ('VL', 'R'), ('VR', 'L'), ('VR', 'R')]

    # Plot handles
    f, axa = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7, 6))
    f.subplots_adjust(right=.8)
    for ax, (channel, side) in zip(axa.flatten(), configs):
        # Get counts
        try:
            sub = counts.loc[channel].loc[side]
        except KeyError:
            continue
        
        # Plot for each wave
        for wave_name, grp in sub.groupby('wave_name'):
            ax.plot(grp.index.get_level_values('level'), grp.values,
                marker='o', ms=3, color=palette[wave_name], label=wave_name)
        
        # Pretty
        ax.set_title(f'{channel} {side}', size='small')
        ax.set_ylim(bottom=0)
        ax.set_xticks((30, 50, 70))
        my.plot.despine(ax)

    # Legend
    axa[0, 1].legend(fontsize='small', loc='upper left', bbox_to_anchor=(1.02, 1))
    for ax in axa[-1]:
        ax.set_xlabel('level (dB)')
    for ax in axa[:, 0]:
        ax.set_ylabel('count')

    # Savefig
    f.savefig('figures/PLOT_PROPORTION_OF_WAVES_DETECTED.png', dpi=300)
    f.savefig('figures/PLOT_PROPORTION_OF_WAVES_DETECTED.svg')

    
    
plt.show()