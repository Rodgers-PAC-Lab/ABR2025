## Run the ridge-tracing algorithm to identify waves

import os
import json
import numpy as np
import pandas
import my
import my.plot
import opensabr.peak_picking
import shared


## Paths
# Load the required file filepaths.json (see README)
with open('filepaths.json') as fi:
    paths = json.load(fi)

# Parse into paths to raw data and output directory
raw_data_directory = paths['raw_data_directory']
output_directory = paths['output_directory']


## Params
sampling_rate = 16000

# The minimum length of a ridge in levels. Shorter ridges are discarded
# Lower is more sensitive for weak responses but might cause false labels
minimum_ridge_length = 5

# This is the maximum allowable cost of a wave assignment to a ridge
# This param is tricky to set
# When it's too low, we lose the waves we need for analysis
# When it's too high (lenient), it's not just that there are more false
# positives, it can also cause a "frame shift" because those false positives
# grab waves belonging to others, causing a cascade of errors
# Current value is fairly strict
max_label_cost = 0.15

# Other params are defined in opensabr.peak_picking, including the typical
# peak latencies used for assigning wave names


## Load previous results
# Load results of Step2b_avg
averaged_abrs_by_date = pandas.read_parquet(
    os.path.join(output_directory, 'averaged_abrs_by_date'))

# Loudest dB
loudest_db = averaged_abrs_by_date.index.get_level_values('label').max()

# Include only vertex-ear channels
averaged_abrs_by_date = averaged_abrs_by_date.reindex(
    ['VL', 'VR'], level='channel')
averaged_abrs_by_date = averaged_abrs_by_date.sort_index()

# Convert to uV
averaged_abrs_by_date = averaged_abrs_by_date * 1e6


## Trace ridges for all recordings
# Groupby each heatmap
group_levels = [
    'HL_type', 'after_HL', 'n_experiment', 'mouse', 'channel', 'speaker_side',
    ]
groupby = averaged_abrs_by_date.groupby(group_levels)

# Iterate over recordings
ridges_l = []
ridges_keys_l = []
for this_recording_keys, this_recording in groupby:
    
    # Drop level
    this_recording = this_recording.droplevel(group_levels).copy()
    
    # Trace positive and negative ridges for this heatmap
    this_ridges_pos = opensabr.peak_picking.trace_ridges(this_recording)
    this_ridges_neg = opensabr.peak_picking.trace_ridges(-this_recording)
    
    # Concat both peaks and troughs
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

# Compute latency in ms
big_ridges['latency_ms'] = big_ridges['timepoint'] / sampling_rate * 1000


## Extract the height of each peak
# Form a slicing MultiIndex that contains the right timepoint
slicing_midx = big_ridges['timepoint'].reset_index().rename(
    columns={'level': 'label'})[
    group_levels + ['label', 'timepoint']]
slicing_midx = pandas.MultiIndex.from_frame(slicing_midx)

# Slice to get height
heights = averaged_abrs_by_date.stack().loc[slicing_midx]

# Store height - the index has the wrong levels but the order is correct
big_ridges['height'] = heights.values


## Label each ridge with a wave name
# Iterate over recordings
labeled_waves_l = []
labeled_waves_keys_l = []
for recording_keys, recording_ridges in big_ridges.groupby(group_levels):

    # Iterate over ridges from that recording
    coef_l = []
    for (sign, n_ridge), ridge in recording_ridges.groupby(['sign', 'n_ridge']):
        
        # Extract the regressors
        # Reference levels to `loudest_db` (so the "intercept" becomes the
        # latency at this reference level)
        levels_db = ridge.index.get_level_values('level').values - loudest_db
        assert len(levels_db) == len(np.unique(levels_db))
        latency_us = ridge['timepoint'].values / sampling_rate * 1e6

        # Compute the slope and intercept of each ridge's latency-vs-level
        slope, intercept = np.polyfit(levels_db, latency_us, deg=1)
        coef_l.append({
            'sign': sign,
            'n_ridge': n_ridge,
            'slope_us_per_db': slope, 
            'latency_ms_at_ref_level': intercept / 1e3,
            })

    # DataFrame
    recording_ridge_coefs = pandas.DataFrame(coef_l).set_index(
        ['sign', 'n_ridge'])
    
    # Label the ridges - pos and neg separately
    this_labeled_l = []
    this_labeled_keys_l = []
    for sign, this_rrc in recording_ridge_coefs.groupby('sign'):
        
        # Droplevel
        this_rrc = this_rrc.droplevel('sign')
        
        # Pick peaks according to the sign
        # The centroids are fit to the typical locations of the waves in 
        # our data and would have to be adjusted for other data
        assert sign in ['pos', 'neg']
        if sign == 'pos':
            this_labeled = opensabr.peak_picking.label_ridges(
                recording_ridge_coefs.loc['pos'], 
                opensabr.peak_picking.wave_centroids_pos, 
                max_cost=max_label_cost)
        
        elif sign == 'neg':
            this_labeled = opensabr.peak_picking.label_ridges(
                recording_ridge_coefs.loc['neg'], 
                opensabr.peak_picking.wave_centroids_neg, 
                max_cost=max_label_cost)
        
        # Reindex
        this_labeled = this_labeled.set_index('n_ridge')
        
        # Store
        this_labeled_l.append(this_labeled)
        this_labeled_keys_l.append(sign)

    # Concat
    labeled = pandas.concat(
        this_labeled_l, keys=this_labeled_keys_l, names=['sign'])

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
bad_l = []
for keys, subdf in big_ridges_with_order.groupby(group_levels + ['level']):
    
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

# Deal with out of order waves, if any
if len(bad_l) > 0:
    
    # Concat all out of order
    out_of_order = pandas.concat(bad_l, ignore_index=True)

    # Construct a MultiIndex to unlabel big_labeled_waves and big_ridges
    midx = pandas.MultiIndex.from_frame(
        out_of_order[
        group_levels + ['sign', 'n_ridge', 'wave_name']
        ].drop_duplicates())

    # Print warning (before dropping wave_name)
    print(f'warning: unlabeling {len(midx)} waves')
    print(midx.to_frame(index=False).to_string())
    midx = midx.droplevel('wave_name')
    
    # Unlabel - this two-step process avoids ChainedAssignmentError
    big_ridges.loc[
        my.misc.slice_df_by_some_levels(big_ridges, midx).index, 
        'wave_name'] = np.nan
    big_labeled_waves = big_labeled_waves.drop(
        my.misc.slice_df_by_some_levels(big_labeled_waves, midx).index, 
        )


## Print out highest cost assignments
# This can be used as a diagnostic to set the cost threshold
print(big_labeled_waves.sort_values('cost').iloc[-30:])


## Save picked peaks
big_ridges.to_parquet(os.path.join(output_directory, 'big_ridges'))
big_labeled_waves.to_parquet(os.path.join(output_directory, 'big_labeled_waves'))

