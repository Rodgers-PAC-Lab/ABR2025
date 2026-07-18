## Comparison of our peak picking with ABRA's
# Skip this script if you didn't run ABRA

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
import opensabr.peak_picking
import shared


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



# Wave colors for plotting
cmap = plt.get_cmap('tab20')
wave_colors = {}
for i in range(8):
    wave_colors[f'W{i}p'] = cmap(2 * i)      # dark
    wave_colors[f'W{i}n'] = cmap(2 * i + 1)  # light


## Load metadata
metadata = shared.load_metadata(raw_data_directory)

# Parse out
mouse_metadata = metadata['mouse_metadata'].copy()
recording_metadata = metadata['recording_metadata'].copy()
experiment_metadata = metadata['experiment_metadata'].copy()
    

## Load previous results
# Load results of Step2b_avg
averaged_abrs_by_date = pandas.read_parquet(
    os.path.join(output_directory, 'averaged_abrs_by_date'))
trial_counts = pandas.read_parquet(
    os.path.join(output_directory, 'trial_counts'))

# Loudest dB
loudest_db = averaged_abrs_by_date.index.get_level_values('label').max()

# Include only vertex-ear channels
averaged_abrs_by_date = averaged_abrs_by_date.reindex(
    ['VL', 'VR'], level='channel')
averaged_abrs_by_date = averaged_abrs_by_date.sort_index()

# Convert to uV
averaged_abrs_by_date = averaged_abrs_by_date * 1e6

# Results of peak-picking
big_ridges = pandas.read_parquet(
    os.path.join(output_directory, 'big_ridges'))
big_labeled_waves = pandas.read_parquet(
    os.path.join(output_directory, 'big_labeled_waves'))

# Load abranalysis peaks
big_abra_peaks = pandas.read_parquet(
    os.path.join(output_directory, 'big_abra_peaks'))
assert (big_abra_peaks.index.levels[1] == ['VL', 'VR']).all()

# Stack peak and trough into one column
original_levels = list(big_abra_peaks.index.names)
big_abra_peaks.columns.name = 'typ'
big_abra_peaks = big_abra_peaks.stack().rename('timepoint').reset_index()

# Use canonical wave_name (W1p, W1n, ...)
big_abra_peaks['wave_name'] = (
    'W' + 
    big_abra_peaks['wave'].astype(str) + # wave num
    big_abra_peaks['typ'].map({'peak': 'p', 'trough': 'n'})
    )

# Reindex
big_abra_peaks = big_abra_peaks.set_index(
    original_levels[:-1] + ['wave_name']).sort_index()[['timepoint']]

# Drop null latencies
big_abra_peaks = big_abra_peaks.dropna()
big_abra_peaks['timepoint'] = big_abra_peaks['timepoint'].astype(int)

# Compute latency in milliseconds
big_abra_peaks['latency_ms'] = (
    big_abra_peaks['timepoint'] / sampling_rate * 1000)


## Plot
STRIP_PLOT_ABRA_LATENCIES = True
PLOT_EXAMPLE_WATERFALL_ABRA = True
PLOT_ABRA_PEAKS_AT_LOUDEST_ACROSS_MICE_VRL_ONLY = True
PLOT_LATENCY_ABRA_VS_OURS = True


if STRIP_PLOT_ABRA_LATENCIES:
    """Strip plot the latencies from ABRA for comparison
    
    TODO: re-run ABRA separately by recording, both pre and post HL
    """

    # Levels to plot
    levels = sorted(big_ridges.index.get_level_values('level').unique())[::-1]

    # Which waves to include
    include_waves = [
        'W1p', 'W1n', 'W2p', 'W2n', 'W3p', 'W3n', 'W4p', 'W4n', 'W5p', 'W5n']
    
    # Slice out waves to plot
    topl = big_abra_peaks.reindex(include_waves, level='wave_name')
    
    # A single ax with each swarm at its own ypos
    f, ax = plt.subplots(figsize=(4, 4.7))
    f.subplots_adjust(left=.15, bottom=.13, top=.95, right=.95)

    # Strip plot the latency
    seaborn.stripplot(
        data=topl, x='latency_ms', y='sound_level', hue='wave_name', 
        hue_order=include_waves, order=levels, orient='h', 
        palette=wave_colors, size=2, alpha=1, jitter=0.3, 
        ax=ax,
        )

    # Pretty
    ax.get_legend().set_visible(False)
    my.plot.despine(ax)
    ax.set_yticks((0, len(levels) - 1))
    ax.set_yticklabels((levels[0], levels[-1]))
    ax.set_xlabel('time from click (ms)')
    ax.set_ylabel('sound level (dB SPL)')
    ax.set_xlim((1, 7))
    ax.set_xticks((1, 2, 3, 4, 5, 6, 7))


    ## Write out stats
    stats_filename = 'figures/STATS__STRIP_PLOT_ABRA_LATENCIES'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write(f"n = {len(topl.groupby('mouse').size())} mice\n")
    
    # Echo
    with open(stats_filename) as fi:
        print(''.join(fi.readlines()))    
        

    ## Savefig
    f.savefig('figures/STRIP_PLOT_ABRA_LATENCIES.png', dpi=300)
    f.savefig('figures/STRIP_PLOT_ABRA_LATENCIES.svg')


if PLOT_EXAMPLE_WATERFALL_ABRA:
    """Example waterfall with ABRA peaks
    
    """
    # Choose example
    HL_type = 'sham'
    after_HL = False
    n_experiment = 0
    mouse = 'Cat_227'
    channel = 'VR'
    speaker_side = 'L'

    # Get ABR for this config
    example_abr = averaged_abrs_by_date.loc[
        (HL_type, after_HL, n_experiment, mouse, channel, speaker_side)]

    # Get this config's labeled ridges
    example_ridges = big_ridges.loc[
        (HL_type, after_HL, n_experiment, mouse, channel, speaker_side)]

    # Time range to plot
    start_timepoint = int(np.rint(-.001 * sampling_rate))
    stop_timepoint = int(np.rint(.007 * sampling_rate))

    # Levels to plot
    levels = sorted(example_abr.index)

    # Which waves to include
    include_waves = [
        'W1p', 'W1n', 'W2p', 'W2n', 'W3p', 'W3n', 'W4p', 'W4n', 'W5p', 'W5n']
    
    # Figure
    f, ax = my.plot.figure_1x1_standard()

    # Plot each level, offset by its rank
    level_offset_y = 2.0
    for n_level, level in enumerate(levels):
        # Slice
        topl = example_abr.loc[
            level, start_timepoint:stop_timepoint-1].copy()
        
        # Offset
        topl += n_level * level_offset_y
        
        # Plot
        ax.plot(
            topl.index / sampling_rate * 1000, 
            topl.values,
            color='k', lw=.5, clip_on=False,
            )

    # Iterate over each ridge
    for wave_name in include_waves:

        # Color by wave name
        color = wave_colors[wave_name]

        # Pull out x and y of each peak
        for sound_level in levels:
            
            # Time in ms of this peak
            try:
                timepoint = big_abra_peaks.loc[(
                    mouse, channel, speaker_side, 
                    sound_level, wave_name
                    ),
                    'timepoint']
            except KeyError:
                continue
            
            # Continue if timepoint outside plot range
            if timepoint < start_timepoint or timepoint >= stop_timepoint:
                continue
            
            # Convert to ms
            t = timepoint / sampling_rate * 1000
            
            # Voltage of this peak
            y = example_abr.loc[sound_level, timepoint]
            
            # Add level offset
            y += levels.index(sound_level) * level_offset_y
            
            # Plot
            ax.plot(
                [t], [y], marker='o', color=color, ms=3, mew=0, clip_on=False)

    
    ## Pretty
    my.plot.despine(ax)
    ax.set_yticks((0, level_offset_y * (len(levels) - 1)))
    ax.set_yticklabels((levels[0], levels[-1]))
    ax.set_xlabel('time from click (ms)')
    ax.set_ylabel('sound level (dB SPL)')
    ax.set_xlim((-1, 7))
    ax.set_xticks((0, 2, 4, 6))
    ax.set_ylim(-level_offset_y, len(levels) * level_offset_y)
    
    
    ## Savefig
    f.savefig('figures/PLOT_EXAMPLE_WATERFALL_ABRA.png', dpi=300)
    f.savefig('figures/PLOT_EXAMPLE_WATERFALL_ABRA.svg')


if PLOT_ABRA_PEAKS_AT_LOUDEST_ACROSS_MICE_VRL_ONLY:
    """ABRA peaks over-plotted on VR-L response to loudest sound"""

    # Include only these
    abrs_pre_HL = averaged_abrs_by_date.xs(
        False, level='after_HL').xs(0, level='n_experiment').droplevel('HL_type')
    ridges_pre_HL = big_ridges.xs(
        False, level='after_HL').xs(0, level='n_experiment').droplevel('HL_type')

    # Mice to plot
    mouse_l = sorted(
        abrs_pre_HL.index.get_level_values('mouse').unique())

    # Levels to plot
    levels = sorted(
        abrs_pre_HL.index.get_level_values('label').unique())

    # Which waves to include
    include_waves = [
        'W1p', 'W1n', 'W2p', 'W2n', 'W3p', 'W3n', 'W4p', 'W4n', 'W5p', 'W5n']
    
    # Make figure handles
    f, ax = my.plot.figure_1x1_standard()
    channel = 'VR'
    speaker_side = 'L'

    # Iterate over mice (lines)
    for mouse in mouse_l:

        # Slice ABR for this mouse at the loudest level
        this_abr = abrs_pre_HL.loc[
            (mouse, channel, speaker_side, loudest_db)]
        
        # Slice peaks from this mouse at the loudest level
        this_peaks = ridges_pre_HL.loc[(mouse, channel, speaker_side)].xs(
            loudest_db, level='level')

        # Plot the ABR trace for this mouse
        ax.plot(
            this_abr.index.values / sampling_rate * 1000, 
            this_abr.values, 
            color='k', lw=.5, alpha=.5)

        # Iterate over each ridge
        for wave_name in include_waves:

            # Color by wave name
            color = wave_colors[wave_name]

            # Time in ms of this peak
            try:
                timepoint = big_abra_peaks.loc[
                    (mouse, channel, speaker_side, loudest_db, wave_name),
                    'timepoint']
            except KeyError:
                continue
            
            # Convert to ms
            t = timepoint / sampling_rate * 1000
            
            # Voltage of this peak
            try:
                y = this_abr.loc[timepoint]
            except KeyError:
                continue
            
            # Plot
            ax.plot(
                [t], [y], 
                marker='o', color=color, ms=4, alpha=.75)
    

    ## Pretty
    ax.set_xlim((-1, 7))
    ax.set_xticks((0, 2, 4, 6))
    ax.set_yticks((-3, 0, 3))
    my.plot.despine(ax)
    ax.set_xlabel('time from click (ms)')
    ax.set_ylabel(f'ABR ({MU}V)')
    
    f.savefig('figures/PLOT_ABRA_PEAKS_AT_LOUDEST_ACROSS_MICE_VRL_ONLY.svg')
    f.savefig('figures/PLOT_ABRA_PEAKS_AT_LOUDEST_ACROSS_MICE_VRL_ONLY.png', dpi=300)


if PLOT_LATENCY_ABRA_VS_OURS:
    """Connected-pairs plot of our latency vs ABRA's"""

    # Include only these
    abrs_pre_HL = averaged_abrs_by_date.xs(
        False, level='after_HL').xs(0, level='n_experiment').droplevel('HL_type')
    ridges_pre_HL = big_ridges.xs(
        False, level='after_HL').xs(0, level='n_experiment').droplevel('HL_type')

    # Match the datasets by these columns
    match_keys = ['mouse', 'channel', 'speaker_side', 'level', 'wave_name']

    # Waves to include
    wave_order = [
        'W1p', 'W1n', 'W2p', 'W2n', 'W3p', 'W3n', 'W4p', 'W4n', 'W5p', 'W5n']

    # Ours
    ours = ridges_pre_HL[
        ridges_pre_HL['wave_name'].isin(wave_order)
        ].reset_index().set_index(match_keys)['latency_ms']

    # ABRA
    theirs = big_abra_peaks.reset_index().rename(
        columns={'sound_level': 'level'}).set_index(match_keys)['latency_ms']

    # Pair ours with theirs
    paired = pandas.concat(
        [ours.rename('our_ms'), theirs.rename('their_ms')], 
        axis=1)

    # Drop nulls (found by ours but not ABRA, this is very rare)
    drop_mask = paired['their_ms'].isnull()
    print(f'dropping {drop_mask.sum()} points')
    paired = paired.loc[~drop_mask]

    # Drop nulls (found by ABRA but not ours, this is common)
    drop_mask = paired['our_ms'].isnull()
    print(f'dropping {drop_mask.sum()} points')
    paired = paired.loc[~drop_mask]
    assert not paired.isnull().any().any()
    

    ## Pair each wave
    f, ax = plt.subplots(figsize=(6.6, 3.4))
    f.subplots_adjust(bottom=.15)

    # Difference between ours and theirs on the x-axis (around a base point)
    wave_offset = 0.15
    
    # Jitter applied for visualization
    jitter = 0.04

    # Iterate over waves
    for wave_name, this_wave in paired.groupby('wave_name'):

        # Slice VL-R only to decrease the number of pairs
        this_wave = this_wave.xs('VL', level='channel').xs(
            'R', level='speaker_side')

        # Determine the central x-point for this wave
        this_x = wave_order.index(wave_name)
        
        # Apply a jitter to each
        n_points = len(this_wave)
        x_ours = this_x - wave_offset + np.random.uniform(
            -jitter, jitter, n_points)
        x_theirs = this_x + wave_offset + np.random.uniform(
            -jitter, jitter, n_points)

        # Draw pairing lines
        for this_xo, this_xt, this_yo, this_yt in zip(
                x_ours, x_theirs, this_wave['our_ms'], this_wave['their_ms']):
            ax.plot(
                [this_xo, this_xt], [this_yo, this_yt],
                color='gray', lw=.5, alpha=.3, zorder=0)

        # Points: ours red, theirs blue
        ax.scatter(x_ours, this_wave['our_ms'], s=4, color='red', alpha=.3)
        ax.scatter(x_theirs, this_wave['their_ms'], s=4, color='blue', alpha=.3)


    ## Pretty
    ax.set_xticks(range(len(wave_order)))
    ax.set_xticklabels(wave_order)
    ax.set_ylim((0, 10))
    ax.set_yticks((0, 2, 4, 6, 8, 10))
    ax.set_ylabel('time from click (ms)')
    ax.set_xlabel('wave name')
    my.plot.despine(ax)
    
    
    ## Savefig
    f.savefig('figures/PLOT_LATENCY_ABRA_VS_OURS.svg')
    f.savefig('figures/PLOT_LATENCY_ABRA_VS_OURS.png', dpi=300)


plt.show()