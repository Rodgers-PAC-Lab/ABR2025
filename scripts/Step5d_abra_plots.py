## Comparison of our peak picking with ABRA's
# Skip this script if you didn't run ABRA

import os
import json
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

# Parse into path to output directory
output_directory = paths['output_directory']


## Params
sampling_rate = 16000

# Which waves to include in the plots
include_waves = [
    'W1p', 'W1n', 'W2p', 'W2n', 'W3p', 'W3n', 'W4p', 'W4n', 'W5p', 'W5n']

# Wave colors for plotting
cmap = plt.get_cmap('tab20')
wave_colors = {}
for i in range(8):
    wave_colors[f'W{i}p'] = cmap(2 * i)      # dark
    wave_colors[f'W{i}n'] = cmap(2 * i + 1)  # light


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

# Results of peak-picking
big_ridges = pandas.read_parquet(
    os.path.join(output_directory, 'big_ridges'))

# Load abranalysis peaks
big_abra_peaks = pandas.read_parquet(
    os.path.join(output_directory, 'big_abra_peaks'))
assert (
    sorted(big_abra_peaks.index.get_level_values('channel').unique()) ==
    ['VL', 'VR'])

# Stack peak and trough into one column
big_abra_peaks.columns.name = 'typ'
big_abra_peaks = big_abra_peaks.stack().rename('timepoint').reset_index()

# Use canonical wave_name (W1p, W1n, ...)
big_abra_peaks['wave_name'] = (
    'W' + 
    big_abra_peaks['wave'].astype(str) + # wave num
    big_abra_peaks['typ'].map({'peak': 'p', 'trough': 'n'})
    )

# Index identically to big_ridges (sound_level is called level there)
big_abra_peaks = big_abra_peaks.rename(columns={'sound_level': 'level'})
big_abra_peaks = big_abra_peaks.set_index([
    'HL_type', 'after_HL', 'n_experiment', 'mouse', 'channel',
    'speaker_side', 'level', 'wave_name']).sort_index()[['timepoint']]

# Drop null latencies
big_abra_peaks = big_abra_peaks.dropna()
big_abra_peaks['timepoint'] = big_abra_peaks['timepoint'].astype(int)

# Compute latency in milliseconds
big_abra_peaks['latency_ms'] = (
    big_abra_peaks['timepoint'] / sampling_rate * 1000)


## Plot
STRIP_PLOT_ABRA_LATENCIES = True
PLOT_EXAMPLE_WATERFALL_ABRA = True
PLOT_ABRA_PEAKS_AT_LOUDEST_ACROSS_MICE = True
PLOT_LATENCY_ABRA_VS_OURS = True


if STRIP_PLOT_ABRA_LATENCIES:
    """Strip plot the latencies from ABRA for comparison
    
    This is run only for the first recording of each mouse (one per mouse)
    """

    # Levels to plot
    levels = sorted(big_ridges.index.get_level_values('level').unique())[::-1]

    # Slice big_abra_peaks to first pre-HL recording only
    this_abra_peaks = big_abra_peaks.xs(
        False, level='after_HL').xs(0, level='n_experiment').droplevel('HL_type')
    
    # Slice out waves to plot
    topl = this_abra_peaks.reindex(
        include_waves, level='wave_name').reset_index()
    
    # A single ax with each swarm at its own ypos
    f, ax = plt.subplots(figsize=(3.1, 4.7))
    f.subplots_adjust(left=.2, bottom=.13, top=.95, right=.95)

    # Strip plot the latency
    alpha = 0.5
    seaborn.stripplot(
        data=topl, x='latency_ms', y='level', hue='wave_name', 
        hue_order=include_waves, order=levels, orient='h', 
        palette=wave_colors, size=2, alpha=alpha, jitter=0.3, 
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
        fi.write(f"n = {topl['mouse'].nunique()} mice\n")
    
    # Echo
    with open(stats_filename) as fi:
        print(''.join(fi.readlines()))    
        

    ## Savefig
    f.savefig('figures/STRIP_PLOT_ABRA_LATENCIES.png', dpi=300)
    f.savefig('figures/STRIP_PLOT_ABRA_LATENCIES.svg')


if PLOT_EXAMPLE_WATERFALL_ABRA:
    """Example waterfall with ABRA peaks"""
    
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

    # Time range to plot
    start_timepoint = int(np.rint(-.001 * sampling_rate))
    stop_timepoint = int(np.rint(.007 * sampling_rate))

    # Levels to plot
    levels = sorted(example_abr.index)

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

    # Iterate over each wave
    for wave_name in include_waves:

        # Color by wave name
        color = wave_colors[wave_name]

        # Pull out x and y of each peak
        for sound_level in levels:
            
            # Time in ms of this peak
            try:
                timepoint = big_abra_peaks.loc[(
                    HL_type, after_HL, n_experiment, mouse, channel, 
                    speaker_side, sound_level, wave_name
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

if PLOT_ABRA_PEAKS_AT_LOUDEST_ACROSS_MICE:
    """Plot ABR at loudest level across mice with ABRA peaks indicated
    
    """
    
    # Conditions: (suffix, dict of level->value to slice)
    condition_l = [
        #~ ('first_rec_VRL_only', dict(
            #~ after_HL=False, n_experiment=0, channel='VR', speaker_side='L')),
        ('preHL', dict(after_HL=False)),
        #~ ('postHL_bilateral', dict(after_HL=True, HL_type='bilateral')),
        #~ ('postHL_sham', dict(after_HL=True, HL_type='sham')),
        ]
    
    # One figure per condition
    for suffix, sel in condition_l:
        
        # Slice abrs and ABRA peaks to this condition
        this_abrs = averaged_abrs_by_date
        this_abra_peaks = big_abra_peaks
        for k, v in sel.items():
            this_abrs = this_abrs.xs(v, level=k)
            this_abra_peaks = this_abra_peaks.xs(v, level=k)
        
        # Levels that identify a recording (everything except label)
        rec_levels = [n for n in this_abrs.index.names if n != 'label']
        
        # Make figure handles
        f, ax = my.plot.figure_1x1_standard()
        
        # Iterate over recordings (lines)
        for rec_keys, rec_abr in this_abrs.groupby(rec_levels):
            
            # Drop recording levels, leaving label on the index
            rec_abr = rec_abr.droplevel(rec_levels)
            
            # Slice ABR at the loudest level
            try:
                this_abr = rec_abr.loc[loudest_db]
            except KeyError:
                continue
            
            # Plot the ABR trace for this recording
            ax.plot(
                this_abr.index.values / sampling_rate * 1000, 
                this_abr.values, 
                color='k', lw=.5, alpha=.5)
            
            # Slice ABRA peaks from this recording at the loudest level
            try:
                this_peaks = this_abra_peaks.loc[rec_keys].xs(
                    loudest_db, level='level')
            except KeyError:
                continue
            
            # Iterate over each peak
            for wave_name in this_peaks.index:
                
                # Color by wave name
                if wave_name not in include_waves:
                    continue
                color = wave_colors[wave_name]
                
                # Time in ms of this peak
                timepoint = this_peaks.loc[wave_name, 'timepoint']
                
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
        
        
        ## Savefig
        f.savefig(f'figures/PLOT_ABRA_PEAKS_AT_LOUDEST_ACROSS_MICE__{suffix}.svg')
        f.savefig(f'figures/PLOT_ABRA_PEAKS_AT_LOUDEST_ACROSS_MICE__{suffix}.png', dpi=300)
        

if PLOT_LATENCY_ABRA_VS_OURS:
    """Connected-pairs plot of our latency vs ABRA's"""

    # Seed so the jitter is reproducible
    np.random.seed(0)

    # Include only first pre-HL recording per mouse
    ridges_pre_HL = big_ridges.xs(
        False, level='after_HL').xs(0, level='n_experiment').droplevel('HL_type')
    abra_pre_HL = big_abra_peaks.xs(
        False, level='after_HL').xs(0, level='n_experiment').droplevel('HL_type')

    # Match the datasets by these columns
    match_keys = ['mouse', 'channel', 'speaker_side', 'level', 'wave_name']

    # Ours
    ours = ridges_pre_HL[
        ridges_pre_HL['wave_name'].isin(include_waves)
        ].reset_index().set_index(match_keys)['latency_ms']

    # ABRA
    theirs = abra_pre_HL.reset_index().set_index(match_keys)['latency_ms']
    
    # Duplicates would cause a cartesian expansion in the concat below
    assert not ours.index.duplicated().any()
    assert not theirs.index.duplicated().any()

    # Pair ours with theirs
    paired = pandas.concat(
        [ours.rename('our_ms'), theirs.rename('their_ms')], 
        axis=1)

    # Drop nulls (found by ours but not ABRA, this is very rare)
    drop_mask = paired['their_ms'].isnull()
    print(f'dropping {drop_mask.sum()} peaks found by ours but not ABRA')
    paired = paired.loc[~drop_mask]

    # Drop nulls (found by ABRA but not ours, this is common)
    drop_mask = paired['our_ms'].isnull()
    print(f'dropping {drop_mask.sum()} peaks found by ABRA but not ours')
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
        this_x = include_waves.index(wave_name)
        
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
    ax.set_xticks(range(len(include_waves)))
    ax.set_xticklabels(include_waves)
    ax.set_ylim((0, 10))
    ax.set_yticks((0, 2, 4, 6, 8, 10))
    ax.set_ylabel('time from click (ms)')
    ax.set_xlabel('wave name')
    my.plot.despine(ax)
    
    
    ## Savefig
    f.savefig('figures/PLOT_LATENCY_ABRA_VS_OURS.svg')
    f.savefig('figures/PLOT_LATENCY_ABRA_VS_OURS.png', dpi=300)


plt.show()