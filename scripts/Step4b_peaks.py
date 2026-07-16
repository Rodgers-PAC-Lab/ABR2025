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
import opensabr.peak_picking


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

# For loading results from abranalysis
abranalysis_path = os.path.expanduser(
    '~/mnt/cuttlefish/chris/data/20251214_abr_data/output_20260704_10ms')


## Params
sampling_rate = 16000

# The minimum length of a ridge in levels. Shorter ridges are discarded
# Lower is more sensitive for weak responses but might cause false labels
minimum_ridge_length = 5

# Define a set of priors on where waves typically are, which are used as 
# starting points for labeling waves
# Each tuple is (slope_us_per_db, latency_ms)
wave_centroids_pos = pandas.DataFrame.from_dict({
    'W0p': (-4.5, 0.6),
    'W1p': (-5.2, 1.36),
    'W2p': (-8.7, 2.3),
    'W3p': (-8.3, 3.2),
    'W4p': (-13.0, 4.2),
    'W5p': (-13.2, 5.2),
    'W6p': (-7.9, 6.1),
    'W7p': (-8.5, 7.0),
    }, 
    orient='index', 
    columns=['slope_us_per_db', 'latency_ms_at_ref_level'])
wave_centroids_pos.index.name = 'wave_name'

wave_centroids_neg = pandas.DataFrame.from_dict({
    'W0n': (-3.9, 0.9),
    'W1n': (-6.4, 1.8),
    'W2n': (-8.8, 2.7),
    'W3n': (-9.1, 3.6),
    'W4n': (-15.3, 4.7),
    'W5n': (-11.5, 5.7),
    'W6n': (-8.8, 6.4),
    'W7n': (-10.4, 7.2),
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
#~ big_abrs = pandas.read_pickle(
    #~ os.path.join(output_directory, 'big_abrs'))
#~ averaged_abrs_by_mouse = pandas.read_pickle(
    #~ os.path.join(output_directory, 'averaged_abrs_by_mouse'))
averaged_abrs_by_date = pandas.read_pickle(
    os.path.join(output_directory, 'averaged_abrs_by_date'))
trial_counts = pandas.read_pickle(
    os.path.join(output_directory, 'trial_counts'))

# Loudest dB
loudest_db = averaged_abrs_by_date.index.get_level_values('label').max()
    

## Set up data for ridge tracing
# Include only vertex-ear channels
averaged_abrs_by_date = averaged_abrs_by_date.reindex(
    ['VL', 'VR'], level='channel')
averaged_abrs_by_date = averaged_abrs_by_date.sort_index()

# Convert to uV
averaged_abrs_by_date = averaged_abrs_by_date * 1e6


## Load abranalysis peaks
# TODO: double check that post HL was excluded
big_abra_peaks = pandas.read_pickle(
    os.path.join(abranalysis_path, 'big_abra_peaks'))

# Drop RL
# TODO: this should never have been computed in the first place, fix upstream
big_abra_peaks = big_abra_peaks.drop('RL', level='channel')
big_abra_peaks.index = big_abra_peaks.index.remove_unused_levels()
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
    ['HL_type', 'after_HL', 'n_experiment', 'mouse', 'channel', 'speaker_side', 'label', 'timepoint']]
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

        # Fit a line
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
    # Of W1p, W1n, W2p and W4p (the ones we care about), the hardest one is 
    # Cacti_225 VL-L W4p with a cost of 0.14
    # There are a variety of assignments between 0.14 and 0.2 but not waves
    # we care about (so include them)
    # Then nothing above 0.253 (except for out-of-order assignments that get
    # dropped anyway)
    labeled_pos = opensabr.peak_picking.label_ridges(
        recording_ridge_coefs.loc['pos'], wave_centroids_pos, max_cost=0.15)
    labeled_neg = opensabr.peak_picking.label_ridges(
        recording_ridge_coefs.loc['neg'], wave_centroids_neg, max_cost=0.15)

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
group_levels = [
    'HL_type', 'after_HL', 'n_experiment', 'mouse', 'channel', 'speaker_side', 'level']
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
    ['HL_type', 'after_HL', 'n_experiment', 'mouse', 'channel', 'speaker_side', 'sign', 'n_ridge', 'wave_name']
    ].drop_duplicates())

# Unlabel - this two-step process avoids ChainedAssignmentError
if len(midx) > 0:
    print(f'warning: unlabeling {len(midx)} waves')
    print(midx.to_frame(index=False).to_string())
midx = midx.droplevel('wave_name')
big_ridges.loc[
    my.misc.slice_df_by_some_levels(big_ridges, midx).index, 
    'wave_name'] = np.nan
big_labeled_waves = big_labeled_waves.drop(
    my.misc.slice_df_by_some_levels(big_labeled_waves, midx).index, 
    )


## Print out highest cost assignments
print(big_labeled_waves.sort_values('cost').iloc[-30:])


## Save picked peaks
big_ridges.to_pickle(os.path.join(output_directory, 'big_ridges'))
big_labeled_waves.to_pickle(os.path.join(output_directory, 'big_labeled_waves'))


## Plot
PLOT_RIDGES_ACROSS_MICE = False
HISTOGRAM_WAVE_LATENCIES = False
STRIP_PLOT_LATENCIES = False
STRIP_PLOT_ABRA_LATENCIES = False
PLOT_EXAMPLE_WATERFALL = False
PLOT_EXAMPLE_WATERFALL_ABRA = False
PLOT_EXAMPLE_HEATMAP = False
PLOT_PEAKS_AT_LOUDEST_ACROSS_MICE_VRL_ONLY = False
PLOT_ABRA_PEAKS_AT_LOUDEST_ACROSS_MICE_VRL_ONLY = False
PLOT_PEAK_LATENCY_BY_WAVE_AND_CONFIG = False
PLOT_PEAK_HEIGHT_BY_WAVE_AND_CONFIG = False
PLOT_LATENCY_ABRA_VS_OURS = False
PLOT_PEAK_GROWTH_FUNCTIONS = False
PLOT_LABELING_DIAGNOSTIC = False
PLOT_DIAGNOSTIC_HEATMAP_ONE_RECORDING = False
PLOT_PEAK_GROWTH_FUNCTIONS_AFTER_HL = True
PLOT_PEAK_W1_W4_RATIO_AFTER_HL = True


if PLOT_RIDGES_ACROSS_MICE:
    """Plot ABR heatmaps with detected ridges overlaid for many mice
    
    This is a supplemental figure to show how the ridge-tracing algorithm
    handles variability in the ABR.
    
    Separately plots by after_HL * n_experiment
    """
    
    # channel * speaker_side configurations - RL is not analyzed
    config_l = [('VL', 'L'), ('VL', 'R'), ('VR', 'L'), ('VR', 'R')]

    # Color cycle for ridges
    cmap = plt.get_cmap('tab10')

    
    ## One figure per n_experiment * after_HL
    for n_experiment in [0, 1]:
        for after_HL in [False, True]:
        
            # Slice this after_HL and n_experiment
            this_abrs = averaged_abrs_by_date.xs(
                after_HL, level='after_HL').xs(n_experiment, level='n_experiment')
            this_ridges_all = big_ridges.xs(
                after_HL, level='after_HL').xs(n_experiment, level='n_experiment')
                
            # Pre-HL: HL_type doesn't matter, drop it
            if not after_HL:
                this_abrs = this_abrs.droplevel('HL_type')
                this_ridges_all = this_ridges_all.droplevel('HL_type')
            
            # Which mice to include -- subset because it's too many otherwise
            mouse_l = sorted(
                this_abrs.index.get_level_values('mouse').unique()
                )
            
            
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

                    # Get ABR for this mouse (xs is order-independent)
                    try:
                        abr_heatmap = this_abrs.xs(mouse, level='mouse').xs(
                            channel, level='channel').xs(
                            speaker_side, level='speaker_side')
                    except KeyError:
                        continue
                    
                    # Drop residual HL_type (post-HL) so index is just level
                    if 'HL_type' in abr_heatmap.index.names:
                        abr_heatmap = abr_heatmap.droplevel('HL_type')

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
                        this_ridges = this_ridges_all.xs(mouse, level='mouse').xs(
                            channel, level='channel').xs(
                            speaker_side, level='speaker_side')
                    except KeyError:
                        this_ridges = None
                    
                    # Drop residual HL_type (post-HL)
                    if this_ridges is not None and (
                            'HL_type' in this_ridges.index.names):
                        this_ridges = this_ridges.droplevel('HL_type')

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
            suffix = f"{'postHL' if after_HL else 'preHL'}_exp{n_experiment}"
            f.savefig(f'figures/PLOT_RIDGES_ACROSS_MICE_{suffix}.png', dpi=300)
            f.savefig(f'figures/PLOT_RIDGES_ACROSS_MICE_{suffix}.svg')

    
if HISTOGRAM_WAVE_LATENCIES:
    """Histogram the wave times separately by level"""
    
    # Three figures: pre-HL, post-HL bilateral, post-HL sham.
    # Pooled across n_experiment.

    # Bin by sample
    bins = averaged_abrs_by_date.columns.values / sampling_rate * 1000
    
    # Which waves to include
    include_waves = [
        'W1p', 'W1n', 'W2p', 'W2n', 'W3p', 'W3n', 'W4p', 'W4n', 'W5p', 'W5n']
    
    # Conditions: (suffix, sliced big_ridges)
    condition_l = [
        ('preHL', 
            big_ridges.xs(False, level='after_HL')),
        ('postHL_bilateral', 
            big_ridges.xs(True, level='after_HL').xs('bilateral', level='HL_type')),
        ('postHL_sham', 
            big_ridges.xs(True, level='after_HL').xs('sham', level='HL_type')),
        ]

    # All levels (from full dataset, so every figure has the same rows)
    all_levels = sorted(
        averaged_abrs_by_date.index.get_level_values('label').unique())[::-1]

    # One figure per condition
    for suffix, this_ridges in condition_l:
        
        # Levels to plot (fixed across conditions)
        levels = all_levels
        
        # One ax per level
        f, axa = plt.subplots(
            len(levels), 1, 
            sharex=True, sharey=True, 
            figsize=(4, 4.7)
            )
        f.subplots_adjust(left=.15, bottom=.13, top=.95, right=.95)

        # Plot each
        for ax, level in zip(axa, levels):
            
            # Slice (may be empty if no peaks at this level)
            try:
                level_ridges = this_ridges.xs(level, level='level')
            except KeyError:
                level_ridges = None
            
            # only hist if we have data
            if level_ridges is not None:
                
                # group
                grouped = level_ridges.groupby('wave_name', dropna=False)
                
                # Iterate over wave
                for wave_name, this_wave_df in grouped:
                
                    # Skip if we don't want to plot this one
                    if wave_name not in include_waves + [np.nan]:
                        continue
                    
                    # Color for this wave
                    if pandas.isnull(wave_name):
                        # There are very few unlabeled waves, so just skip them
                        continue
                    else:
                        color = wave_colors[wave_name]
                        histtype='stepfilled' if wave_name.endswith('p') else 'step'
                    
                    # Hist
                    ax.hist(
                        this_wave_df['timepoint'] / sampling_rate * 1000,
                        bins=bins, 
                        color=color, 
                        histtype=histtype,
                        alpha=1, 
                        lw=1,
                        )
            
            # ylabel
            if ax is axa[0] or ax is axa[-1]:
                ax.set_ylabel(level, rotation=0, va='center', ha='right')
            
            # Despine
            if ax is axa[-1]:
                my.plot.despine(ax, which=('left', 'top', 'right'))
            else:
                my.plot.despine(ax, which=('left', 'top', 'right', 'bottom'))
            ax.set_yticks([])

        # Pretty
        axa[len(axa) // 2].set_ylabel('sound level (dB SPL)', labelpad=20)
        axa[-1].set_xlabel('time from click (ms)')
        ax.set_xlim((1, 7))
        ax.set_xticks((1, 2, 3, 4, 5, 6, 7))
        
        
        ## Savefig
        f.savefig(f'figures/HISTOGRAM_WAVE_LATENCIES_{suffix}.png', dpi=300)
        f.savefig(f'figures/HISTOGRAM_WAVE_LATENCIES_{suffix}.svg')


if STRIP_PLOT_LATENCIES:
    """Strip plot the latencies (alternative to HISTOGRAM_WAVE_LATENCIES)
    
    Four figures are generated:
    - first_rec_VRL_only - only shows first recording per mouse, only shows
      VR-L, thus there is only one point per mouse * wave * level
    - preHL - pools over recordings, experiments, channels, and speaker side
      Showing all the points like this is useful to look for outlierse
    - postHL_bilateral, post_HL_sham - like preHL for the other groups
    """

    # Which waves to include
    include_waves = [
        'W1p', 'W1n', 'W2p', 'W2n', 'W3p', 'W3n', 'W4p', 'W4n', 'W5p', 'W5n']
    
    # Conditions: (suffix, sliced big_ridges)
    condition_l = [
        ('first_rec_VRL_only',
            big_ridges.xs(False, level='after_HL').xs(
                0, level='n_experiment').xs(
                'VR', level='channel').xs(
                'L', level='speaker_side')),
        ('preHL', 
            big_ridges.xs(False, level='after_HL')),
        ('postHL_bilateral', 
            big_ridges.xs(True, level='after_HL').xs('bilateral', level='HL_type')),
        ('postHL_sham', 
            big_ridges.xs(True, level='after_HL').xs('sham', level='HL_type')),
        ]
    
    # All levels (from full dataset, so every figure has the same rows)
    all_levels = sorted(
        averaged_abrs_by_date.index.get_level_values('label').unique())[::-1]
    
    # One figure per condition
    for suffix, this_ridges in condition_l:
        
        # Levels to plot (fixed across conditions)
        levels = all_levels
        
        # Slice out waves to plot
        topl = this_ridges[this_ridges['wave_name'].isin(include_waves)]

        # Assert max one point per wave per mouse per level
        if suffix == 'first_rec_VRL_only':
            assert topl.groupby(
                ['mouse', 'level', 'wave_name']).size().max() <= 1        
        
        # A single ax with each swarm at its own ypos
        f, ax = plt.subplots(figsize=(4, 4.7))
        f.subplots_adjust(left=.15, bottom=.13, top=.95, right=.95)

        # Strip plot the latency
        seaborn.stripplot(
            data=topl, x='latency_ms', y='level', hue='wave_name', 
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


        ## Savefig
        f.savefig(f'figures/STRIP_PLOT_LATENCIES_{suffix}.png', dpi=300)
        f.savefig(f'figures/STRIP_PLOT_LATENCIES_{suffix}.svg')


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


if PLOT_EXAMPLE_WATERFALL:
    """Example ABR, waterfall plot, labeled peaks"""
    
    # Pre-HL, first experiment
    this_abrs = averaged_abrs_by_date.xs(
        False, level='after_HL').xs(0, level='n_experiment')
    this_ridges = big_ridges.xs(
        False, level='after_HL').xs(0, level='n_experiment')
    
    # Choose example
    example_mouse = 'Cat_227'
    channel = 'VR'
    speaker_side = 'L'

    # Get ABR for this config
    example_abr = this_abrs.xs(example_mouse, level='mouse').xs(
        channel, level='channel').xs(
        speaker_side, level='speaker_side').droplevel('HL_type')

    # Get this config's labeled ridges
    example_ridges = this_ridges.xs(example_mouse, level='mouse').xs(
        channel, level='channel').xs(
        speaker_side, level='speaker_side').droplevel('HL_type')

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
    grouped = example_ridges.groupby(['sign', 'n_ridge'])
    for (sign, n_ridge), this_ridge in grouped:

        # Droplevel
        this_ridge = this_ridge.droplevel(['sign', 'n_ridge'])

        # Color by wave name
        wave_name = this_ridge['wave_name'].unique().item()
        if wave_name not in include_waves:
            continue
        color = wave_colors[wave_name]

        # Pull out x and y of each peak
        for sound_level in this_ridge.index:
            
            # Time in ms of this peak
            timepoint = this_ridge.loc[sound_level, 'timepoint']
            
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
    f.savefig('figures/PLOT_EXAMPLE_WATERFALL.png', dpi=300)
    f.savefig('figures/PLOT_EXAMPLE_WATERFALL.svg')


if PLOT_EXAMPLE_WATERFALL_ABRA:
    """Example waterfall with ABRA peaks
    
    This one is broken because averaged_abrs_by_mouse is no longer loaded,
    need to recompute ABRA peaks on the data used here    
    """
    # Choose example
    example_mouse = 'Cat_227'
    channel = 'VR'
    speaker_side = 'L'

    # Get ABR for this config
    example_abr = averaged_abrs_by_mouse.loc[
        example_mouse].loc[channel].loc[speaker_side]

    # Get this config's labeled ridges
    example_ridges = big_ridges.loc[example_mouse].loc[channel].loc[
        speaker_side]

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
                    example_mouse, channel, speaker_side, 
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


if PLOT_EXAMPLE_HEATMAP:
    """Like EXAMPLE_WATERFALL but as a heatmap"""
    
    # Pre-HL, first experiment
    this_abrs = averaged_abrs_by_date.xs(
        False, level='after_HL').xs(0, level='n_experiment')
    this_ridges = big_ridges.xs(
        False, level='after_HL').xs(0, level='n_experiment')

    # Choose example
    example_mouse = 'Cat_227'
    channel = 'VR'
    speaker_side = 'L'

    # Get ABR for this config
    example_abr = this_abrs.xs(example_mouse, level='mouse').xs(
        channel, level='channel').xs(
        speaker_side, level='speaker_side').droplevel('HL_type')

    # Get this config's labeled ridges
    example_ridges = this_ridges.xs(example_mouse, level='mouse').xs(
        channel, level='channel').xs(
        speaker_side, level='speaker_side').droplevel('HL_type')

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
    
    # Heatmap
    im = my.plot.imshow(
        example_abr, ax=ax, cmap='gray',
        x=example_abr.columns.values / sampling_rate * 1000,
        y=example_abr.index.values, origin='lower')
    im.set_clim((-2, 2))

    # Iterate over each ridge
    grouped = example_ridges.groupby(['sign', 'n_ridge'])
    for (sign, n_ridge), this_ridge in grouped:

        # Color by wave name
        wave_name = this_ridge['wave_name'].unique().item()
        if wave_name not in include_waves:
            continue
        color = wave_colors[wave_name]
        
        # Plot
        ax.plot(
            this_ridge['timepoint'] / sampling_rate * 1000, 
            this_ridge.index.get_level_values('level').values, 
            color=color, 
            ls='-', ms=2, mew=0)

    # Pretty
    ax.set_ylim((levels[0] - 2, levels[-1] + 2))
    ax.set_yticks((levels[0], levels[-1]))
    ax.set_xlabel('time from click (ms)')
    ax.set_ylabel('sound level (dB SPL)')
    ax.set_xlim((-1, 7))
    ax.set_xticks((0, 2, 4, 6))
    
    #~ # Colorbar spanning both columns
    #~ cb = f.colorbar(im, ax=ax, fraction=.05, pad=.02)
    #~ cb.set_label('ABR (' + MU + 'V)')
    #~ cb.set_ticks([-3, 0, 3])
    #~ f.subplots_adjust(right=.8)

    # Savefig
    f.savefig('figures/PLOT_EXAMPLE_HEATMAP.png', dpi=300)
    f.savefig('figures/PLOT_EXAMPLE_HEATMAP.svg')
    

if PLOT_PEAKS_AT_LOUDEST_ACROSS_MICE_VRL_ONLY:
    """Plot VR-L ABR at loudest level across mice with peaks circled
    
    Four conditions:
    - first_rec_VRL_only : for an example with one line per mouse
    - preHL : all vertex-ear pre-HL (possibly multiple recording per mouse)
    - postHL_bilateral, postHL_sham : analogous to preHL
    """
    
    # Which waves to include
    include_waves = [
        'W1p', 'W1n', 'W2p', 'W2n', 'W3p', 'W3n', 'W4p', 'W4n', 'W5p', 'W5n']
    
    # Conditions: (suffix, dict of level->value to slice)
    condition_l = [
        ('first_rec_VRL_only', dict(after_HL=False, n_experiment=0, channel='VR', speaker_side='L')),
        ('preHL', dict(after_HL=False)),
        ('postHL_bilateral', dict(after_HL=True, HL_type='bilateral')),
        ('postHL_sham', dict(after_HL=True, HL_type='sham')),
        ]
    
    # One figure per condition
    for suffix, sel in condition_l:
        
        # Slice abrs and ridges to this condition
        this_abrs = averaged_abrs_by_date
        this_ridges = big_ridges
        for k, v in sel.items():
            this_abrs = this_abrs.xs(v, level=k)
            this_ridges = this_ridges.xs(v, level=k)
        
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

            # Slice peaks from this recording at the loudest level
            try:
                this_peaks = this_ridges.loc[rec_keys].xs(
                    loudest_db, level='level')
            except KeyError:
                continue
        
            # Iterate over each peak
            for (sign, n_ridge) in this_peaks.index:

                # Color by wave name
                wave_name = this_peaks.loc[(sign, n_ridge), 'wave_name']
                if wave_name not in include_waves:
                    continue
                color = wave_colors[wave_name]

                # Time in ms of this peak
                timepoint = this_peaks.loc[(sign, n_ridge), 'timepoint']
                
                # Convert to ms
                t = timepoint / sampling_rate * 1000
                
                # Voltage of this peak
                y = this_abr.loc[timepoint]
                
                # Plot
                ax.plot(
                    [t], [y], 
                    marker='o', color=color, ms=4, alpha=.75, clip_on=False)
        

        ## Pretty
        ax.set_xlim((-1, 7))
        ax.set_xticks((0, 2, 4, 6))
        ax.set_yticks((-3, 0, 3))
        my.plot.despine(ax)
        ax.set_xlabel('time from click (ms)')
        ax.set_ylabel(f'ABR ({MU}V)')
        
        # Savefig
        f.savefig(f'figures/PLOT_PEAKS_AT_LOUDEST_ACROSS_MICE_VRL_ONLY_{suffix}.svg')
        f.savefig(f'figures/PLOT_PEAKS_AT_LOUDEST_ACROSS_MICE_VRL_ONLY_{suffix}.png', dpi=300)

if PLOT_ABRA_PEAKS_AT_LOUDEST_ACROSS_MICE_VRL_ONLY:
    """Currently broken till we re-run ABR on individual recordings"""

    # Mice to plot
    mouse_l = sorted(
        averaged_abrs_by_mouse.index.get_level_values('mouse').unique())

    # Levels to plot
    levels = sorted(
        averaged_abrs_by_mouse.index.get_level_values('label').unique())

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
        this_abr = averaged_abrs_by_mouse.loc[
            (mouse, channel, speaker_side, loudest_db)]
        
        # Slice peaks from this mouse at the loudest level
        this_peaks = big_ridges.loc[(mouse, channel, speaker_side)].xs(
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


if PLOT_PEAK_LATENCY_BY_WAVE_AND_CONFIG:
    """Connected pairs strip plot of the latency for each wave * config
    
    Waves in subplots, channel * speaker_side on x-axis.
    Four conditions:
    - first_rec : one recording per mouse; mouse is the paired unit; stats run
    - preHL, postHL_bilateral, postHL_sham : pooled, recording is the unit,
      separate lines per recording, no stats
    """

    # Which waves to include
    wave_l = ['W1p', 'W1n', 'W2p', 'W4p']

    # Config order for the x-axis
    config_order = ['VL L', 'VL R', 'VR L', 'VR R']

    # Conditions: (suffix, sel dict, run_stats)
    condition_l = [
        ('first_rec', dict(after_HL=False, n_experiment=0), True),
        ('preHL', dict(after_HL=False), False),
        ('postHL_bilateral', dict(after_HL=True, HL_type='bilateral'), False),
        ('postHL_sham', dict(after_HL=True, HL_type='sham'), False),
        ]

    for suffix, sel, run_stats in condition_l:

        # Slice big_ridges to this condition
        this_ridges = big_ridges
        for k, v in sel.items():
            this_ridges = this_ridges.xs(v, level=k)

        # Extract labeled peaks at the loudest level only
        loudest = this_ridges.dropna(subset='wave_name').xs(
            loudest_db, level='level').copy()
        loudest = loudest.reset_index()

        # Form "config" as a single ordered label for the x-axis
        loudest['config'] = loudest['channel'] + ' ' + loudest['speaker_side']

        # Also add an ipsi column
        loudest['ipsi'] = loudest['speaker_side'] == loudest['channel'].str[1]

        # Replicate unit (recording); collapses to per-mouse for first_rec
        unit_levels = [
            c for c in ['HL_type', 'after_HL', 'n_experiment', 'mouse']
            if c in loudest.columns]

        # One subplot per wave, in this order
        f, axa = plt.subplots(
            1, len(wave_l),
            sharex=True, sharey=True,
            figsize=(8, 2.5)
            )
        f.subplots_adjust(bottom=.24, left=.1, right=.95, top=.89, wspace=.4)

        # Stats accumulators (first_rec only)
        aov_pvals_l = []
        tt_pvals_l = []
        stats_data_l = []
        stats_keys_l = []

        # Iterate over waves (subplots)
        for wave_name, ax in zip(wave_l, axa):

            # Slice this wave
            this_wave = loudest[loudest['wave_name'] == wave_name].copy()

            # Strip plot the latency for each config
            seaborn.stripplot(
                this_wave, x='config', y='latency_ms',
                marker='$\circ$', color='k', alpha=.5,
                order=config_order, ax=ax)

            # Connect configs within a recording
            for unit_keys, unit_df in this_wave.groupby(unit_levels):
                unit_df = unit_df.set_index('config').reindex(config_order)
                ax.plot(
                    range(len(config_order)),
                    unit_df['latency_ms'].values,
                    ls='-', color='gray', alpha=.5, lw=.75, clip_on=False,
                    )

            # Fancy x-axis
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels(['L', 'R', 'L', 'R'], rotation=0)
            ax.text(0.5, 0, 'VL', ha='center', va='center')
            ax.text(2.5, 0, 'VR', ha='center', va='center')

            # Pretty
            ax.set_title(wave_name)
            ax.set_xlabel('')
            ax.set_ylabel('latency (ms)')
            ax.set_ylim((1, 5))
            ax.set_yticks((1, 3, 5))
            my.plot.despine(ax)

            # Stats (first_rec only, mouse is the paired unit)
            if run_stats:
                # Assert that we have data from all mice on all configs
                assert (this_wave.groupby('mouse')['config'].nunique() == 4).all()

                this_wave['latency_centered'] = (
                    this_wave['latency_ms'] - this_wave['latency_ms'].mean())
                aov = my.stats.anova(
                    this_wave, 'latency_centered ~ ipsi + speaker_side + mouse')
                aov_pvals_l.append(aov['pvals'])

                # Post hoc
                to_test = this_wave.set_index(
                    ['channel', 'ipsi', 'mouse'])['latency_ms'].unstack('mouse').T
                ttp_VL = scipy.stats.ttest_rel(
                    to_test[('VL', True)].values, to_test[('VL', False)].values
                    ).pvalue
                ttp_VR = scipy.stats.ttest_rel(
                    to_test[('VR', True)].values, to_test[('VR', False)].values
                    ).pvalue
                tt_pvals_l.append(pandas.Series({'VL': ttp_VL, 'VR': ttp_VR}))
                stats_data_l.append(to_test)
                stats_keys_l.append(wave_name)

        # Stats output (first_rec only)
        if run_stats:
            big_aov = pandas.concat(aov_pvals_l, keys=stats_keys_l, names=['wave'])
            big_tt = pandas.concat(tt_pvals_l, keys=stats_keys_l, names=['wave'])
            big_stats_data = pandas.concat(
                stats_data_l, keys=stats_keys_l, names=['wave'])

            n_configs_by_mouse = big_stats_data.groupby('mouse').size()
            assert (n_configs_by_mouse == 4).all()
            n_mice = len(n_configs_by_mouse)

            mean_latency = big_stats_data.groupby('wave').mean().T.groupby('ipsi').mean().T
            mean_latency['diff'] = mean_latency.loc[:, True] - mean_latency.loc[:, False]
            mean_latency = mean_latency.T

            stats_filename = 'figures/STATS__PLOT_PEAK_LATENCY_BY_WAVE_AND_CONFIG'
            with open(stats_filename, 'w') as fi:
                fi.write(stats_filename + '\n')
                fi.write(f"n = {n_mice} mice\n")
                fi.write(f"AOV by wave:\n{big_aov.unstack('wave')}\n")
                fi.write(f"paired t-test by wave:\n{big_tt.unstack('wave')}\n")
                fi.write(f"mean latency by wave and ipsi:\n{mean_latency}\n")
            with open(stats_filename) as fi:
                print(''.join(fi.readlines()))

        ## Savefig
        f.savefig(f'figures/PLOT_PEAK_LATENCY_BY_WAVE_AND_CONFIG_{suffix}.svg')
        f.savefig(f'figures/PLOT_PEAK_LATENCY_BY_WAVE_AND_CONFIG_{suffix}.png', dpi=300)


if PLOT_PEAK_HEIGHT_BY_WAVE_AND_CONFIG:
    """Connected pairs strip plot of the height for each wave * config
    
    Waves in subplots, channel * speaker_side on x-axis.
    Four conditions:
    - first_rec : one recording per mouse; mouse is the paired unit; stats run
    - preHL, postHL_bilateral, postHL_sham : pooled, recording is the unit,
      separate lines per recording, no stats
    """

    # Which waves to include
    wave_l = ['W1p', 'W1n', 'W2p', 'W4p']

    # Config order for the x-axis
    config_order = ['VL L', 'VL R', 'VR L', 'VR R']

    # Conditions: (suffix, sel dict, run_stats)
    condition_l = [
        ('first_rec', dict(after_HL=False, n_experiment=0), True),
        ('preHL', dict(after_HL=False), False),
        ('postHL_bilateral', dict(after_HL=True, HL_type='bilateral'), False),
        ('postHL_sham', dict(after_HL=True, HL_type='sham'), False),
        ]

    for suffix, sel, run_stats in condition_l:

        # Slice big_ridges to this condition
        this_ridges = big_ridges
        for k, v in sel.items():
            this_ridges = this_ridges.xs(v, level=k)

        # Extract labeled peaks at the loudest level only
        loudest = this_ridges.dropna(subset='wave_name').xs(
            loudest_db, level='level').copy()
        loudest = loudest.reset_index()

        # Form "config" as a single ordered label for the x-axis
        loudest['config'] = loudest['channel'] + ' ' + loudest['speaker_side']

        # Also add an ipsi column
        loudest['ipsi'] = loudest['speaker_side'] == loudest['channel'].str[1]

        # Replicate unit (recording); collapses to per-mouse for first_rec
        unit_levels = [
            c for c in ['HL_type', 'after_HL', 'n_experiment', 'mouse']
            if c in loudest.columns]

        # One subplot per wave, in this order
        f, axa = plt.subplots(
            1, len(wave_l),
            sharex=True, sharey=True,
            figsize=(8, 2.5)
            )
        f.subplots_adjust(bottom=.24, left=.1, right=.95, top=.89, wspace=.4)

        # Stats accumulators (first_rec only)
        aov_pvals_l = []
        aov_fit_l = []
        tt_pvals_l = []
        stats_data_l = []
        stats_keys_l = []

        # Iterate over waves (subplots)
        for wave_name, ax in zip(wave_l, axa):

            # Slice this wave
            this_wave = loudest[loudest['wave_name'] == wave_name].copy()

            # Invert if trough
            if wave_name.endswith('n'):
                this_wave['height'] = -this_wave['height']

            # Strip plot the height for each config
            seaborn.stripplot(
                this_wave, x='config', y='height',
                marker='$\circ$', color='k', alpha=.5,
                order=config_order, ax=ax)

            # Connect configs within a recording
            for unit_keys, unit_df in this_wave.groupby(unit_levels):
                unit_df = unit_df.set_index('config').reindex(config_order)
                ax.plot(
                    range(len(config_order)),
                    unit_df['height'].values,
                    ls='-', color='gray', alpha=.5, lw=.75,
                    )

            # Fancy x-axis
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels(['L', 'R', 'L', 'R'], rotation=0)
            ax.text(
                0.5, -0.25, 'VL', ha='center', va='center',
                transform=ax.get_xaxis_transform())
            ax.text(
                2.5, -0.25, 'VR', ha='center', va='center',
                transform=ax.get_xaxis_transform())

            # Pretty
            ax.set_title(wave_name)
            ax.set_xlabel('')
            ax.set_ylabel(f'peak height ({MU}V)')
            ax.set_ylim((-0.2, 6))
            ax.set_yticks((0, 3, 6))
            my.plot.despine(ax)

            # Stats (first_rec only, mouse is the paired unit)
            if run_stats:
                # Assert that we have data from all mice on all configs
                assert (this_wave.groupby('mouse')['config'].nunique() == 4).all()

                aov = my.stats.anova(
                    this_wave, 'height ~ channel + ipsi + speaker_side + mouse')
                aov_pvals_l.append(aov['pvals'])
                aov_fit_l.append(
                    aov['fit'].loc[~aov['fit'].index.str.startswith('fit_mouse')]
                    )

                # Post hoc
                to_test = this_wave.set_index(
                    ['channel', 'ipsi', 'speaker_side', 'mouse'])['height'].unstack('mouse').T
                ttp_VL = scipy.stats.ttest_rel(
                    to_test[('VL', True)].values, to_test[('VL', False)].values
                    ).pvalue
                ttp_VR = scipy.stats.ttest_rel(
                    to_test[('VR', True)].values, to_test[('VR', False)].values
                    ).pvalue
                tt_pvals_l.append(pandas.Series({'VL': ttp_VL, 'VR': ttp_VR}))
                stats_data_l.append(to_test)
                stats_keys_l.append(wave_name)

        # Stats output (first_rec only)
        if run_stats:
            big_aov_pvals = pandas.concat(aov_pvals_l, keys=stats_keys_l, names=['wave'])
            big_aov_fit = pandas.concat(aov_fit_l, keys=stats_keys_l, names=['wave'])
            big_tt = pandas.concat(tt_pvals_l, keys=stats_keys_l, names=['wave'])
            big_stats_data = pandas.concat(
                stats_data_l, keys=stats_keys_l, names=['wave'])

            n_configs_by_mouse = big_stats_data.groupby('mouse').size()
            assert (n_configs_by_mouse == 4).all()
            n_mice = len(n_configs_by_mouse)

            # Sigstr
            big_tt_sigstr = big_tt.apply(my.stats.pvalue_to_significance_string)
            big_aov_sigstr = big_aov_pvals.apply(my.stats.pvalue_to_significance_string)

            # Mean height and diff
            mean_height = big_stats_data.groupby('wave').mean().T.groupby('ipsi').mean().T
            mean_height['diff'] = mean_height.loc[:, True] - mean_height.loc[:, False]
            mean_height = mean_height.T

            stats_filename = 'figures/STATS__PLOT_PEAK_HEIGHT_BY_WAVE_AND_CONFIG'
            with open(stats_filename, 'w') as fi:
                fi.write(stats_filename + '\n')
                fi.write(f"n = {n_mice} mice\n")
                fi.write(f"mean heights:\n{big_stats_data.groupby('wave').mean().mean(axis=1)}\n")
                fi.write(f"AOV pvals by wave:\n{big_aov_pvals.unstack('wave')}\n")
                fi.write(f"AOV fit by wave:\n{big_aov_fit.unstack('wave')}\n")
                fi.write(f"AOV sig by wave:\n{big_aov_sigstr.unstack('wave')}\n")
                fi.write(f"paired t-test p-value by wave:\n{big_tt.unstack('wave')}\n")
                fi.write(f"paired t-test sigstr by wave:\n{big_tt_sigstr.unstack('wave')}\n")
                fi.write(f"mean height by wave and ipsi:\n{mean_height}\n")
            with open(stats_filename) as fi:
                print(''.join(fi.readlines()))

        ## Savefig
        f.savefig(f'figures/PLOT_PEAK_HEIGHT_BY_WAVE_AND_CONFIG_{suffix}.svg')
        f.savefig(f'figures/PLOT_PEAK_HEIGHT_BY_WAVE_AND_CONFIG_{suffix}.png', dpi=300)


if PLOT_LATENCY_ABRA_VS_OURS:
    """Broken until ABRA is re-run"""
    ## Connected-pairs plot of our latency vs ABRA's
    # Match the datasets by these columns
    match_keys = ['mouse', 'channel', 'speaker_side', 'level', 'wave_name']

    # Waves to include
    wave_order = [
        'W1p', 'W1n', 'W2p', 'W2n', 'W3p', 'W3n', 'W4p', 'W4n', 'W5p', 'W5n']

    # Ours
    ours = big_ridges[
        big_ridges['wave_name'].isin(wave_order)
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


if PLOT_PEAK_GROWTH_FUNCTIONS:
    """Plot peak-amplitude growth functions vs sound level, colored by wave
    
    Uses positive peaks (height of Wn_p); no normalization.
    Four conditions:
    - first_rec : one recording per mouse (mouse is the replicate)
    - preHL, postHL_bilateral, postHL_sham : pooled, recording is the replicate
      (no mean within mouse), so sem is over recordings
    """

    # Waves to include / color
    wave_l = ['W1p', 'W2p', 'W3p', 'W4p', 'W5p']
    wave_l = ['W1p', 'W4p']
    aut_colorbar = my.plot.generate_colorbar(
        len(wave_l), mapname='viridis_r', start=0, stop=1)
    aut_colorbar = np.array(plt.cm.tab10.colors)[[1, 4]]

    # Panels
    channel_l = ['VL', 'VR']
    speaker_side_l = ['L', 'R']

    # Conditions: (suffix, sel dict, plot_individual)
    condition_l = [
        ('first_rec', dict(after_HL=False, n_experiment=0), False),
        ('preHL', dict(after_HL=False), False),
        ('postHL_bilateral', dict(after_HL=True, HL_type='bilateral'), True),
        ('postHL_sham', dict(after_HL=True, HL_type='sham'), True),
        ]

    for suffix, sel, plot_individual in condition_l:

        # Slice big_ridges to this condition
        this_ridges = big_ridges
        for k, v in sel.items():
            this_ridges = this_ridges.xs(v, level=k)

        # Positive peaks only, keep only the waves we plot
        this = this_ridges.xs('pos', level='sign')
        this = this[this['wave_name'].isin(wave_l)]

        # Recording levels that survive the slice (the replicate unit)
        recording_levels = [
            c for c in ['HL_type', 'after_HL', 'n_experiment', 'mouse']
            if c in this.index.names]

        # height indexed by recording * channel * speaker_side * level * wave_name
        h = this.set_index('wave_name', append=True)['height'].droplevel('n_ridge')

        # Collapse any duplicate ridges mapped to same recording*config*level*wave
        group_keys = recording_levels + [
            'channel', 'speaker_side', 'level', 'wave_name']
        h = h.groupby(group_keys).mean()

        # Recordings as replicates
        to_agg = h.unstack(recording_levels).sort_index()

        # Aggregate (only used for the mean+sem path)
        if not plot_individual:
            # index=(channel, speaker_side, level), cols=wave
            agg_mean = to_agg.mean(axis=1).unstack('wave_name')
            agg_err = to_agg.sem(axis=1).unstack('wave_name')

        # Figure
        f, axa = plt.subplots(
            len(channel_l), len(speaker_side_l),
            sharex=True, sharey=True, figsize=(5, 4))
        f.subplots_adjust(
            left=.17, right=.89, top=.95, bottom=.15, hspace=.15, wspace=.12)

        # Plot each channel * speaker_side
        for channel in channel_l:
            for speaker_side in speaker_side_l:

                # Get ax
                ax = axa[
                    channel_l.index(channel),
                    speaker_side_l.index(speaker_side),
                ]

                if plot_individual:
                    # One line per recording, colored by wave
                    # Slice this config; index=(recording..., level), cols=wave
                    try:
                        sub = to_agg.xs(
                            channel, level='channel').xs(
                            speaker_side, level='speaker_side')
                    except KeyError:
                        continue

                    # Plot each wave
                    for n_wave, wave in enumerate(wave_l):
                        color = aut_colorbar[n_wave]

                        # level on rows, recordings on columns
                        try:
                            traces = sub.xs(wave, level='wave_name')
                        except KeyError:
                            continue

                        # One line per recording (columns), x=level (index)
                        for rec in traces.columns:
                            col = traces[rec]
                            ax.plot(
                                col.index, col.values,
                                color=color, lw=.75, alpha=1)
                
                else:
                    # Mean +- sem over recordings
                    try:
                        subdf = agg_mean.xs(
                            channel, level='channel').xs(
                            speaker_side, level='speaker_side')
                        subdf_err = agg_err.xs(
                            channel, level='channel').xs(
                            speaker_side, level='speaker_side')
                    except KeyError:
                        continue

                    # Plot each wave
                    for n_wave, wave in enumerate(wave_l):
                        color = aut_colorbar[n_wave]

                        ax.plot(
                            subdf.index, subdf[wave], color=color, lw=1)
                        ax.fill_between(
                            subdf.index,
                            (subdf[wave] - subdf_err[wave]),
                            (subdf[wave] + subdf_err[wave]),
                            color=color, alpha=.5, lw=0)

                # Pretty
                my.plot.despine(ax)
        
        # Legend
        for n_wave, wave in enumerate(wave_l):
            f.text(
                .95, .68 - n_wave * .05, f'wave {wave[1]}',
                color=aut_colorbar[n_wave], ha='center', va='center', size=12)

        # Pretty
        f.text(.52, .01, 'sound level (dB SPL)', ha='center', va='bottom')
        f.text(.02, .56, f'peak amplitude ({MU}V)',
            rotation=90, ha='center', va='center')

        ax.set_yscale('log')
        ax.set_ylim((.1, 10))
        ax.set_xlim((20, 80))
        ax.set_xticks((30, 50, 70))

        # Label the channel
        for n_channel, channel in enumerate(channel_l):
            axa[n_channel, 0].set_ylabel(channel)

        # Label the speaker side
        axa[0, 0].set_title('sound from left')
        axa[0, 1].set_title('sound from right')


        ## Savefig
        f.savefig(os.path.join('figures',
            f"PLOT_PEAK_GROWTH_FUNCTIONS_{suffix}.svg"))
        f.savefig(os.path.join('figures',
            f"PLOT_PEAK_GROWTH_FUNCTIONS_{suffix}.png"), dpi=300)    

if PLOT_LABELING_DIAGNOSTIC:
    """Scatter ridge coefs in (slope, latency) space vs wave centroids
    
    Checks whether the centroid priors are positioned so the Hungarian
    assignment labels ridges correctly. Each point is one labeled ridge,
    colored by its assigned wave; stars mark the centroid priors.
    Rows: pre-HL, post-HL bilateral, post-HL sham. Columns: pos, neg.
    """

    # Rows: (row label, sel dict); columns: sign
    condition_l = [
        ('preHL', dict(after_HL=False)),
        ('postHL_bilateral', dict(after_HL=True, HL_type='bilateral')),
        ('postHL_sham', dict(after_HL=True, HL_type='sham')),
        ]
    sign_l = ['pos', 'neg']
    centroids_by_sign = {'pos': wave_centroids_pos, 'neg': wave_centroids_neg}

    # Figure
    f, axa = plt.subplots(
        len(condition_l), len(sign_l),
        sharex=True, sharey=True, figsize=(9, 9))
    f.subplots_adjust(
        left=.1, right=.87, bottom=.08, top=.94, wspace=.15, hspace=.15)

    # Plot each condition (row) * sign (column)
    for (suffix, sel), axa_row in zip(condition_l, axa):

        # Slice big_labeled_waves to this condition
        this_condition = big_labeled_waves
        for k, v in sel.items():
            this_condition = this_condition.xs(v, level=k)

        # Plot each sign
        for sign, ax in zip(sign_l, axa_row):

            # Slice this sign's labeled ridges
            this = this_condition.xs(sign, level='sign')

            # Plot each labeled ridge, colored by assigned wave
            for wave_name, this_wave in this.groupby('wave_name'):
                color = wave_colors[wave_name]

                # Scatter the ridge coefs
                ax.scatter(
                    this_wave['latency_ms_at_ref_level'],
                    this_wave['slope_us_per_db'],
                    s=8, color=color, alpha=.4, lw=0)

            # Overplot the centroid priors as stars
            centroids = centroids_by_sign[sign]
            for wave_name, centroid in centroids.iterrows():

                # Star at the prior, black-edged so it reads over the cloud
                ax.scatter(
                    centroid['latency_ms_at_ref_level'],
                    centroid['slope_us_per_db'],
                    marker='*', s=140, color=wave_colors[wave_name],
                    edgecolor='k', linewidth=.5, zorder=5)

                # Label the wave next to its centroid
                ax.annotate(
                    wave_name,
                    (centroid['latency_ms_at_ref_level'],
                     centroid['slope_us_per_db']),
                    textcoords='offset points', xytext=(4, 4), size='small')

            # Pretty
            my.plot.despine(ax)

    # Column titles on the top row
    for sign, ax in zip(sign_l, axa[0]):
        ax.set_title(sign)

    # Row labels on the left column
    for (suffix, sel), ax in zip(condition_l, axa[:, 0]):
        ax.set_ylabel(f"{suffix}\nslope ({MU}s / dB)")

    # x-labels on the bottom row
    for ax in axa[-1]:
        ax.set_xlabel('latency at ref level (ms)')

    ## Savefig
    f.savefig('figures/PLOT_LABELING_DIAGNOSTIC.png', dpi=300)
    f.savefig('figures/PLOT_LABELING_DIAGNOSTIC.svg')

if PLOT_DIAGNOSTIC_HEATMAP_ONE_RECORDING:
    """Heatmap of one recording with labeled ridges overlaid
    
    Diagnostic for W2/W3 resolution. Ridges colored by assigned wave;
    unlabeled ridges in black. 
    
    Some good ones to check:
    Cacti_225 VL L, pre-HL, exp 0
    Pineapple_197 VL R, pre-HL, exp 0
    """

    # Choose recording
    example_mouse = 'Pineapple_197' # 'Cacti_225'
    channel = 'VL'
    speaker_side = 'R' #'L'
    sel = dict(after_HL=False, n_experiment=0)

    # Slice abrs and ridges to this condition
    this_abrs = averaged_abrs_by_date
    this_ridges = big_ridges
    for k, v in sel.items():
        this_abrs = this_abrs.xs(v, level=k)
        this_ridges = this_ridges.xs(v, level=k)

    # Slice this recording, drop residual HL_type
    example_abr = this_abrs.xs(example_mouse, level='mouse').xs(
        channel, level='channel').xs(
        speaker_side, level='speaker_side').droplevel('HL_type')
    example_ridges = this_ridges.xs(example_mouse, level='mouse').xs(
        channel, level='channel').xs(
        speaker_side, level='speaker_side').droplevel('HL_type')

    # Sort ascending by level so heatmap rows align with ridge levels
    example_abr = example_abr.sort_index()

    # Which waves to include
    include_waves = [
        'W1p', 'W1n', 'W2p', 'W2n', 'W3p', 'W3n', 'W4p', 'W4n', 'W5p', 'W5n']

    # Levels for ticks
    levels = sorted(example_abr.index)

    # Figure
    f, ax = my.plot.figure_1x1_standard()

    # Heatmap
    im = my.plot.imshow(
        example_abr, ax=ax, cmap='gray',
        x=example_abr.columns.values / sampling_rate * 1000,
        y=example_abr.index.values, origin='lower')
    im.set_clim((-2, 2))

    # Iterate over each ridge
    for (sign, n_ridge), this_ridge in example_ridges.groupby(['sign', 'n_ridge']):

        # Color by wave name; unlabeled in black
        wave_name = this_ridge['wave_name'].unique().item()
        if pandas.isnull(wave_name):
            color = 'k'
        elif wave_name in include_waves:
            color = wave_colors[wave_name]
        else:
            continue

        # Plot the ridge
        ax.plot(
            this_ridge['timepoint'] / sampling_rate * 1000,
            this_ridge.index.get_level_values('level').values,
            color=color, ls='-', lw=1)

        # Label the ridge at its top (loudest) level
        if not pandas.isnull(wave_name):
            top = this_ridge.iloc[-1]
            ax.annotate(
                wave_name,
                (top['timepoint'] / sampling_rate * 1000,
                 this_ridge.index.get_level_values('level')[-1]),
                textcoords='offset points', xytext=(3, 0),
                color=color, size='small')

    # Pretty
    ax.set_ylim((levels[0] - 2, levels[-1] + 2))
    ax.set_yticks((levels[0], levels[-1]))
    ax.set_xlabel('time from click (ms)')
    ax.set_ylabel('sound level (dB SPL)')
    ax.set_xlim((-1, 7))
    ax.set_xticks((0, 2, 4, 6))
    ax.set_title(f'{example_mouse} {channel} {speaker_side}')

    ## Savefig
    f.savefig('figures/PLOT_DIAGNOSTIC_HEATMAP_ONE_RECORDING.png', dpi=300)
    f.savefig('figures/PLOT_DIAGNOSTIC_HEATMAP_ONE_RECORDING.svg')


if PLOT_PEAK_GROWTH_FUNCTIONS_AFTER_HL:
    """Peak-amplitude growth functions after HL, colored by HL_type
    
    Uses positive peaks (height of Wn_p); no normalization.
    after_HL==True only, pooled across n_experiment. One line per recording,
    colored by HL_type (bilateral red, sham gray). No error bars.
    One figure per wave (W1p, W4p); each is 2x2 over channel * speaker_side.
    """

    # Waves to plot (one figure each)
    wave_l = ['W1p', 'W4p']

    # Color by HL_type
    hl_colors = {'bilateral': 'red', 'sham': 'gray'}

    # Panels
    channel_l = ['VL', 'VR']
    speaker_side_l = ['L', 'R']

    # Slice to after_HL, positive peaks, just the waves we plot
    this_ridges = big_ridges.xs(True, level='after_HL')
    this = this_ridges.xs('pos', level='sign')
    this = this[this['wave_name'].isin(wave_l)]

    # height indexed by recording * config * level * wave_name
    h = this.set_index('wave_name', append=True)['height'].droplevel('n_ridge')

    # Collapse any duplicate ridges mapped to same recording*config*level*wave
    recording_levels = ['HL_type', 'n_experiment', 'mouse']
    group_keys = recording_levels + [
        'channel', 'speaker_side', 'level', 'wave_name']
    h = h.groupby(group_keys).mean()

    # One figure per wave
    for wave in wave_l:

        # Slice this wave
        this_wave = h.xs(wave, level='wave_name')

        # Recordings on columns, (config, level) on rows
        to_agg = this_wave.unstack(recording_levels).sort_index()

        # Figure
        f, axa = plt.subplots(
            len(channel_l), len(speaker_side_l),
            sharex=True, sharey=True, figsize=(5, 4))
        f.subplots_adjust(
            left=.17, right=.89, top=.95, bottom=.15, hspace=.15, wspace=.12)

        # Plot each channel * speaker_side
        for channel in channel_l:
            for speaker_side in speaker_side_l:

                # Get ax
                ax = axa[
                    channel_l.index(channel),
                    speaker_side_l.index(speaker_side),
                ]

                # Slice this config; index=level, cols=recording
                try:
                    sub = to_agg.xs(channel, level='channel').xs(
                        speaker_side, level='speaker_side')
                except KeyError:
                    continue

                # One line per recording, colored by HL_type
                for rec in sub.columns:

                    # rec is a tuple (HL_type, n_experiment, mouse)
                    hl_type = rec[recording_levels.index('HL_type')]
                    color = hl_colors[hl_type]

                    # Plot this recording's growth function
                    col = sub[rec]
                    ax.plot(
                        col.index, col.values,
                        color=color, lw=.75, alpha=1)

                # Pretty
                my.plot.despine(ax)

        # Legend
        for n_hl, (hl_type, color) in enumerate(hl_colors.items()):
            f.text(
                .95, .68 - n_hl * .05, hl_type,
                color=color, ha='center', va='center', size=12)

        # Pretty
        f.text(.52, .01, 'sound level (dB SPL)', ha='center', va='bottom')
        f.text(.02, .56, f'peak amplitude ({MU}V)',
            rotation=90, ha='center', va='center')
        ax.set_yscale('log')
        ax.set_ylim((.1, 10))
        ax.set_xlim((20, 80))
        ax.set_xticks((30, 50, 70))

        # Label the channel
        for n_channel, channel in enumerate(channel_l):
            axa[n_channel, 0].set_ylabel(channel)

        # Label the speaker side
        axa[0, 0].set_title('sound from left')
        axa[0, 1].set_title('sound from right')

        # Title the figure with the wave
        f.suptitle(wave, x=.02, y=.99, ha='left')


        ## Savefig
        f.savefig(os.path.join('figures',
            f"PLOT_PEAK_GROWTH_FUNCTIONS_AFTER_HL_{wave}.svg"))
        f.savefig(os.path.join('figures',
            f"PLOT_PEAK_GROWTH_FUNCTIONS_AFTER_HL_{wave}.png"), dpi=300)


if PLOT_PEAK_W1_W4_RATIO_AFTER_HL:
    """W1p / W4p peak-amplitude ratio after HL, colored by HL_type
    
    Uses positive peaks (height of Wn_p). after_HL==True only, pooled across
    n_experiment. One line per recording; ratio is null at any level where
    either wave is absent for that recording. Colored by HL_type (bilateral
    red, sham gray). 2x2 over channel * speaker_side.
    """

    # Waves forming the ratio (numerator / denominator)
    numer_wave = 'W1p'
    denom_wave = 'W4p'

    # Color by HL_type
    hl_colors = {'bilateral': 'red', 'sham': 'gray'}

    # Panels
    channel_l = ['VL', 'VR']
    speaker_side_l = ['L', 'R']

    # Slice to after_HL, positive peaks, just the two waves
    this_ridges = big_ridges.xs(True, level='after_HL')
    this = this_ridges.xs('pos', level='sign')
    this = this[this['wave_name'].isin([numer_wave, denom_wave])]

    # height indexed by recording * config * level * wave_name
    h = this.set_index('wave_name', append=True)['height'].droplevel('n_ridge')

    # Collapse any duplicate ridges mapped to same recording*config*level*wave
    recording_levels = ['HL_type', 'n_experiment', 'mouse']
    group_keys = recording_levels + [
        'channel', 'speaker_side', 'level', 'wave_name']
    h = h.groupby(group_keys).mean()

    # Wave onto columns so we can divide numer by denom, aligned on all else
    h = h.unstack('wave_name')

    # Ratio; NaN wherever either wave is missing for this recording*config*level
    ratio = (h[numer_wave] / h[denom_wave]).rename('ratio')

    # Recordings onto columns, (config, level) on rows
    to_agg = ratio.unstack(recording_levels)

    # Figure
    f, axa = plt.subplots(
        len(channel_l), len(speaker_side_l),
        sharex=True, sharey=True, figsize=(5, 4))
    f.subplots_adjust(
        left=.17, right=.89, top=.95, bottom=.15, hspace=.15, wspace=.12)

    # Plot each channel * speaker_side
    for channel in channel_l:
        for speaker_side in speaker_side_l:

            # Get ax
            ax = axa[
                channel_l.index(channel),
                speaker_side_l.index(speaker_side),
            ]

            # Slice this config; index=level, cols=recording; sort by level
            try:
                sub = to_agg.xs(channel, level='channel').xs(
                    speaker_side, level='speaker_side').sort_index()
            except KeyError:
                continue

            # One line per recording, colored by HL_type
            for rec in sub.columns:

                # rec is a tuple (HL_type, n_experiment, mouse)
                hl_type = rec[recording_levels.index('HL_type')]
                color = hl_colors[hl_type]

                # Plot this recording's ratio vs level
                col = sub[rec]
                ax.plot(
                    col.index, col.values,
                    color=color, lw=.75, alpha=1)

            # Pretty
            my.plot.despine(ax)

    # Legend
    for n_hl, (hl_type, color) in enumerate(hl_colors.items()):
        f.text(
            .95, .68 - n_hl * .05, hl_type,
            color=color, ha='center', va='center', size=12)

    # Reference line at ratio == 1
    for ax in axa.flat:
        ax.axhline(1, color='k', lw=.5, ls=':')

    # Pretty
    f.text(.52, .01, 'sound level (dB SPL)', ha='center', va='bottom')
    f.text(.02, .56, f'{numer_wave} / {denom_wave} amplitude ratio',
        rotation=90, ha='center', va='center')
    ax.set_yscale('log')
    ax.set_ylim((.1, 10))
    ax.set_xlim((20, 80))
    ax.set_xticks((30, 50, 70))

    # Label the channel
    for n_channel, channel in enumerate(channel_l):
        axa[n_channel, 0].set_ylabel(channel)

    # Label the speaker side
    axa[0, 0].set_title('sound from left')
    axa[0, 1].set_title('sound from right')


    ## Savefig
    f.savefig(os.path.join('figures',
        'PLOT_PEAK_W1_W4_RATIO_AFTER_HL.svg'))
    f.savefig(os.path.join('figures',
        'PLOT_PEAK_W1_W4_RATIO_AFTER_HL.png'), dpi=300)

plt.show()