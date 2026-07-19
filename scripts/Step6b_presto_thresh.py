## Comparison of our thresholds with ABRpresto's
# Skip this script if you didn't run ABRpresto
# Plots
#   PLOT_OUR_VS_PRESTO_THRESHOLDS
#   PLOT_OUR_VS_PRESTO_THRESHOLDS_AFTER_HL
# The threshold calculation is duplicated from Step6a

import os
import json
import numpy as np
import pandas
import my.plot
import my.misc
import matplotlib.pyplot as plt
import seaborn
import shared


## Plotting 
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


## Load metadata
metadata = shared.load_metadata(raw_data_directory)

# Parse out
mouse_metadata = metadata['mouse_metadata'].copy()
recording_metadata = metadata['recording_metadata'].copy()
experiment_metadata = metadata['experiment_metadata'].copy()


## Load previous results
# Load results of Step2b_avg
big_abrs = pandas.read_parquet(
    os.path.join(output_directory, 'big_abrs'))


## Calculate our threshold
# This block is duplicated from Step5a_rms_and_threshold
# Smooth to get rms(ABR) as a function of level
big_abr_stds = big_abrs.T.rolling(window=20, center=True, min_periods=1).std().T

# Use sample 32 (2.0 ms) as the evoked peak
big_abr_evoked_rms = big_abr_stds.loc[:, 32].unstack('label')

# Aggregate over recordings within a date
big_abr_evoked_rms = big_abr_evoked_rms.groupby(
    [lev for lev in big_abr_evoked_rms.index.names if lev != 'recording'],
    ).mean()

# Always take first experiment
big_abr_evoked_rms = big_abr_evoked_rms.xs(0, level='n_experiment')

# Reindex to 0.1 dB resolution
new_sound_levels = pandas.Index(np.arange(
    big_abr_evoked_rms.columns.min(),
    big_abr_evoked_rms.columns.max() + 1e-6,
    0.1,
    ), name=big_abr_evoked_rms.columns.name)

# Interpolate (linearly)
big_abr_evoked_rms = big_abr_evoked_rms.reindex(
    new_sound_levels.union(big_abr_evoked_rms.columns), axis=1).interpolate(
    axis=1, method='index').reindex(new_sound_levels, axis=1)
assert not big_abr_evoked_rms.isnull().any().any()

# Apply a fixed threshold in uV
over_thresh = big_abr_evoked_rms.T > 0.3e-6 
over_thresh = over_thresh.T.stack()

# Threshold is the softest level above the fixed threshold
threshold_db = over_thresh.loc[over_thresh.values].groupby(
    [lev for lev in over_thresh.index.names if lev != 'label'],
    ).apply(
    lambda df: df.index[0][-1])

# Reindex to get those that are never above threshold
threshold_db = threshold_db.reindex(big_abr_evoked_rms.index)

# Error check that we always have a threshold
assert not threshold_db.isnull().any()


## Load abrpresto results
abr_presto_threshold_df = pandas.read_parquet(
    os.path.join(output_directory, 'abr_presto_threshold_df'))

# Include only threshold
abr_presto_threshold_df = abr_presto_threshold_df.loc[:, ['threshold']]

# Join speaker_side from recording_metadata
abr_presto_threshold_df = my.misc.join_level_onto_index(
    abr_presto_threshold_df, 
    to_join=recording_metadata['speaker_side'], 
    )

# Join after_HL from experiment_metadata
abr_presto_threshold_df = my.misc.join_level_onto_index(
    abr_presto_threshold_df, 
    to_join=experiment_metadata[['after_HL', 'n_experiment']], 
    )

# Join HL_type from mouse_metadata
abr_presto_threshold_df = my.misc.join_level_onto_index(
    abr_presto_threshold_df, 
    to_join=mouse_metadata['HL_type'], 
    )

# Reorder levels
abr_presto_threshold_df = abr_presto_threshold_df.reorder_levels([
    'HL_type', 'after_HL', 'mouse', 'date', 'n_experiment', 'recording', 
    'speaker_side', 'channel', 
    ]).sort_index()


## Aggregate ABRpresto results
# First aggregate over recording
abr_presto_threshold_by_date = abr_presto_threshold_df.groupby(
    [lev for lev in abr_presto_threshold_df.index.names if lev != 'recording']
    ).mean()

# Always take first experiment, matching our threshold
abr_presto_threshold_by_mouse = abr_presto_threshold_by_date.xs(
    0, level='n_experiment').droplevel('date')


## Pair our thresholds with ABRpresto's
paired = abr_presto_threshold_by_mouse.rename(
    columns={'threshold': 'abrpresto'}).join(
    threshold_db.rename('ours'))
assert not paired.isnull().any().any()
assert len(abr_presto_threshold_by_mouse) == len(threshold_db)
assert len(abr_presto_threshold_by_mouse) == len(paired)


## Plots
PLOT_OUR_VS_PRESTO_THRESHOLDS = True
PLOT_OUR_VS_PRESTO_THRESHOLDS_AFTER_HL = True
PLOT_OUR_VS_PRESTO_THRESHOLDS_EXAMPLE_CONFIG = True


if PLOT_OUR_VS_PRESTO_THRESHOLDS:
    ## Plot our threshold vs ABRPresto's as connected pairs for all configs
    # For this one, do pre_HL only
    
    # An outlier is NoBadVibes10 VL-L, where we had a small signal
    # that wasn't enough to cross our threshold, so we assigned a high threshold
    # but apparently the response became consistent at the lower (typical) level

    # Slice out pre-HL only
    paired_pre_HL = paired.xs(False, level='after_HL').droplevel('HL_type')


    ## Plot 
    # Channels in rows, speaker side in columns
    channel_l = ['VL', 'VR', 'RL']
    speaker_side_l = ['L', 'R']
    
    # 6 panels: channel (VL/VR/RL) x speaker_side (L/R)
    f, axa = plt.subplots(3, 2, figsize=(3.8, 3.5), sharey=True, sharex=True)
    f.subplots_adjust(
        wspace=.3, hspace=.4, left=.15, right=.95, bottom=.2, top=.92)

    # Groupby channel * speaker_side (subplots)
    grouped = paired_pre_HL.groupby(['channel', 'speaker_side'])
    
    # Iterate over channel * speaker_side
    for (this_channel, this_speaker_side), this_paired in grouped:
        
        # Get ax
        ax = axa[
            channel_l.index(this_channel),
            speaker_side_l.index(this_speaker_side)]

        # One connected pair per row
        ax.plot(
            this_paired.loc[:, ['ours', 'abrpresto']].values.T,
            marker='o', color='gray', alpha=.4, markersize=4)

        # Title by speaker side, ylabel by channel
        if ax in axa[0]:
            ax.set_title(this_speaker_side)
        if ax in axa[:, 0]:
            ax.set_ylabel(this_channel, rotation=0, va='center', labelpad=20)
        
        # Pretty
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['ours', 'ABRpresto'], rotation=45)
        ax.set_yticks((30, 40, 50))
        ax.set_ylim((20, 60))
        ax.set_xlim(-.8, 1.8)
        my.plot.despine(ax)

    # Savefig
    f.savefig('figures/PLOT_OUR_VS_PRESTO_THRESHOLDS.svg')
    f.savefig('figures/PLOT_OUR_VS_PRESTO_THRESHOLDS.png', dpi=300)    

    
    ## Swarm the difference
    this_diff = paired_pre_HL['ours'] - paired_pre_HL['abrpresto']

    f, ax = plt.subplots(figsize=(6, 2.25))
    f.subplots_adjust(left=.15, bottom=.1, right=.95)
    seaborn.swarmplot(
        data=this_diff.rename('diff').reset_index(), 
        x='channel', hue='speaker_side', y='diff', 
        palette={'L': 'b', 'R': 'r'},
        dodge=True,
        legend=False,
        )
    ax.axhline(0, color='k', linestyle='-', linewidth=.75)
    ax.set_ylabel('threshold difference (dB)\n(ours - ABRpresto)')
    my.plot.despine(ax, which=('bottom', 'top', 'right'))  
    
    # Savefig
    f.savefig('figures/PLOT_OUR_VS_PRESTO_THRESHOLDS__swarm.svg')
    f.savefig('figures/PLOT_OUR_VS_PRESTO_THRESHOLDS__swarm.png', dpi=300)    
    

if PLOT_OUR_VS_PRESTO_THRESHOLDS_AFTER_HL:
    ## Plot our threshold vs ABRPresto's as connected pairs for all configs

    # Slice out post-HL only
    paired_post_HL = paired.xs(True, level='after_HL')


    ## Plot 
    # Channels in rows, speaker side in columns
    channel_l = ['VL', 'VR', 'RL']
    speaker_side_l = ['L', 'R']
    
    # 6 panels: channel (VL/VR/RL) x speaker_side (L/R)
    # Color: bilateral red, sham gray
    f, axa = plt.subplots(3, 2, figsize=(3.8, 3.5), sharey=True, sharex=True)
    f.subplots_adjust(
        wspace=.3, hspace=.4, left=.15, right=.95, bottom=.2, top=.92)

    # Groupby HL_type * channel * speaker_side
    grouped = paired_post_HL.groupby(['HL_type', 'channel', 'speaker_side'])
    
    # Iterate over HL_type * channel * speaker_side
    for (HL_type, this_channel, this_speaker_side), this_paired in grouped:
        
        # Get ax
        ax = axa[
            channel_l.index(this_channel),
            speaker_side_l.index(this_speaker_side)]

        # Color by HL_type
        if HL_type == 'bilateral':
            color = 'r'
        else:
            color = 'gray'

        # One connected pair per row
        ax.plot(
            this_paired.loc[:, ['ours', 'abrpresto']].values.T,
            marker='o', color=color, alpha=.4, markersize=4)
        
        # Title by speaker side, ylabel by channel
        if ax in axa[0]:
            ax.set_title(this_speaker_side)
        if ax in axa[:, 0]:
            ax.set_ylabel(this_channel, rotation=0, va='center', labelpad=20)
        
        # Pretty
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['ours', 'ABRpresto'], rotation=45)
        ax.set_yticks((30, 50, 70))
        ax.set_ylim((20, 80))
        ax.set_xlim(-.8, 1.8)
        my.plot.despine(ax)

    # Savefig
    f.savefig('figures/PLOT_OUR_VS_PRESTO_THRESHOLDS_AFTER_HL.svg')
    f.savefig('figures/PLOT_OUR_VS_PRESTO_THRESHOLDS_AFTER_HL.png', dpi=300)    

if PLOT_OUR_VS_PRESTO_THRESHOLDS_EXAMPLE_CONFIG:
    """Our threshold vs ABRpresto's as connected pairs for one example config
    
    VR-L, pre-HL only. This example is nice because it shows that the outlier
    mouse with a high threshold has the same threshold with both methods.
    """
    
    # Slice out pre-HL only
    this_paired = paired.xs(False, level='after_HL').droplevel('HL_type')
    
    # Slice out VR-L only
    this_paired = this_paired.xs(
        'L', level='speaker_side').xs('VR', level='channel')
    
    
    ## Connected pairs
    f, ax = my.plot.figure_1x1_small()
    
    # One connected pair per mouse
    ax.plot(
        this_paired.loc[:, ['ours', 'abrpresto']].values.T,
        marker='o', color='gray', alpha=.4, markersize=4)
    
    # Pretty
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['ours', 'ABRpresto'], rotation=45)
    ax.set_ylabel('threshold (dB SPL)')
    ax.set_yticks((30, 40, 50))
    ax.set_ylim((25, 50))
    ax.set_xlim(-.8, 1.8)
    my.plot.despine(ax)
    
    # Savefig
    f.savefig('figures/PLOT_OUR_VS_PRESTO_THRESHOLDS_EXAMPLE_CONFIG.svg')
    f.savefig(
        'figures/PLOT_OUR_VS_PRESTO_THRESHOLDS_EXAMPLE_CONFIG.png', dpi=300)
    
    
    ## Swarm the difference
    this_diff = this_paired['ours'] - this_paired['abrpresto']
    
    # Figure
    f, ax = my.plot.figure_1x1_small()
    f.subplots_adjust(left=.45, bottom=.1)
    
    # Swarm
    seaborn.swarmplot(this_diff, ax=ax)
    ax.axhline(0, color='k', linestyle='-', linewidth=.75)
    
    # Pretty
    ax.set_ylabel('threshold difference (dB)\n(ours - ABRpresto)')
    ax.set_ylim((-10, 10))
    ax.set_yticks((-10, -5, 0, 5, 10))
    my.plot.despine(ax, which=('bottom', 'top', 'right'))  
    
    # Savefig
    f.savefig(
        'figures/PLOT_OUR_VS_PRESTO_THRESHOLDS_EXAMPLE_CONFIG__swarm.svg')
    f.savefig(
        'figures/PLOT_OUR_VS_PRESTO_THRESHOLDS_EXAMPLE_CONFIG__swarm.png', 
        dpi=300)

plt.show()