## Computes rms(ABR) over level and thresholds
# Plots
#   PLOT_ABR_RMS_OVER_TIME
#   PLOT_ABR_POWER_VS_LEVEL
#   PLOT_ABR_POWER_VS_LEVEL_AFTER_HL
#   BASELINE_VS_N_TRIALS
#   HISTOGRAM_EVOKED_RMS_BY_LEVEL

import os
import json
import datetime
import scipy.stats
import numpy as np
import pandas
import my.plot
import matplotlib.pyplot as plt
import matplotlib
import seaborn


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
experiment_metadata = experiment_metadata.set_index(
    ['date', 'mouse']).sort_index()
mouse_metadata = mouse_metadata.set_index('mouse').sort_index()    


## Load previous results
# Load results of Step2b_avg
big_abrs = pandas.read_pickle(
    os.path.join(output_directory, 'big_abrs'))
trial_counts = pandas.read_pickle(
    os.path.join(output_directory, 'trial_counts'))


## Load abrpresto results
abr_presto_threshold_df = pandas.read_pickle(
    os.path.join(output_directory, 'abr_presto_threshold_df'))

# Include only threshold
abr_presto_threshold_df = abr_presto_threshold_df.loc[:, ['threshold']]

# Join speaker_side from recording_metadata
abr_presto_threshold_df = my.misc.join_level_onto_index(
    abr_presto_threshold_df, 
    to_join=recording_metadata['speaker_side'], 
    join_on=['date', 'mouse', 'recording'],
    )

# Join after_HL from experiment_metadata
abr_presto_threshold_df = my.misc.join_level_onto_index(
    abr_presto_threshold_df, 
    to_join=experiment_metadata[['after_HL', 'n_experiment']], 
    join_on=['date', 'mouse'],
    )

# Join HL_type from mouse_metadata
abr_presto_threshold_df = my.misc.join_level_onto_index(
    abr_presto_threshold_df, 
    to_join=mouse_metadata['HL_type'], 
    join_on=['mouse'],
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

# Now either take the first experiment, or aggregate over experiments
# Analyses of peaks use first experiment
# First-pass analyses of threshold and rms used average
# Let's try doing everything as the first experiment
abr_presto_threshold_by_mouse = abr_presto_threshold_by_date.xs(
    0, level='n_experiment').droplevel('date')


## Calculate the stdev(ABR) as a function of level
# window=20 (1.25 ms) seems the best compromise between smoothing the whole
# response and localizing it to a reasonably narrow window (and not extending
# into the baseline period)
# Would be good to extract more baseline to use here
# The peak is around sample 32 (2.0 ms), ie sample 22 - 42, and there is a
# variable later peak.
big_abr_stds = big_abrs.T.rolling(window=20, center=True, min_periods=1).std().T


## Calculate the baseline
# Use samples -40 to -20 as baseline
# Generally this should be 50-100 nV, depending on the number of trials
big_abr_baseline_rms = big_abr_stds.loc[:, -30].unstack('label')

# Choose a baseline for each recording as the median over levels
# It's lognormal so mean might be skewed. A mean of log could be good
big_abr_baseline_rms = big_abr_baseline_rms.median(axis=1)


## Calculate the evoked 
# Use samples 22 - 42 as evoked peak
# Evoked response increases linearly with level in dB
# Interestingly, each recording appears to be multiplicatively scaled
# (shifted up and down on a log plot). The variability in microvolts increases
# with level, but the variability in log-units is consistent over level.
big_abr_evoked_rms = big_abr_stds.loc[:, 32].unstack('label')

# Aggregate over recordings within a date
big_abr_evoked_rms = big_abr_evoked_rms.groupby(
    [lev for lev in big_abr_evoked_rms.index.names if lev != 'recording'],
    ).mean()

# Aggregate over dates within a mouse * after_HL
# This was the orginal way - average 1st and 2nd experiments
#~ big_abr_evoked_rms = big_abr_evoked_rms.groupby(
    #~ [lev for lev in big_abr_evoked_rms.index.names if lev != 'n_experiment'],
    #~ ).mean()
# Newer way: always take first experiment
big_abr_evoked_rms = big_abr_evoked_rms.xs(0, level='n_experiment')


## Do the same for the late response
big_abr_evoked_rms_late = big_abr_stds.loc[:, 96].unstack('label')

# Aggregate over recordings within a date
big_abr_evoked_rms_late = big_abr_evoked_rms_late.groupby(
    [lev for lev in big_abr_evoked_rms_late.index.names if lev != 'recording'],
    ).mean()

# Aggregate over dates within a mouse * after_HL
#~ big_abr_evoked_rms_late = big_abr_evoked_rms_late.groupby(
    #~ [lev for lev in big_abr_evoked_rms_late.index.names if lev != 'n_experiment'],
    #~ ).mean()
# Newer way: always take first experiment
big_abr_evoked_rms_late = big_abr_evoked_rms_late.xs(0, level='n_experiment')


## Calculate the threshold
# Keep a copy before interpolating
big_abr_evoked_rms_orig = big_abr_evoked_rms.copy()

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

# threshold
# typically a bit better on LR even though LR has slightly higher baseline
threshold_db = over_thresh.loc[over_thresh.values].groupby(
    [lev for lev in over_thresh.index.names if lev != 'label'],
    ).apply(
    lambda df: df.index[0][-1])

# reindex to get those that are never above threshold
threshold_db = threshold_db.reindex(big_abr_evoked_rms.index)

# error check that we always have a threshold
assert not threshold_db.isnull().any()


## Pair our thresholds with ABRpresto's
paired = abr_presto_threshold_by_mouse.rename(
    columns={'threshold': 'abrpresto'}).join(
    threshold_db.rename('ours'))
assert not paired.isnull().any().any()
assert len(abr_presto_threshold_by_mouse) == len(threshold_db)
assert len(abr_presto_threshold_by_mouse) == len(paired)

# Compute diff
abr_presto_diff = paired['ours'] - paired['abrpresto']


## Plots
NORMALIZE_TO_WAVE1 = False
WHICH_MICE = 'pre_HL'

PLOT_OUR_VS_PRESTO_THRESHOLDS = True
PLOT_OUR_VS_PRESTO_THRESHOLDS_AFTER_HL = True
PLOT_OUR_VS_PRESTO_THRESHOLDS_EXAMPLE_CONFIG = True
PLOT_ABR_RMS_OVER_TIME = True
PLOT_GROWTH_FUNCTIONS = True
PLOT_ABR_POWER_VS_AGE = True
PLOT_ABR_POWER_VS_LEVEL = True
PLOT_ABR_POWER_VS_LEVEL_AFTER_HL = True
PLOT_ABR_POWER_VS_LEVEL_EARLY_VS_LATE_AFTER_HL = True
BASELINE_VS_N_TRIALS = True
HISTOGRAM_EVOKED_RMS_BY_LEVEL = True

if PLOT_OUR_VS_PRESTO_THRESHOLDS:
    ## Plot our threshold vs ABRPresto's as connected pairs for all configs
    # For this one, do pre_HL only
    
    # The major outlier is Pineapple_197 RL-R, and I think ABRpresto correctly
    # reveals noise in that recording, though there is also enough signal
    # for our method to work
    # Another outlier is NoBadVibes10 VL-L, where we had a small signal
    # that wasn't enough to cross our threshold, and ABRpresto is probably
    # more correct

    # Slice out pre-HL only
    paired_pre_HL = paired.xs(False, level='after_HL').droplevel('HL_type')


    ## Plot 
    # Channels in rows, speaker side in columns
    channel_l = ['VL', 'VR', 'RL']
    speaker_side_l = ['L', 'R']
    
    # 4 panels: channel (VL/VR) x speaker_side (L/R)
    # Color: bilateral+after_HL red, sham+after_HL blue, before_HL black
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
    f.savefig('figures/PLOT_OUR_VS_PRESTO_THRESHOLDS_SWARM.svg')
    f.savefig('figures/PLOT_OUR_VS_PRESTO_THRESHOLDS_SWARM.png', dpi=300)    
    

if PLOT_OUR_VS_PRESTO_THRESHOLDS_AFTER_HL:
    ## Plot our threshold vs ABRPresto's as connected pairs for all configs

    # Slice out post-HL only
    paired_post_HL = paired.xs(True, level='after_HL')


    ## Plot 
    # Channels in rows, speaker side in columns
    channel_l = ['VL', 'VR', 'RL']
    speaker_side_l = ['L', 'R']
    
    # 4 panels: channel (VL/VR) x speaker_side (L/R)
    # Color: bilateral+after_HL red, sham+after_HL blue, before_HL black
    f, axa = plt.subplots(3, 2, figsize=(3.8, 3.5), sharey=True, sharex=True)
    f.subplots_adjust(
        wspace=.3, hspace=.4, left=.15, right=.95, bottom=.2, top=.92)

    # Groupby channel * speaker_side (subplots)
    grouped = paired_post_HL.groupby(['HL_type', 'channel', 'speaker_side'])
    
    # Iterate over channel * speaker_side
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
    ## Plot our threshold vs ABRPresto's as connected pairs for example config

    # Slice out pre-HL only
    this_paired = paired.xs(False, level='after_HL').droplevel('HL_type')
    
    # Slice out VR-L only
    # This example is nice because it shows that the outlier mouse has the
    # same thresh with both methods
    this_paired = this_paired.xs(
        'L', level='speaker_side').xs('VR', level='channel')


    ## Plot 
    # One panel
    f, ax = my.plot.figure_1x1_small()

    # One connected pair per mouse
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
    ax.set_ylim((25, 50))
    ax.set_xlim(-.8, 1.8)
    my.plot.despine(ax)

    # Savefig
    f.savefig('figures/PLOT_OUR_VS_PRESTO_THRESHOLDS_EXAMPLE_CONFIG.svg')
    f.savefig('figures/PLOT_OUR_VS_PRESTO_THRESHOLDS_EXAMPLE_CONFIG.png', dpi=300)    
    
    
    ## Hist the difference
    this_diff = this_paired['ours'] - this_paired['abrpresto']

    #~ f, ax = my.plot.figure_1x1_small()
    #~ ax.hist(this_diff.values, bins=np.linspace(-20, 20, 31), histtype='stepfilled', color='gray')
    #~ ax.axvline(0, color='k', linestyle='-', linewidth=.75)
    #~ ax.set_xlabel('threshold difference (dB)\n(ours - ABRpresto)')
    #~ ax.set_ylim(bottom=0)
    #~ ax.set_yticks([])
    #~ ax.set_xlim((-10, 10))
    #~ ax.set_xticks((-10, 0, 10))
    #~ my.plot.despine(ax, which=('left', 'top', 'right'))       

    f, ax = my.plot.figure_1x1_small()
    f.subplots_adjust(left=.45, bottom=.1)
    seaborn.swarmplot(this_diff, ax=ax)
    ax.set_ylim((-10, 10))
    ax.axhline(0, color='k', linestyle='-', linewidth=.75)
    ax.set_ylabel('threshold difference (dB)\n(ours - ABRpresto)')
    my.plot.despine(ax, which=('bottom', 'top', 'right'))  
    
    # Savefig
    f.savefig('figures/PLOT_OUR_VS_PRESTO_THRESHOLDS_EXAMPLE_CONFIG_SWARM.svg')
    f.savefig('figures/PLOT_OUR_VS_PRESTO_THRESHOLDS_EXAMPLE_CONFIG_SWARM.png', dpi=300)    


if PLOT_ABR_RMS_OVER_TIME:
    ## Plot the smoothed rms of the ABR over time by condition
    # Shared t
    t = big_abrs.columns / sampling_rate * 1000

    ## Select mice
    if WHICH_MICE == 'pre_HL':
        # Slice out pre-HL only
        this_big_abr_stds = big_abr_stds.xs(
            False, level='after_HL').droplevel('HL_type')

    elif WHICH_MICE == 'bilateral':
        # Slice out post-HL only
        this_big_abr_stds = big_abr_stds.xs(
            True, level='after_HL').xs(
            'bilateral', level='HL_type')

    elif WHICH_MICE == 'sham':
        # Slice out post-HL only
        this_big_abr_stds = big_abr_stds.xs(
            True, level='after_HL').xs(
            'sham', level='HL_type')
    
    else:
        1/0
    
    # Aggregate over recordings
    to_agg = this_big_abr_stds.groupby(
        [lev for lev in this_big_abr_stds.index.names if lev != 'recording']
        ).mean()

    # Aggregate over date
    # TODO: keep just the first session from each mouse instead?
    to_agg = to_agg.groupby(
        [lev for lev in to_agg.index.names if lev != 'n_experiment']
        ).mean()

    # Make mouse the replicates on the columns
    to_agg = to_agg.stack().unstack('mouse')

    # Normalize to wave 1
    if NORMALIZE_TO_WAVE1:
        to_agg = to_agg.divide(to_agg.xs(32, level='timepoint'))

    # Agg
    # TODO: log10 before agg?
    agg_mean = to_agg.mean(axis=1).unstack('timepoint')
    agg_err = to_agg.sem(axis=1).unstack('timepoint')
    
    # Set up colorbar
    # Always do the lowest labels last
    label_l = sorted(
        agg_mean.index.get_level_values('label').unique(), 
        reverse=True)
    aut_colorbar = my.plot.generate_colorbar(
        len(label_l), mapname='inferno_r', start=0.15, stop=1)  

    # Set up ax_rows and ax_cols
    channel_l = ['VL', 'VR', 'RL']
    speaker_side_l = ['L', 'R']
    
    # Make plot
    f, axa = plt.subplots(
        len(channel_l), len(speaker_side_l),
        sharex=True, sharey=True, figsize=(5, 4))
    f.subplots_adjust(
        left=.17, right=.89, top=.95, bottom=.15, hspace=.15, wspace=.12)

    # Plot each channel * speaker_side
    gobj = agg_mean.groupby(['channel', 'speaker_side'])
    for (channel, speaker_side), subdf in gobj:
        
        # droplevel
        subdf = subdf.droplevel(
            ['channel', 'speaker_side']).sort_index(ascending=False)
    
        # get the error bars
        subdf_err = agg_err.loc[channel].loc[speaker_side]

        # Get ax
        ax = axa[
            channel_l.index(channel),
            speaker_side_l.index(speaker_side),
        ]

        # Plot the evoked peak time
        ax.plot([2, 2], [.03, 3], color='gray', lw=.75)
    
        # Plot each
        for n_level, level in enumerate(subdf.index):
            # Get color
            color = aut_colorbar[label_l.index(level)]

            if NORMALIZE_TO_WAVE1:
                # Plot the mean (as a ratio to wave 1)
                ax.plot(t, subdf.loc[level], color=color, lw=1)  

                # Error bars
                ax.fill_between(t, 
                    (subdf.loc[level] - subdf_err.loc[level]),
                    (subdf.loc[level] + subdf_err.loc[level]),
                    color=color, alpha=.5, lw=0)

            else:
                # Plot the mean (in uV)
                ax.plot(t, subdf.loc[level] * 1e6, color=color, lw=1)  

                # Error bars
                ax.fill_between(t, 
                    (subdf.loc[level] - subdf_err.loc[level]) * 1e6,
                    (subdf.loc[level] + subdf_err.loc[level]) * 1e6,
                    color=color, alpha=.5, lw=0)

        # Pretty
        my.plot.despine(ax) 
        ax.set_yscale('log')
        ax.set_yticks((0.1, 1.0))

        # Nicer log labels
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # Legend
    for n_label, (label, color) in enumerate(zip(label_l, aut_colorbar)):
        if np.mod(n_label, 2) != 0:
            continue
        f.text(
            .95, .68 - n_label * .02, f'{label} dB',
            color=color, ha='center', va='center', size=12)

    # Pretty
    ax.set_xlim((-1, 7))
    ax.set_ylim((.05, 2))
    ax.set_xticks([0, 2, 4, 6])
    f.text(.52, .01, 'time from sound onset (ms)', ha='center', va='bottom')
    f.text(.02, .56, f'response strength ({MU}V rms)', rotation=90, ha='center', va='center')

    # Label the channel
    for n_channel, channel in enumerate(channel_l):
        axa[n_channel, 0].set_ylabel(channel)
    
    # Label the speaker side
    axa[0, 0].set_title('sound from left')
    axa[0, 1].set_title('sound from right')

    
    ## Savefig
    f.savefig(os.path.join('figures', 
        f"PLOT_ABR_RMS_OVER_TIME__{'norm' if NORMALIZE_TO_WAVE1 else 'raw'}__{WHICH_MICE}.svg"))
    f.savefig(os.path.join('figures', 
        f"PLOT_ABR_RMS_OVER_TIME__{'norm' if NORMALIZE_TO_WAVE1 else 'raw'}__{WHICH_MICE}.png"), dpi=300)

    
    ## Stats
    stats_filename = 'figures/STATS__PLOT_ABR_RMS_OVER_TIME'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write(f'n = {to_agg.shape[1]} mice\n')
        fi.write(
            'compute rolling RMS for each recording * level, '
            'then mean over recordings within date, '
            'then mean over date within mouse, '
            'then mean and SEM over mice, '
            'then plot on log scale\n')
    
    # Echo
    with open(stats_filename) as fi:
        print(''.join(fi.readlines()))


if PLOT_GROWTH_FUNCTIONS:
    ## Plot the smoothed rms of the ABR over time by condition
    # Shared t
    t = big_abrs.columns / sampling_rate * 1000


    ## Select mice
    if WHICH_MICE == 'pre_HL':
        # Slice out pre-HL only
        this_big_abr_stds = big_abr_stds.xs(
            False, level='after_HL').droplevel('HL_type')

    elif WHICH_MICE == 'bilateral':
        # Slice out post-HL only
        this_big_abr_stds = big_abr_stds.xs(
            True, level='after_HL').xs(
            'bilateral', level='HL_type')

    elif WHICH_MICE == 'sham':
        # Slice out post-HL only
        this_big_abr_stds = big_abr_stds.xs(
            True, level='after_HL').xs(
            'sham', level='HL_type')
    
    else:
        1/0
        
    # Aggregate over recordings
    to_agg = this_big_abr_stds.groupby(
        [lev for lev in this_big_abr_stds.index.names if lev != 'recording']
        ).mean()

    # Aggregate over date
    # TODO: keep just the first session from each mouse instead?
    to_agg = to_agg.groupby(
        [lev for lev in to_agg.index.names if lev != 'n_experiment']
        ).mean()

    # Make mouse the replicates on the columns
    to_agg = to_agg.stack().unstack('mouse')

    # Normalize to wave 1
    if NORMALIZE_TO_WAVE1:
        to_agg = to_agg.divide(to_agg.xs(32, level='timepoint'))

    # Agg
    # TODO: log10 before agg?
    agg_mean = to_agg.mean(axis=1).unstack('timepoint')
    agg_err = to_agg.sem(axis=1).unstack('timepoint')

    # Set up colorbar by time, rather than level
    timepoint_l = sorted(agg_mean.columns)
    aut_colorbar = my.plot.generate_colorbar(
        len(timepoint_l), mapname='viridis_r', start=0, stop=1)  

    # Set up ax_rows and ax_cols
    channel_l = ['VL', 'VR', 'RL']
    speaker_side_l = ['L', 'R']
    
    # Make plot
    f, axa = plt.subplots(
        len(channel_l), len(speaker_side_l),
        sharex=True, sharey=True, figsize=(5, 4))
    f.subplots_adjust(
        left=.17, right=.89, top=.95, bottom=.15, hspace=.15, wspace=.12)

    # Plot each channel * speaker_side
    gobj = agg_mean.groupby(['channel', 'speaker_side'])
    for (channel, speaker_side), subdf in gobj:
        
        # droplevel
        subdf = subdf.droplevel(
            ['channel', 'speaker_side'])
    
        # get the error bars
        subdf_err = agg_err.loc[channel].loc[speaker_side]

        # Get ax
        ax = axa[
            channel_l.index(channel),
            speaker_side_l.index(speaker_side),
        ]

        # Plot each
        for n_timepoint, timepoint in enumerate(subdf.columns[24:-8:16]):
            # Get color
            color = aut_colorbar[timepoint_l.index(timepoint)]


            if NORMALIZE_TO_WAVE1:
                # Plot the mean (as a ratio to wave 1)
                ax.plot(
                    subdf.index, 
                    subdf.loc[:, timepoint],
                    color=color, 
                    lw=1)  
                
                # Error bars
                ax.fill_between(
                    subdf.index,
                    (subdf.loc[:, timepoint] - subdf_err.loc[:, timepoint]),
                    (subdf.loc[:, timepoint] + subdf_err.loc[:, timepoint]),
                    color=color, alpha=.5, lw=0)

            else:
                # Plot the mean (in uV)
                ax.plot(
                    subdf.index, 
                    subdf.loc[:, timepoint] * 1e6, 
                    color=color, 
                    lw=1)  
                
                # Error bars
                ax.fill_between(
                    subdf.index,
                    (subdf.loc[:, timepoint] - subdf_err.loc[:, timepoint]) * 1e6,
                    (subdf.loc[:, timepoint] + subdf_err.loc[:, timepoint]) * 1e6,
                    color=color, alpha=.5, lw=0)

        # Pretty
        my.plot.despine(ax) 
        ax.set_yscale('log')
        ax.set_yticks((0.1, 1.0))

        # Nicer log labels
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # Legend
    for n_label, (label, color) in enumerate(zip(timepoint_l[24:-8:16], aut_colorbar[24:-8:16])):
        # Check we are correctly plotting once per ms
        assert label // 16 == int(np.rint(label / 16))
        
        # Label
        f.text(
            .95, .68 - n_label * .04, f't = {label // 16} ms',
            color=color, ha='center', va='center', size=12)

    # Pretty
    ax.set_xlim((20, 80))
    ax.set_ylim((.05, 2))
    ax.set_xticks([30, 50, 70])
    f.text(.52, .01, 'sound level (dB SPL)', ha='center', va='bottom')
    f.text(.02, .56, f'response strength ({MU}V rms)', rotation=90, ha='center', va='center')

    # Label the channel
    for n_channel, channel in enumerate(channel_l):
        axa[n_channel, 0].set_ylabel(channel)
    
    # Label the speaker side
    axa[0, 0].set_title('sound from left')
    axa[0, 1].set_title('sound from right')


    ## Savefig
    f.savefig(os.path.join('figures', 
        f"PLOT_GROWTH_FUNCTIONS__{'norm' if NORMALIZE_TO_WAVE1 else 'raw'}__{WHICH_MICE}.svg"))
    f.savefig(os.path.join('figures', 
        f"PLOT_GROWTH_FUNCTIONS__{'norm' if NORMALIZE_TO_WAVE1 else 'raw'}__{WHICH_MICE}.png"), dpi=300)


if PLOT_ABR_POWER_VS_AGE:
    ## Correlate response magnitude with age of mouse
    # Slice out pre-HL only
    this_threshold_db = threshold_db.xs(
        False, level='after_HL').droplevel('HL_type')
    this_big_abr_evoked_rms = big_abr_evoked_rms.xs(
        False, level='after_HL').droplevel('HL_type')
    
    # Compute average mouse age at time of recordings
    mouse_age = experiment_metadata[
        experiment_metadata['after_HL'] == False
        ].groupby('mouse')['age'].mean()
    
    # Average response by mouse across speaker_side * channel
    # Note: This is significant for all channels and speaker_side separately,
    # though perhaps weaker for LR
    mouse_response = this_big_abr_evoked_rms.iloc[:, -1].groupby('mouse').mean()
    
    # Average thresh by mouse across speaker_side * channel
    mouse_thresh = this_threshold_db.groupby('mouse').mean()
    
    # Concat
    age_data = pandas.DataFrame({
        'age': mouse_age,
        'response': mouse_response,
        'threshold': mouse_thresh,
        })
    
    # Join sex
    age_data = age_data.join(mouse_metadata['sex'])
    
    # Join experimenter
    mouse_experimenter = experiment_metadata.groupby(['mouse'])[
        'experimenter'].apply(lambda df:df.unique().item())
    age_data = age_data.join(mouse_experimenter)
    
    
    ## Plot response magnitude
    f, ax = my.plot.figure_1x1_small()
    
    # Plot males and females separately
    ax.plot(
        age_data.loc[age_data['sex'] == 'M', 'age'], 
        age_data.loc[age_data['sex'] == 'M', 'response'] * 1e6, 
        'bo', mfc='none')
    ax.plot(
        age_data.loc[age_data['sex'] == 'F', 'age'], 
        age_data.loc[age_data['sex'] == 'F', 'response'] * 1e6, 
        'ro', mfc='none')
    
    # Pretty
    ax.set_xticks((90, 180, 270))
    ax.set_xlabel('age (days)')
    ax.set_yticks((1, 1.5, 2))
    ax.set_ylabel(f'response strength\n({MU}V rms)')    
    my.plot.despine(ax)
    
    # Savefig
    f.savefig('figures/PLOT_ABR_POWER_VS_AGE__response.svg')
    f.savefig('figures/PLOT_ABR_POWER_VS_AGE__response.png', dpi=300)
    
    
    ## Plot thresh
    f, ax = my.plot.figure_1x1_small()
    
    # Plot males and females separately
    ax.plot(
        age_data.loc[age_data['sex'] == 'M', 'age'], 
        age_data.loc[age_data['sex'] == 'M', 'threshold'], 
        'bo', mfc='none')
    ax.plot(
        age_data.loc[age_data['sex'] == 'F', 'age'], 
        age_data.loc[age_data['sex'] == 'F', 'threshold'], 
        'ro', mfc='none')    

    # Pretty
    ax.set_xticks((90, 180, 270))
    ax.set_xlabel('age (days)')
    ax.set_yticks((30, 35, 40))
    ax.set_ylabel('threshold (dB SPL)')
    my.plot.despine(ax)
    
    # Savefig
    f.savefig('figures/PLOT_ABR_POWER_VS_AGE__thresh.svg')
    f.savefig('figures/PLOT_ABR_POWER_VS_AGE__thresh.png', dpi=300)
    
    
    ## Stats
    # Correlate both
    vs_response = scipy.stats.linregress(age_data['age'], age_data['response'])
    vs_thresh = scipy.stats.linregress(age_data['age'], age_data['threshold'])
    
    # Write out stats
    n_mice = len(age_data)
    stats_filename = 'figures/STATS__PLOT_ABR_POWER_VS_AGE'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write(f'n = {n_mice} mice, pre-HL only\n')
        fi.write(
            'including pre-HL sessions only, '
            'meaned thresh and rms response at highest level over '
            'channel * speaker_side (already meaned within mouse)\n'
            'then correlated with age (meaned over recordings)\n'
            )
        fi.write(
            f'response vs age: p={vs_response.pvalue:.4f}; r={vs_response.rvalue:.2f}; r2={vs_response.rvalue ** 2:.2f}\n'
            f'thresh vs age: p={vs_thresh.pvalue:.4f}; r={vs_thresh.rvalue:.2f}; r2={vs_thresh.rvalue ** 2:.2f}\n'
            )
    
    # Echo
    with open(stats_filename) as fi:
        print(''.join(fi.readlines()))    

if PLOT_ABR_POWER_VS_LEVEL:
    ## Plot ABR rms power vs sound level for all mice together
    # Both the mean and standard deviation of evoked power strongly increase
    # with sound level, suggesting we should take log of power. 
    # On a semilog plot, the variance is similar across level, and evoked
    # power shows diminishing increases with level, not quite plateauing. 
    
    # Slice out pre-HL only
    this_threshold_db = threshold_db.xs(
        False, level='after_HL').droplevel('HL_type')
    this_big_abr_evoked_rms = big_abr_evoked_rms.xs(
        False, level='after_HL').droplevel('HL_type')

    # Set up ax_rows and ax_cols
    channel_l = ['VL', 'VR', 'RL']
    speaker_side_l = ['L', 'R']
    
    # Make plot
    f, axa = plt.subplots(
        len(channel_l), len(speaker_side_l),
        sharex=True, sharey=True, figsize=(5, 4))
    f.subplots_adjust(
        left=.17, right=.89, top=.95, bottom=.15, hspace=.15, wspace=.12)

    # Iterate over channel * speaker_side
    gobj = this_big_abr_evoked_rms.groupby(['channel', 'speaker_side'])
    for (channel, speaker_side), subdf in gobj:
        
        # Drop grouping keys
        subdf = subdf.droplevel(['channel', 'speaker_side'])

        # Get ax
        ax = axa[
            channel_l.index(channel),
            speaker_side_l.index(speaker_side),
        ]

        # Plot all mice for this config
        for mouse in subdf.index:
            # Get power vs level for this mouse
            topl = subdf.loc[mouse].copy()

            # Get avg threshold for this mouse
            thresh = this_threshold_db.loc[
                mouse].loc[channel].loc[speaker_side].item()

            # Because thresh is a mean over recordings, it doesn't
            # align with evoked RMS. Resample to make the point lie
            # on the line
            thresh_y = np.interp([thresh], topl.index, topl.values)

            # Plot evoked RMS
            ax.semilogy(topl * 1e6, color='gray', lw=.75, alpha=.5)
            
            # Plot thresh
            ax.semilogy(
                [thresh], [thresh_y * 1e6], 
                marker='o', color='red', mfc='none', ms=2.5, alpha=.5)


        # Despine
        my.plot.despine(ax)
        ax.set_yticks([0.1, 1])
        ax.set_xticks([30, 50, 70])
        ax.set_xlim((20, 80))
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())


    ## Pretty
    # Shared axis labes
    f.text(.52, .01, 'sound level (dB SPL)', ha='center', va='bottom')
    f.text(
        .02, .56, f'response strength ({MU}V rms)', 
        rotation=90, ha='center', va='center')

    # Label the channel
    for n_channel, channel in enumerate(channel_l):
        axa[n_channel, 0].set_ylabel(channel)
    
    # Label the speaker side
    axa[0, 0].set_title('sound from left')
    axa[0, 1].set_title('sound from right')

    
    ## Savefig
    f.savefig('figures/PLOT_ABR_POWER_VS_LEVEL.svg')
    f.savefig('figures/PLOT_ABR_POWER_VS_LEVEL.png', dpi=300)


    ## Stats
    n_mice = len(this_threshold_db.index.get_level_values('mouse').unique())
    thresh_by_mouse = this_threshold_db.groupby('mouse').mean()
    thresh_by_mouse_mu = thresh_by_mouse.mean()
    thresh_by_mouse_std = thresh_by_mouse.std()
    thresh_by_mouse_sem = thresh_by_mouse.sem()
    thresh_by_channel = this_threshold_db.groupby('channel').mean()
    ipsi_thresh = this_threshold_db.unstack('mouse').loc[
        [('VL', 'L'), ('VR', 'R')]].mean().mean()
    contra_thresh = this_threshold_db.unstack('mouse').loc[
        [('VR', 'L'), ('VL', 'R')]].mean().mean()
    stats_filename = 'figures/STATS__PLOT_ABR_POWER_VS_LEVEL'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write(f'n = {n_mice} mice, pre-HL only\n')
        fi.write(
            'mean power vs level over recordings within date, '
            'then mean over date within mouse.\n'
            'computed one threshold (interpolated) per mouse\n'
            )
        fi.write(
            f'thresh over mice: mean {thresh_by_mouse_mu:.1f}, '
            f'std {thresh_by_mouse_std:.1f}, sem {thresh_by_mouse_sem:.2f}\n')
        fi.write(f'thresh over channel: \n{thresh_by_channel} \n')
        fi.write(f'thresh ipsi: {ipsi_thresh:.1f}; thresh contra: {contra_thresh:.1f}\n')
    
    # Echo
    with open(stats_filename) as fi:
        print(''.join(fi.readlines()))


if PLOT_ABR_POWER_VS_LEVEL_AFTER_HL:
    ## Plot sham and bilateral pre- and post-HL
    
    # Include only HL mice
    this_threshold_db = threshold_db.drop('none', level='HL_type')
    this_big_abr_evoked_rms = big_abr_evoked_rms.drop('none', level='HL_type')

    # Aggregate threshold over date and recording, maintaining after_HL
    threshold_db_agg = this_threshold_db.groupby(
        [lev for lev in this_threshold_db.index.names if lev != 'recording']
        ).mean()
    threshold_db_agg = threshold_db_agg.groupby(
        [lev for lev in threshold_db_agg.index.names if lev != 'n_experiment']
        ).mean()

    # Aggregate evoked RMS over date and recording, maintaining after_HL
    big_abr_evoked_rms_agg = this_big_abr_evoked_rms.groupby(
        [lev for lev in this_big_abr_evoked_rms.index.names if lev != 'recording']
        ).mean()
    big_abr_evoked_rms_agg = big_abr_evoked_rms_agg.groupby(
        [lev for lev in big_abr_evoked_rms_agg.index.names if lev != 'n_experiment']
        ).mean()
        
    
    ## To iterate over
    channel_l = ['VL', 'VR', 'RL']
    speaker_side_l = ['L', 'R']
    HL_type_l = ['bilateral', 'sham']
    after_HL_l = [False, True]
    
    
    ## Plot
    # Iterate over speaker_side (figures)
    for speaker_side in speaker_side_l:
        
        ## First figure: evoked power vs sound level
        f, axa = plt.subplots(
            len(channel_l), len(HL_type_l),
            sharex=True, sharey=True, figsize=(5, 4))
        f.subplots_adjust(
            left=.17, right=.89, top=.95, bottom=.15, hspace=.15, wspace=.12)

        # Iterate over channel * HL_type
        gobj = big_abr_evoked_rms_agg.xs(
            speaker_side, level='speaker_side').groupby(
            ['channel', 'HL_type'])
        for (channel, HL_type), subdf in gobj:
            
            # Get ax
            ax = axa[
                channel_l.index(channel),
                HL_type_l.index(HL_type),
            ]
            
            # Drop grouping keys
            subdf = subdf.droplevel(['channel', 'HL_type'])
    
            # Iterate over mice
            for mouse in subdf.index.get_level_values('mouse').unique():
                
                # Iterate over after_HL
                for after_HL in after_HL_l:

                    # Color by after_HL
                    color = 'magenta' if after_HL else 'green'
                    
                    # Slice evoked RMS and threshold
                    topl = subdf.loc[after_HL].loc[mouse]
                    thresh = threshold_db_agg.loc[
                        HL_type].loc[after_HL].loc[mouse].loc[channel].loc[
                        speaker_side].item()                

                    # Because thresh is a mean over recordings, it doesn't
                    # align with evoked RMS. Resample to make the point lie
                    # on the line
                    thresh_y = np.interp([thresh], topl.index, topl.values)

                    # Plot evoked RMS
                    ax.semilogy(topl * 1e6, color=color, lw=.75)
                    
                    # Plot thresh
                    ax.semilogy(
                        [thresh], [thresh_y * 1e6], 
                        marker='o', color=color, mfc='none')

            # Label the y-axis with the channel
            if ax in axa[:, 0]:
                ax.set_ylabel(channel)
            if ax == axa[1, 0]:
                ax.set_ylabel(f'responses strength ({MU}V)\n{channel}')
            
            # Despine
            my.plot.despine(ax)
            ax.set_yticks([0.1, 1])
            ax.set_xticks([30, 50, 70])
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

        # Shared x-axis
        f.text(.52, .01, 'sound level (dB SPL)', ha='center', va='bottom')

        # Label the HL_type
        axa[0, 0].set_title(HL_type_l[0])
        axa[0, 1].set_title(HL_type_l[1])
        
        # Legend
        f.text(.95, .6, 'pre', color='green', ha='center', va='center')
        f.text(.95, .54, 'post', color='magenta', ha='center', va='center')

        # Savefig
        savename = f'figures/PLOT_ABR_POWER_VS_LEVEL_AFTER_HL__{speaker_side}'
        f.savefig(savename + '.svg')
        f.savefig(savename + '.png', dpi=300)        
        
        
        ## Second plot of threshold
        f, axa = plt.subplots(
            len(channel_l), len(HL_type_l),
            sharex=True, sharey=True, figsize=(5, 4))
        f.subplots_adjust(
            left=.17, right=.89, top=.95, bottom=.15, hspace=.15, wspace=.12)

        # Iterate over channel * HL_type
        gobj = threshold_db_agg.xs(
            speaker_side, level='speaker_side').groupby(
            ['channel', 'HL_type'])
        for (channel, HL_type), subdf in gobj:
            
            # Drop grouping keys
            subdf = subdf.droplevel(['channel', 'HL_type'])
            
            # Get mouse on columns
            subdf = subdf.unstack('mouse')
            
            # Make sure it's pre before post
            subdf = subdf.reindex([False, True])
            
            # Get ax
            ax = axa[
                channel_l.index(channel),
                HL_type_l.index(HL_type),
            ]        
            
            # Plot
            ax.plot(subdf.values, ls='-', marker='o', color='k', mfc='none')

            # Label the y-axis with the channel
            if ax in axa[:, 0]:
                ax.set_ylabel(channel)
            if ax == axa[1, 0]:
                ax.set_ylabel(f'threshold (dB SPL)\n{channel}')
            
            # Despine
            my.plot.despine(ax)

        # Shared x-axis
        f.text(.52, .01, 'before or after hearing loss', ha='center', va='bottom')

        # Label the HL_type
        axa[0, 0].set_title(HL_type_l[0])
        axa[0, 1].set_title(HL_type_l[1])
        
        # Consistent axis limits
        ax.set_ylim((20, 70))
        ax.set_yticks((30, 45, 60))
        ax.set_xticks((0, 1))
        ax.set_xlim((-.5, 1.5))
        ax.set_xticklabels(('pre', 'post'))
        
        # Savefig
        savename = f'figures/PLOT_ABR_POWER_VS_LEVEL_AFTER_HL__thresh__{speaker_side}'
        f.savefig(savename + '.svg')
        f.savefig(savename + '.png', dpi=300)


        ## Stats
        # Threshold (meaning over channel * speaker_side)
        thresh_by_mouse = threshold_db_agg.groupby(
            ['HL_type', 'after_HL', 'mouse']).mean().unstack('after_HL')
        thresh_by_mouse_mu = thresh_by_mouse.groupby('HL_type').mean()
        thresh_by_mouse_err = thresh_by_mouse.groupby('HL_type').sem()
        thresh_by_mouse_N = thresh_by_mouse.groupby('HL_type').size()
        thresh_by_mouse_diff = thresh_by_mouse.diff(axis=1).iloc[:, 1]
        thresh_by_mouse_diff_mu = thresh_by_mouse_diff.groupby('HL_type').mean()
        thresh_by_mouse_diff_std = thresh_by_mouse_diff.groupby('HL_type').std()
        thresh_by_mouse_diff_err = thresh_by_mouse_diff.groupby('HL_type').sem()
        thresh_by_mouse_diff_N = thresh_by_mouse_diff.groupby('HL_type').size()
        
        # Write out
        # This is redundant over speaker_side
        stats_filename = f'figures/PLOT_ABR_POWER_VS_LEVEL_AFTER_HL__thresh__{speaker_side}'
        with open(stats_filename, 'w') as fi:
            fi.write(stats_filename + '\n')
            fi.write(f'these calculations are all meaned over channel * speaker_side\n')
            fi.write(f'raw changes after_HL by mouse:\n{thresh_by_mouse.diff(axis=1).iloc[:, 1]}\n')
            fi.write(f'mean threshold:\n{thresh_by_mouse_mu}\n')
            fi.write(f'SEM threshold:\n{thresh_by_mouse_err}\n')
            fi.write(f'N threshold:\n{thresh_by_mouse_N}\n')
            fi.write(f'mean diff threshold:\n{thresh_by_mouse_diff_mu}\n')
            fi.write(f'STD diff threshold:\n{thresh_by_mouse_diff_std}\n')
            fi.write(f'SEM diff threshold:\n{thresh_by_mouse_diff_err}\n')
            fi.write(f'N diff threshold:\n{thresh_by_mouse_diff_N}\n')
        
        # Echo
        with open(stats_filename) as fi:
            print(''.join(fi.readlines()))        


if PLOT_ABR_POWER_VS_LEVEL_EARLY_VS_LATE_AFTER_HL:
    ## Plot early and late waves post-bilateral HL
    
    # Include only HL mice
    this_big_abr_evoked_rms_early = big_abr_evoked_rms_orig.drop(
        'none', level='HL_type')
    this_big_abr_evoked_rms_late = big_abr_evoked_rms_late.drop(
        'none', level='HL_type')

    # Aggregate evoked RMS over date and recording, maintaining after_HL
    # Early
    big_abr_evoked_rms_agg_early = this_big_abr_evoked_rms_early.groupby(
        [lev for lev in this_big_abr_evoked_rms_early.index.names if lev != 'recording']
        ).mean()
    big_abr_evoked_rms_agg_early = big_abr_evoked_rms_agg_early.groupby(
        [lev for lev in big_abr_evoked_rms_agg_early.index.names if lev != 'n_experiment']
        ).mean()

    # Late
    big_abr_evoked_rms_agg_late = this_big_abr_evoked_rms_late.groupby(
        [lev for lev in this_big_abr_evoked_rms_late.index.names if lev != 'recording']
        ).mean()
    big_abr_evoked_rms_agg_late = big_abr_evoked_rms_agg_late.groupby(
        [lev for lev in big_abr_evoked_rms_agg_late.index.names if lev != 'n_experiment']
        ).mean()
    

    ## To iterate over
    channel_l = ['VL', 'VR', 'RL']
    speaker_side_l = ['L', 'R']
    HL_type_l = ['bilateral', 'sham']
    after_HL_l = [False, True]
    
    
    ## Plot
    # Iterate over speaker_side (figures)
    for speaker_side in speaker_side_l:
        
        ## First figure: evoked power vs sound level
        f, axa = plt.subplots(
            len(channel_l), len(HL_type_l),
            sharex=True, sharey=True, figsize=(5, 4))
        f.subplots_adjust(
            left=.17, right=.89, top=.95, bottom=.15, hspace=.15, wspace=.12)

        # Iterate over channel * HL_type
        for channel in channel_l:
            for HL_type in HL_type_l:
            
                # Get ax
                ax = axa[
                    channel_l.index(channel),
                    HL_type_l.index(HL_type),
                ]
                
                # Slice
                subdf_early = big_abr_evoked_rms_agg_early.xs(
                    channel, level='channel').xs(
                    HL_type, level='HL_type').xs(
                    True, level='after_HL').xs(
                    speaker_side, level='speaker_side')
                subdf_late = big_abr_evoked_rms_agg_late.xs(
                    channel, level='channel').xs(
                    HL_type, level='HL_type').xs(
                    True, level='after_HL').xs(
                    speaker_side, level='speaker_side')
                
                # Iterate over mice
                for mouse in subdf_early.index.get_level_values('mouse').unique():
                    
                    # Slice evoked RMS
                    topl_early = subdf_early.loc[mouse]
                    topl_late = subdf_late.loc[mouse]

                    # Plot evoked RMS
                    ax.semilogy(topl_early * 1e6, color='k', lw=.75)
                    ax.semilogy(topl_late * 1e6, color='r', lw=.75)

                # Label the y-axis with the channel
                if ax in axa[:, 0]:
                    ax.set_ylabel(channel)
                if ax == axa[1, 0]:
                    ax.set_ylabel(f'responses strength ({MU}V)\n{channel}')
                
                # Despine
                my.plot.despine(ax)
                ax.set_yticks([0.1, 1])
                ax.set_xticks([30, 50, 70])
                ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

        # Shared x-axis
        f.text(.52, .01, 'sound level (dB SPL)', ha='center', va='bottom')

        # Label the HL_type
        axa[0, 0].set_title(HL_type_l[0])
        axa[0, 1].set_title(HL_type_l[1])
        
        # Legend
        f.text(.95, .6, 'pre', color='green', ha='center', va='center')
        f.text(.95, .54, 'post', color='magenta', ha='center', va='center')

        #~ # Savefig
        #~ savename = f'figures/PLOT_ABR_POWER_VS_LEVEL_EARLY_VS_LATE_AFTER_HL__{speaker_side}'
        #~ f.savefig(savename + '.svg')
        #~ f.savefig(savename + '.png', dpi=300)        


if BASELINE_VS_N_TRIALS:
    ## Plot the noise level as a function of trial count
    
    # Take the noise level as the baseline rms (median over levels)
    # Only slightly higher in LR (80 nV vs 75 nV) so pool over channels
    med_baseline_rms = big_abr_baseline_rms

    # Take the trial count as the median over levels
    med_trial_count = trial_counts.unstack('label').median(axis=1)

    # Concat these two so they line up
    joined = med_trial_count.rename(
        'trial_count').to_frame().join(med_baseline_rms.rename('noise'))

    # Plot
    f, ax = plt.subplots(figsize=(3.5, 2.5))
    f.subplots_adjust(bottom=.24, left=.25, right=.95, top=.89)
    ax.plot(
        np.log10(joined['trial_count']), 
        np.log10(joined['noise'] * 1e9), # log(nV)
        color='k', marker='o', mfc='none', ls='none', alpha=.3)

    # Pretty
    my.plot.despine(ax)
    ax.set_xticks(np.log10([30, 100, 300]))
    ax.set_xticklabels((30, 100, 300))
    ax.set_yticks(np.log10([30, 100, 300])) # log(nV)
    ax.set_yticklabels((0.03, 0.1, 0.3))
    ax.axis('scaled')
    ax.set_ylim((1.4, 2.6))
    ax.set_xlim((1.1, 2.9))

    # Labels
    ax.set_ylabel(f'baseline ({MU}V rms)')
    ax.set_xlabel('number of trials')

    # Plot a slope line
    # TODO: actually fit this and extract slope
    ax.plot(np.log10([10, 1000]), np.log10([300, 30]) - .2, 'k--', lw=.75)    

    # Savefig
    f.savefig('figures/BASELINE_VS_N_TRIALS.svg')
    f.savefig('figures/BASELINE_VS_N_TRIALS.png', dpi=300)


    ## Stats
    n_recordings_channels = len(joined)
    n_recordings = n_recordings_channels // 3
    n_mice = len(joined.index.get_level_values('mouse').unique())
    stats_filename = 'figures/STATS__BASELINE_VS_N_TRIALS'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write(
            f'n = {n_recordings} recordings * 3 channels = '
            f'{n_recordings_channels} points from {n_mice} mice, '
            'pre- and post-HL\n'
            )
        fi.write(
            'trial count is median over sound level\n'
            'baseline is rms in the (-2.5, -1.25) ms range, '
            'median over sound level\n'
            )
    
    # Echo
    with open(stats_filename) as fi:
        print(''.join(fi.readlines()))
    

if HISTOGRAM_EVOKED_RMS_BY_LEVEL:
    # Plot a histogram of the evoked signal by level, to aid in choosing
    # a consistent threshold
    # The threshold should be set to be greater than the baseline in 99% 
    # of recordings, in order to avoid false postives
    
    # Set the bins
    bins = np.linspace(-8, -4, 101)
    
    # Plot hist
    f, ax = my.plot.figure_1x1_standard()
    ax.hist(
        np.log10(big_abr_evoked_rms_orig.values), 
        histtype='step', cumulative=True, bins=bins, density=True) 

    # Pretty
    ax.set_ylim((0, 1))
    ax.set_yticks((0, .5, 1))
    ax.set_xlim((-8, -5))
    ax.set_xticks((-8, -7, -6, -5))
    ax.set_xticklabels(('10nV', '100nV', '1uV', '10uV'))
    my.plot.despine(ax)
    
    # Labels
    #~ f.text(.9, .7, 'baseline', color='k', ha='center', va='center')
    #~ f.text(.9, .6, '49 dB', color='b', ha='center', va='center')
    #~ f.text(.9, .5, 'etc', color='orange', ha='center', va='center')

    # Plot the threshold we will use 
    ax.plot(np.log10([0.3e-6, 0.3e-6]), [-.1, 1.1], color='gray', ls='--', clip_on=False)
    ax.set_xlabel('response strength (RMS)')
    ax.set_ylabel('fraction of recordings')

    f.savefig('figures/HISTOGRAM_EVOKED_RMS_BY_LEVEL.svg')
    f.savefig('figures/HISTOGRAM_EVOKED_RMS_BY_LEVEL.png', dpi=300)
