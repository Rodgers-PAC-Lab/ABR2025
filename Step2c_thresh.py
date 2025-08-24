## Computes rms(ABR) over level and thresholds
# Plots
#   PLOT_ABR_RMS_OVER_TIME
#   PLOT_ABR_POWER_VS_LEVEL
#   PLOT_ABR_POWER_VS_LEVEL_AFTER_HL
#   BASELINE_VS_N_TRIALS
#   HISTOGRAM_EVOKED_RMS_BY_LEVEL

import os
import datetime
import glob
import json
import scipy.signal
import numpy as np
import pandas
import paclab.abr
import paclab.abr.abr_plotting
import my.plot
import matplotlib.pyplot as plt
import tqdm
import matplotlib


## Plotting
my.plot.manuscript_defaults()
my.plot.font_embed()


## Paths
# Load the required file filepaths.json (see README)
with open('filepaths.json') as fi:
    paths = json.load(fi)

# Parse into paths to raw data and output directory
raw_data_directory = paths['raw_data_directory']
output_directory = paths['output_directory']


## Load previous results
# Load results of Step1
mouse_metadata = pandas.read_pickle(
    os.path.join(output_directory, 'mouse_metadata'))
experiment_metadata = pandas.read_pickle(
    os.path.join(output_directory, 'experiment_metadata'))
recording_metadata = pandas.read_pickle(
    os.path.join(output_directory, 'recording_metadata'))

# Load results of Step2
big_abrs = pandas.read_pickle(
    os.path.join(output_directory, 'big_abrs'))
trial_counts = pandas.read_pickle(
    os.path.join(output_directory, 'trial_counts'))


## Join after_bilateral_HL on big_abrs
# Join after_HL onto big_abrs
big_abrs = my.misc.join_level_onto_index(
    big_abrs, 
    experiment_metadata.set_index(['mouse', 'date'])['after_HL'], 
    join_on=['mouse', 'date']
    )

# Join HL_type onto big_abrs
big_abrs = my.misc.join_level_onto_index(
    big_abrs, 
    mouse_metadata.set_index('mouse')['HL_type'], 
    join_on='mouse',
    )


## Params
sampling_rate = 16000


## Calculate the stdev(ABR) as a function of level
# window=20 (1.25 ms) seems the best compromise between smoothing the whole
# response and localizing it to a reasonably narrow window (and not extending
# into the baseline period)
# Would be good to extract more baseline to use here
# The peak is around sample 34 (2.1 ms), ie sample 24 - 44, and there is a
# variable later peak.
big_abr_stds = big_abrs.T.rolling(window=20, center=True, min_periods=1).std().T

# Use samples -40 to -20 as baseline
# Generally this should be <0.25 uV, but the actual value depends on how
# the averaging was done
big_abr_baseline_rms = big_abr_stds.loc[:, -30].unstack('label')

# Choose a baseline for each recording as the median over levels
# It's lognormal so mean might be skewed. A mean of log could be good
big_abr_baseline_rms = big_abr_baseline_rms.median(axis=1)

# Use samples 24 - 44 as evoked peak
# Evoked response increases linearly with level in dB
# Interestingly, each recording appears to be multiplicatively scaled
# (shifted up and down on a log plot). The variability in microvolts increases
# with level, but the variability in log-units is consistent over level.
big_abr_evoked_rms = big_abr_stds.loc[:, 34].unstack('label')

# TODO: consider smoothing traces before finding threshold crossing

# Determine threshold crossing as 3*baseline. Note: more averaging will
# decrease baseline and therefore threshold, as will better noise levels.
# Update: now using a fixed threshold in uV
over_thresh = big_abr_evoked_rms.T > 0.3e-6 # 3 * big_abr_baseline_rms
over_thresh = over_thresh.T.stack()

# threshold
# typically a bit better on LR even though LR has slightly higher baseline
threshold_db = over_thresh.loc[over_thresh.values].groupby(
    [lev for lev in over_thresh.index.names if lev != 'label'],
    dropna=False,
    ).apply(
    lambda df: df.index[0][-1])

# reindex to get those that are never above threshold
threshold_db = threshold_db.reindex(big_abr_baseline_rms.index)

# error check that we always have a threshold
assert not threshold_db.isnull().any()


## Plots
PLOT_ABR_RMS_OVER_TIME = True
PLOT_ABR_POWER_VS_LEVEL = True
PLOT_ABR_POWER_VS_LEVEL_AFTER_HL = False
BASELINE_VS_N_TRIALS = True
HISTOGRAM_EVOKED_RMS_BY_LEVEL = True

if PLOT_ABR_RMS_OVER_TIME:
    ## Plot the smoothed rms of the ABR over time by condition
    # Shared t
    t = big_abrs.columns / sampling_rate * 1000

    # Slice out pre-HL only
    this_big_abr_stds = big_abr_stds.xs(
        False, level='after_HL').droplevel('HL_type')

    # Aggregate over recordings
    to_agg = this_big_abr_stds.groupby(
        [lev for lev in this_big_abr_stds.index.names if lev != 'recording']
        ).mean()

    # Aggregate over date
    # TODO: keep just the first session from each mouse instead?
    to_agg = to_agg.groupby(
        [lev for lev in to_agg.index.names if lev != 'date']
        ).mean()

    # Make mouse the replicates on the columns
    to_agg = to_agg.stack().unstack('mouse')

    # Agg
    # TODO: log10 before agg?
    agg_mean = to_agg.mean(axis=1).unstack('timepoint')
    agg_err = to_agg.sem(axis=1).unstack('timepoint')

    # Plot
    channel_l = ['LV', 'RV', 'LR']
    speaker_side_l = ['L', 'R']

    # Set up colorbar
    # Always do the lowest labels last
    label_l = sorted(
        agg_mean.index.get_level_values('label').unique(), 
        reverse=True)
    aut_colorbar = paclab.abr.abr_plotting.generate_colorbar(
        len(label_l), mapname='inferno_r', start=0.15, stop=1)[::-1]  

    # Set up ax_rows and ax_cols
    channel_l = ['LV', 'RV', 'LR']
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

        # Plot each
        for n_level, level in enumerate(subdf.index):
            # Get color
            color = aut_colorbar[label_l.index(level)]

            # Plot in uV
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

        # Mark the evoked period
        ax.fill_betweenx(y=(.03, 3), x1=.875, x2=3.375, color='gray', alpha=.25)        

        # Nicer log labels
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    #~ # Legend
    #~ for n_label, (label, color) in enumerate(zip(label_l, aut_colorbar)):
        #~ if np.mod(n_label, 2) != 0:
            #~ continue
        #~ f.text(
            #~ .95, .85 - n_label * .02, f'{label} dB',
            #~ color=color, ha='center', va='center', size=12)

    # Pretty
    ax.set_xlim((-2, 8))
    ax.set_ylim((.03, 3))
    ax.set_xticks([0, 3, 6])
    f.text(.51, .01, 'time (ms)', ha='center', va='bottom')
    f.text(.05, .55, 'rolling RMS of ABR (uV)', rotation=90, ha='center', va='center')

    #~ # Label the channel
    #~ for n_channel, channel in enumerate(channel_l):
        #~ axa[n_channel, 0].set_ylabel(channel, labelpad=20)
    #~ axa[1, 1].set_ylabel
    
    # Label the speaker side
    axa[0, 0].set_title('sound from left')
    axa[0, 1].set_title('sound from right')
    
    # Savefig
    f.savefig(os.path.join('figures', 'PLOT_ABR_RMS_OVER_TIME.svg'))
    f.savefig(os.path.join('figures', 'PLOT_ABR_RMS_OVER_TIME.png'), dpi=300)
    
    # Stats
    stats_filename = 'figures/STATS__PLOT_ABR_RMS_OVER_TIME'
    with open(stats_filename, 'w') as fi:
        fi.write(f'n = {to_agg.shape[1]} mice\n')
        fi.write(
            'compute rolling RMS for each recording * level, '
            'then mean over recordings within date, '
            'then mean over date within mouse, '
            'then mean and SEM over mice, '
            'then plot on log scale\n')

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

    # Calculate mean threshold per mouse
    avg_thresh = this_threshold_db.groupby(
        ['channel', 'speaker_side', 'mouse']).mean()
    
    # Set up ax_rows and ax_cols
    channel_l = ['LV', 'RV', 'LR']
    speaker_side_l = ['L', 'R']
    
    # Make plot
    f, axa = plt.subplots(
        len(channel_l), len(speaker_side_l),
        sharex=True, sharey=True, figsize=(5, 4))
    f.subplots_adjust(
        left=.17, right=.89, top=.95, bottom=.15, hspace=.15, wspace=.12)

    # Average and iterate over channel and speaker side configs
    agged_rms = this_big_abr_evoked_rms.groupby(
        ['channel', 'speaker_side', 'mouse']).mean()
    gobj = agged_rms.groupby(['channel','speaker_side'])
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
            mouse_evoked = subdf.loc[mouse].copy()
            ax.semilogy(mouse_evoked* 1e6,  lw=.75, color='gray', alpha=.6)

            # Get avg threshold
            mouse_thresh = avg_thresh.loc[channel].loc[speaker_side].loc[mouse].item()

            # Round mouse_thresh to nearest actual sound level
            thresh_diffs = np.abs(mouse_evoked.index - mouse_thresh)
            lowest_diff_idx = np.argmin(thresh_diffs)
            near_thresh = mouse_evoked.index[lowest_diff_idx]

            # Plot mouse threshold (rounded)
            if not pandas.isnull(mouse_thresh):
                ax.semilogy(
                    [near_thresh],
                    [mouse_evoked.loc[near_thresh] * 1e6],
                    'ro', ms=2.5, alpha=.5)

        # Despine
        my.plot.despine(ax)
        ax.set_yticks([0.1, 1])
        ax.set_xticks([50, 70, 90])
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # Axis labels on the bottom
    for ax in axa[2,:]:
        ax.set_xlabel('sound level (dB)')
    axa[1,0].set_ylabel('evoked ABR (uV RMS)')
    
    # Savefig
    f.savefig('figures/PLOT_ABR_POWER_VS_LEVEL.svg')
    f.savefig('figures/PLOT_ABR_POWER_VS_LEVEL.png', dpi=300)

if PLOT_ABR_POWER_VS_LEVEL_AFTER_HL:
    ## Plot sham and bilateral pre- and post-HL
    
    # Include only HL mice
    this_threshold_db = threshold_db.drop(np.nan, level='HL_type')
    this_big_abr_evoked_rms = big_abr_evoked_rms.drop(np.nan, level='HL_type')

    # Aggregate threshold over date and recording, maintaining after_HL
    threshold_db_agg = this_threshold_db.groupby(
        [lev for lev in this_threshold_db.index.names if lev != 'recording']
        ).mean()
    threshold_db_agg = threshold_db_agg.groupby(
        [lev for lev in threshold_db_agg.index.names if lev != 'date']
        ).mean()

    # Aggregate evoked RMS over date and recording, maintaining after_HL
    big_abr_evoked_rms_agg = this_big_abr_evoked_rms.groupby(
        [lev for lev in this_big_abr_evoked_rms.index.names if lev != 'recording']
        ).mean()
    big_abr_evoked_rms_agg = big_abr_evoked_rms_agg.groupby(
        [lev for lev in big_abr_evoked_rms_agg.index.names if lev != 'date']
        ).mean()
        
    # To iterate over
    channel_l = ['LV', 'RV', 'LR']
    speaker_side_l = ['L', 'R']
    HL_type_l = ['sham', 'bilateral']
    after_HL_l = [False, True]
    
    # Iterate over HL_type (figures)
    for HL_type in HL_type_l:
        
        ## Plot evoked RMS vs level
        f, axa = plt.subplots(
            len(channel_l), len(speaker_side_l),
            sharex=True, sharey=True, figsize=(5, 4))
        f.subplots_adjust(
            left=.17, right=.89, top=.95, bottom=.15, hspace=.15, wspace=.12)

        # Iterate over channel * speaker_side
        gobj = big_abr_evoked_rms_agg.xs(HL_type, level='HL_type').groupby(
            ['channel','speaker_side'])
        for (channel, speaker_side), subdf in gobj:
            
            # Get ax
            ax = axa[
                channel_l.index(channel),
                speaker_side_l.index(speaker_side),
            ]
            
            # Iterate over mice
            for mouse in subdf.index.get_level_values('mouse').unique():
                
                # Iterate over after_HL
                for after_HL in after_HL_l:
                    
                    # Color by after_HL
                    color = 'b' if after_HL else 'orange'
                    
                    # Slice evoked RMS and threshold
                    topl = subdf.loc[after_HL].loc[mouse].loc[channel].loc[
                        speaker_side]
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

            # Despine
            my.plot.despine(ax)
            ax.set_yticks([0.1, 1])
            ax.set_xticks([50, 70, 90])
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

        # Axis labels on the bottom
        for ax in axa[2,:]:
            ax.set_xlabel('sound level (dB)')
        axa[1,0].set_ylabel('evoked ABR (uV RMS)')
        
        # Savefig
        savename = f'figures/PLOT_ABR_POWER_VS_LEVEL_AFTER_HL__{HL_type}'
        f.savefig(savename + '.svg')
        f.savefig(savename + '.png', dpi=300)        
        
        
        ## Second plot of threshold
        f, axa = plt.subplots(
            len(channel_l), len(speaker_side_l),
            sharex=True, sharey=True, figsize=(5, 4))
        f.subplots_adjust(
            left=.17, right=.89, top=.95, bottom=.15, hspace=.15, wspace=.12)

        # Iterate over channel * speaker_side
        gobj = threshold_db_agg.xs(HL_type, level='HL_type').groupby(
            ['channel','speaker_side'])
        for (channel, speaker_side), subdf in gobj:
            
            # Drop grouping keys
            subdf = subdf.droplevel(['channel', 'speaker_side'])
            
            # Get mouse on columns
            subdf = subdf.unstack('mouse')
            
            # Make sure it's pre before post
            subdf = subdf.reindex([False, True])
            
            # Get ax
            ax = axa[
                channel_l.index(channel),
                speaker_side_l.index(speaker_side),
            ]        
            
            # Plot
            ax.plot(subdf.values, ls='-', marker='o', color='k', mfc='none')
            
            # Despine
            my.plot.despine(ax)
        
        # Consistent axis limits
        ax.set_ylim((50, 85))
        ax.set_yticks((50, 65, 80))
        ax.set_xticks((0, 1))
        ax.set_xlim((-.5, 1.5))
        ax.set_xticklabels(('pre', 'post'))
        
        # Savefig
        savename = f'figures/PLOT_ABR_POWER_VS_LEVEL_AFTER_HL__thresh__{HL_type}'
        f.savefig(savename + '.svg')
        f.savefig(savename + '.png', dpi=300)
    

if BASELINE_VS_N_TRIALS:
    ## Plot the noise level as a function of trial count
    
    # Take the noise level as the baselie rms (median over levels)
    noise_by_config = big_abr_baseline_rms

    # Take the trial count as the median over levels
    med_trial_count = trial_counts.unstack('label').median(axis=1)

    # Concat these two so they line up
    joined = med_trial_count.rename(
        'trial_count').to_frame().join(noise_by_config.rename('noise'))

    # Take the evoked signal by level
    evoked_by_level = big_abr_evoked_rms.median()

    fit = 1.1e-7 * np.sqrt(100) / np.sqrt(med_trial_count)

    f, ax = my.plot.figure_1x1_standard()
    ax.plot(
        np.log10(joined['trial_count']), 
        np.log10(joined['noise'] * 1e9), # log(nV)
        color='k', marker='o', mfc='none', ls='none', alpha=.3)

    ax.set_xticks((1.5, 2, 2.5))
    ax.set_xticklabels((30, 100, 300))
    ax.set_yticks((1.5, 2, 2.5)) # log(nV)
    ax.set_yticklabels((0.03, 0.1, 0.3))
    ax.axis('scaled')
    ax.set_ylim((1.4, 2.6))
    ax.set_xlim((1.2, 2.7))
    ax.set_ylabel('RMS of baseline\nin trial average (uV)')
    ax.set_xlabel('number of trials')

    my.plot.despine(ax)
    #~ for label, evoked in evoked_by_level[:3].items():
        #~ ax.plot(np.log10([50, 250]), np.log10([evoked * 1e9, evoked * 1e9]), '-')

    f.savefig('figures/BASELINE_VS_N_TRIALS.svg')
    f.savefig('figures/BASELINE_VS_N_TRIALS.png', dpi=300)


if HISTOGRAM_EVOKED_RMS_BY_LEVEL:
    # Plot a histogram of the evoked signal by level, to aid in choosing
    # a consistent threshold
    # The threshold should be set to be greater than the baseline in 99% 
    # of recordings, in order to avoid false postives
    
    # Set the bins
    bins = np.linspace(-8, -4, 101)
    
    f, ax = my.plot.figure_1x1_standard()
    ax.hist(
        np.log10(big_abr_evoked_rms.values), 
        histtype='step', cumulative=True, bins=bins, density=True) 

    ax.hist(
        np.log10(big_abr_baseline_rms.values), 
        histtype='step', cumulative=True, bins=bins, density=True, color='k') 

    ax.set_ylim((0, 1))
    ax.set_yticks((0, .5, 1))
    ax.set_xlim((-8, -5))
    ax.set_xticks((-8, -7, -6, -5))
    ax.set_xticklabels(('10nV', '100nV', '1uV', '10uV'))
    my.plot.despine(ax)
    f.text(.9, .7, 'baseline', color='k', ha='center', va='center')
    f.text(.9, .6, '49 dB', color='b', ha='center', va='center')
    f.text(.9, .5, 'etc', color='orange', ha='center', va='center')
    ax.plot([-6.5, -6.5], [-.1, 1.1], color='gray', ls='--', clip_on=False)
    ax.set_xlabel('evoked signal (RMS)')
    ax.set_ylabel('fraction of recordings')

    f.savefig('figures/HISTOGRAM_EVOKED_RMS_BY_LEVEL.svg')
    f.savefig('figures/HISTOGRAM_EVOKED_RMS_BY_LEVEL.png', dpi=300)
