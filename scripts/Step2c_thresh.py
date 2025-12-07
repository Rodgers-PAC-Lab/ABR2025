## Computes rms(ABR) over level and thresholds
# Plots
#   PLOT_ABR_RMS_OVER_TIME
#   PLOT_ABR_POWER_VS_LEVEL
#   PLOT_ABR_POWER_VS_LEVEL_AFTER_HL
#   BASELINE_VS_N_TRIALS
#   HISTOGRAM_EVOKED_RMS_BY_LEVEL

import os
import json
import numpy as np
import pandas
import my.plot
import matplotlib.pyplot as plt
import matplotlib


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


## Load previous results
# Load results of Step1
mouse_metadata = pandas.read_pickle(
    os.path.join(output_directory, 'mouse_metadata'))
experiment_metadata = pandas.read_pickle(
    os.path.join(output_directory, 'experiment_metadata'))
recording_metadata = pandas.read_pickle(
    os.path.join(output_directory, 'recording_metadata'))

# Load results of Step2b_avg
big_abrs = pandas.read_pickle(
    os.path.join(output_directory, 'big_abrs'))
trial_counts = pandas.read_pickle(
    os.path.join(output_directory, 'trial_counts'))


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
# TODO: consider just taking the first date instead
big_abr_evoked_rms = big_abr_evoked_rms.groupby(
    [lev for lev in big_abr_evoked_rms.index.names if lev != 'n_experiment'],
    ).mean()


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


## Plots
PLOT_ABR_RMS_OVER_TIME = True
PLOT_ABR_POWER_VS_LEVEL = True
PLOT_ABR_POWER_VS_LEVEL_AFTER_HL = True
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
        [lev for lev in to_agg.index.names if lev != 'n_experiment']
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
    aut_colorbar = my.plot.generate_colorbar(
        len(label_l), mapname='inferno_r', start=0.15, stop=1)  

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

        # Plot the evoked peak time
        ax.plot([2, 2], [.03, 3], color='gray', lw=.75)
    
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
    f.savefig(os.path.join('figures', 'PLOT_ABR_RMS_OVER_TIME.svg'))
    f.savefig(os.path.join('figures', 'PLOT_ABR_RMS_OVER_TIME.png'), dpi=300)
    
    
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

    #~ ## TODO: analyze effect of speaker_side and channel on evoked response here
    #~ # huge effect of label and mouse, of course
    #~ # very strong effect of ipsi (contra is stronger)
    #~ # strong effect of channel (LV is stronger)
    #~ # mild effect of speaker_side (L is stronger)
    #~ # However, the actual effect size of contra is pretty small (around the 
    #~ # size of a 4 dB difference, and only about twice the size of the effect
    #~ # of channel or speaker_side), so I don't know that it's worth
    #~ # highlighting
    #~ actual_data = np.log10(
        #~ this_big_abr_evoked_rms.drop('LR', level='channel').iloc[:, ::40])
    
    #~ # prep for anova
    #~ to_aov = actual_data.stack().rename('resp').reset_index()
    #~ to_aov['ipsi'] = to_aov['speaker_side'] == to_aov['channel'].str[0]
    
    #~ # run anova
    #~ my.stats.anova(
        #~ to_aov, 'resp ~ mouse + label + channel + speaker_side + ipsi')
    
    # Set up ax_rows and ax_cols
    channel_l = ['LV', 'RV', 'LR']
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
    stats_filename = 'figures/STATS__PLOT_ABR_POWER_VS_LEVEL'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write(f'n = {n_mice} mice, pre-HL only\n')
        fi.write(
            'mean power vs level over recordings within date, '
            'then mean over date within mouse.\n'
            'computed one threshold (interpolated) per mouse\n'
            )
    
    # Echo
    with open(stats_filename) as fi:
        print(''.join(fi.readlines()))
        
    # TODO: grouped bar plot on thresholds
    # my.plot.grouped_bar_plot(
    #   this_threshold_db.unstack('mouse'), index2plot_kwargs=lambda idx: 
    #   {'fc': 'b' if idx['speaker_side'] == 'L' else 'r'})

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
    channel_l = ['LV', 'RV', 'LR']
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
        thresh_by_mouse_diff_err = thresh_by_mouse_diff.groupby('HL_type').sem()
        thresh_by_mouse_diff_N = thresh_by_mouse_diff.groupby('HL_type').size()
        
        # Write out
        stats_filename = f'figures/PLOT_ABR_POWER_VS_LEVEL_AFTER_HL__thresh__{speaker_side}'
        with open(stats_filename, 'w') as fi:
            fi.write(stats_filename + '\n')
            fi.write(f'mean threshold:\n{thresh_by_mouse_mu}\n')
            fi.write(f'SEM threshold:\n{thresh_by_mouse_err}\n')
            fi.write(f'N threshold:\n{thresh_by_mouse_N}\n')
            fi.write(f'mean diff threshold:\n{thresh_by_mouse_diff_mu}\n')
            fi.write(f'SEM diff threshold:\n{thresh_by_mouse_diff_err}\n')
            fi.write(f'N diff threshold:\n{thresh_by_mouse_diff_N}\n')
        
        # Echo
        with open(stats_filename) as fi:
            print(''.join(fi.readlines()))        
        
        

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
    f, ax = my.plot.figure_1x1_standard()
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
    ax.set_xlim((1.2, 2.8))

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
