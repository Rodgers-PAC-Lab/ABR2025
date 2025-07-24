## This directory analyzes all data from 241126_cohort ABR
# Rowan: What does this script do? It doesn't save date. Many plots seem out
# of date. -CR
#
# This script pulls out neural responses around clicks
# Quantifies correlation in between-ears recordings
# Quantifies threshold
#
# Mice with hearing loss are identified as higher threshold in vertex-ear,
# zero or positive correlation in between-ears, and flipped primary peak
# amplitude in between-ears

#
# TODO: determine if LR = LV - RV
#
# Generally, the stdev of the ABR (averaging over all trials in a recording)
# in the baseline period (-40 to -20 samples) should be about 0.25 uV (-6.6 log units). 
# If it's above 1 uV it's definitely too high. 
# Obviously this somewhat varies depending on the number of trials, whether
# recordings are averaged together, etc

import os
import datetime
import glob
import json
import matplotlib
import scipy.signal
import numpy as np
import matplotlib
import pandas
import paclab.abr
import paclab.abr.abr_plotting
import my.plot
import matplotlib.pyplot as plt

my.plot.font_embed()

def pre_or_post(timepoint):
    if type(timepoint)==float:
        print(timepoint)
    if 'pre' in timepoint:
        res = 'pre'
    elif 'post' in timepoint:
        res = 'post'
    else:
        res = 'ERROR, not pre or post!'
    return res
def HL_type(mouse):
    if mouse in bilateral_mouse_l:
        res = 'bilateral'
    elif mouse in sham_mouse_l:
        res = 'sham'
    else:
        res = 'ERROR, UNKNOWN MOUSE'
    return res


def plot_single_ax_abr(abr_subdf, ax, sampling_rate=16000):
    t = abr_subdf.columns/ sampling_rate * 1000
    for label_i in abr_subdf.index.sort_values(ascending=False):
        aut_colorbar = paclab.abr.abr_plotting.generate_colorbar(
            len(abr_subdf.index), mapname='inferno_r', start=0.15, stop=1)
        color_df = pandas.DataFrame(aut_colorbar,
                                    index=abr_subdf.index.sort_values(ascending=True))
        ax.plot(t, abr_subdf.loc[label_i].T * 1e6, lw=.75,
                color=color_df.loc[label_i], label=label_i)
    return ax

## Params
sampling_rate = 16000 # TODO: store in recording_metadata
loudest_dB = 91

# Tenatative because I'm blinded, but come on it's obvious
sham_mouse_l = ['Cat_227', 'Cat_228']
bilateral_mouse_l = ['Cat_226', 'Cat_229']
pre_times_l = ['apreA','apreB']
post_times_l = ['postA','postB']


## Paths
with open('filepaths.json') as fi:
    paths = json.load(fi)

# Parse into paths to raw data and output directory
raw_data_directory = paths['raw_data_directory']
output_directory = paths['output_directory']


## Load results of Step1
cohort_experiments = pandas.read_pickle(
    os.path.join(output_directory, 'cohort_experiments'))
recording_metadata = pandas.read_pickle(
    os.path.join(output_directory, 'recording_metadata'))

# Fillna
cohort_experiments['HL'] = cohort_experiments['HL'].fillna('none')

# Drop those with 'include' == False
recording_metadata = recording_metadata[recording_metadata['include'] == True]


## Load results of Step2
big_triggered_neural = pandas.read_pickle(
    os.path.join(output_directory, 'big_triggered_neural'))
big_abrs = pandas.read_pickle(
    os.path.join(output_directory,'big_abrs'))
threshold_db = pandas.read_pickle(
    os.path.join(output_directory,'thresholds'))

# Average ABRs across recordings
avged_abrs = big_abrs.groupby(['date', 'mouse', 'channel', 'speaker_side','label']).mean()


## Calculate the stdev(ABR) as a function of level
# window=20 (1.25 ms) seems the best compromise between smoothing the whole
# response and localizing it to a reasonably narrow window (and not extending
# into the baseline period)

big_abr_stds = big_abrs.T.rolling(window=20, center=True, min_periods=1).std().T

# Use samples 24 - 44 as evoked peak
# Evoked response increases linearly with level in dB
# Interestingly, each recording appears to be multiplicatively scaled
# (shifted up and down on a log plot). The variability in microvolts increases
# with level, but the variability in log-units is consistent over level.
big_abr_evoked_rms = big_abr_stds.loc[:, 34].unstack('label')

evoked_rms_timepoint = big_abr_evoked_rms.join(
    cohort_experiments.set_index(['mouse', 'date'])[['timepoint','HL']], on=('mouse', 'date'), how='left')
evoked_rms_timepoint['config'] = evoked_rms_timepoint.index.get_level_values(
    'channel') + evoked_rms_timepoint.index.get_level_values('speaker_side')
evoked_rms_timepoint = evoked_rms_timepoint.droplevel(level=['date', 'channel', 'speaker_side'], axis='index')
evoked_rms_timepoint = evoked_rms_timepoint.reset_index().set_index(
    ['mouse', 'HL', 'recording', 'timepoint', 'config'])
# MEAN OVER RECORDING
evoked_rms_timepoint = evoked_rms_timepoint.groupby(
    ['mouse', 'HL', 'timepoint', 'config']).mean()

# Get the peak
# The peak is generally around sample 35, 25, 45-50, or more rarely 75
# But anything is possible
# Rather than be too strict, I'll just allow anything after the initial click,
# although this will generally be noise for the softest condition
# In the end the plots look almost identical with _peak or _rms
big_abr_evoked_peak = big_abrs.loc[:, 10:].abs().max(axis=1)
big_abr_evoked_peak = pandas.DataFrame(big_abr_evoked_peak).join(cohort_experiments.set_index(['date','mouse'])[['timepoint','HL']],on=(['date','mouse']))
big_abr_evoked_peak = big_abr_evoked_peak.rename(columns={0:'pk_V'})
big_abr_evoked_peak = big_abr_evoked_peak.reset_index()
big_abr_evoked_peak['prevpost'] = big_abr_evoked_peak['timepoint'].apply(lambda x: pre_or_post(x))

threshold_avgs_db = threshold_db.groupby(['date', 'mouse','speaker_side', 'channel']).mean()
threshold_avgs_db['config'] = (threshold_avgs_db._get_label_or_level_values('channel',axis=0) +
                          threshold_avgs_db._get_label_or_level_values('speaker_side',axis=0))

threshold_avgs_db = threshold_avgs_db.reset_index().set_index(['date', 'mouse','config'])
threshold_avgs_db = threshold_avgs_db.join(cohort_experiments.set_index(['mouse','date'])[['timepoint','HL']],on=['mouse','date'])
threshold_avgs_db = threshold_avgs_db.reset_index().set_index(['date', 'mouse','config','timepoint'])



## Identify primary peak in between-ears recording
# The most prominent feature is a large peak between 31 and 40 samples
# (1.94-2.5 ms) post stimulus onset. Median is 35 samples (2.2 ms).
# There should only be one peak in this range
# The peak before this is usually smaller, but not always. When it's late
# and large, it can get confused with this peak. This happens with Saturn_7
# and Sodapop_8. So we can't make the lower end too low without spuriously 
# identifying that earlier peak, but we also can't make the lower end too
# high without losing some true early major peaks.
#
# The upper end of the border is easier to define - subsequent peaks are 
# generally smaller, later, or both


# Include only loudest label
LR_loud = big_abrs.xs('LR', level='channel').xs(loudest_dB, level='label').sort_index()

# Identify peak time
peak_sample_l = []
peak_sample_keys_l = []
for idx in LR_loud.index:
    # Slice
    start = 31 # cannot be less than this
    stop = 42 # inclusive. This one is insensitive - no better later peaks
    peak_samples, peak_props = scipy.signal.find_peaks(
        LR_loud.loc[idx, start:stop].abs(),
        height=1e-7,
        distance=(stop - start + 1),
        )
    
    if len(peak_samples) != 1:
        1/0
    
    peak_sample = peak_samples[0] + start
    
    peak_sample_l.append(peak_sample)
    peak_sample_keys_l.append(idx)

# Concat
midx = pandas.MultiIndex.from_tuples(
    peak_sample_keys_l, names=LR_loud.index.names)

# Form a DataFrame of info about this primary peak
primary_peak_LR = pandas.Series(
    peak_sample_l, index=midx).rename('latency').sort_index().to_frame()

# Join on peak value
primary_peak_LR['amplitude'] = [
    LR_loud.loc[idx, latency] for idx, latency in primary_peak_LR['latency'].items()]



## Correlate L and R response in LR_loud (or all)
# TODO: Check for latency shifts between L and R for vertex-ear recording here
rec_l = []
rec_stds_l = []
for mouse in big_abrs.index.get_level_values('mouse').unique():
    for i_day in big_abrs.loc[:,mouse,:].index.get_level_values('date').unique():
        # Get the correlations
        r_LR_all = np.corrcoef(
            avged_abrs.loc[i_day, mouse, 'LR','L'].values.flatten(),
            avged_abrs.loc[i_day, mouse, 'LR', 'R'].values.flatten(),
)[0, 1]

        # Get the corresponding amplitude and latency of the primary peaks (for the loudest sound)
        i_L_amp = primary_peak_LR.loc[i_day, mouse, :, 'L']['amplitude'].mean()
        i_R_amp = primary_peak_LR.loc[i_day, mouse, :, 'R']['amplitude'].mean()
        i_L_lat = primary_peak_LR.loc[i_day, mouse, :, 'L']['latency'].mean()
        i_R_lat = primary_peak_LR.loc[i_day, mouse, :, 'R']['latency'].mean()
        rec_l.append([mouse, i_day, r_LR_all, i_L_amp, i_R_amp, i_L_lat, i_R_lat])

        # Make a separate frame for the standard deviations
        i_L_std = big_abr_stds.loc[i_day,mouse, :, 'LR','L'].groupby(
            ['label']).mean()
        i_R_std = big_abr_stds.loc[i_day, mouse,:, 'LR', 'R'].groupby(
            ['label']).mean()
        rec_stds_l.append([mouse, i_day, i_L_std, i_R_std])


between_ears_diagnostics = pandas.DataFrame(rec_l, columns=[
    'mouse','date','corr','L_amp','R_amp','L_lat','R_lat'])
between_ears_diagnostics = between_ears_diagnostics.set_index(['date','mouse'])
between_ears_diagnostics['L_amp'] *= 1e6
between_ears_diagnostics['R_amp'] *= 1e6

between_ears_diagnostics = between_ears_diagnostics.join(
    cohort_experiments[['HL', 'strain', 'genotype', 'mouse','date']].set_index(['date','mouse']))
between_ears_diagnostics = between_ears_diagnostics.reset_index().set_index(
    ['HL', 'strain', 'genotype', 'mouse','date'])
# Typically corr should be -.8 to -.3, median -.45
# As of 6-4-25, median is -0.7. Huh.

between_ears_stds = pandas.DataFrame(rec_stds_l, columns=['mouse','date','L_std','R_std'])
between_ears_stds = between_ears_stds.set_index(['date','mouse'])


avged_abrs_timepoints = avged_abrs.join(
    cohort_experiments.set_index(['mouse', 'date'])[['timepoint','HL']],
    on=('mouse', 'date'), how='left')
avged_abrs_timepoints = avged_abrs_timepoints.droplevel(level='date',axis='index')
avged_abrs_timepoints = avged_abrs_timepoints.reset_index().set_index(
    ['timepoint','mouse', 'HL','channel', 'speaker_side', 'label'])


# Using 'HL_group' here because I need a way to group sham vs bilateral mice
# even in pre-HL conditions before I average data
abrs_avged_by_HL = avged_abrs_timepoints.copy().reset_index()
abrs_avged_by_HL['prevpost'] = abrs_avged_by_HL['timepoint'].apply(pre_or_post)
abrs_avged_by_HL['HL_group'] = abrs_avged_by_HL['mouse'].apply(HL_type)
# Drop any mice who aren't sham or bilateral
abrs_avged_by_HL = abrs_avged_by_HL.loc[abrs_avged_by_HL['HL_group'].isin(['bilateral','sham'])]
abrs_avged_by_HL = abrs_avged_by_HL.drop(['mouse','timepoint','HL'],axis='columns')
abrs_avged_by_HL = abrs_avged_by_HL.set_index(['prevpost', 'HL_group', 'channel', 'speaker_side', 'label'])
abrs_avged_by_HL = abrs_avged_by_HL.groupby(['prevpost', 'HL_group', 'channel', 'speaker_side', 'label']).mean()
abrs_avged_by_HL = abrs_avged_by_HL.sort_index()

evoked_rms_by_HL = evoked_rms_timepoint.copy().reset_index()
evoked_rms_by_HL['prevpost'] = evoked_rms_by_HL['timepoint'].apply(pre_or_post)
evoked_rms_by_HL['HL_group'] = evoked_rms_by_HL['mouse'].apply(HL_type)
evoked_rms_by_HL = evoked_rms_by_HL.loc[evoked_rms_by_HL['HL_group'].isin(['bilateral','sham'])]
evoked_rms_by_HL = evoked_rms_by_HL.drop(['mouse','timepoint','HL'],axis='columns').set_index(['prevpost','HL_group','config'])
evoked_rms_by_HL = evoked_rms_by_HL.groupby(['prevpost','HL_group','config']).mean()

mouse_l = big_abrs.index.get_level_values('mouse').unique()


## Plot
IMSHOW_COMPARE_DAYS = False
PLOT_ABR_COMPARE_DAYS = False
ABR_POWER_VS_LEVEL_MOUSE_COMPARE_DAYS = False

IMSHOW_ABRS = False
PLOT_ABR_POWER_VS_LEVEL_ALL_MICE = False
PLOT_ABR_POWER_POSTHL_SHIFTED = False

PLOT_ABR_LR_PREHL_BY_HL = False
PLOT_ABR_LR_POSTHL_BY_HL = False

PLOT_LR_IPSVCONT = False
PLOT_ABR_PRE_V_POST_BY_HL = False
IMSHOW_PRE_V_POST_BY_HL = False
PLOT_ABR_POWER_PRE_V_POST = False

# Turns out all the plots in this have been carried over since like February and are not up to date or helpful
# Starting fresh here
if IMSHOW_COMPARE_DAYS:
    ## Plot the ABR for every config, date tested, and mouse
    # Each mouse is a figure, configs are rows, and dates arr columns
    
    # Get time in ms
    t = avged_abrs.columns / sampling_rate * 1000
    # Iterate over mice
    for mouse in mouse_l:
        # Slice
        mouse_abrs = avged_abrs.loc[:, mouse, :].copy()
        mouse_datestrings_l = sorted(
            mouse_abrs.index.get_level_values('date').unique())

        config_l = mouse_abrs.index.droplevel(
            ['date', 'label']).drop_duplicates().tolist()

        # Make plot
        f, axa = plt.subplots(
            len(config_l),  len(mouse_datestrings_l),
            sharex=True, figsize=(10, 10))
        f.subplots_adjust(left=.095, right=.95, top=.9, bottom=.08)
        f.suptitle(mouse + ' compared by date', fontsize=18,fontweight='bold')

        # Label plot rows with the speaker side and channel config
        y_pos = 0.82
        spacer = ((0.9-0.08)/len(config_l))
        check = []
        for i, config in enumerate(config_l):
            f.text(0.01, y_pos, config[0] + ' ' + config[1], fontsize=15)
            y_pos = y_pos - spacer-0.005
            check.append(y_pos)

        # Label plot columns with the date(s)
        if len(mouse_datestrings_l)==1:
            axa[0].set_title(mouse_datestrings_l[0],fontsize=14)
        else:
            for date in mouse_datestrings_l:
                axa[0,mouse_datestrings_l.index(date)].set_title(date,fontsize=14)

        # Imshow each one
        gobj = mouse_abrs.groupby(['date', 'channel', 'speaker_side'])
        for (date,channel, speaker_side), subdf in gobj:
            # droplevel
            subdf = subdf.droplevel(['date','speaker_side','channel'])

            # Get ax
            if axa.ndim==1:
                ax = axa[config_l.index((channel, speaker_side))]
                axa[-1].set_xlabel('time (ms)')
            else:
                ax = axa[
                    config_l.index((channel,speaker_side)),
                    mouse_datestrings_l.index(date)
                ]
                if ax in axa[-1]:
                    ax.set_xlabel('time (ms)')
            # Plot
            config = my.plot.imshow(subdf, x=t, center_clim=True, origin='lower', ax=ax)

            # Title
            # ax.set_title('{} {}'.format(speaker_side , channel))
            # Set date title at the top of the column

            # Pretty
            ax.set_xlim((t[0], t[-1]))


            # if ax in axa[:,0]:
            #     ax.set_ylabel('sound level (dB)')

        # Remove any empty axes
        for ax in axa.flatten():
            if len(ax.images) == 0:
                ax.set_visible(False)

        # Harmonize
        my.plot.harmonize_clim_in_subplots(fig=f, center_clim=True, clim=(-3e-6,3e-6), trim=.999)
        if axa.ndim == 1:
            f.colorbar(config, ax=axa[:])
        else:
            f.colorbar(config,ax=axa[:,-1])
        # Savefig
        savename = f'COMPARE_DAYS_{mouse}_IMSHOW'
        f.savefig(os.path.join(output_directory, savename + '.svg'))
        f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)

if PLOT_ABR_COMPARE_DAYS:
    ## Just like IMSHOW_COMPARE_DAYS, but as a multi-line plot
    # TODO: combine with IMSHOW_COMPARE_DAYS
    
    # Get list of mice

    # Get time in ms
    t = avged_abrs.columns / sampling_rate * 1000

    # Iterate over mice
    for mouse in mouse_l:
        # Slice
        mouse_abrs = avged_abrs.loc[:, mouse, :].copy()

        # Get dates and configurations used for this mouse
        mouse_datestrings_l = sorted(
            mouse_abrs.index.get_level_values('date').unique())
        config_l = mouse_abrs.index.droplevel(
            ['date', 'label']).drop_duplicates().tolist()

        # Make plot
        f, axa = plt.subplots(
            len(config_l),  len(mouse_datestrings_l),
            sharex=True, figsize=(10, 10))
        f.subplots_adjust(left=.1, right=.9, top=.9, bottom=.06, wspace=0.025)
        f.suptitle(mouse + ' compared by date', fontsize=18,fontweight='bold')

        y_pos = 0.82
        spacer = ((0.9-0.06)/len(config_l))
        check = []
        for i, config in enumerate(config_l):
            f.text(0.01, y_pos, config[0] + ' ' + config[1], fontsize=15)
            y_pos = y_pos - spacer-0.001
            check.append(y_pos)

        # Label plot columns with the date(s)
        if len(mouse_datestrings_l)==1:
            axa[0].set_title(mouse_datestrings_l[0],fontsize=14)
        else:
            for date in mouse_datestrings_l:
                axa[0,mouse_datestrings_l.index(date)].set_title(date,fontsize=14)

        # Plot each one
        gobj = mouse_abrs.groupby(['date', 'channel', 'speaker_side'])
        for (date, channel, speaker_side), subdf in gobj:
            # droplevel
            subdf = subdf.droplevel(['date', 'channel', 'speaker_side'])
            subdf = subdf.sort_index(ascending=False)

            # Get ax
            if axa.ndim==1:
                ax = axa[config_l.index((channel, speaker_side))]
                axa[-1].set_xlabel('time (ms)')
            else:
                ax = axa[
                    config_l.index((channel,speaker_side)),
                    mouse_datestrings_l.index(date)
                ]
                if ax in axa[-1]:
                    ax.set_xlabel('time (ms)')
            # Plot
            # Make colorbar
            aut_colorbar = paclab.abr.abr_plotting.generate_colorbar(
                len(subdf.index), mapname='inferno_r', start=0.12, stop=1)
            color_df = pandas.DataFrame(aut_colorbar,
                                        index=subdf.index.sort_values(ascending=True))
            for label_i in subdf.index:
                ax.plot(t, subdf.loc[label_i].T * 1e6, lw=.75,
                        color=color_df.loc[label_i], label=label_i)

            # Pretty
            ax.set_xlim(-0.5, 5)
            ax.set_ylim(-4.2,4.2)
            if axa.ndim!=1 and ax not in axa[:,0]:
                ax.set_yticklabels([])

#
#             if ax in axa[:, 0]:
#                 ax.set_ylabel('ABR (uV)')
#
        # Remove any empty axes
        for ax in axa.flatten():
            if len(ax.lines) == 0:
                ax.set_visible(False)
        # Make a global legend
        handles_l = []
        for label_i in color_df.index:
            handle_i = matplotlib.lines.Line2D([], [], color=color_df.loc[label_i], lw=2, label=label_i)
            handles_l.append(handle_i)
        if len(mouse_datestrings_l)==4:
            axa[0,-1].legend(handles=handles_l, bbox_to_anchor=(1.55,-0.5), fontsize=12, labelspacing=1, title='Sound (dB)')
        elif len(mouse_datestrings_l)==2:
            axa[0, -1].legend(handles=handles_l, bbox_to_anchor=(1.27, -0.5),fontsize=12,
                          labelspacing=1, title='Sound (dB)')
        elif len(mouse_datestrings_l)==1:
            axa[-1].legend(handles=handles_l, bbox_to_anchor=(1, 1.5),fontsize=12,
                          labelspacing=1, title='Sound (dB)')
        else:
            axa[0, -1].legend(handles=handles_l, bbox_to_anchor=(1.33, -0.5),fontsize=12,
                          labelspacing=1, title='Sound (dB)')
        # Savefig
        savename = f'COMPARE_DAYS_{mouse}_PLOT_ABR'
        f.savefig(os.path.join(output_directory, savename + '.svg'))
        f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)

if ABR_POWER_VS_LEVEL_MOUSE_COMPARE_DAYS:
    ## Plot ABR power vs sound level, for every config, date, and mouse
    # Each mouse is a figure, each config is a row, each date is a column
    
    # Get time in ms
    t = big_abrs.columns / sampling_rate * 1000
    for mouse in mouse_l:
        mouse_evoked = big_abr_evoked_rms.loc[:, mouse, :].copy()
        mouse_evoked.index = mouse_evoked.index.remove_unused_levels()

        # Determine included channels and speaker sides
        channel_l = sorted(
            mouse_evoked.index.get_level_values('channel').unique())
        speaker_side_l = sorted(
            mouse_evoked.index.get_level_values('speaker_side').unique())
        mouse_datestrings_l = sorted(
            mouse_evoked.index.get_level_values('date').unique())
        config_l = sorted(mouse_evoked.index.droplevel(
            ['date', 'recording']).drop_duplicates().tolist())
        # Make plot
        f, axa = plt.subplots(
            len(channel_l)*len(speaker_side_l), # each row is one config
            len(mouse_datestrings_l),   # each column is a date
            sharex=True, sharey=True, figsize=(7, 8))
        f.subplots_adjust(left=.15, right=.95, top=.9, bottom=.1,
                          hspace=.1, wspace=.1)
        f.suptitle(mouse + ' compared by date', fontsize=18,fontweight='bold')

        y_pos = 0.82
        spacer = ((0.9-0.06)/len(config_l))
        check = []
        for i, config in enumerate(config_l):
            f.text(0.01, y_pos, config[0] + ' ' + config[1], fontsize=15)
            y_pos = y_pos - spacer-0.001
            check.append(y_pos)

        # Label plot columns with the date(s)
        if len(mouse_datestrings_l)==1:
            axa[0].set_title(mouse_datestrings_l[0],fontsize=14)
        else:
            for date in mouse_datestrings_l:
                axa[0,mouse_datestrings_l.index(date)].set_title(date,fontsize=14)

        # Plot
        gobj = mouse_evoked.groupby(['date', 'channel', 'speaker_side'])
        for (date, channel, speaker_side), subdf in gobj:
            # droplevel
            subdf = subdf.droplevel(['date', 'channel', 'speaker_side'])
            subdf = subdf.sort_index(ascending=False)
            # Get ax
            if axa.ndim==1:
                ax = axa[config_l.index((channel, speaker_side))]
                axa[-1].set_xlabel('sound level (dB)')
                axa[-1].set_xticks((10, 30, 50))
                ax.set_ylabel('ABR power (uV)')
            else:
                ax = axa[
                    config_l.index((channel,speaker_side)),
                    mouse_datestrings_l.index(date)
                ]
                if ax in axa[-1]:
                    ax.set_xlabel('sound level (dB)')
                    ax.set_xticks((10, 30, 50))
                if ax in axa[:, 0]:
                    ax.set_ylabel('ABR power (uV)')
            ax.set_ylim(0,2.5)
            # Plot each recording
            for recording in subdf.index:
                ax.plot(subdf.loc[recording] * 1e6, lw=.75)
                # Get the threshold
                thresh = threshold_db.loc[date].loc[mouse,recording,channel,speaker_side]
                if not pandas.isnull(thresh.values):
                    ax.plot([thresh], [subdf.loc[recording].loc[thresh] * 1e6], 'ko', mfc='none')


        # Remove any empty axes
        for ax in axa.flatten():
            if len(ax.lines) == 0:
                ax.set_visible(False)

        # Savefig
        savename = f'COMPARE_DAYS_{mouse}_ABR_POWER_VS_LEVEL'
        f.savefig(os.path.join(output_directory, savename + '.svg'))
        f.savefig(os.path.join(output_directory,savename + '.png'), dpi=300)
        # plt.close(f)

if IMSHOW_ABRS:
    ## This is a confusing plot and I think we should get rid of it -CR
    
    mouse_l = sorted(big_abrs.index.get_level_values('mouse').unique())[::-1]
    channel_l = sorted(big_abrs.index.get_level_values('channel').unique())
    speaker_side_l = sorted(big_abrs.index.get_level_values('speaker_side').unique())

    # Get time
    t = big_abrs.columns / sampling_rate

    # Make plot
    f, axa = plt.subplots(
        len(channel_l) * len(speaker_side_l), len(mouse_l),
        sharex=True, figsize=(16, 9))
    f.subplots_adjust(left=.02, right=.98, top=.96, bottom=.04)

    # Prep for iteration
    axa = axa.reshape((len(channel_l), len(speaker_side_l), len(mouse_l)))

    # Imshow each one
    gobj = big_abrs.groupby(['mouse', 'channel', 'speaker_side'])
    for (mouse, channel, speaker_side), subdf in gobj:
        # droplevel
        subdf = subdf.droplevel(['mouse', 'channel', 'speaker_side'])

        # Get ax
        ax = axa[
            channel_l.index(channel),
            speaker_side_l.index(speaker_side),
            mouse_l.index(mouse),
        ]

        # Plot
        my.plot.imshow(subdf, x=t, center_clim=True, origin='lower', ax=ax)

        # Title top row by mouse
        if channel_l.index(channel) == 0 and speaker_side_l.index(speaker_side) == 0:
            ax.set_title(mouse, size='x-small')

        # Title first column with config
        if mouse_l.index(mouse) == 0:
            ax.set_ylabel('{} {}'.format(channel, speaker_side))

        # Pretty
        ax.set_xlim((t[0], t[-1]))
        ax.set_xticks([])
        ax.set_yticks([])

    # Remove any empty axes
    for ax in axa.flatten():
        if len(ax.images) == 0:
            ax.set_visible(False)

    # Harmonize
    my.plot.harmonize_clim_in_subplots(fig=f, center_clim=True, trim=.999)
    savename = os.path.join(output_directory, 'IMSHOW_ABRS')
    f.savefig((savename + '.png'), dpi=300)
    f.savefig((savename + '.svg'))

if PLOT_ABR_POWER_VS_LEVEL_ALL_MICE:
    ## This plot fails to run -CR
    
    # Compare mice averaged pre v post-HL.
    # Get time in ms
    t = big_abrs.columns / sampling_rate * 1000
    timepoints_l = sorted(
        evoked_rms_timepoint.index.get_level_values('timepoint').unique())
    config_l = sorted(evoked_rms_timepoint.index.get_level_values(
        'config').drop_duplicates().tolist())
    # Make plot
    f, axa = plt.subplots(
        len(config_l),  # each row is one config
        len(timepoints_l),  # each column is a timepoint
        sharex=True, sharey=True, figsize=(10, 8))
    f.subplots_adjust(left=.11, right=.88, top=.9, bottom=.1,
                      hspace=.1, wspace=.1)
    f.suptitle('ABR power compared by timepoint', fontsize=18, fontweight='bold')

    y_pos = 0.82
    spacer = ((0.9 - 0.06) / len(config_l))
    check = []
    for i, config in enumerate(config_l):
        f.text(0.01, y_pos, config, fontsize=15)
        y_pos = y_pos - spacer - 0.001
        check.append(y_pos)

    # Label plot columns with the timepoints
    for timep in timepoints_l:
        axa[0, timepoints_l.index(timep)].set_title(timep, fontsize=14)

    # Start plotting
    gobj = evoked_rms_timepoint.groupby(['config','timepoint'])
    for (config, timep), subdf in gobj:
        # Get ax
        ax = axa[config_l.index(config), timepoints_l.index(timep)]
        if ax in axa[-1]:
            ax.set_xlabel('sound level (dB)')
            ax.set_xticks((10, 30, 50))
        if ax in axa[:, 0]:
            ax.set_ylabel('ABR power (uV)')
        # ax.set_ylim(0, 2.5)
        for mouse in sorted(subdf.index.get_level_values('mouse'),reverse=True):
            ax.plot(subdf.loc[mouse,timep,config] * 1e6, lw=.8,label=mouse)
            # Get the threshold
            # thresh = threshold_avgs_db.loc[:,mouse,config,timep]['threshold']
            # if not pandas.isnull(thresh.values):
            #     ax.plot([thresh], [subdf.loc[mouse,timep,config].loc[thresh] * 1e6], 'ko', mfc='none')

    axa.flatten()[0].legend(loc='upper right', bbox_to_anchor=(5.05, -2))
    # Savefig
    savename = 'ABR_POWER_VS_LEVEL_ALL_MICE'
    f.savefig(os.path.join(output_directory, savename + '.svg'))
    f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)

if PLOT_ABR_POWER_POSTHL_SHIFTED:
    post_evoked_rms = evoked_rms_timepoint.loc[:,['postA','postB'],:]
    config_l = sorted(post_evoked_rms.index.get_level_values(
        'config').drop_duplicates().tolist())
    timepoints_l = sorted(
        post_evoked_rms.index.get_level_values('timepoint').unique())
    # Make plot
    f, axa = plt.subplots(
        len(config_l),  # each row is one config
        2,  # each column is a timepoint
        sharex=True, sharey=True, figsize=(8, 8))
    f.subplots_adjust(left=.11, right=.88, top=.9, bottom=.1,
                      hspace=.1, wspace=.1)
    f.suptitle('ABR power post-HL, bilateral HL mice shifted left 25 dB', fontsize=18, fontweight='bold')

    y_pos = 0.82
    spacer = ((0.9 - 0.06) / len(config_l))
    check = []
    for i, config in enumerate(config_l):
        f.text(0.01, y_pos, config, fontsize=15)
        y_pos = y_pos - spacer - 0.001
        check.append(y_pos)

    # Label plot columns with the timepoints
    for timep in timepoints_l:
        axa[0, timepoints_l.index(timep)].set_title(timep, fontsize=14)

    # Start plotting
    gobj = post_evoked_rms.groupby(['config','timepoint'])
    for (config, timep), subdf in gobj:
        # Get ax
        ax = axa[config_l.index(config), timepoints_l.index(timep)]
        if ax in axa[-1]:
            ax.set_xlabel('sound level (dB)')
            ax.set_xticks((10, 30, 50))
        if ax in axa[:, 0]:
            ax.set_ylabel('ABR power (uV)')
        # ax.set_ylim(0, 2.5)
        for mouse in sorted(subdf.index.get_level_values('mouse'),reverse=True):
            if mouse in ['Cat_226','Cat_229']:
                ax.plot(subdf.loc[mouse, timep, config].index-25, subdf.loc[mouse, timep, config] * 1e6, lw=.8, label=mouse)
            else:
                ax.plot(subdf.loc[mouse,timep,config] * 1e6, lw=.8,label=mouse)
            # Get the threshold
            # thresh = threshold_avgs_db.loc[:,mouse,config,timep]['threshold']
            # if not pandas.isnull(thresh.values):
            #     ax.plot([thresh], [subdf.loc[mouse,timep,config].loc[thresh] * 1e6], 'ko', mfc='none')

    axa.flatten()[0].legend(loc='upper right', bbox_to_anchor=(2.45, -1))
    # Savefig
    savename = 'ABR_POWER_POSTHL_SHIFTED'
    f.savefig(os.path.join(output_directory, savename + '.svg'))
    f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)

if PLOT_ABR_LR_PREHL_BY_HL:
    f,axa = plt.subplots(2,2, sharex=True, sharey=True)
    f.subplots_adjust(left=.21, right=.89, top=.87, bottom=.1, wspace=0.05)
    f.suptitle('Pre-HL sham vs bilateral HL', fontsize=18, fontweight='bold')
    t = avged_abrs_timepoints.columns / sampling_rate * 1000
    # Pre-HL bilateral mice LRL
    subdf = avged_abrs_timepoints.loc[
        ['apreA','apreB'],bilateral_mouse_l,:,'LR','L'].groupby('label').mean()
    ax = plot_single_ax_abr(subdf, axa[0,0])
    subdf = avged_abrs_timepoints.loc[
        ['apreA','apreB'],bilateral_mouse_l,:,'LR','R'].groupby('label').mean()
    ax = plot_single_ax_abr(subdf, axa[1,0])
    subdf = avged_abrs_timepoints.loc[
        ['apreA','apreB'],sham_mouse_l, :, 'LR', 'L'].groupby('label').mean()
    ax = plot_single_ax_abr(subdf, axa[0,1])
    subdf = avged_abrs_timepoints.loc[
        ['apreA','apreB'],sham_mouse_l, :, 'LR', 'R'].groupby('label').mean()
    ax = plot_single_ax_abr(subdf, axa[1,1])

    for ax in axa.flatten():
        my.plot.despine(ax)
        ax.set_xlim(-0.2,6)
        ax.set_ylim(-2.5,3)
    for ax in axa[1,:]:
        ax.set_xlabel('time (ms)')
    for ax in axa[:, 0]:
        ax.set_ylabel('ABR (uV)')
    axa[1, 1].legend(loc='upper left', bbox_to_anchor=(1, 1.7))
    f.text(0.01, 0.7, 'LR L', fontsize=15)
    f.text(0.01, 0.2, 'LR R', fontsize=15)
    f.text(0.37, 0.85, 'Bilateral\nHL', fontsize=15)
    f.text(0.69, 0.85, 'Sham', fontsize=15)
    # Save figure
    savename = 'PLOT_ABR_LR_PREHL_BY_HL'
    f.savefig(os.path.join(output_directory, savename + '.svg'))
    f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)

if PLOT_ABR_LR_POSTHL_BY_HL:
    f,axa = plt.subplots(2,2, sharex=True, sharey=True)
    f.subplots_adjust(left=.21, right=.89, top=.87, bottom=.1, wspace=0.05)
    f.suptitle('Post-HL sham vs bilateral HL', fontsize=18, fontweight='bold')
    t = avged_abrs_timepoints.columns / sampling_rate * 1000
    # Pre-HL bilateral mice LRL
    subdf = avged_abrs_timepoints.loc[
        post_times_l,bilateral_mouse_l, :, 'LR','L'].groupby('label').mean()
    ax = plot_single_ax_abr(subdf, axa[0,0])
    subdf = avged_abrs_timepoints.loc[
        post_times_l, sham_mouse_l, :, 'LR', 'L'].groupby('label').mean()
    ax = plot_single_ax_abr(subdf, axa[0,1])

    subdf = avged_abrs_timepoints.loc[
        post_times_l,bilateral_mouse_l,:, 'LR','R'].groupby('label').mean()
    ax = plot_single_ax_abr(subdf, axa[1,0])
    subdf = avged_abrs_timepoints.loc[
        post_times_l, sham_mouse_l, :, 'LR', 'R'].groupby('label').mean()
    ax = plot_single_ax_abr(subdf, axa[1,1])

    for ax in axa.flatten():
        my.plot.despine(ax)
        ax.set_xlim(-0.2,6)
        ax.set_ylim(-2.5,3)
    for ax in axa[1,:]:
        ax.set_xlabel('time (ms)')
    for ax in axa[:, 0]:
        ax.set_ylabel('ABR (uV)')
    axa[1, 1].legend(loc='upper left', bbox_to_anchor=(1, 1.7))
    f.text(0.01, 0.7, 'LR L', fontsize=15)
    f.text(0.01, 0.2, 'LR R', fontsize=15)
    f.text(0.37, 0.85, 'Bilateral\nHL', fontsize=15)
    f.text(0.69, 0.85, 'Sham', fontsize=15)
    # Save figure
    savename = 'PLOT_ABR_LR_POSTHL_BY_HL'
    f.savefig(os.path.join(output_directory, savename + '.svg'))
    f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)

if PLOT_ABR_PRE_V_POST_BY_HL:
    f,axa = plt.subplots(6,4, sharex=True, sharey=True,
                         figsize=(9,9))
    f.subplots_adjust(left=.15, right=.89, top=.9, bottom=.06, wspace=0.07, hspace=0.05)
    f.suptitle('Sham vs bilateral HL', fontsize=18, fontweight='bold')
    t = abrs_avged_by_HL.columns / sampling_rate * 1000
    channel_l = abrs_avged_by_HL.index.get_level_values('channel').unique()
    speaker_side_l = abrs_avged_by_HL.index.get_level_values('speaker_side').unique()
    config_l = abrs_avged_by_HL.index.droplevel(
        ['prevpost', 'HL_group', 'label']).drop_duplicates().tolist()

    y_pos = 0.82
    spacer = ((0.9 - 0.06) / len(config_l))
    check = []
    for i, config in enumerate(config_l):
        f.text(0.01, y_pos, config[0]+config[1], fontsize=15)
        y_pos = y_pos - spacer - 0.001
        check.append(y_pos)

    gobj = abrs_avged_by_HL.groupby(['prevpost', 'HL_group', 'channel', 'speaker_side'])

    for (prevpost, HL_group, channel, speaker_side),subdf in gobj:
        subdf = subdf.droplevel(['prevpost', 'HL_group', 'channel', 'speaker_side'])
        # print(prevpost, ' ' ,HL, ' ', channel, speaker_side)
        # Get ax
        if prevpost=='pre':
            if HL_group=='sham':
                ax_ncol = 0
            if HL_group=='bilateral':
                ax_ncol =1
        if prevpost=='post':
            if HL_group == 'sham':
                ax_ncol = 2
            if HL_group == 'bilateral':
                ax_ncol = 3

        ax = axa[config_l.index((channel, speaker_side)),
            ax_ncol]
        plot_single_ax_abr(subdf, ax)
        if ax in axa[5,:]:
            ax.set_xlabel('time (ms)')
        if ax in axa[0,:]:
            ax.set_title(prevpost + '-HL ' + HL_group)
        if ax in axa[:, 0]:
            ax.set_ylabel('ABR power (uV)')
    for ax in axa.flatten():
        my.plot.despine(ax)
        ax.set_xlim(-0.2,6)
        ax.set_ylim(-3,3)
    axa[3, 3].legend(loc='upper left', bbox_to_anchor=(1, 1.5))

    # Savefig
    savename = 'PRE_V_POST_PLOT_ABR_BY_HL'
    f.savefig(os.path.join(output_directory, savename + '.svg'))
    f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)

if IMSHOW_PRE_V_POST_BY_HL:
    f,axa = plt.subplots(6,4, sharex=True, sharey=True,
                         figsize=(9,9))
    f.subplots_adjust(left=.12, right=.89, top=.9, bottom=.06, wspace=0.15, hspace=0.05)
    f.suptitle('Sham vs bilateral HL', fontsize=18, fontweight='bold')
    t = abrs_avged_by_HL.columns / sampling_rate * 1000
    channel_l = abrs_avged_by_HL.index.get_level_values('channel').unique()
    speaker_side_l = abrs_avged_by_HL.index.get_level_values('speaker_side').unique()
    config_l = abrs_avged_by_HL.index.droplevel(
        ['prevpost', 'HL_group', 'label']).drop_duplicates().tolist()

    y_pos = 0.82
    spacer = ((0.9 - 0.06) / len(config_l))
    check = []
    for i, config in enumerate(config_l):
        f.text(0.01, y_pos, config[0]+config[1], fontsize=15)
        y_pos = y_pos - spacer - 0.001
        check.append(y_pos)

    gobj = abrs_avged_by_HL.groupby(['prevpost', 'HL_group', 'channel', 'speaker_side'])

    for (prevpost, HL_group, channel, speaker_side),subdf in gobj:
        subdf = subdf.droplevel(['prevpost', 'HL_group', 'channel', 'speaker_side'])
        # print(prevpost, ' ' ,HL, ' ', channel, speaker_side)
        # Get ax
        if prevpost=='pre':
            if HL_group=='sham':
                ax_ncol = 0
            if HL_group=='bilateral':
                ax_ncol =1
        if prevpost=='post':
            if HL_group == 'sham':
                ax_ncol = 2
            if HL_group == 'bilateral':
                ax_ncol = 3

        ax = axa[config_l.index((channel, speaker_side)),
            ax_ncol]
        config = my.plot.imshow(subdf, x=t, y=subdf.index.get_level_values('label'), center_clim=True, origin='lower', ax=ax)
        if ax in axa[5,:]:
            ax.set_xlabel('time (ms)')
        if ax in axa[0,:]:
            ax.set_title(prevpost + '-HL ' + HL_group)
        if ax in axa[:, 0]:
            ax.set_ylabel('sound level (dB)')
            ax.set_yticks([50,70,90])
    for ax in axa.flatten():
        my.plot.despine(ax)
        ax.set_xlim(-0.2,6)
        # ax.set_ylim(-3,3)
    my.plot.harmonize_clim_in_subplots(fig=f, center_clim=True, clim=(-3e-6,3e-6), trim=.999)
    f.colorbar(config, ax=axa[:, -1])

    # Savefig
    savename = 'PRE_V_POST_IMSHOW_BY_HL'
    f.savefig(os.path.join(output_directory, savename + '.svg'))
    f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)

if PLOT_ABR_POWER_PRE_V_POST:
    # Compare mice averaged pre v post-HL.
    # Get time in ms
    t = big_abrs.columns / sampling_rate * 1000
    timepoints_l = sorted(
        evoked_rms_by_HL.index.get_level_values('prevpost').unique(),reverse=True)
    config_l = sorted(evoked_rms_by_HL.index.get_level_values(
        'config').drop_duplicates().tolist())
    # Make plot
    f, axa = plt.subplots(
        len(config_l),  # each row is one config
        len(timepoints_l),  # each column is a timepoint
        sharex=True, sharey=True, figsize=(6, 8))
    f.subplots_adjust(left=.15, right=.83, top=.9, bottom=.1,
                      hspace=.1, wspace=.06)
    f.suptitle('ABR power - sham v bilateral HL', fontsize=18, fontweight='bold')

    y_pos = 0.82
    spacer = ((0.9 - 0.06) / len(config_l))
    check = []
    for i, config in enumerate(config_l):
        f.text(0.01, y_pos, config, fontsize=15)
        y_pos = y_pos - spacer - 0.001
        check.append(y_pos)

    # Label plot columns with the timepoints
    for timep in timepoints_l:
        axa[0, timepoints_l.index(timep)].set_title(timep, fontsize=14)

    # Start plotting
    gobj = evoked_rms_by_HL.groupby(['config','prevpost'])
    for (config, prevpost), subdf in gobj:
        # Get ax
        ax = axa[config_l.index(config), timepoints_l.index(prevpost)]
        if ax in axa[-1]:
            ax.set_xlabel('sound level (dB)')
            ax.set_xticks((50, 70, 90))
        if ax in axa[:, 0]:
            ax.set_ylabel('ABR power (uV)')
        # ax.set_ylim(0, 2.5)
        for HL_group in sorted(subdf.index.get_level_values('HL_group').unique(),reverse=True):
            ax.plot(subdf.loc[prevpost,HL_group,config].T * 1e6, lw=.8,label=HL_group)
            # Get the threshold

    axa[3, 1].legend(loc='upper right', bbox_to_anchor=(1.53, 1))
    # Savefig
    savename = 'PRE_V_POST_PLOT_ABR'
    f.savefig(os.path.join(output_directory, savename + '.svg'))
    f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)

if PLOT_LR_IPSVCONT:
    LR_loud_avgs = LR_loud.groupby('speaker_side').mean()
    t = LR_loud_avgs.columns / sampling_rate * 1000
    f, ax = plt.subplots()
    ax.plot(t,LR_loud_avgs.loc['L']*1e6,label='Left (ipsilateral)')
    ax.plot(t,LR_loud_avgs.loc['R']*-1e6,label='Right (contralateral)')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('ABR (uV)')
    f.legend()
    f.suptitle('Ipsi vs contra for LR recordings')
    savename = 'LR_ipsvcont'
    f.savefig(os.path.join(output_directory, savename + '.svg'))
    f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)