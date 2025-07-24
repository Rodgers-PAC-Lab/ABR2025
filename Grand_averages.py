## Grand average cohort plots for a 'normal' cohort where there's not any
# dramatic differences to compare like HL or FAD status or whatever

import os
import datetime
import glob
import json
import matplotlib
import scipy.signal
import numpy as np
import matplotlib

#~ matplotlib.use('TkAgg')
import pandas
import paclab.abr
from paclab.abr import abr_plotting, abr_analysis
import my.plot
import matplotlib.pyplot as plt

plt.ion()
my.plot.font_embed()


def pre_or_post(timepoint):
    if type(timepoint) == float:
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


def plot_single_ax_abr(abr_subdf, ax,t=np.arange(160)/16000000):
    """
    PARAMETERS:
        abr_subdf: a subdf where the index is sound levels and the columns are voltages
        ax: the axis to plot it on
        t: the x axis in ms
    RETURNS:
        ax: the axis object with the plot made
    """

    for label_i in abr_subdf.index.sort_values(ascending=False):
        aut_colorbar = paclab.abr.abr_plotting.generate_colorbar(
            len(abr_subdf.index), mapname='inferno_r', start=0.15, stop=1)
        color_df = pandas.DataFrame(aut_colorbar,
                                    index=abr_subdf.index.sort_values(ascending=True))
        ax.plot(t, abr_subdf.loc[label_i].T * 1e6, lw=.75,
                color=color_df.loc[label_i], label=label_i)
    return ax


## Params
sampling_rate = 16000  # TODO: store in recording_metadata

## Cohort Analysis' Information
datestring = '250630'
day_directory = "_cohort"
loudest_dB = 91

# Tenatative because I'm blinded, but come on it's obvious
sham_mouse_l = ['Cat_227', 'Cat_228']
bilateral_mouse_l = ['Cat_226', 'Cat_229']
pre_times_l = ['apreA', 'apreB']
post_times_l = ['postA', 'postB']


## Paths
# Load the required file filepaths.json (see README)
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

# Get demographic info for 'methods' section purposes
demographic_l = []
by_mouse = cohort_experiments.set_index(['mouse']).sort_index()
for mouse in by_mouse.index.unique():
    subdf = by_mouse.loc[mouse]
    if subdf.ndim==1:
        n_testdays = 1
    else:
        n_testdays = subdf['date'].count()
    max_age = subdf['age'].max()
    min_age = subdf['age'].min()
    demo_dct = {'mouse': mouse,
                'sex': subdf['sex'][0],
                'n_testdays': n_testdays,
                'min_age': min_age,
                'max_age' : max_age
                }
    demographic_l.append(pandas.Series(demo_dct))
demographic_df = pandas.DataFrame(demographic_l)


# Drop those with 'include' == False
recording_metadata = recording_metadata[recording_metadata['include'] == True]

## Load results of Step2
big_triggered_neural = pandas.read_pickle(
    os.path.join(output_directory, 'big_triggered_neural'))
big_abrs = pandas.read_pickle(
    os.path.join(output_directory, 'big_abrs'))
threshold_db = pandas.read_pickle(
    os.path.join(output_directory, 'thresholds'))

# Average ABRs across recordings
avged_abrs = big_abrs.groupby(['date', 'mouse', 'channel', 'speaker_side', 'label']).mean()
avged_abrs_lat = avged_abrs.copy()
# avged_abrs_lat = avged_abrs.copy().drop('LR',axis='index',level='channel')
avged_abrs_lat['laterality'] = [
    paclab.abr.abr_analysis.laterality_check(i_channel, i_speaker) for i_channel, i_speaker in zip(
        avged_abrs_lat.index.get_level_values('channel'),
        avged_abrs_lat.index.get_level_values('speaker_side'))
]
avged_abrs_lat['laterality'] = avged_abrs_lat['laterality'].fillna('LR')
avged_abrs_lat = avged_abrs_lat.set_index('laterality',append=True)
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
    cohort_experiments.set_index(['mouse', 'date'])[['timepoint', 'HL']], on=('mouse', 'date'), how='left')
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
big_abr_evoked_peak = pandas.DataFrame(big_abr_evoked_peak).join(
    cohort_experiments.set_index(['date', 'mouse'])[['timepoint', 'HL']], on=(['date', 'mouse']))
big_abr_evoked_peak = big_abr_evoked_peak.rename(columns={0: 'pk_V'})
big_abr_evoked_peak = big_abr_evoked_peak.reset_index()
big_abr_evoked_peak['prevpost'] = big_abr_evoked_peak['timepoint'].apply(lambda x: pre_or_post(x))

threshold_avgs_db = threshold_db.groupby(['date', 'mouse', 'speaker_side', 'channel']).mean()
threshold_avgs_db['config'] = (threshold_avgs_db._get_label_or_level_values('channel', axis=0) +
                               threshold_avgs_db._get_label_or_level_values('speaker_side', axis=0))

threshold_avgs_db = threshold_avgs_db.reset_index().set_index(['date', 'mouse', 'config'])
threshold_avgs_db = threshold_avgs_db.join(cohort_experiments.set_index(['mouse', 'date'])[['timepoint', 'HL']],
                                           on=['mouse', 'date'])
threshold_avgs_db = threshold_avgs_db.reset_index().set_index(['date', 'mouse', 'config', 'timepoint'])

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
    start = 31  # cannot be less than this
    stop = 42  # inclusive. This one is insensitive - no better later peaks
    peak_samples, peak_props = scipy.signal.find_peaks(
        LR_loud.loc[idx, start:stop].abs(),
        height=1e-7,
        distance=(stop - start + 1),
    )

    if len(peak_samples) != 1:
        1 / 0

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
    for i_day in big_abrs.loc[:, mouse, :].index.get_level_values('date').unique():
        # Get the correlations
        r_LR_all = np.corrcoef(
            avged_abrs.loc[i_day, mouse, 'LR', 'L'].values.flatten(),
            avged_abrs.loc[i_day, mouse, 'LR', 'R'].values.flatten(),
        )[0, 1]

        # Get the corresponding amplitude and latency of the primary peaks (for the loudest sound)
        i_L_amp = primary_peak_LR.loc[i_day, mouse, :, 'L']['amplitude'].mean()
        i_R_amp = primary_peak_LR.loc[i_day, mouse, :, 'R']['amplitude'].mean()
        i_L_lat = primary_peak_LR.loc[i_day, mouse, :, 'L']['latency'].mean()
        i_R_lat = primary_peak_LR.loc[i_day, mouse, :, 'R']['latency'].mean()
        rec_l.append([mouse, i_day, r_LR_all, i_L_amp, i_R_amp, i_L_lat, i_R_lat])

        # Make a separate frame for the standard deviations
        i_L_std = big_abr_stds.loc[i_day, mouse, :, 'LR', 'L'].groupby(
            ['label']).mean()
        i_R_std = big_abr_stds.loc[i_day, mouse, :, 'LR', 'R'].groupby(
            ['label']).mean()
        rec_stds_l.append([mouse, i_day, i_L_std, i_R_std])

between_ears_diagnostics = pandas.DataFrame(rec_l, columns=[
    'mouse', 'date', 'corr', 'L_amp', 'R_amp', 'L_lat', 'R_lat'])
between_ears_diagnostics = between_ears_diagnostics.set_index(['date', 'mouse'])
between_ears_diagnostics['L_amp'] *= 1e6
between_ears_diagnostics['R_amp'] *= 1e6

between_ears_diagnostics = between_ears_diagnostics.join(
    cohort_experiments[['HL', 'strain', 'genotype', 'mouse', 'date']].set_index(['date', 'mouse']))
between_ears_diagnostics = between_ears_diagnostics.reset_index().set_index(
    ['HL', 'strain', 'genotype', 'mouse', 'date'])
# Typically corr should be -.8 to -.3, median -.45
# As of 6-4-25, median is -0.7. Huh.

between_ears_stds = pandas.DataFrame(rec_stds_l, columns=['mouse', 'date', 'L_std', 'R_std'])
between_ears_stds = between_ears_stds.set_index(['date', 'mouse'])

mouse_l = big_abrs.index.get_level_values('mouse').unique()

GRAND_AVG_ABR_PLOT = True
GRAND_AVG_IMSHOW = False
GRAND_AVG_IPSVCONT = False
ALL_CHANNELS_IPSVCONT = False
LR_vs_LVminusRV = False
GRAND_AVG_RIGHT_SPKR = False

if GRAND_AVG_ABR_PLOT:
    f, axa = plt.subplots(3, 2, sharex=True, sharey=True)
    f.set_gid('grand_avg_abr_plt')
    f.subplots_adjust(left=.16, right=.85, top=.87, bottom=.1,
                      hspace=0.06, wspace=0.05)
    f.suptitle('Average ABR', fontsize=18, fontweight='bold')
    t = avged_abrs.columns / sampling_rate * 1000
    ax_rows = {'LR':0, 'LV':1, 'RV':2}
    ax_cols = {'L':0, 'R':1}
    gobj = avged_abrs.groupby(['channel','speaker_side'])
    for (channel, speaker_side),subdf in gobj:
        # Mean across sound level
        subdf = subdf.groupby('label').mean()
        config_id = str(channel) + (speaker_side)
        # Get ax
        ax = axa[ax_rows[channel], ax_cols[speaker_side]]
        config_id = str(channel) + (speaker_side)
        ax.set_gid(config_id)
        ax_plt = plot_single_ax_abr(subdf, ax, t)

        if channel!='RV':
            my.plot.despine(ax, which=('top','bottom'))

    for ax in axa.flatten():
        ax.tick_params(labelsize=12)
        my.plot.despine(ax)
        ax.set_xlim(-0.2, 6)
        ax.set_ylim(-3, 3)
    for ax in axa[2, :]:
        ax.set_xlabel('time (ms)', fontsize=12)
    for ax in axa[:, 0]:
        ax.set_ylabel('ABR (uV)', fontsize=12)

    # If you want to make the lines thicker in the legend,
    # unfortunately you need custom legend handlers.
    handles_l = []
    for label_i in subdf.index.sort_values(ascending=False):
        aut_colorbar = paclab.abr.abr_plotting.generate_colorbar(
            len(subdf.index), mapname='inferno_r', start=0.15, stop=1)
        color_df = pandas.DataFrame(aut_colorbar,
                                    index=subdf.index.sort_values(ascending=True))
        thick_line = matplotlib.lines.Line2D([], [], color=color_df.loc[label_i], linewidth=5, label=label_i)
        handles_l.append(thick_line)
    # fig_leg = axa[1, 1].legend(handles=handles_l, loc='upper left', bbox_to_anchor=(1, 1.7), fontsize=12)
    fig_leg = f.legend(handles=handles_l, loc='upper left', bbox_to_anchor=(0.84, 0.8), fontsize=12)
    fig_leg.set_gid('legend')

    f.text(0.01, 0.72, 'LR', fontsize=15)
    f.text(0.01, 0.46, 'LV', fontsize=15)
    f.text(0.01, 0.19, 'RV', fontsize=15)
    f.text(0.25, 0.85, 'Left speaker', fontsize=15)
    f.text(0.6, 0.85, 'Right speaker', fontsize=15)
    # Save figure
    savename = 'GRAND_AVG_ABR_PLOT'
    f.savefig(os.path.join(output_directory, savename + '.svg'))
    f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)

if GRAND_AVG_IMSHOW:
    f, axa = plt.subplots(3, 2, sharex=True, sharey=True)
    f.set_gid('grand_avg_imshow')
    f.subplots_adjust(left=.16, right=.99, top=.87, bottom=.1,
                      hspace=0.06, wspace=0.05)
    f.suptitle('Average ABR', fontsize=18, fontweight='bold')
    t = avged_abrs.columns / sampling_rate * 1000
    ax_rows = {'LR':0, 'LV':1, 'RV':2}
    ax_cols = {'L':0, 'R':1}
    gobj = avged_abrs.groupby(['channel','speaker_side'])
    for (channel, speaker_side),subdf in gobj:
        # Mean across sound level
        subdf = subdf.groupby('label').mean()
        # Get ax
        ax = axa[ax_rows[channel], ax_cols[speaker_side]]
        config_id = str(channel) + (speaker_side)
        ax.set_gid(config_id)
        config = my.plot.imshow(subdf, x=t,y=subdf.index.get_level_values('label'), center_clim=True, origin='lower', ax=ax)
        if channel!='RV':
            my.plot.despine(ax, which=('top','bottom'))

    my.plot.harmonize_clim_in_subplots(fig=f, center_clim=True, clim=(-3e-6, 3e-6), trim=.999)
    f_cb = f.colorbar(config,ax=axa[:])
    f_cb.ax.set_gid('colorbar')
    for ax in axa.flatten():
        ax.tick_params(labelsize=12)
        my.plot.despine(ax)
        ax.set_xlim(-0.2, 6)
    for ax in axa[2, :]:
        ax.set_xlabel('time (ms)', fontsize=12)
    for ax in axa[:, 0]:
        ax.set_ylabel('sound level (dB)', fontsize=12)
        ax.set_yticks([50, 70, 90])

    f.text(0.01, 0.72, 'LR', fontsize=15)
    f.text(0.01, 0.46, 'LV', fontsize=15)
    f.text(0.01, 0.19, 'RV', fontsize=15)
    f.text(0.25, 0.88, 'Left speaker', fontsize=15)
    f.text(0.6, 0.88, 'Right speaker', fontsize=15)
    # Save figure
    savename = 'GRAND_AVG_IMSHOW'
    f.savefig(os.path.join(output_directory, savename + '.svg'))
    f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)

if GRAND_AVG_IPSVCONT:
    # Get only the loudest sounds
    t = avged_abrs_lat.columns / sampling_rate * 1000
    loud_laterality = avged_abrs_lat.xs(loudest_dB,axis='index',level='label')
    loud_laterality = loud_laterality.groupby(['channel','speaker_side','laterality']).mean()

    f, ax=plt.subplots(figsize=(6,2.5))
    f.subplots_adjust(right=0.7, bottom=0.18)
    # f.subplots_adjust(bottom=0.18)
    f.set_gid('grand_ipsvcont')
    colors_dict = {'contralateral':'magenta', 'ipsilateral':'green'}
    gobj = loud_laterality.loc[['LV','RV'],:,:].groupby(['laterality','channel','speaker_side'])
    for (laterality, channel, speaker_side), subdf in gobj:
        if speaker_side=='R':
            ax.plot(t, subdf.T*1e6, color=colors_dict[laterality], linestyle='--',
                    label=str(channel) + ', Right speaker ('+str(laterality.removesuffix('lateral'))+')')
        if speaker_side=='L':
            ax.plot(t, subdf.T*1e6, color=colors_dict[laterality],
                    label=str(channel) + ', Left speaker('+str(laterality.removesuffix('lateral'))+')')
    fig_leg = f.legend(loc='upper left', bbox_to_anchor=(0.64, 0.5), frameon=False)
    # fig_leg = f.legend()
    fig_leg.set_gid('legend')
    ax.tick_params(labelsize=12)
    my.plot.despine(ax)
    ax.set_xlim(0, 5)
    ax.set_ylim(-3.75, 3)
    ax.set_ylabel('ABR (uV)', fontsize=12)
    ax.set_xlabel('time (ms)', fontsize=12)
    # Save figure
    savename = 'GRAND_AVG_IPSVCONT'
    f.savefig(os.path.join(output_directory, savename + '.svg'))
    f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)

if GRAND_AVG_RIGHT_SPKR:
    # Get only the loudest sounds
    t = avged_abrs_lat.columns / sampling_rate * 1000
    loud_laterality = avged_abrs_lat.xs(loudest_dB,axis='index',level='label')
    loud_laterality = loud_laterality.groupby(['channel','speaker_side','laterality']).mean()

    f, ax=plt.subplots(figsize=(6,2.5))
    f.subplots_adjust(right=0.7, bottom=0.18)
    # f.subplots_adjust(bottom=0.18)
    f.set_gid('grand_rightspkr')
    colors_dict = {'contralateral':'magenta', 'ipsilateral':'green','LR':'black'}
    gobj = loud_laterality.loc[:,'R',:].groupby(['laterality','channel'])
    for (laterality, channel), subdf in gobj:
        if laterality=='LR':
            ax.plot(t, subdf.T*1e6, color=colors_dict[laterality], alpha=0.85,
                    label=str(channel) + ', Right speaker')
        else:
            ax.plot(t, subdf.T*1e6, color=colors_dict[laterality], alpha=0.8,
                    label=str(channel) + ', Right speaker ('+str(laterality.removesuffix('lateral'))+')')
    fig_leg = f.legend(loc='upper left', bbox_to_anchor=(0.64, 0.5), frameon=False)
    # fig_leg = f.legend()
    fig_leg.set_gid('legend')
    ax.tick_params(labelsize=12)
    my.plot.despine(ax)
    ax.set_xlim(0, 5)
    ax.set_ylim(-3.75, 3)
    ax.set_ylabel('ABR (uV)', fontsize=12)
    ax.set_xlabel('time (ms)', fontsize=12)
    # Save figure
    savename = 'GRAND_AVG_RIGHT_SPKR'
    f.savefig(os.path.join(output_directory, savename + '.svg'))
    f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)

if ALL_CHANNELS_IPSVCONT:
    # Get only the loudest sounds
    t = avged_abrs_lat.columns / sampling_rate * 1000
    loud_laterality = avged_abrs_lat.xs(loudest_dB,axis='index',level='label')
    loud_laterality = loud_laterality.groupby(['channel','speaker_side','laterality']).mean()

    f, ax=plt.subplots(figsize=(6,2.5))
    f.subplots_adjust(right=0.7, bottom=0.18)
    # f.subplots_adjust(bottom=0.18)
    f.set_gid('grand_ipsvcont')
    colors_dict = {'contralateral':'magenta', 'ipsilateral':'green', 'LR': 'black'}
    gobj = loud_laterality.groupby(['laterality','channel','speaker_side'])
    for (laterality, channel, speaker_side), subdf in gobj:
        if laterality=='LR':
            if speaker_side=='R':
                ax.plot(t, subdf.T*-1e6, color=colors_dict[laterality], linestyle='--',
                    label='LR , Right speaker')
            if speaker_side=='L':
                ax.plot(t, subdf.T*1e6, color=colors_dict[laterality],
                        label='LR , Left speaker')
        else:
            if speaker_side=='R':
                ax.plot(t, subdf.T*1e6, color=colors_dict[laterality], linestyle='--',
                    label=str(channel) + ', Right speaker ('+str(laterality.removesuffix('lateral'))+')')
            if speaker_side=='L':
                ax.plot(t, subdf.T*1e6, color=colors_dict[laterality],
                        label=str(channel) + ', Left speaker('+str(laterality.removesuffix('lateral'))+')')
    fig_leg = f.legend(loc='upper left', bbox_to_anchor=(0.64, 0.5), frameon=False)
    # fig_leg = f.legend()
    fig_leg.set_gid('legend')
    ax.tick_params(labelsize=12)
    my.plot.despine(ax)
    ax.set_xlim(0, 5)
    ax.set_ylim(-3.75, 3)
    ax.set_ylabel('ABR (uV)', fontsize=12)
    ax.set_xlabel('time (ms)', fontsize=12)
    # Save figure
    savename = 'ALL_CHANNELS_IPSVCONT'
    f.savefig(os.path.join(output_directory, savename + '.svg'))
    f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)
if LR_vs_LVminusRV:
    LVsRV_Right = (loud_laterality.loc['LV','R','contralateral']-
                   loud_laterality.loc['RV','R','ipsilateral'])
    LVsRV_Left = (loud_laterality.loc['LV','L','ipsilateral']-
                   loud_laterality.loc['RV','L','contralateral'])
    fig,ax = plt.subplots()
    ax.plot(t, LVsRV_Right*-1e6, color='#06249c', label='LV-RV Right')
    ax.plot(t, LVsRV_Left*1e6, color='#b5250e', label='LV-RV Left')
    ax.plot(t, loud_laterality.loc['LR','L',:].T*1e6, ls ='--', color='red',label='LR Left')
    ax.plot(t, loud_laterality.loc['LR', 'R', :].T*-1e6, ls ='--', color='blue', label='LR Right')
    fig.legend()
    savename = 'LR_vs_LVminusRV'
    f.savefig(os.path.join(output_directory, savename + '.svg'))
    f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)
