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
matplotlib.use('TkAgg')
import pandas
import paclab.abr
from paclab.abr import abr_plotting, abr_analysis
import my.plot
import matplotlib.pyplot as plt
import seaborn

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



## Params
sampling_rate = 16000  # TODO: store in recording_metadata

## Cohort Analysis' Information
datestring = '250630'
day_directory = "_cohort"
loudest_dB = 91

# Tenatative because I'm blinded, but come on it's obvious
sham_mouse_l = ['Cat_227', 'Cat_228']
bilateral_mouse_l = ['Cat_226', 'Cat_229']

## Paths
GUIdata_directory, Pickle_directory = (paclab.abr.loading.get_ABR_data_paths())
cohort_name = datestring + day_directory
# Use cohort pickle directory
cohort_pickle_directory = os.path.join(Pickle_directory, cohort_name)
if not os.path.exists(cohort_pickle_directory):
    try:
        os.mkdir(cohort_pickle_directory)
    except:
        print("No pickle directory exists and this script doesn't have permission to create one.")
        print("Check your Pickle_directory file path.")

## Load results of Step1
cohort_experiments = pandas.read_pickle(
    os.path.join(cohort_pickle_directory, 'cohort_experiments'))
recording_metadata = pandas.read_pickle(
    os.path.join(cohort_pickle_directory, 'recording_metadata'))
# Fillna
cohort_experiments['HL'] = cohort_experiments['HL'].fillna('none')

# Drop those with 'include' == False
recording_metadata = recording_metadata[recording_metadata['include'] == True]

## Load results of Step2
big_triggered_neural = pandas.read_pickle(
    os.path.join(cohort_pickle_directory, 'big_triggered_neural'))
big_abrs = pandas.read_pickle(
    os.path.join(cohort_pickle_directory, 'big_abrs'))
threshold_db = pandas.read_pickle(
    os.path.join(cohort_pickle_directory, 'thresholds'))


# Add timepoint to big_abrs
big_abrs = big_abrs.join(cohort_experiments.set_index(['date','mouse'])['timepoint'],
              on=(['date','mouse']))
big_abrs['laterality'] = [
    paclab.abr.abr_analysis.laterality_check(i_channel, i_speaker) for i_channel, i_speaker in zip(
        big_abrs.index.get_level_values('channel'),
        big_abrs.index.get_level_values('speaker_side'))
]
big_abrs['laterality'] = big_abrs['laterality'].fillna('LR')
big_abrs = big_abrs.reset_index()
big_abrs['config'] = big_abrs['channel']+big_abrs['speaker_side']
big_abrs = big_abrs.set_index(['date','mouse','timepoint','recording',
        'config', 'laterality', 'channel','speaker_side','label'])


averaged_abrs = big_abrs.groupby(
    [lev for lev in big_abrs.index.names if lev != 'recording']
    ).mean()

## Consistent t, used throughout
t = big_abrs.columns / sampling_rate * 1000
loudest = averaged_abrs.xs(loudest_dB, level='label')

# Find peaks for each
peak_params_l = []
peak_params_keys_l = []
for idx in loudest.index:
    # Slice
    topl = loudest.loc[idx] * 1e6
    # topl = topl.loc[idx]

    # See notes in identify_click_times about how find_peaks works
    fp_kwargs = {
        'distance': 8,  # refractory period, should be about 0.5 ms
        'prominence': 0.2,  # peak prominence
        'wlen': 20,  # samples to seek for nearest valley in prominence
    }

    # First positive and then negative
    # The refractory keeps us from finding multiple peaks per cycle, and a low
    # prominence threshold helps avoid false negatives, but it's still possible
    # to miss some and pos and neg won't alternate
    pos_peak_samples, pos_peak_props = scipy.signal.find_peaks(
        topl, **fp_kwargs)
    neg_peak_samples, neg_peak_props = scipy.signal.find_peaks(
        -topl, **fp_kwargs)

    # DataFrame
    # Invert negative peak params prom because of the negative above
    pos_peak_params = pandas.Series(pos_peak_samples, name='idx').to_frame()
    pos_peak_params['prom'] = pos_peak_props['prominences']
    pos_peak_params['typ'] = 'pos'
    neg_peak_params = pandas.Series(neg_peak_samples, name='idx').to_frame()
    neg_peak_params['prom'] = -neg_peak_props['prominences']
    neg_peak_params['typ'] = 'neg'

    # Concat
    peak_params = pandas.concat([pos_peak_params, neg_peak_params], ignore_index=True)

    # Add more stuff
    peak_params['val'] = topl.values[peak_params['idx'].values]
    peak_params['t'] = t[peak_params['idx'].values]

    # Sort by t
    peak_params = peak_params.sort_values('t').reset_index(drop=True)
    # Name the index, which is the peak sequence of the peaks per topl
    peak_params.index = peak_params.index.set_names('n_pk')
    # Store
    peak_params_l.append(peak_params)
    peak_params_keys_l.append(idx)

# Concat
big_peak_df = pandas.concat(
    peak_params_l, keys=peak_params_keys_l, names=loudest.index.names,
).sort_index()

## Parameterize the primary peak
# Invert the sign for the LR-R recordings
to_invert = big_peak_df.loc[
    (big_peak_df.index.get_level_values('speaker_side') == 'R') &
    (big_peak_df.index.get_level_values('channel') == 'LR')
    ].copy()
to_not_invert = big_peak_df.drop(to_invert.index).copy()

# Invert, although preserve 'typ'
to_invert['prom'] *= -1
to_invert['val'] *= -1

# Re-concat
big_peak_df_rect = pandas.concat([to_invert, to_not_invert]).sort_index()

big_rect_pks_config = big_peak_df_rect.reset_index().drop(columns='date')

## Find the most negative peak
# The most positive peak is more variable
# Using prominence is more variable than value
most_negative = big_rect_pks_config.sort_values('val').groupby(
    ['timepoint', 'mouse', 'config', 'laterality']).first()
most_negative = most_negative.sort_values('speaker_side')

# most_negative picks a few peaks that, based on their latency, are clearly wrong.
# Get rid of these.
error_pks = [most_negative.loc[most_negative['t']>2.4].loc[most_negative['channel']=='LR']]
error_pks.append(most_negative.loc[most_negative['t']>2].loc[most_negative['channel']!='LR'])
error_pks = pandas.concat(error_pks)
# Go ahead and get ABR waveform of these to plot them tho
error_pks_ABRs = error_pks.reset_index().set_index(
    ['mouse', 'timepoint', 'config', 'laterality', 'channel', 'speaker_side'])
error_pks_ABRs = error_pks_ABRs.join(
    loudest.droplevel('date',axis=0),on=error_pks_ABRs.index.names,how='left')
most_negative = most_negative.loc[~most_negative.index.isin(error_pks.index)]

# Chris' names
PLOT_DELAY_VS_LEVEL = False
PLOT_DIVERSITY_LOUDEST = False
PLOT_ABR_POWER_VS_LEVEL_ALL_MICE = False



mouse_l = big_abrs.index.get_level_values('mouse').unique()
font_size = 14

# My plot names
PLOT_ERROR_PKS = False

PEAK_HT_BY_CONFIG = False
PEAK_LATENCY_BY_CONFIG = False
PEAK_HT_BY_LAT = False
PEAK_LATENCY_BY_LAT = False
PLOT_LOUDEST_PKS = False
PLOT_RAWTRIALS_ALIGNED = False

PLOT_ABR_RMS_OVER_TIME = False
PLOT_PSD = True

if PLOT_ERROR_PKS:
    f, axa = my.plot.auto_subplot(len(error_pks))
    f.subplots_adjust(hspace=0.3)
    gobj = error_pks_ABRs.groupby(['mouse', 'timepoint', 'config'])
    i = 0
    for (mouse, timepoint, config), subdf in gobj:
        ax = axa.flatten()[i]
        subdf = subdf
        ax.plot((1e6 * subdf.iloc[:, 6:]).T)
        pk_x = int(subdf['idx'].values[0]) - 40
        if config == 'LRR':
            pk_y = -subdf['val']
        else:
            pk_y = subdf['val']
        ax.plot(pk_x, pk_y, marker='.')
        ax.set_title(mouse+' '+timepoint+' '+config)
        i += 1
    savename = 'error_peaks_plotted'
    f.savefig(os.path.join(cohort_pickle_directory, savename + '.svg'))
    f.savefig(os.path.join(cohort_pickle_directory, savename + '.png'), dpi=300)

if PEAK_HT_BY_CONFIG:
    # Plot distribution of primary peak heights

    # Histogram the time at which this occurs
    f, ax = plt.subplots(figsize=(5.2, 3))
    f.subplots_adjust(left=.16, bottom=.17, right=.71,top=0.98)
    f.set_gid('Pk_ht_by_config')
    box = seaborn.boxplot(most_negative, x='channel', y='val', hue='speaker_side', hue_order=['L','R'],
                          fill=False, showfliers=False, legend=False, ax=ax)
    swarm = seaborn.swarmplot(most_negative, x='channel', y='val', hue='speaker_side', hue_order=['L','R'],
                              ax=ax, dodge=True, legend=False)
    handles_l = [
        matplotlib.lines.Line2D([], [], color=seaborn.color_palette()[0],
                linewidth=2, label='Left speaker'),
        matplotlib.lines.Line2D([], [], color=seaborn.color_palette()[1],
                linewidth=2, label='Right speaker')
    ]
    fig_leg = f.legend(handles=handles_l, loc='upper left', bbox_to_anchor=(0.68, 0.6),
                       handlelength=1, handletextpad=0.5, fontsize=font_size, frameon=False)
    fig_leg.set_gid('legend')
    # Pretty
    ax.tick_params(labelsize=font_size)
    ax.set_ylabel('primary peak\nheight (uV)', fontsize=font_size)
    # ax.set_xlabel('')
    ax.set_xlabel('channel', fontsize=font_size)
    my.plot.despine(ax)
    # Savefig
    savename = 'PEAK_HT_BY_CONFIG'
    f.savefig(os.path.join(cohort_pickle_directory, savename + '.svg'))
    f.savefig(os.path.join(cohort_pickle_directory, savename + '.png'), dpi=300)
if PEAK_LATENCY_BY_CONFIG:
    # Plot distribution of primary peak heights

    # Histogram the time at which this occurs
    f, ax = plt.subplots(figsize=(5.2, 3))
    f.subplots_adjust(left=.16, bottom=.17, right=.71,top=0.98)
    f.set_gid('Pk_latency_by_config')
    box = seaborn.boxplot(most_negative, x='channel', y='t', hue='speaker_side', hue_order=['L','R'],
                          fill=False, showfliers=False, legend=False, ax=ax)
    swarm = seaborn.swarmplot(most_negative, x='channel', y='t', hue='speaker_side', hue_order=['L','R'],
                              ax=ax, dodge=True, legend=False)
    handles_l = [
        matplotlib.lines.Line2D([], [], color=seaborn.color_palette()[0],
                linewidth=2, label='Left speaker'),
        matplotlib.lines.Line2D([], [], color=seaborn.color_palette()[1],
                linewidth=2, label='Right speaker')
    ]
    fig_leg = f.legend(handles=handles_l, loc='upper left', bbox_to_anchor=(0.68, 0.6),
                       handlelength=1, handletextpad=0.5, fontsize=font_size, frameon=False)
    fig_leg.set_gid('legend')
    # Pretty
    ax.tick_params(labelsize=font_size)
    ax.set_ylabel('primary peak\nlatency (ms)', fontsize=font_size)
    # ax.set_xlabel('')
    ax.set_xlabel('channel', fontsize=font_size)
    my.plot.despine(ax)
    # Savefig
    savename = 'PEAK_LATENCY_BY_CONFIG'
    f.savefig(os.path.join(cohort_pickle_directory, savename + '.svg'))
    f.savefig(os.path.join(cohort_pickle_directory, savename + '.png'), dpi=300)

if PLOT_LOUDEST_PKS:
    ## Plot the loudest sounds only, to show diversity
    # Note:
    # Set t
    t = big_abrs.columns / sampling_rate * 1000

    # One plot per channel
    channel_l = ['LR','LV', 'RV']

    # Slice mice
    this_loudest = loudest.loc[:,mouse_l,:]

    # Make plot
    f, axa = plt.subplots(
        len(channel_l), 2, sharex=True, sharey=True, figsize=(4.5, 5))
    f.subplots_adjust(left=.12, right=.98, bottom=.1, top=.97, hspace=.13)
    this_loudest = this_loudest.droplevel(['date','laterality','timepoint','config'],axis=0)

    gobj = this_loudest.groupby(['channel','speaker_side'])
    for (channel, speaker_side),subdf in gobj:
        n_row = channel_l.index(channel)
        if speaker_side=='L':
            n_col = 0
        if speaker_side=='R':
            n_col = 1
            # if channel=='LR':
                # subdf = -subdf
        ax = axa[n_row,n_col]
        ax.plot(t, (1e6*subdf.T), lw=.7, alpha=.4, color='k')

        this_peaks = big_peak_df.xs(
            (channel, speaker_side), level=('channel', 'speaker_side'), axis='index')
        ax.plot(
            this_peaks[this_peaks['typ'] == 'pos']['t'],
            this_peaks[this_peaks['typ'] == 'pos']['val'],
            mec='r', mfc='none', marker='o', ls='none', ms=4)
        ax.plot(
            this_peaks[this_peaks['typ'] == 'neg']['t'],
            this_peaks[this_peaks['typ'] == 'neg']['val'],
            mec='b', mfc='none', marker='o', ls='none', ms=4)

        # Title by channel
        ax.set_title(channel+'_'+speaker_side,fontsize=font_size,y=0.9)

        # Pretty
        my.plot.despine(ax)
        ax.tick_params(labelsize=font_size)
        ax.set_xlim((-0.5, 6))
        ax.set_xticks((0, 3, 6))

    # Label
    for ax in axa[2,:]:
        ax.set_xlabel('time (ms)', fontsize=font_size, labelpad=-1)
    axa[1,0].set_ylabel('ABR (uV)', fontsize=font_size,labelpad=-8)
    # Savefig
    savename = 'PLOT_LOUDEST_PKS'
    f.savefig(os.path.join(cohort_pickle_directory, savename + '.svg'))
    f.savefig(os.path.join(cohort_pickle_directory, savename + '.png'), dpi=300)

if PLOT_RAWTRIALS_ALIGNED:
    # Set t
    t = big_abrs.columns / sampling_rate * 1000
    raw_aligned = big_triggered_neural.copy()
    raw_aligned.index = raw_aligned.index.reorder_levels(['channel', 'date', 'mouse', 'recording', 'label', 'polarity', 't_samples'])
    # Only take LV and RV so you don't have any stupid LRR inverting to worry about
    raw_aligned = raw_aligned.loc[['LV','RV']].xs(loudest_dB,axis=0,level='label')
    f,ax = plt.subplots(figsize=(5,3))
    f.subplots_adjust(top=0.99, bottom=0.16, right=0.98)
    # Take 300 random trials to plot
    ax.plot(t,(raw_aligned.sample(300)*1e6).T, alpha=0.4)
    ax.set_xlim(-1,6)
    ax.set_ylim(-10,10)
    ax.set_xlabel('time (ms)', fontsize=font_size, labelpad=-1)
    ax.set_ylabel('ABR (uV)', fontsize=font_size,labelpad=-8)
    my.plot.despine(ax)
    ax.tick_params(labelsize=font_size)

    savename = 'PLOT_RAWTRIALS_ALIGNED'
    f.savefig(os.path.join(cohort_pickle_directory, savename + '.svg'))
    f.savefig(os.path.join(cohort_pickle_directory, savename + '.png'), dpi=300)

if PLOT_ABR_RMS_OVER_TIME:
    # Plot the smoothed rms of the ABR over time by condition

    t = big_abrs.columns / sampling_rate * 1000
    big_abr_stds = big_abrs.T.rolling(window=20, center=True, min_periods=1).std().T
    # Drop redundante date level and aggregate over recordings
    # TODO: consider whether to aggregate over recordings before or after RMS
    to_agg = big_abr_stds.droplevel(['date','timepoint','config']).groupby(
        ['mouse', 'speaker_side', 'channel', 'label']).mean()
    # Oops at some point I stopped calling the x timepoints 'timepoint' and
    # started using 'timepoint' to mean experimental timepoint
    # Now it has come bacto_agg.columns = to_agg.columns.rename('timepoint')k to haunt me and I am having regrets
    # I will have to rename many things later to change it to exp_timepoint or something like that
    to_agg.columns = to_agg.columns.rename('timepoint')
    # Make mouse the replicates on the columns
    to_agg = to_agg.stack().unstack('mouse')

    # Agg

    # TODO: log10 before agg?
    agg_mean = to_agg.mean(axis=1).unstack('timepoint')
    agg_err = to_agg.sem(axis=1).unstack('timepoint')

    # Plot
    channel_l = ['LR','LV', 'RV']
    speaker_side_l = ['L', 'R']

    # Color by level
    sound_levels = list(big_abrs.index.get_level_values('label').unique())
    sound_levels.sort(reverse=True)
    colors = my.plot.generate_colorbar(
        len(sound_levels),
        mapname='inferno_r', start=0.15, stop=1)

    # Make plot
    f, axa = plt.subplots(
        len(channel_l), len(speaker_side_l),
        sharex=True, sharey=True, figsize=(5, 4))
    f.subplots_adjust(left=.17, right=.89, top=.95, bottom=.15, hspace=.15, wspace=.12)

    # Imshow each one
    gobj = agg_mean.groupby(['channel', 'speaker_side'])
    for (channel, speaker_side), subdf in gobj:
        # droplevel
        subdf = subdf.droplevel(['channel', 'speaker_side']).sort_index(ascending=False)

        # Get ax
        ax = axa[
            channel_l.index(channel),
            speaker_side_l.index(speaker_side),
        ]

        # Plot each
        for n_level, level in enumerate(subdf.index[::2]):
            # Get color
            color = colors[sound_levels.index(level)]

            # Plot in uV
            ax.plot(t, subdf.loc[level] * 1e6, color=color, lw=1)  # , clip_on=False)

            # Add legend
            if ax == axa[1, 1]:
                ax.text(
                    1, 1 - 0.17 * n_level, f'{level} dB',
                    color=color, ha='left', va='center', clip_on=False,
                    transform=ax.transAxes, fontsize=font_size)

        # Pretty
        ax.set_xlim((-3, 8))
        ax.set_yscale('log')
        ax.set_ylim((0.04, 2.5))
        ax.set_xticks((-2, 0, 2, 4, 6))

        # Nicer log labels
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

        # Label top ax only
        if ax in axa[0]:
            if speaker_side == 'L':
                ax.set_title(f'sound from left', pad=0, fontsize=font_size)
            else:
                ax.set_title(f'sound from right', pad=0, fontsize=font_size)
                # Despine
        my.plot.despine(ax)  # , which=('left', 'top', 'right'))#, 'bottom'))
        ax.tick_params(labelsize=font_size)
        # Start time
        # ~ ax.plot([0, 0], [-2, 2], 'k-', lw=.75)

        # Make rest of legend visible
        ax.patch.set_visible(False)

    # Label config
    axa[0, 0].set_ylabel('LR', rotation=0, labelpad=15, fontsize=font_size)
    axa[1, 0].set_ylabel('LV', rotation=0, labelpad=15, fontsize=font_size)
    axa[2, 0].set_ylabel('RV', rotation=0, labelpad=15, fontsize=font_size)

    # Shared x-axis
    f.text(.5, .02, 'time from sound onset (ms)', ha='center', va='bottom',fontsize=font_size)

    # Savefig
    f.savefig(os.path.join(cohort_pickle_directory,
                           'PLOT_ABR_RMS_OVER_TIME.svg'))
    f.savefig(os.path.join(cohort_pickle_directory,
                           'PLOT_ABR_RMS_OVER_TIME.png'), dpi=300)


if PLOT_PSD:
    print("hey idk what I'm doing yet lol")