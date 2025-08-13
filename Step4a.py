## Peak picking plots

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
from paclab.abr import abr_plotting, abr_analysis
import my.plot
import matplotlib.pyplot as plt
import seaborn


## Plots
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
    

## Params
sampling_rate = 16000  # TODO: store in recording_metadata
loudest_dB = 91


## Aggregate over recordings for each ABR
# TODO: do this upstream
averaged_abrs = big_abrs.groupby(
    [lev for lev in big_abrs.index.names if lev != 'recording']).mean()

# Join after_HL on avged_abrs
averaged_abrs = my.misc.join_level_onto_index(
    averaged_abrs, 
    experiment_metadata.set_index(['mouse', 'date'])['after_HL'], 
    join_on=['mouse', 'date']
    )

# Keep only after_HL == False
averaged_abrs = averaged_abrs.loc[False]


## Pick peaks
# Consistent t, used throughout
t = big_abrs.columns / sampling_rate * 1000

# Pick peaks for loudest sound only
loudest = averaged_abrs.xs(loudest_dB, level='label')

# Find peaks for each
peak_params_l = []
peak_params_keys_l = []
for idx in loudest.index:
    # Slice
    topl = loudest.loc[idx] * 1e6

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


## Invert the sign for the LR-R recordings, so that primary peak is always neg
to_invert = big_peak_df.loc[
    (big_peak_df.index.get_level_values('speaker_side') == 'R') &
    (big_peak_df.index.get_level_values('channel') == 'LR')
    ].copy()
to_not_invert = big_peak_df.drop(to_invert.index).copy()

# Invert, although preserve 'typ'
to_invert['prom'] *= -1
to_invert['val'] *= -1

# Re-concat
big_peak_df = pandas.concat([to_invert, to_not_invert]).sort_index()


## Find the primary peak
# Drop any peak in LR after 2.4 ms, or in the other channels after 2.0 ms
# Without dropping, here are the violations
"""
                                           idx      prom  typ       val       t
date       mouse     channel speaker_side                          
2025-06-06 Cacti_223 LR      R              81 -2.068464  pos -1.423620  2.5625
2025-04-30 Pearl_189 RV      R              74 -2.621651  neg -2.048181  2.1250
2025-05-02 Cacti_1   RV      R              75 -4.300674  neg -2.523503  2.1875
2025-04-30 Pearl_189 LV      R              92 -2.444802  neg -1.490880  3.2500
"""
drop_mask1 = (
    (big_peak_df.index.get_level_values('channel') == 'LR') & 
    (big_peak_df['t'] > 2.4)
    )
drop_mask2 = (
    (big_peak_df.index.get_level_values('channel') != 'LR') & 
    (big_peak_df['t'] > 2)
    )

big_peak_df_filtered = big_peak_df.loc[~drop_mask1 & ~drop_mask2]

# Choose the primary peak
# `first()` means we choose the most negative, which is more consistent
# 'val' tends to be more consistent than 'prom'
primary_peak = big_peak_df_filtered.sort_values('val').groupby(
    [lev for lev in big_peak_df.index.names if lev != 'n_pk']
    ).first()


## Plots
STRIP_PLOT_PEAK_HEIGHT = True
STRIP_PLOT_PEAK_LATENCY = True
PEAK_HT_BY_LAT = False
PEAK_LATENCY_BY_LAT = False
PLOT_LOUDEST_PKS = False
PLOT_ABR_RMS_OVER_TIME = False


if STRIP_PLOT_PEAK_HEIGHT:
    ## Plot distribution of primary peak heights
    # TODO: plot as connected pairs
    # TODO: stats
    # Create figure handles
    f, ax = my.plot.figure_1x1_standard()
    
    # Swarmplot
    swarm = seaborn.stripplot(
        primary_peak, 
        x='channel', 
        y='val', 
        hue='speaker_side', 
        marker="$\circ$",
        alpha=0.5,
        hue_order=['L', 'R'], 
        ax=ax, 
        dodge=True, 
        legend=False,
        palette={'L': 'b', 'R': 'r'},
        )

    # Pretty
    ax.set_ylim((0, -6))
    ax.set_yticks((0, -3, -6))
    ax.set_ylabel('height of primary peak (uV)')
    ax.set_xlabel('channel')
    my.plot.despine(ax)

    # Savefig
    savename = 'STRIP_PLOT_PEAK_HEIGHT'
    f.savefig(os.path.join(output_directory, savename + '.svg'))
    f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)

if STRIP_PLOT_PEAK_LATENCY:
    ## Plot distribution of primary peak latencies by channel and speaker_side
    # Create figure handles
    f, ax = my.plot.figure_1x1_standard()
    
    # Swarmplot
    # https://stackoverflow.com/questions/66404883/seaborn-scatterplot-set-hollow-markers-instead-of-filled-markers
    swarm = seaborn.stripplot(
        primary_peak, 
        x='channel', 
        y='t', 
        hue='speaker_side', 
        marker="$\circ$",
        alpha=0.5,
        hue_order=['L', 'R'], 
        ax=ax, 
        dodge=True, 
        legend=False,
        palette={'L': 'b', 'R': 'r'},
        )
    
    # Pretty
    ax.set_ylim((0, 3))
    ax.set_yticks((0, 1, 2, 3))
    ax.set_ylabel('latency to primary peak (ms)')
    ax.set_xlabel('channel')
    my.plot.despine(ax)

    # Savefig
    savename = 'STRIP_PLOT_PEAK_LATENCY'
    f.savefig(os.path.join(output_directory, savename + '.svg'))
    f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)

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


## TODO: move this to a threshold script
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
