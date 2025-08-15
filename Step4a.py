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
# Slice LR-R peaks
to_invert = big_peak_df.loc[
    (big_peak_df.index.get_level_values('speaker_side') == 'R') &
    (big_peak_df.index.get_level_values('channel') == 'LR')
    ].copy()

# Slice the other peaks
to_not_invert = big_peak_df.drop(to_invert.index).copy()

# Invert the peak heights in `to_invert`, although preserve 'typ'
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

The Cacti_223 recording looks weird. Should we drop it?
For the vertex ear ones, I think it just happens that the later peaks
are slightly bigger than the primary one. 
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
STRIP_PLOT_PEAK_HEIGHT = False
STRIP_PLOT_PEAK_LATENCY = False
OVERPLOT_LOUDEST_WITH_PEAKS = True
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

if OVERPLOT_LOUDEST_WITH_PEAKS:
    ## Plot the ABR to the loudest level, with peaks labeled
    
    # Make figure handles. Channels in columns and speaker side on rows
    channel_l = ['LV', 'RV', 'LR']
    speaker_side_l = ['L', 'R']
    f, axa = plt.subplots(
        len(channel_l), 
        len(speaker_side_l),
        sharex=True, sharey=True, figsize=(5.4, 4))
    f.subplots_adjust(
        left=.1, right=.9, top=.95, bottom=.12, hspace=0.06, wspace=0.2)

    # Plot each channel * speaker_side
    gobj = loudest.groupby(['channel', 'speaker_side'])
    for (channel, speaker_side), subdf in gobj:
        
        # Get ax
        ax = axa[
            channel_l.index(channel),
            speaker_side_l.index(speaker_side),
            ]

        # Plot the ABR
        ax.plot(t, subdf.T * 1e6, lw=.7, alpha=.4, color='k')

        # Get the corresponding peaks
        this_peaks = big_peak_df.xs(
            channel, level='channel').xs(
            speaker_side, level='speaker_side')
        
        # Plot the peaks
        peak_kwargs = {
            'marker': '.',
            'ls': 'none',
            'ms': 4,
            'alpha': .5,
            }
        if channel == 'LR' and speaker_side == 'R':
            # The peaks are flipped for this one!
            # Plot the positive peaks
            ax.plot(
                this_peaks[this_peaks['typ'] == 'pos']['t'],
                -this_peaks[this_peaks['typ'] == 'pos']['val'],
                color='r', **peak_kwargs)
            
            # Plot the negative peaks
            ax.plot(
                this_peaks[this_peaks['typ'] == 'neg']['t'],
                -this_peaks[this_peaks['typ'] == 'neg']['val'],
                color='b', **peak_kwargs)
        
        else:
            # Plot the positive peaks
            ax.plot(
                this_peaks[this_peaks['typ'] == 'pos']['t'],
                this_peaks[this_peaks['typ'] == 'pos']['val'],
                color='r', **peak_kwargs)
            
            # Plot the negative peaks
            ax.plot(
                this_peaks[this_peaks['typ'] == 'neg']['t'],
                this_peaks[this_peaks['typ'] == 'neg']['val'],
                color='b', **peak_kwargs)
        
        # Despine 
        if ax in axa[-1]:
            my.plot.despine(ax, which=('left', 'right', 'top'))
        else:
            my.plot.despine(ax, which=('left', 'right', 'top', 'bottom'))

    # Pretty
    ax.set_xlim((-1, 7))
    ax.set_ylim((-5, 5))
    ax.set_xticks([0, 3, 6])
    ax.set_yticks([])
    f.text(.51, .01, 'time (ms)', ha='center', va='bottom')
    
    # Scale bar
    axa[0, -1].plot([6, 6], [3, 4], 'k-', lw=.75)
    axa[0, -1].text(6.2, 3.5, '1 uV', ha='left', va='center', size=12)
    
    # Label the channel
    for n_channel, channel in enumerate(channel_l):
        axa[n_channel, 0].set_ylabel(channel, labelpad=20)
    
    # Label the speaker side
    axa[0, 0].set_title('sound from left')
    axa[0, 1].set_title('sound from right')
    
    # Savefig
    savename = 'OVERPLOT_LOUDEST_WITH_PEAKS'
    f.savefig(os.path.join(output_directory, savename + '.svg'))
    f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)

