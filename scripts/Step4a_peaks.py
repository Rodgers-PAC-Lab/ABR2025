## Peak picking plots
"""
Makes the following plots:

STRIP_PLOT_PEAK_HEIGHT
STRIP_PLOT_PEAK_LATENCY
OVERPLOT_LOUDEST_WITH_PEAKS
"""

import os
import json
import matplotlib
import scipy.signal
import numpy as np
import pandas
import my.plot
import matplotlib.pyplot as plt
import seaborn


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
averaged_abrs_by_mouse = pandas.read_pickle(
    os.path.join(output_directory, 'averaged_abrs_by_mouse'))
averaged_abrs_by_date = pandas.read_pickle(
    os.path.join(output_directory, 'averaged_abrs_by_date'))
trial_counts = pandas.read_pickle(
    os.path.join(output_directory, 'trial_counts'))

# Loudest dB
loudest_db = big_abrs.index.get_level_values('label').max()
    

## Keep only after_HL == False
big_abrs = big_abrs.xs(False, level='after_HL').droplevel('HL_type')
averaged_abrs_by_mouse = averaged_abrs_by_mouse.xs(False, level='after_HL').droplevel('HL_type')
averaged_abrs_by_date = averaged_abrs_by_date.xs(False, level='after_HL').droplevel('HL_type')


## Use only the loudest sound
# Pick peaks for loudest sound only
loudest = averaged_abrs_by_mouse.xs(loudest_db, level='label')


## Pick peaks
# Consistent t, used throughout
t = big_abrs.columns / sampling_rate * 1000

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
    peak_params['height'] = topl.values[peak_params['idx'].values]
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
to_invert['height'] *= -1

# Re-concat
big_peak_df = pandas.concat([to_invert, to_not_invert]).sort_index()


## Find the primary peak
"""Find violations before dropping them

big_peak_df_filtered = big_peak_df
primary_peak = big_peak_df_filtered.sort_values('height').groupby(
    [lev for lev in big_peak_df.index.names if lev != 'n_pk']
    ).first()
    

## Vertex-ear recordings

print(primary_peak.drop('LR', level='channel').sort_values('t'))

# I checked all these
# These are almost all ones where there's (unusually) a later peak that is 
# larger than the primary peak. That's okay, we'll just specify that we chose
# the primary peak as the largest negative one in the window < 1.9 ms.
# 
# The exception is PizzaSlice2 on 02-28, which has a strangely tiny primary peak,
# which is also bimodal for RV-R. That one looks messed up to me. The late
# peak will be dropped here, leaving only the tiny primary peak (which ends up
# being an outlier on the strip plot).
# TODO: look into whether that whole experiment should be dropped

2025-03-29 YellowStingray7 LV      L              72 -1.890973  neg -1.213548  2.0000
                           RV      R              72 -1.314301  neg -1.242551  2.0000
2025-04-30 Pearl_189       RV      R              74 -2.621651  neg -2.048181  2.1250
2025-02-28 PizzaSlice7     RV      R              74 -3.003374  neg -2.010931  2.1250
2025-05-02 Cacti_1         RV      R              75 -4.300674  neg -2.523503  2.1875
2025-02-28 PizzaSlice2     LV      L              76 -1.186733  neg -0.981259  2.2500
                           RV      R              76 -1.203288  neg -1.139183  2.2500
2025-02-12 Pineapple_197   RV      L              82 -5.428365  neg -3.349833  2.6250
2025-04-30 Pearl_189       LV      R              92 -2.444802  neg -1.490880  3.2500


## LR recordings

print(primary_peak.xs('LR', level='channel').sort_values('t'))

# The Cacti_223 R recording on 6-6 looks a bit off: the peak is small and
# followed too quickly by another peak. However, I think it is in the range
# of normal variation. It was repeated 4 times. Dropping the late peak enables
# us to pick up the correct peak.
#
# The PowerRainbow2 L recording on 2-19 has a bona fide delayed primary peak,
# so it will not be dropped. It may be in range of normal variation. It ends
# up as the LR-R outlier in the latency strip plot.

2025-02-19 PowerRainbow2   L              77 -2.107131  neg -1.795992  2.3125
2025-06-06 Cacti_223       R              81 -2.068464  pos -1.423620  2.5625
"""

# Drop any peak in LR after 2.6 ms, or in the other channels after 1.9 ms
drop_mask1 = (
    (big_peak_df.index.get_level_values('channel') == 'LR') & 
    (big_peak_df['t'] > 2.6)
    )
drop_mask2 = (
    (big_peak_df.index.get_level_values('channel') != 'LR') & 
    (big_peak_df['t'] > 1.9)
    )
big_peak_df_filtered = big_peak_df.loc[~drop_mask1 & ~drop_mask2]

# Choose the primary peak
# `first()` means we choose the most negative, which is more consistent
# 'height' tends to be more consistent than 'prom'
primary_peak = big_peak_df_filtered.sort_values('height').groupby(
    [lev for lev in big_peak_df.index.names if lev != 'n_pk']
    ).first()

# Reindex to see if we dropped any mouse (e.g., no peak found)
new_index = big_peak_df.groupby('mouse').size().index
primary_peak = my.misc.slice_df_by_some_levels(primary_peak, new_index)
assert not primary_peak.isnull().any().any()

# Make sure we found 6 peaks per experiment (2 sides * 3 channels)
assert len(primary_peak) == len(new_index) * 6


## Plots
STRIP_PLOT_PEAK_HEIGHT = True
STRIP_PLOT_PEAK_LATENCY = True
OVERPLOT_LOUDEST_WITH_PEAKS = True


if STRIP_PLOT_PEAK_HEIGHT:
    ## Plot distribution of primary peak heights
    
    # Create figure handles
    f, ax = my.plot.figure_1x1_standard()
    
    # Choose height or prominence
    metric = 'height'
    
    # Swarmplot
    swarm = seaborn.stripplot(
        primary_peak, 
        x='channel', 
        y=metric,
        hue='speaker_side', 
        marker="$\circ$",
        alpha=0.5,
        order=['LV', 'RV', 'LR'],
        hue_order=['L', 'R'], 
        ax=ax, 
        dodge=True, 
        legend=False,
        palette={'L': 'b', 'R': 'r'},
        )

    # Connect the pairs
    for n_channel, channel in enumerate(['LV', 'RV', 'LR']):
        for mouse in primary_peak.index.levels[0]:
            lval = primary_peak[metric].loc[mouse].loc[channel].loc['L']
            rval = primary_peak[metric].loc[mouse].loc[channel].loc['R']
            xval = [
                n_channel - .2,
                n_channel + .2,
                ]
            ax.plot(xval, [lval, rval], '-', color='gray', alpha=.5, lw=.75)

    # Check nothing is outside the axis limits
    assert (primary_peak[metric] > -8).all()
    assert (primary_peak[metric] < 0).all()

    # Pretty
    ax.set_ylim((0, -8))
    ax.set_yticks((0, -4, -8))
    ax.set_ylabel(f'{metric} of\nprimary peak ({MU}V)')
    ax.set_xlabel('channel')
    my.plot.despine(ax)

    
    ## Run stats
    # `metric ~ 'channel' + 'speaker_side'` isn't so informative, because
    # the primary variable is 'ipsi', which isn't well-defined for LR.
    # `metric ~ 'channel' + 'ipsi'`, restricted to vertex-ear only,
    # reveals a significant ipsi preference
    # This is barely significant for L sounds, due to the overall 
    # bias toward R sounds.
    
    # Count mice
    n_mice = len(primary_peak.index.get_level_values('mouse').unique())
    
    # Run anova
    to_anova = primary_peak.reset_index()
    to_anova['ipsi'] = to_anova['channel'].str[0] == to_anova['speaker_side']
    aov_res = my.stats.anova(
        to_anova[to_anova['channel'] != 'LR'], 
        f'{metric} ~ channel + ipsi',
        )
    
    # Get post-hoc t-test for each
    pvalue_l = []
    channel_l = ['LV', 'RV', 'LR']
    for channel in channel_l:
        # Slice channel
        to_compare = primary_peak[metric].xs(
            channel, level='channel').unstack('speaker_side')
        
        # paired t-test
        ttr = scipy.stats.ttest_rel(to_compare['L'], to_compare['R'])
        
        # Store
        pvalue_l.append(ttr.pvalue)
    pvalue_ser = pandas.Series(pvalue_l, index=channel_l)    
    
    
    ## Plot stats
    for n_channel, channel in enumerate(['LV', 'RV', 'LR']):
        # Plot a line
        xval = [
            n_channel - .2,
            n_channel + .2,
            ]
        ax.plot(xval, [-7, -7], '-', color='k', lw=.75)    
        
        # Add asterisks
        sigstr = my.stats.pvalue_to_significance_string(pvalue_ser.loc[channel])
        yval = -7.25 if sigstr == 'n.s.' else -7
        ax.text(n_channel, yval, sigstr, ha='center', va='bottom')
    
    # Aggregate
    mu = primary_peak[metric].groupby(['channel', 'speaker_side']).mean().unstack()
    err = primary_peak[metric].groupby(['channel', 'speaker_side']).sem().unstack()
    
    
    ## Write out stats
    stats_filename = 'figures/STATS__STRIP_PLOT_PEAK_HEIGHT'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write(f'n = {n_mice} mice\n')
        fi.write('using only the first experiment from each mouse\n')
        fi.write(f'paired t-test p-value on L vs R for {metric}:\n')
        fi.write(str(pvalue_ser) + '\n')
        fi.write(f'mean:\n{mu}\n')
        fi.write(f'SEM:\n{err}\n')    
    
    # Echo
    with open(stats_filename) as fi:
        print(''.join(fi.readlines()))    


    ## Savefig
    savename = 'figures/STRIP_PLOT_PEAK_HEIGHT'
    f.savefig(savename + '.svg')
    f.savefig(savename + '.png', dpi=300)

if STRIP_PLOT_PEAK_LATENCY:
    ## Plot distribution of primary peak latencies by channel and speaker_side
    
    # Run everything on 't'
    metric = 't' 
    
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
        order=['LV', 'RV', 'LR'],
        hue_order=['L', 'R'], 
        ax=ax, 
        dodge=True, 
        legend=False,
        palette={'L': 'b', 'R': 'r'},
        )

    # Connect the pairs
    for n_channel, channel in enumerate(['LV', 'RV', 'LR']):
        for mouse in primary_peak.index.levels[0]:
            lval = primary_peak[metric].loc[mouse].loc[channel].loc['L']
            rval = primary_peak[metric].loc[mouse].loc[channel].loc['R']
            xval = [
                n_channel - .2,
                n_channel + .2,
                ]
            ax.plot(xval, [lval, rval], '-', color='gray', alpha=.5, lw=.75)
            
    # Pretty
    ax.set_ylim((0, 3))
    ax.set_yticks((0, 1, 2, 3))
    ax.set_ylabel('latency to\nprimary peak (ms)')
    ax.set_xlabel('channel')
    my.plot.despine(ax)


    ## Run stats
    # Count mice
    n_mice = len(primary_peak.index.get_level_values('mouse').unique())
    
    # Run anova
    to_anova = primary_peak.reset_index()
    to_anova['ipsi'] = to_anova['channel'].str[0] == to_anova['speaker_side']
    aov_res = my.stats.anova(
        to_anova[to_anova['channel'] != 'LR'], 
        f'{metric} ~ channel + ipsi',
        )
    
    # Get post-hoc t-test for each
    pvalue_l = []
    channel_l = ['LV', 'RV', 'LR']
    for channel in channel_l:
        # Slice channel
        to_compare = primary_peak[metric].xs(
            channel, level='channel').unstack('speaker_side')
        
        # paired t-test
        ttr = scipy.stats.ttest_rel(to_compare['L'], to_compare['R'])
        
        # Store
        pvalue_l.append(ttr.pvalue)
    pvalue_ser = pandas.Series(pvalue_l, index=channel_l)    
    
    
    ## Plot stats
    for n_channel, channel in enumerate(['LV', 'RV', 'LR']):
        # Plot a line
        xval = [
            n_channel - .2,
            n_channel + .2,
            ]
        ax.plot(xval, [2.5, 2.5], '-', color='k', lw=.75)    
        
        # Add asterisks
        sigstr = my.stats.pvalue_to_significance_string(pvalue_ser.loc[channel])
        yval = 2.6 if sigstr == 'n.s.' else 2.5
        ax.text(n_channel, yval, sigstr, ha='center', va='bottom')

    
    ## Write out stats
    # Aggregate by channel * speaker_side over mouse
    mu = primary_peak[metric].groupby(['channel', 'speaker_side']).mean().unstack()
    err = primary_peak[metric].groupby(['channel', 'speaker_side']).sem().unstack()
    
    # Simple mean by ipsi and contra
    mu_ipsi = (mu.loc[('LV', 'L')] + mu.loc[('RV', 'R')]) / 2
    mu_contra = (mu.loc[('LV', 'R')] + mu.loc[('RV', 'L')]) / 2
    
    # Aggregate vertex-ear by mouse over channel * speaker_side
    by_mouse = primary_peak[metric].drop(
        'LR', level='channel').groupby('mouse').mean()
    mu2 = by_mouse.mean()
    err2 = by_mouse.sem()
    
    

    # Write out stats
    stats_filename = 'figures/STATS__STRIP_PLOT_PEAK_LATENCY'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write(f'n = {n_mice} mice\n')
        fi.write('using only the first experiment from each mouse\n')
        fi.write(f'paired t-test p-value on L vs R for {metric}:\n')
        fi.write(str(pvalue_ser) + '\n')
        fi.write(f'mean:\n{mu}\n')
        fi.write(f'SEM:\n{err}\n')
        fi.write(f'ipsi: {mu_ipsi:.4f}; contra: {mu_contra:.4f}\n')
        fi.write(f'vertex-ear latency across '
            f'{len(by_mouse)} mice: {mu2:.4f} +/- {err2:.5f} (SEM)')
    
    # Echo
    with open(stats_filename) as fi:
        print('\n' + ''.join(fi.readlines()))    


    ## Savefig
    savename = 'figures/STRIP_PLOT_PEAK_LATENCY'
    f.savefig(savename + '.svg')
    f.savefig(savename + '.png', dpi=300)

if OVERPLOT_LOUDEST_WITH_PEAKS:
    ## Plot the ABR to the loudest level, with peaks labeled
    
    # Make figure handles. Channels in columns and speaker side on rows
    channel_l = ['LV', 'RV']#, 'LR']
    speaker_side_l = ['L', 'R']
    f, axa = plt.subplots(
        len(channel_l), 
        len(speaker_side_l),
        sharex=True, sharey=True, figsize=(8, 3))
    f.subplots_adjust(
        left=.07, right=.99, top=.92, bottom=.15, hspace=0, wspace=0.2)

    # Plot each channel * speaker_side
    gobj = loudest.groupby(['channel', 'speaker_side'])
    for (channel, speaker_side), subdf in gobj:
        
        # Get ax
        try:
            ax = axa[
                channel_l.index(channel),
                speaker_side_l.index(speaker_side),
                ]
        except ValueError:
            continue

        # Plot the ABR
        ax.plot(t, subdf.T * 1e6, lw=.75, alpha=.4, color='k')

        # Get the corresponding peaks
        this_peaks = primary_peak.xs(
            channel, level='channel').xs(
            speaker_side, level='speaker_side')
        
        # Plot the peaks
        peak_kwargs = {
            'marker': 'o',
            #~ 'mfc': 'none',
            'ls': 'none',
            'ms': 4,
            'mew': 0,
            'alpha': .5,
            }
        pos_color = 'g'
        neg_color = 'magenta'
        
        if channel == 'LR' and speaker_side == 'R':
            # The peaks are flipped for this one!
            # Plot the positive peaks
            ax.plot(
                this_peaks[this_peaks['typ'] == 'pos']['t'],
                -this_peaks[this_peaks['typ'] == 'pos']['height'],
                color=pos_color, **peak_kwargs)
            
            # Plot the negative peaks
            ax.plot(
                this_peaks[this_peaks['typ'] == 'neg']['t'],
                -this_peaks[this_peaks['typ'] == 'neg']['height'],
                color=neg_color, **peak_kwargs)
        
        else:
            # Plot the positive peaks
            ax.plot(
                this_peaks[this_peaks['typ'] == 'pos']['t'],
                this_peaks[this_peaks['typ'] == 'pos']['height'],
                color=pos_color, **peak_kwargs)
            
            # Plot the negative peaks
            ax.plot(
                this_peaks[this_peaks['typ'] == 'neg']['t'],
                this_peaks[this_peaks['typ'] == 'neg']['height'],
                color=neg_color, **peak_kwargs)
        
        # Despine 
        if ax in axa[-1]:
            my.plot.despine(ax, which=('left', 'right', 'top'))
        else:
            my.plot.despine(ax, which=('left', 'right', 'top', 'bottom'))

        # Label the primary peak zone
        ax.fill_betweenx(y=(-6, 6), x1=1.1, x2=1.7, color='gray', alpha=.25, lw=0)

    # Legend the primary peak pink point
    axa[0, 0].plot([-2], [4], color='magenta', clip_on=False, **peak_kwargs)
    axa[0, 0].text(-1.75, 4, 'primary peak', ha='left', va='center', size=12)

    # Pretty
    ax.set_xlim((-1, 7))
    ax.set_ylim((-6, 6))
    ax.set_xticks([0, 3, 6])
    ax.set_yticks([])
    f.text(.51, .01, 'time from sound onset (ms)', ha='center', va='bottom')
    
    # Scale bar
    axa[0, -1].plot([6, 6], [3, 5], 'k-', lw=.75, clip_on=False)
    axa[0, -1].text(6.2, 4, '2 uV', ha='left', va='center', size=12)
    
    # Label the channel
    for n_channel, channel in enumerate(channel_l):
        axa[n_channel, 0].set_ylabel(channel, labelpad=20)
    
    # Label the speaker side
    axa[0, 0].set_title('sound from left')
    axa[0, 1].set_title('sound from right')
    
    
    ## Savefig
    savename = 'figures/OVERPLOT_LOUDEST_WITH_PEAKS'
    f.savefig(savename + '.svg')
    f.savefig(savename + '.png', dpi=300)


    ## Write out stats
    n_mice = len(loudest.index.get_level_values('mouse').unique())
    
    # Write out stats
    stats_filename = 'figures/STATS__OVERPLOT_LOUDEST_WITH_PEAKS\n'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write(f'n = {n_mice} mice\n')
        fi.write('using only the first experiment from each mouse\n')
    
    # Echo
    with open(stats_filename) as fi:
        print(''.join(fi.readlines()))    
    
plt.show()