## Remove outliers and aggregates ABRs over trials
# Writes out:
#   big_abrs - averaged ABRs
#   trial_counts - trial counts
#
# Plots:
#   PLOT_SINGLE_TRIAL_ABR
#   PLOT_TRIAL_AVERAGED_ABR
#   PLOT_POSITIVE_AND_NEGATIVE_CLICKS

import os
import datetime
import glob
import json
import scipy.signal
import numpy as np
import pandas
from paclab import abr
import my.plot
import matplotlib.pyplot as plt
import tqdm


## Paths
# Load the required file filepaths.json (see README)
with open('filepaths.json') as fi:
    paths = json.load(fi)

# Parse into paths to raw data and output directory
raw_data_directory = paths['raw_data_directory']
output_directory = paths['output_directory']


## Params
# Outlier params
abs_max_sigma = 3
stdev_sigma = 3


## Load previous results
# Load results of Step1
recording_metadata = pandas.read_pickle(
    os.path.join(output_directory, 'recording_metadata'))

# Load results of Step2
big_triggered_ad = pandas.read_pickle(
    os.path.join(output_directory, 'big_triggered_ad'))
big_triggered_neural = pandas.read_pickle(
    os.path.join(output_directory, 'big_triggered_neural'))
big_click_params = pandas.read_pickle(
    os.path.join(output_directory, 'big_click_params'))


## Join on speaker_side
# Should have done this at the same time as joining channel
idx = big_triggered_neural.index.to_frame().reset_index(drop=True)
idx = idx.join(
    recording_metadata['speaker_side'], on=['date', 'mouse', 'recording'])
big_triggered_neural.index = pandas.MultiIndex.from_frame(idx)

# Reorder levels
big_triggered_neural = big_triggered_neural.reorder_levels([
    'date', 'mouse', 'recording', 'channel', 'speaker_side', 'label', 
    'polarity', 't_samples']).sort_index()


## Drop outlier trials, separately by channel
# Consider outliers separately for every channel on every recording
group_by = ['date', 'mouse', 'recording', 'channel', 'speaker_side']
gobj = big_triggered_neural.groupby(group_by)

# We use this helper function to drop the groupby levels, otherwise
# the levels get duplicated by gobj.apply
def drop_outliers(df):
    res = abr.signal_processing.trim_outliers(
        df.droplevel(group_by),
        abs_max_sigma=abs_max_sigma,
        stdev_sigma=stdev_sigma,
        )
    
    return res

# Apply the drop
big_triggered_neural2 = gobj.apply(drop_outliers)

# Reorder levels to be like triggered_neural
big_triggered_neural2 = big_triggered_neural2.reorder_levels(
    big_triggered_neural.index.names).sort_index()


## Count the number of trials remaining in each recording
trial_counts = big_triggered_neural2.groupby(
    ['date', 'mouse', 'speaker_side', 'recording', 'label', 'channel']).size()


## Aggregate
# Average out the trial
by_polarity = big_triggered_neural2.groupby(
    [lev for lev in big_triggered_neural2.index.names if lev != 't_samples']
    ).mean()

# Compute the big_abrs by adding over polarity
big_abrs = 0.5 * (
    by_polarity.xs(True, level='polarity') + 
    by_polarity.xs(False, level='polarity')
    )

# Compute the big_arts by subtracting over polarity
big_arts = 0.5 * (
    by_polarity.xs(True, level='polarity') - 
    by_polarity.xs(False, level='polarity')
    )


## Plots
PLOT_SINGLE_TRIAL_ABR = True
PLOT_TRIAL_AVERAGED_ABR = True
PLOT_POSITIVE_AND_NEGATIVE_CLICKS = True

if PLOT_SINGLE_TRIAL_ABR:
    # Cat_229 rec 10 on 2025-05-20 is about top 25th percentile of response
    # strength. So it's stronger than median, but not unusually strong.
    # This is an RV recording with speaker_side L
    single_trial_abr = big_triggered_neural2.loc[
        datetime.date(2025, 5, 20)].loc['Cat_229'].loc[10].xs(
        'RV', level='channel').droplevel('speaker_side')

    # Slice out loudest only
    single_trial_abr = single_trial_abr.xs(91, level='label')

    f, ax = plt.subplots(figsize=(4, 2.5))
    f.subplots_adjust(bottom=.24, left=.15, right=.95, top=.89)
    ax.plot(
        single_trial_abr.columns.values / 16e3 * 1000,
        single_trial_abr.T * 1e6, 
        color='k', alpha=.05, lw=1)
    ax.set_xlabel('time from sound (ms)')
    ax.set_ylabel('ABR (uV)')
    ax.set_yticks((-8, 0, 8))
    ax.set_ylim((-8, 8))
    ax.set_xlim((-2.5, 7.5))
    ax.set_xticks((-2, 0, 2, 4, 6, ))
    my.plot.despine(ax)

    # Plot the stimulus
    ax.plot([0, 0.1], [5.5 / 6 * 8] * 2, 'k-', lw=3)
    
    f.savefig('PLOT_SINGLE_TRIAL_ABR.svg')
    f.savefig('PLOT_SINGLE_TRIAL_ABR.png', dpi=300)

if PLOT_TRIAL_AVERAGED_ABR:
    # Plot the corresponding trial averaged ABR
    trial_averaged_abr = big_triggered_neural2.loc[
        datetime.date(2025, 5, 20)].loc['Cat_229'].loc[10].xs(
        'RV', level='channel').droplevel('speaker_side')

    # Aggregate
    trial_averaged_abr = trial_averaged_abr.groupby('label').mean()
    
    # Plot
    f, ax = plt.subplots(figsize=(4, 2.5))
    f.subplots_adjust(bottom=.24, left=.15, right=.95, top=.89)
    ax.plot(
        trial_averaged_abr.columns.values / 16e3 * 1000,
        trial_averaged_abr.T * 1e6, 
        color='k', lw=1)
    ax.set_xlabel('time from sound (ms)')
    ax.set_ylabel('ABR (uV)')
    ax.set_yticks((-6, 0, 6))
    ax.set_ylim((-6, 6))
    ax.set_xlim((-2.5, 7.5))
    ax.set_xticks((-2, 0, 2, 4, 6, ))
    my.plot.despine(ax)
    
    # Plot the stimulus
    ax.plot([0, 0.1], [5.5, 5.5], 'k-', lw=3)
    
    # Plot the baseline period
    ax.fill_between([-2.5, -1.25], y1=-6, y2=6, color='orange', alpha=0.5)
    
    f.savefig('PLOT_TRIAL_AVERAGED_ABR.svg')
    f.savefig('PLOT_TRIAL_AVERAGED_ABR.png', dpi=300)

if PLOT_POSITIVE_AND_NEGATIVE_CLICKS:
    # Plot the corresponding trial averaged ABR
    trial_averaged_abr = big_triggered_neural2.loc[
        datetime.date(2025, 5, 20)].loc['Cat_229'].loc[10].xs(
        'RV', level='channel').droplevel('speaker_side')

    # Slice out loudest only
    trial_averaged_abr = trial_averaged_abr.xs(91, level='label')
    
    # Aggregate
    trial_averaged_abr = trial_averaged_abr.groupby('polarity').mean()
    
    # Plot
    f, ax = plt.subplots(figsize=(4, 2.5))
    f.subplots_adjust(bottom=.24, left=.15, right=.95, top=.89)
    ax.plot(
        trial_averaged_abr.columns.values / 16e3 * 1000,
        trial_averaged_abr.loc[True] * 1e6, 
        color='k', lw=1)
    ax.plot(
        trial_averaged_abr.columns.values / 16e3 * 1000,
        trial_averaged_abr.loc[False] * 1e6, 
        color='k', lw=1, ls='--')
    ax.set_xlabel('time from sound (ms)')
    ax.set_ylabel('ABR (uV)')
    ax.set_yticks((-6, 0, 6))
    ax.set_ylim((-6, 6))
    ax.set_xlim((-2.5, 7.5))
    ax.set_xticks((-2, 0, 2, 4, 6, ))
    my.plot.despine(ax)
    
    # Plot the stimulus
    ax.plot([0, 0.1], [5.5, 5.5], 'k-', lw=3)
    
    
    f.savefig('PLOT_POSITIVE_AND_NEGATIVE_CLICKS.svg')
    f.savefig('PLOT_POSITIVE_AND_NEGATIVE_CLICKS.png', dpi=300)


## Store
big_abrs.to_pickle(os.path.join(output_directory, 'big_abrs'))
trial_counts.to_pickle(os.path.join(output_directory, 'trial_counts'))