## Remove outliers, aggregate ABRs, and calculate thresholds
# Writes out:
#   big_abrs - averaged ABRs
#   trial_counts - trial counts

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


## Drop outlier trials, separately by channel
# Consider outliers separately for every channel on every recording
group_by = ['date', 'mouse', 'recording', 'channel']
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
    ['date', 'mouse', 'recording', 'label', 'channel']).size()


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


## TODO: review each individual recording and decide about keeping it


## Arbitrarily choose Cat_229 on 2025-05-15 as the example to show
single_trial_abr = big_triggered_neural2.loc[
    datetime.date(2025, 5, 15)].loc['Cat_229'].loc[1]

single_trial_abr = single_trial_abr.xs(91, level='label').xs('LV', level='channel')

f, ax = plt.subplots()
ax.plot(single_trial_abr.T, color='k', alpha=.2)


## Join on speaker_side
# Should do this at the same time as joining channel
idx = big_abrs.index.to_frame().reset_index(drop=True)
idx = idx.join(
    recording_metadata['speaker_side'], on=['date', 'mouse', 'recording'])
big_abrs.index = pandas.MultiIndex.from_frame(idx)
big_abrs = big_abrs.reorder_levels(
    ['date', 'mouse', 'speaker_side', 'recording', 'channel', 'label']
    ).sort_index()

# Same for arts
idx = big_arts.index.to_frame().reset_index(drop=True)
idx = idx.join(
    recording_metadata['speaker_side'], on=['date', 'mouse', 'recording'])
big_arts.index = pandas.MultiIndex.from_frame(idx)
big_arts = big_arts.reorder_levels(
    ['date', 'mouse', 'speaker_side', 'recording', 'channel', 'label']
    ).sort_index()


## Further aggregate over recordings within an experiment
# Average recordings together where everything else is the same
big_abrs = big_abrs.groupby(
    ['date', 'mouse', 'recording', 'channel', 'speaker_side', 'label']).mean()


## Store
big_abrs.to_pickle(os.path.join(output_directory, 'big_abrs'))
trial_counts.to_pickle(os.path.join(output_directory, 'trial_counts'))