## Remove outliers, aggregate ABRs, and calculate thresholds
# Writes out:
#   big_abrs - averaged ABRs

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


## Aggregate
# Count the number of trials in each experiment
trial_counts = big_triggered_neural.groupby(
    ['date', 'mouse', 'recording', 'label', 'channel']).size()

# Iterate over recordings
abrs_l = []
arts_l = []
keys_l = []
for date, mouse, recording in recording_metadata.index:

    # Slice
    triggered_neural = big_triggered_neural.loc[date].loc[mouse].loc[recording]


    ## Identify outlier trials, separately by channel
    # Trim outliers
    triggered_neural2 = triggered_neural.groupby('channel').apply(
        lambda df: abr.signal_processing.trim_outliers(
            df.droplevel('channel'),
            abs_max_sigma=abs_max_sigma,
            stdev_sigma=stdev_sigma,
        ))

    # Reorder levels to be like triggered_neural
    triggered_neural2 = triggered_neural2.reorder_levels(
        triggered_neural.index.names).sort_index()


    ## Aggregate
    # Average by polarity, label, channel over t_samples
    avg_by_condition = triggered_neural2.groupby(
        ['polarity', 'channel', 'label']).mean()

    # The ABR adds over polarity
    avg_abrs = (avg_by_condition.loc[True] + avg_by_condition.loc[False]) / 2

    # The artefact subtracts over polarity
    avg_arts = (avg_by_condition.loc[True] - avg_by_condition.loc[False]) / 2


    ## Store
    abrs_l.append(avg_abrs)
    arts_l.append(avg_arts)
    keys_l.append((date, mouse, recording))

# Concat
big_abrs = pandas.concat(abrs_l, keys=keys_l, names=['date', 'mouse', 'recording'])
big_arts = pandas.concat(arts_l, keys=keys_l, names=['date', 'mouse', 'recording'])

# TODO: identify which recordings have large arts and drop them


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