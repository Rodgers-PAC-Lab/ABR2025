## Write out correlations
# Writes out
# - wthn_mouse_R2
# - btwn_mouse_R2
# - wthn_mouse_R2_all_sound

import os
import datetime
import glob
import json
import matplotlib
import scipy.signal
import numpy as np
import pandas
import paclab.abr
import my.plot
import matplotlib.pyplot as plt
from itertools import combinations


def flatten_across_sounds_or_time(big_abrs,time_ilocs='None',
                    sound_levels=[54, 58, 61, 65, 69, 73, 77, 81, 85, 88, 91],
                    keep_recordings=False):
    """
    Selects the desired times and/or sound levels from big_abrs and concats them all together
    In the resulting flattened_df, each row is one mouse/config/recording combo (recording is optional)
    and the columns are a million voltage values including all the sound levels

    Arguments:
        big_abrs:   pandas.DataFrame, with the normal format from the pickles
        time_ilocs: Either 'None' or a list of ints [start,stop] which are the ilocs of
            the times within an ABR you want to use. So [20,140] would take columns -15:119 for each sound level.
        sound_levels: list of ints. The sound levels you want to concat over.
            Often we don't want the softest sounds because they usually have low correlations.
        keep_recordings: T/F. Do you want to average over all the recordings of a given config?

    Returns:
        flattened_df:   pandas.DataFrame flattened as specified.
    """
    subdf = big_abrs
    if keep_recordings==False:
        subdf = subdf.drop(columns='recording')
        subdf = subdf.groupby(subdf.index.names).mean()
    else:
        subdf = subdf.set_index('recording',append=True)
    df_index = subdf.index.to_frame()
    if time_ilocs!='None':
        subdf = subdf.iloc[:,time_ilocs[0]:time_ilocs[1]]

    flat_values_l = []
    configs_l = []
    if keep_recordings:
        gobj = subdf.groupby(['mouse','timepoint','channel','speaker_side','recording'])
    else:
        gobj = subdf.groupby(['mouse','timepoint','channel','speaker_side'])
    for groups,cond_subdf in gobj:
        if keep_recordings:
            to_concat = cond_subdf.droplevel(['mouse','timepoint', 'speaker_side', 'channel'], axis=0)
            col_names = ['mouse','timepoint','channel','speaker_side','recording']
        else:
            to_concat = cond_subdf.droplevel(['mouse','timepoint', 'speaker_side', 'channel'], axis=0)
            col_names = ['mouse', 'timepoint', 'channel', 'speaker_side']

        lblconcat_l = []
        for label in sound_levels:
            lblconcat_l.append(to_concat.loc[label])
        concat_flat = np.array(lblconcat_l).flatten()


        # Append the flattened values and index keys to their respective lists
        flat_values_l.append(concat_flat)
        configs_l.append(groups)

    data_df = pandas.DataFrame(flat_values_l)
    # configs_df = pandas.DataFrame(configs_l, columns=['mouse', 'timepoint', 'channel', 'speaker_side'])
    configs_df = pandas.DataFrame(configs_l,columns=col_names)
    flattened_df = pandas.concat([configs_df, data_df], axis=1)
    flattened_df = flattened_df.set_index(col_names)
    return flattened_df



## Params
sampling_rate = 16000  # TODO: store in recording_metadata
my.plot.manuscript_defaults()
my.plot.font_embed()

loudest_dB = 91


## Paths
# Load the required file filepaths.json (see README)
with open('filepaths.json') as fi:
    paths = json.load(fi)

# Parse into paths to raw data and output directory
raw_data_directory = paths['raw_data_directory']
output_directory = paths['output_directory']


## Load results of Step1-3
cohort_experiments = pandas.read_pickle(os.path.join(output_directory, 'cohort_experiments'))
recording_metadata = pandas.read_pickle(os.path.join(output_directory, 'recording_metadata'))
big_abrs = pandas.read_pickle(os.path.join(output_directory, 'big_abrs'))
big_triggered_neural = pandas.read_pickle(os.path.join(output_directory, 'big_triggered_neural'))

# Drop those with 'include' == False
recording_metadata = recording_metadata[recording_metadata['include'] == True]

# Fillna
cohort_experiments['HL'] = cohort_experiments['HL'].fillna('none')

## Remove outliers, aggregate, and average ABRs
# Outlier params
abs_max_sigma = 3
stdev_sigma = 3

# Count the number of trials in each experiment
trial_counts = big_triggered_neural.groupby(['date', 'mouse', 'recording', 'label', 'channel']).count()


## Calculate the stdev(ABR) as a function of level
# window=20 (1.25 ms) seems the best compromise between smoothing the whole
# response and localizing it to a reasonably narrow window (and not extending
# into the baseline period)
# Would be good to extract more baseline to use here
# The peak is around sample 34 (2.1 ms), ie sample 24 - 44, and there is a
# variable later peak.
big_abr_stds = big_abrs.T.rolling(window=20, center=True, min_periods=1).std().T

# Use samples -40 to -20 as baseline
big_abr_baseline_std = big_abr_stds.loc[:, -30]

# Use samples 24 - 44 as evoked peak
big_abr_evoked_std = big_abr_stds.loc[:, 34]

# Replace the date with the timepoint
dated_big_abrs = big_abrs.copy()
big_abrs = big_abrs.join(cohort_experiments.set_index(['date','mouse'])['timepoint'],on=['date','mouse'])
big_abrs = big_abrs.reset_index().set_index(['mouse','timepoint','channel','speaker_side','label'])
big_abrs = big_abrs.drop(columns=['date'])
big_abrs = big_abrs.sort_index(level='timepoint')

# List of all mice
mouse_l = big_abrs.index.get_level_values('mouse').unique()

# List of mice who have more than one recording, and therefore can have a within-mouse correlation
mouse_counts = cohort_experiments['mouse'].value_counts()
repeated_mouse_l = mouse_counts.loc[mouse_counts>1].index.to_list()


flattened_attempt = flatten_across_sounds_or_time(big_abrs)
flat_wrecordings = flatten_across_sounds_or_time(big_abrs, keep_recordings=True)

flat_sound = flatten_across_sounds_or_time(big_abrs)


# Get the within-mouse correlation from the df flattened by sound
# You can only do this on mice who have been tested more than one day
dicts_l = []
for mouse in repeated_mouse_l:
    subdf = flat_sound.loc[mouse]
    gobj = subdf.groupby(['channel','speaker_side'])
    for (channel, speaker_side), cond_subdf in gobj:
        # print(channel, speaker_side)
        cond_subdf = cond_subdf.droplevel(['speaker_side', 'channel'], axis=0)
        cond_subdf = cond_subdf.sort_index(level='timepoint')

        correls_l = []
        combos_l = []
        for (colA, colB) in combinations(cond_subdf.index, 2):
            combo_n = colA + colB
            pr=scipy.stats.pearsonr(cond_subdf.loc[colA], cond_subdf.loc[colB])[0]
                        # correls_l.append(abs(pr))
            correls_l.append(pr**2)
            combos_l.append(combo_n)
        correl_d = {
            'mouse': mouse,
            'channel': channel,
            'speaker_side': speaker_side,
            'combo': combos_l,
            'pearsonR2': correls_l
        }
        correl_df = pandas.DataFrame(correl_d)
        dicts_l.append(correl_df)
wthn_mouse_R2 = pandas.concat(dicts_l)
wthn_mouse_R2.to_pickle(os.path.join(output_directory, 'wthn_mouse_R2'))


# Now get the between-mouse correlations from pre-HL days.
# This currently only takes the combos across one timepoint,
# Eg it won't compare mouseAday1 to mouseBday2.
pre_flat = flat_sound.loc[flat_sound.index.get_level_values('timepoint').isin(['apreA','apreB'])]
gobj = pre_flat.groupby(['timepoint','channel','speaker_side'])
dicts_l = []
for (timepoint, channel,speaker_side),subdf in gobj:
    subdf = subdf.droplevel(['channel','speaker_side','timepoint'])
    correls_l = []
    mouseA_l = []
    mouseB_l = []
    combos_l = []
    for (colA, colB) in combinations(subdf.index, 2):
        combo_n = colA + colB
        pr = scipy.stats.pearsonr(subdf.loc[colA], subdf.loc[colB])[0]
        # correls_l.append(abs(pr))
        correls_l.append(pr ** 2)
        mouseA_l.append(colA)
        mouseB_l.append(colB)
        # combos_l.append([colA,colB])
    correl_d = {
        'timepoint':    timepoint,
        'channel': channel,
        'speaker_side': speaker_side,
        # 'combo': combos_l,
        'mouseA': mouseA_l,
        'mouseB': mouseB_l,
        'pearsonR2': correls_l
    }
    correl_df = pandas.DataFrame(correl_d)
    dicts_l.append(correl_df)
btwn_mouse_R2 = pandas.concat(dicts_l)
btwn_mouse_R2.to_pickle(os.path.join(output_directory, 'btwn_mouse_R2'))


# Get correlation within-mouse across timepoints WITHOUT flattening it by sound level
dicts_l = []
for mouse in repeated_mouse_l:
    subdf = big_abrs.loc[mouse].groupby(['timepoint', 'channel', 'speaker_side','label']).mean()
    subdf = subdf.drop(columns='recording')
    gobj = subdf.groupby(['channel','speaker_side','label'])
    for (channel,speaker_side,label),subberdf in gobj:
        subberdf = subberdf.loc[:,channel,speaker_side, label].sort_index()
        subberdf = subberdf.reset_index()
        subberdf = subberdf.set_index('timepoint')
        # subberdf = subberdf.drop(columns='date')
        correls_l = []
        combos_l = []
        for (colA, colB) in combinations(subberdf.index, 2):
            combo_n = colA + colB
            pr=scipy.stats.pearsonr(subberdf.loc[colA], subberdf.loc[colB])[0]
                        # correls_l.append(abs(pr))
            correls_l.append(pr**2)
            combos_l.append(combo_n)
        correl_d = {
            'mouse' :   mouse,
            'channel': channel,
            'speaker_side': speaker_side,
            'label':    label,
            'combo':    combos_l,
            'pearsonR2': correls_l
        }
        correl_df = pandas.DataFrame(correl_d)
        dicts_l.append(correl_df)
wthn_mouse_R2_all_sound = pandas.concat(dicts_l)
wthn_mouse_R2_all_sound.to_pickle(os.path.join(output_directory, 'wthn_mouse_R2_all_sound'))
