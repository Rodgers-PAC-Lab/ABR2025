# This script goes through 'mouse_info.csv'
# For each date and mouse, it gets the recording metadata for those recordings
# and checks to make sure it can load the data files.
# At the end of the script it creates these pickles:
#   recording_metadata: all the recording metadata from all dates and mice
#   cohort_experiments: mouse_info.csv with proper formats and with age computed

import os
import datetime
import glob
import json
import numpy as np
import pandas
from paclab import abr
import tqdm


## Paths
# Load paths to raw data and output directory
with open('filepaths.json') as fi:
    paths = json.load(fi)

raw_data_directory = paths['raw_data_directory']
output_directory = paths['output_directory']

# Create output_directory if it doesn't exist already (e.g., if this is the
# first run)
if not os.path.exists(output_directory):
    os.mkdir(output_directory)


## Load notes about the cohort
# Columns: date, mouse, sex, strain, genotype, DOB, HL
cohort_experiments = pandas.read_csv('mouse_info.csv')
cohort_mice = cohort_experiments['mouse'].unique()

# Turn the dates into actual datetime dates
cohort_experiments['date'] = cohort_experiments['date'].apply(
    lambda x: datetime.datetime.strptime(x,'%Y-%m-%d').date())
cohort_experiments['DOB'] = cohort_experiments['DOB'].apply(
    lambda x: datetime.datetime.strptime(x,'%Y-%m-%d').date())

# Calculate age on the day of the experiment
cohort_experiments.insert(5,'age',
                (cohort_experiments['date']-cohort_experiments['DOB']).values)
cohort_experiments['age'] = cohort_experiments['age'].apply(lambda x: x.days)


## Load metadata from each session
# An "experiment" is a series of "recordings" on a single day from a single mouse
# One mouse might be tested on multiple days, and these are different experiments
# Multiple mice might be tested on the same day, and these are different experiments
#
# On disk, all data from the same day are combined into the same dated directory,
# so experiments are mixed together
#
# For every day, there is a _notes_v5.csv file in the GUI data directory, and
# that csv file labels the mouse for every recording on that day
#
# Here, we loop over the experiments (e.g., mouse * day), and get the metadata
# for each recording on that experiment
day_notes_l = []
for i_day in cohort_experiments['date'].unique():
    print(i_day)
    # Form the folder name for the daily recording folder
    folder_datestr = datetime.datetime.strftime(i_day, '%Y-%m-%d')
    day_metadata = cohort_experiments.loc[cohort_experiments['date']==i_day]
    # We shouldn't have any mice run twice in one day
    assert len(day_metadata['mouse'].unique()) == len(day_metadata)
    day_mouses = day_metadata['mouse'].unique()
    day_metadata = day_metadata.set_index('mouse')
    # Most of the time, it'll be the same experimenter on the same day.
    # Then we can just load that metadata csv once instead of per-mouse.
    if len(day_metadata['experimenter'].unique()) == 1:
        # We ABSOLUTELY should not have a case where the same experimenter, same day,
        # is using two different versions of metadata
        assert len(day_metadata['metadata_version'].unique())==1

        experimenter = day_metadata['experimenter'].unique()[0]
        metadata_version = day_metadata['metadata_version'].unique()[0]
        # Form data_directory where both the notes csv and recording data are stored
        data_directory = os.path.join(raw_data_directory, folder_datestr, experimenter)
        # Load metadata for the day
        day_notes = abr.loading.get_metadata(data_directory, i_day.strftime('%y%m%d'), metadata_version)
        # Remove mice that aren't in the cohort
        day_notes = day_notes.loc[day_notes['mouse'].isin(day_mouses)]

        # Insert date
        day_notes['date'] = i_day
        day_notes = day_notes.reset_index().set_index(['date', 'mouse', 'recording'])

        day_notes_l.append(day_notes)
    else:
        # Loop through the day's mice and append their metadata to day_notes
        mouse_notes_l = []
        for mouse in day_metadata.index:
            experimenter = day_metadata.loc[mouse]['experimenter']
            metadata_version = day_metadata.loc[mouse]['metadata_version']
            # Form data_directory where both the notes csv and recording data are stored
            data_directory = os.path.join(raw_data_directory, folder_datestr, experimenter)
            # Load metadata for the day
            day_notes = abr.loading.get_metadata(data_directory, i_day.strftime('%y%m%d'), metadata_version)
            mouse_notes = day_notes.loc[day_notes['mouse']==mouse]
            mouse_notes_l.append(mouse_notes)
        day_notes = pandas.concat(mouse_notes_l)
        # Insert date
        day_notes['date'] = i_day
        day_notes = day_notes.reset_index().set_index(['date', 'mouse', 'recording'])
        day_notes_l.append(day_notes)


# Concat
# Indexed by date * mouse * recording, with 1 row per recording
recording_metadata = pandas.concat(day_notes_l)
recording_metadata['include'] = recording_metadata['include'].fillna(True)
# There's some recordings with v6 metadata which has an extra 'ch4_config' column
# Re-order the columns so that's next to the other channel configs because it's annoying
recording_metadata = recording_metadata.reindex(columns=['ch0_config', 'ch2_config', 'ch4_config', 'speaker_side',
        'amplitude', 'include', 'notes', 'recording_name', 'datafile'])

print('Cohort metadata loaded. Checking that each recording can be loaded...')
## Ensure binary data can be loaded for each recording
for recording in tqdm.tqdm( recording_metadata.index):
    i_metadata = recording_metadata.loc[recording]
    # Continue if 'include' is already False
    if i_metadata['include'] == False:
        continue
    # Get the filename
    recording_folder = os.path.join(i_metadata['datafile'])
    # Check if data can be loaded
    try:
        # Load raw data in volts
        data = abr.loading.load_recording(recording_folder)
    except IOError as e:
        # If not, mark 'include' as False
        #~ print(f'cannot load data from {i_metadata['datafile']}, marking to exclude')
        print(e)
        recording_metadata.loc[recording, 'include'] = False


## Store
cohort_experiments.to_pickle(os.path.join(output_directory, 'cohort_experiments'))
recording_metadata.to_pickle(os.path.join(output_directory, 'recording_metadata'))

print('Done')