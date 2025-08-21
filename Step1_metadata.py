# This script goes through 'mouse_info.csv'
# For each date and mouse, it gets the recording metadata for those recordings
# and checks to make sure it can load the data files.
#
# An "experiment" is a series of "recordings" on a single day from a single mouse
# One mouse might be tested on multiple days, and these are different experiments
# Multiple mice might be tested on the same day, and these are different experiments
# On disk, all data from the same day are combined into the same dated directory,
# so experiments are mixed together. This script will split them apart.
#
# At the end of the script it creates these pickles:
#   mouse_metadata : one row per mouse
#   experiment_metadata : one row per experiment (mouse * date)
#   recording_metadata: one row per recording, indexed by date * mouse * recording

import os
import datetime
import glob
import json
import numpy as np
import pandas
from paclab import abr
import tqdm


## Paths
# Load the required file filepaths.json (see README)
with open('filepaths.json') as fi:
    paths = json.load(fi)

# Parse into paths to raw data and output directory
raw_data_directory = paths['raw_data_directory']
output_directory = paths['output_directory']

# Create output_directory if it doesn't exist already (for example, 
# if this is the first run)
if not os.path.exists(output_directory):
    os.mkdir(output_directory)


## Params
# In this dataset, all clicks should have this amplitude
expected_amplitude = [
    0.01,  0.0063,  0.004, 0.0025, 0.0016, 0.001, 0.00063, 
    0.0004, 0.00025, 0.00016, 0.0001, 6.3e-05, 4e-05,
    ]


## Load mouse metadata
# The source data is in the lab google drive -- download to CSVs here
# The mice in the dataset

mouse_metadata = pandas.read_csv(
    './metadata/2025-07-25 ABR paper metadata - mouse metadata.csv')

# The experiments in the dataset
experiment_metadata = pandas.read_csv(
    './metadata/2025-07-25 ABR paper metadata - experiment metadata.csv')

# Error check for duplicates
assert not mouse_metadata.duplicated().any()
assert not experiment_metadata.duplicated().any()

# Presently, all experiments were done by 'rowan'
experiment_metadata['experimenter'] = 'rowan'

# Turn the dates into actual datetime dates
experiment_metadata['date'] = experiment_metadata['date'].apply(
    lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())
mouse_metadata['DOB'] = mouse_metadata['DOB'].apply(
    lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())

# Special case this one because it can be null
mouse_metadata['HL_date'] = mouse_metadata['HL_date'].apply(
    lambda x: None if pandas.isnull(x) else 
    datetime.datetime.strptime(x, '%Y-%m-%d').date())

# Calculate age on the day of the experiment
experiment_metadata = experiment_metadata.join(
    mouse_metadata.set_index('mouse')['DOB'], on='mouse')
experiment_metadata['age'] = (
    experiment_metadata['date'] - experiment_metadata['DOB']
    ).apply(lambda x: x.days)
experiment_metadata = experiment_metadata.drop('DOB', axis=1)

# Calculate whether each experiment was before or after hearing loss
experiment_metadata = experiment_metadata.join(
    mouse_metadata.set_index('mouse')['HL_date'], on='mouse')
experiment_metadata['after_HL'] = (
    experiment_metadata['date'] > experiment_metadata['HL_date']
    )
experiment_metadata = experiment_metadata.drop('HL_date', axis=1)


## Number the sessions within for each group of mouse * after_HL
# This will define n_experiment as {0, 1, ...} before hearing loss and
# (if applicable) {0, 1, ...} after hearing loss

# Sort by date
experiment_metadata = experiment_metadata.sort_values('date')

# Group
gobj = experiment_metadata.groupby(['mouse', 'after_HL'])

# Number sessions within each group
subdf_l = []
for (mouse, after_HL), subdf in gobj:
    subdf['n_experiment'] = range(len(subdf))
    
    # Store
    subdf_l.append(subdf)

# Concat
experiment_metadata = pandas.concat(subdf_l).sort_index()


## Load metadata from each recording
# For every day, there is a _notes_v6.csv file in the GUI data directory, and
# that csv file labels the mouse for every recording on that day
# Here, we loop over the experiments (e.g., experimenter * mouse * day), 
# and get the metadata for each recording on that experiment
# Load data from every date that is included
recording_metadata_l = []

for idx in experiment_metadata.index:
    # Get the experimenter, date, and mouse
    experimenter, experiment_date, mouse = experiment_metadata.loc[
        idx, ['experimenter', 'date', 'mouse']]

    # Form data_directory where both the notes csv and recording data are stored
    folder_datestr = datetime.datetime.strftime(experiment_date, '%Y-%m-%d')
    data_directory = os.path.join(
        raw_data_directory, folder_datestr, experimenter)
    
    # Load metadata for the day (hardcoding v6)
    this_date_metadata = abr.loading.get_metadata(
        data_directory, experiment_date.strftime('%y%m%d'), 'v6')
    
    # Include only data from this mouse
    this_date_metadata = this_date_metadata[
        this_date_metadata['mouse'] == mouse]
    
    # Insert date
    this_date_metadata['date'] = experiment_date
    
    # Drop datafile, which contains the full path, which can differ based
    # on user and filesystem
    this_date_metadata = this_date_metadata.drop('datafile', axis=1)
    
    # Replace datafile with a relative path within raw_data_directory
    def recording_name2short_datafile(recording_name):
        """Turn bare recording name into a path within raw_data_directory.
        
        Join folder_datestr and experimenter before recording_name        
        """
        return os.path.join(folder_datestr, experimenter, recording_name)

    this_date_metadata['short_datafile'] = this_date_metadata[
        'recording_name'].apply(recording_name2short_datafile)
    
    # Reindex
    this_date_metadata = this_date_metadata.set_index(
        ['date', 'mouse', 'recording'])    

    # Store
    recording_metadata_l.append(this_date_metadata)

# Concat
# Indexed by date * mouse * recording, with 1 row per recording
recording_metadata = pandas.concat(recording_metadata_l)


## Include only some recordings
# Default value for 'include' is True
recording_metadata['include'] = recording_metadata['include'].fillna(True)

# Drop those with 'include' == False
recording_metadata = recording_metadata[recording_metadata['include'] == True]


## Error check that amplitude is always the same
for actual_amplitude in recording_metadata['amplitude'].values:
    assert actual_amplitude == expected_amplitude

# Drop
recording_metadata = recording_metadata.drop('amplitude', axis=1)


## Sort
mouse_metadata = mouse_metadata.sort_index().sort_index(axis=1)
experiment_metadata = experiment_metadata.sort_index().sort_index(axis=1)
recording_metadata = recording_metadata.sort_index().sort_index(axis=1)


## Store
mouse_metadata.to_pickle(
    os.path.join(output_directory, 'mouse_metadata'))
experiment_metadata.to_pickle(
    os.path.join(output_directory, 'experiment_metadata'))
recording_metadata.to_pickle(
    os.path.join(output_directory, 'recording_metadata'))
