## Takes metadata from google sheets and generates metadata for sharing
# This scripts converts metadata from lab-specific format to a shareable
# format. Ultimately this script will be removed from the repository because
# it won't work outside the lab.
#
# Steps:
# * Load google sheets for "mouse metadata" and "experiment metadata"
# * Get the metadata from the surgery partition on cuttlefish for each of 
#   those experiments, using the "notes_v6.csv" in those directories.
# * Writes out mouse_metadata, experiment_metadata, and recording_metadata
#   in a shareable CSV format that all other shared scripts will use.
# * Writes out "files_from" to enable an rsync from cuttlefish to a folder
#   of data to share.
#
# An "experiment" is a series of "recordings" on a single day from a single mouse
# One mouse might be tested on multiple days, and these are different experiments
# Multiple mice might be tested on the same day, and these are different experiments
# On disk, all data from the same day are combined into the same dated directory,
# so experiments are mixed together. This script will split them apart.
#
# At the end of the script it creates these CSV files:
#   mouse_metadata.csv : one row per mouse
#   experiment_metadata.csv : one row per experiment (mouse * date)
#   recording_metadata.csv: one row per recording, indexed by date * mouse * recording
#
# Run this to generate the data to share. It copies the relevant files
# from cuttlefish to a new location, which will ultimately be uploaded
# (and then downloaded, serving as the input directory)
"""
rsync -van \
    /home/chris/mnt/cuttlefish/surgery/abr_data \
    /home/chris/mnt/cuttlefish/chris/data/20251214_abr_data/ABR2025_data \
    --files-from=/home/chris/mnt/cuttlefish/chris/data/20251214_abr_data/ABR2025_data/metadata/files_from

Replace paths where appropriate with 
paths['original_cuttlefish_directory'] and paths['raw_data_directory']

I was getting CRC errors with zip
zip -r ABR2025_data.zip ABR2025_data

Try tar
tar -cvzf ABR2025_data.tar.gz ABR2025_data
"""
import os
import datetime
import json
import numpy as np
import pandas
import paclab # okay to keep this import, since this script won't be shared
import paclab.abr_misc
import ABR2025
import tqdm


## Paths
# Load the required file filepaths.json (see README)
with open('filepaths.json') as fi:
    paths = json.load(fi)

# Parse into paths to raw data and output directory
cuttlefish_directory = paths['original_cuttlefish_directory']
raw_data_directory = paths['raw_data_directory']
output_directory = paths['output_directory']

# Create raw_data_directory and output_directory if needed 
# (for example, if this is the first run)
if not os.path.exists(raw_data_directory):
    os.mkdir(raw_data_directory)
if not os.path.exists(output_directory):
    os.mkdir(output_directory)


## Params
# In this dataset, all clicks should have this amplitude
expected_amplitude = [
    0.01,  0.0063,  0.004, 0.0025, 0.0016, 0.001, 0.00063, 
    0.0004, 0.00025, 0.00016, 0.0001, 6.3e-05, 4e-05,
    ]


## Load mouse metadata
# For convenience now, the google doc is the hot copy: 
#   https://docs.google.com/spreadsheets/d/1J0AFGFZp5V91GxNgEJvWDqN1v0S22znYZwBFbDCbEes/edit?gid=1105382891#gid=1105382891
# Before sharing repository, download these as CSV in the ./metadata directory
# so the user doesn't have to rely on a google doc

# The mice in the dataset
mouse_metadata = paclab.load_gsheet.load(
    '1J0AFGFZp5V91GxNgEJvWDqN1v0S22znYZwBFbDCbEes', 
    sheet_name='mouse metadata', normalize_case=False)

# The experiments in the dataset
experiment_metadata = paclab.load_gsheet.load(
    '1J0AFGFZp5V91GxNgEJvWDqN1v0S22znYZwBFbDCbEes', 
    sheet_name='experiment metadata', normalize_case=False)


## Clean metadata
# When exporting to CSV first, datetime-like columns are str
# Turn the dates into actual datetime dates
# For compatibility with the code below, force them back into str and drop the
# time component
experiment_metadata['date'] = experiment_metadata['date'].apply(
    lambda ts: str(ts.date()))
mouse_metadata['DOB'] = mouse_metadata['DOB'].apply(
    lambda ts: str(ts.date()))
mouse_metadata['HL_date'] = mouse_metadata['HL_date'].apply(
    lambda ts: None if pandas.isnull(ts) else str(ts.date()))

# Error check for duplicates
assert not mouse_metadata.duplicated().any()
assert not experiment_metadata.duplicated().any()

# Replace null HL_type with the string 'none', because otherwise it is
# awkwardly dropped by `groupby`
mouse_metadata['HL_type'] = mouse_metadata['HL_type'].fillna('none')

# When downloading directly from google, datetime-like columns are Timestamp
experiment_metadata['date'] = experiment_metadata['date'].apply(
    lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())
mouse_metadata['DOB'] = mouse_metadata['DOB'].apply(
    lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())

# Special case this one because it can be null
mouse_metadata['HL_date'] = mouse_metadata['HL_date'].apply(
    lambda x: None if pandas.isnull(x) else 
    datetime.datetime.strptime(x, '%Y-%m-%d').date())


## Calculate more metadata
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

for idx in tqdm.tqdm(experiment_metadata.index):
    # Get the experimenter, date, and mouse
    experimenter, experiment_date, mouse = experiment_metadata.loc[
        idx, ['experimenter', 'date', 'mouse']]

    # Form data_directory where both the notes csv and recording data are stored
    folder_datestr = datetime.datetime.strftime(experiment_date, '%Y-%m-%d')
    data_directory = os.path.join(
        cuttlefish_directory, folder_datestr, experimenter)
    
    # Load metadata for the day (hardcoding v6)
    this_date_metadata = paclab.abr_misc.get_metadata(
        data_directory, experiment_date.strftime('%y%m%d'), 'v6')
    
    # Check that we actually found data
    assert len(this_date_metadata) > 0
    assert mouse in this_date_metadata['mouse'].values
    
    # Include only data from this mouse
    this_date_metadata = this_date_metadata[
        this_date_metadata['mouse'] == mouse]
    
    # Insert date
    this_date_metadata['date'] = experiment_date
    
    # Drop datafile, which contains the full path, which can differ based
    # on user and filesystem
    this_date_metadata = this_date_metadata.drop('datafile', axis=1)
    
    # Replace datafile with a relative path within cuttlefish_directory
    def recording_name2short_datafile(recording_name):
        """Turn bare recording name into a path within cuttlefish_directory.
        
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


## Error check that we always have >1 recordings per speaker_side
recordings_per_side = recording_metadata.groupby(
    ['date', 'mouse', 'speaker_side']).size().unstack('speaker_side')

assert (recordings_per_side > 1).all().all()


## Error check that amplitude is always the same
for actual_amplitude in recording_metadata['amplitude'].values:
    assert actual_amplitude == expected_amplitude

# Drop this column
recording_metadata = recording_metadata.drop('amplitude', axis=1)


## Sort
mouse_metadata = mouse_metadata.sort_index().sort_index(axis=1)
experiment_metadata = experiment_metadata.sort_index().sort_index(axis=1)
recording_metadata = recording_metadata.sort_index().sort_index(axis=1)


## Store metadata
# Drop the notes columns
mouse_metadata = mouse_metadata.drop('notes', axis=1)
experiment_metadata = experiment_metadata.drop('notes', axis=1)
recording_metadata = recording_metadata.drop('notes', axis=1)

# Drop redundant
assert recording_metadata['include'].all()
recording_metadata = recording_metadata.drop('include', axis=1)

# Make a place for metadata
make_dir = os.path.join(raw_data_directory, 'metadata')
if not os.path.exists(make_dir):
    os.mkdir(make_dir)

# Write out as csv in the raw data directory
# These will be shared alongside the raw data
# recording_metadata is the only one with a useful index
mouse_metadata.to_csv(
    os.path.join(raw_data_directory, 'metadata', 'mouse_metadata.csv'),
    index=False)
experiment_metadata.to_csv(
    os.path.join(raw_data_directory, 'metadata', 'experiment_metadata.csv'),
    index=False)
recording_metadata.to_csv(
    os.path.join(raw_data_directory, 'metadata', 'recording_metadata.csv'))

# Write out a files_from for rsync
files_from_text = '\n'.join(
    recording_metadata['short_datafile'].apply(lambda s: s + '/'))
with open(os.path.join(raw_data_directory, 'metadata', 'files_from'), 'w') as fi:
    fi.write(files_from_text + '\n')

# Figure output will go here
if not os.path.exists('figures'):
    os.mkdir('figures')
