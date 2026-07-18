"""Shared methods among demo scripts

"""

import os
import datetime
import my
import pandas

def load_metadata(raw_data_directory):
    """Load and format all metadata for demo scripts.
    
    Coerces date columns to `datetime.date`
    Sets MultiIndex on each DataFrame
    
    raw_data_directory : Path-like
        Path to raw Zenodo data, generally specified in filepaths.json
    
    Returns: dict
        mouse_metadata : DataFrame
        experiment_metadata : DataFrame
        recording_metadata : DataFrame
    """
    # Load all CSV
    mouse_metadata = pandas.read_csv(
        os.path.join(raw_data_directory, 'metadata', 'mouse_metadata.csv'))
    experiment_metadata = pandas.read_csv(
        os.path.join(raw_data_directory, 'metadata', 'experiment_metadata.csv'))
    recording_metadata = pandas.read_csv(
        os.path.join(raw_data_directory, 'metadata', 'recording_metadata.csv'))

    # Coerce
    recording_metadata['date'] = recording_metadata['date'].apply(
        lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())
    experiment_metadata['date'] = experiment_metadata['date'].apply(
        lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())
    mouse_metadata['DOB'] = mouse_metadata['DOB'].apply(
        lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())

    # Coerce: special case this one because it can be null
    mouse_metadata['HL_date'] = mouse_metadata['HL_date'].apply(
        lambda x: None if pandas.isnull(x) else 
        datetime.datetime.strptime(x, '%Y-%m-%d').date())

    # Index
    recording_metadata = recording_metadata.set_index(
        ['date', 'mouse', 'recording']).sort_index()
    experiment_metadata = experiment_metadata.set_index(
        ['date', 'mouse']).sort_index()
    mouse_metadata = mouse_metadata.set_index('mouse').sort_index()        


    ## TODO: exclude these upstream in Step 1
    # Drop whole mouse (noisy)
    mouse_metadata = mouse_metadata.drop('Pineapple_197')
    recording_metadata = recording_metadata.drop('Pineapple_197', level='mouse')
    experiment_metadata = experiment_metadata.drop('Pineapple_197', level='mouse')

    # Drop specific recordings (done under iso)
    midx = pandas.MultiIndex.from_tuples([
        (datetime.date(2025, 6, 6), 'Cacti_223', 14),
        (datetime.date(2025, 6, 6), 'Cacti_223', 15),
        ], names=['date', 'mouse', 'recording']
        )
    recording_metadata = my.misc.slice_df_by_some_levels(
        recording_metadata, midx, drop=True)

    
    ## Return
    return {
        'mouse_metadata': mouse_metadata,
        'experiment_metadata': experiment_metadata,
        'recording_metadata': recording_metadata,
        }