## Run ABRA on our data
"""
You can skip this script if you don't have ABRA
This script must be run in a conda environment that satisfies ABRA's 
requirements, and ABRA must be on the PYTHONPATH, and you must be using
our fork where the path to the CNN is fixed to allow running outside the repo
See README for further instructions

Runs utils.calculate.calculate_and_plot_wave_exact outside streamlit

The ABRAnalysis core algorithm is in utils/calculate.py, function peak_finding.
* W1 is the only one that is modeled, using the CNN. This is used only to 
  set the search window for W1. 
* Standard find_peaks is used with no height or prominence filters, only a 
  distance of 16 samples (at 24.4 kHz). This returns tallest peaks that 
  satisfy this spacing criterion.
* W1 is the closest peak found by find_peaks within the search window predicted
  by CNN. 
* W2-W5 are numbered with argsort after W1, so extra or missing will frame shift
"""

import os
import json
import numpy as np
import pandas
import easydict


## Complex imports
# importing streamlit requires an environment variable
# Some version info with protobuf, possibly streamlit's protobuf-generated
# code may have been built against a newer version than what I have
# This workaround fixes it by using the pure Python (slower) implementation
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
import streamlit

# Provide the session state keys needed by streamlit to set the stage for
# our analysis
streamlit.session_state = easydict.EasyDict(
    time_scale=10, # total window length in ms
    units='Microvolts', # Tell it to expect microvolts
    multiply_y_factor=1e6, # Convert our data to microvolts
    atten=False, # we will specify level as 'Level(dB)'
    )

# Now import the streamlit code
import abranalysis.utils.calculate


## Paths
# Load the required file filepaths.json (see README)
with open('filepaths.json') as fi:
    paths = json.load(fi)

# Parse into paths to raw data and output directory
raw_data_directory = paths['raw_data_directory']
output_directory = paths['output_directory']


## Params
duration_ms = streamlit.session_state.time_scale
duration_samples = int(np.rint(16e3 * duration_ms / 1e3))


## Load data
big_abr = pandas.read_parquet(
    os.path.join(output_directory, 'averaged_abrs_by_mouse'))

# Keep only after_HL == False
big_abr = big_abr.xs(False, level='after_HL').droplevel('HL_type')

# Keep only vertex-ear channels
big_abr = big_abr.drop('RL', level='channel')


## Run through abranalysis
# Run each mouse * channel * speaker_side separately, leaving only `level`
grouping_keys = ['mouse', 'channel', 'speaker_side']
wave_peaks_l = []
wave_peaks_keys_l = []

for keys, this_abr in big_abr.groupby(grouping_keys):
    
    # Droplevel on slice
    this_abr = this_abr.droplevel(grouping_keys).copy()
    
    # Slice out 0 - 10 ms
    this_abr = this_abr.loc[:, 0:159]
    
    # Set columns as strings (required by abranalysis)
    string_labeled_columns = pandas.Index([str(n) for n in this_abr.columns])
    this_abr.columns = string_labeled_columns
    
    # Put metadata as columns not on MultiIndex
    this_abr['Level(dB)'] = this_abr.index.values
    this_abr['Freq(Hz)'] = 'Click'
    this_abr = this_abr.reset_index(drop=True)
    
    # abranalysis assumes this column order
    this_abr = this_abr[
        ['Freq(Hz)', 'Level(dB)'] + list(string_labeled_columns)
        ]
    
    # Iterate over sound levels 
    row_l = []
    for sound_level in this_abr['Level(dB)']:
        
        # Get peaks from abranalysis
        res = abranalysis.utils.calculate.calculate_and_plot_wave_exact(
            this_abr, 'Click', sound_level, return_peaks=True,
            )
        orig_x, orig_y, peaks, troughs = res
 
        # Continue if no peaks found
        if orig_y is None:
            assert orig_x is None
            assert peaks is None
            assert troughs is None
            continue
        
        # Error check peaks and troughs detected
        assert len(peaks) > 0
        assert len(troughs) > 0
        assert not pandas.isnull(peaks).any()
        assert not pandas.isnull(troughs).any()
        
        # Error check index order
        assert len(orig_y) == duration_samples
        assert orig_y.index.equals(string_labeled_columns)
        
        # Convert to array (all other variables already arrays)
        orig_y = orig_y.loc[string_labeled_columns].values

        # `troughs` can be one shorter than `peaks` if the last peak is 
        # too close to the end of the series
        if len(troughs) == len(peaks) - 1:
            troughs = np.concatenate([troughs, [np.nan]])
        
        # Convert to Series for storage
        wave_peaks = pandas.DataFrame.from_dict({
            'peak': peaks,
            'trough': troughs,
            }).astype(float)
        wave_peaks.index = pandas.Index(
            1 + np.arange(len(wave_peaks)), name='wave')

        # Check that peaks always occur before troughs
        null_mask = wave_peaks['trough'].isnull()
        assert wave_peaks['peak'].is_monotonic_increasing
        assert wave_peaks['trough'].dropna().is_monotonic_increasing
        assert (
            wave_peaks.loc[~null_mask, 'peak'] < 
            wave_peaks.loc[~null_mask, 'trough']
            ).all()
        
        # Check that it always finds 5 waves - not guaranteed but seems to
        # be the case in this data
        assert len(wave_peaks) == 5
        
        # Store
        wave_peaks_l.append(wave_peaks)
        wave_peaks_keys_l.append(tuple(list(keys) + [sound_level]))


## Concat
big_abra_peaks = pandas.concat(
    wave_peaks_l, keys=wave_peaks_keys_l, names=grouping_keys + ['sound_level'])


## Store
big_abra_peaks.to_parquet(os.path.join(output_directory, 'big_abra_peaks'))