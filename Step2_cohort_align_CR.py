## Identify and categorize clicks, and slice neural and audio data around them
# This script takes a while.
#
# Writes out the following in the output directory
#   big_triggered_ad - audio data triggered on clicks
#   big_triggered_neural - neural data triggered on clicks
#   big_click_params - click metadata
#   big_abrs - averaged ABRs
#   thresholds - computed ABR thresholds

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


## Params
abr_start_sample = -40
abr_stop_sample = 120
abr_highpass_freq = 300
abr_lowpass_freq = 3000
sampling_rate = 16000 # TODO: store in recording_metadata?
neural_channel_numbers = [0, 2, 4]

# Outlier params
abs_max_sigma = 3
stdev_sigma = 3


## Paths
# Load the required file filepaths.json (see README)
with open('filepaths.json') as fi:
    paths = json.load(fi)

# Parse into paths to raw data and output directory
raw_data_directory = paths['raw_data_directory']
output_directory = paths['output_directory']

# Put click figures here
# TODO: bring these figures back, for error checking
click_data_dir = os.path.join(output_directory, 'click_validation')
if not os.path.exists(click_data_dir):
    os.mkdir(click_data_dir)


## Load results of main1
cohort_experiments = pandas.read_pickle(
    os.path.join(output_directory, 'cohort_experiments'))
recording_metadata = pandas.read_pickle(
    os.path.join(output_directory, 'recording_metadata'))

# Drop those with 'include' == False
recording_metadata = recording_metadata[recording_metadata['include'] == True]


## Load data from each recording
click_params_l = []
triggered_ad_l = []
triggered_neural_l = []
keys_l = []

# Iterate over recordings
for date, mouse, recording in tqdm.tqdm(recording_metadata.index):
    
    # Get the recording info
    this_recording = recording_metadata.loc[date].loc[mouse].loc[recording]

    
    ## Load raw data in volts
    # Get the filename
    folder_datestr = datetime.datetime.strftime(date, '%Y-%m-%d')
    recording_folder = os.path.normpath(this_recording['datafile'])
    
    # Load the data
    data = abr.loading.load_recording(recording_folder)
    data = data['data']
    
    # Parse into neural and speaker data
    # Presently, neural data is always on channels 0, 2, and 4 at most (maybe fewer)
    speaker_signal_V = data[:, 7]
    neural_data_V = data[:, neural_channel_numbers]


    ## Set up the click categories
    # Convert the autopilot amplitudes to voltages
    # This 1.34 is empirically determined to align autopilot with measured voltage
    log10_voltage = np.sort(np.log10(this_recording['amplitude']) + 1.34)
    
    # Generate labels .. convert voltage to dB and normalize to 70 dB, just 
    # to make it positive, this is not SPL
    amplitude_labels_old = np.rint(20 * log10_voltage + 70).astype(int)
    
    # SPL as measured with the fancy microphone
    amplitude_labels = np.array(
        [49, 51, 54, 58, 61, 65, 69, 73, 77, 81, 85, 88, 91])

    # Convert the voltages to cuts
    amplitude_cuts = (log10_voltage[1:] + log10_voltage[:-1]) / 2

    # Add a first and last amplitude cut
    diff_cut = np.mean(np.diff(amplitude_cuts))
    amplitude_cuts = np.concatenate([
        [amplitude_cuts[0] - diff_cut],
        amplitude_cuts,
        [amplitude_cuts[-1] + diff_cut],
        ])
    
    
    ## Identify and categorize clicks
    # Use the least cut as the threshold
    threshold_V = 10 ** amplitude_cuts.min()
    
    # Identify clicks
    identified_clicks = abr.signal_processing.identify_click_times(
        speaker_signal_V, 
        threshold_V=threshold_V,
        sampling_rate=sampling_rate, 
        slice_start_sample=abr_start_sample, 
        slice_stop_sample=abr_stop_sample,
        )
    
    # Pull out the highpassed signal because we need it later
    speaker_signal_hp = identified_clicks['highpassed']

    # Categorize them
    click_params = abr.signal_processing.categorize_clicks(
        identified_clicks['peak_time_samples'], 
        speaker_signal_hp, 
        amplitude_cuts, 
        amplitude_labels,
        )    


    ## Extract each trigger from the audio signal
    triggered_ad = abr.signal_processing.slice_audio_on_clicks(
        speaker_signal_hp, click_params)


    ## Extract neural data locked to onsets
    # Highpass neural data
    nyquist_freq = sampling_rate / 2
    ahi, bhi = scipy.signal.butter(
        2, (
        abr_highpass_freq / nyquist_freq, 
        abr_lowpass_freq / nyquist_freq), 
        btype='bandpass')
    neural_data_hp = scipy.signal.filtfilt(ahi, bhi, neural_data_V, axis=0)

    # Extract highpassed neural data around triggers
    # Shape is (n_triggers, n_timepoints, n_channels)
    triggered_neural = np.array([
        neural_data_hp[trigger + abr_start_sample:trigger + abr_stop_sample]
        for trigger in click_params['t_samples']])

    # Remove channel as a level
    triggered_neural = triggered_neural.reshape(
        [triggered_neural.shape[0], -1])        

    # DataFrame
    triggered_neural = pandas.DataFrame(triggered_neural)
    triggered_neural.index = pandas.MultiIndex.from_frame(
        click_params[['label', 'polarity', 't_samples']])
    triggered_neural = triggered_neural.reorder_levels(
        ['label', 'polarity', 't_samples']).sort_index()

    # The columns are interdigitated samples and channels
    triggered_neural.columns = pandas.MultiIndex.from_product([
        pandas.Index(range(abr_start_sample, abr_stop_sample), name='timepoint'),
        pandas.Index(neural_channel_numbers, name='channel')
        ])
    
    # Rename the channels meaningfully
    triggered_neural = triggered_neural.rename(columns={
        0: this_recording.loc['ch0_config'], 
        2: this_recording.loc['ch2_config'],
        4: this_recording.loc['ch4_config']
        }, level='channel')
    
    # Drop NN if any
    triggered_neural = triggered_neural.drop(
        'NN', level='channel', axis=1, errors='ignore')
    
    # Stack channel
    triggered_neural = triggered_neural.stack('channel', future_stack=True)
    

    ## Store
    click_params_l.append(click_params)
    triggered_ad_l.append(triggered_ad)
    triggered_neural_l.append(triggered_neural)
    keys_l.append((date, mouse, recording))


## Concat
big_triggered_ad = pandas.concat(
    triggered_ad_l, keys=keys_l, names=['date', 'mouse', 'recording'])
big_triggered_neural = pandas.concat(
    triggered_neural_l, keys=keys_l, names=['date', 'mouse', 'recording'])
big_click_params = pandas.concat(
    click_params_l, keys=keys_l, names=['date', 'mouse', 'recording']
    ).set_index('t_samples', append=True).droplevel(None).sort_index()


## Store
big_triggered_ad.to_pickle(
    os.path.join(output_directory, 'big_triggered_ad'))
big_triggered_neural.to_pickle(
    os.path.join(output_directory, 'big_triggered_neural'))
big_click_params.to_pickle(
    os.path.join(output_directory, 'big_click_params'))
