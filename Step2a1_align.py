## Identify and categorize clicks, and slice neural and audio data around them
# This script takes a while - 12 minutes or so
# Everything that involves loading the raw data for all sesssions should be
# done in this script, to avoid having to load multiple times.
#
# TODO: pull out longer times to see what happens in a bigger window
#
# Workflow:
# * Loads data and checks for glitches
# * Calculates PSD
# * Extracts heartbeats
# * Extract and categorizes clicks
# * Slices neural and audio data around clicks
# * Concat everything across sessions and stores
#
# This script is the one that assigns amplitude labels.
#
# Writes out the following in the output directory
#   big_triggered_ad - audio data triggered on clicks
#   big_triggered_neural - neural data triggered on clicks
#   big_click_params - click metadata
#   big_Pxx - PSDs
#   big_heartbeat_info - time of each beat
#   big_heartbeat_waveform - mean EKG waveform of each recording


import os
import datetime
import glob
import json
import scipy.signal
import numpy as np
import pandas
from paclab import abr
import paclab
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


## Load results of main1
recording_metadata = pandas.read_pickle(
    os.path.join(output_directory, 'recording_metadata'))


## Params
abr_start_sample = -40
abr_stop_sample = 120
abr_highpass_freq = 300
abr_lowpass_freq = 3000

# ECG params
heartbeat_highpass_freq = 20
heartbeat_lowpass_freq = 500

# Recording params
# TODO: store in recording_metadata?
sampling_rate = 16000 
neural_channel_numbers = [0, 2, 4]
audio_channel_number = 7


## Set up the click categories
# In this dataset, all clicks should have this amplitude
expected_amplitude = [
    0.01,  0.0063,  0.004, 0.0025, 0.0016, 0.001, 0.00063, 
    0.0004, 0.00025, 0.00016, 0.0001, 6.3e-05, 4e-05,
    ]

# Convert the autopilot amplitudes to voltages
# This 1.34 is empirically determined to align autopilot with measured voltage
log10_voltage = np.sort(np.log10(expected_amplitude) + 1.34)

# SPL as measured with the ultrasonic microphone
# Note: Subtract 30 dB here (average over 50 ms instead of 0.05 ms)
amplitude_labels = np.linspace(45, 93, 13).astype(int) - 30 

# Convert the voltages to cuts
amplitude_cuts = (log10_voltage[1:] + log10_voltage[:-1]) / 2

# Add a first and last amplitude cut
diff_cut = np.mean(np.diff(amplitude_cuts))
amplitude_cuts = np.concatenate([
    [amplitude_cuts[0] - diff_cut],
    amplitude_cuts,
    [amplitude_cuts[-1] + diff_cut],
    ])


## Load data from each recording
click_params_l = []
triggered_ad_l = []
triggered_neural_l = []
heartbeats_l = []
heartbeats_waveform_l = []
Pxx_df_l = []
keys_l = []

# Iterate over recordings
for date, mouse, recording in tqdm.tqdm(recording_metadata.index):
    
    # Get the recording info
    this_recording = recording_metadata.loc[date].loc[mouse].loc[recording]

    
    ## Load raw data in volts
    # Get the filename
    recording_folder = os.path.normpath(
        os.path.join(raw_data_directory, this_recording['short_datafile']))
    
    # Load the data
    data = abr.loading.load_recording(recording_folder)
    data = data['data']
    
    # Parse into neural and speaker data
    speaker_signal_V = data[:, audio_channel_number]
    neural_data_V = data[:, neural_channel_numbers]


    ## Check for glitches
    # The maximum voltage I ever see in real data is ~0.1 V, and that's only
    # when there's a substantial DC offset. The demeaned absmax is like ~1 mV.
    assert np.abs(neural_data_V).max() < 0.3
    
    
    ## Label the neural data by channel name
    # DataFrame labeled by channel number
    neural_data_df = pandas.DataFrame(
        neural_data_V, 
        columns=neural_channel_numbers)

    # Rename the channels meaningfully
    # This fixes the inconsistent channel order for Pineapple_197 on 2025-02-12
    neural_data_df.columns.name = 'channel'
    neural_data_df = neural_data_df.rename(columns={
        0: this_recording.loc['ch0_config'], 
        2: this_recording.loc['ch2_config'],
        4: this_recording.loc['ch4_config']
        }, level='channel')

    # Check no NN or other channels
    assert neural_data_df.columns.isin(['LR', 'LV', 'RV']).all()
    neural_data_df = neural_data_df.sort_index(axis=1)


    ## Bandpass heartbeat
    # Define heartbeat filter
    nyquist_freq = sampling_rate / 2
    ahi, bhi = scipy.signal.butter(
        2, (
        heartbeat_highpass_freq / nyquist_freq, 
        heartbeat_lowpass_freq / nyquist_freq), 
        btype='bandpass')

    # Bandpass neural data into ECG band
    # Becomes an array at this point
    ekg_signal = scipy.signal.filtfilt(ahi, bhi, neural_data_df, axis=0)

    # Restore DataFrame metadata
    ekg_signal = pandas.DataFrame(
        ekg_signal, columns=neural_data_df.columns, index=neural_data_df.index)

    # Find heartbeats, indexed into ekg_signal
    # 
    # SHAPE OF EKG
    # Use the first channel (LR), which is biggest
    # The peak is always positive on LR and LV, and negative on RV
    # LV and RV are nearly opposites, so LR is about double
    # The central peak is maybe 5 ms wide and the whole thing is maybe 25 ms
    #
    # HEIGHT and PROMINENCE
    # The lowest SNR recording is Cat_229 on 2025-05-15, esp recording 1
    # On this recording, the heights are 40-50 uV and prominences 50-60
    # Prominences are larger because of the dip around the peak
    # Breathing artefacts can get up to 25
    # There is a kind of bimodality in the raw EKG signal with a dip around 27
    # A threshold of 35 seems appropriate, we probably would prefer to lose
    # a few heartbeats than to pick up too much noise
    #
    # INTER-BEAT INTERVAL ("DISTANCE")
    # The inter-beat-interval is 3000-7500 samples (187-469 ms)
    # Enforce a minimum of 1000 samples
    #
    # WIDTH
    # The main peak is about 5 ms wide (80 samples), so set wlen to 150
    # The 'width' criterion is actually a half-width if rel_height is 0.5
    # For some reason there's another, narrower mode in widths, around half-width 35
    # So use a wide range of (10, 100) on width
    ekg_threshold = 35e-6 # V
    peak_times, peak_props = scipy.signal.find_peaks(
        ekg_signal.loc[:, 'LR'], 
        height=ekg_threshold, 
        distance=1000,
        prominence=ekg_threshold,
        wlen=150,
        width=(10, 150),
        rel_height=0.5,
        )    
    
    # DataFrame
    heartbeats = pandas.DataFrame.from_dict(peak_props)
    heartbeats['sample'] = peak_times

    # Exclude too close to edge
    slice_halfwidth = 400
    heartbeats = heartbeats[
        (heartbeats['sample'] >= slice_halfwidth) &
        (heartbeats['sample'] < len(ekg_signal) - slice_halfwidth)
        ].reset_index(drop=True)
    heartbeats.index.name = 'beat'

    # Right now this fails on ToyCar1 2025-7-7 rec 8 
    # Bring this check back after we drop that session
    #~ # Error check
    #~ if len(heartbeats) < 10:
        #~ 1/0
    
    # Extract (n_trials, n_timepoints, 3)
    sliced_l = []
    for peak in heartbeats['sample']:
        # Slice
        sliced = ekg_signal.iloc[
            peak - slice_halfwidth:peak + slice_halfwidth
            ]
        sliced.index = pandas.Index(
            np.arange(-slice_halfwidth, slice_halfwidth, dtype=int), 
            name='timepoint')
        
        # Store
        sliced_l.append(sliced)
    
    # Concat
    sliced_beats_df = pandas.concat(sliced_l, keys=heartbeats.index)

    # Mean waveform over beats within recording
    mean_beat = sliced_beats_df.groupby('timepoint').mean()
    
    # Store
    heartbeats_l.append(heartbeats)
    heartbeats_waveform_l.append(mean_beat)
    

    ## Run PSD on each column
    Pxx_l = []
    for col in neural_data_df.values.T:
        # Data is in V
        # Result is in V**2/Hz (if scale_by_freq == True, which is default)
        # Check: converting to uV yields a PSD that is 1e12 greater
        Pxx, freqs = paclab.misc.psd(col, NFFT=16384, Fs=sampling_rate)
        Pxx_l.append(Pxx)

    # DataFrame
    Pxx_df = pandas.DataFrame(
        np.transpose(Pxx_l),
        columns=neural_data_df.columns, index=freqs)
    Pxx_df.index.name = 'freq'
    Pxx_df.columns.name = 'channel'

    # Store
    Pxx_df_l.append(Pxx_df)
    

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
    # ABR band params
    nyquist_freq = sampling_rate / 2
    ahi, bhi = scipy.signal.butter(
        2, (
        abr_highpass_freq / nyquist_freq, 
        abr_lowpass_freq / nyquist_freq), 
        btype='bandpass')
    
    # Bandpass neural data into ABR band
    # Becomes an array at this point
    neural_data_hp = scipy.signal.filtfilt(ahi, bhi, neural_data_df, axis=0)

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
    # The channels are ordered in the same way as neural_data_df.columns
    triggered_neural.columns = pandas.MultiIndex.from_product([
        pandas.Index(range(abr_start_sample, abr_stop_sample), name='timepoint'),
        neural_data_df.columns,
        ])
    
    # Stack channel
    triggered_neural = triggered_neural.stack('channel', future_stack=True)


    ## Store
    click_params_l.append(click_params)
    triggered_ad_l.append(triggered_ad)
    triggered_neural_l.append(triggered_neural)
    keys_l.append((date, mouse, recording))


## Concat
big_triggered_ad = pandas.concat(
    triggered_ad_l, 
    keys=keys_l, names=['date', 'mouse', 'recording'])
big_triggered_neural = pandas.concat(
    triggered_neural_l, 
    keys=keys_l, names=['date', 'mouse', 'recording'])
big_click_params = pandas.concat(
    click_params_l, 
    keys=keys_l, names=['date', 'mouse', 'recording']
    ).set_index('t_samples', append=True).droplevel(None).sort_index()
big_Pxx = pandas.concat(
    Pxx_df_l, 
    keys=keys_l, names=['date', 'mouse', 'recording'])
big_heartbeat_info = pandas.concat(
    heartbeats_l, 
    keys=keys_l, names=['date', 'mouse', 'recording'])
big_heartbeat_waveform = pandas.concat(
    heartbeats_waveform_l, 
    keys=keys_l, names=['date', 'mouse', 'recording'])
    

## Store
big_triggered_ad.to_pickle(
    os.path.join(output_directory, 'big_triggered_ad'))
big_triggered_neural.to_pickle(
    os.path.join(output_directory, 'big_triggered_neural'))
big_click_params.to_pickle(
    os.path.join(output_directory, 'big_click_params'))
big_Pxx.to_pickle(
    os.path.join(output_directory, 'big_Pxx'))
big_heartbeat_info.to_pickle(
    os.path.join(output_directory, 'big_heartbeat_info'))
big_heartbeat_waveform.to_pickle(
    os.path.join(output_directory, 'big_heartbeat_waveform'))