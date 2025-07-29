## Pull out EKG
# 

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


## Load results of main1
recording_metadata = pandas.read_pickle(
    os.path.join(output_directory, 'recording_metadata'))


## Params
heartbeat_highpass_freq = 20
heartbeat_lowpass_freq = 500

# Recording params
# TODO: store in recording_metadata?
sampling_rate = 16000 
neural_channel_numbers = [0, 2, 4]
audio_channel_number = 7


## Load data from each recording
click_params_l = []
triggered_ad_l = []
triggered_neural_l = []
keys_l = []

# Iterate over recordings
heartbeats_l = []
heartbeats_keys_l = []
heartbeats_sliced_l = []
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


    ## Bandpass heartbeat
    # Bandpass all neural channels
    nyquist_freq = sampling_rate / 2
    ahi, bhi = scipy.signal.butter(
        2, (
        heartbeat_highpass_freq / nyquist_freq, 
        heartbeat_lowpass_freq / nyquist_freq), 
        btype='bandpass')
    ekg_signal = scipy.signal.filtfilt(ahi, bhi, neural_data_V, axis=0)

    # Convert to uV
    ekg_signal = ekg_signal * 1e6

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
    ekg_threshold = 35 # uV
    peak_times, peak_props = scipy.signal.find_peaks(
        ekg_signal[:, 0], 
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

    # Error check
    if len(heartbeats) < 10:
        1/0
    
    # Extract (n_trials, n_timepoints, 3)
    sliced_arr = np.array([
        ekg_signal[peak - slice_halfwidth:peak + slice_halfwidth]
        for peak in heartbeats['sample']])
    
    # Squeeze into (n_trials, (n_timepoints * 3))
    sliced_arr = sliced_arr.reshape((len(sliced_arr), -1))
    
    # DataFrame it
    # Index is the same as the heartbeats
    sliced_df = pandas.DataFrame(sliced_arr, index=heartbeats.index)
    
    # Form the columns, taking into account the three channels
    level0 = pandas.Series(
        np.arange(-slice_halfwidth, slice_halfwidth, dtype=int), 
        name='timepoint')
    level1 = pandas.Series(neural_channel_numbers, name='channel')
    sliced_df.columns = pandas.MultiIndex.from_product([level0, level1])
    
    # Store
    heartbeats_l.append(heartbeats)
    heartbeats_keys_l.append((date, mouse, recording))
    heartbeats_sliced_l.append(sliced_df)

# Concat
heart_df = pandas.concat(
    heartbeats_l, keys=heartbeats_keys_l, 
    names=['date', 'mouse', 'recording'])
beats_df = pandas.concat(
    heartbeats_sliced_l, keys=heartbeats_keys_l, 
    names=['date', 'mouse', 'recording'])


## Mean waveform by session
mean_by_session = beats_df.groupby(['date', 'mouse', 'recording']).mean()


## Summarize waveform shape by session
stats_by_session = heart_df.groupby(['date', 'mouse', 'recording']).median()[
    ['peak_heights', 'prominences', 'widths']]

# Add on inter-beat interval
stats_by_session['IBI'] = heart_df['sample'].diff().dropna().groupby(
    ['date', 'mouse', 'recording']).median()


## Plots
PLOT_EKG_GRAND_MEAN = True
PLOT_EKG_SESSION_MEAN = True
PLOT_EKG_STATS = True

if PLOT_EKG_GRAND_MEAN:
    # Plot grand mean by channel
    f, ax = my.plot.figure_1x1_standard()
    grand_mean = mean_by_session.mean().unstack('channel')
    ax.plot(
        grand_mean.index.values / sampling_rate * 1000,
        grand_mean.values,
        )
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('EKG (uV)')
    ax.legend(['LR', 'LV', 'RV'])
    ax.set_xlim((-15, 15))
    my.plot.despine(ax)
    f.savefig('PLOT_EKG_GRAND_MEAN.svg')
    f.savefig('PLOT_EKG_GRAND_MEAN.png', dpi=300)

if PLOT_EKG_SESSION_MEAN:
    # Plot LR by session (each channel seems equally variable)
    f, ax = my.plot.figure_1x1_standard()
    LR_mean = mean_by_session.xs(0, level='channel', axis=1)
    LR_mean_by_session = LR_mean.groupby(['date', 'mouse', 'recording']).mean()
    ax.plot(
        LR_mean_by_session.columns.values / sampling_rate * 1000,
        LR_mean_by_session.T,
        color='k', alpha=.1, lw=1)
    ax.set_xlabel('time (ms)')
    ax.set_xlim((-15, 15))
    ax.set_ylabel('EKG (uV)')
    my.plot.despine(ax)
    f.savefig('PLOT_EKG_SESSION_MEAN.svg')
    f.savefig('PLOT_EKG_SESSION_MEAN.png', dpi=300)

if PLOT_EKG_STATS:
    # Histogram height, prominence, width, and IBI by session
    # Height: mode at 66, range 50-140, long tail to 210, min 50
    # Prominence: mode at 90, range 60-150, long tail to 250, min 55
    # Width: bimodal; mode at 40, dip at 48, mode at 56, long tail to 68
    # IBI: range 4000-6000, mode 5000, min 3200, max 6500
    f, axa = plt.subplots(1, 4, figsize=(12, 3))
    axa[0].hist(by_session['peak_heights'], bins=21)
    axa[0].set_xlabel('height (uV)')
    axa[1].hist(by_session['prominences'], bins=21)
    axa[1].set_xlabel('prominence (uV)')
    axa[2].hist(by_session['widths'], bins=21)
    axa[2].set_xlabel('width (samples)')
    axa[3].hist(by_session['IBI'] / sampling_rate * 1000, bins=21)
    axa[3].set_xlabel('IBI (ms)')
    f.tight_layout()
    f.savefig('PLOT_EKG_STATS.svg')
    f.savefig('PLOT_EKG_STATS.png', dpi=300)

    # These narrow and wide ones seem to have fairly similar shapes, just stretched
    #~ mask = by_session['widths'] < 47

plt.show()


## Save
# TODO: Correlate EKG size and evoked ABR size
heart_df.to_pickle(os.path.join(output_directory, 'EKG_heartbeats'))
stats_by_session.to_pickle(os.path.join(output_directory, 'EKG_stats'))
mean_by_session.to_pickle(os.path.join(output_directory, 'EKG_waveform'))