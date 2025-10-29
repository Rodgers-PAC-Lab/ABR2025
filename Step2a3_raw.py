## Plot raw data 
# 2025-02-28 Cedric rec 2? PizzaSlice7?

import os
import datetime
import glob
import json
import scipy.signal
import numpy as np
import pandas
import paclab.abr
from paclab import abr
import my.plot
import matplotlib.pyplot as plt


## Plotting defaults
my.plot.manuscript_defaults()
my.plot.font_embed()
MU = chr(956)

# Squelch this warning
#~ pandas.set_option('future.no_silent_downcasting', True)


## Paths
# Load the required file filepaths.json (see README)
with open('filepaths.json') as fi:
    paths = json.load(fi)

# Parse into paths to raw data and output directory
raw_data_directory = paths['raw_data_directory']
output_directory = paths['output_directory']


## Params
# Specify which date and experimenter to process
abr_date = datetime.date(year=2025, month=2, day=28)
mouse = 'PizzaSlice7'
experimenter = 'Cedric'
recording = 2

# Form experimenter dir
date_s = abr_date.strftime('%Y-%m-%d')
experimenter_dir = os.path.join(raw_data_directory, date_s, experimenter)

# ABR params
abr_start_sample = -40
abr_stop_sample = 120
abr_highpass_freq = 300
abr_lowpass_freq = 3000
audio_extract_win_samples = 10
speaker_channel = 7
neural_channel_numbers = [0, 2, 4]
sampling_rate = 16000



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
    
    
## Load results of main1
recording_metadata = pandas.read_pickle(os.path.join(output_directory, 'recording_metadata'))

# Drop those with 'include' == False
recording_metadata = recording_metadata[recording_metadata['include'] == True]


## Set up click categories
# Get the recording info
this_recording = recording_metadata.loc[abr_date].loc[mouse].loc[recording]


## Load raw data in volts
# Get the filename
recording_folder = os.path.normpath(
    os.path.join(raw_data_directory, this_recording['short_datafile']))

# Load the data
data = abr.loading.load_recording(recording_folder)
data = data['data']

# Parse into neural and speaker data
speaker_signal_V = data[:, speaker_channel]
neural_data_V = data[:, neural_channel_numbers]


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


## Check for glitches
# The maximum voltage I ever see in real data is ~0.1 V, and that's only
# when there's a substantial DC offset. The demeaned absmax is like ~1 mV.
assert np.abs(neural_data_V).max() < 0.3


## Barely highpass neural data just for visualizing raw data
nyquist_freq = sampling_rate / 2
ahi, bhi = scipy.signal.butter(
    2, 1 / nyquist_freq, 
    btype='high')
neural_data_V = scipy.signal.filtfilt(ahi, bhi, neural_data_V, axis=0)


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

# Put time on index
neural_data_df.index = np.arange(len(neural_data_df)) / sampling_rate



## Bandpass in the ABR band
# ABR band params
nyquist_freq = sampling_rate / 2
ahi, bhi = scipy.signal.butter(
    2, (
    abr_highpass_freq / nyquist_freq, 
    abr_lowpass_freq / nyquist_freq), 
    btype='bandpass')

neural_data_hp = scipy.signal.filtfilt(ahi, bhi, neural_data_df, axis=0)
neural_data_hp_df = pandas.DataFrame(
    neural_data_hp, index=neural_data_df.index, columns=neural_data_df.columns)


## Plot
f, axa = plt.subplots(7, 1, sharex=True, figsize=(10, 6))
f.subplots_adjust(left=.05, right=.98, bottom=.1, top=.95)

ds_ratio = 10
channel_order = ['LV', 'RV', 'LR']
channel2color = {'LR': 'purple', 'LV': 'b', 'RV': 'r'}

# Plot raw data on each channel in the first three axes
for n_channel, channel in enumerate(channel_order):
    # Get color
    color = channel2color[channel]
    
    # Get ax
    ax = axa[1 + n_channel]
    
    # Plot raw data (in microvolts)
    ax.plot(neural_data_df.loc[:, channel].iloc[::ds_ratio] * 1e6, color=color, lw=.75)
    ax.set_ylim((-150, 150))
    ax.set_yticks((-150, 0, 150))

# Plot highpass data on each channel in the last three axes
for n_channel, channel in enumerate(channel_order):
    # Get color
    color = channel2color[channel]
    
    # Get ax
    ax = axa[4 + n_channel]
    
    # Plot raw data (in microvolts)
    ax.plot(neural_data_hp_df.loc[:, channel].iloc[::ds_ratio] * 1e6, color=color, lw=.75)
    ax.set_ylim((-20, 20))
    ax.set_yticks((-20, 0, 20))

# Plot click params
to_raster = []
for label in amplitude_labels:
    # Slice
    this_click_times = click_params.loc[
        click_params['label'] == label, 't_samples'].values / sampling_rate
    
    # Store
    to_raster.append(this_click_times)
to_raster = click_params['t_samples'] / sampling_rate
axa[0].eventplot(to_raster, color='k')
#~ axa[0].set_yticks(range(len(amplitude_labels)))
#~ axa[0].set_yticklabels(amplitude_labels)

# Slice out a reasonable period of time
axa[6].set_xlim((300, 303))
axa[6].set_xlabel('time (s)')

# Pretty
for ax in axa[:-1]:
    my.plot.despine(ax, which=('bottom', 'right', 'top'))
my.plot.despine(axa[-1])

plt.show()