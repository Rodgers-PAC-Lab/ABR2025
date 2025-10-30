## Example plot of raw data
# Plots:
#   PLOT_RAW_DATA

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


## Paths
# Load the required file filepaths.json (see README)
with open('filepaths.json') as fi:
    paths = json.load(fi)

# Parse into paths to raw data and output directory
raw_data_directory = paths['raw_data_directory']
output_directory = paths['output_directory']


## Params
# Specify which date and experimenter to process
# Previously: PizzaSlice7 rec2 on 2025-2-28
abr_date = datetime.date(year=2025, month=5, day=20)
mouse = 'Cat_229'
experimenter = 'rowan'
recording = 10

# Form experimenter dir
date_s = abr_date.strftime('%Y-%m-%d')
experimenter_dir = os.path.join(raw_data_directory, date_s, experimenter)

# ABR params
abr_start_sample = -40
abr_stop_sample = 120
abr_highpass_freq = 300
abr_lowpass_freq = 3000

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
    
    
## Load results of main1
recording_metadata = pandas.read_pickle(
    os.path.join(output_directory, 'recording_metadata'))


## Load raw data in volts
# Get the recording info
this_recording = recording_metadata.loc[abr_date].loc[mouse].loc[recording]

# Get the filename
recording_folder = os.path.normpath(
    os.path.join(raw_data_directory, this_recording['short_datafile']))

# Load the data
data = abr.loading.load_recording(recording_folder)
data = data['data']

# Parse into neural and speaker data
speaker_signal_V = data[:, audio_channel_number]
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
    2, 0.1 / nyquist_freq, 
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
f, axa = plt.subplots(
    2, 1, sharex=True, figsize=(7.5, 4.5), height_ratios=[4, 3])
f.subplots_adjust(left=.1, right=.98, bottom=.14, top=.92, hspace=.4)

# Timing params
t_start = 295.5
t_stop = 297.5
ds_ratio = 10

# Color and channel params
channel_order = ['LV', 'RV', 'LR'][::-1]
channel2color = {'LR': 'k', 'LV': 'b', 'RV': 'r'}


## Plot raw data on each channel in the first axis
for n_channel, channel in enumerate(channel_order):
    # Get color
    color = channel2color[channel]
    
    # Get ax
    ax = axa[0]
    
    # Plot raw data (in microvolts)
    ax.plot(
        n_channel * 300 + neural_data_df.loc[:, channel].iloc[::ds_ratio] * 1e6, 
        color=color, lw=.75)

# Pretty
ax.set_title('raw data')    
axa[0].set_ylim((-150, 1000))


## Plot highpass data on each channel in the last three axes
for n_channel, channel in enumerate(channel_order):
    # Get color
    color = channel2color[channel]
    
    # Get ax
    ax = axa[1]
    
    # Plot raw data (in microvolts)
    ax.plot(
        n_channel * 30 + neural_data_hp_df.loc[:, channel].iloc[::ds_ratio] * 1e6, 
        color=color, lw=.75)

# Pretty
axa[1].set_ylim((-20, 75))


## Plot stimulus bars
axa[0].plot(
    click_params['t_samples'] / sampling_rate, 
    [900] * len(click_params), 
    '|', color='k', ms=4)


## Set time axis
axa[1].set_title('bandpass filtered (ABR band)')
axa[1].set_xlim((t_start, t_stop))
axa[1].set_xlabel('time (s)')
axa[1].set_xticks([t_start, t_start + 1, t_start+2])
axa[1].set_xticklabels([0, 1, 2])


## Pretty
# Scale bar
legend_xval = t_start - .2
axa[0].plot([legend_xval, legend_xval], [100, 400], 'k-', clip_on=False)
axa[0].text(legend_xval, 250, f'300 {MU}V', ha='right', va='center', rotation=90)
axa[1].plot([legend_xval, legend_xval], [10, 40], 'k-', clip_on=False)
axa[1].text(legend_xval, 25, f'30 {MU}V', ha='right', va='center', rotation=90)

# Labels
label_xval = t_start - .1
axa[0].text(label_xval, 900, 'audio', color='k', ha='center', va='center')
axa[0].text(label_xval, 600, 'LV', color='b', ha='center', va='center')
axa[0].text(label_xval, 300, 'RV', color='r', ha='center', va='center')
axa[0].text(label_xval, 0, 'LR', color='k', ha='center', va='center')
axa[1].text(label_xval, 60, 'LV', color='b', ha='center', va='center')
axa[1].text(label_xval, 30, 'RV', color='r', ha='center', va='center')
axa[1].text(label_xval, 0, 'LR', color='k', ha='center', va='center')

# Despine
for ax in axa[:-1]:
    my.plot.despine(ax, which=('left', 'bottom', 'right', 'top'))
    ax.set_yticks([])
my.plot.despine(axa[-1], which=('left', 'right', 'top'))
axa[-1].set_yticks([])


## Savefig
f.savefig('figures/PLOT_RAW_DATA.svg')
f.savefig('figures/PLOT_RAW_DATA.png', dpi=300)

plt.show()