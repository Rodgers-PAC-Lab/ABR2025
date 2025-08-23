## Make PSD plots
# This script takes a while - 12 minutes or so
#
# Plots
#   PSD_BY_CHANNEL and STATS__PSD_BY_CHANNEL

import os
import datetime
import glob
import json
import scipy.signal
import numpy as np
import pandas
import paclab
from paclab import abr
import my.plot
import matplotlib.pyplot as plt
import tqdm


## Plotting
my.plot.manuscript_defaults()
my.plot.font_embed()


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
# Recording params
# TODO: store in recording_metadata?
sampling_rate = 16000 
neural_channel_numbers = [0, 2, 4]


## Load data from each recording
Pxx_df_l = []
keys_l = []

# Iterate over recordings
for date, mouse, recording in tqdm.tqdm(recording_metadata.index):
    
    # TODO: mark these as exclude
    if date == datetime.date(2025, 3, 10) and mouse == 'Ketchup_208' and recording == 9:
        continue
    if date == datetime.date(2025, 3, 10) and mouse == 'Ketchup_209' and recording == 5:
        continue
    
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
    neural_data_V = data[:, neural_channel_numbers]


    ## Check for glitches
    # TODO: do this earlier, in data loading
    # The maximum voltage I ever see in real data is ~0.1 V, and that's only
    # when there's a substantial DC offset. The demeaned absmax is like ~1 mV.
    assert np.abs(neural_data_V).max() < 0.3
    
    
    ## Label the neural data by channel name
    # Get the channel names
    # These must match neural_channel_numbers above
    neural_channel_names = [
        this_recording.loc['ch0_config'],
        this_recording.loc['ch2_config'],
        this_recording.loc['ch4_config'],
        ]

    # DataFrame labeled by channel
    neural_data_df = pandas.DataFrame(
        neural_data_V, columns=neural_channel_names)

    # Drop NN
    neural_data_df = neural_data_df.drop('NN', axis=1, errors='ignore')


    ## Run PSD on each column
    Pxx_l = []
    for col in neural_data_df.values.T:
        # Data is in V
        Pxx, freqs = paclab.misc.psd(col, NFFT=16384, Fs=sampling_rate)
        Pxx_l.append(Pxx)

    # DataFrame
    Pxx_df = pandas.DataFrame(
        np.transpose(Pxx_l),
        columns=neural_data_df.columns, index=freqs)
    Pxx_df.index.name = 'freq'
    Pxx_df.columns.name = 'channel'


    ## Store
    Pxx_df_l.append(Pxx_df)
    keys_l.append((date, mouse, recording))


## Concat
big_Pxx = pandas.concat(
    Pxx_df_l, keys=keys_l,
    names=['date', 'mouse', 'recording'])
    

## Plot
# Convert to db re 1 uV
topl = 10 * np.log10(big_Pxx * 1e12)

# Get freq on columns, and channels on index
topl = topl.unstack('freq').stack('channel', future_stack=True)

# Drop the nyquist frequency
topl = topl.iloc[:, :-1]

# Groupby channel
topl_mu = topl.groupby('channel').mean()
topl_err = topl.groupby('channel').std()
n_recordings = topl.groupby('channel').size().unique().item()

# Figure handles
f, ax = my.plot.figure_1x1_standard()

# Plot each channel
# The three channels are similar, except LR has less ~1 Hz and more ~100 Hz
for channel in ['LV', 'RV', 'LR']:
    if channel == 'LV':
        color = 'b'
    elif channel == 'RV':
        color = 'r'
    elif channel == 'LR':
        color = 'k'

    line, = ax.plot(topl_mu.loc[channel], lw=1, label=channel, color=color)
    ax.fill_between(
        x=topl_mu.loc[channel].index,
        y1=topl_mu.loc[channel] - topl_err.loc[channel],
        y2=topl_mu.loc[channel] + topl_err.loc[channel],
        color=color, alpha=.5, lw=0)

# log x
ax.set_xscale('log')
ax.set_xlabel('frequency (Hz)')
ax.set_xticks((1e0, 1e1, 1e2, 1e3, 1e4))
ax.set_xlim((1e0, 1e4))

# y axis
ax.set_ylabel('power spectral density\n(uV/Hz)')
ax.set_yticks((-40, -20, 0, 20, 40))
ax.set_yticklabels((.01, .1, 1, 10, 100))
ax.set_ylim((-40, 40))

# pretty
my.plot.despine(ax)

# legend
f.text(.9, .9, 'LR', ha='center', va='center', color='k')
f.text(.9, .82, 'LV', ha='center', va='center', color='b')
f.text(.9, .74, 'RV', ha='center', va='center', color='r')

# Save figure
f.savefig('figures/PSD_BY_CHANNEL.svg')
f.savefig('figures/PSD_BY_CHANNEL.png', dpi=300)

# Stats
with open('figures/STATS__PSD_BY_CHANNEL', 'w') as fi:
    fi.write('n = {n_recordings} recordings\n')
    fi.write('error bars: standard deviation over recordings\n')