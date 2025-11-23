## Make PSD plots
# Load PSDs and aggregate them
# There are some mice with higher 60 Hz or other types of noise, but
# they aren't extreme outliers
#
# TODO: check these PSDs by sex and hearing loss condition
#
# Plots
#   PSD_BY_CHANNEL and STATS__PSD_BY_CHANNEL

import os
import json
import numpy as np
import pandas
import my.plot
import matplotlib.pyplot as plt


## Plotting
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
# Recording params
# TODO: store in recording_metadata?
sampling_rate = 16000 
neural_channel_numbers = [0, 2, 4]


## Load previous results
# Load results of Step1
recording_metadata = pandas.read_pickle(
    os.path.join(output_directory, 'recording_metadata'))

# Load results of Step2
big_Pxx = pandas.read_pickle(
    os.path.join(output_directory, 'big_Pxx'))


## Compute average PSD
# Convert to db re 1 uV**2
topl = 10 * np.log10(big_Pxx * 1e12)

# Get freq on columns, and channels on index
topl = topl.unstack('freq').stack('channel', future_stack=True)

# Drop the nyquist frequency
topl = topl.iloc[:, :-1]

# Sample size
n_recordings = topl.groupby('channel').size().unique().item()
n_mice = len(topl.groupby('mouse').size())

# Aggregate within mouse first
# TODO: drop post-HL (and maybe all after 1st session per mouse)
by_mouse = topl.groupby(['mouse', 'channel']).mean()

# Aggregate across mice
topl_mu = by_mouse.groupby('channel').mean()
topl_err = by_mouse.groupby('channel').sem()


## Plot
# Figure handles
f, ax = my.plot.figure_1x1_standard()
f.subplots_adjust(left=.35, right=.85)

# Plot each channel
# The three channels are similar, except LR has less ~1 Hz and more ~100 Hz
#   (ie, LR cancels shared low-frequency noise and emphasizes ECG/ABR)
# Also, LV is generally above RV
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
# https://stackoverflow.com/questions/21226868/superscript-in-python-plots
ax.set_ylabel(
    f'power spectral density\n(dB re 1 {MU}'
    '$\mathregular{V^2}$/Hz)')
ax.set_yticks((-40, -20, 0, 20, 40))
ax.set_ylim((-40, 40))

# 1/f line
ax.plot([1, 1000], [20, -40], 'k--', lw=.75)
#~ ax.text(10, -20, '1/f')

# Label bands
# ABR - 300-3000
# ECG - 20-500
ax.plot([20, 500], [30, 30], 'k-')
ax.text(100, 31, 'ECG', ha='center', va='bottom')
ax.plot([300, 3000], [25, 25], 'k-')
ax.text(900, 23, 'ABR', ha='center', va='top')

# pretty
my.plot.despine(ax)

# legend
f.text(.9, .9, 'LV', ha='center', va='center', color='b')
f.text(.9, .82, 'RV', ha='center', va='center', color='r')
f.text(.9, .74, 'LR', ha='center', va='center', color='k')

# Save figure
f.savefig('figures/PSD_BY_CHANNEL.svg')
f.savefig('figures/PSD_BY_CHANNEL.png', dpi=300)

# Stats
stats_filename = 'figures/STATS__PSD_BY_CHANNEL'
with open(stats_filename, 'w') as fi:
    fi.write(stats_filename + '\n')
    fi.write(f'n = {n_recordings} recordings from n = {n_mice} mice\n')
    fi.write('aggregated within mouse and then across mice\n')
    fi.write('error bars: SEM over mice\n')

# Echo
with open(stats_filename) as fi:
    print(''.join(fi.readlines()))   

# Show
plt.show()

## Store
big_Pxx.to_pickle(os.path.join(output_directory, 'big_Pxx'))