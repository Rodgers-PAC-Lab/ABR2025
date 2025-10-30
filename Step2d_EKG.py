## Plot stats about ECG
# TODO: correlate ECG size with ABR size
#
# Plots
#   PLOT_EKG_GRAND_MEAN
#   PLOT_EKG_SESSION_MEAN
#   PLOT_EKG_STATS

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


## Plotting defaults
my.plot.manuscript_defaults()
my.plot.font_embed()


## Paths
# Load the required file filepaths.json (see README)
with open('filepaths.json') as fi:
    paths = json.load(fi)

# Parse into paths to raw data and output directory
raw_data_directory = paths['raw_data_directory']
output_directory = paths['output_directory']


## Load previous results
# Load results of main1
recording_metadata = pandas.read_pickle(
    os.path.join(output_directory, 'recording_metadata'))

# Load results of Step2a1_align
big_heartbeat_info = pandas.read_pickle(
    os.path.join(output_directory, 'big_heartbeat_info'))
big_heartbeat_waveform = pandas.read_pickle(
    os.path.join(output_directory, 'big_heartbeat_waveform'))

# Drop ToyCar1
# PowerRainbow2 looks a bit blunted and slow, but not totally out of the realm
big_heartbeat_info = big_heartbeat_info.drop('ToyCar1', level='mouse')
big_heartbeat_waveform = big_heartbeat_waveform.drop('ToyCar1', level='mouse')


## Params
heartbeat_highpass_freq = 20
heartbeat_lowpass_freq = 500

# Recording params
# TODO: store in recording_metadata?
sampling_rate = 16000 
neural_channel_numbers = [0, 2, 4]
audio_channel_number = 7


## Aggregate
# Mean waveform over recordings within session
mean_by_session = big_heartbeat_waveform.groupby(
    [lev for lev in big_heartbeat_waveform.index.names if lev != 'recording']
    ).mean()

# Mean waveform over sessions within mouse
# TODO: check whether there is substantial variability within mouse
mean_by_mouse = mean_by_session.groupby(
    [lev for lev in mean_by_session.index.names if lev != 'date']
    ).mean()


## Summarize waveform shape by session
stats_by_session = big_heartbeat_info.groupby(
    ['date', 'mouse', 'recording']).median()[
    ['peak_heights', 'prominences', 'widths']]

# Add on inter-beat interval
stats_by_session['IBI'] = big_heartbeat_info['sample'].diff().dropna().groupby(
    ['date', 'mouse', 'recording']).median()


## Plots
PLOT_EKG_GRAND_MEAN = True
PLOT_EKG_BY_MOUSE = True
PLOT_EKG_STATS = True

if PLOT_EKG_GRAND_MEAN:
    # Mean over mouse
    grand_mean = mean_by_mouse.groupby('timepoint').mean()

    # Plot grand mean by channel
    f, ax = my.plot.figure_1x1_standard()
    ax.plot(
        grand_mean.index.values / sampling_rate * 1000,
        grand_mean.values * 1e6,
        )
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('EKG (uV)')
    #~ ax.legend(['LR', 'LV', 'RV'])
    ax.set_xlim((-15, 15))
    my.plot.despine(ax)
    f.savefig('figures/PLOT_EKG_GRAND_MEAN.svg')
    f.savefig('figures/PLOT_EKG_GRAND_MEAN.png', dpi=300)

if PLOT_EKG_BY_MOUSE:
    # Plot LR by mouse (each channel seems equally variable)
    f, ax = my.plot.figure_1x1_standard()
    LR_mean = mean_by_mouse.loc[:, 'LR'].unstack('mouse')

    ax.plot(
        LR_mean.index.values / sampling_rate * 1000,
        LR_mean * 1e6,
        color='k', alpha=.1, lw=1)
    ax.set_xlabel('time (ms)')
    ax.set_xlim((-15, 15))
    ax.set_ylabel('EKG (uV)')
    my.plot.despine(ax)
    f.savefig('figures/PLOT_EKG_BY_MOUSE.svg')
    f.savefig('figures/PLOT_EKG_BY_MOUSE.png', dpi=300)

if PLOT_EKG_STATS:
    # Histogram height, prominence, width, and IBI by session
    # Height: mode at 66, range 50-140, long tail to 210, min 50
    # Prominence: mode at 90, range 60-150, long tail to 250, min 55
    # Width: bimodal; mode at 40, dip at 48, mode at 56, long tail to 68
    # IBI: range 4000-6000, mode 5000, min 3200, max 6500
    f, axa = plt.subplots(1, 4, figsize=(12, 3))
    axa[0].hist(stats_by_session['peak_heights'] * 1e6, bins=21)
    axa[0].set_xlabel('height (uV)')
    axa[1].hist(stats_by_session['prominences'] * 1e6, bins=21)
    axa[1].set_xlabel('prominence (uV)')
    axa[2].hist(stats_by_session['widths'], bins=21)
    axa[2].set_xlabel('width (samples)')
    axa[3].hist(stats_by_session['IBI'] / sampling_rate * 1000, bins=21)
    axa[3].set_xlabel('IBI (ms)')
    f.tight_layout()
    f.savefig('figures/PLOT_EKG_STATS.svg')
    f.savefig('figures/PLOT_EKG_STATS.png', dpi=300)

    # These narrow and wide ones seem to have fairly similar shapes, just stretched
    #~ mask = by_session['widths'] < 47

plt.show()

