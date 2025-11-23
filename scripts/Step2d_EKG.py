## Plot stats about ECG
# TODO: correlate ECG size with ABR size
#
# Plots
#   PLOT_EKG_GRAND_MEAN
#   PLOT_EKG_SESSION_MEAN
#   PLOT_EKG_STATS

import os
import json
import numpy as np
import pandas
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
heartbeat_highpass_freq = 20
heartbeat_lowpass_freq = 500

# Recording params
sampling_rate = 16000 
neural_channel_numbers = [0, 2, 4]
audio_channel_number = 7


## Load previous results
# Load results of main1
recording_metadata = pandas.read_pickle(
    os.path.join(output_directory, 'recording_metadata'))

# Load results of Step2a1_align
# PowerRainbow2 looks a bit blunted and slow, but not totally out of the realm
big_heartbeat_info = pandas.read_pickle(
    os.path.join(output_directory, 'big_heartbeat_info'))
big_heartbeat_waveform = pandas.read_pickle(
    os.path.join(output_directory, 'big_heartbeat_waveform'))


## Aggregate
# Mean waveform over recordings within session
mean_by_session = big_heartbeat_waveform.groupby(
    [lev for lev in big_heartbeat_waveform.index.names if lev != 'recording']
    ).mean()

# Mean waveform over sessions within mouse
# TODO: check whether there is substantial variability within mouse over days
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
    grand_err = mean_by_mouse.groupby('timepoint').sem()

    # Plot grand mean by channel
    f, ax = plt.subplots(figsize=(3.5, 2.5))
    f.subplots_adjust(bottom=.24, left=.25, right=.95, top=.89)
    for channel in ['LV', 'RV', 'LR']:
        if channel == 'LV':
            color = 'b'
        elif channel == 'RV':
            color = 'r'
        else:
            color = 'k'
            
        ax.plot(
            grand_mean.index.values / sampling_rate * 1000,
            grand_mean[channel].values * 1e6,
            color=color
            )

        ax.fill_between(
            x=grand_mean.index.values / sampling_rate * 1000,
            y1=(grand_mean[channel].values - grand_err[channel]) * 1e6,
            y2=(grand_mean[channel].values + grand_err[channel]) * 1e6,
            color=color,
            lw=0,
            alpha=.5
            )
    
    # Pretty
    ax.set_xlabel('time (ms)')
    ax.set_ylabel(f'ECG ({MU}V)')
    ax.set_xlim((-15, 15))
    ax.set_xticks((-10, 0, 10))
    ax.set_ylim((-75, 150))
    ax.set_yticks((-75, 0, 75, 150))
    my.plot.despine(ax)

    # Legend
    f.text(.85, .9, 'LV', color='b', ha='center', va='center')
    f.text(.85, .82, 'RV', color='r', ha='center', va='center')
    f.text(.85, .74, 'LR', color='k', ha='center', va='center')

    # Savefig
    f.savefig('figures/PLOT_EKG_GRAND_MEAN.svg')
    f.savefig('figures/PLOT_EKG_GRAND_MEAN.png', dpi=300)


    ## Stats
    n_mice = len(mean_by_mouse.index.get_level_values('mouse').unique())
    n_sessions = len(mean_by_session.groupby(['date', 'mouse']).size())
    
    stats_filename = 'figures/STATS__PLOT_EKG_GRAND_MEAN'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write(f'n = {n_sessions} sessions from {n_mice} mice\n')
        fi.write('error bars: SEM\n')
    
    # Echo
    with open(stats_filename) as fi:
        print(''.join(fi.readlines()))

if PLOT_EKG_BY_MOUSE:
    # Plot LR by mouse (this seems to be the largest and most consistent)
    f, ax = plt.subplots(figsize=(3.5, 2.5))
    f.subplots_adjust(bottom=.24, left=.25, right=.95, top=.89)

    # Slice LR
    LR_mean = mean_by_mouse.loc[:, 'LR'].unstack('mouse')
    
    # Slice temporally
    LR_mean = LR_mean.loc[-240:240].copy()

    ax.plot(
        LR_mean.index.values / sampling_rate * 1000,
        LR_mean * 1e6,
        color='k', alpha=.5, lw=1, clip_on=False)

    # Pretty
    ax.set_xlabel('time (ms)')
    ax.set_ylabel(f'ECG ({MU}V)')
    ax.set_xlim((-15, 15))
    ax.set_xticks((-10, 0, 10))
    ax.set_ylim((-100, 200))
    ax.set_yticks((-100, 0, 100, 200))
    my.plot.despine(ax)

    # Legend
    f.text(.85, .9, 'LR', color='k', ha='center', va='center')
    
    # Savefig
    f.savefig('figures/PLOT_EKG_BY_MOUSE.svg')
    f.savefig('figures/PLOT_EKG_BY_MOUSE.png', dpi=300)

    #~ # LV
    #~ f, ax = my.plot.figure_1x1_standard()
    #~ LR_mean = mean_by_mouse.loc[:, 'LV'].unstack('mouse')

    #~ ax.plot(
        #~ LR_mean.index.values / sampling_rate * 1000,
        #~ LR_mean * 1e6,
        #~ color='k', alpha=.1, lw=1)
    #~ ax.set_xlabel('time (ms)')
    #~ ax.set_xlim((-15, 15))
    #~ ax.set_ylabel('EKG (uV)')
    #~ my.plot.despine(ax)

    #~ # RV
    #~ f, ax = my.plot.figure_1x1_standard()
    #~ LR_mean = mean_by_mouse.loc[:, 'RV'].unstack('mouse')

    #~ ax.plot(
        #~ LR_mean.index.values / sampling_rate * 1000,
        #~ LR_mean * 1e6,
        #~ color='k', alpha=.1, lw=1)
    #~ ax.set_xlabel('time (ms)')
    #~ ax.set_xlim((-15, 15))
    #~ ax.set_ylabel('EKG (uV)')
    #~ my.plot.despine(ax)    

if PLOT_EKG_STATS:
    # Histogram height, prominence, width, and IBI by session
    # Height: mode at 72, range 50-140, long tail to 210, min 50
    # Prominence: mode at 100, range 60-150, long tail to 250, min 55
    # Width: bimodal; mode at 40, dip at 48, mode at 56, long tail to 68
    # IBI: range 250-375, mode 275, min 200, max 500
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

