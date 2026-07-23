## Plot stats about ECG
# TODO: correlate ECG size with ABR size
#
# Plots
#   PLOT_EKG_GRAND_MEAN
#   PLOT_EKG_BY_MOUSE

import os
import json
import pandas
import my.plot
import matplotlib.pyplot as plt
import shared


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


## Load metadata
metadata = shared.load_metadata(raw_data_directory)

# Parse out
mouse_metadata = metadata['mouse_metadata'].copy()
recording_metadata = metadata['recording_metadata'].copy()
experiment_metadata = metadata['experiment_metadata'].copy()
    

## Load previous results
# Load results of Step2a1_align
big_heartbeat_waveform = pandas.read_parquet(
    os.path.join(output_directory, 'big_heartbeat_waveform'))

# Like elsewhere, keep only the first pre-HL experiment
big_heartbeat_waveform = my.misc.join_level_onto_index(
    big_heartbeat_waveform, experiment_metadata[['after_HL', 'n_experiment']])
big_heartbeat_waveform = big_heartbeat_waveform.xs(
    False, level='after_HL').xs(0, level='n_experiment')
    

## Aggregate
# Mean waveform over recordings within session
mean_by_session = big_heartbeat_waveform.groupby(
    [lev for lev in big_heartbeat_waveform.index.names if lev != 'recording']
    ).mean()

# Mean waveform over sessions within mouse
# Now that we select only one session per mouse, this no longer does anything
mean_by_mouse = mean_by_session.groupby(
    [lev for lev in mean_by_session.index.names if lev != 'date']
    ).mean()


## Plots
PLOT_EKG_GRAND_MEAN = True
PLOT_EKG_BY_MOUSE = True

if PLOT_EKG_GRAND_MEAN:
    # Mean over mouse
    grand_mean = mean_by_mouse.groupby('timepoint').mean()
    grand_err = mean_by_mouse.groupby('timepoint').sem()

    # Plot grand mean by channel
    f, ax = plt.subplots(figsize=(3.5, 2.5))
    f.subplots_adjust(bottom=.24, left=.25, right=.95, top=.89)
    for channel in ['VL', 'VR', 'RL']:
        if channel == 'VL':
            color = 'b'
        elif channel == 'VR':
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
    ax.set_ylim((-150, 75))
    ax.set_yticks((-150, -75, 0, 75))
    my.plot.despine(ax)

    # Legend
    f.text(.85, .9, 'VL', color='b', ha='center', va='center')
    f.text(.85, .82, 'VR', color='r', ha='center', va='center')
    f.text(.85, .74, 'RL', color='k', ha='center', va='center')

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
        fi.write(f'first pre-HL session only\n')
        fi.write(f'aggregated over recordings within mouse, then across mice\n')
        fi.write('error bars: SEM\n')
    
    # Echo
    with open(stats_filename) as fi:
        print(''.join(fi.readlines()))

if PLOT_EKG_BY_MOUSE:
    # Plot LR by mouse (this seems to be the largest and most consistent)
    f, ax = plt.subplots(figsize=(3.5, 2.5))
    f.subplots_adjust(bottom=.24, left=.25, right=.95, top=.89)

    # Slice LR
    LR_mean = mean_by_mouse.loc[:, 'RL'].unstack('mouse')
    
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
    ax.set_ylim((-200, 100))
    ax.set_yticks((-200, -100, 0, 100))
    my.plot.despine(ax)

    # Legend
    f.text(.85, .9, 'RL', color='k', ha='center', va='center')
    
    # Savefig
    f.savefig('figures/PLOT_EKG_BY_MOUSE.svg')
    f.savefig('figures/PLOT_EKG_BY_MOUSE.png', dpi=300)


plt.show()

