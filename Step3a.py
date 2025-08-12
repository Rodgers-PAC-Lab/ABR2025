## Grand average cohort plots for a 'normal' cohort where there's not any
# dramatic differences to compare like HL or FAD status or whatever

import os
import datetime
import glob
import json
import matplotlib
import scipy.signal
import numpy as np
import matplotlib
import pandas
import paclab.abr
from paclab.abr import abr_plotting, abr_analysis
import my.plot
import matplotlib.pyplot as plt


## Plots
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
# Load results of Step1
mouse_metadata = pandas.read_pickle(
    os.path.join(output_directory, 'mouse_metadata'))
experiment_metadata = pandas.read_pickle(
    os.path.join(output_directory, 'experiment_metadata'))
recording_metadata = pandas.read_pickle(
    os.path.join(output_directory, 'recording_metadata'))

# Load results of Step2
big_abrs = pandas.read_pickle(
    os.path.join(output_directory, 'big_abrs'))
    

## Params
sampling_rate = 16000  # TODO: store in recording_metadata


## Load previous results
big_abrs = pandas.read_pickle(
    os.path.join(output_directory, 'big_abrs'))


## Aggregate over recordings for each ABR
# TODO: do this upstream
avged_abrs = big_abrs.groupby(
    [lev for lev in big_abrs.index.names if lev != 'recording']).mean()

# Join after_HL on avged_abrs
avged_abrs = my.misc.join_level_onto_index(
    avged_abrs, 
    experiment_metadata.set_index(['mouse', 'date'])['after_HL'], 
    join_on=['mouse', 'date']
    )

# Keep only after_HL == False
avged_abrs = avged_abrs.loc[False]

# Calculate the grand average (averaging out date and mouse)
grand_average = avged_abrs.groupby(
    [lev for lev in avged_abrs.index.names if lev not in ['date', 'mouse']]).mean()


## Plots
GRAND_AVG_ABR_PLOT = True
GRAND_AVG_IMSHOW = True
GRAND_AVG_IPSI_VS_CONTRA = True
GRAND_AVG_LR_LEFT_VS_RIGHT = True
GRAND_AVG_ONE_SIDE_ONLY = True

# Define a global t-axis for all plots
t = avged_abrs.columns / sampling_rate * 1000

if GRAND_AVG_ABR_PLOT:
    ## Plot the grand average ABR for every channel * speaker_side
    # Set up ax_rows and ax_cols
    channel_l = ['LV', 'RV', 'LR']
    speaker_side_l = ['L', 'R']

    # Set up colorbar
    # Always do the lowest labels last
    label_l = sorted(
        grand_average.index.get_level_values('label').unique(), 
        reverse=True)
    aut_colorbar = paclab.abr.abr_plotting.generate_colorbar(
        len(label_l), mapname='inferno_r', start=0.15, stop=1)[::-1]
    
    # Make handles
    f, axa = plt.subplots(3, 2, sharex=True, sharey=True)
    f.subplots_adjust(
        left=.1, right=.9, top=.95, bottom=.12, hspace=0.06, wspace=0.2)

    # Group
    gobj = grand_average.groupby(['channel', 'speaker_side'])
    
    # Iterate over gruops
    for (channel, speaker_side), subdf in gobj:
        # Get ax
        ax = axa[
            channel_l.index(channel),
            speaker_side_l.index(speaker_side),
            ]
        
        # Drop the grouping keys
        topl = subdf.droplevel(['channel', 'speaker_side']).copy()
        
        # Plot each label, ending with the softest
        for n_label, label in enumerate(label_l):
            ax.plot(
                t, topl.loc[label] * 1e6, 
                lw=.75, color=aut_colorbar[n_label],
                )
        
        # Despine 
        if ax in axa[-1]:
            my.plot.despine(ax, which=('left', 'right', 'top'))
        else:
            my.plot.despine(ax, which=('left', 'right', 'top', 'bottom'))

    # Legend
    for n_label, (label, color) in enumerate(zip(label_l, aut_colorbar)):
        if np.mod(n_label, 2) != 0:
            continue
        f.text(
            .95, .85 - n_label * .02, f'{label} dB',
            color=color, ha='center', va='center', size=12)

    # Pretty
    ax.set_xlim((-1, 7))
    ax.set_ylim((-3.3, 3.3))
    ax.set_xticks([0, 3, 6])
    ax.set_yticks([])
    f.text(.51, .01, 'time (ms)', ha='center', va='bottom')

    # Legend 
    aut_colorbar = paclab.abr.abr_plotting.generate_colorbar(
        len(topl.index), mapname='inferno_r', start=0.15, stop=1)    

    # Scale bar
    axa[0, -1].plot([6, 6], [1, 2], 'k-', lw=.75)
    axa[0, -1].text(6.2, 1.5, '1 uV', ha='left', va='center', size=12)
    
    # Label the channel
    for n_channel, channel in enumerate(channel_l):
        axa[n_channel, 0].set_ylabel(channel, labelpad=20)
    
    # Label the speaker side
    axa[0, 0].set_title('sound from left')
    axa[0, 1].set_title('sound from right')
    
    # Save figure
    savename = 'GRAND_AVG_ABR_PLOT'
    f.savefig(os.path.join(output_directory, savename + '.svg'))
    f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)


if GRAND_AVG_IMSHOW:
    ## Plot the grand average ABR as an imshow for each channel * speaker_side
    # Set up ax_rows and ax_cols
    channel_l = ['LV', 'RV', 'LR']
    speaker_side_l = ['L', 'R']

    # Make handles
    f, axa = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(6.4, 4.8))
    f.subplots_adjust(
        left=.18, right=.98, top=.95, bottom=.12, hspace=0.2, wspace=0.2)

    # Separate figure just for colorbar
    f_cb, ax_cb = plt.subplots(figsize=(1, 4.8))
    f_cb.subplots_adjust(
        left=.3, right=.5, top=.95, bottom=.12)

    # Group
    gobj = grand_average.groupby(['channel', 'speaker_side'])
    
    # Iterate over gruops
    for (channel, speaker_side), subdf in gobj:
        # Get ax
        ax = axa[
            channel_l.index(channel),
            speaker_side_l.index(speaker_side),
            ]
        
        # Drop the grouping keys
        topl = subdf.droplevel(['channel', 'speaker_side']).copy()

        # Imshow
        im = my.plot.imshow(
            topl * 1e6, 
            x=t,
            y=topl.index.get_level_values('label'), 
            center_clim=True, 
            origin='lower', 
            ax=ax,
            )
        
        # Pretty
        ax.set_yticks((50, 90))
        ax.set_xticks((0, 3, 6))
        ax.set_xlim((-1, 7))
    
    # Harmonize clim
    my.plot.harmonize_clim_in_subplots(
        fig=f, center_clim=True, clim=(-3, 3), trim=.999)
    
    # Add the color bar
    cb = f.colorbar(im, cax=ax_cb)

    # Label the channel
    for n_channel, channel in enumerate(channel_l):
        axa[n_channel, 0].set_ylabel(
            channel, labelpad=40, rotation=0, va='center')
    
    # Label the speaker side
    axa[0, 0].set_title('sound from left')
    axa[0, 1].set_title('sound from right')

    # Shared y-label
    f.text(
        .11, .53, 'sound level (dB SPL)', 
        ha='center', va='center', rotation=90)
    
    # Shared x-label
    f.text(.51, .01, 'time (ms)', ha='center', va='bottom')

    # Save figure
    savename = 'GRAND_AVG_IMSHOW'
    f.savefig(os.path.join(output_directory, savename + '.svg'))
    f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)

if GRAND_AVG_IPSI_VS_CONTRA:
    # Get only the loudest sounds
    loudest = grand_average.xs(np.max(label_l), level='label') * 1e6

    # Plot handles
    f, ax = plt.subplots(figsize=(4.5, 2.5))
    f.subplots_adjust(bottom=.24, left=.15, right=.93, top=.89)
    
    # Plot each
    # green=ipsi, pink=contra
    # solid=left speaker, dashed=right speaker
    ax.plot(
        t, loudest.loc['LV'].loc['L'], color='green', ls='-', lw=1)
    ax.plot(
        t, loudest.loc['RV'].loc['R'], color='green', ls='--', lw=1)
    
    ax.plot(
        t, loudest.loc['LV'].loc['R'], color='magenta', ls='--', lw=1)
    ax.plot(
        t, loudest.loc['RV'].loc['L'], color='magenta', ls='-', lw=1)
    
    # Legend
    ax.text(6, 4, 'ipsi', color='green', ha='center', size=12)
    ax.text(6, 3, 'contra', color='magenta', ha='center', size=12)
    
    # Pretty
    my.plot.despine(ax)
    ax.set_title('LV and RV')
    ax.set_xlim(-1, 7)
    ax.set_xticks((0, 3, 6))
    ax.set_ylim(-4, 4)
    ax.set_yticks((-4, 0, 4))
    ax.set_ylabel('ABR (uV)')
    ax.set_xlabel('time (ms)')
    
    # Save figure
    savename = 'GRAND_AVG_IPSI_VS_CONTRA'
    f.savefig(os.path.join(output_directory, savename + '.svg'))
    f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)

if GRAND_AVG_LR_LEFT_VS_RIGHT:
    # Get only the loudest sounds
    loudest = grand_average.xs(np.max(label_l), level='label') * 1e6

    # Plot handles
    f, ax = plt.subplots(figsize=(4.5, 2.5))
    f.subplots_adjust(bottom=.24, left=.15, right=.93, top=.89)
    
    # Plot each
    # solid=left speaker, dashed=right speaker
    ax.plot(
        t, loudest.loc['LR'].loc['L'], color='k', ls='-', lw=1, label='left')
    ax.plot(
        t, loudest.loc['LR'].loc['R'], color='k', ls='--', lw=1, label='right')
    ax.set_title('LR')
    
    # Legend
    ax.legend(loc='right', frameon=False, bbox_to_anchor=(1, 1), prop={'size': 12})
    
    # Pretty
    my.plot.despine(ax)
    ax.set_xlim(-1, 7)
    ax.set_xticks((0, 3, 6))
    ax.set_ylim(-3, 3)
    ax.set_yticks((-3, 0, 3))
    ax.set_ylabel('ABR (uV)')
    ax.set_xlabel('time (ms)')
    
    # Save figure
    savename = 'GRAND_AVG_LR_LEFT_VS_RIGHT'
    f.savefig(os.path.join(output_directory, savename + '.svg'))
    f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)

if GRAND_AVG_ONE_SIDE_ONLY:
    # Get only the loudest sounds
    loudest = grand_average.xs(np.max(label_l), level='label') * 1e6

    # Plot handles
    f, ax = plt.subplots(figsize=(4.5, 2.5))
    f.subplots_adjust(bottom=.24, left=.15, right=.93, top=.89)
    
    # Plot each
    # solid=left speaker, dashed=right speaker
    ax.plot(
        t, loudest.loc['LV'].loc['R'], color='b', ls='-', lw=1, label='LV')
    ax.plot(
        t, loudest.loc['RV'].loc['R'], color='r', ls='-', lw=1, label='RV')
    ax.plot(
        t, loudest.loc['LR'].loc['R'], color='k', ls='-', lw=1, label='LR')
    ax.set_title('sound from right')
    
    # Legend
    ax.legend(loc='right', frameon=False, bbox_to_anchor=(1, 1), prop={'size': 12})
    
    # Pretty
    my.plot.despine(ax)
    ax.set_xlim(-1, 7)
    ax.set_xticks((0, 3, 6))
    ax.set_ylim(-4, 4)
    ax.set_yticks((-4, 0, 4))
    ax.set_ylabel('ABR (uV)')
    ax.set_xlabel('time (ms)')
    
    # Save figure
    savename = 'GRAND_AVG_ONE_SIDE_ONLY'
    f.savefig(os.path.join(output_directory, savename + '.svg'))
    f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)