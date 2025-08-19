## Plots of the "grand average" over all mouse * date (pre-HL only)
# Plots
#   GRAND_AVG_ABR_PLOT
#   GRAND_AVG_IMSHOW
#   GRAND_AVG_IPSI_VS_CONTRA 
#   GRAND_AVG_LR_LEFT_VS_RIGHT
#   GRAND_AVG_ONE_SIDE_ONLY 
#   PLOT_DELAY_VS_LEVEL

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


## Params
sampling_rate = 16000  # TODO: store in recording_metadata


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
    

## Join HL metadata on big_abrs
# Join after_HL onto big_abrs
big_abrs = my.misc.join_level_onto_index(
    big_abrs, 
    experiment_metadata.set_index(['mouse', 'date'])['after_HL'], 
    join_on=['mouse', 'date']
    )

# Join HL_type onto big_abrs
big_abrs = my.misc.join_level_onto_index(
    big_abrs, 
    mouse_metadata.set_index('mouse')['HL_type'], 
    join_on='mouse',
    )


## Aggregate over recordings for each ABR
# TODO: do this upstream
averaged_abrs = big_abrs.groupby(
    [lev for lev in big_abrs.index.names if lev != 'recording']
    ).mean()

# Calculate the grand average (averaging out date and mouse)
grand_average = averaged_abrs.groupby(
    [lev for lev in averaged_abrs.index.names if lev not in ['date', 'mouse']]
    ).mean()


## Cross-correlate over level to measure delay
rec_l = []
rec_keys_l = []
grouping_keys = [lev for lev in averaged_abrs.index.names if lev != 'label']
for (grouped_keys), subdf in averaged_abrs.groupby(grouping_keys):
    
    # Droplevel, leaving only label
    subdf = subdf.droplevel(grouping_keys)
    
    # Correlate each with the loudest
    for level in subdf.index:
        # Slice and convert to uV
        x1 = subdf.loc[level] * 1e6
        x2 = subdf.loc[91] * 1e6
        
        # Xcorr
        # TODO: confirm that positive delays means x1 lags x2
        counts, corrn = my.misc.correlate(x1, x2, mode='same')
        
        # Minimize noise by looking for values in expected range (<0.75 ms)
        keep_mask = np.abs(corrn) <= 12
        counts = counts[keep_mask]
        corrn = corrn[keep_mask]

        # Find peak
        peak_idx = corrn[counts.argmax()]
        peak_val = counts.max()
        
        # Normalize
        # Max achievable peak_val is max_val * norm(x1)
        # If x1 == x2, max achievable is max_val ** 2
        # Typicall x1 will be less than x2
        max_val = np.sqrt(np.sum(x2 ** 2))
        
        # Store
        rec_l.append((peak_idx, peak_val, max_val))
        rec_keys_l.append(list(grouped_keys) + [level])

# Concat
midx = pandas.MultiIndex.from_tuples(
    rec_keys_l, names=list(grouping_keys) + ['label'])
corr_df = pandas.DataFrame(
    rec_l, columns=['idx', 'val', 'max'], index=midx)

# Norm
# 1 is achieved with equality
# 0 is achieved with very low-power signal
# Almost all are intermediate
corr_df['norm'] = corr_df['val'] / corr_df['max'] ** 2

# The delays become highly variable (random) for the lower levels
# Dropping delays with low 'norm' introduces strange selection effects
# Better to just give up on lowest levels
corr_df = corr_df[corr_df.index.get_level_values('label') >= 52]


## Plots
GRAND_AVG_ABR_PLOT = True
GRAND_AVG_IMSHOW = True
GRAND_AVG_IPSI_VS_CONTRA = True
GRAND_AVG_LR_LEFT_VS_RIGHT = True
GRAND_AVG_ONE_SIDE_ONLY = True
PLOT_DELAY_VS_LEVEL = True

# Define a global t-axis for all plots
t = averaged_abrs.columns / sampling_rate * 1000

if GRAND_AVG_ABR_PLOT:
    ## Plot the grand average ABR for every channel * speaker_side
    # Do this in three ways: control, sham, bilateral
    for plot_type in ['healthy', 'sham', 'bilateral']:
        
        ## Slice data
        if plot_type == 'healthy':
            this_averaged_abrs = averaged_abrs.xs(
                False, level='after_HL').droplevel('HL_type')
        
        elif plot_type == 'sham':
            this_averaged_abrs = averaged_abrs.xs(
                True, level='after_HL').xs('sham', level='HL_type')
        
        elif plot_type == 'bilateral':
            this_averaged_abrs = averaged_abrs.xs(
                True, level='after_HL').xs('bilateral', level='HL_type')
        
        else:
            1/0

        # Calculate the grand average (averaging out date and mouse)
        this_grand_average = this_averaged_abrs.groupby(
            [lev for lev in this_averaged_abrs.index.names if lev not in ['date', 'mouse']]
            ).mean()
        
        
        ## Set up plot
        # Set up ax_rows and ax_cols
        channel_l = ['LV', 'RV', 'LR']
        speaker_side_l = ['L', 'R']

        # Set up colorbar
        # Always do the lowest labels last
        label_l = sorted(
            this_grand_average.index.get_level_values('label').unique(), 
            reverse=True)
        aut_colorbar = paclab.abr.abr_plotting.generate_colorbar(
            len(label_l), mapname='inferno_r', start=0.15, stop=1)[::-1]
        
        # Make handles
        f, axa = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(5.4, 4))
        f.subplots_adjust(
            left=.1, right=.9, top=.95, bottom=.12, hspace=0.06, wspace=0.2)

        # Group
        gobj = this_grand_average.groupby(['channel', 'speaker_side'])
        
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
        savename = f'GRAND_AVG_ABR_PLOT__{plot_type}'
        f.savefig(os.path.join(output_directory, savename + '.svg'))
        f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)


if GRAND_AVG_IMSHOW:
    ## Plot the grand average ABR as an imshow for each channel * speaker_side
    # Do this in three ways: control, sham, bilateral
    for plot_type in ['healthy', 'sham', 'bilateral']:
        
        ## Slice data
        if plot_type == 'healthy':
            this_averaged_abrs = averaged_abrs.xs(
                False, level='after_HL').droplevel('HL_type')
        
        elif plot_type == 'sham':
            this_averaged_abrs = averaged_abrs.xs(
                True, level='after_HL').xs('sham', level='HL_type')
        
        elif plot_type == 'bilateral':
            this_averaged_abrs = averaged_abrs.xs(
                True, level='after_HL').xs('bilateral', level='HL_type')
        
        else:
            1/0

        # Calculate the grand average (averaging out date and mouse)
        this_grand_average = this_averaged_abrs.groupby(
            [lev for lev in this_averaged_abrs.index.names if lev not in ['date', 'mouse']]
            ).mean()
        
        # Set up ax_rows and ax_cols
        channel_l = ['LV', 'RV', 'LR']
        speaker_side_l = ['L', 'R']

        # Make handles
        f, axa = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(5.4, 4))
        f.subplots_adjust(
            left=.18, right=.98, top=.95, bottom=.12, hspace=0.2, wspace=0.2)

        # Separate figure just for colorbar
        f_cb, ax_cb = plt.subplots(figsize=(1, 4.8))
        f_cb.subplots_adjust(
            left=.3, right=.5, top=.95, bottom=.12)

        # Group
        gobj = this_grand_average.groupby(['channel', 'speaker_side'])
        
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
        savename = f'GRAND_AVG_IMSHOW__{plot_type}'
        f.savefig(os.path.join(output_directory, savename + '.svg'))
        f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)
        f_cb.savefig(os.path.join(output_directory, savename + '.colorbar.svg'))
        f_cb.savefig(os.path.join(output_directory, savename + '.colorbar.png'), dpi=300)

if GRAND_AVG_IPSI_VS_CONTRA:
    ## Slice data
    # For this analysis, use only after_HL == False
    this_averaged_abrs = averaged_abrs.xs(
        False, level='after_HL').droplevel('HL_type')    

    # Calculate the grand average (averaging out date and mouse)
    this_grand_average = this_averaged_abrs.groupby(
        [lev for lev in this_averaged_abrs.index.names if lev not in ['date', 'mouse']]
        ).mean()
    
    # Slice loudest sound
    loudest = this_grand_average.xs(91, level='label') * 1e6


    ## Plot handles
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

    ## Slice data
    # For this analysis, use only after_HL == False
    this_averaged_abrs = averaged_abrs.xs(
        False, level='after_HL').droplevel('HL_type')    
    
    # Calculate the grand average (averaging out date and mouse)
    this_grand_average = this_averaged_abrs.groupby(
        [lev for lev in this_averaged_abrs.index.names if lev not in ['date', 'mouse']]
        ).mean()
    
    # Slice loudest sound
    loudest = this_grand_average.xs(91, level='label') * 1e6

    
    ## Set up plot
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
    # Slice data
    # for this analysis, use after_HL == False
    this_averaged_abrs = averaged_abrs.xs(
        False, level='after_HL').droplevel('HL_type')
    
    # Calculate the grand average (averaging out date and mouse)
    this_grand_average = this_averaged_abrs.groupby(
        [lev for lev in this_averaged_abrs.index.names if lev not in ['date', 'mouse']]
        ).mean()
    
    # Slice loudest sound
    loudest = this_grand_average.xs(91, level='label') * 1e6

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

if PLOT_DELAY_VS_LEVEL:
    ## Plot the delay versus sound level
    
    ## Slice data
    # For this analysis, use only after_HL == False    
    this_corr_df = corr_df.xs(
        False, level='after_HL').droplevel('HL_type')
    

    ## Fit a line to the delay vs level
    # TODO: do this in one shot, by finding a single slope that explains all levels
    rec_l = []
    rec_keys_l = []
    for (date, mouse, speaker_side, channel), subdf in this_corr_df.groupby(
            ['date', 'mouse', 'speaker_side', 'channel']):
        
        # droplevel
        subdf = subdf.droplevel(['date', 'mouse', 'speaker_side', 'channel'])
        
        # fit
        # slope units are delay in samples per dB
        # Multiply by 1000/16 to convert to us / dB
        # Should be about 7 us / dB, or ~300 us over the tested range
        fit = scipy.stats.linregress(subdf.index.values, subdf['idx'])
        
        # store slope, rval
        # rval should generally be between -1 and -0.85
        # slope should be mean -0.11, std 0.02
        rec_l.append((fit.slope, fit.rvalue))
        rec_keys_l.append((date, mouse, speaker_side, channel))

    # Concat
    # The fits are worse for speaker_side R
    midx = pandas.MultiIndex.from_tuples(
        rec_keys_l, names=['date', 'mouse', 'speaker_side', 'channel'])
    slope_df = pandas.DataFrame(
        rec_l, columns=['slope', 'rval'], index=midx)

    # Mean slope (in samples/dB)
    slope_by_mouse = slope_df['slope'].groupby('mouse').mean()
    
    # Convert to us/dB
    slope_by_mouse = slope_by_mouse / 16e3 * 1e6
    
    # Agg
    slope_by_mouse_mu = slope_by_mouse.mean()
    slope_by_mouse_sem = slope_by_mouse.sem()
    
    
    ## Make plot
    f, ax = my.plot.figure_1x1_standard()
    
    # First mean within mouse
    # This averages over all channels * speaker_side, for better or for worse
    # LV and RV are closer to linear, LR accelerates more with level, and
    # has a weird jog at the loudest levels
    to_agg = this_corr_df['idx'].groupby(
        ['mouse', 'label']).mean().unstack('mouse') / 16e3 * 1e3
    
    # Now aggregate with mouse as N
    n_mice = to_agg.shape[1]
    topl_mu = to_agg.mean(axis=1)
    topl_err = to_agg.sem(axis=1)
    ax.plot(topl_mu, color='k')
    ax.fill_between(
        x=topl_mu.index,
        y1=topl_mu - topl_err,
        y2=topl_mu + topl_err,
        alpha=.5, lw=0, color='k',
        )

    # Pretty
    my.plot.despine(ax)
    ax.set_xlim((45, 95))
    ax.set_xticks((50, 70, 90))
    ax.set_ylim((0, 0.4))
    ax.set_yticks((0, 0.2, 0.4))
    
    # Label
    ax.set_xlabel('sound level (dB)')
    ax.set_ylabel('delay (ms)')
    
    # Legend
    ax.text(80, 0.35, f'n = {n_mice} mice', ha='center', va='center')
    
    # Save figure
    savename = 'PLOT_DELAY_VS_LEVEL'
    f.savefig(os.path.join(output_directory, savename + '.svg'))
    f.savefig(os.path.join(output_directory, savename + '.png'), dpi=300)
    
    # Stats
    with open('STATS__PLOT_DELAY_VS_LEVEL', 'w') as fi:
        fi.write('mean over xcorr peak for all channels, speaker sides, and recordings\n')
        fi.write(f'n = {n_mice} mice\n')
        fi.write(f'mean slope in us/dB: {slope_by_mouse_mu}\n')
        fi.write(f'SEM slope in us/dB: {slope_by_mouse_sem}\n')
    
    # Echo
    with open('STATS__PLOT_DELAY_VS_LEVEL') as fi:
        for line in fi.readlines():
            print(line.strip())