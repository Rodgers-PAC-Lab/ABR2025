## Plots of the "grand average" over all mouse * date (pre-HL only)
# Plots
#   GRAND_AVG_ABR_PLOT
#   GRAND_AVG_IMSHOW
#   GRAND_AVG_IPSI_VS_CONTRA 
#   GRAND_AVG_LR_LEFT_VS_RIGHT
#   GRAND_AVG_ONE_SIDE_ONLY 
#   PLOT_DELAY_VS_LEVEL

import os
import json
import datetime
import matplotlib
import scipy.signal
import numpy as np
import matplotlib
import pandas
import my.plot
import matplotlib.pyplot as plt


## Plots
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
sampling_rate = 16000


## Load metadata
mouse_metadata = pandas.read_csv(
    os.path.join(raw_data_directory, 'metadata', 'mouse_metadata.csv'))
experiment_metadata = pandas.read_csv(
    os.path.join(raw_data_directory, 'metadata', 'experiment_metadata.csv'))
recording_metadata = pandas.read_csv(
    os.path.join(raw_data_directory, 'metadata', 'recording_metadata.csv'))

# Coerce
recording_metadata['date'] = recording_metadata['date'].apply(
    lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())
experiment_metadata['date'] = experiment_metadata['date'].apply(
    lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())
mouse_metadata['DOB'] = mouse_metadata['DOB'].apply(
    lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())

# Coerce: special case this one because it can be null
mouse_metadata['HL_date'] = mouse_metadata['HL_date'].apply(
    lambda x: None if pandas.isnull(x) else 
    datetime.datetime.strptime(x, '%Y-%m-%d').date())

# Index
recording_metadata = recording_metadata.set_index(
    ['date', 'mouse', 'recording']).sort_index()


## Load previous results
# Load results of Step2b_avg
big_abrs = pandas.read_pickle(
    os.path.join(output_directory, 'big_abrs'))
averaged_abrs_by_mouse = pandas.read_pickle(
    os.path.join(output_directory, 'averaged_abrs_by_mouse'))
averaged_abrs_by_date = pandas.read_pickle(
    os.path.join(output_directory, 'averaged_abrs_by_date'))
trial_counts = pandas.read_pickle(
    os.path.join(output_directory, 'trial_counts'))

# Loudest dB
loudest_db = big_abrs.index.get_level_values('label').max()
  

## Cross-correlate over level to measure delay
rec_l = []
rec_keys_l = []
grouping_keys = [lev for lev in averaged_abrs_by_mouse.index.names if lev != 'label']
for (grouped_keys), subdf in averaged_abrs_by_mouse.groupby(grouping_keys):
    
    # Droplevel, leaving only label
    subdf = subdf.droplevel(grouping_keys)
    
    # Correlate each with the loudest
    for level in subdf.index:
        # Slice and convert to uV
        x1 = subdf.loc[level] * 1e6
        x2 = subdf.loc[loudest_db] * 1e6
        
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
corr_df = corr_df[corr_df.index.get_level_values('label') >= 34]


## Plots
GRAND_AVG_ABR_PLOT = True
GRAND_AVG_ABR_PLOT_PERI_HL = True
GRAND_AVG_IMSHOW = True
GRAND_AVG_IMSHOW_PERI_HL = True
GRAND_AVG_IPSI_VS_CONTRA = True
GRAND_AVG_LR_LEFT_VS_RIGHT = True
GRAND_AVG_ONE_SIDE_ONLY = True
PLOT_DELAY_VS_LEVEL = True

if GRAND_AVG_ABR_PLOT:
    ## Plot the grand average ABR for every channel * speaker_side
    # Do this in three ways: control, sham, bilateral
    for plot_type in ['healthy']:#, 'sham', 'bilateral']:
        
        ## Slice data
        if plot_type == 'healthy':
            # This will include mice that never received HL
            this_averaged_abrs = averaged_abrs_by_mouse.xs(
                False, level='after_HL').droplevel('HL_type')
        
        elif plot_type == 'sham':
            this_averaged_abrs = averaged_abrs_by_mouse.xs(
                True, level='after_HL').xs('sham', level='HL_type')
        
        elif plot_type == 'bilateral':
            this_averaged_abrs = averaged_abrs_by_mouse.xs(
                True, level='after_HL').xs('bilateral', level='HL_type')
        
        else:
            1/0

        # Calculate the grand average (averaging out mouse)
        this_grand_average = this_averaged_abrs.groupby(
            [lev for lev in this_averaged_abrs.index.names if lev != 'mouse']
            ).mean()

        # Slice temporally
        this_grand_average = this_grand_average.loc[:, -16:112]
        this_t = this_grand_average.columns / sampling_rate * 1000

        # Count mice
        n_mice = len(this_averaged_abrs.index.get_level_values('mouse').unique())
        
        
        ## Set up plot
        # Set up ax_rows and ax_cols
        channel_l = ['LV', 'RV', 'LR']
        speaker_side_l = ['L', 'R']

        # Set up colorbar
        # Always do the lowest labels last
        label_l = sorted(
            this_grand_average.index.get_level_values('label').unique(), 
            reverse=True)
        aut_colorbar = my.plot.generate_colorbar(
            len(label_l), mapname='inferno_r', start=0.15, stop=1)
        
        # Make handles
        f, axa = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(5.4, 4))
        f.subplots_adjust(
            left=.1, right=.9, top=.95, bottom=.12, hspace=0.06, wspace=0.2)

        # Group
        gobj = this_grand_average.groupby(['channel', 'speaker_side'])
        
        # Iterate over groups
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
                    this_t, 
                    topl.loc[label] * 1e6, 
                    lw=.75, 
                    color=aut_colorbar[n_label],
                    clip_on=False,
                    )
            
            # Invisible background
            ax.set_facecolor('none')
            
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
                .95, .68 - n_label * .02, f'{label} dB',
                color=color, ha='center', va='center', size=12)

        # Pretty
        ax.set_xlim((-1, 7))
        ax.set_ylim((-2.5, 2.5))
        ax.set_xticks([0, 3, 6])
        ax.set_yticks([])
        f.text(.51, .01, 'time (ms)', ha='center', va='bottom')

        # Scale bar
        axa[0, -1].plot([5, 5], [1, 2], 'k-', lw=.75, clip_on=False)
        axa[0, -1].text(5.2, 1.5, f'1 {MU}V', ha='left', va='center', size=12)
        
        # Label the channel
        for n_channel, channel in enumerate(channel_l):
            axa[n_channel, 0].set_ylabel(channel)
        
        # Label the speaker side
        axa[0, 0].set_title('sound from left')
        axa[0, 1].set_title('sound from right')
        
        
        ## Save figure
        savename = f'figures/GRAND_AVG_ABR_PLOT__{plot_type}'
        f.savefig(savename + '.svg')
        f.savefig(savename + '.png', dpi=300)

        
        ## Stats
        stats_filename = f'figures/STATS__GRAND_AVG_ABR_PLOT__{plot_type}'
        with open(stats_filename, 'w') as fi:
            fi.write(f'n = {n_mice} mice\n')
        
        # Echo
        print(stats_filename)
        with open(stats_filename) as fi:
            print(''.join(fi.readlines()))

if GRAND_AVG_ABR_PLOT_PERI_HL:
    ## Plot the grand average ABR for each speaker_side

    # Slice data for this analysis - after_HL only
    averaged_abrs_peri_HL = averaged_abrs_by_mouse.drop(
        'none', level='HL_type').xs(True, level='after_HL')  
    
    # Separate figures for each speaker_side
    for speaker_side in ['L', 'R']:
        
        # Slice speaker_side
        this_averaged_abrs = averaged_abrs_peri_HL.xs(
            speaker_side, level='speaker_side')

        # Calculate the grand average (averaging out mouse)
        this_grand_average = this_averaged_abrs.groupby(
            [lev for lev in this_averaged_abrs.index.names if lev != 'mouse']
            ).mean()
        
        # Slice temporally
        this_grand_average = this_grand_average.loc[:, -16:112]
        this_t = this_grand_average.columns / sampling_rate * 1000

        # Count mice by HL_type
        n_mice = this_averaged_abrs.groupby(
            ['HL_type', 'mouse']).size().groupby(['HL_type']).size()
        
        
        ## Set up plot
        # Set up ax_rows and ax_cols
        channel_l = ['LV', 'RV', 'LR']
        HL_type_l = ['bilateral', 'sham']

        # Set up colorbar
        # Always do the lowest labels last
        label_l = sorted(
            this_grand_average.index.get_level_values('label').unique(), 
            reverse=True)
        aut_colorbar = my.plot.generate_colorbar(
            len(label_l), mapname='inferno_r', start=0.15, stop=1)
        
        # Make handles
        f, axa = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(5.4, 4))
        f.subplots_adjust(
            left=.1, right=.9, top=.95, bottom=.12, hspace=0.06, wspace=0.2)

        # Group
        gobj = this_grand_average.groupby(['channel', 'HL_type'])
        
        # Iterate over groups
        for (channel, HL_type), subdf in gobj:
            # Get ax
            ax = axa[
                channel_l.index(channel),
                HL_type_l.index(HL_type),
                ]
            
            # Drop the grouping keys
            topl = subdf.droplevel(['channel', 'HL_type']).copy()
            
            # Plot each label, ending with the softest
            for n_label, label in enumerate(label_l):
                ax.plot(
                    this_t, 
                    topl.loc[label] * 1e6, 
                    lw=.75, 
                    color=aut_colorbar[n_label],
                    clip_on=False,
                    )
            
            # Invisible background
            ax.set_facecolor('none')
            
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
                .95, .68 - n_label * .02, f'{label} dB',
                color=color, ha='center', va='center', size=12)

        # Pretty
        ax.set_xlim((-1, 7))
        ax.set_ylim((-2.5, 2.5))
        ax.set_xticks([0, 3, 6])
        ax.set_yticks([])
        f.text(.51, .01, 'time (ms)', ha='center', va='bottom')

        # Scale bar
        axa[0, -1].plot([5, 5], [1, 2], 'k-', lw=.75, clip_on=False)
        axa[0, -1].text(5.2, 1.5, f'1 {MU}V', ha='left', va='center', size=12)
        
        # Label the channel
        for n_channel, channel in enumerate(channel_l):
            axa[n_channel, 0].set_ylabel(channel)
        
        # Label the HL_type
        axa[0, 0].set_title(HL_type_l[0])
        axa[0, 1].set_title(HL_type_l[1])
        
        
        ## Save figure
        savename = f'figures/GRAND_AVG_ABR_PLOT_PERI_HL__{speaker_side}'
        f.savefig(savename + '.svg')
        f.savefig(savename + '.png', dpi=300)

        
        ## Stats
        stats_filename = f'figures/STATS__GRAND_AVG_ABR_PLOT_PERI_HL__{speaker_side}'
        with open(stats_filename, 'w') as fi:
            fi.write(f'n = {n_mice} mice\n')
        
        # Echo
        print(stats_filename)
        with open(stats_filename) as fi:
            print(''.join(fi.readlines()))

if GRAND_AVG_IMSHOW:
    ## Plot the grand average ABR as an imshow for each channel * speaker_side
    # Do this in three ways: control, sham, bilateral
    for plot_type in ['healthy', 'sham', 'bilateral']:
        
        ## Slice data
        if plot_type == 'healthy':
            this_averaged_abrs = averaged_abrs_by_mouse.xs(
                False, level='after_HL').droplevel('HL_type')
        
        elif plot_type == 'sham':
            this_averaged_abrs = averaged_abrs_by_mouse.xs(
                True, level='after_HL').xs('sham', level='HL_type')
        
        elif plot_type == 'bilateral':
            this_averaged_abrs = averaged_abrs_by_mouse.xs(
                True, level='after_HL').xs('bilateral', level='HL_type')
        
        else:
            1/0

        # Calculate the grand average (averaging out mouse)
        this_grand_average = this_averaged_abrs.groupby(
            [lev for lev in this_averaged_abrs.index.names if lev != 'mouse']
            ).mean()
        
        # Set up ax_rows and ax_cols
        channel_l = ['LV', 'RV', 'LR']
        speaker_side_l = ['L', 'R']

        # Make handles
        f, axa = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(5.4, 4))
        f.subplots_adjust(
            left=.18, right=.98, top=.95, bottom=.12, hspace=0.2, wspace=0.2)

        # Separate figure just for colorbar
        f_cb, ax_cb = plt.subplots(figsize=(.7, 3.2))
        f_cb.subplots_adjust(
            left=.07, right=.22, top=.95, bottom=.12)

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
                x=topl.columns / sampling_rate * 1000,
                y=topl.index.get_level_values('label'), 
                center_clim=True, 
                origin='lower', 
                ax=ax,
                )
            
            # Pretty
            ax.set_yticks((30, 70))
            ax.set_xticks((0, 3, 6))
            ax.set_xlim((-1, 7))
        
        # Harmonize clim
        my.plot.harmonize_clim_in_subplots(
            fig=f, center_clim=True, clim=(-3, 3), trim=.999)
        
        # Add the color bar
        cb = f.colorbar(im, cax=ax_cb)
        ax_cb.set_ylabel(f'ABR ({MU}V)')

        # Label the channel
        for n_channel, channel in enumerate(channel_l):
            axa[n_channel, 0].set_ylabel(channel)
        axa[1, 0].set_ylabel(f'sound level (dB SPL)\n{channel_l[1]}')
        
        # Label the speaker side
        axa[0, 0].set_title('sound from left')
        axa[0, 1].set_title('sound from right')

        # Shared x-label
        f.text(.58, .01, 'time (ms)', ha='center', va='bottom')

        
        ## Save figure
        savename = f'figures/GRAND_AVG_IMSHOW__{plot_type}'
        f.savefig(savename + '.svg')
        f.savefig(savename + '.png', dpi=300)
        f_cb.savefig(savename + '.colorbar.svg')
        f_cb.savefig(savename + '.colorbar.png', dpi=300)

        
        ## Stats
        n_mice = len(this_averaged_abrs.index.get_level_values('mouse').unique())
        
        stats_filename = f'figures/STATS__GRAND_AVG_IMSHOW__{plot_type}'
        with open(stats_filename, 'w') as fi:
            fi.write(f'n = {n_mice} mice\n')
        
        # Echo
        print(stats_filename)
        with open(stats_filename) as fi:
            print(''.join(fi.readlines()))

if GRAND_AVG_IMSHOW_PERI_HL:
    ## Plot the grand average ABR as an imshow for each channel * after_HL

    # Slice data for this analysis - after_HL only
    averaged_abrs_peri_HL = averaged_abrs_by_mouse.drop(
        'none', level='HL_type').xs(True, level='after_HL')  
    
    # Separate figures for each speaker_side
    for speaker_side in ['L', 'R']:
        
        # Slice speaker_side
        this_averaged_abrs = averaged_abrs_peri_HL.xs(
            speaker_side, level='speaker_side')

        # Calculate the grand average (averaging out mouse)
        this_grand_average = this_averaged_abrs.groupby(
            [lev for lev in this_averaged_abrs.index.names if lev != 'mouse']
            ).mean()
        
        # Set up ax_rows and ax_cols
        channel_l = ['LV', 'RV', 'LR']
        HL_type_l = ['bilateral', 'sham']

        # Make handles
        f, axa = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(5.4, 4))
        f.subplots_adjust(
            left=.18, right=.98, top=.95, bottom=.12, hspace=0.2, wspace=0.2)

        # Separate figure just for colorbar
        f_cb, ax_cb = plt.subplots(figsize=(.7, 3.2))
        f_cb.subplots_adjust(
            left=.07, right=.22, top=.95, bottom=.12)

        # Group
        gobj = this_grand_average.groupby(['channel', 'HL_type'])
        
        # Iterate over groups
        for (channel, HL_type), subdf in gobj:
            # Get ax
            ax = axa[
                channel_l.index(channel),
                HL_type_l.index(HL_type),
                ]
            
            # Drop the grouping keys
            topl = subdf.droplevel(['channel', 'HL_type']).copy()

            # Imshow
            im = my.plot.imshow(
                topl * 1e6, 
                x=topl.columns / sampling_rate * 1000,
                y=topl.index.get_level_values('label'), 
                center_clim=True, 
                origin='lower', 
                ax=ax,
                )
            
            # Pretty
            ax.set_yticks((30, 70))
            ax.set_xticks((0, 3, 6))
            ax.set_xlim((-1, 7))
        
        # Harmonize clim
        my.plot.harmonize_clim_in_subplots(
            fig=f, center_clim=True, clim=(-3, 3), trim=.999)
        
        # Add the color bar
        cb = f.colorbar(im, cax=ax_cb)
        ax_cb.set_ylabel(f'ABR ({MU}V)')

        # Label the channel
        for n_channel, channel in enumerate(channel_l):
            axa[n_channel, 0].set_ylabel(channel)
        axa[1, 0].set_ylabel(f'sound level (dB SPL)\n{channel_l[1]}')
        
        # Label the speaker side
        axa[0, 0].set_title(HL_type_l[0])
        axa[0, 1].set_title(HL_type_l[1])

        # Shared x-label
        f.text(.58, .01, 'time (ms)', ha='center', va='bottom')

        
        ## Save figure
        savename = f'figures/GRAND_AVG_IMSHOW_PERI_HL__{speaker_side}'
        f.savefig(savename + '.svg')
        f.savefig(savename + '.png', dpi=300)
        f_cb.savefig(savename + '.colorbar.svg')
        f_cb.savefig(savename + '.colorbar.png', dpi=300)

        
        ## Stats
        n_mice = len(this_averaged_abrs.index.get_level_values('mouse').unique())
        
        stats_filename = f'figures/STATS__GRAND_AVG_IMSHOW_PERI_HL__{speaker_side}'
        with open(stats_filename, 'w') as fi:
            fi.write(f'n = {n_mice} mice\n')
        
        # Echo
        print(stats_filename)
        with open(stats_filename) as fi:
            print(''.join(fi.readlines()))

if GRAND_AVG_IPSI_VS_CONTRA:
    ## Slice data
    # For this analysis, use only after_HL == False
    this_averaged_abrs = averaged_abrs_by_mouse.xs(
        False, level='after_HL').droplevel('HL_type')    

    # Average over mouse
    this_grand_average = this_averaged_abrs.groupby(
        [lev for lev in this_averaged_abrs.index.names if lev != 'mouse']
        ).mean()
    
    # Slice loudest sound
    loudest = this_grand_average.xs(loudest_db, level='label') * 1e6


    ## Plot handles
    f, ax = plt.subplots(figsize=(4.5, 2.5))
    f.subplots_adjust(bottom=.24, left=.15, right=.93, top=.89)
    
    # Plot each
    # green=ipsi, pink=contra
    # solid=left speaker, dashed=right speaker
    this_t = loudest.columns / sampling_rate * 1000
    ax.plot(
        this_t, loudest.loc['LV'].loc['L'], color='green', ls='-', lw=1)
    ax.plot(
        this_t, loudest.loc['RV'].loc['R'], color='green', ls='--', lw=1)
    
    ax.plot(
        this_t, loudest.loc['LV'].loc['R'], color='magenta', ls='--', lw=1)
    ax.plot(
        this_t, loudest.loc['RV'].loc['L'], color='magenta', ls='-', lw=1)
    
    # Legend
    ax.text(6, 3, 'ipsi', color='green', ha='center', size=12)
    ax.text(6, 2, 'contra', color='magenta', ha='center', size=12)
    
    # Pretty
    my.plot.despine(ax)
    ax.set_title('LV and RV')
    ax.set_xlim(-1, 7)
    ax.set_xticks((0, 2, 4, 6))
    ax.set_ylim(-3, 3)
    ax.set_yticks((-3, 0, 3))
    ax.set_ylabel(f'ABR ({MU}V)')
    ax.set_xlabel('time (ms)')
    
    # Save figure
    savename = 'figures/GRAND_AVG_IPSI_VS_CONTRA'
    f.savefig(savename + '.svg')
    f.savefig(savename + '.png', dpi=300)


    ## Stats
    n_mice = len(this_averaged_abrs.index.get_level_values('mouse').unique())
    
    stats_filename = f'figures/STATS__GRAND_AVG_IPSI_VS_CONTRA'
    with open(stats_filename, 'w') as fi:
        fi.write(f'n = {n_mice} mice\n')
    
    # Echo
    print(stats_filename)
    with open(stats_filename) as fi:
        print(''.join(fi.readlines()))


## Graveyard code for estimating delay between ipsi and contra
if False:
    ## Resample
    # This will ring for about 20 samples near the window edges, but that
    # is far from the time that we care about
    resample_factor = 10
    new_len = this_averaged_abrs.shape[1] * resample_factor
    resampled, new_t = scipy.signal.resample(
        this_averaged_abrs, new_len, t=loudest.columns, axis=1)
    
    # The columns are weirdly a float into the original sample count
    resampled_df = pandas.DataFrame(
        resampled,
        index=this_averaged_abrs.index,
        columns=pandas.Index(new_t, name=this_averaged_abrs.columns.name),
        )


    ## Plot the delay between ipsi and contra
    winsize_ms = 1
    halfwinsize_samples = int(np.rint(winsize_ms * sampling_rate / 1000 * resample_factor / 2))

    # Iterate
    peak_res_l = []
    for mouse in this_averaged_abrs.index.get_level_values('mouse').unique():
        for channel in ['LV', 'RV']:
            # Determine which is which
            if channel == 'LV':
                ipsi_speaker = 'L'
                contra_speaker = 'R'
            else:
                ipsi_speaker = 'R'
                contra_speaker = 'L'
            
            # Slice
            ipsi_resp = resampled_df.loc[mouse].loc[channel].loc[ipsi_speaker]
            contra_resp = resampled_df.loc[mouse].loc[channel].loc[contra_speaker]
    
            # Moving xcorr
            for n_center_col in range(len(resampled_df.columns)):
                # Get start and stop indices
                start = n_center_col - halfwinsize_samples
                stop = n_center_col + halfwinsize_samples
                if start < 0 or stop >= len(resampled_df.columns):
                    continue
                
                # Convert n_center_col to actual time
                center_t = resampled_df.columns[n_center_col] / sampling_rate
                
                # Correlate ipsi vs contra
                # This will be biased toward zero-lag because of the zero-padding
                counts, corrn = my.misc.correlate(
                    ipsi_resp.iloc[:, start:stop].T, #loc[63],
                    contra_resp.iloc[:, start:stop].T, #.loc[63],
                    mode='same')   
    
                # Mean over sound levels
                counts = counts.mean(axis=1)
    
                # Find peak
                # Positive peak_idx means that the second signal leads the first one
                peak_idx = corrn[counts.argmax()]
                peak_val = counts.max()
                
                # Store
                peak_res_l.append({
                    'mouse': mouse,
                    'channel': channel,
                    'center_t': center_t,
                    'peak_idx': peak_idx,
                    })

    # DataFrame
    peak_df = pandas.DataFrame.from_records(peak_res_l)
    
    # Aggregate over channel
    peak_df2 = peak_df.groupby(['mouse', 'center_t'])['peak_idx'].mean().unstack('mouse')
    
    # Plot
    f, ax = plt.subplots()
    ax.plot(
        peak_df2.index * 1000,
        (peak_df2.values / resample_factor / sampling_rate * 1000),
        )


if GRAND_AVG_LR_LEFT_VS_RIGHT:

    ## Slice data
    # For this analysis, use only after_HL == False
    this_averaged_abrs = averaged_abrs_by_mouse.xs(
        False, level='after_HL').droplevel('HL_type')    
    
    # Calculate the grand average (averaging out mouse)
    this_grand_average = this_averaged_abrs.groupby(
        [lev for lev in this_averaged_abrs.index.names if lev != 'mouse']
        ).mean()
    
    # Slice loudest sound
    loudest = this_grand_average.xs(loudest_db, level='label') * 1e6

    
    ## Set up plot
    # Plot handles
    f, ax = plt.subplots(figsize=(4.5, 2.5))
    f.subplots_adjust(bottom=.24, left=.15, right=.93, top=.89)
    
    # Plot each
    # solid=left speaker, dashed=right speaker
    this_t = loudest.columns / sampling_rate * 1000
    ax.plot(
        this_t, loudest.loc['LR'].loc['L'], color='k', ls='-', lw=1, label='left')
    ax.plot(
        this_t, loudest.loc['LR'].loc['R'], color='k', ls='--', lw=1, label='right')
    ax.set_title('LR')
    
    # Legend
    ax.legend(loc='right', frameon=False, bbox_to_anchor=(1, 1), prop={'size': 12})
    
    # Pretty
    my.plot.despine(ax)
    ax.set_xlim(-1, 7)
    ax.set_xticks((0, 2, 4, 6))
    ax.set_ylim(-3, 3)
    ax.set_yticks((-3, 0, 3))
    ax.set_ylabel(f'ABR ({MU}V)')
    ax.set_xlabel('time (ms)')
    
    # Save figure
    savename = 'figures/GRAND_AVG_LR_LEFT_VS_RIGHT'
    f.savefig(savename + '.svg')
    f.savefig(savename + '.png', dpi=300)


    ## Stats
    n_mice = len(this_averaged_abrs.index.get_level_values('mouse').unique())
    
    stats_filename = f'figures/STATS__GRAND_AVG_LR_LEFT_VS_RIGHT'
    with open(stats_filename, 'w') as fi:
        fi.write(f'n = {n_mice} mice\n')
    
    # Echo
    print(stats_filename)
    with open(stats_filename) as fi:
        print(''.join(fi.readlines()))
    
    
if GRAND_AVG_ONE_SIDE_ONLY:
    # Slice data
    # for this analysis, use after_HL == False
    this_averaged_abrs = averaged_abrs_by_mouse.xs(
        False, level='after_HL').droplevel('HL_type')
    
    # Calculate the grand average (averaging out date and mouse)
    this_grand_average = this_averaged_abrs.groupby(
        [lev for lev in this_averaged_abrs.index.names if lev != 'mouse']
        ).mean()
    
    # Slice loudest sound
    loudest = this_grand_average.xs(loudest_db, level='label') * 1e6

    # Plot handles
    f, ax = plt.subplots(figsize=(4.5, 2.5))
    f.subplots_adjust(bottom=.24, left=.15, right=.93, top=.89)
    
    # Plot each
    # solid=left speaker, dashed=right speaker
    this_t = loudest.columns / sampling_rate * 1000
    ax.plot(
        this_t, loudest.loc['LV'].loc['R'], color='b', ls='-', lw=1, label='LV')
    ax.plot(
        this_t, loudest.loc['RV'].loc['R'], color='r', ls='-', lw=1, label='RV')
    ax.plot(
        this_t, loudest.loc['LR'].loc['R'], color='k', ls='-', lw=1, label='LR')
    ax.set_title('sound from right')
    
    # Legend
    ax.legend(loc='right', frameon=False, bbox_to_anchor=(1, 1), prop={'size': 12})
    
    # Pretty
    my.plot.despine(ax)
    ax.set_xlim(-1, 7)
    ax.set_xticks((0, 2, 4, 6))
    ax.set_ylim(-3, 3)
    ax.set_yticks((-3, 0, 3))
    ax.set_ylabel(f'ABR ({MU}V)')
    ax.set_xlabel('time (ms)')
    
    # Save figure
    savename = 'figures/GRAND_AVG_ONE_SIDE_ONLY'
    f.savefig(savename + '.svg')
    f.savefig(savename + '.png', dpi=300)

    
    ## Stats
    n_mice = len(this_averaged_abrs.index.get_level_values('mouse').unique())
    
    stats_filename = f'figures/STATS__GRAND_AVG_ONE_SIDE_ONLY'
    with open(stats_filename, 'w') as fi:
        fi.write(f'n = {n_mice} mice\n')
    
    # Echo
    print(stats_filename)
    with open(stats_filename) as fi:
        print(''.join(fi.readlines()))

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
    for (mouse, speaker_side, channel), subdf in this_corr_df.groupby(
            ['mouse', 'speaker_side', 'channel']):
        
        # droplevel
        subdf = subdf.droplevel(['mouse', 'speaker_side', 'channel'])
        
        # fit
        # slope units are delay in samples per dB
        # Multiply by 1000/16 to convert to us / dB
        # Should be about 7 us / dB, or ~300 us over the tested range
        fit = scipy.stats.linregress(subdf.index.values, subdf['idx'])
        
        # store slope, rval
        # rval should generally be between -1 and -0.85
        # slope should be mean -0.11, std 0.02
        rec_l.append((fit.slope, fit.rvalue))
        rec_keys_l.append((mouse, speaker_side, channel))

    # Concat
    # The fits are worse for speaker_side R
    midx = pandas.MultiIndex.from_tuples(
        rec_keys_l, names=['mouse', 'speaker_side', 'channel'])
    slope_df = pandas.DataFrame(
        rec_l, columns=['slope', 'rval'], index=midx)

    # Mean slope (in samples/dB)
    slope_by_mouse = slope_df['slope'].groupby('mouse').mean()
    
    # Convert to us/dB
    slope_by_mouse = slope_by_mouse / 16e3 * 1e6
    
    # Agg
    slope_by_mouse_mu = slope_by_mouse.mean()
    slope_by_mouse_std = slope_by_mouse.std()
    slope_by_mouse_sem = slope_by_mouse.sem()
    
    
    ## Make plot
    f, ax = my.plot.figure_1x1_standard()
    
    # First mean within mouse
    # This averages over all channels * speaker_side, for better or for worse
    # LV and RV are closer to linear, LR accelerates more with level, and
    # has a weird jog at the loudest levels
    to_agg = this_corr_df['idx'].groupby(
        ['mouse', 'label']).mean().unstack('mouse') / 16e3 * 1e3
    
    # Plot each mouse
    ax.plot(to_agg, lw=1, alpha=.5, color='k')

    # Now aggregate with mouse as N
    n_mice = to_agg.shape[1]
    topl_mu = to_agg.mean(axis=1)
    topl_err = to_agg.sem(axis=1)
    #~ ax.plot(topl_mu, lw=1.5, color='r')
    #~ ax.fill_between(
        #~ x=topl_mu.index,
        #~ y1=topl_mu - topl_err,
        #~ y2=topl_mu + topl_err,
        #~ alpha=.5, lw=0, color='k',
        #~ )
    
    # Pretty
    my.plot.despine(ax)
    ax.set_xlim((35, 75))
    ax.set_xticks((40, 50, 60, 70))
    ax.set_ylim((0, 0.4))
    ax.set_yticks((0, 0.2, 0.4))
    
    # Label
    ax.set_xlabel('sound level (dB)')
    ax.set_ylabel('delay (ms)')
    
    # Legend
    ax.text(50, 0.35, f'n = {n_mice} mice', ha='center', va='center')
    
    
    ## Save figure
    savename = 'figures/PLOT_DELAY_VS_LEVEL'
    f.savefig(savename + '.svg')
    f.savefig(savename + '.png', dpi=300)
    
    
    ## Stats
    with open('figures/STATS__PLOT_DELAY_VS_LEVEL', 'w') as fi:
        fi.write('take corr_df defined above, excluding after_HL ')
        fi.write('(mouse * channel * speaker_side * label)\n')
        fi.write('groupby mouse * label, meaning out channel * speaker_side\n')
        fi.write('plot the above, with SEM over mice\n')
        fi.write(f'n = {n_mice} mice\n')
        fi.write('also fit a slope over level for each (mouse * channel * speaker_side)\n')
        fi.write('then average those slopes within mouse over channel * speaker_side\n')
        fi.write('then aggregate those slopes over mouse as below\n')
        fi.write(f'mean slope in {MU}s/dB: {slope_by_mouse_mu}\n')
        fi.write(f'SEM slope in {MU}s/dB: {slope_by_mouse_sem}\n')
        fi.write(f'STD slope in {MU}s/dB: {slope_by_mouse_std}\n')
    
    # Echo
    with open('figures/STATS__PLOT_DELAY_VS_LEVEL') as fi:
        for line in fi.readlines():
            print(line.strip())