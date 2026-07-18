## Computes rms(ABR) over level and thresholds
# Plots
#   PLOT_ABR_RMS_OVER_TIME
#   PLOT_GROWTH_FUNCTIONS
#   PLOT_ABR_POWER_VS_AGE
#   PLOT_ABR_POWER_VS_LEVEL
#   PLOT_ABR_POWER_VS_LEVEL_AFTER_HL
#   PLOT_ABR_POWER_VS_LEVEL_EARLY_VS_LATE_AFTER_HL
#   BASELINE_VS_N_TRIALS
#   HISTOGRAM_EVOKED_RMS_BY_LEVEL
#   PLOT_RMS_GROWTH_FUNCTIONS_W1_W4
#   PLOT_RMS_GROWTH_FUNCTIONS_AFTER_HL
# See Step5b_abrpresto for everything that depends on ABRpresto

import os
import json
import scipy.stats
import numpy as np
import pandas
import my.plot
import matplotlib.pyplot as plt
import matplotlib
import shared


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
sampling_rate = 16000

# Conditions used by the blocks that iterate over mice
# (suffix, dict of level->value to slice)
condition_l = [
    ('preHL', dict(after_HL=False)),
    ('postHL_bilateral', dict(after_HL=True, HL_type='bilateral')),
    ('postHL_sham', dict(after_HL=True, HL_type='sham')),
    ]


## Load metadata
metadata = shared.load_metadata(raw_data_directory)

# Parse out
mouse_metadata = metadata['mouse_metadata'].copy()
experiment_metadata = metadata['experiment_metadata'].copy()

# TODO: drop upstream
mouse_metadata = mouse_metadata.drop('Pineapple_197')
experiment_metadata = experiment_metadata.drop('Pineapple_197', level='mouse')


## Load previous results
# Load results of Step2b_avg
big_abrs = pandas.read_parquet(
    os.path.join(output_directory, 'big_abrs'))
trial_counts = pandas.read_parquet(
    os.path.join(output_directory, 'trial_counts'))


## Calculate the stdev(ABR) as a function of level
# window=20 (1.25 ms) seems the best compromise between smoothing the whole
# response and localizing it to a reasonably narrow window (and not extending
# into the baseline period)
# Would be good to extract more baseline to use here
# The peak is around sample 32 (2.0 ms), ie sample 22 - 42, and there is a
# variable later peak.
big_abr_stds = big_abrs.T.rolling(window=20, center=True, min_periods=1).std().T


## Calculate the baseline
# Use samples -40 to -20 as baseline
# Generally this should be 50-100 nV, depending on the number of trials
big_abr_baseline_rms = big_abr_stds.loc[:, -30].unstack('label')

# Choose a baseline for each recording as the median over levels
# It's lognormal so mean might be skewed. A mean of log could be good
big_abr_baseline_rms = big_abr_baseline_rms.median(axis=1)


## Calculate the evoked 
# Use samples 22 - 42 as evoked peak
# Evoked response increases linearly with level in dB
# Interestingly, each recording appears to be multiplicatively scaled
# (shifted up and down on a log plot). The variability in microvolts increases
# with level, but the variability in log-units is consistent over level.
big_abr_evoked_rms = big_abr_stds.loc[:, 32].unstack('label')

# Aggregate over recordings within a date
big_abr_evoked_rms = big_abr_evoked_rms.groupby(
    [lev for lev in big_abr_evoked_rms.index.names if lev != 'recording'],
    ).mean()

# Always take first experiment
big_abr_evoked_rms = big_abr_evoked_rms.xs(0, level='n_experiment')


## Do the same for the late response
big_abr_evoked_rms_late = big_abr_stds.loc[:, 96].unstack('label')

# Aggregate over recordings within a date
big_abr_evoked_rms_late = big_abr_evoked_rms_late.groupby(
    [lev for lev in big_abr_evoked_rms_late.index.names if lev != 'recording'],
    ).mean()

# Always take first experiment
big_abr_evoked_rms_late = big_abr_evoked_rms_late.xs(0, level='n_experiment')


## Calculate the threshold
# Keep a copy before interpolating
big_abr_evoked_rms_orig = big_abr_evoked_rms.copy()

# Reindex to 0.1 dB resolution
new_sound_levels = pandas.Index(np.arange(
    big_abr_evoked_rms.columns.min(),
    big_abr_evoked_rms.columns.max() + 1e-6,
    0.1,
    ), name=big_abr_evoked_rms.columns.name)

# Interpolate (linearly)
big_abr_evoked_rms = big_abr_evoked_rms.reindex(
    new_sound_levels.union(big_abr_evoked_rms.columns), axis=1).interpolate(
    axis=1, method='index').reindex(new_sound_levels, axis=1)
assert not big_abr_evoked_rms.isnull().any().any()

# Apply a fixed threshold in uV
over_thresh = big_abr_evoked_rms.T > 0.3e-6 
over_thresh = over_thresh.T.stack()

# threshold
# typically a bit better on LR even though LR has slightly higher baseline
threshold_db = over_thresh.loc[over_thresh.values].groupby(
    [lev for lev in over_thresh.index.names if lev != 'label'],
    ).apply(
    lambda df: df.index[0][-1])

# reindex to get those that are never above threshold
threshold_db = threshold_db.reindex(big_abr_evoked_rms.index)

# error check that we always have a threshold
assert not threshold_db.isnull().any()


## Plots
PLOT_ABR_RMS_OVER_TIME = True
PLOT_GROWTH_FUNCTIONS = True
PLOT_ABR_POWER_VS_AGE = True
PLOT_ABR_POWER_VS_LEVEL = True
PLOT_ABR_POWER_VS_LEVEL_AFTER_HL = True
BASELINE_VS_N_TRIALS = True
PLOT_RMS_GROWTH_FUNCTIONS_W1_W4 = True
PLOT_RMS_GROWTH_FUNCTIONS_AFTER_HL = True


if PLOT_ABR_RMS_OVER_TIME:
    """Plot the smoothed rms of the ABR over time, colored by sound level
    
    One figure per condition (preHL, postHL_bilateral, postHL_sham).
    Mouse is the replicate; plots mean +- sem over mice.
    """
    
    ## Params
    # Shared t
    t = big_abrs.columns / sampling_rate * 1000
    
    # Panels
    channel_l = ['VL', 'VR', 'RL']
    speaker_side_l = ['L', 'R']
    
    
    ## One figure per condition
    for suffix, sel in condition_l:
        
        # Skip sham
        if suffix == 'postHL_sham':
            continue
        
        # Slice big_abr_stds to this condition
        this_stds = big_abr_stds
        for k, v in sel.items():
            this_stds = this_stds.xs(v, level=k)
        
        # HL_type survives the preHL slice; drop it so all conditions match
        if 'HL_type' in this_stds.index.names:
            this_stds = this_stds.droplevel('HL_type')
        
        # Aggregate over recordings within a date
        to_agg = this_stds.groupby(
            [lev for lev in this_stds.index.names if lev != 'recording']
            ).mean()
        
        # Always take first experiment
        to_agg = to_agg.xs(0, level='n_experiment')
        
        # Make mouse the replicates on the columns
        to_agg = to_agg.stack().unstack('mouse')
        
        # Mean and sem over mice
        agg_mean = to_agg.mean(axis=1).unstack('timepoint')
        agg_err = to_agg.sem(axis=1).unstack('timepoint')
        
        # Set up colorbar
        # Always do the lowest labels last
        label_l = sorted(
            agg_mean.index.get_level_values('label').unique(), 
            reverse=True)
        aut_colorbar = my.plot.generate_colorbar(
            len(label_l), mapname='inferno_r', start=0.15, stop=1)  
        
        # Make plot
        f, axa = plt.subplots(
            len(channel_l), len(speaker_side_l),
            sharex=True, sharey=True, figsize=(5, 4))
        f.subplots_adjust(
            left=.17, right=.89, top=.95, bottom=.15, hspace=.15, wspace=.12)
        
        # Plot each channel * speaker_side
        for n_channel, channel in enumerate(channel_l):
            for n_speaker_side, speaker_side in enumerate(speaker_side_l):
                
                # Get ax
                ax = axa[n_channel, n_speaker_side]
                
                # Slice this config
                try:
                    subdf = agg_mean.xs(channel, level='channel').xs(
                        speaker_side, level='speaker_side')
                    subdf_err = agg_err.xs(channel, level='channel').xs(
                        speaker_side, level='speaker_side')
                except KeyError:
                    continue
                
                # Loudest first so the softest are drawn on top
                subdf = subdf.sort_index(ascending=False)
                
                # Plot the evoked peak time
                ax.plot([2, 2], [.03, 3], color='gray', lw=.75)
                
                # Plot each level
                for level in subdf.index:
                    
                    # Get color
                    color = aut_colorbar[label_l.index(level)]
                    
                    # Plot the mean (in uV)
                    ax.plot(t, subdf.loc[level] * 1e6, color=color, lw=1)  
                    
                    # Error bars
                    ax.fill_between(t, 
                        (subdf.loc[level] - subdf_err.loc[level]) * 1e6,
                        (subdf.loc[level] + subdf_err.loc[level]) * 1e6,
                        color=color, alpha=.5, lw=0)
                
                # Pretty
                my.plot.despine(ax) 
                ax.set_yscale('log')
                ax.set_yticks((0.1, 1.0))
                
                # Nicer log labels
                ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        
        
        ## Pretty
        # Legend
        for n_label, (label, color) in enumerate(zip(label_l, aut_colorbar)):
            if np.mod(n_label, 2) != 0:
                continue
            f.text(
                .95, .68 - n_label * .02, f'{label} dB',
                color=color, ha='center', va='center', size=12)
        
        # Shared limits
        ax.set_xlim((-1, 7))
        ax.set_ylim((.05, 2))
        ax.set_xticks([0, 2, 4, 6])
        
        # Axis labels
        f.text(.52, .01, 'time from sound onset (ms)', ha='center', va='bottom')
        f.text(.02, .56, f'response strength ({MU}V rms)', 
            rotation=90, ha='center', va='center')
        
        # Label the channel
        for n_channel, channel in enumerate(channel_l):
            axa[n_channel, 0].set_ylabel(channel)
        
        # Label the speaker side
        axa[0, 0].set_title('sound from left')
        axa[0, 1].set_title('sound from right')
        
        
        ## Savefig
        f.savefig(os.path.join('figures', 
            f'PLOT_ABR_RMS_OVER_TIME__{suffix}.svg'))
        f.savefig(os.path.join('figures', 
            f'PLOT_ABR_RMS_OVER_TIME__{suffix}.png'), dpi=300)
        
        
        ## Stats
        stats_filename = f'figures/STATS__PLOT_ABR_RMS_OVER_TIME__{suffix}'
        with open(stats_filename, 'w') as fi:
            fi.write(stats_filename + '\n')
            fi.write(f'n = {to_agg.shape[1]} mice\n')
            fi.write(
                'compute rolling RMS for each recording * level, '
                'then mean over recordings within date, '
                'then first experiment only, '
                'then mean and SEM over mice, '
                'then plot on log scale\n')
        
        # Echo
        with open(stats_filename) as fi:
            print(''.join(fi.readlines()))



if PLOT_GROWTH_FUNCTIONS:
    """Plot the smoothed rms of the ABR vs sound level, colored by wave
    
    One figure per condition (preHL, postHL_bilateral, postHL_sham).
    Mouse is the replicate; plots mean +- sem over mice.
    """
    
    ## Params
    # Sample the RMS at these timepoints (samples), approximating each wave
    wave_timepoints = list(np.rint(
        np.array([1.36, 2.3, 3.2, 4.2, 5.2]) * 16).astype(int))
    growth_wave_colors = plt.cm.tab10.colors[1:]
    
    # Panels
    channel_l = ['VL', 'VR', 'RL']
    speaker_side_l = ['L', 'R']
    
    
    ## One figure per condition
    for suffix, sel in condition_l:

        # Skip sham
        if suffix == 'postHL_sham':
            continue
            
        # Slice big_abr_stds to this condition
        this_stds = big_abr_stds
        for k, v in sel.items():
            this_stds = this_stds.xs(v, level=k)
        
        # HL_type survives the preHL slice; drop it so all conditions match
        if 'HL_type' in this_stds.index.names:
            this_stds = this_stds.droplevel('HL_type')
        
        # Aggregate over recordings within a date
        to_agg = this_stds.groupby(
            [lev for lev in this_stds.index.names if lev != 'recording']
            ).mean()
        
        # Always take first experiment
        to_agg = to_agg.xs(0, level='n_experiment')
        
        # Make mouse the replicates on the columns
        to_agg = to_agg.stack().unstack('mouse')
        
        # Mean and sem over mice
        agg_mean = to_agg.mean(axis=1).unstack('timepoint')
        agg_err = to_agg.sem(axis=1).unstack('timepoint')
        
        # Make plot
        f, axa = plt.subplots(
            len(channel_l), len(speaker_side_l),
            sharex=True, sharey=True, figsize=(5, 4))
        f.subplots_adjust(
            left=.17, right=.89, top=.95, bottom=.15, hspace=.15, wspace=.12)
        
        # Plot each channel * speaker_side
        for n_channel, channel in enumerate(channel_l):
            for n_speaker_side, speaker_side in enumerate(speaker_side_l):
                
                # Get ax
                ax = axa[n_channel, n_speaker_side]
                
                # Slice this config
                try:
                    subdf = agg_mean.xs(channel, level='channel').xs(
                        speaker_side, level='speaker_side')
                    subdf_err = agg_err.xs(channel, level='channel').xs(
                        speaker_side, level='speaker_side')
                except KeyError:
                    continue
                
                # Plot each wave timepoint
                for n_timepoint, timepoint in enumerate(wave_timepoints):
                    
                    # Get color
                    color = growth_wave_colors[n_timepoint]
                    
                    # Plot the mean (in uV)
                    ax.plot(
                        subdf.index, 
                        subdf.loc[:, timepoint] * 1e6, 
                        color=color, 
                        lw=1)  
                    
                    # Error bars
                    ax.fill_between(
                        subdf.index,
                        (subdf.loc[:, timepoint] - subdf_err.loc[:, timepoint]) * 1e6,
                        (subdf.loc[:, timepoint] + subdf_err.loc[:, timepoint]) * 1e6,
                        color=color, alpha=.5, lw=0)
                
                # Pretty
                my.plot.despine(ax) 
                ax.set_yscale('log')
                ax.set_yticks((0.1, 1.0))
                
                # Nicer log labels
                ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        
        
        ## Pretty
        # Legend
        for n_timepoint, timepoint in enumerate(wave_timepoints):
            f.text(
                .95, .68 - n_timepoint * .04, f't ~ Wave {n_timepoint + 1}',
                color=growth_wave_colors[n_timepoint], 
                ha='center', va='center', size=12)
        
        # Shared limits
        ax.set_xlim((20, 80))
        ax.set_ylim((.05, 2))
        ax.set_xticks([30, 50, 70])
        
        # Axis labels
        f.text(.52, .01, 'sound level (dB SPL)', ha='center', va='bottom')
        f.text(.02, .56, f'response strength ({MU}V rms)', 
            rotation=90, ha='center', va='center')
        
        # Label the channel
        for n_channel, channel in enumerate(channel_l):
            axa[n_channel, 0].set_ylabel(channel)
        
        # Label the speaker side
        axa[0, 0].set_title('sound from left')
        axa[0, 1].set_title('sound from right')
        
        
        ## Savefig
        f.savefig(os.path.join('figures', 
            f'PLOT_GROWTH_FUNCTIONS__{suffix}.svg'))
        f.savefig(os.path.join('figures', 
            f'PLOT_GROWTH_FUNCTIONS__{suffix}.png'), dpi=300)


if PLOT_ABR_POWER_VS_AGE:
    ## Correlate response magnitude with age of mouse
    # Slice out pre-HL only
    this_threshold_db = threshold_db.xs(
        False, level='after_HL').droplevel('HL_type')
    this_big_abr_evoked_rms = big_abr_evoked_rms.xs(
        False, level='after_HL').droplevel('HL_type')
    
    # Compute average mouse age at time of recordings
    mouse_age = experiment_metadata[
        experiment_metadata['after_HL'] == False
        ].groupby('mouse')['age'].mean()
    
    # Average response by mouse across speaker_side * channel
    # Note: This is significant for all channels and speaker_side separately,
    # though perhaps weaker for LR
    mouse_response = this_big_abr_evoked_rms.iloc[:, -1].groupby('mouse').mean()
    
    # Average thresh by mouse across speaker_side * channel
    mouse_thresh = this_threshold_db.groupby('mouse').mean()
    
    # Concat
    age_data = pandas.DataFrame({
        'age': mouse_age,
        'response': mouse_response,
        'threshold': mouse_thresh,
        })
    
    # Join sex
    age_data = age_data.join(mouse_metadata['sex'])
    
    # Join experimenter
    mouse_experimenter = experiment_metadata.groupby(['mouse'])[
        'experimenter'].apply(lambda df:df.unique().item())
    age_data = age_data.join(mouse_experimenter)
    
    
    ## Plot response magnitude
    f, ax = my.plot.figure_1x1_small()
    
    # Plot males and females separately
    ax.plot(
        age_data.loc[age_data['sex'] == 'M', 'age'], 
        age_data.loc[age_data['sex'] == 'M', 'response'] * 1e6, 
        'bo', mfc='none')
    ax.plot(
        age_data.loc[age_data['sex'] == 'F', 'age'], 
        age_data.loc[age_data['sex'] == 'F', 'response'] * 1e6, 
        'ro', mfc='none')
    
    # Pretty
    ax.set_xticks((90, 180, 270))
    ax.set_xlabel('age (days)')
    ax.set_yticks((1, 1.5, 2))
    ax.set_ylabel(f'response strength\n({MU}V rms)')    
    my.plot.despine(ax)
    
    # Savefig
    f.savefig('figures/PLOT_ABR_POWER_VS_AGE__response.svg')
    f.savefig('figures/PLOT_ABR_POWER_VS_AGE__response.png', dpi=300)
    
    
    ## Plot thresh
    f, ax = my.plot.figure_1x1_small()
    
    # Plot males and females separately
    ax.plot(
        age_data.loc[age_data['sex'] == 'M', 'age'], 
        age_data.loc[age_data['sex'] == 'M', 'threshold'], 
        'bo', mfc='none')
    ax.plot(
        age_data.loc[age_data['sex'] == 'F', 'age'], 
        age_data.loc[age_data['sex'] == 'F', 'threshold'], 
        'ro', mfc='none')    

    # Pretty
    ax.set_xticks((90, 180, 270))
    ax.set_xlabel('age (days)')
    ax.set_yticks((30, 35, 40))
    ax.set_ylabel('threshold (dB SPL)')
    my.plot.despine(ax)
    
    # Savefig
    f.savefig('figures/PLOT_ABR_POWER_VS_AGE__thresh.svg')
    f.savefig('figures/PLOT_ABR_POWER_VS_AGE__thresh.png', dpi=300)
    
    
    ## Stats
    # Correlate both
    vs_response = scipy.stats.linregress(age_data['age'], age_data['response'])
    vs_thresh = scipy.stats.linregress(age_data['age'], age_data['threshold'])
    
    # Write out stats
    n_mice = len(age_data)
    stats_filename = 'figures/STATS__PLOT_ABR_POWER_VS_AGE'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write(f'n = {n_mice} mice, pre-HL only\n')
        fi.write(
            'including pre-HL sessions only, '
            'meaned thresh and rms response at highest level over '
            'channel * speaker_side (already meaned within mouse)\n'
            'then correlated with age (meaned over recordings)\n'
            )
        fi.write(
            f'response vs age: p={vs_response.pvalue:.4f}; r={vs_response.rvalue:.2f}; r2={vs_response.rvalue ** 2:.2f}\n'
            f'thresh vs age: p={vs_thresh.pvalue:.4f}; r={vs_thresh.rvalue:.2f}; r2={vs_thresh.rvalue ** 2:.2f}\n'
            )
    
    # Echo
    with open(stats_filename) as fi:
        print(''.join(fi.readlines()))    


if PLOT_ABR_POWER_VS_LEVEL:
    ## Plot ABR rms power vs sound level for all mice together
    # Both the mean and standard deviation of evoked power strongly increase
    # with sound level, suggesting we should take log of power. 
    # On a semilog plot, the variance is similar across level, and evoked
    # power shows diminishing increases with level, not quite plateauing. 
    
    # Slice out pre-HL only
    this_threshold_db = threshold_db.xs(
        False, level='after_HL').droplevel('HL_type')
    this_big_abr_evoked_rms = big_abr_evoked_rms.xs(
        False, level='after_HL').droplevel('HL_type')

    # Set up ax_rows and ax_cols
    channel_l = ['VL', 'VR', 'RL']
    speaker_side_l = ['L', 'R']
    
    # Make plot
    f, axa = plt.subplots(
        len(channel_l), len(speaker_side_l),
        sharex=True, sharey=True, figsize=(5, 4))
    f.subplots_adjust(
        left=.17, right=.89, top=.95, bottom=.15, hspace=.15, wspace=.12)

    # Iterate over channel * speaker_side
    gobj = this_big_abr_evoked_rms.groupby(['channel', 'speaker_side'])
    for (channel, speaker_side), subdf in gobj:
        
        # Drop grouping keys
        subdf = subdf.droplevel(['channel', 'speaker_side'])

        # Get ax
        ax = axa[
            channel_l.index(channel),
            speaker_side_l.index(speaker_side),
        ]

        # Plot all mice for this config
        for mouse in subdf.index:
            # Get power vs level for this mouse
            topl = subdf.loc[mouse].copy()

            # Get avg threshold for this mouse
            thresh = this_threshold_db.loc[
                mouse].loc[channel].loc[speaker_side].item()

            # Because thresh is a mean over recordings, it doesn't
            # align with evoked RMS. Resample to make the point lie
            # on the line
            thresh_y = np.interp([thresh], topl.index, topl.values)

            # Plot evoked RMS
            ax.semilogy(topl * 1e6, color='gray', lw=.75, alpha=.5)
            
            # Plot thresh
            ax.semilogy(
                [thresh], [thresh_y * 1e6], 
                marker='o', color='red', mfc='none', ms=2.5, alpha=.5)


        # Despine
        my.plot.despine(ax)
        ax.set_yticks([0.1, 1])
        ax.set_xticks([30, 50, 70])
        ax.set_xlim((20, 80))
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())


    ## Pretty
    # Shared axis labes
    f.text(.52, .01, 'sound level (dB SPL)', ha='center', va='bottom')
    f.text(
        .02, .56, f'response strength ({MU}V rms)', 
        rotation=90, ha='center', va='center')

    # Label the channel
    for n_channel, channel in enumerate(channel_l):
        axa[n_channel, 0].set_ylabel(channel)
    
    # Label the speaker side
    axa[0, 0].set_title('sound from left')
    axa[0, 1].set_title('sound from right')

    
    ## Savefig
    f.savefig('figures/PLOT_ABR_POWER_VS_LEVEL.svg')
    f.savefig('figures/PLOT_ABR_POWER_VS_LEVEL.png', dpi=300)


    ## Stats
    n_mice = len(this_threshold_db.index.get_level_values('mouse').unique())
    thresh_by_mouse = this_threshold_db.groupby('mouse').mean()
    thresh_by_mouse_mu = thresh_by_mouse.mean()
    thresh_by_mouse_std = thresh_by_mouse.std()
    thresh_by_mouse_sem = thresh_by_mouse.sem()
    thresh_by_channel = this_threshold_db.groupby('channel').mean()
    ipsi_thresh = this_threshold_db.unstack('mouse').loc[
        [('VL', 'L'), ('VR', 'R')]].mean().mean()
    contra_thresh = this_threshold_db.unstack('mouse').loc[
        [('VR', 'L'), ('VL', 'R')]].mean().mean()
    stats_filename = 'figures/STATS__PLOT_ABR_POWER_VS_LEVEL'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write(f'n = {n_mice} mice, pre-HL only\n')
        fi.write(
            'mean power vs level over recordings within date, '
            'then first experiment only.\n'
            'computed one threshold (interpolated) per mouse\n'
            )
        fi.write(
            f'thresh over mice: mean {thresh_by_mouse_mu:.1f}, '
            f'std {thresh_by_mouse_std:.1f}, sem {thresh_by_mouse_sem:.2f}\n')
        fi.write(f'thresh over channel: \n{thresh_by_channel} \n')
        fi.write(f'thresh ipsi: {ipsi_thresh:.1f}; thresh contra: {contra_thresh:.1f}\n')
    
    # Echo
    with open(stats_filename) as fi:
        print(''.join(fi.readlines()))


if PLOT_ABR_POWER_VS_LEVEL_AFTER_HL:
    ## Plot sham and bilateral pre- and post-HL
    
    # Include only HL mice
    this_threshold_db = threshold_db.drop('none', level='HL_type')
    this_big_abr_evoked_rms = big_abr_evoked_rms.drop('none', level='HL_type')

    # Aggregate threshold over recording, maintaining after_HL
    threshold_db_agg = this_threshold_db.groupby(
        [lev for lev in this_threshold_db.index.names if lev != 'recording']
        ).mean()

    # Aggregate evoked RMS over recording, maintaining after_HL
    big_abr_evoked_rms_agg = this_big_abr_evoked_rms.groupby(
        [lev for lev in this_big_abr_evoked_rms.index.names if lev != 'recording']
        ).mean()
        
    
    ## To iterate over
    channel_l = ['VL', 'VR', 'RL']
    speaker_side_l = ['L', 'R']
    HL_type_l = ['bilateral', 'sham']
    after_HL_l = [False, True]
    
    
    ## Plot
    # Iterate over speaker_side (figures)
    for speaker_side in speaker_side_l:
        
        ## First figure: evoked power vs sound level
        f, axa = plt.subplots(
            len(channel_l), len(HL_type_l),
            sharex=True, sharey=True, figsize=(5, 4))
        f.subplots_adjust(
            left=.17, right=.89, top=.95, bottom=.15, hspace=.15, wspace=.12)

        # Iterate over channel * HL_type
        gobj = big_abr_evoked_rms_agg.xs(
            speaker_side, level='speaker_side').groupby(
            ['channel', 'HL_type'])
        for (channel, HL_type), subdf in gobj:
            
            # Get ax
            ax = axa[
                channel_l.index(channel),
                HL_type_l.index(HL_type),
            ]
            
            # Drop grouping keys
            subdf = subdf.droplevel(['channel', 'HL_type'])
    
            # Iterate over mice
            for mouse in subdf.index.get_level_values('mouse').unique():
                
                # Iterate over after_HL
                for after_HL in after_HL_l:

                    # Color by after_HL
                    color = 'magenta' if after_HL else 'green'
                    
                    # Slice evoked RMS and threshold
                    topl = subdf.loc[after_HL].loc[mouse]
                    thresh = threshold_db_agg.loc[
                        HL_type].loc[after_HL].loc[mouse].loc[channel].loc[
                        speaker_side].item()                

                    # Because thresh is a mean over recordings, it doesn't
                    # align with evoked RMS. Resample to make the point lie
                    # on the line
                    thresh_y = np.interp([thresh], topl.index, topl.values)

                    # Plot evoked RMS
                    ax.semilogy(topl * 1e6, color=color, lw=.75)
                    
                    # Plot thresh
                    ax.semilogy(
                        [thresh], [thresh_y * 1e6], 
                        marker='o', color=color, mfc='none')

            # Label the y-axis with the channel
            if ax in axa[:, 0]:
                ax.set_ylabel(channel)
            if ax == axa[1, 0]:
                ax.set_ylabel(f'responses strength ({MU}V)\n{channel}')
            
            # Despine
            my.plot.despine(ax)
            ax.set_yticks([0.1, 1])
            ax.set_xticks([30, 50, 70])
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

        # Shared x-axis
        f.text(.52, .01, 'sound level (dB SPL)', ha='center', va='bottom')

        # Label the HL_type
        axa[0, 0].set_title(HL_type_l[0])
        axa[0, 1].set_title(HL_type_l[1])
        
        # Legend
        f.text(.95, .6, 'pre', color='green', ha='center', va='center')
        f.text(.95, .54, 'post', color='magenta', ha='center', va='center')

        # Savefig
        savename = f'figures/PLOT_ABR_POWER_VS_LEVEL_AFTER_HL__{speaker_side}'
        f.savefig(savename + '.svg')
        f.savefig(savename + '.png', dpi=300)        
        
        
        ## Second plot of threshold
        f, axa = plt.subplots(
            len(channel_l), len(HL_type_l),
            sharex=True, sharey=True, figsize=(5, 4))
        f.subplots_adjust(
            left=.17, right=.89, top=.95, bottom=.15, hspace=.15, wspace=.12)

        # Iterate over channel * HL_type
        gobj = threshold_db_agg.xs(
            speaker_side, level='speaker_side').groupby(
            ['channel', 'HL_type'])
        for (channel, HL_type), subdf in gobj:
            
            # Drop grouping keys
            subdf = subdf.droplevel(['channel', 'HL_type'])
            
            # Get mouse on columns
            subdf = subdf.unstack('mouse')
            
            # Make sure it's pre before post
            subdf = subdf.reindex([False, True])
            
            # Get ax
            ax = axa[
                channel_l.index(channel),
                HL_type_l.index(HL_type),
            ]        
            
            # Plot
            ax.plot(subdf.values, ls='-', marker='o', color='k', mfc='none')

            # Label the y-axis with the channel
            if ax in axa[:, 0]:
                ax.set_ylabel(channel)
            if ax == axa[1, 0]:
                ax.set_ylabel(f'threshold (dB SPL)\n{channel}')
            
            # Despine
            my.plot.despine(ax)

        # Shared x-axis
        f.text(.52, .01, 'before or after hearing loss', ha='center', va='bottom')

        # Label the HL_type
        axa[0, 0].set_title(HL_type_l[0])
        axa[0, 1].set_title(HL_type_l[1])
        
        # Consistent axis limits
        ax.set_ylim((20, 70))
        ax.set_yticks((30, 45, 60))
        ax.set_xticks((0, 1))
        ax.set_xlim((-.5, 1.5))
        ax.set_xticklabels(('pre', 'post'))
        
        # Savefig
        savename = f'figures/PLOT_ABR_POWER_VS_LEVEL_AFTER_HL__thresh__{speaker_side}'
        f.savefig(savename + '.svg')
        f.savefig(savename + '.png', dpi=300)


        ## Stats
        # Threshold (meaning over channel * speaker_side)
        thresh_by_mouse = threshold_db_agg.groupby(
            ['HL_type', 'after_HL', 'mouse']).mean().unstack('after_HL')
        thresh_by_mouse_mu = thresh_by_mouse.groupby('HL_type').mean()
        thresh_by_mouse_err = thresh_by_mouse.groupby('HL_type').sem()
        thresh_by_mouse_N = thresh_by_mouse.groupby('HL_type').size()
        thresh_by_mouse_diff = thresh_by_mouse.diff(axis=1).iloc[:, 1]
        thresh_by_mouse_diff_mu = thresh_by_mouse_diff.groupby('HL_type').mean()
        thresh_by_mouse_diff_std = thresh_by_mouse_diff.groupby('HL_type').std()
        thresh_by_mouse_diff_err = thresh_by_mouse_diff.groupby('HL_type').sem()
        thresh_by_mouse_diff_N = thresh_by_mouse_diff.groupby('HL_type').size()
        
        # Write out
        # This is redundant over speaker_side
        stats_filename = (
            'figures/STATS__PLOT_ABR_POWER_VS_LEVEL_AFTER_HL__thresh__'
            f'{speaker_side}')
        with open(stats_filename, 'w') as fi:
            fi.write(stats_filename + '\n')
            fi.write(f'these calculations are all meaned over channel * speaker_side\n')
            fi.write(f'raw changes after_HL by mouse:\n{thresh_by_mouse.diff(axis=1).iloc[:, 1]}\n')
            fi.write(f'mean threshold:\n{thresh_by_mouse_mu}\n')
            fi.write(f'SEM threshold:\n{thresh_by_mouse_err}\n')
            fi.write(f'N threshold:\n{thresh_by_mouse_N}\n')
            fi.write(f'mean diff threshold:\n{thresh_by_mouse_diff_mu}\n')
            fi.write(f'STD diff threshold:\n{thresh_by_mouse_diff_std}\n')
            fi.write(f'SEM diff threshold:\n{thresh_by_mouse_diff_err}\n')
            fi.write(f'N diff threshold:\n{thresh_by_mouse_diff_N}\n')
        
        # Echo
        with open(stats_filename) as fi:
            print(''.join(fi.readlines()))        


if BASELINE_VS_N_TRIALS:
    ## Plot the noise level as a function of trial count
    
    # Take the noise level as the baseline rms (median over levels)
    # Only slightly higher in LR (80 nV vs 75 nV) so pool over channels
    med_baseline_rms = big_abr_baseline_rms

    # Take the trial count as the median over levels
    med_trial_count = trial_counts.unstack('label').median(axis=1)

    # Concat these two so they line up
    joined = med_trial_count.rename(
        'trial_count').to_frame().join(med_baseline_rms.rename('noise'))

    # Plot
    f, ax = plt.subplots(figsize=(3.5, 2.5))
    f.subplots_adjust(bottom=.24, left=.25, right=.95, top=.89)
    ax.plot(
        np.log10(joined['trial_count']), 
        np.log10(joined['noise'] * 1e9), # log(nV)
        color='k', marker='o', mfc='none', ls='none', alpha=.3)

    # Pretty
    my.plot.despine(ax)
    ax.set_xticks(np.log10([30, 100, 300]))
    ax.set_xticklabels((30, 100, 300))
    ax.set_yticks(np.log10([30, 100, 300])) # log(nV)
    ax.set_yticklabels((0.03, 0.1, 0.3))
    ax.axis('scaled')
    ax.set_ylim((1.4, 2.6))
    ax.set_xlim((1.1, 2.9))

    # Labels
    ax.set_ylabel(f'baseline ({MU}V rms)')
    ax.set_xlabel('number of trials')

    # Plot a slope line
    # TODO: actually fit this and extract slope
    ax.plot(np.log10([10, 1000]), np.log10([300, 30]) - .2, 'k--', lw=.75)    

    # Savefig
    f.savefig('figures/BASELINE_VS_N_TRIALS.svg')
    f.savefig('figures/BASELINE_VS_N_TRIALS.png', dpi=300)


    ## Stats
    n_recordings_channels = len(joined)
    n_recordings = n_recordings_channels // 3
    n_mice = len(joined.index.get_level_values('mouse').unique())
    stats_filename = 'figures/STATS__BASELINE_VS_N_TRIALS'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write(
            f'n = {n_recordings} recordings * 3 channels = '
            f'{n_recordings_channels} points from {n_mice} mice, '
            'pre- and post-HL\n'
            )
        fi.write(
            'trial count is median over sound level\n'
            'baseline is rms in the (-2.5, -1.25) ms range, '
            'median over sound level\n'
            )
    
    # Echo
    with open(stats_filename) as fi:
        print(''.join(fi.readlines()))
    



if PLOT_RMS_GROWTH_FUNCTIONS_W1_W4:
    """RMS growth functions at the W1 and W4 timepoints
    
    RMS-based analogue of PLOT_PEAK_GROWTH_FUNCTIONS, so it can include RL.
    Samples the smoothed RMS at the W1 and W4 timepoints (not true wave
    amplitudes; below threshold these approach the noise floor).
    This is run only for the first recording of each mouse (one per mouse),
    so mouse is the replicate. One line per mouse.
    """
    
    ## Params
    # Timepoints for W1 and W4 (samples), matching PLOT_GROWTH_FUNCTIONS
    wave_timepoints = {
        'W1': int(np.rint(1.36 * 16)),
        'W4': int(np.rint(4.2 * 16)),
        }
    rms_wave_colors = {'W1': 'tab:orange', 'W4': 'tab:purple'}
    
    # Panels
    channel_l = ['VL', 'VR', 'RL']
    speaker_side_l = ['L', 'R']
    
    
    ## Slice big_abr_stds to first pre-HL recording only
    this_stds = big_abr_stds.xs(False, level='after_HL').droplevel('HL_type')
    
    # Aggregate over recordings within a date
    this_stds = this_stds.groupby(
        [lev for lev in this_stds.index.names if lev != 'recording']).mean()
    
    # Always take first experiment
    this_stds = this_stds.xs(0, level='n_experiment')
    
    # Slice RMS at the two timepoints; wide on timepoint -> stack to a
    # 'wave_name' level. index=(mouse, channel, speaker_side, label, wave_name)
    rms = this_stds.loc[:, list(wave_timepoints.values())]
    rms.columns = pandas.Index(wave_timepoints.keys(), name='wave_name')
    rms = rms.stack()
    
    # Mice as replicates (one column per mouse)
    rms_by_mouse = rms.unstack('mouse').sort_index()
    
    
    ## Figure
    f, axa = plt.subplots(
        len(channel_l), len(speaker_side_l),
        sharex=True, sharey=True, figsize=(5, 5.5))
    f.subplots_adjust(
        left=.17, right=.89, top=.93, bottom=.11, hspace=.15, wspace=.12)
    
    # Plot each channel * speaker_side
    for n_channel, channel in enumerate(channel_l):
        for n_speaker_side, speaker_side in enumerate(speaker_side_l):
            
            # Get ax
            ax = axa[n_channel, n_speaker_side]
            
            # Slice this config; index=(label, wave_name), cols=mouse
            this_config = rms_by_mouse.xs(channel, level='channel').xs(
                speaker_side, level='speaker_side')
            
            # Plot each wave
            for wave_name in wave_timepoints:
                
                # label on rows, mice on columns
                this_traces = this_config.xs(wave_name, level='wave_name')
                
                # One line per mouse, x=label
                for mouse in this_traces.columns:
                    ax.plot(
                        this_traces.index, this_traces[mouse].values * 1e6,
                        color=rms_wave_colors[wave_name], lw=.75)
            
            # Pretty
            my.plot.despine(ax)
            ax.set_yscale('log')
            ax.set_ylim((.05, 2))
            ax.set_xlim((20, 80))
            ax.set_xticks((30, 50, 70))
            ax.set_yticks((0.1, 1.0))
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    
    ## Pretty
    # Legend
    for n_wave, wave_name in enumerate(wave_timepoints):
        f.text(
            .95, .66 - n_wave * .05, wave_name,
            color=rms_wave_colors[wave_name], 
            ha='center', va='center', size=12)
    
    # Axis labels
    f.text(.52, .01, 'sound level (dB SPL)', ha='center', va='bottom')
    f.text(.02, .55, f'response strength ({MU}V rms)',
        rotation=90, ha='center', va='center')
    
    # Label the channel
    for n_channel, channel in enumerate(channel_l):
        axa[n_channel, 0].set_ylabel(channel)
    
    # Label the speaker side
    axa[0, 0].set_title('sound from left')
    axa[0, 1].set_title('sound from right')
    
    
    ## Savefig
    f.savefig(os.path.join('figures', 'PLOT_RMS_GROWTH_FUNCTIONS_W1_W4.svg'))
    f.savefig(os.path.join('figures', 'PLOT_RMS_GROWTH_FUNCTIONS_W1_W4.png'), 
        dpi=300)


if PLOT_RMS_GROWTH_FUNCTIONS_AFTER_HL:
    """RMS growth functions after HL, one figure per wave, colored by HL_type
    
    RMS-based analogue of PLOT_PEAK_GROWTH_FUNCTIONS_AFTER_HL. Samples the
    smoothed RMS at the W1 and W4 timepoints. after_HL==True only. One line
    per recording, colored by HL_type (bilateral red, sham gray). One figure
    per wave (W1, W4); each is a 3x2 channel * speaker_side grid.
    """
    
    ## Params
    # Timepoints for W1 and W4 (samples), matching PLOT_GROWTH_FUNCTIONS
    wave_timepoints = {
        'W1': int(np.rint(1.36 * 16)),
        'W4': int(np.rint(4.2 * 16)),
        }

    # Color by HL_type
    hl_colors = {'bilateral': 'red', 'sham': 'gray'}

    # Panels
    channel_l = ['VL', 'VR', 'RL']
    speaker_side_l = ['L', 'R']

    # The replicate unit (one line per recording)
    recording_levels = ['HL_type', 'mouse']


    ## Slice to post-HL, drop unoperated
    this_stds = big_abr_stds.xs(True, level='after_HL').drop(
        'none', level='HL_type')

    # Aggregate over recordings within a date
    to_agg = this_stds.groupby(
        [lev for lev in this_stds.index.names if lev != 'recording']
        ).mean()

    # Take first experiment only (matches peak analyses)
    to_agg = to_agg.xs(0, level='n_experiment')


    ## One figure per wave
    for wave_name, timepoint in wave_timepoints.items():

        # RMS at this timepoint; index=(HL_type, mouse, channel,
        # speaker_side, label), value=rms
        rms = to_agg.loc[:, timepoint]

        # Recordings onto columns, (config, label) on rows
        to_plot = rms.unstack(recording_levels)

        # Figure
        f, axa = plt.subplots(
            len(channel_l), len(speaker_side_l),
            sharex=True, sharey=True, figsize=(5, 5.5))
        f.subplots_adjust(
            left=.17, right=.89, top=.93, bottom=.11, hspace=.15, wspace=.12)

        # Plot each channel * speaker_side
        for n_channel, channel in enumerate(channel_l):
            for n_speaker_side, speaker_side in enumerate(speaker_side_l):

                # Get ax
                ax = axa[n_channel, n_speaker_side]

                # Slice this config; index=label, cols=recording
                try:
                    this_traces = to_plot.xs(channel, level='channel').xs(
                        speaker_side, level='speaker_side').sort_index()
                except KeyError:
                    continue

                # One line per recording, colored by HL_type
                for recording in this_traces.columns:

                    # recording is a tuple (HL_type, mouse)
                    hl_type = recording[recording_levels.index('HL_type')]

                    # Plot this recording's growth function (in uV)
                    ax.plot(
                        this_traces.index, 
                        this_traces[recording].values * 1e6,
                        color=hl_colors[hl_type], lw=.75)

                # Pretty
                my.plot.despine(ax)
                ax.set_yscale('log')
                ax.set_ylim((.05, 2))
                ax.set_xlim((20, 80))
                ax.set_xticks((30, 50, 70))
                ax.set_yticks((0.1, 1.0))
                ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())


        ## Pretty
        # Legend
        for n_hl, (hl_type, color) in enumerate(hl_colors.items()):
            f.text(
                .95, .66 - n_hl * .05, hl_type,
                color=color, ha='center', va='center', size=12)

        # Axis labels
        f.text(.52, .01, 'sound level (dB SPL)', ha='center', va='bottom')
        f.text(.02, .55, f'response strength ({MU}V rms)',
            rotation=90, ha='center', va='center')

        # Label the channel
        for n_channel, channel in enumerate(channel_l):
            axa[n_channel, 0].set_ylabel(channel)

        # Label the speaker side
        axa[0, 0].set_title('sound from left')
        axa[0, 1].set_title('sound from right')

        # Title the figure with the wave
        f.suptitle(wave_name, y=.99)


        ## Savefig
        f.savefig(os.path.join('figures',
            f'PLOT_RMS_GROWTH_FUNCTIONS_AFTER_HL__{wave_name}.svg'))
        f.savefig(os.path.join('figures',
            f'PLOT_RMS_GROWTH_FUNCTIONS_AFTER_HL__{wave_name}.png'), dpi=300)


plt.show()