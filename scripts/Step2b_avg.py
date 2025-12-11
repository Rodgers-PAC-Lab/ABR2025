## Remove outliers and aggregates ABRs over trials
# Writes out:
#   big_abrs - averaged ABRs
#   trial_counts - trial counts
#   averaged_abrs_by_date - averaged ABRs aggregated by date
#   averaged_abrs_by_mouse - averaged ABRs aggregated by mouse
#
# Plots:
#   PLOT_SINGLE_TRIAL_ABR
#   PLOT_TRIAL_AVERAGED_ABR
#   PLOT_POSITIVE_AND_NEGATIVE_CLICKS


import os
import datetime
import json
import numpy as np
import pandas
import ABR2025
import my.plot
import matplotlib.pyplot as plt
import tqdm


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
# Outlier params
abs_max_sigma = 3
stdev_sigma = 3


## Load previous results
# Load results of Step1
mouse_metadata = pandas.read_pickle(
    os.path.join(output_directory, 'mouse_metadata'))
experiment_metadata = pandas.read_pickle(
    os.path.join(output_directory, 'experiment_metadata'))
recording_metadata = pandas.read_pickle(
    os.path.join(output_directory, 'recording_metadata'))

# Load results of Step2a1_align
big_triggered_neural = pandas.read_pickle(
    os.path.join(output_directory, 'big_triggered_neural'))
big_click_params = pandas.read_pickle(
    os.path.join(output_directory, 'big_click_params'))

# Loudest dB
loudest_db = big_triggered_neural.index.get_level_values('label').max()


## Join on speaker_side
# Should have done this at the same time as joining channel
idx = big_triggered_neural.index.to_frame().reset_index(drop=True)
idx = idx.join(
    recording_metadata['speaker_side'], on=['date', 'mouse', 'recording'])
big_triggered_neural.index = pandas.MultiIndex.from_frame(idx)

# Reorder levels
big_triggered_neural = big_triggered_neural.reorder_levels([
    'date', 'mouse', 'recording', 'channel', 'speaker_side', 'label', 
    'polarity', 't_samples']).sort_index()


## Drop outlier trials, separately by channel
# Consider outliers separately for every channel on every recording
group_by = ['date', 'mouse', 'recording', 'channel', 'speaker_side']
gobj = big_triggered_neural.groupby(group_by)

# We use this helper function to drop the groupby levels, otherwise
# the levels get duplicated by gobj.apply
def drop_outliers(df):
    res = ABR2025.signal_processing.trim_outliers(
        df.droplevel(group_by),
        abs_max_sigma=abs_max_sigma,
        stdev_sigma=stdev_sigma,
        )
    
    return res

# Apply the drop
big_triggered_neural2 = gobj.apply(drop_outliers)

# Reorder levels to be like triggered_neural
big_triggered_neural2 = big_triggered_neural2.reorder_levels(
    big_triggered_neural.index.names).sort_index()


## Count the number of trials remaining in each recording
trial_counts = big_triggered_neural2.groupby(
    ['date', 'mouse', 'speaker_side', 'recording', 'label', 'channel']).size()


## Aggregate
# Average out the trial
by_polarity = big_triggered_neural2.groupby(
    [lev for lev in big_triggered_neural2.index.names if lev != 't_samples']
    ).mean()

# Compute the big_abrs by adding over polarity
big_abrs = 0.5 * (
    by_polarity.xs(True, level='polarity') + 
    by_polarity.xs(False, level='polarity')
    )

# Compute the big_arts by subtracting over polarity
big_arts = 0.5 * (
    by_polarity.xs(True, level='polarity') - 
    by_polarity.xs(False, level='polarity')
    )


## Join metadata on big_abrs
# Join after_HL and n_experiment onto big_abrs
big_abrs = my.misc.join_level_onto_index(
    big_abrs, 
    experiment_metadata.set_index(['mouse', 'date'])[['after_HL', 'n_experiment']], 
    join_on=['mouse', 'date']
    )

# Drop the now unnecessary level 'date' (replaced with n_experiment)
big_abrs = big_abrs.droplevel('date')

# Join HL_type onto big_abrs
big_abrs = my.misc.join_level_onto_index(
    big_abrs, 
    mouse_metadata.set_index('mouse')['HL_type'], 
    join_on='mouse',
    )


## Join metadata on trial_counts in the same way
# Join after_HL and n_experiment onto trial_counts
trial_counts = my.misc.join_level_onto_index(
    trial_counts, 
    experiment_metadata.set_index(['mouse', 'date'])[['after_HL', 'n_experiment']], 
    join_on=['mouse', 'date']
    )

# Drop the now unnecessary level 'date' (replaced with n_experiment)
trial_counts = trial_counts.droplevel('date')

# Join HL_type onto big_abrs
trial_counts = my.misc.join_level_onto_index(
    trial_counts, 
    mouse_metadata.set_index('mouse')['HL_type'], 
    join_on='mouse',
    )

# Reorder level to match big_abrs
trial_counts = trial_counts.reorder_levels(big_abrs.index.names).sort_index()


## Further aggregate big_abrs
# Mean out recording, leaving n_experiment
averaged_abrs_by_date = big_abrs.groupby(
    [lev for lev in big_abrs.index.names if lev != 'recording']
    ).mean()

# Keep only the first experiment from each mouse
# Better to keep the first than to average, because the two experiments
# may be different-looking or out of phase
averaged_abrs_by_mouse = averaged_abrs_by_date.xs(0, level='n_experiment')


## Dump overall N
# Estimate duration as the last click time
recording_duration = big_click_params.groupby(
    ['date', 'mouse', 'recording']).apply(lambda df: df.index[-1][-1]
    ) / 16e3
n_recordings = len(recording_duration)
n_experiments = len(recording_duration.groupby(['date', 'mouse']).sum())
n_mice = len(recording_duration.groupby('mouse').sum())
quantiles = recording_duration.quantile((0, .25, .5, .75, 1))

# Get range of surgical experiment dates
surgery_df = experiment_metadata.join(
    mouse_metadata.set_index('mouse')['HL_date'], on='mouse')
surgery_dates = (surgery_df['date'] - surgery_df['HL_date']).dropna().sort_values()
surgery_dates = surgery_dates.value_counts().sort_index()

# Print N
stats_filename = 'figures/STATS__N_OVERALL'
with open(stats_filename, 'w') as fi:
    fi.write(stats_filename + '\n')
    fi.write(f'n = {n_mice} mice; {n_recordings} recordings; {n_experiments} experiments\n')
    fi.write(f'duration quantiles:\n{str(quantiles)}\n')
    fi.write(f'surgery date range:\n{str(surgery_dates)}')

# Echo
with open(stats_filename) as fi:
    print(''.join(fi.readlines()))


## Plots
PLOT_SINGLE_TRIAL_ABR = True
PLOT_TRIAL_AVERAGED_ABR = True
PLOT_POSITIVE_AND_NEGATIVE_CLICKS = True

if PLOT_SINGLE_TRIAL_ABR:
    # Cat_229 rec 10 on 2025-05-20 is about top 25th percentile of response
    # strength. So it's stronger than median, but not unusually strong.
    # This is an RV recording with speaker_side L
    single_trial_abr = big_triggered_neural2.loc[
        datetime.date(2025, 5, 20)].loc['Cat_229'].loc[10].xs(
        'RV', level='channel').droplevel('speaker_side')

    # Slice out loudest only
    single_trial_abr = single_trial_abr.xs(loudest_db, level='label')

    f, ax = plt.subplots(figsize=(3.75, 2.5))
    f.subplots_adjust(bottom=.24, left=.18, right=.95, top=.89)
    ax.plot(
        single_trial_abr.columns.values / 16e3 * 1000,
        single_trial_abr.T * 1e6, 
        color='k', alpha=.05, lw=1)
    ax.set_xlabel('time from sound (ms)')
    ax.set_ylabel(f'ABR ({MU}V)')
    ax.set_yticks((-8, 0, 8))
    ax.set_ylim((-8, 8))
    ax.set_xlim((-2.5, 7.5))
    ax.set_xticks((-2, 0, 2, 4, 6, ))
    my.plot.despine(ax)

    # Plot the stimulus
    ax.plot([0, 0.1], [5.5 / 6 * 8] * 2, 'k-', lw=3)

    # Savefig
    f.savefig('figures/PLOT_SINGLE_TRIAL_ABR.svg')
    f.savefig('figures/PLOT_SINGLE_TRIAL_ABR.png', dpi=300)

if PLOT_TRIAL_AVERAGED_ABR:
    ## Plot the ABR averaged by level
    
    # Slice out Cat_229 rec 10 on 2025-05-20, RV for L speaker
    to_agg = big_triggered_neural2.loc[
        datetime.date(2025, 5, 20)].loc[
        'Cat_229'].loc[
        10].loc[
        'RV'].loc[
        'L']

    # Aggregate
    trial_averaged_abr = to_agg.groupby('label').mean()
    
    # Set up colorbar
    # Always do the lowest labels last
    label_l = sorted(trial_averaged_abr.index, reverse=True)
    aut_colorbar = my.plot.generate_colorbar(
        len(label_l), mapname='inferno_r', start=0.15, stop=1)[::-1]

    # Plot
    f, ax = plt.subplots(figsize=(3.75, 2.5))
    f.subplots_adjust(bottom=.24, left=.18, right=.95, top=.89)
    for label in label_l:
        xtopl = trial_averaged_abr.columns.values / 16e3 * 1000
        ytopl = trial_averaged_abr.loc[label] * 1e6
        color = aut_colorbar[label_l.index(label)]       
        ax.plot(xtopl, ytopl, color=color, lw=1)
    
    # Legend
    for n_label, (label, color) in enumerate(zip(label_l, aut_colorbar)):
        if np.mod(n_label, 2) != 0:
            continue
        f.text(
            1.02, .75 - n_label * .035, f'{label} dB',
            color=color, ha='center', va='center', size=12)

    # Pretty
    ax.set_xlabel('time from sound (ms)')
    ax.set_ylabel(f'ABR ({MU}V)')
    ax.set_yticks((-4, 0, 4))
    ax.set_ylim((-4, 4))
    ax.set_xlim((-2.5, 7.5))
    ax.set_xticks((-2, 0, 2, 4, 6, ))
    my.plot.despine(ax)
    
    # Plot the stimulus
    ax.plot([0, 0.1], [3.5, 3.5], 'k-', lw=3)
    
    # Plot the baseline period
    #~ ax.fill_between([-2.5, -1.25], y1=-6, y2=6, color='orange', alpha=0.5)
    
    f.savefig('figures/PLOT_TRIAL_AVERAGED_ABR.svg')
    f.savefig('figures/PLOT_TRIAL_AVERAGED_ABR.png', dpi=300)

if PLOT_POSITIVE_AND_NEGATIVE_CLICKS:
    ## Plot response to positive and negative clicks for a single recording
    # Only Pineapple_197 recs 6 and 7 shows speaker artefact, and even 
    # that one is not so bad
    
    # Slice out Cat_229 rec 10 on 2025-05-20, RV for L speaker
    to_agg = big_triggered_neural2.loc[
        datetime.date(2025, 5, 20)].loc[
        'Cat_229'].loc[
        10].loc[
        'RV'].loc[
        'L']

    # Slice out loudest only
    to_agg = to_agg.xs(loudest_db, level='label')
    
    # Aggregate
    trial_averaged_abr = to_agg.groupby('polarity').mean()
    
    # Plot
    f, ax = plt.subplots(figsize=(3.75, 2.5))
    f.subplots_adjust(bottom=.24, left=.18, right=.95, top=.89)
    ax.plot(
        trial_averaged_abr.columns.values / 16e3 * 1000,
        trial_averaged_abr.loc[True] * 1e6, 
        color='k', lw=1, label='positive')
    ax.plot(
        trial_averaged_abr.columns.values / 16e3 * 1000,
        trial_averaged_abr.loc[False] * 1e6, 
        color='k', lw=1, ls='--', label='negative')
    ax.set_xlabel('time from sound (ms)')
    ax.set_ylabel(f'ABR ({MU}V)')
    ax.set_yticks((-4, 0, 4))
    ax.set_ylim((-4, 4))
    ax.set_xlim((-2.5, 7.5))
    ax.set_xticks((-2, 0, 2, 4, 6, ))
    my.plot.despine(ax)
    
    # Plot the stimulus
    ax.plot([0, 0.1], [3.5, 3.5], 'k-', lw=3)
    
    # Legend
    ax.legend(frameon=False, fontsize=12, loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    f.savefig('figures/PLOT_POSITIVE_AND_NEGATIVE_CLICKS.svg')
    f.savefig('figures/PLOT_POSITIVE_AND_NEGATIVE_CLICKS.png', dpi=300)


## Show
plt.show()


## Store
big_abrs.to_pickle(os.path.join(output_directory, 'big_abrs'))
averaged_abrs_by_date.to_pickle(os.path.join(output_directory, 'averaged_abrs_by_date'))
averaged_abrs_by_mouse.to_pickle(os.path.join(output_directory, 'averaged_abrs_by_mouse'))
trial_counts.to_pickle(os.path.join(output_directory, 'trial_counts'))