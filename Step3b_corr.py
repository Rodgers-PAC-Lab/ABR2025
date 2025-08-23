## Correlation across mice
# Plots
#   HIST_CORRELATION_ACROSS_MICE
#   HEATMAP_CORRELATION_ACROSS_MICE

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


## Cross-correlate within and between mice to measure consistency
# Get date * mouse on the rows
unstacked = avged_abrs.unstack(
    ['channel', 'speaker_side', 'label']).reorder_levels(
    ['mouse', 'date']).sort_index()

# Include only mice with multiple sessions
sessions_per_mouse = unstacked.groupby('mouse').size()
assert sessions_per_mouse.max() == 2
mice_with_multiple_sessions = sorted(
    sessions_per_mouse.index[sessions_per_mouse == 2])
unstacked = unstacked.reindex(mice_with_multiple_sessions, level='mouse')

# Correlate each row
corr_across_sessions = unstacked.T.corr()

# Null the self-comparisons and redundant comparisons
null_mask = np.tri(len(unstacked)).astype(bool)
corr_across_sessions.values[null_mask] = np.nan

# Relabel the columns, fully stack, and reset index
corr_across_sessions.columns.names = ['mouse2', 'date2']
corr_across_sessions = corr_across_sessions.stack(
    future_stack=True).stack(future_stack=True).rename('corr')
corr_across_sessions = corr_across_sessions.reset_index()

# Drop nulls, including self comparisons
corr_across_sessions = corr_across_sessions.dropna()

# Label 'within' and 'between'
within_mask = corr_across_sessions['mouse'] == corr_across_sessions['mouse2']
corr_across_sessions['typ'] = 'between'
corr_across_sessions.loc[within_mask, 'typ'] = 'within'


## Now also include recordings within a session
grouping_keys = ['date', 'mouse', 'channel', 'speaker_side']
mean_corr_l = []
keys_l = []
for grouped_key, subdf in big_abrs.groupby(grouping_keys):
    # Drop grouping keys
    subdf = subdf.droplevel(grouping_keys)
    
    # Unstack label
    subdf = subdf.unstack('label')
    
    # Continue if not enough to compare
    if len(subdf) == 1:
        continue
    
    # Corr
    this_corr = subdf.T.corr()
    this_corr[np.tri(len(this_corr)).astype(bool)] = np.nan
    mean_corr = this_corr.stack().mean()
    
    # Store
    mean_corr_l.append(mean_corr)
    keys_l.append(grouped_key)

# Concat
within_experiment_corr = pandas.Series(
    mean_corr_l, 
    index=pandas.MultiIndex.from_tuples(keys_l, names=grouping_keys))

# Keep only mice with multiple sessions
within_experiment_corr = within_experiment_corr.reindex(
    mice_with_multiple_sessions, level='mouse')

# Mean over channel and speaker side
# Note that LR is slightly less corr than LV and RV
within_session_corr = within_experiment_corr.groupby(['date', 'mouse']).mean()

# Mean over session
within_mouse_corr = within_session_corr.groupby('mouse').mean()


## Plots
HIST_CORRELATION_ACROSS_MICE = True
HEATMAP_CORRELATION_ACROSS_MICE = True

if HIST_CORRELATION_ACROSS_MICE:
    ## Histogram Perason's R for across mice, within mice, and within sessions
    # If you want to plot R**2, you have to square before aggregating above
    
    # Plot the distribution
    bins = np.linspace(0, 1, 21)
    f, ax = my.plot.figure_1x1_standard()
    ax.hist(corr_across_sessions.loc[~within_mask, 'corr'], bins=bins, color='green', histtype='step')
    ax.hist(corr_across_sessions.loc[within_mask, 'corr'], bins=bins, color='k', alpha=.5)
    #~ ax.hist(within_mouse_corr, bins=bins, color='r', alpha=.5)
    ax.set_xlim((0, 1))
    ax.set_xticks((0, .5, 1))
    ax.set_yticks((0, 3, 6))
    ax.set_xlabel('correlation')
    ax.set_ylabel('# of comparisons')
    my.plot.despine(ax)
    
    # Rather than a third histogram, plot the within-recording corr as a line
    # This one is calculated slightly differently because within-recording
    # must always be the same speaker_side (and possibly subset of configs)
    ax.plot([within_mouse_corr.mean()] * 2, ax.get_ylim(), 'k--', lw=1)
    
    f.text(.5, .9, 'across mice', color='g', ha='center')
    f.text(.5, .82, 'within mouse', color='gray', ha='center')

    # Savefig
    f.savefig('HIST_CORRELATION_ACROSS_MICE.svg')
    f.savefig('HIST_CORRELATION_ACROSS_MICE.png', dpi=300)

    with open('STATS__HIST_CORRELATION_ACROSS_MICE', 'w') as fi:
        n_mice = len(within_mouse_corr)
        fi.write(f'n = {n_mice} with multiple sessions\n')
        fi.write('across mice mean R: {:.4f}\n'.format(
            corr_across_sessions.groupby('typ')['corr'].mean().loc['between']))
        fi.write('within mouse, across sessions mean R: {:.4f}\n'.format(
            corr_across_sessions.groupby('typ')['corr'].mean().loc['within']))
        fi.write('within sessions mean R: {:.4f}\n'.format(
            within_mouse_corr.mean()))

    with open('STATS__HIST_CORRELATION_ACROSS_MICE') as fi:
        for line in fi.readlines():
            print(line.strip())

if HEATMAP_CORRELATION_ACROSS_MICE:
    # Group comparisons
    # There is only one way to compare a mouse to itself, but there are
    # 4 ways to compare one mouse to another
    topl = corr_across_sessions.groupby(['mouse', 'mouse2'])['corr'].mean().unstack() ** 2

    # Plot
    f, ax = my.plot.figure_1x1_standard()
    my.plot.imshow(topl, ax=ax, axis_call='scaled', cmap='viridis', clim=(0, 1))
    ax.set_xticks(range(topl.shape[1]))
    #~ ax.set_xticklabels(topl.columns, rotation=90)
    ax.set_yticks(range(topl.shape[0]))
    #~ ax.set_yticklabels(topl.index)
    ax.set_xlabel('mouse 1')
    ax.set_ylabel('mouse 2')
    
    cb = my.plot.colorbar(fig=f)

    # Savefig
    f.savefig('HEATMAP_CORRELATION_ACROSS_MICE.svg')
    f.savefig('HEATMAP_CORRELATION_ACROSS_MICE.png', dpi=300)



plt.show()