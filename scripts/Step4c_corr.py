## Correlation across mice
# Plots
#   HIST_CORRELATION_ACROSS_MICE
#   HEATMAP_CORRELATION_ACROSS_MICE

import os
import json
import numpy as np
import pandas
import my.plot
import matplotlib.pyplot as plt
import shared


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
sampling_rate = 16000


## Load previous results
# Load results of Step2b_avg
averaged_abrs_by_date = pandas.read_parquet(
    os.path.join(output_directory, 'averaged_abrs_by_date'))

# Keep only after_HL == False
averaged_abrs_by_date = averaged_abrs_by_date.xs(
    False, level='after_HL').droplevel('HL_type')


## Cross-correlate within and between mice
# Get date * mouse on the rows
unstacked = averaged_abrs_by_date.unstack(
    ['channel', 'speaker_side', 'label']).reorder_levels(
    ['mouse', 'n_experiment']).sort_index()

# Slice to the (0, 7.5) ms window used for most of the manuscript
# (Data before t=0 can only add to noise so it is excluded)
unstacked = unstacked.loc[:, 0:119]

# Include only mice with multiple sessions
sessions_per_mouse = unstacked.groupby('mouse').size()
assert sessions_per_mouse.max() == 2
mice_with_multiple_sessions = sorted(
    sessions_per_mouse.index[sessions_per_mouse == 2])
unstacked = unstacked.reindex(mice_with_multiple_sessions, level='mouse')

# Correlate each row
corr_across_sessions = unstacked.T.corr()

# Relabel the columns, fully stack, and reset index
corr_across_sessions.columns.names = ['mouse2', 'n_experiment2']
corr_across_sessions = corr_across_sessions.stack(
    future_stack=True).stack(future_stack=True).rename('corr')
corr_across_sessions = corr_across_sessions.reset_index()

# Label 'within' and 'between'
within_mask = corr_across_sessions['mouse'] == corr_across_sessions['mouse2']
corr_across_sessions['typ'] = 'between'
corr_across_sessions.loc[within_mask, 'typ'] = 'within'

# Label the 'order'
# 'forward': experiment 0 for mouse and experiment 1 for mouse2
# 'reverse': experiment 1 for mouse and experiment 0 for mouse2
# 'same0': experiment 0 for mouse and experiment 0 for mouse2
# 'same1': experiment 1 for mouse and experiment 1 for mouse2
# There will be N 'within-mouse' comparisons of each 'order', but the 'same0'
# and 'same1' comparisons will be 1 by definition.
# There will be N*(N-1) 'between-mouse' comparisons of each order.
# In all cases, 'forward' is redundant with 'reverse'
corr_across_sessions['order'] = 'blank'
corr_across_sessions.loc[
    corr_across_sessions['n_experiment2'] > corr_across_sessions['n_experiment'],
    'order'] = 'forward'
corr_across_sessions.loc[
    corr_across_sessions['n_experiment2'] < corr_across_sessions['n_experiment'],
    'order'] = 'reverse'
corr_across_sessions.loc[
    (corr_across_sessions['n_experiment2'] == corr_across_sessions['n_experiment']) &
    (corr_across_sessions['n_experiment2'] == 0),
    'order'] = 'same0'
corr_across_sessions.loc[
    (corr_across_sessions['n_experiment2'] == corr_across_sessions['n_experiment']) &
    (corr_across_sessions['n_experiment2'] == 1),
    'order'] = 'same1'

# Drop the trivial case of 'same0' and 'same1' for 'within'
corr_across_sessions = corr_across_sessions.loc[~(
    corr_across_sessions['order'].isin(['same0', 'same1']) & 
    (corr_across_sessions['typ'] == 'within')
    )]

# Drop the 'reverse' case everywhere, since it is always redundant with 'forward'
corr_across_sessions = corr_across_sessions.loc[~(
    (corr_across_sessions['order'] == 'reverse')
    )]

# For unknown reasons, the between-mouse corr is higher on day0 than on day1,
# with the 'forward' comparison being intermediate.
# The within-mouse corr is similar to the highest of those corrs (day0)
# The fairest comparison is to use only 'forward' in all cases, because
# 'same0' and 'same1' are trivial for within-mouse
corr_across_sessions = corr_across_sessions[
    corr_across_sessions['order'] == 'forward']


## Plots
HIST_CORRELATION_ACROSS_MICE = True
HEATMAP_CORRELATION_ACROSS_MICE = True

if HIST_CORRELATION_ACROSS_MICE:
    ## Histogram Pearson's R for across mice, within mice, and within sessions
    
    # Consistent bins
    bins = np.linspace(0, 1, 26)
    
    # Figure handles
    f, ax = my.plot.figure_1x1_standard()
    
    # Histogram across mice
    ax.hist(
        corr_across_sessions.loc[corr_across_sessions['typ'] == 'between', 'corr'],
        bins=bins, color='green', histtype='step')
    
    # Histogram within mouse
    ax.hist(
        corr_across_sessions.loc[corr_across_sessions['typ'] == 'within', 'corr'],
        bins=bins, color='k', alpha=.5)
    
    # Pretty
    ax.set_xlim((0, 1))
    ax.set_xticks((0, .5, 1))
    ax.set_ylim((0, 8))
    ax.set_yticks((0, 4, 8))
    ax.set_xlabel('correlation')
    ax.set_ylabel('# of comparisons')
    my.plot.despine(ax)

    # Legend
    f.text(.5, .9, 'across mice', color='g', ha='center')
    f.text(.5, .82, 'within mouse', color='gray', ha='center')

    # Savefig
    f.savefig('figures/HIST_CORRELATION_ACROSS_MICE.svg')
    f.savefig('figures/HIST_CORRELATION_ACROSS_MICE.png', dpi=300)

    
    ## Stats
    # Mean and std of the correlation, within vs between mice
    corr_by_typ = corr_across_sessions.groupby('typ')['corr']
    corr_mean = corr_by_typ.mean()
    corr_std = corr_by_typ.std()

    with open('figures/STATS__HIST_CORRELATION_ACROSS_MICE', 'w') as fi:
        n_mice = len(corr_across_sessions.loc[within_mask])
        fi.write(f'n = {n_mice} with multiple sessions\n')
        fi.write(f'forward comparisons only\n')
        fi.write('across mice mean R: {:.4f} (std = {:.4f})\n'.format(
            corr_mean['between'],
            corr_std['between'],
            ))
        fi.write('within mouse, across sessions mean R: {:.4f} (std = {:.4f})\n'.format(
            corr_mean['within'],
            corr_std['within'],
            ))

    with open('figures/STATS__HIST_CORRELATION_ACROSS_MICE') as fi:
        for line in fi.readlines():
            print(line.strip())

if HEATMAP_CORRELATION_ACROSS_MICE:
    # Because only forward comparisons are included, there is only one entry
    # for each pair of mice
    topl = corr_across_sessions.set_index(
        ['mouse', 'mouse2'])['corr'].unstack('mouse2')

    # Plot
    # Cacti_223 on day 1 and Lighthouse_232 on day 1 are notably lower
    # I suspect those are just "weird days"
    f, ax = my.plot.figure_1x1_standard()
    my.plot.imshow(topl, ax=ax, axis_call='scaled', cmap='viridis', clim=(0, 1))
    ax.set_xticks(range(topl.shape[1]))
    ax.set_xticklabels(np.arange(topl.shape[1], dtype=int) + 1)
    ax.set_yticks(range(topl.shape[0]))
    ax.set_yticklabels(np.arange(topl.shape[0], dtype=int) + 1)
    ax.set_xlabel('mouse on day 2')
    ax.set_ylabel('mouse on day 1')

    # Add a colorbar
    cb = my.plot.colorbar(fig=f)
    cb.set_label('correlation')
    cb.set_ticks((0, 0.5, 1))
    
    # Pretty
    f.subplots_adjust(left=.1, right=.85)

    # Savefig
    f.savefig('figures/HEATMAP_CORRELATION_ACROSS_MICE.svg')
    f.savefig('figures/HEATMAP_CORRELATION_ACROSS_MICE.png', dpi=300)



plt.show()