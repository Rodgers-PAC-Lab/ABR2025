## Run ABRpresto on our data
"""
You can skip this script if you don't have ABRpresto
This script must be run in a conda environment that satisfies ABRpresto's 
requirements, and ABRpresto must be installed, and you must be using
our fork with a few parameter changes
See README for further instructions

Expected warnings:
---
warning: 2025-05-09__Cat_227__1__RL has min 18 reps; skipping                                                                     
warning: 2025-05-09__Cat_227__1__VL has min 18 reps; skipping                                                                     
warning: 2025-05-09__Cat_227__1__VR has min 18 reps; skipping                                                                     
warning: 2025-06-06__Cacti_223__11__RL has min 7 reps; skipping                                                                   
warning: 2025-06-06__Cacti_223__11__VL has min 7 reps; skipping                                                                   
warning: 2025-06-06__Cacti_223__11__VR has min 7 reps; skipping                                                                   
warning: 2025-06-06__Cacti_223__12__RL has min 13 reps; skipping                                                                  
warning: 2025-06-06__Cacti_223__12__VL has min 13 reps; skipping                                                                  
warning: 2025-06-06__Cacti_223__12__VR has min 13 reps; skipping                                                                  
warning: 2025-06-06__Cacti_223__15__RL has min 5 reps; skipping                                                                   
warning: 2025-06-06__Cacti_223__15__VL has min 5 reps; skipping                                                                   
warning: 2025-06-06__Cacti_223__15__VR has min 5 reps; skipping                                                                   
warning: 2025-06-06__Cacti_224__7__RL has min 18 reps; skipping                                                                   
warning: 2025-06-06__Cacti_224__7__VL has min 18 reps; skipping                                                                   
warning: 2025-06-06__Cacti_224__7__VR has min 18 reps; skipping  
"""
import os
import json
import numpy as np
import pandas
import ABRpresto
import matplotlib.pyplot as plt
import tqdm
import logging


## Disable ABRpresto warnings
class _abr_filter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        drop = (
            'Minimum needed (arbitrary)' in msg or 
            'Different level combinations' in msg
            )
        return not drop

logging.getLogger('ABRpresto.XCsub').addFilter(_abr_filter())


## Paths
# Load the required file filepaths.json (see README)
with open('filepaths.json') as fi:
    paths = json.load(fi)

# Parse into paths to raw data and output directory
raw_data_directory = paths['raw_data_directory']
output_directory = paths['output_directory']

# Put figs here
fig_dir = os.path.join(output_directory, 'abr_presto')
if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)


## Params
XCsubargs = {
    # RandomState seed; fixed for reproducible shuffles
    'seed': 0,               
    
    # post-stim window (s) for cross-correlation: 0.5-6 ms
    'pst_range': [0.0005, 0.006],  
    
    # random trial splits per level (can lower for speed, e.g. 100)
    'N_shuffles': 500,       
    
    # subaverage by median (vs 'mean')
    'avmode': 'median',      
    
    # criterion: threshold = level where the fit crosses this XC value
    # CR raised XC0m_threshold to 0.7 to force a fit on most sessions
    'XC0m_threshold': 0.7,   
    
    # sigmoid: force positive slope, midpoint within [min-step, max+step]
    'XC0m_sigbounds': 'increasing, midpoint within one step of x range', 
    
    # power-law: force positive slope
    'XC0m_plbounds': 'increasing',  
    
    # disable the filtering since we'll provide already-filtered data
    'second_filter': None,   
    
    # only the XC0-mean method (the paper's main one); skip KS/XCp alternates
    'calc_XC0m_only': True,  
    
    # whether to make plots
    'plot_results': True,    
    
    # no rounding
    'round_results': False,
    }
sampling_rate = 16000
min_trials_per_polarity = 20


## Load data
# Raw data
big_triggered_neural = pandas.read_parquet(
    os.path.join(output_directory, 'big_triggered_neural'))
big_triggered_neural_orig = big_triggered_neural.copy()


## Format: columns samples -> seconds; polarity False/True -> -1/1
# Label columns by time in seconds
big_triggered_neural.columns = (big_triggered_neural.columns / sampling_rate).values
big_triggered_neural.columns.name = 'time'

# Label sound_level as "level"
big_triggered_neural = big_triggered_neural.rename_axis(index={'label': 'level'})

# Label polarity as +1 or -1
big_triggered_neural = big_triggered_neural.rename(
    index={True: 1, False: -1}, level='polarity')

# Pull levels for later recomputation
levels = np.sort(
    big_triggered_neural.index.get_level_values('level').unique().values)


## Fit each recording
keys_l = []
scalar_l = []
sigmoid_params_l = []
xc0mean_l = []
xc0std_l = []
for keys, this_group in tqdm.tqdm(big_triggered_neural.groupby(
        ['date', 'mouse', 'recording', 'channel'])):

    # Form session_name
    session_name = '__'.join(map(str, keys))

    # Count min trials per polarity
    this_n = this_group.groupby(['polarity', 'level']).size()
    min_n = this_n.min()

    # Fit; estimate_threshold drops all index levels but polarity/level
    this_fit, this_fig = ABRpresto.XCsub.estimate_threshold(this_group, **XCsubargs)

    # Check whether we succeeded
    if this_fit['status'] != 'Success':
        # Some kind of failure, assert that this only happens when n is too low
        assert min_n < min_trials_per_polarity
        
        # A blank figure is created in this case
        # This doesn't reliably work, I think - sometimes get blank figure anyway
        plt.close(this_fig)
        
        # Warn and continue
        tqdm.tqdm.write(f'warning: {session_name} has min {min_n} reps; skipping')
        continue
    else:
        # Succeeded
        assert min_n >= min_trials_per_polarity

    # Savefig
    if this_fig is not None:
        this_fig.savefig(os.path.join(fig_dir, session_name + '.svg'))
        this_fig.savefig(os.path.join(fig_dir, session_name + '.png'), dpi=300)
        plt.close(this_fig)

    # This is sometimes None
    if this_fit['fit_XC0m']['sigmoid_fit'] is None:
        sigmoid_sse = None
        sigmoid_params = [np.nan, np.nan, np.nan, np.nan]
    else:
        sigmoid_sse = this_fit['fit_XC0m']['sigmoid_fit']['sse']
        sigmoid_params = this_fit['fit_XC0m']['sigmoid_fit']['params']

    # Store scalars
    scalar_l.append({
        'threshold':      this_fit['threshold'],
        'status':         this_fit['status'],
        'status_message': this_fit['status_message'],
        'bestFitType':    this_fit['fit_XC0m']['bestFitType'],
        'sigmoid_sse':    sigmoid_sse,
        'N_min_global':   this_fit['datapars']['N_min_global'],
        })

    # Store arrays
    sigmoid_params_l.append(sigmoid_params)
    xc0mean_l.append(this_fit['datapars']['xc0mean'])
    xc0std_l.append(this_fit['datapars']['xc0std'])

    # Assert recomputation works
    if sigmoid_sse is not None:
        yfit = ABRpresto.utils.sigmoid(levels, *sigmoid_params)
        assert np.allclose(yfit, this_fit['fit_XC0m']['sigmoid_fit']['yfit'])

    # Store keys
    keys_l.append(keys)


## Close any leftover figures - something about the skipped ones
plt.close('all')


## Assemble DataFrames
# Keys
keys_midx = pandas.MultiIndex.from_tuples(
    keys_l, names=['date', 'mouse', 'recording', 'channel'])

# DataFrames
threshold_df = pandas.DataFrame(scalar_l, index=keys_midx)
sigmoid_params_df = pandas.DataFrame(
    sigmoid_params_l, index=keys_midx,
    columns=['amplitude', 'slope', 'x0', 'baseline'])
xc0mean_df = pandas.DataFrame(
    xc0mean_l, index=keys_midx,
    columns=levels)
xc0std_df = pandas.DataFrame(
    xc0std_l, index=keys_midx,
    columns=levels)

# Save
threshold_df.to_parquet(os.path.join(output_directory, 'abr_presto_threshold_df'))
sigmoid_params_df.to_parquet(os.path.join(output_directory, 'abr_presto_sigmoid_params_df'))
xc0mean_df.to_parquet(os.path.join(output_directory, 'abr_presto_xc0mean_df'))
xc0std_df.to_parquet(os.path.join(output_directory, 'abr_presto_xc0std_df'))
