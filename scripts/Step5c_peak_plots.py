## Plots relating to peak picking, excluding those that require ABRA results

import os
import json
import scipy.stats
import numpy as np
import pandas
import my.stats
import my.plot
import matplotlib.pyplot as plt
import seaborn


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
sampling_rate = 16000

# Which waves to include in the plots
include_waves = [
    'W1p', 'W1n', 'W2p', 'W2n', 'W3p', 'W3n', 'W4p', 'W4n', 'W5p', 'W5n']

# Wave colors for plotting
cmap = plt.get_cmap('tab20')
wave_colors = {}
for i in range(8):
    wave_colors[f'W{i}p'] = cmap(2 * i)      # dark
    wave_colors[f'W{i}n'] = cmap(2 * i + 1)  # light


## Load previous results
# Load results of Step2b_avg
averaged_abrs_by_date = pandas.read_parquet(
    os.path.join(output_directory, 'averaged_abrs_by_date'))

# Loudest dB
loudest_db = averaged_abrs_by_date.index.get_level_values('label').max()

# Include only vertex-ear channels
averaged_abrs_by_date = averaged_abrs_by_date.reindex(
    ['VL', 'VR'], level='channel')
averaged_abrs_by_date = averaged_abrs_by_date.sort_index()

# All levels (used to ensure every figure has the same rows)
all_levels = sorted(
    averaged_abrs_by_date.index.get_level_values('label').unique())

# Convert to uV
averaged_abrs_by_date = averaged_abrs_by_date * 1e6

# Results of peak-picking
big_ridges = pandas.read_parquet(
    os.path.join(output_directory, 'big_ridges'))


## Plot
STRIP_PLOT_LATENCIES = False
PLOT_EXAMPLE_WATERFALL = False
PLOT_EXAMPLE_HEATMAP = False
PLOT_PEAKS_AT_LOUDEST_ACROSS_MICE = False
PLOT_PEAK_METRIC_BY_WAVE_AND_CONFIG = True
PLOT_PEAK_GROWTH_FUNCTIONS = True
PLOT_PEAK_GROWTH_FUNCTIONS_AFTER_HL = True


if STRIP_PLOT_LATENCIES:
    """Strip plot the latencies to each peak
    
    Makes two figures:
    - first_rec_VRL_only - only shows first recording per mouse, only shows
        VR-L, thus there is only one point per mouse * wave * level
    - preHL - pools over recordings, experiments, channels, and speaker side
        Showing all the points like this is useful to look for outliers across
        all configs
    """

    # Conditions: (suffix, sliced big_ridges)
    condition_l = [
        ('first_rec_VRL_only',
            big_ridges.xs(False, level='after_HL').xs(
                0, level='n_experiment').xs(
                'VR', level='channel').xs(
                'L', level='speaker_side')),
        ('preHL', 
            big_ridges.xs(False, level='after_HL')),
        ]
    
    # One figure per condition
    for suffix, this_ridges in condition_l:
        
        # Slice out waves to plot
        topl = this_ridges[this_ridges['wave_name'].isin(include_waves)]

        # Assert max one point per wave per mouse per level
        if suffix == 'first_rec_VRL_only':
            assert topl.groupby(
                ['mouse', 'level', 'wave_name']).size().max() <= 1        
        
        # A single ax with each swarm at its own ypos
        f, ax = plt.subplots(figsize=(3.1, 4.7))
        f.subplots_adjust(left=.2, bottom=.13, top=.95, right=.95)

        # Strip plot the latency
        alpha = 0.5 if suffix == 'preHL' else 1
        seaborn.stripplot(
            data=topl, x='latency_ms', y='level', hue='wave_name', 
            hue_order=include_waves, order=all_levels[::-1], orient='h', 
            palette=wave_colors, size=2, alpha=alpha, jitter=0.3, 
            ax=ax,
            )

        # Pretty
        ax.get_legend().set_visible(False)
        my.plot.despine(ax)
        ax.set_yticks((0, len(all_levels) - 1))
        ax.set_yticklabels((all_levels[-1], all_levels[0]))
        ax.set_xlabel('time from click (ms)')
        ax.set_ylabel('sound level (dB SPL)')
        ax.set_xlim((1, 7))
        ax.set_xticks((1, 2, 3, 4, 5, 6, 7))


        ## Savefig
        f.savefig(f'figures/STRIP_PLOT_LATENCIES_{suffix}.png', dpi=300)
        f.savefig(f'figures/STRIP_PLOT_LATENCIES_{suffix}.svg')

if PLOT_EXAMPLE_WATERFALL:
    """Example ABR: waterfall plot with labeled peaks"""
    
    # Choose example
    HL_type = 'sham'
    after_HL = False
    n_experiment = 0
    mouse = 'Cat_227'
    channel = 'VR'
    speaker_side = 'L'

    # Get ABR for this config
    example_abr = averaged_abrs_by_date.loc[
        (HL_type, after_HL, n_experiment, mouse, channel, speaker_side)]

    # Get this config's labeled ridges
    example_ridges = big_ridges.loc[
        (HL_type, after_HL, n_experiment, mouse, channel, speaker_side)]

    # Time range to plot
    start_timepoint = int(np.rint(-.001 * sampling_rate))
    stop_timepoint = int(np.rint(.007 * sampling_rate))

    # Figure
    f, ax = my.plot.figure_1x1_standard()

    # Plot each level, offset by its rank
    level_offset_y = 2.0
    for n_level, level in enumerate(all_levels):
        
        # Slice
        topl = example_abr.loc[
            level, start_timepoint:stop_timepoint-1].copy()
        
        # Offset
        topl += n_level * level_offset_y
        
        # Plot
        ax.plot(
            topl.index / sampling_rate * 1000, 
            topl.values,
            color='k', lw=.5, clip_on=False,
            )

    # Iterate over each ridge
    grouped = example_ridges.groupby(['sign', 'n_ridge'])
    for (sign, n_ridge), this_ridge in grouped:

        # Droplevel
        this_ridge = this_ridge.droplevel(['sign', 'n_ridge'])

        # Color by wave name
        wave_name = this_ridge['wave_name'].unique().item()
        if wave_name not in include_waves:
            continue
        color = wave_colors[wave_name]

        # Pull out x and y of each peak
        for sound_level in this_ridge.index:
            
            # Time in ms of this peak
            timepoint = this_ridge.loc[sound_level, 'timepoint']
            
            # Continue if timepoint outside plot range
            if timepoint < start_timepoint or timepoint >= stop_timepoint:
                continue
            
            # Convert to ms
            t = timepoint / sampling_rate * 1000
            
            # Voltage of this peak
            y = example_abr.loc[sound_level, timepoint]
            
            # Add level offset
            y += all_levels.index(sound_level) * level_offset_y
            
            # Plot
            ax.plot(
                [t], [y], marker='o', color=color, ms=3, mew=0, clip_on=False)

    
    ## Pretty
    my.plot.despine(ax)
    ax.set_yticks((0, level_offset_y * (len(all_levels) - 1)))
    ax.set_yticklabels((all_levels[0], all_levels[-1]))
    ax.set_xlabel('time from click (ms)')
    ax.set_ylabel('sound level (dB SPL)')
    ax.set_xlim((-1, 7))
    ax.set_xticks((0, 2, 4, 6))
    ax.set_ylim(-level_offset_y, len(all_levels) * level_offset_y)
    
    
    ## Savefig
    f.savefig('figures/PLOT_EXAMPLE_WATERFALL.png', dpi=300)
    f.savefig('figures/PLOT_EXAMPLE_WATERFALL.svg')


if PLOT_EXAMPLE_HEATMAP:
    """Like PLOT_EXAMPLE_WATERFALL but as a heatmap"""
    
    # Choose example
    HL_type = 'sham'
    after_HL = False
    n_experiment = 0
    mouse = 'Cat_227'
    channel = 'VR'
    speaker_side = 'L'

    # Get ABR for this config
    example_abr = averaged_abrs_by_date.loc[
        (HL_type, after_HL, n_experiment, mouse, channel, speaker_side)]

    # Get this config's labeled ridges
    example_ridges = big_ridges.loc[
        (HL_type, after_HL, n_experiment, mouse, channel, speaker_side)]

    # Time range to plot
    start_timepoint = int(np.rint(-.001 * sampling_rate))
    stop_timepoint = int(np.rint(.007 * sampling_rate))

    # Figure
    f, ax = my.plot.figure_1x1_standard()
    
    # Heatmap
    im = my.plot.imshow(
        example_abr, ax=ax, cmap='gray',
        x=example_abr.columns.values / sampling_rate * 1000,
        y=example_abr.index.values, origin='lower')
    im.set_clim((-2, 2))

    # Iterate over each ridge
    grouped = example_ridges.groupby(['sign', 'n_ridge'])
    for (sign, n_ridge), this_ridge in grouped:

        # Color by wave name
        wave_name = this_ridge['wave_name'].unique().item()
        if wave_name not in include_waves:
            continue
        color = wave_colors[wave_name]
        
        # Plot
        ax.plot(
            this_ridge['timepoint'] / sampling_rate * 1000, 
            this_ridge.index.get_level_values('level').values, 
            color=color, 
            ls='-', ms=2, mew=0)

    # Pretty
    ax.set_ylim((all_levels[0] - 2, all_levels[-1] + 2))
    ax.set_yticks((all_levels[0], all_levels[-1]))
    ax.set_xlabel('time from click (ms)')
    ax.set_ylabel('sound level (dB SPL)')
    ax.set_xlim((-1, 7))
    ax.set_xticks((0, 2, 4, 6))
    
    # Colorbar spanning both columns
    f_cb, ax_cb = my.plot.figure_1x1_standard()
    cb = f_cb.colorbar(im, ax=ax_cb, fraction=.05, pad=.02)
    cb.set_label('ABR (' + MU + 'V)')
    cb.set_ticks([-2, 0, 2])

    # Savefig
    f.savefig('figures/PLOT_EXAMPLE_HEATMAP.png', dpi=300)
    f.savefig('figures/PLOT_EXAMPLE_HEATMAP.svg')
    f_cb.savefig('figures/PLOT_EXAMPLE_HEATMAP_colorbar.png', dpi=300)
    f_cb.savefig('figures/PLOT_EXAMPLE_HEATMAP_colorbar.svg')
    

if PLOT_PEAKS_AT_LOUDEST_ACROSS_MICE:
    """Plot ABR at loudest level across mice with peaks indicated
    
    Four conditions:
    - first_rec_VRL_only : for an example with one line per mouse
    - preHL : all vertex-ear pre-HL (possibly multiple recording per mouse)
    - postHL_bilateral, postHL_sham : analogous to preHL
    """
    
    # Conditions: (suffix, dict of level->value to slice)
    condition_l = [
        ('first_rec_VRL_only', dict(
            after_HL=False, n_experiment=0, channel='VR', speaker_side='L')),
        ('preHL', dict(after_HL=False)),
        #~ ('postHL_bilateral', dict(after_HL=True, HL_type='bilateral')),
        #~ ('postHL_sham', dict(after_HL=True, HL_type='sham')),
        ]
    
    # One figure per condition
    for suffix, sel in condition_l:
        
        # Slice abrs and ridges to this condition
        this_abrs = averaged_abrs_by_date
        this_ridges = big_ridges
        for k, v in sel.items():
            this_abrs = this_abrs.xs(v, level=k)
            this_ridges = this_ridges.xs(v, level=k)
        
        # Levels that identify a recording (everything except label)
        rec_levels = [n for n in this_abrs.index.names if n != 'label']
        
        # Make figure handles
        f, ax = my.plot.figure_1x1_standard()

        # Iterate over recordings (lines)
        for rec_keys, rec_abr in this_abrs.groupby(rec_levels):
            
            # Drop recording levels, leaving label on the index
            rec_abr = rec_abr.droplevel(rec_levels)
            
            # Slice ABR at the loudest level
            try:
                this_abr = rec_abr.loc[loudest_db]
            except KeyError:
                continue
            
            # Plot the ABR trace for this recording
            ax.plot(
                this_abr.index.values / sampling_rate * 1000, 
                this_abr.values, 
                color='k', lw=.5, alpha=.5)

            # Slice peaks from this recording at the loudest level
            try:
                this_peaks = this_ridges.loc[rec_keys].xs(loudest_db, level='level')
            except KeyError:
                continue
        
            # Iterate over each peak
            for (sign, n_ridge) in this_peaks.index:

                # Color by wave name
                wave_name = this_peaks.loc[(sign, n_ridge), 'wave_name']
                if wave_name not in include_waves:
                    continue
                color = wave_colors[wave_name]

                # Time in ms of this peak
                timepoint = this_peaks.loc[(sign, n_ridge), 'timepoint']
                
                # Convert to ms
                t = timepoint / sampling_rate * 1000
                
                # Voltage of this peak
                y = this_abr.loc[timepoint]
                
                # Plot
                ax.plot(
                    [t], [y], 
                    marker='o', color=color, ms=4, alpha=.75, clip_on=False)
        

        ## Pretty
        ax.set_xlim((-1, 7))
        ax.set_xticks((0, 2, 4, 6))
        ax.set_yticks((-3, 0, 3))
        my.plot.despine(ax)
        ax.set_xlabel('time from click (ms)')
        ax.set_ylabel(f'ABR ({MU}V)')
        
        # Savefig
        f.savefig(f'figures/PLOT_PEAKS_AT_LOUDEST_ACROSS_MICE__{suffix}.svg')
        f.savefig(f'figures/PLOT_PEAKS_AT_LOUDEST_ACROSS_MICE__{suffix}.png', dpi=300)

if PLOT_PEAK_METRIC_BY_WAVE_AND_CONFIG:
    """Connected pairs plot of latency and height for each wave * config
    This is run only for the first recording of each mouse (one per mouse)
    Waves in subplots, channel * speaker_side on x-axis.
    Points are colored by ipsi (green) vs contra (magenta), and a
    significance bracket spans the ipsi-contra pair within each channel.
    """
    
    ## Params
    # Which waves to include
    wave_l = ['W1p', 'W1n', 'W2p', 'W4p']
    pretty_name_l = ['wave 1 peak', 'wave 1 trough', 'wave 2 peak', 'wave 4 peak']
    
    # Config order for the x-axis
    config_order = ['VL L', 'VL R', 'VR L', 'VR R']
    
    # Color the points by ipsi vs contra
    ipsi_color_d = {True: 'green', False: 'magenta'}
    
    # x-positions of the two configs to compare within each channel
    bracket_x_d = {'VL': (0, 1), 'VR': (2, 3)}
    
    # Per-metric plotting params
    metric_params = {
        'latency_ms': {
            'label': 'latency (ms)',
            'ylim': (1, 6),
            'yticks': (1, 3, 5),
            'invert_trough': False,
            },
        'height': {
            'label': f'peak height ({MU}V)',
            'ylim': (-0.2, 6),
            'yticks': (0, 3, 6),
            'invert_trough': True,
            },
        }
    
    
    ## Slice big_ridges to first pre-HL recording only
    this_ridges = big_ridges.xs(
        False, level='after_HL').xs(0, level='n_experiment').droplevel('HL_type')
    
    # Slice loudest peaks only
    loudest = this_ridges.dropna(subset='wave_name').xs(
        loudest_db, level='level').reset_index()
    
    # Form "config" as a single ordered label for the x-axis
    loudest['config'] = loudest['channel'] + ' ' + loudest['speaker_side']
    
    # Also add an ipsi column
    loudest['ipsi'] = loudest['speaker_side'] == loudest['channel'].str[1]
    
    
    ## Iterate over metrics
    for metric, params in metric_params.items():
        
        # One subplot per wave
        f, axa = plt.subplots(
            1, len(wave_l),
            sharex=True, sharey=True,
            figsize=(7, 2.1)
            )
        f.subplots_adjust(bottom=.24, left=.1, right=.95, top=.89, wspace=.4)
        
        # Stats accumulators
        aov_pvals_l = []
        aov_fit_l = []
        tt_pvals_l = []
        stats_data_l = []
        stats_keys_l = []
        
        # Iterate over waves (subplots)
        for wave_name, ax in zip(wave_l, axa):
            
            # Slice this wave
            this_wave = loudest[loudest['wave_name'] == wave_name].copy()
            
            # Invert if trough (height is negative at a trough)
            if params['invert_trough'] and wave_name.endswith('n'):
                this_wave[metric] = -this_wave[metric]
            
            # Strip plot the metric for each config, colored by ipsi
            seaborn.stripplot(
                this_wave, x='config', y=metric,
                hue='ipsi', palette=ipsi_color_d, dodge=False, legend=False,
                marker=r'$\circ$', alpha=.5,
                order=config_order, ax=ax)
            
            # Connect configs within a mouse
            for mouse, unit_df in this_wave.groupby('mouse'):
                unit_df = unit_df.set_index('config').reindex(config_order)
                ax.plot(
                    range(len(config_order)),
                    unit_df[metric].values,
                    ls='-', color='gray', alpha=.5, lw=.75, clip_on=False,
                    )
            
            # Fancy x-axis
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels(['L', 'R', 'L', 'R'], rotation=0)
            ax.text(
                0.5, -0.3, 'VL', ha='center', va='center',
                transform=ax.get_xaxis_transform())
            ax.text(
                2.5, -0.3, 'VR', ha='center', va='center',
                transform=ax.get_xaxis_transform())            
            
            # Pretty
            if metric == 'latency_ms':
                ax.set_title(pretty_name_l[wave_l.index(wave_name)])
            ax.set_xlabel('')
            ax.set_ylabel(params['label'])
            ax.set_ylim(params['ylim'])
            ax.set_yticks(params['yticks'])
            my.plot.despine(ax)
            
            
            ## Stats
            # Assert that we have data from all mice on all configs
            assert (this_wave.groupby('mouse')['config'].nunique() == 4).all()
            
            # AOV
            # Drop the mouse fits (not interesting)
            aov = my.stats.anova(
                this_wave, f'{metric} ~ channel + ipsi + speaker_side + mouse')
            aov_pvals_l.append(aov['pvals'])
            aov_fit_l.append(
                aov['fit'].loc[~aov['fit'].index.str.startswith('fit_mouse')]
                )
            
            # Post hoc
            to_test = this_wave.set_index(
                ['channel', 'ipsi', 'mouse'])[metric].unstack('mouse').T
            ttp_VL = scipy.stats.ttest_rel(
                to_test[('VL', True)].values, to_test[('VL', False)].values
                ).pvalue
            ttp_VR = scipy.stats.ttest_rel(
                to_test[('VR', True)].values, to_test[('VR', False)].values
                ).pvalue
            tt_pvals_l.append(pandas.Series({'VL': ttp_VL, 'VR': ttp_VR}))
            
            # Store
            stats_data_l.append(to_test)
            stats_keys_l.append(wave_name)
            
            
            ## Mark each within-channel ipsi-contra comparison
            for this_channel, pvalue in [('VL', ttp_VL), ('VR', ttp_VR)]:
                
                # Get sigstr (continue if n.s.)
                sigstr = my.stats.pvalue_to_significance_string(pvalue)
                if sigstr == 'n.s.':
                    continue
                
                # Bracket in data x and axes y, so it ignores ylim
                x_left, x_right = bracket_x_d[this_channel]
                ax.plot(
                    [x_left, x_right], [.90, .90],
                    ls='-', color='k', lw=.75, clip_on=False,
                    transform=ax.get_xaxis_transform())
                ax.text(
                    (x_left + x_right) / 2, .85, sigstr,
                    ha='center', va='bottom',
                    transform=ax.get_xaxis_transform())
        
        
        ## Stats output
        # Concat over waves
        big_aov_pvals = pandas.concat(
            aov_pvals_l, keys=stats_keys_l, names=['wave'])
        big_aov_fit = pandas.concat(
            aov_fit_l, keys=stats_keys_l, names=['wave'])
        big_tt = pandas.concat(tt_pvals_l, keys=stats_keys_l, names=['wave'])
        big_stats_data = pandas.concat(
            stats_data_l, keys=stats_keys_l, names=['wave'])

        # Error check
        n_configs_by_mouse = big_stats_data.groupby('mouse').size()
        assert (n_configs_by_mouse == 4).all()
        n_mice = len(n_configs_by_mouse)
        
        # Sigstr (Intercept is uninteresting, Residual has no pvalue)
        drop_rows = ['p_Intercept', 'p_Residual']
        big_aov_sigstr = big_aov_pvals.drop(drop_rows, level=1).apply(
            my.stats.pvalue_to_significance_string)
        big_tt_sigstr = big_tt.apply(my.stats.pvalue_to_significance_string)
        
        # Mean metric and ipsi-contra diff
        mean_metric = big_stats_data.groupby(
            'wave').mean().T.groupby('ipsi').mean().T
        mean_metric['diff'] = (
            mean_metric.loc[:, True] - mean_metric.loc[:, False])
        mean_metric = mean_metric.T
        
        # Calculate grand mean and SD (averaging over channel * ipsi first)
        grand_data_to_agg = big_stats_data.mean(axis=1).unstack('wave')
        grand_data = pandas.concat({
            'grand_mean': grand_data_to_agg.mean(), 
            'grand_std': grand_data_to_agg.std(), 
            'grand_sem': grand_data_to_agg.sem(),
            }, axis=1)
        
        # Write out stats
        stats_filename = f'figures/STATS__PLOT_PEAK_METRIC_BY_WAVE_AND_CONFIG__{metric}'
        with open(stats_filename, 'w') as fi:
            fi.write(stats_filename + '\n')
            fi.write(f"n = {n_mice} mice\n")
            fi.write(f"grand mean by wave:\n {grand_data}\n")
            fi.write(f"AOV pvals by wave:\n{big_aov_pvals.unstack('wave')[wave_l]}\n")
            fi.write(f"AOV fit by wave:\n{big_aov_fit.unstack('wave')[wave_l]}\n")
            fi.write(f"AOV sig by wave:\n{big_aov_sigstr.unstack('wave')[wave_l]}\n")
            fi.write(f"paired t-test p-value by wave:\n{big_tt.unstack('wave')[wave_l]}\n")
            fi.write(f"paired t-test sigstr by wave:\n{big_tt_sigstr.unstack('wave')[wave_l]}\n")
            fi.write(f"mean by wave and ipsi:\n{mean_metric[wave_l]}\n")
        with open(stats_filename) as fi:
            print(''.join(fi.readlines()))
            
        
        ## Savefig
        f.savefig(f"figures/PLOT_PEAK_METRIC_BY_WAVE_AND_CONFIG__{metric}.svg")
        f.savefig(f"figures/PLOT_PEAK_METRIC_BY_WAVE_AND_CONFIG__{metric}.png", dpi=300)

if PLOT_PEAK_GROWTH_FUNCTIONS:
    """Plot peak-amplitude growth functions vs sound level, colored by wave
    
    This is run only for the first recording of each mouse (one per mouse),
    so mouse is the replicate. One line per mouse.
    """
    
    ## Params
    # Waves to include / color
    wave_l = ['W1p', 'W4p']
    growth_wave_colors = np.array(plt.cm.tab10.colors)[[1, 4]]
    
    # Panels
    channel_l = ['VL', 'VR']
    speaker_side_l = ['L', 'R']
    
    
    ## Slice big_ridges to first pre-HL recording only
    this_ridges = big_ridges.xs(
        False, level='after_HL').xs(0, level='n_experiment').droplevel('HL_type')
    
    # Positive peaks only
    this_ridges = this_ridges.xs('pos', level='sign')
    
    # Keep only the waves we plot
    this_ridges = this_ridges[this_ridges['wave_name'].isin(wave_l)]
    
    # peak_height indexed by mouse * channel * speaker_side * level * wave_name
    peak_height = this_ridges.set_index(
        'wave_name', append=True)['height'].droplevel('n_ridge')
    assert not peak_height.index.duplicated().any()
    
    # Mice as replicates (one column per mouse)
    peak_height_by_mouse = peak_height.unstack('mouse').sort_index()
    
    
    ## Figure
    f, axa = plt.subplots(
        len(channel_l), len(speaker_side_l),
        sharex=True, sharey=True, figsize=(4.2, 4))
    f.subplots_adjust(
        left=.25, right=.89, top=.95, bottom=.15, hspace=.15, wspace=.12)
    
    # Plot each channel * speaker_side
    for n_channel, channel in enumerate(channel_l):
        for n_speaker_side, speaker_side in enumerate(speaker_side_l):
            
            # Get ax
            ax = axa[n_channel, n_speaker_side]
            
            # Slice this config; index=(level, wave_name), cols=mouse
            this_config = peak_height_by_mouse.xs(
                channel, level='channel').xs(
                speaker_side, level='speaker_side')
            
            # Plot each wave
            for n_wave, wave_name in enumerate(wave_l):
                
                # level on rows, mice on columns
                this_traces = this_config.xs(wave_name, level='wave_name')
                
                # One line per mouse, x=level
                for mouse in this_traces.columns:
                    ax.plot(
                        this_traces.index, this_traces[mouse].values,
                        color=growth_wave_colors[n_wave], lw=.75)
            
            # Pretty
            my.plot.despine(ax)
    
    
    ## Pretty
    # Legend
    for n_wave, wave_name in enumerate(wave_l):
        f.text(
            .95, .68 - n_wave * .05, f'wave {wave_name[1]}',
            color=growth_wave_colors[n_wave], ha='center', va='center', size=12)
    
    # Axis labels
    f.text(.52, .01, 'sound level (dB SPL)', ha='center', va='bottom')
    f.text(.02, .56, f'peak amplitude ({MU}V)',
        rotation=90, ha='center', va='center')
    
    # Shared limits
    ax.set_yscale('log')
    ax.set_ylim((.1, 10))
    ax.set_xlim((20, 80))
    ax.set_xticks((30, 50, 70))
    
    # Label the channel
    for n_channel, channel in enumerate(channel_l):
        axa[n_channel, 0].set_ylabel(channel)
    
    # Label the speaker side
    axa[0, 0].set_title('sound from left')
    axa[0, 1].set_title('sound from right')
    
    
    ## Savefig
    f.savefig(os.path.join('figures', 'PLOT_PEAK_GROWTH_FUNCTIONS.svg'))
    f.savefig(os.path.join('figures', 'PLOT_PEAK_GROWTH_FUNCTIONS.png'), dpi=300)


if PLOT_PEAK_GROWTH_FUNCTIONS_AFTER_HL:
    """Peak-amplitude growth functions after HL, colored by HL_type
    One figure per wave (W1 and W4 only) showing all configs
    after_HL==True only, but showing ALL recordings (one line per recording)
    """
    
    ## Params
    # Waves to plot (one figure each)
    wave_l = ['W1p', 'W4p']
    
    # Color by HL_type
    hl_colors = {'bilateral': 'red', 'sham': 'gray'}
    
    # Panels
    channel_l = ['VL', 'VR']
    speaker_side_l = ['L', 'R']
    
    # The replicate unit (one line per recording)
    recording_levels = ['HL_type', 'n_experiment', 'mouse']
    
    
    ## Slice big_ridges to after-HL recordings only
    this_ridges = big_ridges.xs(True, level='after_HL')
    
    # Positive peaks only
    this_ridges = this_ridges.xs('pos', level='sign')
    
    # Keep only the waves we plot
    this_ridges = this_ridges[this_ridges['wave_name'].isin(wave_l)]
    
    # peak_height indexed by recording * channel * speaker_side * level * wave_name
    peak_height = this_ridges.set_index(
        'wave_name', append=True)['height'].droplevel('n_ridge')
    assert not peak_height.index.duplicated().any()    
    
    
    ## One figure per wave
    for wave_name in wave_l:
        
        # Slice this wave; recordings as replicates (one column per recording)
        peak_height_by_recording = peak_height.xs(
            wave_name, level='wave_name').unstack(recording_levels).sort_index()
        
        # Figure
        f, axa = plt.subplots(
            len(channel_l), len(speaker_side_l),
            sharex=True, sharey=True, figsize=(4.2, 4))
        f.subplots_adjust(
            left=.25, right=.89, top=.95, bottom=.15, hspace=.15, wspace=.12)
        
        # Plot each channel * speaker_side
        for n_channel, channel in enumerate(channel_l):
            for n_speaker_side, speaker_side in enumerate(speaker_side_l):
                
                # Get ax
                ax = axa[n_channel, n_speaker_side]
                
                # Slice this config; index=level, cols=recording
                this_config = peak_height_by_recording.xs(
                    channel, level='channel').xs(
                    speaker_side, level='speaker_side')
                
                # One line per recording, colored by HL_type
                for recording in this_config.columns:
                    
                    # recording is a tuple (HL_type, n_experiment, mouse)
                    hl_type = recording[recording_levels.index('HL_type')]
                    
                    # Plot this recording's growth function
                    ax.plot(
                        this_config.index, this_config[recording].values,
                        color=hl_colors[hl_type], lw=.75)
                
                # Pretty
                my.plot.despine(ax)
        
        
        ## Pretty
        # Legend
        for n_hl, (hl_type, color) in enumerate(hl_colors.items()):
            f.text(
                .95, .68 - n_hl * .05, hl_type,
                color=color, ha='center', va='center', size=12)
        
        # Axis labels
        f.text(.52, .01, 'sound level (dB SPL)', ha='center', va='bottom')
        f.text(.02, .56, f'peak amplitude ({MU}V)',
            rotation=90, ha='center', va='center')
        
        # Shared limits
        ax.set_yscale('log')
        ax.set_ylim((.1, 10))
        ax.set_xlim((20, 80))
        ax.set_xticks((30, 50, 70))
        
        # Label the channel
        for n_channel, channel in enumerate(channel_l):
            axa[n_channel, 0].set_ylabel(channel)
        
        # Label the speaker side
        axa[0, 0].set_title('sound from left')
        axa[0, 1].set_title('sound from right')
        
        # Title the figure with the wave
        f.suptitle(wave_name, x=.02, y=.99, ha='left')
        
        
        ## Savefig
        f.savefig(os.path.join('figures',
            f'PLOT_PEAK_GROWTH_FUNCTIONS_AFTER_HL__{wave_name}.svg'))
        f.savefig(os.path.join('figures',
            f'PLOT_PEAK_GROWTH_FUNCTIONS_AFTER_HL__{wave_name}.png'), dpi=300)


plt.show()