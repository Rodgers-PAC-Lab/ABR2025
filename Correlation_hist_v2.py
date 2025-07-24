# Just a clean script to play with historgrams or swarmplots for the correlation data
import os
import datetime
import glob
import json
import matplotlib
matplotlib.use('TkAgg')
import scipy.signal
import numpy as np
import pandas
import paclab.abr
import my.plot
import matplotlib.pyplot as plt
import seaborn
from itertools import combinations

def check_combo(combo):
    if combo=='apreAapreB':
        type = 'btwn_apre'
    elif combo=='postApostB':
        type='btwn_post'
    else:
        type='pre_v_post'
    return type
def HL_type(mouse):
    if mouse in bilateral_mouse_l:
        res = 'bilateral'
    elif mouse in sham_mouse_l:
        res = 'sham'
    else:
        res = 'ERROR, UNKNOWN MOUSE'
    return res

## Plot params
my.plot.manuscript_defaults()
my.plot.font_embed()

## Params
sampling_rate = 16000  # TODO: store in recording_metadata

## Cohort Analysis' Information
datestring = '250630'
day_directory = "_cohort"

# Tenatative because I'm blinded, but come on it's obvious
sham_mouse_l = ['Cat_227', 'Cat_228']
bilateral_mouse_l = ['Cat_226', 'Cat_229']

## Paths
GUIdata_directory, Pickle_directory = (paclab.abr.loading.get_ABR_data_paths())
cohort_name = datestring + day_directory
# Use cohort pickle directory
cohort_pickle_directory = os.path.join(Pickle_directory, cohort_name)
if not os.path.exists(cohort_pickle_directory):
    try:
        os.mkdir(cohort_pickle_directory)
    except:
        print("No pickle directory exists and this script doesn't have permission to create one.")
        print("Check your Pickle_directory file path.")

## Load results of Step1
cohort_experiments = pandas.read_pickle(os.path.join(cohort_pickle_directory, 'cohort_experiments'))
recording_metadata = pandas.read_pickle(os.path.join(cohort_pickle_directory, 'recording_metadata'))
wthn_mouse_R2_all_sound = pandas.read_pickle(os.path.join(cohort_pickle_directory, 'wthn_mouse_R2_all_sound'))
wthn_mouse_R2 = pandas.read_pickle(os.path.join(cohort_pickle_directory,'wthn_mouse_R2'))
btwn_mice_preHL = pandas.read_pickle(os.path.join(cohort_pickle_directory,'btwn_mouse_R2'))

# Drop those with 'include' == False
recording_metadata = recording_metadata[recording_metadata['include'] == True]

## Load results of Step2

big_triggered_neural = pandas.read_pickle(
    os.path.join(cohort_pickle_directory, 'big_triggered_neural'))
# Fillna
cohort_experiments['HL'] = cohort_experiments['HL'].fillna('none')

wthn_mouse_R2['type'] = wthn_mouse_R2['combo'].apply(check_combo)

correl_small = wthn_mouse_R2_all_sound.set_index(['channel', 'speaker_side', 'label'])
correl_small['type'] = correl_small['combo'].apply(check_combo)
correl_small = correl_small.reset_index().set_index([
    'type', 'channel', 'speaker_side', 'label'])
correl_small = correl_small.sort_index()
btwnpre_df = correl_small.loc['btwn_apre'].drop(columns='combo')



wthn_mouse_R2['config'] = wthn_mouse_R2['channel'] + wthn_mouse_R2['speaker_side']
wthn_mouse_R2 = wthn_mouse_R2.set_index(['mouse','type','config','channel','speaker_side'])
wthn_mouse_R2= wthn_mouse_R2.sort_index(level='type')
flat_correls_small = wthn_mouse_R2.drop(columns='combo').reset_index()
flat_correls_small['HL_group'] = flat_correls_small['mouse'].apply(HL_type)
flat_correls_small = flat_correls_small.drop(['channel','speaker_side'],axis='columns')

# Get the combos of mice in one field
btwn_mice_preHL['mouse'] = btwn_mice_preHL['mouseA'] + btwn_mice_preHL['mouseB']


mouse_l = cohort_experiments['mouse'].unique()
palette = seaborn.color_palette(n_colors=len(mouse_l))
mouse_colors = dict(zip(sorted(mouse_l), palette))

flat_correls_small['m_hue'] = flat_correls_small['mouse'].apply(lambda x: mouse_colors[x])
flat_correls_small = flat_correls_small.set_index(['mouse','type','config','HL_group'])

order = mouse_l.tolist()

#Make manual legend handles
handles_l = []
for mouse in order:
    if mouse in sham_mouse_l:
        m_handle = matplotlib.lines.Line2D([], [], marker='o', color=mouse_colors[mouse], label=mouse + ' - sham', lw=0)
    elif mouse in bilateral_mouse_l:
        m_handle = matplotlib.lines.Line2D([], [], marker='x', color=mouse_colors[mouse], label=mouse + ' - bilateral', lw=0)
    else:
        m_handle = matplotlib.lines.Line2D([], [], marker='*', color=mouse_colors[mouse], label=mouse, lw=0)
    handles_l.append(m_handle)

# Plots using data NOT flattened by sound level
PLOT_R2_BY_CHANNEL = False

# Plots using data flattened by sound level
PLOT_R2_BY_MOUSE = False
PLOT_PREHL_WITHIN_MOUSE = False
HIST_PREHL_WITHIN_MOUSE = True
PLOT_PREHL_WITHIN_V_BETWEEN = False

if PLOT_R2_BY_CHANNEL:
    fig = seaborn.catplot(data=btwnpre_df, kind='swarm', x='channel', y='pearsonR2',
                          hue='label', dodge=True,
                          palette=seaborn.color_palette("dark:#5A9_r", as_cmap=True),
                          legend_out=True, legend='full')

    ax = fig.ax
    seaborn.boxenplot(data=btwnpre_df, x='channel', y='pearsonR2', fill=False, ax=ax, legend=False, showfliers=False)
    ax.set_title('Correlation between pre-HL recordings\nGrouped by channel')
    fig.tight_layout()
    plt.show()
    savename = 'R2_by_channel'
    fig.savefig(os.path.join(cohort_pickle_directory, savename + '.svg'))
    fig.savefig(os.path.join(cohort_pickle_directory, savename + '.png'), dpi=300)

if PLOT_R2_BY_MOUSE:
    # Plot histograms for data flattened by sound level
    # Plot bilateral mice with an 'x' as a marker and sham mice with a 'o'
    # The legend

    fig, ax = plt.subplots(figsize=(6, 5))

    for HL_group in flat_correls_small.index.get_level_values('HL_group').unique():
        subdf = flat_correls_small.xs(HL_group, level='HL_group')
        if HL_group == 'sham':
            seaborn.swarmplot(data=subdf, x='type', y='pearsonR2',
                          hue='mouse', hue_order=order, ax=ax,alpha=0.8)
        elif HL_group == 'bilateral':
            seaborn.swarmplot(data=flat_correls_small.xs(HL_group, level='HL_group'), x='type', y='pearsonR2',
                          hue='mouse', hue_order=order, marker='x', linewidth=1.5, ax=ax)
        else:
            seaborn.swarmplot(data=flat_correls_small.xs(HL_group, level='HL_group'), x='type', y='pearsonR2',
                          hue='mouse', hue_order=order, marker='*', linewidth=1.5, ax=ax)

    ax.legend(handles=handles_l, loc='upper left', bbox_to_anchor=(0.98, 0.6), handlelength=0)
    fig.suptitle('Correlation for ABR recordings\nConcatted across all sound levels')
    fig.subplots_adjust(bottom=0.14, left=0.15, right=0.67)
    plt.show()
    savename = 'R2_by_mouse'
    fig.savefig(os.path.join(cohort_pickle_directory, savename + '.svg'))
    fig.savefig(os.path.join(cohort_pickle_directory, savename + '.png'), dpi=300)

if PLOT_PREHL_WITHIN_MOUSE:
    subdf = wthn_mouse_R2.xs('btwn_apre',axis=0,level='type')
    # fig, ax = plt.subplots(figsize=(6, 5))
    fig, ax = plt.subplots(figsize=(4, 3.5))
    seaborn.boxplot(subdf, y='pearsonR2', ax=ax, fill=None, showfliers=False, legend=False)
    seaborn.swarmplot(subdf, y='pearsonR2', hue='mouse', dodge=True, ax=ax)
    ax.get_legend().set(loc='upper left', bbox_to_anchor=(0.97, 0.7))
    my.plot.despine(ax)
    fig.tight_layout()
    savename = 'PREHL_WITHIN_MOUSE'
    fig.savefig(os.path.join(cohort_pickle_directory, savename + '.svg'))
    fig.savefig(os.path.join(cohort_pickle_directory, savename + '.png'), dpi=300)

    fig, ax = plt.subplots(figsize=(4, 3.5))
    seaborn.boxplot(subdf, y='pearsonR2', ax=ax, fill=None, showfliers=False, legend=False)
    seaborn.swarmplot(subdf, y='pearsonR2', hue='mouse', ax=ax)
    ax.get_legend().set(loc='upper left', bbox_to_anchor=(0.85, 0.7), frame_on=False)
    my.plot.despine(ax)
    fig.tight_layout()
    savename = 'PREHL_WITHIN_MOUSE_MIXED'
    fig.savefig(os.path.join(cohort_pickle_directory, savename + '.svg'))
    fig.savefig(os.path.join(cohort_pickle_directory, savename + '.png'), dpi=300)

    fig, ax = plt.subplots(figsize=(5, 4))
    seaborn.boxplot(subdf, y='pearsonR2', x='mouse', ax=ax, fill=None, showfliers=False, legend=False)
    seaborn.swarmplot(subdf, y='pearsonR2', hue='mouse', x='mouse', ax=ax)
    my.plot.despine(ax)
    ax.tick_params(axis='x', labelrotation=30)
    # fig.tight_layout()
    fig.subplots_adjust(top=0.95,bottom=0.27,right=0.99)
    savename = 'PREHL_WITHIN_MOUSE_SEPERATED'
    fig.savefig(os.path.join(cohort_pickle_directory, savename + '.svg'))
    fig.savefig(os.path.join(cohort_pickle_directory, savename + '.png'), dpi=300)
if HIST_PREHL_WITHIN_MOUSE:
    subdf = wthn_mouse_R2.xs('btwn_apre',axis=0,level='type')
    # fig,ax = plt.subplots(figsize=(4,4))
    # ax.hist(subdf['pearsonR2'],bins=np.linspace(0, 1, 21))
    # ax.set_ylabel('n recordings')
    # ax.set_xlabel('pearson R2')
    # fig.tight_layout()
    # savename = 'PREHL_WITHIN_MOUSE_HISTOGRAM'
    # fig.savefig(os.path.join(cohort_pickle_directory, savename + '.svg'))
    # fig.savefig(os.path.join(cohort_pickle_directory, savename + '.png'), dpi=300)

    fig,ax = plt.subplots(figsize=(4,4))
    seaborn.histplot(subdf, x='pearsonR2', bins=np.linspace(0, 1, 21))
    fig.tight_layout()
    savename = 'PREHL_WITHIN_MOUSE_HISTOGRAM'
    fig.savefig(os.path.join(cohort_pickle_directory, savename + '.svg'))
    fig.savefig(os.path.join(cohort_pickle_directory, savename + '.png'), dpi=300)