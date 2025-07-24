import seaborn as sns
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
import paclab.abr.abr_analysis
import my.plot
import matplotlib.pyplot as plt
plt.ion()
my.plot.font_embed()

def HL_type(mouse):
    if mouse in bilateral_mouse_l:
        res = 'bilateral'
    elif mouse in sham_mouse_l:
        res = 'sham'
    else:
        res = 'ERROR, UNKNOWN MOUSE'
    return res

## Params
sampling_rate = 16000 # TODO: store in recording_metadata


## Cohort Analysis' Information
datestring = '250620'
day_directory = "_HL_cohort"

sham_mouse_l = ['Cat_227', 'Cat_228']
bilateral_mouse_l = ['Cat_226', 'Cat_229']

## Paths
GUIdata_directory,Pickle_directory = (paclab.abr.loading.get_ABR_data_paths())
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
cohort_experiments = pandas.read_pickle(
    os.path.join(cohort_pickle_directory, 'cohort_experiments'))
recording_metadata = pandas.read_pickle(
    os.path.join(cohort_pickle_directory, 'recording_metadata'))
# Fillna
cohort_experiments['HL'] = cohort_experiments['HL'].fillna('none')
cohort_experiments['HL_group'] = cohort_experiments['mouse'].apply(HL_type)


# Drop those with 'include' == False
recording_metadata = recording_metadata[recording_metadata['include'] == True]

threshold_db = pandas.read_pickle(os.path.join(cohort_pickle_directory,'thresholds'))
threshold_db['config'] = (threshold_db._get_label_or_level_values('channel',axis=0) +
                          threshold_db._get_label_or_level_values('speaker_side',axis=0))
threshold_db = threshold_db.join(
    cohort_experiments.set_index(['date', 'mouse'])['timepoint'], on=['date', 'mouse'])
threshold_db = threshold_db.reset_index().set_index(['mouse','timepoint','config'])
threshold_db = threshold_db.drop(columns=['date'])
threshold_db = threshold_db.sort_index(level='timepoint')

threshold_avgs_db = threshold_db.groupby(['timepoint', 'mouse','config','speaker_side', 'channel']).mean()

timepoint_avg_thresholds = threshold_db.copy().reset_index()
timepoint_avg_thresholds['pre_v_post'] = timepoint_avg_thresholds['timepoint'].apply(paclab.abr.abr_analysis.pre_or_post)
timepoint_avg_thresholds = timepoint_avg_thresholds.set_index(['mouse','config','pre_v_post'])
timepoint_avg_thresholds = timepoint_avg_thresholds.drop(columns='recording')
timepoint_avg_thresholds = timepoint_avg_thresholds.groupby(['mouse','config','pre_v_post','speaker_side','channel']).mean('threshold')
# Sort so 'pre' comes first
timepoint_avg_thresholds = timepoint_avg_thresholds.sort_index(axis=0,level='pre_v_post',ascending=False)

pre_v_post_thresholds = timepoint_avg_thresholds.copy().reset_index()
pre_v_post_thresholds['HL_group'] = pre_v_post_thresholds['mouse'].apply(HL_type)
pre_v_post_thresholds = pre_v_post_thresholds.set_index(['pre_v_post', 'HL_group', 'config','speaker_side','channel'])
avg_drop_bilateral = pre_v_post_thresholds.loc['post','bilateral',:]['threshold'].mean() - pre_v_post_thresholds.loc['pre','bilateral',:]['threshold'].mean()
avg_drop_sham = pre_v_post_thresholds.loc['post','sham',:]['threshold'].mean() - pre_v_post_thresholds.loc['pre','sham',:]['threshold'].mean()

prevpost_laterality = timepoint_avg_thresholds.copy().drop('LR',axis='index',level='channel')
prevpost_laterality = prevpost_laterality.reset_index()
prevpost_laterality['HL_group'] = prevpost_laterality['mouse'].apply(HL_type)
prevpost_laterality['laterality'] = [
    paclab.abr.abr_analysis.laterality_check(i_channel, i_speaker) for i_channel, i_speaker in zip(
        prevpost_laterality['channel'],
        prevpost_laterality['speaker_side']) ]
prevpost_laterality = prevpost_laterality.set_index(['pre_v_post','HL_group','laterality'])
prevpost_laterality = prevpost_laterality.drop(['mouse','config','speaker_side','channel'],axis='columns')
# prevpost_laterality = prevpost_laterality.groupby(['pre_v_post','HL_group','laterality']).mean()

avgs_unstacked = timepoint_avg_thresholds.unstack('pre_v_post')
avgs_unstacked = avgs_unstacked.rename(columns={'pre':'pre_threshold','post':'post_threshold'})
avgs_unstacked.columns = avgs_unstacked.columns.droplevel(0)
avgs_unstacked['diff'] = avgs_unstacked['post_threshold']-avgs_unstacked['pre_threshold']

mouse_l = threshold_avgs_db.index.get_level_values('mouse').unique()
configs_l = timepoint_avg_thresholds.index.get_level_values('config').unique()
subplot_nrows,subplot_ncols = my.plot.auto_subplot(len(mouse_l),return_fig=False)


print('Threshold drop:')
threshold_drop = []
for mouse in mouse_l:
    subdf=avgs_unstacked.loc[mouse]
    avg = subdf['diff'].mean()
    threshold_drop.append([mouse,avg])
    try:
        print(mouse+ ' ' + str(avg.round(2)))
    except:
        print("Couldn't calculate threshold drop, probably NaN")
threshold_drops_db = pandas.DataFrame(threshold_drop).dropna().rename(columns={0:'mouse',1:'threshold_drop'})

pre_HL_thresh = threshold_db.loc[:,['apreA','apreB'],:]
pre_HL_thresh = pre_HL_thresh.reset_index().set_index(['mouse','speaker_side','channel'])
pre_HL_thresh = pre_HL_thresh.drop(columns=['timepoint','config'])

compare_days = threshold_avgs_db.loc[['apreA','apreB']].unstack('mouse').droplevel(level=('channel','speaker_side'),axis='index')
compare_days = compare_days.sort_index(level='config')
check = compare_days.stack()['threshold']
check = check.unstack('mouse')
check = check.sort_index(axis='index',level='config')
checkdiff = check.loc['apreA'] - check.loc['apreB']

## Plotting
# Select plots
PLOT_THRESHOLD_BY_MOUSE = True
PLOT_AVG_THRESHOLD_PREVPOST = True
PLOT_PREHL_THRESH_BY_CONFIG = True
BARPLOT_PREVPOST_AVGED_BY_HL = True
BARPLOT_PREVPOST_AVGED_BY_LAT = True

if PLOT_THRESHOLD_BY_MOUSE:
    if subplot_ncols == 2:
        fig_size = (9, 7)
    else:
        fig_size = (9, 8)
    fig,axs = plt.subplots(round(len(mouse_l)/2), 2, sharey=True, sharex=True, figsize=fig_size)
    fig.suptitle('Threshold over time- by mouse')
    for idx, mouse in enumerate(mouse_l):
        ax = axs.flatten()[idx]
        subdf = threshold_avgs_db.loc[:,mouse,:].reset_index()
        sns.boxplot(subdf,x='timepoint', y='threshold', ax=ax,
                    fill=None)
        if idx==subplot_ncols-1:
            sns.swarmplot(subdf, x='timepoint', y='threshold', ax=ax,
                          hue='config', hue_order=['LRL', 'LRR', 'LVL', 'LVR', 'RVL', 'RVR'])
        else:
            sns.swarmplot(subdf,x='timepoint', y='threshold',ax=ax,
                      hue='config', hue_order=['LRL', 'LRR', 'LVL', 'LVR', 'RVL', 'RVR'],
                      legend=False)
        ax.set_title(mouse)
    # Place legend and adjust subplots based on layout
    ax = axs.flatten()[subplot_ncols-1]
    if subplot_ncols == 2 and subplot_nrows==3:
        ax.legend(bbox_to_anchor=(1.3, 1.5), markerscale=1.5)
        plt.subplots_adjust(left=0.08, right=0.85, top=0.9, bottom=0.1, wspace=0.05, hspace=0.1)
    elif subplot_ncols == 2 and subplot_nrows==2:
        ax.legend(bbox_to_anchor=(1.25, 0.2), markerscale=1.5)
        plt.subplots_adjust(left=0.08, right=0.85, top=0.9, bottom=0.1, wspace=0.05, hspace=0.1)
    else:
        ax.legend(loc='upper right', bbox_to_anchor=(1.4, 0.1), markerscale=1.5)
        plt.subplots_adjust(left=0.08, right=0.89, top=0.9, bottom=0.1, wspace=0.05, hspace=0.1)

    savename = os.path.join(cohort_pickle_directory,(cohort_name+'_THRESHOLD_BY_MOUSE'))
    fig.savefig((savename + '.png'), dpi=300)
    fig.savefig((savename + '.svg'))

if PLOT_AVG_THRESHOLD_PREVPOST:
    f,axa = plt.subplots(int(len(configs_l)/2),2, sharex=True, sharey=True)
    gobj = timepoint_avg_thresholds.groupby(['mouse','config'])
    for (mouse, config),subdf in gobj:
        ax = axa.flatten()[configs_l.get_loc(config)]
        ax.plot(subdf.index.get_level_values('pre_v_post'),subdf['threshold'],marker='.', label=mouse)
        ax.set_title(config)
        # ax.set_xticks([0,1],labels=['pre-HL','post-HL'])
        ax.set_xlim(-0.25,1.25)

    for ax in axa[-1,:]:
        ax.set_xlabel('timepoint')
    for ax in axa[:,0]:
        ax.set_ylabel('threshold (dB)')
    if subplot_ncols==2:
        axa[-1, -1].legend(bbox_to_anchor=(1.6, 2))
    else:
        axa[-1, -1].legend(loc='upper right', bbox_to_anchor=(2.1, 1.5))
    f.suptitle('Avg threshold pre-v-post')
    f.subplots_adjust(right=0.8)
    savename = os.path.join(cohort_pickle_directory, (cohort_name + '_AVG_THRESHOLD_PREVPOST'))
    f.savefig((savename + '.png'), dpi=300)
    f.savefig((savename + '.svg'))

if PLOT_PREHL_THRESH_BY_CONFIG:
    f,axa = plt.subplots(1,3,sharex=True,sharey=True,figsize=(9,8))
    for idx,channel in enumerate(['LR', 'LV', 'RV']):
        ax = axa[idx]
        # sns.boxplot(pre_HL_thresh.loc[:,:,channel], x="speaker_side", y="threshold", fill=None,ax=ax)
        sns.barplot(pre_HL_thresh.loc[:,:,channel], x="speaker_side", y="threshold", fill=None,ax=ax)
        if idx != 2:
            sns.swarmplot(pre_HL_thresh.loc[:,:,channel],x="speaker_side", y="threshold",hue='mouse',ax=ax,legend=False)
            # sns.stripplot(pre_HL_thresh.loc[:, :, channel], x="speaker_side", y="threshold", hue='mouse', ax=ax,
            #               legend=False, jitter=True, marker='x')
        elif idx==2:
            sns.swarmplot(pre_HL_thresh.loc[:, :, channel], x="speaker_side", y="threshold", hue='mouse', ax=ax)
            ax.legend(bbox_to_anchor=(1.05,0.5))
        ax.set_title(channel + ' channel')
        f.suptitle('Average thresholds')
        f.subplots_adjust(left=0.1,right=0.8)
    savename = os.path.join(cohort_pickle_directory, (cohort_name + '_PRE-HL_THRESH_BY_CONFIG'))
    f.savefig((savename + '.png'), dpi=300)
    f.savefig((savename + '.svg'))

if BARPLOT_PREVPOST_AVGED_BY_HL:
    f,ax = plt.subplots(figsize=(4,4))
    f.suptitle('Threshold average pre-HL v post-HL')
    sns.barplot(pre_v_post_thresholds,x='pre_v_post',y='threshold',hue='HL_group',hue_order=('sham','bilateral'),ax=ax)
    ax.set_xticks((0,1),labels=['pre-HL', 'post-HL'])
    diffs_l = []
    for c in ax.containers:
        ax.bar_label(c, fmt='%0.0f', padding=3)
        diff = c.datavalues[1] - c.datavalues[0]
        diffs_l.append(diff)
    ax.set_ylabel('threshold (dB)')
    ax.set_xlabel('')
    savename = os.path.join(cohort_pickle_directory, ('PREVPOST_BARPLOT_AVGED_BY_HL'))
    f.savefig((savename + '.png'), dpi=300)
    f.savefig((savename + '.svg'))

if BARPLOT_PREVPOST_AVGED_BY_LAT:
    f,axa = plt.subplots(1,2,figsize=(8,4))
    f.suptitle('Threshold average pre-HL v post-HL')
    for lat in prevpost_laterality.index.get_level_values('laterality').unique():
        if lat=='ipsilateral':
            sns.barplot(prevpost_laterality.xs(lat,level='laterality'),
                x='pre_v_post',y='threshold',hue='HL_group',hue_order=('sham','bilateral'),ax=axa[0])
            axa[0].set_title(lat)
        if lat=='contralateral':
            sns.barplot(prevpost_laterality.xs(lat,level='laterality'),
                x='pre_v_post',y='threshold',hue='HL_group',hue_order=('sham','bilateral'),ax=axa[1])
            axa[1].set_title(lat)
    for ax in axa:
        ax.set_xticks((0,1),labels=['pre-HL', 'post-HL'])
        diffs_l = []
        for c in ax.containers:
            ax.bar_label(c, fmt='%0.0f', padding=3)
            diff = c.datavalues[1] - c.datavalues[0]
            diffs_l.append(diff)
        ax.set_ylabel('threshold (dB)')
        ax.set_xlabel('')
    savename = os.path.join(cohort_pickle_directory, ('PREVPOST_BARPLOT_AVGED_BY_LAT'))
    f.savefig((savename + '.png'), dpi=300)
    f.savefig((savename + '.svg'))