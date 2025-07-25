## Remove outliers, aggregate, and average ABRs

# Count the number of trials in each experiment
trial_counts = big_triggered_neural.groupby(['date', 'mouse', 'recording', 'label', 'channel']).count()

# Iterate over recordings
abrs_l = []
arts_l = []
keys_l = []
for date, mouse, recording in recording_metadata.index:
    # Slice
    triggered_neural = big_triggered_neural.loc[date].loc[mouse].loc[recording]

    ## Identify outlier trials, separately by channel
    triggered_neural2 = triggered_neural.groupby('channel').apply(
        lambda df: abr.signal_processing.trim_outliers(
            df.droplevel('channel'),
            abs_max_sigma=abs_max_sigma,
            stdev_sigma=stdev_sigma,
        ))

    # Reorder levels to be like triggered_neural
    triggered_neural2 = triggered_neural2.reorder_levels(
        triggered_neural.index.names).sort_index()

    ## Aggregate
    # Average by polarity, label, channel over t_samples
    avg_by_condition = triggered_neural2.groupby(
        ['polarity', 'channel', 'label']).mean()

    # The ABR adds over polarity
    avg_abrs = (avg_by_condition.loc[True] + avg_by_condition.loc[False]) / 2

    # The artefact subtracts over polarity
    avg_arts = (avg_by_condition.loc[True] - avg_by_condition.loc[False]) / 2

    ## Store
    abrs_l.append(avg_abrs)
    arts_l.append(avg_arts)
    keys_l.append((date, mouse, recording))

# Concat
big_abrs = pandas.concat(abrs_l, keys=keys_l, names=['date', 'mouse', 'recording'])
big_arts = pandas.concat(arts_l, keys=keys_l, names=['date', 'mouse', 'recording'])

# TODO: identify which recordings have large arts and drop them


## Join on speaker_side
# Should do this at the same time as joining channel
idx = big_abrs.index.to_frame().reset_index(drop=True)
idx = idx.join(recording_metadata['speaker_side'], on=['date', 'mouse', 'recording'])
big_abrs.index = pandas.MultiIndex.from_frame(idx)
big_abrs = big_abrs.reorder_levels(
    ['date', 'mouse', 'speaker_side', 'recording', 'channel', 'label']
).sort_index()

# Same for arts
idx = big_arts.index.to_frame().reset_index(drop=True)
idx = idx.join(recording_metadata['speaker_side'], on=['date', 'mouse', 'recording'])
big_arts.index = pandas.MultiIndex.from_frame(idx)
big_arts = big_arts.reorder_levels(
    ['date', 'mouse', 'speaker_side', 'recording', 'channel', 'label']
).sort_index()

## Further aggregate over recordings
# Average recordings together where everything else is the same
big_abrs = big_abrs.groupby(['date', 'mouse', 'recording', 'channel', 'speaker_side', 'label']).mean()

## Calculate the stdev(ABR) as a function of level
# window=20 (1.25 ms) seems the best compromise between smoothing the whole
# response and localizing it to a reasonably narrow window (and not extending
# into the baseline period)
# Would be good to extract more baseline to use here
# The peak is around sample 34 (2.1 ms), ie sample 24 - 44, and there is a
# variable later peak.
big_abr_stds = big_abrs.T.rolling(window=20, center=True, min_periods=1).std().T

# Use samples -40 to -20 as baseline
# Generally this should be <0.25 uV, but the actual value depends on how
# the averaging was done
big_abr_baseline_rms = big_abr_stds.loc[:, -30].unstack('label')

# Choose a baseline for each recording as the median over levels
# It's lognormal so mean might be skewed. A mean of log could be good
big_abr_baseline_rms = big_abr_baseline_rms.median(axis=1)

# Use samples 24 - 44 as evoked peak
# Evoked response increases linearly with level in dB
# Interestingly, each recording appears to be multiplicatively scaled
# (shifted up and down on a log plot). The variability in microvolts increases
# with level, but the variability in log-units is consistent over level.
big_abr_evoked_rms = big_abr_stds.loc[:, 34].unstack('label')

# Determine threshold crossing as 3*baseline. Note: more averaging will
# decrease baseline and therefore threshold, as will better noise levels.
# But this still seems better than a fixed threshold in microvolts.
# TODO: consider smoothing traces before finding threshold crossing
over_thresh = big_abr_evoked_rms.T > 3 * big_abr_baseline_rms
over_thresh = over_thresh.T.stack()
# threshold
# typically a bit better on LR even though LR has slightly higher baseline
threshold_db = over_thresh.loc[over_thresh.values].groupby(
    ['date', 'mouse', 'recording', 'channel', 'speaker_side']).apply(
    lambda df: df.index[0][-1])

# reindex to get those that are never above threshold
threshold_db = threshold_db.reindex(big_abr_baseline_rms.index)
threshold_db = pandas.DataFrame(threshold_db, columns=['threshold'])


## Store
big_triggered_ad.to_pickle(os.path.join(output_directory, 'big_triggered_ad'))
big_triggered_neural.to_pickle(os.path.join(output_directory, 'big_triggered_neural'))
big_click_params.to_pickle(os.path.join(output_directory, 'big_click_params'))
big_abrs.to_pickle(os.path.join(output_directory, 'big_abrs'))
threshold_db.to_pickle(os.path.join(output_directory, 'thresholds'))