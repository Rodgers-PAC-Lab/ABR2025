

import os
import datetime
import glob
import json
import scipy.signal
import numpy as np
import pandas
import abr
import abr.signal_processing
import my.plot
import matplotlib.pyplot as plt
import tqdm

# This script identifies and categorizes clicks and error-checks them
plt.ion()

# import importlib
# importlib.reload(paclab.abr)
# importlib.reload(paclab.abr.signal_processing)

## Params
abr_start_sample = -40
abr_stop_sample = 120
abr_highpass_freq = 300
abr_lowpass_freq = 3000
sampling_rate = 16000 # TODO: store in recording_metadata?
neural_channel_numbers = [0, 2, 4]

# Outlier params
abs_max_sigma = 3
stdev_sigma = 3

## Cohort Analysis' Information
cohort_name = '250630_cohort'

## Paths
json_filepath = os.path.normpath(os.path.expanduser(
    '~/dev/scripts/rowan/ABR_data/filepaths.json'))
GUIdata_directory,Pickle_directory = abr.loading.get_ABR_data_paths(json_filepath)
# Use cohort pickle directory
cohort_pickle_directory = os.path.join(Pickle_directory, cohort_name)
if not os.path.exists(cohort_pickle_directory):
    try:
        os.mkdir(cohort_pickle_directory)
    except:
        print("No pickle directory exists and this script doesn't have permission to create one.")
        print("Check your Pickle_directory file path.")
# Put click figures here
click_data_dir = os.path.join(cohort_pickle_directory, 'click_validation')
if not os.path.exists(click_data_dir):
    os.mkdir(click_data_dir)


## Load results of main1
cohort_experiments = pandas.read_pickle(os.path.join(cohort_pickle_directory, 'cohort_experiments'))
recording_metadata = pandas.read_pickle(os.path.join(cohort_pickle_directory, 'recording_metadata'))

# Drop those with 'include' == False
recording_metadata = recording_metadata[recording_metadata['include'] == True]


## Load data from each recording
click_params_l = []
triggered_ad_l = []
triggered_neural_l = []
keys_l = []

# Iterate over recordings
for date, mouse, recording in tqdm.tqdm(recording_metadata.index):
    # print((date, mouse, recording))
    
    # Get the recording info
    this_recording = recording_metadata.loc[date].loc[mouse].loc[recording]

    
    ## Load raw data in volts
    # Get the filename
    folder_datestr = datetime.datetime.strftime(date, '%Y-%m-%d')
    recording_folder = os.path.normpath(this_recording['datafile'])
    # Load the data
    data = abr.loading.load_recording(recording_folder)
    data = data['data']
    # Parse into neural and speaker data
    # Presently, neural data is always on channels 0, 2, and 4 at most (maybe fewer)
    speaker_signal_V = data[:, 7]
    neural_data_V = data[:, neural_channel_numbers]


    ## Set up the click categories
    # Convert the autopilot amplitudes to voltages
    # This 1.34 is empirically determined to align autopilot with measured voltage
    log10_voltage = np.sort(np.log10(this_recording['amplitude']) + 1.34)
    
    # Generate labels .. convert voltage to dB and normalize to 70 dB, just 
    # to make it positive, this is not SPL
    amplitude_labels_old = np.rint(20 * log10_voltage + 70).astype(int)
    # SPL as measured with the fancy microphone
    amplitude_labels = np.array([49, 51, 54, 58, 61, 65, 69, 73, 77, 81, 85, 88, 91])

    # Convert the voltages to cuts
    amplitude_cuts = (log10_voltage[1:] + log10_voltage[:-1]) / 2

    # Add a first and last amplitude cut
    diff_cut = np.mean(np.diff(amplitude_cuts))
    amplitude_cuts = np.concatenate([
        [amplitude_cuts[0] - diff_cut],
        amplitude_cuts,
        [amplitude_cuts[-1] + diff_cut],
        ])
    
    
    ## Identify and categorize clicks
    # Use the least cut as the threshold
    threshold_V = 10 ** amplitude_cuts.min()
    
    # Identify clicks
    identified_clicks = abr.signal_processing.identify_click_times(
        speaker_signal_V, 
        threshold_V=threshold_V,
        sampling_rate=sampling_rate, 
        slice_start_sample=abr_start_sample, 
        slice_stop_sample=abr_stop_sample,
        )
    
    # Pull out the highpassed signal because we need it later
    speaker_signal_hp = identified_clicks['highpassed']

    # Categorize them
    click_params = abr.signal_processing.categorize_clicks(
        identified_clicks['peak_time_samples'], 
        speaker_signal_hp, 
        amplitude_cuts, 
        amplitude_labels,
        )    


    ## Extract each trigger from the audio signal
    triggered_ad = abr.signal_processing.slice_audio_on_clicks(
        speaker_signal_hp, click_params)


    ## Extract neural data locked to onsets
    # Highpass neural data
    nyquist_freq = sampling_rate / 2
    ahi, bhi = scipy.signal.butter(
        2, (
        abr_highpass_freq / nyquist_freq, 
        abr_lowpass_freq / nyquist_freq), 
        btype='bandpass')
    neural_data_hp = scipy.signal.filtfilt(ahi, bhi, neural_data_V, axis=0)

    # Extract highpassed neural data around triggers
    # Shape is (n_triggers, n_timepoints, n_channels)
    triggered_neural = np.array([
        neural_data_hp[trigger + abr_start_sample:trigger + abr_stop_sample]
        for trigger in click_params['t_samples']])

    # Remove channel as a level
    triggered_neural = triggered_neural.reshape(
        [triggered_neural.shape[0], -1])        

    # DataFrame
    triggered_neural = pandas.DataFrame(triggered_neural)
    triggered_neural.index = pandas.MultiIndex.from_frame(
        click_params[['label', 'polarity', 't_samples']])
    triggered_neural = triggered_neural.reorder_levels(
        ['label', 'polarity', 't_samples']).sort_index()

    # The columns are interdigitated samples and channels
    triggered_neural.columns = pandas.MultiIndex.from_product([
        pandas.Index(range(abr_start_sample, abr_stop_sample), name='timepoint'),
        pandas.Index(neural_channel_numbers, name='channel')
        ])
    
    # Rename the channels meaningfully
    triggered_neural = triggered_neural.rename(columns={
        0: this_recording.loc['ch0_config'], 
        2: this_recording.loc['ch2_config'],
        4: this_recording.loc['ch4_config']
        }, level='channel')
    
    # Drop NN if any
    triggered_neural = triggered_neural.drop(
        'NN', level='channel', axis=1, errors='ignore')
    
    # Stack channel
    triggered_neural = triggered_neural.stack('channel', future_stack=True)
    

    ## Store
    click_params_l.append(click_params)
    triggered_ad_l.append(triggered_ad)
    triggered_neural_l.append(triggered_neural)
    keys_l.append((date, mouse, recording))


## Concat
big_triggered_ad = pandas.concat(
    triggered_ad_l, keys=keys_l, names=['date', 'mouse', 'recording'])
big_triggered_neural = pandas.concat(
    triggered_neural_l, keys=keys_l, names=['date', 'mouse', 'recording'])
big_click_params = pandas.concat(
    click_params_l, keys=keys_l, names=['date', 'mouse', 'recording']
    ).set_index('t_samples', append=True).droplevel(None).sort_index()


## Store
big_triggered_ad.to_pickle(os.path.join(cohort_pickle_directory, 'big_triggered_ad'))
big_triggered_neural.to_pickle(os.path.join(cohort_pickle_directory, 'big_triggered_neural'))
big_click_params.to_pickle(os.path.join(cohort_pickle_directory, 'big_click_params'))

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

# Get the peak
# The peak is generally around sample 35, 25, 45-50, or more rarely 75
# But anything is possible
# Rather than be too strict, I'll just allow anything after the initial click,
# although this will generally be noise for the softest condition
# In the end the plots look almost identical with _peak or _rms
big_abr_evoked_peak = big_abrs.loc[:, 10:].abs().max(axis=1).unstack('label')

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
big_abrs.to_pickle(os.path.join(cohort_pickle_directory, 'big_abrs'))
threshold_db.to_pickle(os.path.join(cohort_pickle_directory, 'thresholds'))