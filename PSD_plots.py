### This script uses cohort_experiments and recording_metadata from Rowan's old
# Step1_cohort_info script. It goes through recording_metadata and
# loads the raw neural data. Then it finds the PSD of the raw data and
# a histogram of power in the ABR band, and plot both of these.
# Writes the following to the output directory:
#   big_Pxx: dataframe of PSD and freqiencies
#   big_rms: dataframe of rms over a recording, for both raw and hp data
#   PSD_BY_CHANNEL .png and .svg: PSD plot of all 3 channels
#   RMS_ABR_BAND .png and .svg: Histogram of rms, filtered for ABR band

import os
import datetime
import glob
import json
import scipy.signal
import numpy as np
import pandas
import paclab.abr
import my.plot
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tqdm
import importlib
import seaborn

def psd(data, NFFT=None, Fs=None, detrend='mean', window=None, noverlap=None,
        scale_by_freq=None, **kwargs):
    """Compute power spectral density.

    A wrapper around mlab.psd with more documentation and slightly different
    defaults.

    Arguments
    ---
    data : The signal to analyze. Must be 1d
    NFFT : defaults to 256 in mlab.psd
    Fs : defaults to 2 in mlab.psd
    detrend : default is 'mean', overriding default in mlab.psd
    window : defaults to Hanning in mlab.psd
    noverlap : defaults to 0 in mlab.psd
        50% or 75% of NFFT is a good choice in data-limited situations
    scale_by_freq : defaults to True in mlab.psd
    **kwargs : passed to mlab.psd

    Notes on scale_by_freq
    ---
    Using scale_by_freq = False makes the sum of the PSD independent of NFFT
    Using scale_by_freq = True makes the values of the PSD comparable for
    different NFFT
    In both cases, the result is independent of the length of the data
    With scale_by_freq = False, ppxx.sum() is roughly comparable to
      the mean of the data squared (but about half as much, for some reason)
    With scale_by_freq = True, the returned results are smaller by a factor
      roughly equal to sample_rate, but not exactly, because the window
      correction is done differently

    With scale_by_freq = True
      The sum of the PSD is proportional to NFFT/sample_rate
      Multiplying the PSD by sample_rate/NFFT and then summing it
        gives something that is roughly equal to np.mean(signal ** 2)
      To sum up over a frequency range, could ignore NFFT and multiply
        by something like bandwidth/sample_rate, but I am not sure.
    With scale_by_freq = False
      The sum of the PSD is independent of NFFT and sample_rate
      The sum of the PSD is slightly more than np.mean(signal ** 2)
      To sum up over a frequency range, need to account for the number of
        points in that range, which depends on NFFT.
    In both cases
      The sum of the PSD is independent of the length of the signal
    The reason that the answers are not proportional to each other
    is because the window correction is done differently.

    scale_by_freq = True generally seems to be more accurate
    I imagine scale_by_freq = False might be better for quickly reading
    off a value of a peak
    """
    # Run PSD
    Pxx, freqs = matplotlib.mlab.psd(
        data,
        NFFT=NFFT,
        Fs=Fs,
        detrend=detrend,
        window=window,
        noverlap=noverlap,
        scale_by_freq=scale_by_freq,
        **kwargs,
    )

    # Return
    return Pxx, freqs

plt.ion()

importlib.reload(paclab.abr)
importlib.reload(paclab.abr.signal_processing)

## Params
abr_start_sample = -40
abr_stop_sample = 120
abr_highpass_freq = 300
abr_lowpass_freq = 3000
sampling_rate = 16000  # TODO: store in recording_metadata?
neural_channel_numbers = [0, 2, 4]

# Outlier params
abs_max_sigma = 3
stdev_sigma = 3

## Cohort Analysis' Information
datestring = '250630'
day_directory = "_cohort"

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
Pxx_df_l = []
rec_l = []
rms_l = []

# Iterate over recordings
all_neural_V = []
# Temporarily using the first 10 recordings so it's not absurdly huge
for date, mouse, recording in tqdm.tqdm(recording_metadata.head(20).index):
    # print((date, mouse, recording))

    ## Get metadata about recording
    experimenter = cohort_experiments.set_index(['date','mouse']).loc[date,mouse]['experimenter']
    # Get experimenter dir
    # experimenter_dir = cohort_experiments.set_index(['date','experimenter']).loc[date].loc[experimenter].loc['full_path']

    # Get the recording info
    this_recording = recording_metadata.loc[date].loc[mouse].loc[recording]

    ## Load raw data in volts
    # Get the filename
    folder_datestr = datetime.datetime.strftime(date, '%Y-%m-%d')
    recording_folder = os.path.normpath(this_recording['datafile'])
    # Load the data
    data = paclab.abr.loading.load_recording(recording_folder)
    data = data['data']
    # Parse into neural and speaker data
    # Presently, neural data is always on channels 0, 2, and 4 at most (maybe fewer)
    speaker_signal_V = data[:, 7]
    neural_data_V = data[:, neural_channel_numbers]

    # Get the channel names
    # These must match neural_channel_numbers above
    neural_channel_names = [
        this_recording.loc['ch0_config'],
        this_recording.loc['ch2_config'],
        this_recording.loc['ch4_config'],
        ]

    # DataFrame labeled by channel
    neural_data_df = pandas.DataFrame(neural_data_V, columns=neural_channel_names)

    # Drop NN
    neural_data_df = neural_data_df.drop('NN', axis=1, errors='ignore')

    ## Identify and fix glitches
    # Certainly any time it is above 1, something has gone wrong, I don't think
    # it can even go above a few hundred mV
    # A more conservative threshold might be better, but there are few values
    # in between 1 and the real data
    excursions = (neural_data_df.abs() > 0.3).any(axis=1)
    excursions = np.where(excursions)[0]

    # Blank out anything within 1000 samples of an excursion
    # Packet size is about 500 so this feels about right
    drop_mask = my.misc.times_near_times(
        excursions,
        np.arange(len(neural_data_df)),
        dstart=-1000, dstop=1000)

    ## Drop drop mask
    if drop_mask.any():
        print(f'{recording_dir} dropping {drop_mask.mean()}')
    neural_data_df = neural_data_df.loc[~drop_mask].reset_index(drop=True)

    ## Run PSD on each columns
    Pxx_l = []
    for col in neural_data_df.values.T:
        # Data is in V
        Pxx, freqs = psd(col, NFFT=16384, Fs=sampling_rate)
        Pxx_l.append(Pxx)

    # DataFrame
    Pxx_df = pandas.DataFrame(
        np.transpose(Pxx_l),
        columns=neural_data_df.columns, index=freqs)
    Pxx_df.index.name = 'freq'
    Pxx_df.columns.name = 'channel'

    ## Highpass neural data
    nyquist_freq = sampling_rate / 2
    ahi, bhi = scipy.signal.butter(
        2, (
            abr_highpass_freq / nyquist_freq,
            abr_lowpass_freq / nyquist_freq),
        btype='bandpass')
    neural_data_hp = scipy.signal.filtfilt(ahi, bhi, neural_data_df, axis=0)
    neural_data_hp = pandas.DataFrame(
        neural_data_hp, columns=neural_data_df.columns)

    ## Compute rms
    # This is in V
    raw_rms = neural_data_df.std()
    hp_rms = neural_data_hp.std()
    rms = pandas.concat([raw_rms, hp_rms], keys=['raw', 'hp'], axis=1)
    rms.columns.name = 'signal'

    ## Store
    Pxx_df_l.append(Pxx_df)
    rms_l.append(rms)
    rec_l.append({
        'drop_frac': drop_mask.mean(),
        'recording_dir': recording_folder,
        })
    keys_l.append((date, experimenter, mouse, recording))

## Concat
big_Pxx = pandas.concat(
    Pxx_df_l, keys=keys_l,
    names=['date', 'experimenter', 'mouse', 'recording'])
big_rms = pandas.concat(
    rms_l, keys=keys_l,
    names=['date', 'experimenter', 'mouse', 'recording'])

## Plot
# Convert to db re 1 uV
topl = 10 * np.log10(big_Pxx * 1e12)

# Get freq on columns, and channels on index
topl = topl.unstack('freq').stack('channel', future_stack=True)

# Drop the nyquist frequency
topl = topl.iloc[:, :-1]

# Groupby channel
topl_mu = topl.groupby('channel').mean()
topl_err = topl.groupby('channel').std()

# Plot each channel
# The three channels are similar, except LR has less ~1 Hz and more ~100 Hz
f, ax = plt.subplots(figsize=(4.2, 3.5))
f.subplots_adjust(left=.2, bottom=.2, right=.8)

for channel in ['LV', 'RV', 'LR']:
    if channel == 'LV':
        color = 'b'
    elif channel == 'RV':
        color = 'r'
    elif channel == 'LR':
        color = 'k'

    line, = ax.plot(topl_mu.loc[channel], lw=1, label=channel, color=color)
    ax.fill_between(
        x=topl_mu.loc[channel].index,
        y1=topl_mu.loc[channel] - topl_err.loc[channel],
        y2=topl_mu.loc[channel] + topl_err.loc[channel],
        color=color, alpha=.5, lw=0)

# log x
ax.set_xscale('log')
ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('power spectral density\n(dB re 1 uV/Hz)')
ax.set_yticks((-40, -20, 0, 20, 40))

ax2 = ax.twinx()
ax2.set_yticks((-40, -20, 0, 20, 40))
ax2.set_yticklabels(('10 nV', '100 nV', '1 uV', '10 uV', '100 uV'))
ax.legend()

# Save figure
savename = 'PSD_BY_CHANNEL'
f.savefig(os.path.join(cohort_pickle_directory, savename + '.svg'))
f.savefig(os.path.join(cohort_pickle_directory, savename + '.png'), dpi=300)


## RMS histogram
# Histogram the noise levels
to_hist = np.log10(big_rms['hp'])
bins = np.linspace(-6.5, -4.5, 21)
binwidth = np.mean(np.diff(bins))
bincenters = (bins[:-1] + bins[1:]) / 2
edges_l = []
keys_l = []
for channel, to_hist2 in to_hist.groupby('channel'):
    counts, edges = np.histogram(to_hist2, bins=bins)
    edges_l.append(pandas.Series(counts, name=channel, index=bincenters))
    keys_l.append(channel)
all_counts = pandas.DataFrame(edges_l)
all_counts.index.name = 'channel'
all_counts.columns.name = 'log_rms'

f, ax = my.plot.figure_1x1_small()
for n_channel, channel in enumerate(['LV', 'RV', 'LR']):
    if channel == 'LV':
        color = 'b'
    elif channel == 'RV':
        color = 'r'
    elif channel == 'LR':
        color = 'k'

    ax.bar(
        all_counts.columns + n_channel / 3 * binwidth,
        all_counts.loc[channel],
        width=binwidth / 3 * .95, alpha=1, label=channel, edgecolor=color, facecolor=color)
ax.legend(fontsize='xx-small')
ax.set_xlim((-6.2, -4.7))
ax.set_xticks((-6, -5.5, -5))
ax.set_xticklabels(('1 uV', '3 uV', '10 uV'))
my.plot.despine(ax)
ax.set_ylabel('number of recordings')
ax.set_xlabel('rms(ABR band)')
f.savefig(os.path.join(cohort_pickle_directory, 'RMS_ABR_BAND.svg'))
f.savefig(os.path.join(cohort_pickle_directory, 'RMS_ABR_BAND.png'), dpi=300)

plt.figure()
seaborn.histplot(big_rms,x='hp',hue='channel',multiple='dodge',bins=bins)
## Store
big_Pxx.to_pickle(os.path.join(cohort_pickle_directory, 'big_Pxx'))
big_rms.to_pickle(os.path.join(data_dir, 'big_rms'))