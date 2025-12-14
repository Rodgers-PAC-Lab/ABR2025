# ABR2025 - analysis scripts

## Introduction

The Python files in this folder will generate the figures presented in the
Gargiullo 2025 paper. 

## Getting started

### Downloading the data

First, download the data that we collected using the ABR measuring system. The
data can be downloaded from Zenodo at this link (TODO: add link)

Unzip the downloaded file, which will create a directory called "ABR2025_data".
Take note of the location of this directory - you will use it in the next step.

### Specifying filepaths

Next, you will need to create a file telling the scripts where to load the
data from, and where to save intermediate data files and figures. To do this,
create a file called `filepaths.json` in the same directory as this README
file, which is also where all of the analysis scripts are. 

The `filepaths.json` file should contain JSON text like the following:
'''
{
  "raw_data_directory": "/path_to_data_you_downloaded/ABR2025_data",
  "output_directory": "/path_to_save_output_files"
}
'''

Replace the paths with locations that work on your computer.
- `raw_data_directory` should be the path to the data that you downloaded and
unzipped in the previous step. It should end in "ABR2025_data", unless you
renamed the folder after unzipping it.
- `output_directory` should be the location where you want intermediate files
and figures to be saved. 

Note that `filepaths.json` has been added to `.gitignore`, since
the same paths are unlikely to work on a different computer.

## Description of the downloaded data

The raw data directory that you download and unzipped contains:
- A folder called `metadata` that contains CSV files about the data
- Many other directories named by date, which contain recorded ABR data

The metadata directory contains the following files:
- mouse_metadata.csv - A list of all mice
- experiment_metadata.csv - A list of all experiments (i.e., sessions)
- recording_metadata.csv - A list of all recordings

An "experiment" is a set of "recordings" that were made on one mouse on one
date. The speaker side can differ across recordings. Parameters like 
channel configuration are the same across all recordings, but they can differ
across experiments.

Each mouse can have multiple experiments, and each experiment can have 
multiple recordings. All recordings within the same experiment were taken
right after each other on the same day. 

The mouse_metadata csv file has the following columns:
- DOB: the mouse's date of birth
- HL_date: the date of the hearing loss surgery, or blank if no surgery
- HL_type: the string 'none', 'sham', or 'bilateral', indicating the type of
surgery
- genotype: the string 'WT' or '5xFAD (-/-)'. WT mice had two wildtype parents.
5xFAD (-/-) mice had one WT parent and one parent carrying the 5xFAD alleles 
(JAX 008730). However, we only included mice in the dataset that did not
inherit the 5xFAD alleles, so effectively they are all wildtype.
- mouse: string indicating the mouse name
- sex: the string 'M' or 'F' indicating the mouse's sex

The experiment_metadata csv file has the following columns
- after_HL: whether the experiment took place after the hearing loss surgery 
(True or False). If this mouse never had a hearing loss surgery, this value 
will always be False.
- age: age of the mouse in days at the time of the experiment
- date: date of the experiment
- experimenter: name of the person who did the experiment
- mouse: the mouse's name (matching the 'mouse' column in mouse_metadata.csv)
- n_experiment: an integer 0 or 1 indicating whether this was the first or 
second recording. The count restarts after hearing loss surgery, so a mouse
that received hearing loss surgery will have two experiment labeled "0": the 
first one ever (for which after_HL is False), and the first one after hearing 
loss (for which after_HL is True).

The recording_metadata csv file has the following columns
- date: date of the experiment (matches experiment_metadata.csv)
- mouse: the name of the mouse (matches experiment_metadata.csv and mouse_metadata.csv)
- recording: an integer indicating the recording number. No two recordings can
have the same tuple (date, mouse, recording).
- ch0_config: the string 'LV', 'RV', 'LR' indicating which differential pair
was connected to channel 0 in the ADS1299
- ch2_config, ch4_config: similar
- recording_name: a string of the integer 'recording' beginning with two zeros
- short_datafile: date, experimenter, and recording_name joined with '/'
- speaker_side: the string 'L' or 'R' indicating which speaker was playing sound

Each data directory is named by the date (YYYY-MM-DD) matching the date column
in experiment_metadata. Within the data directory is one or more subdirectories
named for the experimenter (matching the 'experimenter' column of 
experiment_metadata). Within the experimenter subdirecoty is one or more 
subdirectories named for the recording_name column of recording_metadata. For
example, the recording that took place on 2025-02-12 and was conducted by 
experimenter 'rowan' and was recording named "006" will be located at 
the path '2025-02-12/rowan/006' within the raw data directory. The column
"short_datafile" within recording_metadata specifies this relative path. 

Within each recording subdirectory will be three files. The best way to load
the data is to use the function `ABR2025.loading.load_recording`.
- data.bin : A binary data file of dtype int32 including 8 channels of data. 
The samples are stored in the following order "t0c0,t0c1,...,t0c7,t1c0,t1c1,...",
that is, the channel dimension iterates faster than the timepoint dimension.
- config.json : A JSON file of configuration information, such as the start 
time of the recording, and other parameters that can be ignroed because they
never change. 
- packet_headers.csv : A CSV file with one row per received packet. This can
be ignored because it includes only debugging information. 

## Running the analysis scripts

Running the scripts in this directory will reproduce all of the plots in the
paper. The scripts should be run in the specified (alphabetical) order, 
because some of them generate outputs that are needed by later ones. You 
should be able to run them just by calling `python3 script_name.py`, or 
however else you run Python scripts.

Note that there is not "Step1" for you to run, because that step is the 
one we used to compile all the data to share.

Each of these scripts will use the `filepaths.json` file that you created in the
previous step, in order for it to know where to load data from and write output
to. The figures will appear on the screen and they will also be saved to
SVG and PNG files in the output directory.

- Step2a1_align.py : Loads all binary data, slices it on click times, and writes
out the `big_triggered_ad`, `big_triggered_neural`, `big_click_params` used
by many downstream scripts.
- Step2a2_PSD.py : Analyzes power spectral density
- Step2a3_raw.py : Example plot of raw data
- Step2b_avg.py : Averages the ABRs over trials and writes out `big_abrs` and 
avearaged ABRs used by many downstream scripts.
- Step2c_thresh.py : Analyzes response strength and computes threshold
- Step2d_EKG.py : Analyzes the electrocardiogram
- Step3a_avgplot.py : Plots the grand average
- Step3b_corr.py : Correlates ABR across mice
- Step4a_peaks.py : Picks the primary peak
