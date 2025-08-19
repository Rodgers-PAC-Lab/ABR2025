# ABR2025

## Introduction

This repository contains all the scripts you need to reproduce the figures
in Gargiullo et al. 

## Requirements

Install the following dependencies into your environment:
- conda install numpy
- pip install ffmpeg-python

TODO: Add version numbers

Clone the following repositories in a location on your PYTHONPATH:
- git clone git@github.com:Rodgers-PAC-Lab/paclab.git

## Getting started

Do these steps before running these scripts.
- Download the data from XYZ. 
- Create a file called `filepaths.json` in the same directory as this README.
  This file should contain JSON text like the following (without the triple quotes):

'''
{
 "raw_data_directory": "/home/chris/mnt/cuttlefish/surgery/abr_data",
 "output_directory": "/home/chris/mnt/cuttlefish/chris/data/20250720_abr_paper_data"
}
'''

Replace the paths with locations that work on your computer.
`raw_data_directory` should be the path to the data.
`output_directory` should be the location where you want intermediate files
and figures to go.

Note that this `filepaths.json` file has been added to .gitignore, since
the same paths are unlikely to work on a different computer.

## Metadata

The subdirectory `./metadata/` contains two metadata CSV files. 

* experiment_metadata.csv - A list of all experiments
* mouse_metadata.csv - A list of all mice

An "experiment" is a set of "recordings" that were made on one mouse on one
date. The speaker side can differ across recordings. Parameters like 
channel configuration are the same across all recordings, but they can differ
across experiments.

The mouse metadata also includes the type of hearing loss (bilateral or sham)
and the date of hearing loss surgery, or null if no surgery was performed.

The data from each experiment is contained within a directory located at
RAW_DATA_DIRECTORY/DATE/EXPERIMENTER. 

* RAW_DATA_DIRECTORY : the string from `filepaths.json`
* DATE : YYYY-MM-DD format
* EXPERIMENTER : always the string "rowan"

Within that directory, a file ending in "_v6.csv" contains the metadata
for each recording, such as the speaker side and whether it should be included.

## Scripts

Running the scripts in this directory will reproduce all of the plots in the
paper. The scripts should be run in the specified order, because some of them
generate outputs that are needed by later ones. 

This google sheet keeps track of which scripts generate which figures:
https://docs.google.com/spreadsheets/d/1OOjzReItH3UJshRRVqz2gxoI22vuh4sFMQ0LG2jgTPE/edit?gid=0#gid=0

TODO: Make sure that all figures are written out to `./figures`
TODO: replace `from paclab import abr` with `import paclab.abr` throughout
TODO: remove any unnecessary module imports

### Step 1
This step is for loading metadata and choosing what data to include.

* Step1.py : Loads metadata for every recording and generates metadata pickles 
  used by the rest of the scripts. 

### Step 2

This step is for loading binary data, slicing it, and aggregating it.
TODO: Move Step2b and Step2c to Step 3 instead, so that this step only operates
on full ABR recordings, not on sliced recordings.

* Step2a1.py : Loads all binary data, slices it on click times, and writes
  out the `big_triggered_ad`, `big_triggered_neural`, `big_click_params` used
  by many downstream scripts.
* Step2a2.py : Loads all binary data and runs PSD analysis
* Step2b.py : Averages the ABRs over trials and writes out `big_abrs` and 
  `trial_counts`
* Step2c.py : Computes the rms(ABR) by sound level and calculates thresholds
* Step2d.py : EKG analysis

### Step 3

This step is for averaging the ABR across mice and correlating across mice.

* Step3a.py : Compute the grand average ABRs, and the delay vs level
* Step3b.py : Correlates ABR across mice

### Step 4

This step is for peak-picking.

* Step4a.py : Plots that involve peak picking


