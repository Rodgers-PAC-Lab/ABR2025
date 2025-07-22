# ABR2025

## Introduction

This repository contains all the scripts you need to reproduce the figures
in Gargiullo et al. 

## Getting started

Do these steps before running these scripts.
- Download the data from XYZ. 
- Create a file called `filepaths.json` in the same directory as this README.
  This file should contain JSON text like the following:

'''
{
 "raw_data_directory": "/home/chris/mnt/cuttlefish/abr_data",
 "output_directory": "/home/chris/mnt/cuttlefish/data/20250720_abr_paper_data"
}
'''

Replace the paths with locations that work on your computer.
`raw_data_directory` should be the path to the data.
`output_directory` should be the location where you want intermediate files
and figures to go.

Note that this `filepaths.json` file has been added to .gitignore, since
the same paths are unlikely to work on a different computer.
