# ABR2025

## Introduction

This repository contains code to implement the ABR measuring system described
in Gargiullo et al, as well as the analysis scripts needed to replicate the 
figures in that paper.

The repository is structured as follows:
- `./` - The toplevel directory, containing this README file
- `./designs` - Design files, bill of materials, and other information about 
physically building the system.
- `./gui` - A Python module for running the graphical user interface (GUI) 
to take ABR data. It can be imported like `import ABR2025.gui`. This code is 
run on a desktop PC connected to the ABR measuring system hardware.
- `./scripts` - A folder of individual scripts that generate the figures in the
paper, using data that we collected and have shared on Zenodo. These scripts 
are meant to be run one at a time, not imported and called by other scripts. 

## Requirements

The following Python modules are required to run the GUI or analysis scripts. 
Other versions will likely work, but these are the ones we used. 

- matplotlib==3.10.0
- numpy==2.0.1
- pandas==2.3.1
- pyqt==5.15.10
- pyqt5-sip==12.13.0
- pyqtgraph==0.13.7
- scipy==1.16.0
- seaborn==0.13.2
- pyserial==3.5
- tqdm==4.67.1

For the analysis scripts (but not the GUI), the following repository is 
required. It should be on your PYTHONPATH so that it can be imported using
the syntax `import my`.
- git clone git@github.com:Rodgers-PAC-Lab/my.git

## Instructions

To build the hardware, refer to the file `./designs/README`. 

To run the GUI and measure ABR, refer to the file `./gui/README`.

To run the analysis scripts and regenerate the figures included in the paper,
refer to the file `./scripts/README`.
