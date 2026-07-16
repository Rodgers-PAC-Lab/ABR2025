# ABR2025

## Introduction

This repository contains code to implement the ABR measuring system described
in Gargiullo et al, as well as the analysis scripts needed to replicate the 
figures in that paper.
Preprint: https://www.biorxiv.org/content/10.64898/2025.12.17.694771v1

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

We used Python 3.11. Newer versions may also work but we did not test them.

requirements-lock.txt includes the exact versions we used

For the analysis scripts (but not the GUI), the following repository is 
required. It should be on your PYTHONPATH so that it can be imported using
the syntax `import my`.
- `git clone git@github.com:Rodgers-PAC-Lab/my.git`

TODO: only my.plot is needed, bring minimal into this repo

## Instructions

To build the hardware, refer to the file `./designs/README`. 

To run the GUI and measure ABR, refer to the file `./gui/README`.

To run the analysis scripts and regenerate the figures included in the paper,
refer to the file `./scripts/README`.


## Installation (updated)

conda create -n opensabr python=3.11
conda activate opensabr
git clone ... # TODO: insert git repo URL here
cd OpenSABR
pip install -e .          # or pip install -e ".[gui]"

For reproducibility (exact versions we used):
pip install -r requirements-lock.txt   # exact dep versions
pip install -e . --no-deps             # this package, deps already satisfied
