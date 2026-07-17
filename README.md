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
- `./src` - source code for analysis and GUI. Import like `import opensabr`
- `./src/gui` - A Python module for running the graphical user interface (GUI) 
to take ABR data. It can be imported like `import opensabr.gui`. This code is 
run on a desktop PC connected to the ABR measuring system hardware.
- `./scripts` - A folder of demo scripts that generate the figures in the
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


## Installing OpenSABR
# Clone repo
cd ~/installed # or wherever desired
git clone git@github.com:Rodgers-PAC-Lab/OpenSABR # TODO: update URL
cd OpenSABR

# --- Install: pick ONE of the two options below ---

# Option A: Typical install (no version pins)
conda create -n opensabr -c conda-forge python=3.11 pip ipython
conda activate opensabr
conda install -c conda-forge numpy scipy pandas matplotlib tqdm pyqt pyqtgraph pyserial pyarrow seaborn
pip install -e . --no-deps

# Option B: Reproducible install (linux-64 only)
conda create -n opensabr --file opensabr-conda-lock.txt
conda activate opensabr
pip install -e . --no-deps


## Installing ABRpresto into its own environment
# pyarrow needed to read OpenSABR's parquet output; tqdm/ipython for convenience
cd ~/installed # or wherever
git clone git@github.com:Rodgers-PAC-Lab/ABRpresto # our fork
cd ABRpresto
git checkout dev # our branch with fixes
conda create -n abrpresto -c conda-forge python=3.11 ipython pip 
conda activate abrpresto
conda install -c conda-forge numpy scipy matplotlib setuptools_scm pandas pyarrow tqdm
pip install -e . --no-deps


## Installing ABRA
cd ~/dev 
git clone git@github.com:Rodgers-PAC-Lab/abranalysis # our fork
cd abranalysis
git checkout dev # our branch with fixes
conda create -n abra -c conda-forge python=3.11 ipython pip 
conda activate abra
conda install -c conda-forge numpy scipy pandas pyarrow scikit-learn pytorch=*=cpu* easydict streamlit tqdm "altair<5" keras tensorflow matplotlib

# Put abranalysis on PYTHONPATH (it cannot be pip-installed)
conda env config vars set PYTHONPATH=$HOME/dev
conda deactivate && conda activate abra   # reload so the var takes effect