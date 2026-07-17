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


## Overall installation instructions
These instructions walk you through the installation of OpenSABR, our fork
of ABRA, and our fork of ABRpresto in three different conda environments.
If you only want to run OpenSABR, you can skip the other two environments --
they are only used to provide a point of comparison in the paper. 

We clone repositories that are installed using a method like `pip install`
into a directory called `~/dev/installed`. We clone repositories that are 
not installable, but need to be imported, into a directory called `~/dev`.
ABRA is the only one that must be cloned into `~/dev`.
This is because it can cause errors to mix these two kinds of repositories.
`~/dev` must be on the PYTHONPATH for this to work. 

For ABRA, we recommend the CPU packages, there is no need for GPU.

The three conda environments share data written to disk in the parquet format.


## Installing OpenSABR
# Clone repo
cd ~/installed
git clone git@github.com:Rodgers-PAC-Lab/OpenSABR
cd OpenSABR

# --- Install: pick ONE of Option A or Option B ---

# Option A: Typical install (no version pins)
conda create -n opensabr -c conda-forge python=3.11 pip ipython
conda activate opensabr
conda install -c conda-forge numpy scipy pandas matplotlib tqdm pyqt pyqtgraph pyserial pyarrow seaborn
pip install -e . --no-deps

# Option B: Reproducible install from lock (linux-64 only)
conda create -n opensabr --file opensabr-conda-lock.txt
conda activate opensabr
pip install -e . --no-deps


## Installing ABRpresto
# Clone repo
cd ~/installed 
git clone git@github.com:Rodgers-PAC-Lab/ABRpresto # our fork
cd ABRpresto
git checkout dev # our branch with fixes

# --- Install: pick ONE of Option A or Option B ---

# Option A: Typical install (no version pins)
conda create -n abrpresto -c conda-forge python=3.11 ipython pip
conda activate abrpresto
conda install -c conda-forge numpy scipy pandas matplotlib setuptools_scm pyarrow tqdm
pip install -e . --no-deps

# Option B: Reproducible install from lock (linux-64 only)
conda create -n abrpresto --file abrpresto-conda-lock.txt
conda activate abrpresto
pip install -e . --no-deps


## Installing ABRA
# Clone repo (must be under ~/dev or some location on PYTHONPATH)
cd ~/dev
git clone git@github.com:Rodgers-PAC-Lab/abranalysis # our fork
cd abranalysis
git checkout dev # our branch with fixes

# --- Install: pick ONE of Option A or Option B ---
# ABRA cannot be pip-installed; it is imported via PYTHONPATH in both options.

# Option A: Typical install (no version pins)
conda create -n abra -c conda-forge python=3.11 ipython pip
conda activate abra
conda install -c conda-forge numpy scipy pandas scikit-learn pytorch=*=cpu* keras tensorflow streamlit easydict "altair<5" pyarrow matplotlib tqdm
conda env config vars set PYTHONPATH=$HOME/dev
conda deactivate && conda activate abra   # reload so PYTHONPATH takes effect

# Option B: Reproducible install from lock (linux-64 only)
conda create -n abra --file abra-conda-lock.txt
conda activate abra
conda env config vars set PYTHONPATH=$HOME/dev
conda deactivate && conda activate abra   # reload so PYTHONPATH takes effect