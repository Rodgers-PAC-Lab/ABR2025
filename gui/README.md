# ABR2025 - graphical user interface (GUI)

This module runs the graphical user interface (GUI) for running an ABR
experiment. 

## Installation and requirements

The GUI was only designed for, tested on, and used in Linux. We use Linux Mint.
Other Linux distributions are somewhat likely to work. Some rewriting would be
required to run the GUI on Windows or Mac.

The installation steps are documented in the repository-wide README, one
level up from this directory.

Before running the GUI, you should:
- Build the hardware described in `../designs`
- Flash the provided `ino` file on the Teensy (see `../designs/README`)
- Connect the ABR measuring system's USB port to your computer
- Ensure that the ABR measuring system is showing up at `/dev/ttyACM0`, a path
which is hardcoded.

Presently the GUI assumes that the audio input is on the last channel of the
ADS1299, and that channels 0, 2, and 4 contain neural data.

## Starting the GUI

To start the GUI, activate whichever environment you used to install ABR2025, 
and type the following:
- python3 -m ABR2025.gui.start_gui

We include also a bash script, `start_gui.sh`, which demonstrates how to
save error messages to a logfile. This bash script will not work out of the
box for you because it references paths that only work in our lab, but you 
may be interested in modifying it for your own usage.

## Running the GUI

If all is working as it should, all you need to do is click "Start Recording"
to begin the experiment. 

The averaged ABR will only be computed and displayed accurately if you are using
click stimuli of the same amplitude that we used. However, data should be 
recorded and stored properly even if other stimuli are used.