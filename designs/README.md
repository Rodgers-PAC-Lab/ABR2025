# ABR2025 - hardware designs

## Introduction 

This folder contains the designs needed to build the hardware for the 
ABR measuring system described in Gargiullo 2025.

It contains the following subfolders:
- bill_of_materials.xlsx: Bill of Materials for all required parts
- interface_pcb/: PCB design files for the "interface PCB" that houses the
Teensy and connects to the other boards
- teensy/: Files for the code that runs on the Teensy

### Bill of Materials

This is an Excel file of all required components. You will also need standard
electronics equipment for soldering, etc.

### Interface PCB

This PCB was designed in Eagle. The original design files are located in the
`eagle` subdirectory and can be opened in the Eagle software. You only need to 
do this if you want to edit the PCB.

If you just want to print the PCB, use the files in the `gerbers` subdirectory.
These are the standard manufacturing format that all PCB printing companies use.
We used OSH Park to print the PCB.

After you receive the PCB, you will need to populate a few components. Details
are available in the "Supplemental Material", which also explains how to 
physically assemble all components.

### Teensy files

This is the software that runs on the Teensy. The software controls the 
ADS1299, and it also interfaces with the desktop computer running the GUI
(see `../gui/README`). Flash the files included here on the Teensy.