"""Top-level OpenSABR module

Use as follows: import opensabr

The GUI is not imported by default because it has additional dependencies
that are not necessary just to run the demo analysis code.

If running the GUI: import opensabr.gui
"""

from . import signal_processing
from . import loading
from . import peak_picking