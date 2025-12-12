# Need to have permission to access ACM0
# sudo adduser mouse dialout
#
# run this like:
#   python3 -m ABR2025.gui.start_cli
# or in ipython --pylab=qt:
#   run -m ABR2025.gui.start_cli

import numpy as np
import pandas
import ABR2025
import time
import os
import multiprocessing

# Initalize abr_device
abr_device = ABR2025.gui.ABR_Device.ABR_Device()

# Run (it will stop on CTRL+C)
abr_device.run_session()

# Load data
loaded_data = ABR2025.loading.load_recording(
    os.path.split(abr_device.tfw.output_filename)[0])

config = loaded_data['config']
header_df = loaded_data['header_df']
data = loaded_data['data']

if np.any(np.diff(header_df['packet_num_unwrapped']) != 1):
    print('warning: data is torn')
    