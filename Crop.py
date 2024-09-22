# Description: 
# Code for cropping interferometry data for DPP 2024 poster. This code
# will also include a filter that flags bad data (i.e., black images) by
# intensity threshold.

# ---------- Imports ----------

import numpy as np
import PIL
import os
import matplotlib.pyplot as plt

# ---------- File Paths ----------

source = '../Interferometry Data (Raw)/8_23_2024/' # Directory containing raw images
target = 'Cropped Data/' # Target directory to deposit cropped images
shot_info = target + 'Shot Info.txt' # Text file containing timing and assoicated shot numbers

# ---------- Functions ----------
