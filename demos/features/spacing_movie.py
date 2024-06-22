from pathlib import Path
import json
import subprocess
import os

import numpy as np
from skimage.io import imsave

import fad as fd


HERE = Path(__file__).resolve().parent
FRLFaceFolder = HERE / "FRL-LS"
bookends = ("","jpg")
ID = "122_03"
margins = (.05, .1)

CreativeCommons = fd.Ensemble(dir_source = FRLFaceFolder, file_bookends = bookends, INTER_PUPILLARY_DISTANCE = 64)
CreativeCommons.add_to_roster(ID)
CreativeCommons.clip_roster_margins(margins=margins)
SHAPE = CreativeCommons.Roster[0].SHAPE

scaling_min, scaling_max = .5, 2
scales_half = np.arange(scaling_min,scaling_max,step=.1)
scales = np.r_[scales_half, np.flip(scales_half[1:-1])]
scale_shape = scales.max()
shape_out = (round(SHAPE[0]*scale_shape), round(SHAPE[1]*scale_shape))
movie = np.empty((shape_out[0], shape_out[1], len(scales)), dtype="uint8")

for i, s in enumerate(scales):
    CreativeCommons.empty_roster()
    CreativeCommons.add_to_roster(ID)
    CreativeCommons.clip_roster_margins(margins=margins)
    CreativeCommons.roster_space_out(scale=s, scale_shape=scale_shape)
    CreativeCommons.combine_roster_features()
    movie[:,:,i] = CreativeCommons.Roster[0].F
    
# Write frames to temporary folder
temp_folder = HERE / "temp"
if not temp_folder.exists():
    temp_folder.mkdir(parents=False, exist_ok=False)
for i in range(len(scales)):
    file_name = "image-" + str(i) + ".png"
    imsave(temp_folder / file_name, movie[:,:,i], check_contrast=False)

# Compose GIF from frames
out_file = HERE / "spacing-movie.gif"
commandFull = "ffmpeg -i " + str(temp_folder) + os.sep + "image-%d.png " + str(out_file)
retCode = subprocess.call(commandFull, shell=True)

# Clear out and remove temporary folder
commandClear = "rm " + str(temp_folder) + os.sep + "image-*.png"
retCode = subprocess.call(commandClear, shell=True)
commandRemove = "rmdir " + str(temp_folder)
retCode = subprocess.call(commandRemove, shell=True)


# End
# -------------------------------------------------------------------