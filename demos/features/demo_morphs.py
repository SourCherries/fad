from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.util import montage

import fad as fd


# HERE = Path(__file__).resolve().parent
HERE = Path("/Users/z10722/Tools/fad/demos/features")
FRL_folder = HERE / "FRL-LS"
stim_folder = HERE / "stimuli"
fig_folder = HERE/ "figs"
ID = ["122_03", "014_03"]  # 124_03

# Perform morph on 2 faces from FRL
num_morphs = 5
bookends = ("","jpg")
FRL = fd.Ensemble(dir_source = FRL_folder, file_bookends = bookends, INTER_PUPILLARY_DISTANCE = 64)
FRL.list_faces()
FRL.add_to_roster(ID[0])
FRL.add_to_roster(ID[1])
results = FRL.roster_morph_between(num_morphs=5)
morph_path = results["morph_path"]

# Wavelet features of morphs
FRL = fd.Ensemble(dir_source = Path(morph_path), file_bookends = ("N","png"), make_windowed_faces=True, INTER_PUPILLARY_DISTANCE = 64)
for i in range(num_morphs):
    FRL.add_to_roster("N" + str(i))
# FRL.add_all_to_roster()

FRL.combine_roster_features()
FRL.clip_roster_margins(margins=(1/6, 1/4))
FRL.display_roster()

# Make figure of morphs (WRFF)
num_morphs = len(FRL.Roster)
all_images = np.zeros((num_morphs*2, FRL.SHAPE[0], FRL.SHAPE[1], 4))
for i in range(num_morphs):
    source = FRL.Roster[i].img    
    all_images[i,:,:,:] = fd.align.make_four_channel_image(source, FRL.aperture)
for i in range(num_morphs):
    wrff = FRL.Roster[i].F
    all_images[i + num_morphs,:,:,:] = fd.align.make_four_channel_image(wrff, FRL.aperture)
im_montage = montage(all_images, grid_shape=(2, num_morphs), channel_axis=3).astype(np.uint8)


fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
fig.set_size_inches(w=2*num_morphs, h=2)
ax.imshow(im_montage, interpolation='nearest')

ax.set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
plt.margins(0,0)
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())
plt.savefig("figure-demos-features-morph.png", bbox_inches = 'tight',
            pad_inches = 0)
plt.show()

# End
# -------------------------------------------------------------------