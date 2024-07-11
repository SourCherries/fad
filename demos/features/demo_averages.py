from pathlib import Path
import os
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from skimage.util import montage
from skimage.transform import rescale
from PIL import Image as PilImage

import fad as fd
from fad import align as af


# Helper functions --------------------------------------------------
def slim_fig(ax):
    ax.set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    return None

def windowed_montage_with_average(dir, files, shape):
    n = len(files)
    rows, cols = shape
    channels = 4
    average_gray = np.zeros(shape)
    images = np.zeros((n, rows, cols, channels), dtype=np.uint8)
    for i, file in enumerate(files):
        infile = dir / (file.split(".")[0] + ".png")
        img = np.array(PilImage.open(infile))
        average_gray += img[:,:,0]
        images[i,:,:,:] = img
    average_gray /= n
    grid = (ceil(n/5), 5)
    im_montage = montage(images, rescale_intensity=True, grid_shape=grid, channel_axis=3)
    inner_map = img[:,:,3] > 16
    average = af.contrast_stretch(average_gray, 
                                inner_locs=inner_map, 
                                type="mean_127")
    average_double = rescale(average.astype(float), scale=2).astype(np.uint8)
    ap_double = rescale(img[:,:,3], scale=2)
    BGRA = af.make_four_channel_image(img=average_double, aperture=ap_double)
    return np.concatenate((im_montage, BGRA), axis=1)

# -------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
mother_ship = HERE / "faces-of-merkel"
bookends = ("merkel-","jpg")

# Originals (aligned and windowed) ----------------------------------
Merk = fd.Ensemble(dir_source=mother_ship,
                   file_bookends=bookends, 
                   INTER_PUPILLARY_DISTANCE=64, 
                   make_windowed_faces=True)

mdir = Merk.dir_windowed
mfiles = Merk.get_face_list()
mshape = Merk.SHAPE
montage_source = windowed_montage_with_average(mdir, mfiles, mshape)

# Faces warped to mean shape ----------------------------------------
af.warp_to_mean_landmarks(str(Merk.dir_aligned) + os.sep,
                          file_prefix=bookends[0],
                          file_postfix=bookends[1])

# Warped (aligned and windowed) -------------------------------------
mother_ship = Path(str(Merk.dir_aligned) + "-warp-to-mean")
Merk = fd.Ensemble(dir_source=mother_ship,
                   file_bookends=("N","png"), 
                   INTER_PUPILLARY_DISTANCE=64, 
                   make_windowed_faces=True)

mdir = Merk.dir_windowed
mfiles = Merk.get_face_list()
mshape = Merk.SHAPE
montage_warped = windowed_montage_with_average(mdir, mfiles, mshape)


# Put all together as single figure ---------------------------------
cols = montage_source.shape[1]
top_strip_v = np.zeros((30,cols,3), dtype=np.uint8)
top_strip_a = np.zeros((30,cols,1), dtype=np.uint8)
top_strip = np.concatenate((top_strip_v, top_strip_a), axis=2)
A = np.concatenate((top_strip, montage_source), axis=0)
B = np.concatenate((top_strip, montage_warped), axis=0)
montage_all = np.concatenate((A, B), axis=0)

fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
plt.imshow(montage_all, cmap="gray")
slim_fig(ax)
ax.text(30, 80, "A", fontsize=20, color="k")
ax.text(30 + Merk.SHAPE[1]*5, 80, "B", fontsize=20, color="k")
ax.text(30, 80 + Merk.SHAPE[0]*2+30, "C", fontsize=20, color="k")
ax.text(30 + Merk.SHAPE[1]*5, 80 + Merk.SHAPE[0]*2+30, "D", fontsize=20, color="k")
plt.savefig("figure-demos-features-enhanced-average.png", bbox_inches = 'tight',
            pad_inches = 0)
plt.show()

# End
# -------------------------------------------------------------------