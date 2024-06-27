from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave

import fad as fd
from fad import align as af


category = "friends"

# Helper functions --------------------------------------------------
def slim_fig(ax):
    ax.set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    return None

def get_row(face):
    """Concatenate windowed source and combined features side by side"""
    id = face.name.split(".")[0]
    B = face.F
    B = np.tile(B[:,:,np.newaxis], (1,1,3))
    file = aperture_path + id + ".png"
    img = Image.open(file)
    A = np.array(img)
    B = np.concatenate((B, A[:,:,3,][:,:,np.newaxis]), axis=2)
    row = np.concatenate((A, B), axis=1)
    return row

# Align/Window/Features ---------------------------------------------
HERE = Path(__file__).resolve().parent
demo_folder = HERE / category
fig_folder = HERE/ "demo_figs"
if not fig_folder.exists():
    fig_folder.mkdir(parents=False, exist_ok=False)
bookends = ("","jpg")
famous = fd.Ensemble(dir_source = demo_folder, file_bookends = bookends, INTER_PUPILLARY_DISTANCE = 120)  # max 120 px
ID = [f.split(".")[0] for f in famous.get_face_list()]
for name in ID:
    famous.add_to_roster(name)
famous.combine_roster_features()
famous.display_roster(include="both")
aligned_path = HERE / (category + "-aligned")
the_aperture, aperture_path = af.place_aperture(aligned_path, bookends[0],
                                                bookends[1],
                                                aperture_type="MossEgg",
                                                contrast_norm="max",
                                                color_of_result="rgb")

# Montage figure ----------------------------------------------------
strip_percent = 0.20
face = famous.Roster[0]
A = get_row(face)
face = famous.Roster[1]
B = get_row(face)
r, c, _ = A.shape
strip = np.zeros((r, round(c*strip_percent), 4), dtype=np.uint8)
R1 = np.concatenate((A,strip,B), axis=1)

face = famous.Roster[2]
C = get_row(face)
face = famous.Roster[3]
D = get_row(face)
R2 = np.concatenate((C,strip,D), axis=1)

montage = np.concatenate((R1, R2), axis=0)
out_file = "collage-" + category + ".png"
imsave(out_file, montage, check_contrast=False)

# Labeled version of figure -----------------------------------------
h, w, c = A.shape
w += strip.shape[1]

fig_height_in = 6
img_frac_x = 0.10
img_frac_y = 0.15
wh = montage.shape[1] / montage.shape[0]

fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
fig.set_size_inches(w=wh*fig_height_in, h=fig_height_in)
plt.imshow(montage)
slim_fig(ax)

labels = ["a", "b", "c", "d"]
px = [img_frac_x * w, img_frac_x * w + w] * 2
py = [img_frac_y * h, img_frac_y * h,
      img_frac_y * h + h*1, img_frac_y * h + h*1]
for L, X, Y in zip(labels, px, py):
    ax.text(X, Y, L, fontsize=36, color="k")
out_file = "collage-" + category + "-labeled.png"
plt.savefig(out_file, bbox_inches = "tight", pad_inches = 0)
plt.show()