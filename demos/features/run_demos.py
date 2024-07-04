from pathlib import Path
import copy

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave

import fad as fd
import fad.features.shift_features as shift


# Write individual Thatcher faces X2    🚀
# Add composite  X2 -- halves swapping
# Add feature spacing X2  -- IPD

# 2 figures
#   raw - wrff - composite  - whole_part    - thatcher
#   raw - wrff - spacing    - crowding      - double_face
HERE = Path(__file__).resolve().parent
FRL_folder = HERE / "FRL-LS"
stim_folder = HERE / "stimuli"
fig_folder = HERE/ "figs"
if not stim_folder.exists():
    stim_folder.mkdir(parents=False, exist_ok=False)
if not fig_folder.exists():
    fig_folder.mkdir(parents=False, exist_ok=False)    

ID = ["122_03", "014_03"]  # 124_03

# Basic usage
bookends = ("","jpg")
FRL = fd.Ensemble(dir_source = FRL_folder, file_bookends = bookends, INTER_PUPILLARY_DISTANCE = 64)
FRL.list_faces()
FRL.add_to_roster(ID[0])
FRL.add_to_roster(ID[1])
FRL.clip_roster_margins(margins=(1/6, 1/4))
FRL.display_roster()

FRL_original = copy.deepcopy(FRL)
FRL_original.combine_roster_features()
for i, face in enumerate(FRL_original.Roster):
    file_raw = "raw-" + str(i) + ".png"
    file_feat = "features-" + str(i) + ".png"
    imsave(stim_folder / file_raw, face.img, check_contrast=False)
    imsave(stim_folder / file_feat, face.F, check_contrast=False)
    

# *******************************************************************
# Feature manipulation demos
# *******************************************************************

# Thatcher face -----------------------------------------------------
FRL.roster_thatcher()
FRL.combine_roster_features()
FRL.display_roster()

# Montage of factorial Thatcher-by-Inverted
#  -------------------------------------
# | upright-normal  | upright-Thatcher  |
# |-------------------------------------|
# | inverted-normal | inverted-Thatcher |
#  -------------------------------------
do_face = 0
Face0_Original = FRL_original.Roster[do_face].F
Face0_Thatcher = FRL.Roster[do_face].F
row0 = np.c_[Face0_Original, Face0_Thatcher]
row1 = np.c_[shift.rotate_180(Face0_Original), shift.rotate_180(Face0_Thatcher)]
montage = np.r_[row0, row1]
plt.figure(figsize = (6,6))
plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)
plt.imshow(montage, cmap="gray")
plt.savefig(fig_folder / "thatcher.png")
plt.show()

for i, face in enumerate(FRL.Roster):
    file_thatcher = "thatcher-" + str(i) + ".png"    
    imsave(stim_folder / file_thatcher, face.F, check_contrast=False)

# Chimeric face -----------------------------------------------------
bookends = ("","jpg")

FRL = fd.Ensemble(dir_source = FRL_folder, file_bookends = bookends, INTER_PUPILLARY_DISTANCE = 64)
FRL.add_to_roster(ID[0])
FRL.add_to_roster(ID[1])
FRL.display_roster()
FRL.clip_roster_margins(margins=(1/6, 1/4))
FRL.display_roster()

feature_id = {"left_eyebrow": 0,
              "right_eyebrow": 0,
              "left_eye": 1,
              "right_eye": 1,
              "nose": 0,
              "mouth_outline": 0}
chimera_a = FRL.roster_chimeric(feature_id)["img"]

feature_id = {"left_eyebrow": 1,
              "right_eyebrow": 1,
              "left_eye": 0,
              "right_eye": 0,
              "nose": 1,
              "mouth_outline": 1}
chimera_b = FRL.roster_chimeric(feature_id)["img"]

imsave(stim_folder / "whole-part-0.png", chimera_a, check_contrast=False)
imsave(stim_folder / "whole-part-1.png", chimera_b, check_contrast=False)

# FRL.combine_roster_features()
# originals = []
# for face in FRL.Roster:
#     originals.append(face.F)
# montage = np.c_[originals[0], originals[1], chimera]
# plt.imshow(montage, cmap="gray")
# plt.tick_params(left = False, right = False , labelleft = False , 
#                 labelbottom = False, bottom = False)
# plt.savefig(fig_folder / "chimera.png")
# plt.show()

# Spaced out features -----------------------------------------------
FRL.empty_roster()
FRL.add_to_roster(ID[0])
FRL.add_to_roster(ID[1])
FRL.clip_roster_margins(margins=(1/6, 1/4))
FRL.roster_space_out(scale=1.8)
FRL.clip_roster_margins(margins=(1/6, 1/4))
fh = FRL.display_roster(include="features", show=False, title=False)
for i, fig in enumerate(fh):
    fig.savefig(fig_folder / ("spaced-out-" + str(i) + ".png"))
    plt.show()

FRL.combine_roster_features()
for i, face in enumerate(FRL.Roster):
    file_feat = "exploded-" + str(i) + ".png"    
    imsave(stim_folder / file_feat, face.F, check_contrast=False)

# Double face illusion ----------------------------------------------
FRL.empty_roster()
FRL.add_to_roster(ID[0])
FRL.add_to_roster(ID[1])
FRL.clip_roster_margins(margins=(1/6, 1/4))
FRL.roster_double_face()
fh = FRL.display_roster(include="features", show=False, title=False)
for i, fig in enumerate(fh):
    fig.savefig(fig_folder / ("double-space-" + str(i) + ".png"))
    plt.show()

FRL.combine_roster_features()
for i, face in enumerate(FRL.Roster):
    file_feat = "double-" + str(i) + ".png"    
    imsave(stim_folder / file_feat, face.F, check_contrast=False)

# *******************************************************************
# Face montage demo (advanced)
# *******************************************************************
FRL.empty_roster()
FRL.add_to_roster(ID[0])
FRL.add_to_roster(ID[1])
FRL.clip_roster_margins(margins=(1/6, 1/4))

# Needed for advanced demo
name, img, ROI, F, SHAPE = [], [], [], [], []
for face in FRL.Roster:
    name.append(face.name)
    img.append(face.img)
    ROI.append(face.ROI)
    F.append(face.F)
    SHAPE.append(face.SHAPE)

# Fat face illusion -------------------------------------------------
CF = F[0].astype(float)
CF = CF - CF[0,0,0]
face_rows, face_cols = 2, 2.4
pixel_rows = round(face_rows * SHAPE[0][0])
pixel_cols = round(face_cols * SHAPE[0][1])
shape_out = (pixel_rows, pixel_cols)
roof = np.zeros(shape_out)
icy = int(shift.center_index(SHAPE[0][0]))
icx = int(shift.center_index(SHAPE[0][1]))
cx = round(pixel_cols/2)
for row in range(2):
    cy = icy + SHAPE[0][0]*row
    for f in range(FRL.NUM_FEATURES):
        roof += shift.tile_placement(CF[:,:,f], shape_out, (cx, cy))

plt.imshow(shift.max_stretch_original_0_is_127(roof), cmap="gray")
plt.savefig(fig_folder / "fat-face.png")
plt.show()

# End
# -------------------------------------------------------------------