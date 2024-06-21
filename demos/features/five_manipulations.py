from pathlib import Path
import copy

import matplotlib.pyplot as plt
import numpy as np

import fad as fd
import fad.features.shift_features as shift


HERE = Path.cwd()
CCFaceFolder = HERE.parent / "align" / "1_basic" / "faces"
FRLFaceFolder = HERE / "FRL-LS"

# Basic usage
bookends = ("","jpg")
# CreativeCommons = fd.Ensemble(dir_source = CCFaceFolder, file_bookends = bookends)
CreativeCommons = fd.Ensemble(dir_source = CCFaceFolder, file_bookends = bookends, INTER_PUPILLARY_DISTANCE = 64)
CreativeCommons.list_faces()
CreativeCommons.add_to_roster("premium/men/corey-haim")
CreativeCommons.add_to_roster("premium/men/eric-mccormack")
CreativeCommons.clip_roster_margins(margins=(1/6, 1/4))
CreativeCommons.display_roster()
CreativeCommonsOriginal = copy.deepcopy(CreativeCommons)


# *******************************************************************
# Feature manipulation demos
# *******************************************************************

# Thatcher face -----------------------------------------------------
CreativeCommons.roster_thatcher()
CreativeCommons.combine_roster_features()
CreativeCommons.display_roster()

# Montage of factorial Thatcher-by-Inverted
#  -------------------------------------
# | upright-normal  | upright-Thatcher  |
# |-------------------------------------|
# | inverted-normal | inverted-Thatcher |
#  -------------------------------------
do_face = 0
CreativeCommonsOriginal.combine_roster_features()
Face0_Original = CreativeCommonsOriginal.Roster[do_face].F
Face0_Thatcher = CreativeCommons.Roster[do_face].F
row0 = np.c_[Face0_Original, Face0_Thatcher]
row1 = np.c_[shift.rotate_180(Face0_Original), shift.rotate_180(Face0_Thatcher)]
montage = np.r_[row0, row1]
plt.imshow(montage, cmap="gray"); plt.show()

# Chimeric face -----------------------------------------------------
bookends = ("","jpg")

FRL = fd.Ensemble(dir_source = FRLFaceFolder, file_bookends = bookends, INTER_PUPILLARY_DISTANCE = 64)
FRL.add_to_roster("014_03")
FRL.add_to_roster("122_03")
FRL.display_roster()
FRL.clip_roster_margins(margins=(1/6, 1/4))
FRL.display_roster()

feature_id = {"left_eyebrow": 1,
              "right_eyebrow": 1,
              "left_eye": 0,
              "right_eye": 0,
              "nose": 1,
              "mouth_outline": 1}

chimera = FRL.roster_chimeric(feature_id)["img"]
FRL.combine_roster_features()
originals = []
for face in FRL.Roster:
    originals.append(face.F)
montage = np.c_[originals[0], originals[1], chimera]
plt.imshow(montage, cmap="gray"); plt.show()


# Spaced out features -----------------------------------------------
CreativeCommons.empty_roster()
CreativeCommons.add_to_roster("premium/men/corey-haim")
CreativeCommons.add_to_roster("premium/men/eric-mccormack")
CreativeCommons.clip_roster_margins(margins=(1/6, 1/4))
CreativeCommons.roster_space_out(scale=1.8)
CreativeCommons.display_roster(include="features")

# Double face illusion ----------------------------------------------
CreativeCommons.empty_roster()
CreativeCommons.add_to_roster("premium/men/corey-haim")
CreativeCommons.add_to_roster("premium/men/eric-mccormack")
CreativeCommons.clip_roster_margins(margins=(1/6, 1/4))
CreativeCommons.roster_double_face()
CreativeCommons.display_roster(include="features")


# *******************************************************************
# Face montage demo (advanced)
# *******************************************************************
CreativeCommons.empty_roster()
CreativeCommons.add_to_roster("premium/men/corey-haim")
CreativeCommons.add_to_roster("premium/men/eric-mccormack")
CreativeCommons.clip_roster_margins(margins=(1/6, 1/4))

# Needed for advanced demo
name, img, ROI, F, SHAPE = [], [], [], [], []
for face in CreativeCommons.Roster:
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
    for f in range(CreativeCommons.NUM_FEATURES):
        roof += shift.tile_placement(CF[:,:,f], shape_out, (cx, cy))

plt.imshow(shift.max_stretch_original_0_is_127(roof), cmap="gray"); plt.show() # image_as_uint8()


# End
# -------------------------------------------------------------------