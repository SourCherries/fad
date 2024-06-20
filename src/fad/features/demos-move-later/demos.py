from pathlib import Path
import copy

import matplotlib.pyplot as plt
import numpy as np

from wrff import Ensemble
from shift_features import (rotate_180, 
                            center_index,
                            tile_placement,
                            max_stretch_original_0_is_127)


# Basic usage
bookends = ("","jpg")
CreativeCommons = Ensemble(dir_source = Path.cwd() / "faces", file_bookends = bookends)
CreativeCommons.list_faces()
CreativeCommons.add_to_roster("corey-haim")
CreativeCommons.add_to_roster("eric-mccormack")
CreativeCommons.clip_roster_margins(margins=(1/6, 1/4))
CreativeCommons.display_roster()
CreativeCommonsOriginal = copy.deepcopy(CreativeCommons)

# # Needed for advanced demo
# name, img, ROI, F, SHAPE = [], [], [], [], []
# for face in CreativeCommons.Roster:
#     name.append(face.name)
#     img.append(face.img)
#     ROI.append(face.ROI)
#     F.append(face.F)
#     SHAPE.append(face.SHAPE)


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
row1 = np.c_[rotate_180(Face0_Original), rotate_180(Face0_Thatcher)]
montage = np.r_[row0, row1]
plt.imshow(montage, cmap="gray"); plt.show()

# Chimeric face -----------------------------------------------------
CreativeCommons.empty_roster()
CreativeCommons.add_to_roster("corey-haim")
CreativeCommons.add_to_roster("eric-mccormack")
CreativeCommons.clip_roster_margins(margins=(1/6, 1/4))

feature_id = {"left_eyebrow": 0,
              "right_eyebrow": 0,
              "left_eye": 0,
              "right_eye": 0,
              "nose": 1,
              "mouth_outline": 0}

chimera = CreativeCommons.roster_chimeric(feature_id)["img"]
CreativeCommons.combine_roster_features()
originals = []
for face in CreativeCommons.Roster:
    originals.append(face.F)
montage = np.c_[originals[0], originals[1], chimera]
plt.imshow(montage, cmap="gray"); plt.show()


# Spaced out features -----------------------------------------------
CreativeCommons.empty_roster()
CreativeCommons.add_to_roster("corey-haim")
CreativeCommons.add_to_roster("eric-mccormack")
CreativeCommons.clip_roster_margins(margins=(1/6, 1/4))
CreativeCommons.roster_space_out(scale=1.8)
CreativeCommons.display_roster(include="features")

# Double face illusion ----------------------------------------------
CreativeCommons.empty_roster()
CreativeCommons.add_to_roster("corey-haim")
CreativeCommons.add_to_roster("eric-mccormack")
CreativeCommons.clip_roster_margins(margins=(1/6, 1/4))
CreativeCommons.roster_double_face()
CreativeCommons.display_roster(include="features")


# *******************************************************************
# Face montage demo (advanced)
# *******************************************************************
CreativeCommons.empty_roster()
CreativeCommons.add_to_roster("corey-haim")
CreativeCommons.add_to_roster("eric-mccormack")
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
icy = int(center_index(SHAPE[0][0]))
icx = int(center_index(SHAPE[0][1]))
cx = round(pixel_cols/2)
for row in range(2):
    cy = icy + SHAPE[0][0]*row
    for f in range(CreativeCommons.NUM_FEATURES):
        roof += tile_placement(CF[:,:,f], shape_out, (cx, cy))

plt.imshow(max_stretch_original_0_is_127(roof), cmap="gray"); plt.show() # image_as_uint8()


# 🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧
# import alignfaces as afa
# import json
# import os


# ens = Ensemble(dir_source = Path.cwd() / "faces", file_bookends = bookends)
# pre, post = "", "jpg"

# source
# def prepare_source(ens, pre=None, post=None):
#     source_error = "Unknown"
#     if not ens.dir_source.exists():
#         source_error = "Source folder does not exist."
#     else:
#         source_files = afa.make_files.get_source_files(ens.dir_source, pre, post)
#         if len(source_files) == 0:
#             source_error = "No files in source folder matching requested pattern."
#         else:
#             landmarks_file = ens.dir_source / "landmarks.txt"
#             if not landmarks_file.exists():
#                 print("Missing file: landmarks.txt")
#                 print("Get landmarks using AFA.")
#                 input_path = str(ens.dir_source) + os.sep
#                 afa.get_landmarks(input_path, file_prefix=pre, file_postfix=post, start_fresh=True)
#                 source_error = None
#             else:
#                 with open(ens.dir_source / "landmarks.txt") as json_data:
#                     landmarks_source = json.load(json_data)
#                     json_data.close()
#                 ens.file_extension = list(landmarks_source)[0].split(".")[1]
#                 if ens.file_extension != post:
#                     print("Image files with landmarks have different file extension than specified.")
#                     print("Get landmarks using AFA.")
#                     input_path = str(ens.dir_source) + os.sep
#                     afa.get_landmarks(input_path, file_prefix=pre, file_postfix=post, start_fresh=True)
#                     source_error = None
#                 else:
#                     files_rel = [str(Path(f).relative_to(ens.dir_source)) for f in source_files]
#                     files_rel_lm =  list(landmarks_source)
#                     if set(files_rel) != set(files_rel_lm):
#                         print("Image files with landmarks do not match list of files in source folder matching requested pattern.")
#                         print("Get landmarks using AFA.")
#                         input_path = str(ens.dir_source) + os.sep
#                         afa.get_landmarks(input_path, file_prefix=pre, file_postfix=post, start_fresh=True)                    
#                     source_error = None
#     return source_error

# source_error = prepare_source(ens, pre=pre, post=post)

# # aligned
# def prepare_aligned(ens, pre=None, post=None, ipd=64):
#     aligned_error = "Unknown"
#     source_files = afa.make_files.get_source_files(ens.dir_source, pre, post)
#     files_rel = [str(Path(f).relative_to(ens.dir_source)) for f in source_files]
#     if not ens.dir_aligned.exists():
#         print("Alignment folder does not exist. Run alignment")
#         adjust_size, size_value = "set_eye_distance", ipd
#         input_path = str(ens.dir_source) + os.sep
#         aligned_path = afa.align_procrustes(input_path, adjust_size=adjust_size, size_value=size_value)
#         afa.get_landmarks(aligned_path, file_prefix=pre, file_postfix=post)
#         aligned_error = None
#     else:
#         aligned_files = afa.make_files.get_source_files(ens.dir_aligned, pre, post)
#         if len(aligned_files) == 0:
#             aligned_error = "No files in alignment folder matching requested pattern."
#         else:
#             files_rel_aligned = [str(Path(f).relative_to(ens.dir_aligned)) for f in aligned_files]
#             if set(files_rel) != set(files_rel_aligned):  # files_rel from above "source" code
#                 aligned_error = "Mismatch between files in aligned and in source folders."
#             else:
#                 specs_file = ens.dir_aligned / "specs.csv"
#                 if not specs_file.exists():
#                     print("Alignment specification file (specs.csv) does not exist.")
#                     print("Re-run alignment.")
#                     adjust_size, size_value = "set_eye_distance", ipd
#                     input_path = str(ens.dir_source) + os.sep
#                     aligned_path = afa.align_procrustes(input_path, adjust_size=adjust_size, size_value=size_value)
#                     afa.get_landmarks(aligned_path, file_prefix=pre, file_postfix=post)
#                     aligned_error = None
#                 else:
#                     landmarks_file = ens.dir_aligned / "landmarks.txt"
#                     if not landmarks_file.exists():
#                         print("Get landmarks using AFA.")
#                         aligned_path = str(ens.dir_aligned) + os.sep
#                         afa.get_landmarks(aligned_path, file_prefix=pre, file_postfix=post)
#                         aligned_error = None           
#                     else:         
#                         aligned_error = None
#                         with open(ens.dir_aligned / "landmarks.txt") as json_data:
#                             landmarks_aligned = json.load(json_data)
#                             json_data.close()
#                         ens.file_extension = list(landmarks_aligned)[0].split(".")[1]
#                         if ens.file_extension != post:
#                             aligned_error = "Aligned image files with landmarks have different file extension than specified."
#                         files_rel_aligned_lm =  list(landmarks_aligned)
#                         if set(files_rel_aligned) != set(files_rel_aligned_lm):
#                             aligned_error = "Aligned image files with landmarks do not match list of files in aligned folder matching requested pattern."
#     return aligned_error

# aligned_error = prepare_aligned(ens, pre=pre, post=post, ipd=64)

# wrff
# from wrff import make_features
# make_features(ens, "richard-pryor.jpg")


# for f in list(ens.landmarks):
#     make_features(ens, f)

# def make_features(ens: Ensemble, file_name: str):
#     L = ens.landmarks[file_name]
#     results = get_feature_roi(ens.SHAPE, L)
#     ROI = results["DilatedBW"]
#     img = np.array(PilImage.open(ens.dir_aligned / file_name).convert("L"))    
#     EO = get_eo(img, ens.ffilters)
#     chosen_coefficients = select_coefficients_by_feature(EO, ROI, ens.CriterionQuantileAmp)
#     F = reconstruct(ens.sfilters, EO, chosen_coefficients, img_format="uint8")
#     ID = file_name.split(".")[0]
#     for i in range(6):
#         file_full = ens.dir_wrff / (ID + "-" + str(i+1) + ".png")
#         F[:,:,i] = np.array(PilImage.open(file_full).convert("L"))
# End
# -------------------------------------------------------------------