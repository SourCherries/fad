from pathlib import Path
import json
import subprocess
import os

import numpy as np
from PIL import Image as PilImage
import matplotlib.pyplot as plt
from skimage.io import imsave

from .reconstruct import (tile_placement,
                         max_stretch_original_0_is_127,
                         combine_features)
from .roi import get_feature_roi


ID = "corey-haim"  # dave-thomas, ron-perlman, william-zabka, corey-haim
SPACING_MOVIE = False


# Helper functions ----------------------------------------------

def center_index(length):
    if length % 2:
        c = (length-1)/2
    else:
        c = length/2
    return(c)

def move_p0_radius_along_path_to_p1(P0, P1, radius):
    """
    P0 center of background image (roof)
    P1 is center of feature (can define in different ways)
    PN is where feature image should be shifted to be in background
    Ensures distance (PN-P0) is radius
    """
    D = P1 - P0
    D = D / np.sqrt((D**2).sum())
    return(P0 + D*radius)

def rotate_180(img):
    return(np.flip(img, (0,1)))

def shift_xy_to_image_center(img, xy):
    d_r = (center_index(img.shape[0]) - xy[1]).round().astype(int)
    d_c = (center_index(img.shape[1]) - xy[0]).round().astype(int)
    img_centered = np.roll(img, (d_r, d_c), axis=(0,1))  # explicit axis in numpy.roll is critical
    return img_centered

def feature_center(ROI):
    floc = np.where(ROI)
    fr = floc[0]
    fc = floc[1]
    xy = np.array([fc.mean(), fr.mean()])    
    return xy

def clip_to_margins(F, ROI, margins=(1/6, 1/4)):
    SHAPE = F.shape[:2]
    vertical_extent = np.where(ROI.sum((1,2)))[0][[0,-1]]
    horizontal_extent = np.where(ROI.sum((0,2)))[0][[0,-1]]
    x0 = round(horizontal_extent[0] - np.diff(horizontal_extent)[0]*margins[0])
    x1 = round(horizontal_extent[1] + np.diff(horizontal_extent)[0]*margins[0])
    y0 = round(vertical_extent[0] - np.diff(vertical_extent)[0]*margins[1])
    y1 = round(vertical_extent[1] + np.diff(vertical_extent)[0]*margins[1])
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(SHAPE[1], x1)
    y1 = min(SHAPE[0], y1)
    F_ = F[y0:y1, x0:x1, :]
    ROI_ = ROI[y0:y1, x0:x1, :]
    SHAPE_ = F_.shape[:2]
    results = {"F": F_, "ROI": ROI_, "SHAPE": SHAPE_}
    return results

def space_out_features(F, ROI, scale, scale_shape=None):
    """
    Shift each feature away from center of original image,
    as a multiple of original distance.

    INPUT
        F               ndarray of wavelet features (rows,cols,features)
        ROI             ndarray of feature regions  (rows,cols,features)
        scale           spacing multiplier        
        scale_shape     final shape multiplier

    OUTPUT
        Fn              ndarray of wavelet features (rows,cols,features)        
    """
    assert F.shape==ROI.shape
    if scale_shape is None:
        scale_shape = scale
    assert scale_shape >= scale
    roof_rows, roof_cols, NUM_FEATURES = F.shape
    new_rows, new_cols = round(roof_rows*scale_shape), round(roof_cols*scale_shape)
    shape_out = (new_rows, new_cols)
    Fn = np.zeros((new_rows, new_cols, NUM_FEATURES))
    ROIn = np.zeros((new_rows, new_cols, NUM_FEATURES))
    for feature in range(NUM_FEATURES):
        tile = F[:,:,feature].astype(float)
        xy_o = feature_center(ROI[:,:,feature])
        tile_centered = shift_xy_to_image_center(tile, xy_o)
        tile_centered -= tile_centered[0,0]
        roof_c = np.array([center_index(roof_cols), center_index(roof_rows)])
        newf_c = np.array([center_index(new_cols), center_index(new_rows)])
        xy_n = ((xy_o - roof_c)*scale + newf_c).round().astype(int)
        Fn[:,:,feature] = tile_placement(tile_centered, shape_out, xy_n)

        roi_centered =  shift_xy_to_image_center(ROI[:,:,feature], xy_o)
        ROIn[:,:,feature] = tile_placement(roi_centered, shape_out, xy_n)
    return {"F": Fn, "ROI": ROIn}


def space_out_ipd(F_, ROI_, scale):
    """
    Shift each eyes and brows away from midpoint between eyes,
    as a multiple of original distance.

    INPUT
        F               ndarray of wavelet features (rows,cols,features)
        ROI             ndarray of feature regions  (rows,cols,features)
        scale           spacing multiplier

    OUTPUT
        Fn              ndarray of wavelet features (rows,cols,features)        
    """
    assert F_.shape == ROI_.shape
    assert type(scale) is float
    assert len(F_.shape)==3
    assert F_.shape[2]==6
    new_rows, new_cols, NUM_FEATURES = F_.shape
    shape_out = (F_.shape[0], F_.shape[1])
    Fn = np.zeros(F_.shape)
    ROIn = np.zeros(F_.shape)

    LB = feature_center(ROI_[:,:,0])
    LE = feature_center(ROI_[:,:,2])
    RB = feature_center(ROI_[:,:,1])
    RE = feature_center(ROI_[:,:,3])
    ipd = ((LE-RE)**2).sum()**(1/2)
    shift_eye_pixels = ipd*(scale-1)/2
    CE = (LE+RE)/2
    CB = (LB+RB)/2

    LE_ = (LE - CE)*((scale-1)/2 + 1) + CE
    RE_ = (RE - CE)*((scale-1)/2 + 1) + CE
    LB_ = (LB + (LE_-LE)).astype(int)
    RB_ = (RB + (RE_-RE)).astype(int)
    LE_ = LE_.astype(int)
    RE_ = RE_.astype(int)

    feature = 0
    tile = F_[:,:,feature].astype(float)
    tile_centered = shift_xy_to_image_center(tile, LB)
    tile_centered -= tile_centered[0,0]
    Fn[:,:,feature] = tile_placement(tile_centered, shape_out, LB_)
    roi_centered =  shift_xy_to_image_center(ROI_[:,:,feature], LB)
    ROIn[:,:,feature] = tile_placement(roi_centered, shape_out, LB_)

    feature = 1
    tile = F_[:,:,feature].astype(float)
    tile_centered = shift_xy_to_image_center(tile, RB)
    tile_centered -= tile_centered[0,0]
    Fn[:,:,feature] = tile_placement(tile_centered, shape_out, RB_)
    roi_centered =  shift_xy_to_image_center(ROI_[:,:,feature], RB)
    ROIn[:,:,feature] = tile_placement(roi_centered, shape_out, RB_)

    feature = 2
    tile = F_[:,:,feature].astype(float)
    tile_centered = shift_xy_to_image_center(tile, LE)
    tile_centered -= tile_centered[0,0]
    Fn[:,:,feature] = tile_placement(tile_centered, shape_out, LE_)
    roi_centered =  shift_xy_to_image_center(ROI_[:,:,feature], LE)
    ROIn[:,:,feature] = tile_placement(roi_centered, shape_out, LE_)

    feature = 3
    tile = F_[:,:,feature].astype(float)
    tile_centered = shift_xy_to_image_center(tile, RE)
    tile_centered -= tile_centered[0,0]
    Fn[:,:,feature] = tile_placement(tile_centered, shape_out, RE_)
    roi_centered =  shift_xy_to_image_center(ROI_[:,:,feature], RE)
    ROIn[:,:,feature] = tile_placement(roi_centered, shape_out, RE_)

    feature = 4
    tile = F_[:,:,feature].astype(float)
    corner_value = tile[0,0]
    Fn[:,:,feature] = tile
    ROIn[:,:,feature] = ROI_[:,:,feature]

    feature = 5
    tile = F_[:,:,feature].astype(float)
    corner_value = tile[0,0]
    Fn[:,:,feature] = tile
    ROIn[:,:,feature] = ROI_[:,:,feature]

    for feature in range(4):
        Fn[:,:,feature] += corner_value

    return {"F": Fn, "ROI": ROIn}


def thatcher_face(F, ROI, feature_index):
    """
    INPUT
        F               ndarray of wavelet features (rows,cols,features)
        ROI             ndarray of feature regions  (rows,cols,features)
        feature_index   dict of indices into F[:,:,i]

    OUTPUT
        Ft              ndarray of wavelet features (rows,cols,features)        
    """
    # Original eyes
    LEFT_EYE = feature_index["left_eye"]
    RIGHT_EYE = feature_index["right_eye"]
    left_eye, right_eye = F[:,:,LEFT_EYE], F[:,:,RIGHT_EYE]
    left_roi, right_roi = ROI[:,:,LEFT_EYE], ROI[:,:,RIGHT_EYE]

    # Invert left eye in place
    xy = feature_center(left_roi).round().astype(int)
    tile_centered = shift_xy_to_image_center(left_eye, xy).astype(float)
    background = tile_centered[0,0]
    tile_centered -= background
    tile_rotated = rotate_180(tile_centered)
    left_eye_rotated = tile_placement(tile_rotated, left_eye.shape, xy)
    left_eye_rotated += background
    left_eye_rotated = left_eye_rotated.astype(int)

    roi_centered = shift_xy_to_image_center(left_roi, xy).astype(float)
    roi_rotated = rotate_180(roi_centered)
    left_roi_rotated = tile_placement(roi_rotated, left_roi.shape, xy)

    # Invert right eye in place
    xy = feature_center(right_roi).round().astype(int)
    tile_centered = shift_xy_to_image_center(right_eye, xy).astype(float)
    background = tile_centered[0,0]
    tile_centered -= background
    tile_rotated = rotate_180(tile_centered)
    right_eye_rotated = tile_placement(tile_rotated, right_eye.shape, xy)
    right_eye_rotated += background
    right_eye_rotated = right_eye_rotated.astype(int)

    roi_centered = shift_xy_to_image_center(right_roi, xy).astype(float)
    roi_rotated = rotate_180(roi_centered)
    right_roi_rotated = tile_placement(roi_rotated, right_roi.shape, xy)

    # Thatcher face with feature channels
    Ft = np.copy(F)
    Ft[:,:,LEFT_EYE] = left_eye_rotated
    Ft[:,:,RIGHT_EYE] = right_eye_rotated
    ROIt = np.copy(ROI)
    ROIt[:,:,LEFT_EYE] = left_roi_rotated
    ROIt[:,:,RIGHT_EYE] = right_roi_rotated

    return {"F": Ft, "ROI": ROIt}


def double_face_illusion(F, ROI, feature_index, percent_down_shift):
    assert type(feature_index) is dict
    assert len(feature_index) == 6
    LEFT_BROW = feature_index["left_eyebrow"]
    RIGHT_BROW = feature_index["right_eyebrow"]
    LEFT_EYE = feature_index["left_eye"]
    RIGHT_EYE = feature_index["right_eye"]    
    MOUTH = feature_index["mouth_outline"]    
    SHAPE = F.shape[:2]
    icy = int(center_index(SHAPE[0]))
    icx = int(center_index(SHAPE[1]))
    brow_y = (feature_center(ROI[:,:,LEFT_BROW])[1] +
            feature_center(ROI[:,:,RIGHT_BROW])[1]) / 2
    eye_y = (feature_center(ROI[:,:,LEFT_EYE])[1] +
            feature_center(ROI[:,:,RIGHT_EYE])[1]) / 2
    dy = (eye_y - brow_y) * percent_down_shift
    CF = F.astype(float)
    CF = CF - CF[0,0,0]
    pxy = (icx, icy + round(dy/2) + 3)
    mxy = (icx, icy - round(dy/2) + 3)
    eyes_shift_down_l = tile_placement(CF[:,:,LEFT_EYE], SHAPE, pxy)
    eyes_shift_down_r = tile_placement(CF[:,:,RIGHT_EYE], SHAPE, pxy)
    mouth_shift_down = tile_placement(CF[:,:,MOUTH], SHAPE, pxy)
    Fextra = np.concatenate((CF, 
                            eyes_shift_down_l[:,:,np.newaxis],
                            eyes_shift_down_r[:,:,np.newaxis],
                            mouth_shift_down[:,:,np.newaxis]), axis=2)
    Fextra[:,:,LEFT_EYE] = tile_placement(CF[:,:,LEFT_EYE], SHAPE, mxy)
    Fextra[:,:,RIGHT_EYE] = tile_placement(CF[:,:,RIGHT_EYE], SHAPE, mxy)
    Fextra[:,:,MOUTH] = tile_placement(CF[:,:,MOUTH], SHAPE, mxy)

    # ROI
    eyes_shift_down_l = tile_placement(ROI[:,:,LEFT_EYE], SHAPE, pxy)
    eyes_shift_down_r = tile_placement(ROI[:,:,RIGHT_EYE], SHAPE, pxy)
    mouth_shift_down = tile_placement(ROI[:,:,MOUTH], SHAPE, pxy)
    Rextra = np.concatenate((ROI, 
                            eyes_shift_down_l[:,:,np.newaxis],
                            eyes_shift_down_r[:,:,np.newaxis],
                            mouth_shift_down[:,:,np.newaxis]), axis=2)
    Rextra[:,:,LEFT_EYE] = tile_placement(ROI[:,:,LEFT_EYE], SHAPE, mxy)
    Rextra[:,:,RIGHT_EYE] = tile_placement(ROI[:,:,RIGHT_EYE], SHAPE, mxy)
    Rextra[:,:,MOUTH] = tile_placement(ROI[:,:,MOUTH], SHAPE, mxy)

    feature_labels = list(feature_index)
    feature_labels.append("left_eye_lower")
    feature_labels.append("right_eye_lower")
    feature_labels.append("mouth_outline_lower")

    return {"F": Fextra, "ROI": Rextra, "feature_labels": feature_labels}

# # -------------------------------------------------------------------
# # Load data for image ensembles

# cwd = Path(__file__).resolve().parent
# aligned_faces = "faces-aligned"
# wrff_faces = aligned_faces + "-wrff"

# # Landmarks of aligned-image ensemble
# with open(cwd / aligned_faces / "landmarks.txt") as json_data:
#     landmarks = json.load(json_data)
#     json_data.close()
#     files = list(landmarks)

# # Constants of aligned-image ensemble
# csvfile = open(cwd / aligned_faces / "specs.csv", "r")
# headers = csvfile.readline()
# values = csvfile.readline()
# adjust_size, h, w, INTER_PUPILLARY_DISTANCE = [v for v in values.split(",")]
# h = int(h)
# w = int(w)
# INTER_PUPILLARY_DISTANCE = int(INTER_PUPILLARY_DISTANCE.split("\n")[0])
# SHAPE = (h, w)


# # -------------------------------------------------------------------
# # Load data for individual face (ID)

# # WRFF in F (rows,cols,features)
# R, C = SHAPE
# F = np.zeros((R,C,6), dtype=np.uint8)
# for i in range(6):
#     file_full = cwd / wrff_faces / (ID + "-" + str(i+1) + ".png")
#     F[:,:,i] = np.array(PilImage.open(file_full).convert("L"))

# # Region-of-interest maps for features
# person = landmarks[ID + ".jpg"]
# results = get_feature_roi(SHAPE, person)
# ROI = results["DilatedBW"]
# feature_labels = results["roi_features"]
# NUM_FEATURES = len(feature_labels)

# # Crop
# results = clip_to_margins(F, ROI, margins=(1/6, 1/4))
# F = results["F"]
# ROI = results["ROI"]
# SHAPE = F.shape[:2]
# plt.imshow(combine_features(F), cmap="gray"); plt.show()

# # 🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪
# # CHECK Place left brow in original place and compare
# # roof_cols, roof_rows = C, R
# # Roof = np.zeros((roof_rows, roof_cols))
# # Tile = F[:,:,0]
# # DestinationXY = (int(center_index(roof_cols)), int(center_index(roof_rows)))
# # TileOnFreshRoof = tile_placement(Tile, (roof_rows, roof_cols), DestinationXY)
# # Roof += TileOnFreshRoof
# # assert np.all(Roof == Tile)


# # ------------------------------------------------------------------- 
# # (1) Thatcher deception for individual face (ID)
# #       🏭 --> FUNCTION 🏭 🚀
# #
# #   Required:   F       ndarray [row,col,feature] uint8     [0,255]
# #               ROI     ndarray [row,col,feature] float64   {0.,1.}

# Ft = thatcher_face(F, ROI)

# # Montage of factorial Thatcher-by-Inverted
# #  -------------------------------------
# # | upright-normal  | upright-Thatcher  |
# # |-------------------------------------|
# # | inverted-normal | inverted-Thatcher |
# #  -------------------------------------
# row0 = np.c_[combine_features(F), combine_features(Ft)]
# row1 = np.c_[rotate_180(combine_features(F)), rotate_180(combine_features(Ft))]
# montage = np.r_[row0, row1]
# plt.imshow(montage, cmap="gray"); plt.show()


# # -------------------------------------------------------------------
# # (2) Spacing out features to alleviate crowding (and the opposite)
# #       🏭 --> FUNCTION 🏭 🚀
# #
# #   Required:   F       ndarray [row,col,feature] uint8     [0,255]
# #               ROI     ndarray [row,col,feature] float64   {0.,1.}
# #
# #               scale       scalar  (0, ∞]
# #               shape_out   tuple (rows,cols)
# #                           * make check this is >= scale X shape_in?

# # 🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪
# # scale = 1
# # shape_out = SHAPE
# # FnS1 = space_out_features(F, ROI, scale=scale, shape_out=shape_out)
# # FnS1_ = image_as_uint8(FnS1)
# # assert np.all(FnS1_==F)


# if SPACING_MOVIE:
#     # Gradual spacing in and out across multiple frames
#     scaling_min, scaling_max = .5, 2
#     scales_half = np.arange(scaling_min,scaling_max,step=.1)
#     scales = np.r_[scales_half, np.flip(scales_half[1:-1])]
#     shape_out = (round(SHAPE[0]*scaling_max), round(SHAPE[1]*scaling_max))
#     movie = np.empty((shape_out[0], shape_out[1], len(scales)))
#     for i, s in enumerate(scales):
#         Fn = space_out_features(F, ROI, scale=s, shape_out=shape_out)
#         frame = Fn.astype(float).mean(-1)
#         movie[:,:,i] = frame - frame[0,0]

#     # Write frames to temporary folder
#     temp_folder = cwd / wrff_faces / ("spacing-" + ID)
#     if not temp_folder.exists():
#         temp_folder.mkdir(parents=False, exist_ok=False)
#     movie = max_stretch_original_0_is_127(movie) # image_as_uint8()
#     for i in range(len(scales)):
#         file_name = "image-" + str(i) + ".png"
#         imsave(temp_folder / file_name, movie[:,:,i], check_contrast=False)

#     # Compose GIF from frames
#     out_file = cwd / wrff_faces / (ID + "-spacing.gif")
#     commandFull = "ffmpeg -i " + str(temp_folder) + os.sep + "image-%d.png " + str(out_file)
#     retCode = subprocess.call(commandFull, shell=True)

#     # Clear out and remove temporary folder
#     commandClear = "rm " + str(temp_folder) + os.sep + "image-*.png"
#     retCode = subprocess.call(commandClear, shell=True)
#     commandRemove = "rmdir " + str(temp_folder)
#     retCode = subprocess.call(commandRemove, shell=True)

# scale = 1.8
# shape_out = (round(SHAPE[0]*scale), round(SHAPE[1]*scale))
# Fn0 = space_out_features(F, ROI, scale=scale, shape_out=shape_out)
# plt.imshow(combine_features(Fn0), cmap="gray")
# plt.title("Spaced out using function")
# plt.show()    


# # -------------------------------------------------------------------
# # (3) Double face illusion 🏭 --> FUNCTION 🏭 🚀
# #     Hancock, P. J., & Foster, C. (2012). The ‘Double Face' Illusion. Perception, 41(1), 57-70.
# #
# #   Required:   F       ndarray [row,col,feature] uint8     [0,255]
# #               ROI     ndarray [row,col,feature] float64   {0.,1.}
# #               SHAPE   tuple (rows, cols)
# #               percent_down_shift  scalar  real (0,∞]
# #                                   downward shift of duplicate features
# #                                   (not nose) as a percentage of vertical
# #                                   distance between brows and eyes.

# Fextra = double_face_illusion(F, ROI, percent_down_shift=0.92)
# NF = combine_features(Fextra)
# plt.imshow(NF, cmap="gray")
# plt.title("Double-feature meme")
# plt.show()


# # ----------------------------------------------------------------
# # (4) Fat face illusion

# CF = F.astype(float)
# CF = CF - CF[0,0,0]
# face_rows, face_cols = 2, 2.4
# pixel_rows = round(face_rows * SHAPE[0])
# pixel_cols = round(face_cols * SHAPE[1])
# shape_out = (pixel_rows, pixel_cols)
# roof = np.zeros(shape_out)
# icy = int(center_index(SHAPE[0]))
# icx = int(center_index(SHAPE[1]))
# # cx = icx + SHAPE[1]
# cx = round(pixel_cols/2)
# for row in range(2):
#     cy = icy + SHAPE[0]*row
#     for f in range(NUM_FEATURES):
#         roof += tile_placement(CF[:,:,f], shape_out, (cx, cy))

# plt.imshow(max_stretch_original_0_is_127(roof), cmap="gray"); plt.show() # image_as_uint8()


# # End
# # -------------------------------------------------------------------    