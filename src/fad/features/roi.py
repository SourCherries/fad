# ONLY FOR TEST
# import json
# from PIL import Image as PilImage
# import matplotlib.pyplot as plt

# REQUIRED FOR SCRIPT
import numpy as np
from skimage.morphology import convex_hull_image, dilation, disk

def interpolate_brow_points(points, number_new_points=20):
    """
    IN
        points                (number-of-points, x-y)
        number_new_points     scalar int, between each pair of points
    OUT
        new_points            ((number-of-points - 1)*number_new_points, x-y)
    """
    number_points = points.shape[0]
    P0 = points[0:-1,:]
    P1 = points[1:,:]
    N0 = np.tile(P0[np.newaxis,:,:], (number_new_points,1,1))
    N1 = np.tile(P1[np.newaxis,:,:], (number_new_points,1,1))
    t = np.tile(np.arange(number_new_points).reshape((number_new_points,1,1)), (1,number_points-1,2)) / number_new_points
    B = ((1-t)*N0 + t*N1).astype(int)
    new_points = B.reshape(((number_points-1)*number_new_points, 2))
    return(new_points)

def get_feature_roi(shape, landmarks):
    x = np.array(landmarks["left_eyebrow"][0::2])
    y = np.array(landmarks["left_eyebrow"][1::2])
    radius_left = np.sqrt(np.diff(x)**2 + np.diff(y)**2).max()
    x = np.array(landmarks["right_eyebrow"][0::2])
    y = np.array(landmarks["right_eyebrow"][1::2])
    radius_right = np.sqrt(np.diff(x)**2 + np.diff(y)**2).max()
    radius_brow = max(radius_left, radius_right) + 1
    ir, ic = shape
    roi_features = ["left_eyebrow", "right_eyebrow", "left_eye", "right_eye", "nose", "mouth_outline"]
    starting_points = [0, 0, 0, 0, 1, 0]
    NUM_FEATURES = len(roi_features)
    CBW = np.zeros((ir, ic, NUM_FEATURES))
    for channel, (feature, start) in enumerate(zip(roi_features, starting_points)):
        LBW = np.zeros((ir, ic))
        x = np.array(landmarks[feature][0::2], dtype=int)[start:]
        y = np.array(landmarks[feature][1::2], dtype=int)[start:]
        if "brow" not in feature:
            LBW[y, x] = 1
            CBW[:, :, channel] = convex_hull_image(LBW)
        else:
            points = np.array([x, y]).T
            new_points = interpolate_brow_points(points, number_new_points=20)
            LBW = np.zeros((ir, ic))
            LBW[new_points[:,1], new_points[:,0]] = 1
            CBW[:, :, channel] = dilation(LBW, footprint=disk(1))
    se = disk(1)  # smallest increment for gradual dilation
    DilatedBW = CBW.copy()
    NBW = CBW.copy()
    SBW = NBW.sum(axis=2)
    while np.where(SBW.flatten()>1)[0].size == 0:
        DilatedBW = NBW.copy()
        for feature_i in range(NUM_FEATURES):
            dilation(NBW[:,:,feature_i], footprint=se, out=NBW[:,:,feature_i])
        SBW = NBW.sum(axis=2)
    results = {"DilatedBW": DilatedBW,
               "roi_features": roi_features}
    return(results)


# Example use of get_feature_roi() ----------------------------------
# with open("faces-aligned/landmarks.txt") as json_data:
#     landmarks = json.load(json_data)
#     json_data.close()

# files = list(landmarks)
# file = files[0]
# person = landmarks[file]

# file_full = "faces-aligned/" + file
# img = np.array(PilImage.open(file_full).convert("L"))
# ir, ic = img.shape
# shape = (ir, ic)

# DilatedBW = get_feature_roi(shape, person)
# plt.imshow(DilatedBW.sum(2)); plt.show()

# fig, ax = plt.subplots(2)
# ax[0].imshow(DilatedBW[:,:,0])
# ax[1].imshow(DilatedBW[:,:,0])
# plt.show()


# End
# -------------------------------------------------------------------