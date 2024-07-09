from pathlib import Path
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave

import fad as fd
import fad.features.shift_features as shift

# -----------------------------------------------
# from .align.make_aligned_faces import get_source_files, get_landmarks, align_procrustes, place_aperture
# from .align.make_files import clone_directory_tree

# from .features.filters import face_filters, get_eo, eo_montage
# from .features.roi import get_feature_roi
# from .features.coefficients import select_coefficients_by_feature
# from .features.reconstruct import wrff_progress as reconstruct
# from .features.reconstruct import combine_features
# from .features.shift_features import thatcher_face, space_out_features, space_out_ipd, double_face_illusion

# -----------------------------------------------
# from fad import align as af
# af.make_aligned_faces.morph_between_two_faces

from fad.align.make_aligned_faces import morph_between_two_faces as morph
# -----------------------------------------------

# HERE = Path(__file__).resolve().parent
HERE = Path("/Users/z10722/Tools/fad/demos/features")
FRL_folder = HERE / "FRL-LS"
stim_folder = HERE / "stimuli"
fig_folder = HERE/ "figs"
ID = ["122_03", "014_03"]  # 124_03

# Basic usage
bookends = ("","jpg")
FRL = fd.Ensemble(dir_source = FRL_folder, file_bookends = bookends, INTER_PUPILLARY_DISTANCE = 64)
FRL.list_faces()
FRL.add_to_roster(ID[0])
FRL.add_to_roster(ID[1])

num_morphs = 5
face_array, p, morph_path = morph(str(FRL.dir_aligned) + os.sep,
                                  do_these=[0, 1],
                                  num_morphs=num_morphs,
                                  file_prefix=FRL.file_bookends[0],
                                  file_postfix=FRL.file_bookends[1])


FRL = fd.Ensemble(dir_source = Path(morph_path), file_bookends = ("N","png"), make_windowed_faces=True, INTER_PUPILLARY_DISTANCE = 64)
ID = ["N"+str(i) for i in range(num_morphs)]
for i in range(num_morphs):
    FRL.add_to_roster(ID[i])
FRL.clip_roster_margins(margins=(1/6, 1/4))
FRL.display_roster()

FRL = fd.Ensemble(dir_source = Path(morph_path), file_bookends = ("N","png"))
for i in range(num_morphs):
    FRL.add_to_roster(ID[i])
FRL.display_roster()