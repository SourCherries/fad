from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wrff import Ensemble


# Face Recognition Lab faces (not yet processed)
source = Path.cwd() / "FRL-LS"
bookends = ("","jpg")

FRL = Ensemble(dir_source = source, file_bookends = bookends)
FRL.list_faces()
FRL.add_to_roster("014_03")
FRL.add_to_roster("122_03")
FRL.display_roster()

FRL = Ensemble(dir_source = source, file_bookends = bookends, INTER_PUPILLARY_DISTANCE = 64)
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