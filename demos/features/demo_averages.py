from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.util import montage

import fad as fd


# HERE = Path(__file__).resolve().parent
HERE = Path("/Users/z10722/Tools/fad/demos/features")
mother_ship = HERE / "faces-of-merkel"
bookends = ("merkel-",".jpg")

# Originals (aligned)
Merk = fd.Ensemble(dir_source = mother_ship, file_bookends = bookends, INTER_PUPILLARY_DISTANCE = 64)


# Roster[:].img 👉🏻 maximize inner-face contrast around 127 👉🏻 Ensemble roster method