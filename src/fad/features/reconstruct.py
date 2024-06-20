import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def RectLeft():
    return(0)

def RectTop():
    return(1)

def RectRight():
    return(2)

def RectBottom():
    return(3)

def RectCenter(r):
    x = int((r[0]+r[2])*.5)
    y = int((r[1]+r[3])*.5)
    return(np.array([x, y]))

def RectHeight(r):
    return(r[RectBottom()]-r[RectTop()])

def RectWidth(r):
    return(r[RectRight()]-r[RectLeft()])

def image_as_uint8(FeatureImages):
    FI8 = FeatureImages - FeatureImages.min()
    FI8 = (255 * FI8 / FI8.max()).astype(np.uint8)
    return(FI8)    

def max_stretch_original_0_is_127(in_image):
    in_image = in_image - in_image[0,0]
    m = abs(in_image).max()
    out_image = (in_image/m) * 127.5 + 127.5
    return out_image.astype(np.uint8)

def combine_features(F):
    M = F.astype(float).mean(-1)
    M = max_stretch_original_0_is_127(M)
    return M

# Valid/transparent reconstruction code -----------------------------
def tile_placement_ranges(tile_length, roof_length, roof_index):
    # check
    assert int(roof_index) == roof_index
    assert roof_index > -1
    assert roof_index < roof_length

    # indices to image tile
    tile_range = [0, tile_length]

    # indices to background
    if tile_length % 2:
        margin_a = int((tile_length-1) / 2)
        margin_b = margin_a
        roof_range = [roof_index - margin_a, roof_index + margin_b + 1]
    else:
        margin_a = int((tile_length) / 2)
        margin_b = int((tile_length-1) / 2)
        roof_range = [roof_index - margin_a, roof_index + margin_b + 1]

    # early clipping
    if roof_range[0] < 0:
        tile_range[0] = np.abs(roof_range[0])
        roof_range[0] = 0

    # late clipping
    if roof_range[1] > roof_length:
        clip = roof_range[1] - roof_length
        roof_range[1] -= clip
        tile_range[1] -= clip

    # package
    locs = {"roof": slice(roof_range[0],roof_range[1]),
            "tile": slice(tile_range[0],tile_range[1])}
    return(locs)

def tile_placement(Tile, RoofShape, DestinationXY):
    """
    note: this way i can reuse this function in other contexts like feature explosion
    """
    destination_x, destination_y = DestinationXY
    rows = tile_placement_ranges(Tile.shape[0], RoofShape[0], destination_y)
    cols = tile_placement_ranges(Tile.shape[1], RoofShape[1], destination_x)
    TileOnFreshRoof = np.zeros(RoofShape)
    TileOnFreshRoof[rows["roof"], cols["roof"]] = Tile[rows["tile"], cols["tile"]]
    return(TileOnFreshRoof)


def wrff_progress(sfilters, EO, chosen_coefficients, img_format="uint8"):    
    MR, MC, nscale, norient, TotalFeatures = chosen_coefficients.shape
    fh, fw = MR, MC
    ImageRect = np.array([0, 0, MC, MR])
    ix, iy = RectCenter(ImageRect)
    OutputRows = RectHeight(ImageRect)
    OutputCols = RectWidth(ImageRect)
    FeatureImages = np.zeros((OutputRows,OutputCols,TotalFeatures))
    for ThisF in range(TotalFeatures):
        print(f'\n\nFeature {ThisF}')
        for si in range(nscale):
            for ai in range(norient):
                counter = si*norient + ai
                print(f'\tcomponent {counter} out of {nscale*norient}')
                WaveletMask = chosen_coefficients[:,:,si,ai,ThisF]
                Wreal = np.real(EO[:,:,si,ai])
                wy, wx = np.where(WaveletMask)
                WR, WC = wy, wx
                wy = wy - fh/2
                wx = wx - fw/2
                wy = (wy + iy).astype(int)
                wx = (wx + ix).astype(int)
                NumWavelets = wy.size
                for wi in tqdm(range(NumWavelets)):
                    ScaledImage = Wreal[WR[wi], WC[wi]] * np.real(sfilters[:,:,si,ai])
                    ScaledImage = ScaledImage - ScaledImage[0,0]
                    ImageBit = np.copy(ScaledImage)
                    TileOnFreshRoof = tile_placement(ImageBit, (OutputRows,OutputCols), (wx[wi], wy[wi]))
                    FeatureImages[:,:,ThisF] += TileOnFreshRoof                    
    if img_format == "uint8":
        FeatureImages = image_as_uint8(FeatureImages)
    return(FeatureImages)


def wrff(sfilters, EO, chosen_coefficients, img_format="uint8"):    
    MR, MC, nscale, norient, TotalFeatures = chosen_coefficients.shape
    fh, fw = MR, MC
    ImageRect = np.array([0, 0, MC, MR])
    ix, iy = RectCenter(ImageRect)
    OutputRows = RectHeight(ImageRect)
    OutputCols = RectWidth(ImageRect)
    FeatureImages = np.zeros((OutputRows,OutputCols,TotalFeatures))
    for ThisF in range(TotalFeatures):
        for si in range(nscale):
            for ai in range(norient):
                WaveletMask = chosen_coefficients[:,:,si,ai,ThisF]
                Wreal = np.real(EO[:,:,si,ai])
                wy, wx = np.where(WaveletMask)
                WR, WC = wy, wx
                wy = wy - fh/2
                wx = wx - fw/2
                wy = (wy + iy).astype(int)
                wx = (wx + ix).astype(int)
                NumWavelets = wy.size
                for wi in range(NumWavelets):
                    ScaledImage = Wreal[WR[wi], WC[wi]] * np.real(sfilters[:,:,si,ai])
                    ScaledImage = ScaledImage - ScaledImage[0,0]
                    ImageBit = np.copy(ScaledImage)
                    # TileOnFreshRoof = tile_placement(ImageBit, FeatureImages[:,:,ThisF], (wx[wi], wy[wi]))
                    TileOnFreshRoof = tile_placement(ImageBit, (OutputRows,OutputCols), (wx[wi], wy[wi]))
                    FeatureImages[:,:,ThisF] += TileOnFreshRoof                    
    if img_format == "uint8":
        FeatureImages = image_as_uint8(FeatureImages)
    return(FeatureImages)


# Display -----------------------------------------------------------
def show_features_separate(img, FI8, feature_labels):
    NUM_FEATURES = FI8.shape[2]
    for f in range(NUM_FEATURES):
        together = np.c_[FI8[:,:,f], img]
        plt.imshow(together, cmap="gray")
        plt.title("WRFF " + feature_labels[f])
        plt.show()    

def show_face(img, FI8):
    M = FI8.astype(float).mean(-1)
    M = M - M.min()
    M = (255* M / M.max()).astype(np.uint8)
    together = np.c_[M, img]
    plt.imshow(together, cmap="gray")
    plt.title("WRFF face")
    plt.show()
    return(M)

# End
# -------------------------------------------------------------------