import numpy as np

# {EO and DilatedBW and CriterionQuantileAmp} --> chosen_coefficients

def select_coefficients_by_feature(EO, DilatedBW, CriterionQuantileAmp=0.50):
    # Initialize wavelet selection
    NUM_FEATURES = DilatedBW.shape[2]
    IMG_ROWS, IMG_COLS, nscale, norient = EO.shape
    chosen_coefficients = np.zeros((IMG_ROWS, IMG_COLS, nscale, norient, NUM_FEATURES))

    # Mean amplitude by scale, PredictedAmp (based on all coefficients)
    num_components = nscale * norient
    num_pixels = IMG_ROWS * IMG_COLS
    AEO = np.zeros((num_components, num_pixels))
    for si in range(nscale):
        for ai in range(norient):
            counter = si*norient + ai
            TempEO = EO[:,:,si,ai]
            AEO[counter,:] = np.abs(TempEO.reshape((1,num_pixels)))
    # NEWA = AEO.T.reshape((num_pixels*norient, nscale)) # inappropriate for C-like index order
    NEWA = AEO.reshape((nscale, num_pixels*norient)) # appropriate for C-like index order
    PredictedAmp = NEWA.mean(axis=1) # (nscale, )

    # Calculate chosen_coefficients for each facial feature
    #   Select criterion percentage of coefficients from within a feature but across scale and orientation.
    #   Criterion is CriterionQuantileAmp.
    #   Coefficients with high amplitude selected.
    #   Coefficients are normalized by mean amplitude by scale before ordering.
    for feature_i in range(NUM_FEATURES):
        CurrentBW = DilatedBW[:,:,feature_i]
        TotalPix = int(CurrentBW.sum())
        AEO = np.zeros((num_components, TotalPix))
        for si in range(nscale):
            for ai in range(norient):
                counter = si*norient + ai
                TempEO = (np.abs(CurrentBW*EO[:,:,si,ai]) / PredictedAmp[si]).flatten()
                AEO[counter,:] = TempEO[TempEO>0].reshape((1,TotalPix))
        MinimumAmp = np.quantile(AEO.flatten(), CriterionQuantileAmp)
        for si in range(nscale):
            for ai in range(norient):
                TempEO = np.abs(EO[:,:,si,ai]) / PredictedAmp[si]
                TempMap = (CurrentBW>0) & (TempEO>MinimumAmp)
                chosen_coefficients[:,:,si,ai,feature_i] = TempMap
    return(chosen_coefficients)
    
# End


# -------------------------------------------------------------------