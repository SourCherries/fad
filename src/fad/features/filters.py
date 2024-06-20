import math
import numpy as np


# Helper functions --------------------------------------------------
def grid_normalised(rows, cols):
    # Set up X and Y matrices with ranges normalised to +/- 0.5
    # The following code adjusts things appropriately for odd and even values
    # of rows and columns.
    y, x = np.mgrid[0:rows, 0:cols]
    if (cols % 2):
        xc = (x - (cols-1)/2) / (cols-1)
    else:
        xc = (x - cols/2) / cols
    if (rows % 2):
        yc = (y - (rows-1)/2) / (rows-1)
    else:
        yc = (y - rows/2) / rows
    return((xc,yc))


def lowpassfilter(xc, yc, cutoff=.45, n=15):
    assert cutoff >= 0  # radius
    assert cutoff <= .5
    assert int(n)==n    # sharpness
    assert n >= 1
    radius = np.sqrt(xc**2 + yc**2)  # A matrix with every pixel = radius relative to centre.
    f = np.fft.ifftshift( 1.0 / (1.0 + (radius / cutoff)**(2*n)) )  # The filter
    return(f)


# Main functions ----------------------------------------------------
def get_ffilters_simple(wavelength_pix, rows, cols, norient=6, sigmaOnf=0.55, dThetaOnSigma=1.5):
    thetaSigma = math.pi/norient/dThetaOnSigma  # Calculate the standard deviation of the
                                                # angular Gaussian function used to
                                                # construct filters in the freq. plane.
    xc, yc = grid_normalised(rows, cols)

    radius = np.sqrt(xc**2 + yc**2)  # Matrix values contain *normalised* radius from centre.
    theta = np.arctan2(-yc, xc)      # Matrix values contain polar angle.
                                     # (note -ve y is used to give +ve
                                     # anti-clockwise angles)

    radius = np.fft.ifftshift(radius)  # Quadrant shift radius and theta so that filters
    theta  = np.fft.ifftshift(theta)   # are constructed with 0 frequency at the corners.
    radius[0,0] = 1                    # Get rid of the 0 radius value at the 0
                                       # frequency point (now at top-left corner)
                                       # so that taking the log of the radius will
                                       # not cause trouble.
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    lp = lowpassfilter(xc, yc)

    nscale = wavelength_pix.size
    logGabor = np.zeros((rows, cols, nscale))
    for s in range(nscale):
        fo = 1.0/wavelength_pix[s]            # Centre frequency of filter.
        logGabor[:,:,s] = np.exp((-(np.log(radius/fo))**2) / (2 * np.log(sigmaOnf)**2))
        logGabor[:,:,s] = logGabor[:,:,s]*lp  # Apply low-pass filter
        logGabor[:,:,s][0,0] = 0              # Set the value at the 0 frequency point of the filter
                                              # back to zero (undo the radius fudge).    

    # Then construct the angular filter components...

    spread = np.zeros((rows, cols, norient))
    for o in range(norient):
        angl = (o-1)*np.pi/norient                                 # Filter angle.
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)     # Difference in sine.
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)     # Difference in cosine.
        dtheta = np.abs(np.arctan2(ds,dc))                         # Absolute angular distance.
        spread[:,:,o] = np.exp((-dtheta**2) / (2 * thetaSigma**2)) # Calculate the
                                                                   # angular filter component.    
    # The main loop...
    Ffilter = np.zeros((rows,cols,nscale,norient))
    for o in range(norient):
        for s in range(nscale):
                Ffilter[:,:,s,o] = logGabor[:,:,s] * spread[:,:,o]
    return(Ffilter)    


def face_filters(shape, INTER_PUPILLARY_DISTANCE):
    IMG_ROWS, IMG_COLS = shape
    mult = 1.66
    CPF = 10 * mult ** np.array([-1, 0, 1])
    wavelength_p = (2 * INTER_PUPILLARY_DISTANCE) / CPF
    ffilters = get_ffilters_simple(wavelength_p, IMG_ROWS, IMG_COLS)
    _, _, nscale, norient = ffilters.shape
    print(f'Cylces per face width {CPF}')
    sfilters = np.zeros((IMG_ROWS, IMG_COLS, nscale, norient), dtype=np.complex128)
    for si in range(nscale):
        for ai in range(norient):
            ctemp = np.fft.ifftshift(np.fft.ifft2( ffilters[:,:,si,ai] ))
            rtemp = np.real(ctemp)
            rtemp = rtemp - rtemp[0,0]
            itemp = np.imag(ctemp)
            itemp = itemp - itemp[0,0]
            sfilters[:,:,si,ai] = rtemp + 1j * itemp    
    results = {"ffilters": ffilters,
               "sfilters": sfilters}
    return(results)


def get_eo(img, ffilters):
    nrows, ncols, nscale, norient = ffilters.shape
    f_np = np.fft.fft2(img)
    EO = np.zeros((nrows,ncols,nscale,norient),dtype=complex)
    for o in range(norient):
        for s in range(nscale):
            filt = ffilters[:,:,s,o] + 1j * ffilters[:,:,s,o]
            EO[:,:,s,o] = np.fft.ifft2( f_np * filt )
    return(EO)


# Display -----------------------------------------------------------
def eo_montage(EO, phase="even"):
    _, _, nscale, _ = EO.shape
    if phase == "even":
        coefficients = np.real(EO)
    elif phase == "odd":
        coefficients = np.imag(EO)
    this_scale = 0
    E = coefficients[:,:,this_scale,0]
    for a in range(1, EO.shape[3]):
        E = np.c_[E, coefficients[:,:,this_scale,a]]
    montage = np.copy(E)
    if nscale > 1:
        for this_scale in range(1,nscale):
            E = coefficients[:,:,this_scale,0]
            for a in range(1, EO.shape[3]):
                E = np.c_[E, coefficients[:,:,this_scale,a]]
            montage = np.r_[montage, E]
    return(montage)


# end
# -------------------------------------------------------------------