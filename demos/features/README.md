# Wavelet-reconstructed facial features

Automatic alignment is first performed to ensure that facial features overlap as much as possible. As shown in [results](../../results/README.md), deviation in feature spacing is minimal. Therefore, once features are generated, feature swapping (e.g., chimeric faces) can be performed with a simple centering operation. Interpupilary distance is set to be constant. While not necessary, constant interpupilary distance makes it easier to generate uniform spatial-frequency characteristics across faces; i.e., a single set of wavelet filters can be defined for an entire set of faces.

Each face is decomposed into a set of 6 images of inner facial features that can be recombined in various ways -- left eyebrow, right eyebrow, left eye, right eye, nose and mouth. Each facial-feature image is a wavelet-based reconstruction, ensuring none of the edge artifacts associated with conventional image-based cutting and pasting. We call each feature a Wavelet-Reconstucted Facial Feature (WRFF). To generate WRFFs, we first establish a region-of-interest (ROI) for each feature, perform a wavelet transform on the image, and then selected wavelet coefficients within each ROI.

Each facial feature is associated with a set of automatically detected landmarks. Region-of-interests (ROI) are based on these landmarks. ROI were initialized differently for eyebrows and other features, but each ROI consists of a binary window the same size as the entire face image. Initial eyebrow ROI were constructed by connecting landmarks into a multi-segment line and then dilating until ROI approximately encompasses eyebrow hair. Initial ROI for other features were the convex hull of the associated landmarks. Final ROI were obtained by dilating all of binary windows incrementally until they are as large as they can be without any overlap among ROI. ROI used to generate the WRFF of a specific face can be easily recomputed for feature rearrangements and background separation (as in this [figure of experimental stimuli](../../demos/features/fig-demos-features.png) and this [figure of celebrities from the show Friends](../../results/collage-friends-labeled.png)).

A wavelet-transform was performed on each aligned face photo. Wavelet parameters were carefully chosen to ensure close correspondence with characteristics of simple cells estimated from human physiology, and the psychophysics of face identification. For example, log Gabor were chosen as the wavelet basis, partly because their transfer functions in the Fourier domain are symmetric in log space - just like human simple cells (Anderson & Burr, 1985; Wilson et al., 1983). Each wavelet has a fullwidth frequency bandwidth (at half-height) of approximately 2 octaves, which falls into the range of human simple cells - from 0.5 to 3 octaves (Daugman, 1985; Webster & De Valois, 1985). We designed wavelets across 6 orientations (increments of 30 degrees) and 3 spatial frequencies. We chose spatial frequencies that center around those that are optimal for face identification by human observers: 6, 10 and 16.6 cycles per face width (Gaspar et al., 2008; J. Gold et al., 1999; Näsänen, 1999).

For each facial feature, 50-percent of the wavelet coefficients within the corresponding ROI with the largest real values (even phase) were selected. Each facial feature was then reconstructed from the sum of those even-phase wavelets (in the spatial domain). Virtually identical results are obtained from selecting and reconstructing from only odd-phase wavelets notwithstanding a simple translation of the entire image by a few pixels. For each individual face, relative contrast across WRFFs was preserved whilst maximizing the grayscale range in a way that keeps the background value constant across all faces (127 is the default).

## Related techniques

Note that the wavelet transform we chose is an overcomplete basis. Unlike wavelet transforms associated with popular image compression methods, there are no off-the-shelf algorithms. However, our code was partially adapted from Matlab code (https://www.peterkovesi.com/matlabfns/) used to estimate perceptually meaningful features based on local phase-congruency (Kovesi, 2000). Unlike prior attempts to deconstruct the human face with wavelets in a perceptually meaningful way (Xu et al., 2014; Yue et al., 2012), the transform procedure of Kovesi (2000) does attempt to mirror human physiology in as close detail as possible. Incidentallly, this same procedure has also been widely used in practical computer vision applications (Despotović et al., 2015; Moccia et al., 2018). 

The **FAD** method is the first to combine wavelet reconstruction with landmark extraction in order to isolate facial features, and it is the first to apply physiological realistic parameters for wavelets in order to model face images for psychological study.

## References

Anderson, S. J., & Burr, D. C. (1985). Spatial and temporal selectivity of the human motion detection system. Vision Research, 25(8), 1147–1154. https://doi.org/10.1016/0042-6989(85)90104-X

Daugman, J. G. (1985). Uncertainty relation for resolution in space, spatial frequency, and orientation optimized by two-dimensional visual cortical filters. JOSA A, 2(7), 1160–1169. https://doi.org/10.1364/JOSAA.2.001160

Despotović, I., Goossens, B., & Philips, W. (2015). MRI Segmentation of the Human Brain: Challenges, Methods, and Applications. Computational and Mathematical Methods in Medicine, 2015(1), 450341. https://doi.org/10.1155/2015/450341

Gaspar, C., Sekuler, A. B., & Bennett, P. J. (2008). Spatial frequency tuning of upright and inverted face identification. Vision Research, 48(28), 2817–2826. https://doi.org/10.1016/j.visres.2008.09.015

Gold, J., Bennett, P. J., & Sekuler, A. B. (1999). Identification of band-pass filtered letters and faces by human and ideal observers. Vision Research, 39(21), 3537–3560. https://doi.org/10.1016/S0042-6989(99)00080-2

Kovesi, P. (2000). Phase congruency: A low-level image invariant. Psychological Research, 64(2), 136–148. https://doi.org/10.1007/s004260000024

Moccia, S., De Momi, E., El Hadji, S., & Mattos, L. S. (2018). Blood vessel segmentation algorithms—Review of methods, datasets and evaluation metrics. Computer Methods and Programs in Biomedicine, 158, 71–91. https://doi.org/10.1016/j.cmpb.2018.02.001

Näsänen, R. (1999). Spatial frequency bandwidth used in the recognition of facial images. Vision Research, 39(23), 3824–3833. https://doi.org/10.1016/S0042-6989(99)00096-6

Webster, M. A., & De Valois, R. L. (1985). Relationship between spatial-frequency and orientation tuning of striate-cortex cells. Journal of the Optical Society of America A, 2(7), 1124. https://doi.org/10.1364/JOSAA.2.001124

Wilson, H. R., Mcfarlane, D. K., & Phillips, G. C. (1983). Spatial frequency tuning of orientation selective units estimated by oblique masking. Vision Research, 23(9), 873–882. https://doi.org/10.1016/0042-6989(83)90055-X

Xu, X., Biederman, I., & Shah, M. P. (2014). A neurocomputational account of the face configural effect. Journal of Vision, 14(8), 9. https://doi.org/10.1167/14.8.9

Yue, X., Biederman, I., Mangini, M. C., Malsburg, C. von der, & Amir, O. (2012). Predicting the psychophysical similarity of faces and non-face complex shapes by image-based measures. Vision Research, 55, 41–46. https://doi.org/10.1016/j.visres.2011.12.012
