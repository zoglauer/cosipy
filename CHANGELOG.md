# Changelog

## Version 0.4.0

Under development. 

Developers: please keep track of notable changes here.

### New FullDetectorResponse 

HDF5 response file format has changed substantially.  The new format,
which is *not* compatible with the .h5 response format from prior
releases, features
  - internal compression using BitShuffle algorithm, which gives chunk
    access times within 50% of uncompressed for smaller responses and
	much faster for larger ones while offering same or better compression 
	on disk.
  - much faster conversion from .rsp.gz to .h5, with greatly reduced
    memory usage
  - ability to convert from .rsp.gz to .h5 and back to 
    .rsp.gz without losing any of the original header information
	(which facilitates creating lower-resolution versions of large
	responses)
  - separation of raw counts from effective area, which allows
    FullDetectorResponse to export the latter as needed and
    enables future performance improvements for code that
	needs to compute and average PSRs.  The dtype of the effective 
    area, and hence the values returned when computing PSRs etc.,
	can now be set at load time; hence, the same response can be
	used in float32 or float64 with no additional overhead.
  - automatic "good chunks" chunking of .h5 responses at creation time
  - HDF5 tweaks to avoid writing timestamps, so that (with support
    from a forthcoming release of histpy) the MD5 signature of an
	.h5 response does not change every time it is written.
	
.rsp.gz conversion has been broken out from FullDetectorResponse into
its own class, RspConverter, which simplifies the FullDetectorResponse
code and in particular its open() interface.

The new response code deprecates and removes the following
response-related functionality:
  - sparse response format
  - reading spectrum for effective area computation from a file
     (was broken/bit-rotted in DC3 release)
  - miniDC2 format support (unused since DC2)

### New image-deconvolution algorithm: MREM

`MREM` was added to `cosipy.image_deconvolution` (also available in the `ImageDeconvolution`
parameter file as `algorithm: "MREM"`): the multi-resolution regularized expectation
maximization of Knoedlseder et al. (1999, A&A 345, 813), used for the COMPTEL 26Al maps.
  - The correction map is normalised by the square root of the Fisher information (unit
    variance per pixel), decomposed into isotropic wavelet scales on the sphere and thresholded
    at `s` times the noise of each scale, which is determined from Poisson realisations of the
    current model. Scales are admitted progressively from the largest to the smallest, and the
    thresholded correction is applied with a likelihood line search.
  - Two safeguards beyond the paper: the step is bounded so that no pixel changes by more than
    a factor (1 + `max_change`) per iteration, and the update is damped per pixel
    (`min_ratio`), which prevents single pixels from overshooting and being trapped at zero.
  - Opt-in `acceleration:line_search: "robust"` (default `"newton"`): the thresholded direction
    is not always an ascent direction; the robust mode drops the planes with a non-positive
    directional derivative from such a direction and finds the step by a safeguarded
    Newton/bisection search.
  - The Fisher normalisation uses a new optional data-interface method,
    `calc_squared_T_product` (sum_i H_i R_ij^2), implemented for the Galactic-coordinate CDS of
    `DataIF_COSI_DC2`, and `DataInterfaceCollection.calc_summed_squared_T_product`. Interfaces
    without it raise `NotImplementedError`, and MREM falls back to the exposure normalisation.
  - Defaults are in `image_deconvolution/constants.py` (`DEFAULT_MREM_*`).

### New wasabi location for development files

To support file format changes in the develop branch that are
incompatible with DC3 and cosipy versions <= 0.3, we have established
a "COSI-SMEX/develop" tree in the public wasabi bucket.  Currently,
this tree holds the new-format .h5 detector responses and the
corresponding .rsp.gz files.  The DC3 tutorials have been updated to
use this new tree for files whose format has changed since DC3.



## Version 0.2.x

Version 0.2.0 is the first released version of the "new" cosipy. It was reimplemented from scratch based on the "old" cosipy, now called "[cosipy classic](https://github.com/cositools/mirror-cosipy-classic)". It is the version used for [Data Challenge 2](https://github.com/cositools/cosi-data-challenge-2). Versions 0.2.x have bug fixes for version 0.2.0 but keep backward compatibility and do not add new features.
