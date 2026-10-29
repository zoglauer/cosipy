"""
Constants for the image deconvolution module.
This module centralizes magic numbers and default values.
"""
import astropy.units as u

# Numerical Constants
NUMERICAL_ZERO = 1e-12 # Small value to avoid division by zero in expectation calculations.
CHUNK_SIZE_FITS = 998 # Maximum columns in FITS table (FITS limit is 1000, using 998 for safety).

# Physical Constants
EARTH_RADIUS_KM = 6378.0 # Earth radius in km

# Default Values - General Deconvolution Algorithm Parameters
DEFAULT_MINIMUM_FLUX = 0.0 # Default minimum flux to enforce non-negativity.
DEFAULT_ITERATION_MAX = 1 # Default maximum number of iterations.
DEFAULT_STOPPING_THRESHOLD = 1e-2 # Default convergence threshold for log-likelihood change.

# Default Values - Background Normalization
DEFAULT_BKG_NORM_RANGE = [0.0, 100.0] # Default allowed range [min, max] for background normalization factors.

# Default Values - Response Weighting
DEFAULT_RESPONSE_WEIGHTING_INDEX = 0.5 # Default power index for response weighting: filter = (exposure/max)^index.

# Default Values - MREM (Knoedlseder+99)
DEFAULT_MREM_WAVELET_SCALES_DEG = [4.0, 8.0, 16.0, 32.0, 64.0] # FWHMs of the wavelet smoothing kernels (deg); the finest should be about the angular resolution.
DEFAULT_MREM_SIGNIFICANCE = 3.0 # Threshold of the wavelet coefficients in units of the noise of each scale.
DEFAULT_MREM_NOISE_REALIZATIONS = 8 # Poisson realisations of the current model used to estimate the noise per scale.
DEFAULT_MREM_MAX_ITERATIONS_PER_SCALE = 40 # Maximum number of iterations per progressive-admission stage.
DEFAULT_MREM_ACCEL_FACTOR_MAX = 10.0 # Maximum step length of the line search.
DEFAULT_MREM_BKG_NORM_EM_STEPS = 3 # EM steps of the background normalizations after each image update.
DEFAULT_MREM_MIN_RATIO = 0.1 # Per-pixel damping: a pixel is multiplied by at least this factor per iteration.
DEFAULT_MREM_MAX_CHANGE = 0.5 # Step bound: no pixel changes by more than a factor (1 + max_change) per iteration.
DEFAULT_MREM_STOPPING_THRESHOLD = 0.05 # Log-likelihood gain per iteration below which a stage is converged.
DEFAULT_MREM_LINE_SEARCH = 'newton' # Line search: 'newton' or 'robust'.
MREM_FLOOR = 1e-30 # Floor of the expected counts in the MREM line search and normalisation.
