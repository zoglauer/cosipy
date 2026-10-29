import numpy as np
import astropy.units as u
import healpy as hp
import logging
logger = logging.getLogger(__name__)

from histpy import Histogram

from .RichardsonLucy import RichardsonLucy
from ..models.allskyimage import AllSkyImageModel
from ..constants import (DEFAULT_MREM_WAVELET_SCALES_DEG, DEFAULT_MREM_SIGNIFICANCE, DEFAULT_MREM_NOISE_REALIZATIONS,
                         DEFAULT_MREM_MAX_ITERATIONS_PER_SCALE, DEFAULT_MREM_ACCEL_FACTOR_MAX, DEFAULT_MREM_BKG_NORM_EM_STEPS,
                         DEFAULT_MREM_MIN_RATIO, DEFAULT_MREM_MAX_CHANGE, DEFAULT_MREM_STOPPING_THRESHOLD,
                         DEFAULT_MREM_LINE_SEARCH, MREM_FLOOR)

LINE_SEARCH_METHODS = ('newton', 'robust')

def _value(x):
    """Plain numpy values of a Quantity or an array."""
    return x.value if hasattr(x, 'unit') else np.asarray(x)

class MREM(RichardsonLucy):
    r"""
    Multi-Resolution Regularized Expectation Maximization (MREM), Knoedlseder+99 (A&A 345, 813, Sect. 2).

    MREM is Richardson-Lucy (ML-EM) in which only the *significant* part of each update is applied. The
    correction of every iteration is split into angular scales (an isotropic wavelet transform on the
    sphere), and only coefficients that are significant compared with the Poisson noise of their scale are
    kept. Noise is therefore not deconvolved, and the iteration converges instead of slowly fitting noise
    as RL does. The regularisation parameter is the significance level :math:`s`.

    **Notation.** :math:`i` indexes the data space (CDS bins), :math:`j` the image pixels,
    :math:`k` the background models and :math:`l` the wavelet planes.

    - :math:`n_i`: observed counts; :math:`f_j`: model (image) value of pixel :math:`j`;
    - :math:`R_{ij}`: response (expected counts in bin :math:`i` per unit of :math:`f_j`), as used by
      ``calc_T_product``;
    - :math:`b_{ki}`, :math:`\beta_k`: background models and their normalisations;
    - expected counts :math:`e_i = \sum_j R_{ij} f_j + \sum_k \beta_k b_{ki}`;
    - log-likelihood :math:`\ln L = \sum_i \left( n_i \ln e_i - e_i \right)`;
    - exposure :math:`\varepsilon_j = \sum_i R_{ij}`; Fisher diagonal :math:`F_j = \sum_i R_{ij}^2 / e_i`.

    All sums over :math:`i` also run over the datasets of the
    :py:class:`DataInterfaceCollection`. With several energy bins, every bin is treated separately
    (its own wavelet transform and its own noise per scale).

    **One iteration** (methods in brackets).

    1. *Normalised correction map* (Eq. 5 of the paper; ``_normalised_correction``). The gradient of
       :math:`\ln L` with respect to :math:`f_j`,

       .. math::
           g_j = \frac{\partial \ln L}{\partial f_j} = \sum_i R_{ij} \left( \frac{n_i}{e_i} - 1 \right),

       is divided by a per-pixel normalisation :math:`N_j` (``_update_normalisation``):

       .. math::
           \delta h_j = \frac{g_j}{N_j}, \qquad
           N_j = \sqrt{F_j} = \sqrt{\sum_i R_{ij}^2 / e_i} \quad (\texttt{normalization: "fisher"}).

       Since :math:`\mathrm{Var}(g_j) = F_j` for Poisson counts with mean :math:`e_i`, the noise of
       :math:`\delta h_j` has unit variance in every pixel, however different the exposure is. The
       Fisher diagonal needs the optional data-interface method ``calc_squared_T_product``
       (:math:`\sum_i H_i R_{ij}^2` with :math:`H_i = 1/e_i`). Without it, MREM falls back to
       :math:`N_j = \varepsilon_j` (``"exposure"``), for which the noise is only approximately uniform.
       Unexposed pixels (:math:`\varepsilon_j = 0`) get :math:`\delta h_j = 0`.

    2. *Wavelet decomposition* (``wavelet_decomposition``, ``_planes``). With :math:`G_\theta` a Gaussian
       smoothing of FWHM :math:`\theta` on the sphere and the scales
       :math:`\theta_1 < \dots < \theta_L` (``wavelet:scales``),

       .. math::
           S_0 = \delta h, \quad S_l = G_{\theta_l} \ast \delta h, \qquad
           w_l = S_{l-1} - S_l \;\; (l = 1 \dots L), \qquad w_{L+1} = S_L,

       so that :math:`\delta h = \sum_{l=1}^{L+1} w_l` exactly (:math:`w_{L+1}` is the coarse residual
       plane). In the code the planes are indexed :math:`0 \dots L-1` (fine to coarse) and :math:`L`
       (residual).

    3. *Noise per scale* (Eq. 6; ``_estimate_noise``, at the start of every stage). :math:`M`
       (``wavelet:noise_realizations``) Poisson realisations :math:`n^{(r)}_i \sim \mathrm{Poisson}(e_i)`
       of the current model are pushed through steps 1 and 2. The noise of plane :math:`l` is

       .. math::
           \sigma_l = \operatorname{median}_j \; \operatorname{std}_r \, w^{(r)}_{l,j},

       the median over exposed pixels of the standard deviation over the realisations. As in the paper,
       the noise is taken from realisations of the model, not estimated from the correction map itself.

    4. *Thresholding* (Eq. 6; ``Mstep``). Only the planes admitted in the current stage
       (:math:`\mathcal{A}`, see below) are used. The DC term :math:`d = \langle w_{L+1} \rangle`
       (mean over exposed pixels) is never thresholded:

       .. math::
           \hat w_{l,j} = \begin{cases} w'_{l,j} & |w'_{l,j}| > s\,\sigma_l \\ 0 & \text{otherwise}
           \end{cases}, \qquad
           w'_l = w_l - d\,[l = L+1], \qquad
           \widehat{\delta h}_j = d + \sum_{l \in \mathcal{A}} \hat w_{l,j}.

       :math:`s` is ``wavelet:significance``.

    5. *Back-transformation* (Eq. 8; ``Mstep``). The thresholded map is turned back into a
       multiplicative RL correction and an update direction,

       .. math::
           c_j = \widehat{\delta h}_j \, \frac{N_j}{\varepsilon_j}, \qquad
           \delta f_j = f_j\, c_j
           \;\left( = f_j\, \widehat{\delta h}_j \frac{\sqrt{F_j}}{\varepsilon_j} \text{ for the Fisher
           normalisation} \right).

       Without thresholding, :math:`c_j = g_j / \varepsilon_j` and :math:`\delta f_j` is exactly the RL
       update :math:`f_j / \varepsilon_j \sum_i R_{ij} (n_i/e_i - 1)`.

    6. *Line search and update* (Eq. 4; ``_line_search``, ``post_processing``). The step :math:`t`
       maximises :math:`\phi(t) = \ln L(f + t\,\delta f)` on :math:`[0, t_{\max}]`. With
       :math:`\Delta e_i = \sum_j R_{ij} \delta f_j`,

       .. math::
           \phi'(t) = \sum_i \Delta e_i \left( \frac{n_i}{e_i + t\,\Delta e_i} - 1 \right), \qquad
           \phi''(t) = -\sum_i \frac{n_i\, \Delta e_i^2}{(e_i + t\,\Delta e_i)^2} \le 0,

       so :math:`\phi` is concave. The step is bounded so that no pixel changes by more than a factor
       :math:`1 + m` (:math:`m` = ``acceleration:max_change``), and by :math:`t_a` =
       ``acceleration:accel_factor_max``:

       .. math::
           t_{\max} = \min\left( t_a, \; \frac{m}{\max_j |c_j|} \right)
           \qquad (\text{without acceleration: } t = \min(1, m / \max_j |c_j|)).

       The update is damped per pixel with :math:`\rho` = ``acceleration:min_ratio``,

       .. math::
           f_j \leftarrow f_j \, \max\left( 1 + t\, c_j, \; \rho \right),

       followed by the ``minimum_flux`` constraint. The paper bounds the step only by positivity. With
       accelerated steps on a thresholded direction, single pixels then overshoot by an order of
       magnitude, bounce and end up at zero, where a multiplicative update cannot revive them; the bound
       and the damping prevent this.

    7. *Background normalisations* (``post_processing``; fixed in the paper). After the image update,
       ``background_normalization_optimization:em_steps`` EM steps at fixed source expectation,

       .. math::
           \beta_k \leftarrow \beta_k \, \frac{\sum_i b_{ki}\, n_i / e_i}{\sum_i b_{ki}},

       each followed by clipping to ``background_normalization_optimization:range``.

    **Progressive admission of scales** (``admitted_planes``, ``_stage_converged``,
    ``check_stopping_criteria``). Stage 0 admits only the residual plane,
    :math:`\mathcal{A} = \{L+1\}`. Stage :math:`m` admits the :math:`m` coarsest detail planes as well,
    :math:`\mathcal{A} = \{L+1-m, \dots, L+1\}`. The next stage starts when the current one has
    converged, i.e. when any of these holds:

    - no coefficient is significant and the DC step :math:`|d|\,t` is below :math:`10^{-3}`;
    - the step :math:`t` is below :math:`10^{-3}` (no likelihood gain along the direction);
    - the gain of :math:`\ln L` was below ``stopping_criteria:threshold`` in two consecutive iterations;
    - the stage has used ``wavelet:max_iterations_per_scale`` iterations.

    The run ends when the stage with all planes (:math:`m = L`) has converged, or at ``iteration_max``.
    With ``wavelet:progressive: False``, all planes are admitted from the start and the run ends at the
    first convergence.

    **Robust line search** (``acceleration:line_search: "robust"``, opt-in; ``Mstep``,
    ``_line_search_robust``). The thresholded direction is not always an ascent direction: when newly
    admitted fine-scale coefficients combine with coarser ones, :math:`\phi'(0)` can be negative. The
    default ``"newton"`` search then returns :math:`t = 0` and ends the stage, or the whole run at the
    finest scale. The robust mode computes the contribution of every admitted plane to :math:`\phi'(0)`,

    .. math::
        \phi'(0) = \sum_j \delta f_j\, g_j = \sum_{l \in \mathcal{A}} \gamma_l, \qquad
        \gamma_l = \sum_j f_j \, \frac{N_j^2}{\varepsilon_j} \, \delta h_j \, \hat w_{l,j}

    (the DC term belongs to the residual plane). If :math:`\sum_l \gamma_l \le 0`, the planes with
    :math:`\gamma_l \le 0` are dropped from this iteration's direction, which is then an ascent direction
    whenever any plane offers a gain. The step is found by a safeguarded Newton/bisection search on
    :math:`\phi'` over :math:`[0, t_{\max}]`. A zero step then means that no admitted plane offers a
    likelihood gain, i.e. genuine convergence of the stage.

    **Choosing the parameters.**

    - ``significance`` :math:`s`: the expected number of noise coefficients above threshold is about
      :math:`N_\mathrm{pix}\, \mathrm{erfc}(s/\sqrt{2})` per plane (two-sided). With
      :math:`N_\mathrm{pix} = 12288` (nside 32), :math:`s = 3` admits about 30 noise coefficients per plane
      and :math:`s = 4` fewer than one. A smaller :math:`s` recovers more faint emission and also more
      spurious structure. The paper uses 2.5-3.5.
    - ``scales``: the finest scale should be about the angular resolution. Finer planes of the correction
      map have so little noise (the response correlates it) that a threshold relative to it admits
      sub-resolution structure, i.e. the noise amplification of plain RL.

    Only :py:class:`cosipy.image_deconvolution.AllSkyImageModel` is supported.

    **Results** stored per iteration: ``iteration``, ``stage``, ``finest_scale`` (FWHM of the finest
    admitted plane, None in stage 0), ``model``, ``background_normalization``, ``log-likelihood``,
    ``accel_factor`` (:math:`t`), ``n_significant`` (number of kept coefficients), ``dc_offset``
    (:math:`d` per energy bin) and ``n_dropped_planes`` (robust line search).

    An example of parameter is as follows.

    iteration_max: 300
    minimum_flux:
        value: 0.0
        unit: "cm-2 s-1 sr-1"
    wavelet:
        scales:
            value: [4.0, 8.0, 16.0, 32.0, 64.0]
            unit: "deg"
        significance: 3.0
        noise_realizations: 8
        max_iterations_per_scale: 40
        normalization: "fisher"
        progressive: True
    acceleration:
        activate: True
        accel_factor_max: 10.0
        max_change: 0.5
        min_ratio: 0.1
        line_search: "newton"      # or "robust" (opt-in, see above)
    background_normalization_optimization:
        activate: True
        range: {"albedo": [0.01, 10.0]}
        em_steps: 3
    stopping_criteria:
        threshold: 0.05
    save_results:
        activate: True
        directory: "./results"
        only_final_result: True
    """

    def __init__(self, initial_model, dataset, mask, parameter):

        super().__init__(initial_model, dataset, mask, parameter)

        if not isinstance(initial_model, AllSkyImageModel):
            raise TypeError("MREM supports only AllSkyImageModel.")

        # wavelet decomposition and thresholding
        scales = parameter.get('wavelet:scales:value', DEFAULT_MREM_WAVELET_SCALES_DEG)
        scales_unit = parameter.get('wavelet:scales:unit', 'deg')
        self.wavelet_scales = np.sort(np.asarray(scales, dtype=float)) * u.Unit(scales_unit)
        self.significance = parameter.get('wavelet:significance', DEFAULT_MREM_SIGNIFICANCE)
        self.noise_realizations = int(parameter.get('wavelet:noise_realizations', DEFAULT_MREM_NOISE_REALIZATIONS))
        self.max_iterations_per_scale = int(parameter.get('wavelet:max_iterations_per_scale', DEFAULT_MREM_MAX_ITERATIONS_PER_SCALE))
        self.normalization = parameter.get('wavelet:normalization', 'fisher')
        self.progressive = bool(parameter.get('wavelet:progressive', True))

        if self.normalization not in ('fisher', 'exposure'):
            raise ValueError(f'Unknown normalization "{self.normalization}". Available: "fisher", "exposure"')

        # acceleration (line search)
        self.acceleration_enabled = parameter.get('acceleration:activate', True)
        self.accel_factor_max = parameter.get('acceleration:accel_factor_max', DEFAULT_MREM_ACCEL_FACTOR_MAX)
        self.min_ratio = parameter.get('acceleration:min_ratio', DEFAULT_MREM_MIN_RATIO)
        self.max_change = parameter.get('acceleration:max_change', DEFAULT_MREM_MAX_CHANGE)
        self.line_search = parameter.get('acceleration:line_search', DEFAULT_MREM_LINE_SEARCH)
        if self.line_search not in LINE_SEARCH_METHODS:
            raise ValueError(f'Unknown line_search "{self.line_search}". Available: {LINE_SEARCH_METHODS}')

        # background normalization
        if self.bkg_norm_optimization_enabled:
            self.bkg_norm_em_steps = parameter.get('background_normalization_optimization:em_steps', DEFAULT_MREM_BKG_NORM_EM_STEPS)

        # stopping criteria: log-likelihood gain per iteration below which a scale is converged
        self.stopping_criteria_threshold = parameter.get('stopping_criteria:threshold', DEFAULT_MREM_STOPPING_THRESHOLD)

        # update parameter summary
        self._parameter_summary += [
            ("wavelet_scales", self.wavelet_scales),
            ("significance", self.significance),
            ("noise_realizations", self.noise_realizations),
            ("max_iterations_per_scale", self.max_iterations_per_scale),
            ("normalization", self.normalization),
            ("progressive", self.progressive),
            ("acceleration_enabled", self.acceleration_enabled),
        ]
        if self.acceleration_enabled:
            self._parameter_summary += [
                ("accel_factor_max", self.accel_factor_max),
                ("line_search", self.line_search),
            ]
        self._parameter_summary += [
            ("min_ratio", self.min_ratio),
            ("max_change", self.max_change),
        ]
        if self.bkg_norm_optimization_enabled:
            self._parameter_summary += [
                ("bkg_norm_em_steps", self.bkg_norm_em_steps),
            ]
        self._parameter_summary += [
            ("stopping_criteria_threshold", self.stopping_criteria_threshold),
        ]

    def initialization(self):
        r"""
        Initialization before running the image deconvolution: the RL initialization (exposure map,
        background models), the mask of exposed pixels (:math:`\varepsilon_j > 0`), the random generator
        of the noise realisations (fixed seed, so runs are reproducible) and the stage counters.
        """

        super().initialization()

        self._exposed = _value(self.summed_exposure_map.contents) > 0
        self._nest = self.model.axes['lb'].scheme == 'NESTED'
        self._rng = np.random.default_rng(1999)
        self._n_scales = len(self.wavelet_scales)          # planes 0 .. n-1 (fine to coarse), n = residual
        self._stage = 0
        self._stage_iteration = 0
        self._finished = False
        self._fallback_warned = False
        self.sigma_history = []
        self.n_dropped_planes = 0

    @property
    def admitted_planes(self):
        r"""
        Indices of the wavelet planes admitted in the current stage, as used in the code (0 = finest
        detail plane, ``n_scales`` = coarse residual plane). Stage :math:`m` admits the residual plane and
        the :math:`m` coarsest detail planes; with ``progressive: False`` all planes are admitted.
        """

        if not self.progressive:
            return list(range(self._n_scales + 1))

        return list(range(self._n_scales - self._stage, self._n_scales + 1))

    # ------------------------------------------------------------------ wavelets
    @staticmethod
    def wavelet_decomposition(image, fwhms):
        r"""
        Isotropic multi-scale decomposition of a HEALPix map (RING scheme),

        .. math::
            S_0 = x, \quad S_l = G_{\theta_l} \ast x, \qquad w_l = S_{l-1} - S_l \;\; (l = 1 \dots L),

        with :math:`G_\theta` a Gaussian smoothing of FWHM :math:`\theta` (``healpy.smoothing``). The
        planes and the coarse residual :math:`S_L` add up to the input map exactly.

        Parameters
        ----------
        image : numpy.ndarray
            HEALPix map (RING scheme)
        fwhms : :py:class:`astropy.units.Quantity`
            Increasing FWHMs :math:`\theta_1 < \dots < \theta_L` of the smoothing kernels

        Returns
        -------
        list of numpy.ndarray, numpy.ndarray
            Wavelet planes :math:`w_1 \dots w_L` (fine to coarse) and the coarse residual :math:`S_L`.
        """

        planes = []
        previous = image
        for fwhm in fwhms:
            smoothed = hp.smoothing(image, fwhm = fwhm.to('rad').value, iter = 1)
            planes.append(previous - smoothed)
            previous = smoothed

        return planes, previous

    def _planes(self, image):
        r"""
        Wavelet planes plus the residual plane of one HEALPix map, as an array of shape
        (n_scales + 1, npix). A NESTED map is reordered to RING for the smoothing and back.
        """

        if self._nest:
            image = hp.reorder(image, n2r = True)

        planes, coarse = self.wavelet_decomposition(image, self.wavelet_scales)
        planes = planes + [coarse]

        if self._nest:
            planes = [hp.reorder(p, r2n = True) for p in planes]

        return np.array(planes)

    # ------------------------------------------------------------------ normalisation and noise
    def _normalised_correction(self, ratio_list):
        r"""
        Normalised correction map (step 1, Eq. 5 of the paper),

        .. math::
            \delta h_j = \frac{1}{N_j} \sum_i R_{ij} \left( \frac{n_i}{e_i} - 1 \right),

        with the normalisation :math:`N_j` set by ``_update_normalisation``. Unexposed pixels are 0.

        Parameters
        ----------
        ratio_list : list of :py:class:`histpy.Histogram` or numpy.ndarray
            :math:`n_i / e_i` per dataset: the data for the correction itself, or a Poisson realisation
            :math:`n^{(r)}_i` for the noise estimate.

        Returns
        -------
        numpy.ndarray
            Dimensionless map of shape (npix, n_energy).
        """

        g = self.dataset.calc_summed_T_product([Histogram(data.data_axes, contents = np.asarray(_value(getattr(ratio, 'contents', ratio))) - 1.0)
                                                for data, ratio in zip(self.dataset, ratio_list)])

        correction = np.zeros(self.model.contents.shape)
        correction[self._exposed] = (_value(g.contents) / self._norm)[self._exposed]

        return correction

    def _update_normalisation(self):
        r"""
        Set the per-pixel normalisation :math:`N_j` (``self._norm``, the unit of the exposure map) of the
        correction map for the current expected counts:

        .. math::
            N_j = \sqrt{F_j} = \sqrt{\sum_i R_{ij}^2 / e_i} \quad (\texttt{"fisher"}), \qquad
            N_j = \varepsilon_j = \sum_i R_{ij} \quad (\texttt{"exposure"}).

        The Fisher diagonal uses ``calc_summed_squared_T_product`` with :math:`H_i = 1/e_i`. If a data
        interface does not provide it, the exposure normalisation is used and a warning is logged once.
        Called at every iteration, since :math:`F_j` depends on the current model.
        """

        exposure = _value(self.summed_exposure_map.contents)

        if self.normalization == 'fisher':
            try:
                fisher = self.dataset.calc_summed_squared_T_product(
                    [Histogram(data.data_axes, contents = 1.0 / np.maximum(np.asarray(_value(expectation.contents)), MREM_FLOOR))
                     for data, expectation in zip(self.dataset, self.expectation_list)])
                self._norm = np.sqrt(np.maximum(_value(fisher.contents), 0.0))
                self._norm[self._norm <= 0] = 1.0
                return
            except NotImplementedError:
                if not self._fallback_warned:
                    logger.warning("The data interface does not provide calc_squared_T_product; MREM uses the exposure normalisation.")
                    self._fallback_warned = True

        self._norm = np.where(self._exposed, exposure, 1.0)

    def _estimate_noise(self):
        r"""
        Noise of every wavelet plane of the normalised correction map (step 3, Eq. 6), from
        ``noise_realizations`` Poisson realisations :math:`n^{(r)}_i \sim \mathrm{Poisson}(e_i)` of the
        current expected counts:

        .. math::
            \sigma_l = \operatorname{median}_{j\ \mathrm{exposed}} \; \operatorname{std}_r \, w_l\!\left[
            \delta h^{(r)} \right]_j,

        separately for every energy bin. Sets ``self._sigma`` (n_energy, n_planes) and appends it to
        ``self.sigma_history``. Called at the first iteration of every stage.
        """

        realisations = []
        for _ in range(self.noise_realizations):
            ratio_list = []
            for expectation in self.expectation_list:
                e = np.asarray(expectation.contents)
                ratio_list.append(self._rng.poisson(e) / np.maximum(e, MREM_FLOOR))
            correction = self._normalised_correction(ratio_list)
            realisations.append(np.array([self._planes(correction[:, i]) for i in range(correction.shape[1])]))
        realisations = np.array(realisations)      # (n_real, n_energy, n_planes, npix)

        sd = realisations.std(axis = 0, ddof = 1)
        self._sigma = np.median(sd[:, :, self._exposed[:, 0]] if self._exposed.ndim > 1 else sd, axis = -1)   # (n_energy, n_planes)
        self.sigma_history.append(self._sigma.tolist())

        logger.info(f"wavelet noise per scale (stage {self._stage}): {self._sigma}")

    # ------------------------------------------------------------------ iteration steps
    def pre_processing(self):
        r"""
        Before every iteration: at the first iteration the expected counts :math:`e_i` of the initial
        model; at every iteration the normalisation :math:`N_j` (``_update_normalisation``); at the first
        iteration of each stage the noise per scale :math:`\sigma_l` (``_estimate_noise``).
        """

        if self.iteration_count == 1:
            self.Estep()
            logger.info("The expected count histograms were calculated with the initial model map.")

        self._update_normalisation()

        if self._stage_iteration == 0:
            self._estimate_noise()

    def processing_core(self):
        r"""
        Core processing of each iteration: the M-step (``Mstep``). The line search, the model update and
        the E-step follow in ``post_processing``.
        """

        self.Mstep()
        # Note that Estep() is performed in self.post_processing().

    def Mstep(self):
        r"""
        M-step: steps 1, 2, 4 and 5 of the algorithm. Computes the normalised correction
        :math:`\delta h`, decomposes it, keeps the coefficients of the admitted planes with
        :math:`|w'_{l,j}| > s\,\sigma_l` plus the DC term :math:`d`, and transforms the result back
        into the multiplicative correction and the update direction,

        .. math::
            c_j = \widehat{\delta h}_j \, \frac{N_j}{\varepsilon_j}, \qquad \delta f_j = f_j\, c_j.

        With the robust line search, the contributions :math:`\gamma_l` of the planes to
        :math:`\phi'(0)` are computed (``self.plane_gradients``); if their sum is not positive, the planes
        with :math:`\gamma_l \le 0` are dropped from the direction.

        Sets ``self.delta_model`` (:math:`\delta f`), ``self._multiplicative`` (:math:`c`),
        ``self.n_significant``, ``self.dc_offset`` and ``self.n_dropped_planes``.
        """

        ratio_list = [ data.event / expectation for data, expectation in zip(self.dataset, self.expectation_list) ]

        correction = self._normalised_correction(ratio_list)

        filtered = np.zeros_like(correction)
        self.n_significant = 0
        self.dc_offset = []
        admitted = self.admitted_planes
        residual = self._n_scales
        # thresholded contribution of each admitted plane (the DC term belongs to the residual plane)
        plane_filtered = np.zeros((len(admitted),) + correction.shape)

        for i_energy in range(correction.shape[1]):

            planes = self._planes(correction[:, i_energy])
            dc = float(planes[residual][self._exposed[:, i_energy]].mean())
            self.dc_offset.append(dc)

            for k, j in enumerate(admitted):
                w = planes[j] - (dc if j == residual else 0.0)
                significant = np.abs(w) > self.significance * self._sigma[i_energy, j]
                self.n_significant += int(np.sum(significant))
                plane_filtered[k, :, i_energy] = np.where(significant, w, 0.0) + (dc if j == residual else 0.0)

        self.n_dropped_planes = 0
        if self.line_search == 'robust':
            # directional derivative of the log-likelihood along each plane's contribution:
            # g_k = sum_j f_j (norm_j^2 / exposure_j) correction_j filtered_kj  (= sum_i dmu_i (n_i/e_i - 1))
            exposure_ = _value(self.summed_exposure_map.contents)
            weight = np.zeros_like(correction)
            weight[self._exposed] = (_value(self.model.contents) * self._norm**2)[self._exposed] / exposure_[self._exposed]
            g_planes = np.array([np.sum(weight * correction * pf) for pf in plane_filtered])
            self.plane_gradients = g_planes.tolist()
            if np.sum(g_planes) <= 0:
                keep = g_planes > 0
                self.n_dropped_planes = int(np.sum(~keep))
                plane_filtered = plane_filtered[keep] if np.any(keep) else np.zeros_like(plane_filtered[:1])
                logger.info(f"MREM robust line search: direction not ascent (plane gradients {g_planes}); "
                            f"dropped {self.n_dropped_planes} plane(s)")

        filtered[:] = np.sum(plane_filtered, axis = 0)

        # back-transformation (Eq. 8): multiplicative correction = filtered * norm / exposure
        exposure = _value(self.summed_exposure_map.contents)
        multiplicative = np.zeros_like(filtered)
        multiplicative[self._exposed] = (filtered * self._norm)[self._exposed] / exposure[self._exposed]

        self._multiplicative = multiplicative
        self.delta_model = self.model.copy()
        self.delta_model[:] = self.model.contents * multiplicative

        if self.mask is not None:
            self.delta_model = self.delta_model.mask_pixels(self.mask)

        logger.debug(f"The delta model was updated. Significant wavelet coefficients: {self.n_significant}, DC offset: {self.dc_offset}")

    def _line_search(self):
        r"""
        Step length :math:`t` along ``self.delta_model`` (step 6). The bound is

        .. math::
            t_{\max} = \min\left( t_a, \; \frac{m}{\max_j |c_j|} \right),

        with :math:`m` = ``max_change`` and :math:`t_a` = ``accel_factor_max``. Without acceleration,
        :math:`t = \min(1, m / \max_j |c_j|)`. Otherwise, with :math:`\Delta e_i = \sum_j R_{ij}\, \delta f_j`,
        up to 10 Newton iterations on :math:`\phi'(t) = 0`,

        .. math::
            t \leftarrow \mathrm{clip}\left( t - \frac{\phi'(t)}{\phi''(t)}, \; 0, \; t_{\max} \right),
            \qquad
            \phi'(t) = \sum_i \Delta e_i \left( \frac{n_i}{e_i + t\,\Delta e_i} - 1 \right), \quad
            \phi''(t) = -\sum_i \frac{n_i\, \Delta e_i^2}{(e_i + t\,\Delta e_i)^2},

        starting from :math:`t = \min(1, t_{\max})` and stopping when :math:`t` changes by less than
        :math:`10^{-3}`. Positivity is ensured by the per-pixel damping in ``post_processing``, not by
        the step length. With ``line_search: "robust"`` the step comes from ``_line_search_robust``.

        Returns
        -------
        float
            The step :math:`t`.
        """

        t_max = self.max_change / max(float(np.max(np.abs(self._multiplicative))), MREM_FLOOR)

        if not self.acceleration_enabled:
            return min(1.0, t_max)

        t_max = min(t_max, self.accel_factor_max)

        delta_expectation_list = self.dataset.calc_source_expectation_list(self.delta_model)

        events = [_value(data.event.contents) for data in self.dataset]
        expectations = [_value(expectation.contents) for expectation in self.expectation_list]
        delta_expectations = [_value(delta_expectation.contents) for delta_expectation in delta_expectation_list]

        if self.line_search == 'robust':
            return self._line_search_robust(events, expectations, delta_expectations, t_max)

        t = min(1.0, t_max)
        for _ in range(10):
            grad, hess = 0.0, 0.0
            for d, mu, dmu in zip(events, expectations, delta_expectations):
                mu_t = np.maximum(mu + t * dmu, MREM_FLOOR)
                grad += np.sum(dmu * (d / mu_t - 1.0))
                hess -= np.sum(dmu**2 * d / mu_t**2)

            if hess >= 0:
                break

            t_new = np.clip(t - grad / hess, 0.0, t_max)

            if abs(t_new - t) < 1e-3:
                t = t_new
                break

            t = t_new

        return float(t)

    @staticmethod
    def _line_derivatives(t, events, expectations, delta_expectations):
        r"""
        First and second derivative of :math:`\phi(t) = \ln L(e + t\,\Delta e)`,

        .. math::
            \phi'(t) = \sum_i \Delta e_i \left( \frac{n_i}{e_i + t\,\Delta e_i} - 1 \right), \qquad
            \phi''(t) = -\sum_i \frac{n_i\, \Delta e_i^2}{(e_i + t\,\Delta e_i)^2},

        summed over the datasets (lists of arrays :math:`n`, :math:`e`, :math:`\Delta e`).
        """
        grad, hess = 0.0, 0.0
        for d, mu, dmu in zip(events, expectations, delta_expectations):
            mu_t = np.maximum(mu + t * dmu, MREM_FLOOR)
            grad += np.sum(dmu * (d / mu_t - 1.0))
            hess -= np.sum(dmu**2 * d / mu_t**2)
        return grad, hess

    def _line_search_robust(self, events, expectations, delta_expectations, t_max, max_iter = 40, rtol = 1e-3):
        r"""
        Safeguarded Newton/bisection search for the maximum of the concave :math:`\phi(t)` on
        :math:`[0, t_{\max}]` (robust line search).

        - If :math:`\phi'(0) \le 0`, the direction is not an ascent direction: returns 0.
        - If :math:`\phi'(t_{\max}) \ge 0`, the maximum is at the bound: returns :math:`t_{\max}`.
        - Otherwise the root of :math:`\phi'` is kept bracketed, :math:`\phi'(t_\mathrm{lo}) > 0 >
          \phi'(t_\mathrm{hi})`. Each iteration takes the Newton step :math:`t - \phi'/\phi''`, or bisects
          when that step leaves the bracket, until :math:`t` or the bracket changes by less than ``rtol``
          (relative).

        An overshooting Newton step therefore falls back to a smaller step instead of ending the stage.

        Returns
        -------
        float
            The step :math:`t \ge 0`.
        """

        g0, _ = self._line_derivatives(0.0, events, expectations, delta_expectations)
        if not g0 > 0:
            return 0.0

        g_max, _ = self._line_derivatives(t_max, events, expectations, delta_expectations)
        if g_max >= 0:
            return float(t_max)

        lo, hi = 0.0, float(t_max)          # g(lo) > 0 > g(hi)
        t = min(1.0, 0.5 * t_max)
        for _ in range(max_iter):
            g, h = self._line_derivatives(t, events, expectations, delta_expectations)
            if g > 0:
                lo = t
            else:
                hi = t
            t_new = t - g / h if h < 0 else 0.5 * (lo + hi)
            if not (lo < t_new < hi):        # Newton step outside the bracket: bisect (smaller step)
                t_new = 0.5 * (lo + hi)
            if abs(t_new - t) <= rtol * max(t, 1e-3) or hi - lo <= rtol * max(lo, 1e-3):
                t = t_new
                break
            t = t_new

        return float(max(t, 0.0))

    def post_processing(self):
        r"""
        Steps 6 and 7: the line search, the damped model update

        .. math::
            f_j \leftarrow f_j \, \max\left( 1 + t\, c_j, \; \rho \right) \qquad (\rho = \texttt{min\_ratio}),

        the ``minimum_flux`` constraint, then ``em_steps`` EM updates of the background normalisations at
        fixed source expectation,

        .. math::
            \beta_k \leftarrow \beta_k \, \frac{\sum_i b_{ki}\, n_i / e_i}{\sum_i b_{ki}},

        each clipped to its range. Finally the E-step (new :math:`e_i`) and the log-likelihood.
        """

        self.accel_factor = self._line_search()

        factor = np.maximum(1.0 + self.accel_factor * self._multiplicative, self.min_ratio)   # damped per pixel
        self.model[:] = self.model.contents * factor
        self._ensure_model_constraints()

        if self.bkg_norm_optimization_enabled:

            self.source_expectation_list = self.dataset.calc_source_expectation_list(self.model)

            for _ in range(self.bkg_norm_em_steps):

                self.bkg_expectation_list = self.dataset.calc_bkg_expectation_list(self.dict_bkg_norm)
                self.expectation_list = self.dataset.combine_expectation_list(self.source_expectation_list, self.bkg_expectation_list)

                ratio_list = [ data.event / expectation for data, expectation in zip(self.dataset, self.expectation_list) ]

                for key in self.dict_bkg_norm.keys():
                    sum_bkg_T_product = self.dataset.calc_summed_bkg_model_product(key, ratio_list)
                    self.dict_bkg_norm[key] = self.dict_bkg_norm[key] * sum_bkg_T_product / self.dict_summed_bkg_model[key]

                self._ensure_bkg_norm_range()

        self.Estep()
        self.log_likelihood_list = self.dataset.calc_log_likelihood_list(self.expectation_list)
        self._stage_iteration += 1

    def register_result(self):
        r"""
        The values below are stored at the end of each iteration.

        - iteration: iteration number
        - stage: index of the progressive-admission stage (0: only the residual plane)
        - finest_scale: FWHM of the finest admitted wavelet plane in deg (None in stage 0)
        - model: updated image
        - background_normalization: optimized background normalization :math:`\beta_k`
        - log-likelihood: log-likelihood
        - accel_factor: step length :math:`t` of the line search
        - n_significant: number of significant (kept) wavelet coefficients
        - dc_offset: the (never thresholded) DC term :math:`d` of the correction, per energy bin
        - n_dropped_planes: planes dropped from the direction by the robust line search (0 otherwise)
        """

        finest = self.wavelet_scales[self.admitted_planes[0]].to_value('deg') if self.admitted_planes[0] < self._n_scales else None

        this_result = {"iteration": self.iteration_count,
                       "stage": self._stage,
                       "finest_scale": finest,
                       "model": self.model.copy(),
                       "background_normalization": self.dict_bkg_norm.copy(),
                       "log-likelihood": self.log_likelihood_list.copy(),
                       "accel_factor": self.accel_factor,
                       "n_significant": self.n_significant,
                       "dc_offset": list(self.dc_offset),
                       "n_dropped_planes": self.n_dropped_planes}

        for key in ["stage", "finest_scale", "background_normalization", "log-likelihood", "accel_factor", "n_significant", "dc_offset"]:
            logger.info(f"{key}: {this_result[key]}")

        self.results.append(this_result)

    def _stage_converged(self):
        r"""
        True if the current stage has converged, i.e. if any of these holds:

        - the stage has used ``max_iterations_per_scale`` iterations;
        - no coefficient is significant and the DC step :math:`\max |d|\, t` is below :math:`10^{-3}`
          (the correction at the admitted scales is structureless, the criterion of the paper);
        - the step :math:`t` is below :math:`10^{-3}` (no likelihood gain along the direction);
        - the gain of :math:`\ln L` was below ``stopping_criteria:threshold`` in each of the last two
          iterations of the stage.
        """

        if self._stage_iteration >= self.max_iterations_per_scale:
            return True

        dc_step = max(abs(dc) for dc in self.dc_offset) * self.accel_factor
        if self.n_significant == 0 and dc_step < 1e-3:
            return True

        if self.accel_factor < 1e-3:
            return True

        stage_results = [r for r in self.results if r["stage"] == self._stage]
        if len(stage_results) >= 3:
            ll = [np.sum(r["log-likelihood"]) for r in stage_results[-3:]]
            if ll[2] - ll[1] < self.stopping_criteria_threshold and ll[1] - ll[0] < self.stopping_criteria_threshold:
                return True

        return False

    def check_stopping_criteria(self):
        r"""
        Advance to the next (finer) scale when the current stage has converged (``_stage_converged``);
        stop after the stage with all planes has converged, or at ``iteration_max``.

        Returns
        -------
        bool
            True if the deconvolution should stop.
        """

        if self.iteration_count >= self.iteration_max:
            return True

        if self._stage_converged():

            if not self.progressive or self._stage >= self._n_scales:
                logger.info("MREM converged: the correction at all scales is structureless.")
                return True

            self._stage += 1
            self._stage_iteration = 0
            logger.info(f"Stage {self._stage}: admitting the {self.wavelet_scales[self.admitted_planes[0]]} scale.")

        return False

    def finalization(self):
        r"""
        Finalization after running the image deconvolution: if ``save_results`` is active, the model
        (``model.hdf5``) and the per-iteration values (``results.fits``: stage, step, number of significant
        coefficients, background normalizations, log-likelihood) are saved.
        """

        if not self.save_results:
            return

        logger.info(f"Saving results in {self.save_results_directory}")

        self._save_standard_results(
            counter_name           = "iteration",
            histogram_keys         = [("model", f"{self.save_results_directory}/model.hdf5", self.save_only_final_result)],
            fits_filename          = f"{self.save_results_directory}/results.fits",
            values_key_name_format = [("stage", "STAGE", "K"), ("accel_factor", "ACCEL_FACTOR", "D"), ("n_significant", "N_SIGNIFICANT", "K")],
            dicts_key_name_format  = [("background_normalization", "BKG_NORM", "D")],
            lists_key_name_format  = [("log-likelihood", "LOG-LIKELIHOOD", "D")],
        )
