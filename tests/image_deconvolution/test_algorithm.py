"""
Integration tests for image deconvolution algorithms.
Each class groups tests for one algorithm. Within a class, tests share
a common parameter template via a fixture, and each test method exercises
a specific scenario (e.g. with/without background normalization, different
stopping criteria). Numerical values are regression checks: they should
only be updated intentionally when the algorithm changes.
"""

import pytest
import numpy as np
import astropy.units as u
from yayc import Configurator

from cosipy.image_deconvolution import (
    RichardsonLucyBasic,
    RichardsonLucy,
    RichardsonLucyAdvanced,
    MAP_RichardsonLucy,
    MREM,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run(algorithm, num_iteration):
    """Initialize and iterate an algorithm. Returns the algorithm instance."""
    algorithm.initialization()
    for _ in range(num_iteration):
        if algorithm.iteration():
            break
    algorithm.finalization()
    return algorithm


# ---------------------------------------------------------------------------
# RichardsonLucyBasic
# ---------------------------------------------------------------------------

class TestRichardsonLucyBasic:

    @pytest.fixture
    def parameter(self, tmp_path):
        return Configurator({
            "iteration_max": 3,
            "minimum_flux": {"value": 0.0, "unit": "cm-2 s-1 sr-1"},
            "save_results": {"activate": True, "directory": str(tmp_path), "only_final_result": True},
        })

    def test_basic(self, dataset, model, mask, parameter):
        alg = RichardsonLucyBasic(model, dataset, mask, parameter)
        run(alg, parameter["iteration_max"])


# ---------------------------------------------------------------------------
# RichardsonLucy
# ---------------------------------------------------------------------------

class TestRichardsonLucy:

    @pytest.fixture
    def parameter(self, tmp_path):
        return Configurator({
            "iteration_max": 3,
            "minimum_flux": {"value": 0.0, "unit": "cm-2 s-1 sr-1"},
            "background_normalization_optimization": {"activate": True, "range": {"bkg": [0.9, 1.1]}},
            "save_results": {"activate": True, "directory": str(tmp_path), "only_final_result": True},
        })

    def test_basic(self, dataset, model, mask, parameter):
        alg = RichardsonLucy(model, dataset, mask, parameter)
        run(alg, parameter["iteration_max"])


# ---------------------------------------------------------------------------
# RichardsonLucyAdvanced
# ---------------------------------------------------------------------------

class TestRichardsonLucyAdvanced:
    """
    Tests for RichardsonLucyAdvanced covering acceleration on/off.
    Numerical assertions are regression checks.
    """

    NUM_ITERATION = 3

    @pytest.fixture
    def base_parameter(self, tmp_path):
        return Configurator({
            "iteration_max": self.NUM_ITERATION,
            "minimum_flux": {"value": 0.0, "unit": "cm-2 s-1 sr-1"},
            "acceleration": {"activate": True, "alpha_max": 10.0},
            "response_weighting": {"activate": True, "index": 0.5},
            "smoothing": {"activate": True, "FWHM": {"value": 2.0, "unit": "deg"}},
            "background_normalization_optimization": {"activate": True, "range": {"bkg": [0.9, 1.1]}},
            "save_results": {"activate": True, "directory": str(tmp_path), "only_final_result": True},
        })

    def test_with_acceleration(self, dataset, model, mask, base_parameter):
        alg = run(RichardsonLucyAdvanced(model, dataset, mask, base_parameter), self.NUM_ITERATION)
        # Regression check
        assert np.isclose(alg.results[-1]['log-likelihood'][0], 5495.120521335304)

    def test_without_acceleration(self, dataset, model, mask, base_parameter):
        base_parameter["acceleration:activate"] = False
        alg = run(RichardsonLucyAdvanced(model, dataset, mask, base_parameter), self.NUM_ITERATION)
        # Regression check
        assert np.isclose(alg.results[-1]['log-likelihood'][0], 5270.562770130176)


# ---------------------------------------------------------------------------
# MAP_RichardsonLucy
# ---------------------------------------------------------------------------

class TestMAPRichardsonLucy:
    """
    Tests for MAP_RichardsonLucy covering different prior configurations
    and stopping criteria. Numerical assertions are regression checks.
    """

    NUM_ITERATION = 10

    @pytest.fixture
    def base_parameter(self, tmp_path):
        return Configurator({
            "iteration_max": self.NUM_ITERATION,
            "minimum_flux": {"value": 0.0, "unit": "cm-2 s-1 sr-1"},
            "response_weighting": {"activate": True, "index": 0.5},
            "background_normalization_optimization": {"activate": True, "range": {"bkg": [0.9, 1.1]}},
            "stopping_criteria": {"statistics": "log-posterior", "threshold": 1e-2},
            "prior": {
                "TSV": {"coefficient": 1e-10},
                "gamma": {
                    "model":      {"theta": {"value": np.inf, "unit": "cm-2 s-1 sr-1"}, "k": {"value": 0.999}},
                    "background": {"theta": {"value": np.inf}, "k": {"value": 1.0}},
                },
            },
            "save_results": {"activate": True, "directory": str(tmp_path), "only_final_result": True},
        })

    def test_with_gamma_and_tsv_prior(self, dataset, model, mask, base_parameter):
        """Full setup: TSV + gamma prior, bkg optimization enabled."""
        alg = run(MAP_RichardsonLucy(model, dataset, mask, base_parameter), self.NUM_ITERATION)
        # Regression check
        assert np.isclose(alg.results[-1]['log-posterior'], 6567.857548203495)

    def test_without_bkg_optimization(self, dataset, model, mask, base_parameter):
        """Background normalization fixed at 1.0."""
        base_parameter["background_normalization_optimization:activate"] = False
        alg = run(MAP_RichardsonLucy(model, dataset, mask, base_parameter), self.NUM_ITERATION)
        # Regression check
        assert np.isclose(alg.results[-1]['log-posterior'], 6202.336733778631)

    def test_stopping_criteria_threshold(self, dataset, model, mask, base_parameter):
        """Large threshold causes early stopping after 2 iterations."""
        base_parameter["stopping_criteria:threshold"] = 1e10
        alg = run(MAP_RichardsonLucy(model, dataset, mask, base_parameter), self.NUM_ITERATION)
        assert len(alg.results) == 2

    def test_stopping_criteria_log_likelihood(self, dataset, model, mask, base_parameter):
        """Use log-likelihood instead of log-posterior as stopping criterion."""
        base_parameter["background_normalization_optimization:activate"] = False
        base_parameter["stopping_criteria:threshold"] = 1e10
        base_parameter["stopping_criteria:statistics"] = "log-likelihood"
        alg = run(MAP_RichardsonLucy(model, dataset, mask, base_parameter), self.NUM_ITERATION)
        # Regression check
        assert np.isclose(alg.results[-1]['log-likelihood'][0], 3931.3528198012773)

    def test_without_gamma_prior(self, dataset, model, mask, base_parameter):
        """TSV prior only, no gamma prior."""
        base_parameter["background_normalization_optimization:activate"] = False
        base_parameter["stopping_criteria:statistics"] = "log-posterior"
        base_parameter["stopping_criteria:threshold"] = 1e10
        base_parameter["prior"] = {"TSV": {"coefficient": 1e-10}}
        alg = run(MAP_RichardsonLucy(model, dataset, mask, base_parameter), self.NUM_ITERATION)
        # Regression check
        assert np.isclose(alg.results[-1]['log-posterior'], 3931.29811442966)

    def test_invalid_stopping_statistics_raises(self, dataset, model, mask, base_parameter):
        """Invalid stopping_criteria:statistics should raise ValueError."""
        base_parameter["stopping_criteria:statistics"] = "likelihooooooooood!!!"
        with pytest.raises(ValueError):
            MAP_RichardsonLucy(model, dataset, mask, base_parameter)


# ---------------------------------------------------------------------------
# MREM
# ---------------------------------------------------------------------------

class TestMREM:
    """
    Tests for MREM (multi-resolution EM).
    """

    NUM_ITERATION = 5

    @pytest.fixture
    def base_parameter(self, tmp_path):
        return Configurator({
            "iteration_max": self.NUM_ITERATION,
            "minimum_flux": {"value": 0.0, "unit": "cm-2 s-1 sr-1"},
            "wavelet": {"scales": {"value": [10.0, 20.0, 40.0], "unit": "deg"}, "significance": 1.0,
                        "noise_realizations": 4, "max_iterations_per_scale": 3, "normalization": "fisher", "progressive": True},
            "acceleration": {"activate": True, "accel_factor_max": 10.0},
            "background_normalization_optimization": {"activate": True, "range": {"bkg": [0.9, 1.1]}, "em_steps": 2},
            "stopping_criteria": {"threshold": 1e-10},
            "save_results": {"activate": True, "directory": str(tmp_path), "only_final_result": True},
        })

    def test_wavelet_decomposition_is_exact(self):
        """The wavelet planes and the coarse residual add up to the input map."""
        rng = np.random.default_rng(0)
        image = rng.normal(size = 12 * 8**2)
        planes, coarse = MREM.wavelet_decomposition(image, np.array([10.0, 20.0]) * u.deg)
        assert len(planes) == 2
        assert np.allclose(sum(planes) + coarse, image)

    def test_likelihood_does_not_decrease(self, dataset, model, mask, base_parameter, tmp_path):
        alg = run(MREM(model, dataset, mask, base_parameter), self.NUM_ITERATION)
        log_likelihood = [np.sum(r['log-likelihood']) for r in alg.results]
        assert np.all(np.diff(log_likelihood) >= -1e-6)
        assert np.all(alg.results[-1]['model'].contents.value >= 0)
        assert (tmp_path / "results.fits").exists()

    def test_progressive_scales(self, dataset, model, mask, base_parameter):
        """Stages are admitted from the residual plane down to the finest scale, never skipping."""
        alg = run(MREM(model, dataset, mask, base_parameter), 20)
        stages = [r['stage'] for r in alg.results]
        assert stages[0] == 0 and stages == sorted(stages) and max(np.diff(stages)) <= 1
        assert alg.results[-1]['finest_scale'] == 10.0 or alg.results[-1]['stage'] < 3

    def test_without_acceleration(self, dataset, model, mask, base_parameter):
        base_parameter["acceleration:activate"] = False
        alg = run(MREM(model, dataset, mask, base_parameter), self.NUM_ITERATION)
        assert all(r['accel_factor'] <= 1.0 for r in alg.results)

    def test_robust_line_search(self, dataset, model, mask, base_parameter):
        """Opt-in robust line search: non-negative steps, likelihood never decreases, default unchanged."""
        base_parameter["acceleration"] = {"activate": True, "accel_factor_max": 10.0, "line_search": "robust"}
        alg = run(MREM(model, dataset, mask, base_parameter), self.NUM_ITERATION)
        log_likelihood = [np.sum(r['log-likelihood']) for r in alg.results]
        assert np.all(np.diff(log_likelihood) >= -1e-6)
        assert all(r['accel_factor'] >= 0 for r in alg.results)
        assert all('n_dropped_planes' in r for r in alg.results)
        base_parameter["acceleration"] = {"activate": True, "line_search": "nonsense"}
        with pytest.raises(ValueError):
            MREM(model, dataset, mask, base_parameter)

    def test_high_significance_keeps_only_dc(self, dataset, model, mask, base_parameter):
        base_parameter["wavelet:significance"] = 1e30
        alg = run(MREM(model, dataset, mask, base_parameter), 2)
        assert all(r['n_significant'] == 0 for r in alg.results)

    def test_exposure_normalisation(self, dataset, model, mask, base_parameter):
        base_parameter["wavelet:normalization"] = "exposure"
        alg = run(MREM(model, dataset, mask, base_parameter), 2)
        assert len(alg.results) == 2

    def test_noise_has_unit_variance(self, dataset, model, mask, base_parameter):
        """With the Fisher normalisation the correction map of a Poisson realisation has unit variance."""
        alg = MREM(model, dataset, mask, base_parameter)
        alg.initialization()
        alg.Estep()
        alg._update_normalisation()
        rng = np.random.default_rng(0)
        maps = []
        for _ in range(30):
            ratio_list = [rng.poisson(np.asarray(e.contents)) / np.maximum(np.asarray(e.contents), 1e-30) for e in alg.expectation_list]
            maps.append(alg._normalised_correction(ratio_list))
        var = np.var(np.array(maps), axis = 0, ddof = 1)[alg._exposed]
        assert 0.6 < np.median(var) < 1.4

    def test_invalid_normalization_raises(self, dataset, model, mask, base_parameter):
        base_parameter["wavelet:normalization"] = "something"
        with pytest.raises(ValueError):
            MREM(model, dataset, mask, base_parameter)
