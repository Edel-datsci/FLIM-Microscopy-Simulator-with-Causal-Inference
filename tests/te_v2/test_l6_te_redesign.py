"""Tests for L6 v2 — Conditional Transfer Entropy with Ensemble Trial Design.

Original 8 tests (preserved) + 4 new tests for:
  - Wibral asymmetry lag criterion
  - ADF diagnostic integration
  - lag_criterion="max_cte" alternative
  - Detrending with diff2 in pipeline
"""

import unittest
import numpy as np

from l6_te_redesign import (
    discretize_uniform,
    transfer_entropy_discrete,
    conditional_te_discrete,
    local_te_discrete,
    preprocess_trials,
    detrend_series,
    trial_shuffle_surrogates,
    ensemble_conditional_te_discrete,
    conditional_te_bidirectional_ensemble,
    select_lag_by_conditional_te,
    CausalPairSpecV2,
    compute_causal_flim_generic_v2,
    _adf_diagnostic_trials,
)


# ═══════════════════════════════════════════════════════════════════════
# Data generators
# ═══════════════════════════════════════════════════════════════════════

def make_direct_driver(n_trials=40, T=120, noise=0.05, seed=0):
    """X(t) directly drives Y(t+1) with small noise."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=(n_trials, T))
    y = np.zeros_like(x)
    for r in range(n_trials):
        for t in range(1, T):
            if rng.random() < noise:
                y[r, t] = 1 - x[r, t - 1]
            else:
                y[r, t] = x[r, t - 1]
    return x.astype(float), y.astype(float)


def make_common_driver(n_trials=40, T=120, noise=0.05, seed=1):
    """Z drives both X and Y with no direct X→Y link."""
    rng = np.random.default_rng(seed)
    z = np.zeros((n_trials, T), dtype=int)
    x = np.zeros_like(z)
    y = np.zeros_like(z)
    z[:, 0] = rng.integers(0, 2, size=n_trials)
    for r in range(n_trials):
        for t in range(1, T):
            z[r, t] = z[r, t - 1] if rng.random() > 0.1 else 1 - z[r, t - 1]
        for t in range(T - 1):
            x[r, t] = z[r, t] if rng.random() > noise else 1 - z[r, t]
            y[r, t + 1] = z[r, t] if rng.random() > noise else 1 - z[r, t]
    x[:, -1] = x[:, -2]
    return x.astype(float), y.astype(float), z.astype(float)


def make_mediated_chain(n_trials=40, T=120, noise=0.05, seed=2):
    """X→Z→Y chain: conditioning on Z should remove X→Y."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=(n_trials, T))
    z = np.zeros_like(x)
    y = np.zeros_like(x)
    for r in range(n_trials):
        for t in range(T - 1):
            z[r, t] = x[r, t] if rng.random() > noise else 1 - x[r, t]
            y[r, t + 1] = z[r, t] if rng.random() > noise else 1 - z[r, t]
        z[r, -1] = z[r, -2]
    return x.astype(float), y.astype(float), z.astype(float)


def make_delayed_driver(n_trials=30, T=150, lag_true=5, noise=0.05, seed=3):
    """X drives Y with a specific lag (for lag selection tests)."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=(n_trials, T))
    y = np.zeros_like(x)
    for r in range(n_trials):
        for t in range(lag_true, T):
            if rng.random() < noise:
                y[r, t] = 1 - x[r, t - lag_true]
            else:
                y[r, t] = x[r, t - lag_true]
    return x.astype(float), y.astype(float)


# ═══════════════════════════════════════════════════════════════════════
# Original 8 tests (preserved exactly)
# ═══════════════════════════════════════════════════════════════════════

class TestL6TeRedesignOriginal(unittest.TestCase):

    def test_01_discretize_constant(self):
        x = np.ones(10)
        bins = discretize_uniform(x, n_bins=4)
        self.assertTrue(np.all(bins == 0))

    def test_02_local_te_average_matches_global(self):
        x, y = make_direct_driver(n_trials=1, T=200, noise=0.0, seed=4)
        xd = discretize_uniform(x[0], 2)
        yd = discretize_uniform(y[0], 2)
        global_te = transfer_entropy_discrete(xd, yd, lag=1)
        local = local_te_discrete(xd, yd, lag=1)
        self.assertAlmostEqual(global_te, float(local.mean()), places=10)

    def test_03_conditional_equals_unconditional_without_conditioner(self):
        x, y = make_direct_driver(n_trials=1, T=200, noise=0.05, seed=5)
        xd = discretize_uniform(x[0], 2)
        yd = discretize_uniform(y[0], 2)
        te = transfer_entropy_discrete(xd, yd, lag=1)
        cte = conditional_te_discrete(xd, yd, lag=1, cond_d=None)
        self.assertAlmostEqual(te, cte, places=12)

    def test_04_trial_shuffle_preserves_trial_shape(self):
        x, _ = make_direct_driver(n_trials=8, T=50, noise=0.05, seed=6)
        xs = trial_shuffle_surrogates(x, np.random.default_rng(0))
        self.assertEqual(xs.shape, x.shape)
        original = {tuple(row) for row in x}
        self.assertTrue(all(tuple(row) in original for row in xs))

    def test_05_ensemble_direct_driver_detected(self):
        x, y = make_direct_driver(n_trials=40, T=120, noise=0.05, seed=7)
        res = conditional_te_bidirectional_ensemble(
            x, y, lag=1, n_bins=2, n_surrogates=50, random_state=1)
        self.assertGreater(res.te_xy_bits, res.te_yx_bits)
        self.assertTrue(res.sig_xy.significant)

    def test_06_conditional_te_removes_common_driver(self):
        x, y, z = make_common_driver(n_trials=60, T=150, noise=0.02, seed=8)
        bivar = conditional_te_bidirectional_ensemble(
            x, y, lag=1, n_bins=2, n_surrogates=40, random_state=2)
        cond = conditional_te_bidirectional_ensemble(
            x, y, lag=1, n_bins=2, cond_trials=z,
            n_surrogates=40, random_state=2)
        self.assertGreater(bivar.te_xy_bits, 0.05)
        self.assertLess(cond.te_xy_bits, bivar.te_xy_bits * 0.5)

    def test_07_conditional_te_removes_mediated_effect(self):
        x, y, z = make_mediated_chain(n_trials=60, T=150, noise=0.02, seed=9)
        bivar = conditional_te_bidirectional_ensemble(
            x, y, lag=1, n_bins=2, n_surrogates=40, random_state=3)
        cond = conditional_te_bidirectional_ensemble(
            x, y, lag=1, n_bins=2, cond_trials=z,
            n_surrogates=40, random_state=3)
        self.assertGreater(bivar.te_xy_bits, 0.05)
        self.assertLess(cond.te_xy_bits, bivar.te_xy_bits * 0.5)

    def test_08_compute_causal_flim_generic_v2_smoke(self):
        x, y, z = make_common_driver(n_trials=20, T=80, noise=0.05, seed=10)
        series = {"x": x, "y": y, "z": z}
        specs = [CausalPairSpecV2(
            source_key="x", target_key="y",
            conditioning_keys=("z",),
            candidate_lags=(1, 2), n_bins=2, n_surrogates=20)]
        res = compute_causal_flim_generic_v2(series, specs)
        self.assertEqual(len(res), 1)
        self.assertIn(res[0].te_result.lag, (1, 2))
        self.assertIn(1, res[0].diagnostics["lag_scores"])


# ═══════════════════════════════════════════════════════════════════════
# New tests for v2 fixes
# ═══════════════════════════════════════════════════════════════════════

class TestL6TeRedesignV2Fixes(unittest.TestCase):

    def test_09_wibral_asymmetry_lag_selects_true_lag(self):
        """Wibral asymmetry criterion should find lag=5 for delayed driver."""
        x, y = make_delayed_driver(n_trials=30, T=150, lag_true=5, seed=20)
        xd = np.array([discretize_uniform(t, 2) for t in x])
        yd = np.array([discretize_uniform(t, 2) for t in y])
        lag, scores = select_lag_by_conditional_te(
            xd, yd, candidate_lags=(1, 2, 5, 10), criterion="asymmetry")
        # True lag=5 should have highest asymmetry
        self.assertEqual(lag, 5,
                         f"Expected lag=5, got lag={lag}. Scores: {scores}")

    def test_10_max_cte_lag_also_finds_delayed_driver(self):
        """max_cte criterion should also find lag=5."""
        x, y = make_delayed_driver(n_trials=30, T=150, lag_true=5, seed=21)
        xd = np.array([discretize_uniform(t, 2) for t in x])
        yd = np.array([discretize_uniform(t, 2) for t in y])
        lag, scores = select_lag_by_conditional_te(
            xd, yd, candidate_lags=(1, 2, 5, 10), criterion="max_cte")
        self.assertEqual(lag, 5,
                         f"Expected lag=5, got lag={lag}. Scores: {scores}")

    def test_11_adf_diagnostic_returns_dict(self):
        """ADF diagnostic should return info dict without crashing."""
        x, y = make_direct_driver(n_trials=5, T=100, seed=22)
        info = _adf_diagnostic_trials(x, y)
        self.assertIn("available", info)
        self.assertIn("warning", info)
        # Should have run (statsmodels typically available)
        if info["available"]:
            self.assertIn("x_adf_pvalue", info)
            self.assertIn("y_adf_pvalue", info)

    def test_12_pipeline_v2_includes_adf_diagnostics(self):
        """compute_causal_flim_generic_v2 should include ADF in diagnostics."""
        x, y = make_direct_driver(n_trials=20, T=80, seed=23)
        series = {"x": x, "y": y}
        specs = [CausalPairSpecV2(
            source_key="x", target_key="y",
            candidate_lags=(1,), n_bins=2, n_surrogates=10)]
        res = compute_causal_flim_generic_v2(series, specs)
        self.assertIn("adf", res[0].diagnostics)
        self.assertIsNotNone(res[0].te_result.adf_diagnostics)


# ═══════════════════════════════════════════════════════════════════════
# Edge case and robustness tests
# ═══════════════════════════════════════════════════════════════════════

class TestL6TeEdgeCases(unittest.TestCase):

    def test_13_discretize_empty(self):
        bins = discretize_uniform(np.array([]), 4)
        self.assertEqual(len(bins), 0)

    def test_14_discretize_single_value(self):
        bins = discretize_uniform(np.array([42.0]), 4)
        self.assertEqual(bins[0], 0)

    def test_15_discretize_two_values(self):
        bins = discretize_uniform(np.array([0.0, 1.0]), 2)
        self.assertEqual(bins[0], 0)
        self.assertEqual(bins[1], 1)

    def test_16_detrend_diff1(self):
        x = np.array([1.0, 3.0, 6.0, 10.0])
        d = detrend_series(x, "diff1")
        np.testing.assert_array_almost_equal(d, [2.0, 3.0, 4.0])

    def test_17_detrend_diff2(self):
        x = np.array([1.0, 3.0, 6.0, 10.0])
        d = detrend_series(x, "diff2")
        np.testing.assert_array_almost_equal(d, [1.0, 1.0])

    def test_18_detrend_none_copies(self):
        x = np.array([1.0, 2.0, 3.0])
        d = detrend_series(x, "none")
        np.testing.assert_array_equal(d, x)
        d[0] = 999
        self.assertEqual(x[0], 1.0)  # original unchanged

    def test_19_preprocess_trials_shape(self):
        trials = np.random.default_rng(0).random((5, 20))
        result = preprocess_trials(trials, "diff1")
        self.assertEqual(result.shape, (5, 19))

    def test_20_conditional_te_with_lag_boundary(self):
        """lag = N-1 should raise ValueError."""
        xd = np.array([0, 1, 0, 1, 0], dtype=np.int64)
        yd = np.array([1, 0, 1, 0, 1], dtype=np.int64)
        with self.assertRaises(ValueError):
            conditional_te_discrete(xd, yd, lag=5)

    def test_21_trial_shuffle_requires_2_trials(self):
        x = np.array([[1, 2, 3]], dtype=float)
        with self.assertRaises(ValueError):
            trial_shuffle_surrogates(x, np.random.default_rng(0))

    def test_22_independent_series_te_near_zero(self):
        """Two independent random series should have TE ≈ 0."""
        rng = np.random.default_rng(42)
        x = rng.integers(0, 4, size=(30, 100)).astype(float)
        y = rng.integers(0, 4, size=(30, 100)).astype(float)
        res = conditional_te_bidirectional_ensemble(
            x, y, lag=1, n_bins=4, n_surrogates=30, random_state=0)
        # After bias correction, should be very small
        self.assertLess(res.te_xy_bits, 0.05)
        self.assertLess(res.te_yx_bits, 0.05)

    def test_23_pipeline_with_asymmetry_criterion(self):
        """Pipeline should accept lag_criterion='asymmetry'."""
        x, y = make_direct_driver(n_trials=20, T=80, seed=30)
        series = {"x": x, "y": y}
        specs = [CausalPairSpecV2(
            source_key="x", target_key="y",
            candidate_lags=(1, 2), n_bins=2, n_surrogates=10,
            lag_criterion="asymmetry")]
        res = compute_causal_flim_generic_v2(series, specs)
        self.assertEqual(len(res), 1)

    def test_24_pipeline_with_max_cte_criterion(self):
        """Pipeline should accept lag_criterion='max_cte'."""
        x, y = make_direct_driver(n_trials=20, T=80, seed=31)
        series = {"x": x, "y": y}
        specs = [CausalPairSpecV2(
            source_key="x", target_key="y",
            candidate_lags=(1, 2), n_bins=2, n_surrogates=10,
            lag_criterion="max_cte")]
        res = compute_causal_flim_generic_v2(series, specs)
        self.assertEqual(len(res), 1)

    def test_25_pipeline_with_diff2_detrend(self):
        """Pipeline should work with diff2 detrending."""
        rng = np.random.default_rng(32)
        # Logistic-like growth + noise
        t = np.linspace(0, 5, 80)
        x = np.array([1.0 / (1.0 + np.exp(-t + 2.5 + rng.normal(0, 0.3)))
                       for _ in range(15)])
        y = np.array([1.0 / (1.0 + np.exp(-t + 3.0 + rng.normal(0, 0.3)))
                       for _ in range(15)])
        series = {"x": x, "y": y}
        specs = [CausalPairSpecV2(
            source_key="x", target_key="y",
            candidate_lags=(1, 2), n_bins=3, n_surrogates=10,
            detrend_method="diff2")]
        res = compute_causal_flim_generic_v2(series, specs)
        self.assertEqual(len(res), 1)
        # After diff2 on logistic, series should be short but valid
        self.assertGreaterEqual(res[0].te_result.lag, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
