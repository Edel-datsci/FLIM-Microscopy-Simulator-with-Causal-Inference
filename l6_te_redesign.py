"""L6 v2 — Conditional Transfer Entropy with Ensemble Trial Design.

Upgrades bivariante TE to CTE(X→Y|C) = I(Y_future ; X_past | Y_past, C_past).
This resolves confounding by conditioning on observed common drivers / mediators.

Surrogate strategy: trial-shuffle (Wollstadt 2014) preserves within-trial
temporal structure while destroying inter-trial X↔Y coupling.

References
----------
Schreiber T (2000) Phys Rev Lett 85:461 — TE definition
Frenzel S, Pompe B (2007) Phys Rev Lett 99:204101 — CMI / CTE
Wollstadt P et al. (2014) PLoS ONE 9:e102833 — Ensemble / trial-shuffle
Lizier JT et al. (2008) Phys Rev E 77:026110 — Local TE
Wibral M et al. (2013) PLoS Comput Biol 9:e1003196 — Lag selection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Dict, Any
import math
import warnings
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SignificanceResult:
    """Result of surrogate-based significance test."""
    observed_bits: float
    null_bits: np.ndarray
    p_value: float
    significant: bool
    effect_bits: float          # bias-corrected: max(observed - mean_null, 0)
    alpha: float = 0.05


@dataclass
class TransferEntropyResultV2:
    """Bidirectional CTE result with significance."""
    te_xy_bits: float
    te_yx_bits: float
    lag: int
    conditioning_keys: Tuple[str, ...] = field(default_factory=tuple)
    sig_xy: Optional[SignificanceResult] = None
    sig_yx: Optional[SignificanceResult] = None
    estimator: str = "discrete"
    notes: Tuple[str, ...] = field(default_factory=tuple)
    adf_diagnostics: Optional[Dict[str, Any]] = None  # optional stationarity info


@dataclass
class CausalPairSpecV2:
    """Specification for one causal pair to test."""
    source_key: str
    target_key: str
    conditioning_keys: Tuple[str, ...] = field(default_factory=tuple)
    candidate_lags: Tuple[int, ...] = (1, 2, 5, 10, 20)
    detrend_method: str = "none"
    n_bins: int = 4
    n_surrogates: int = 200
    surrogate_mode: str = "trial_shuffle"
    lag_criterion: str = "asymmetry"  # "asymmetry" (Wibral) or "max_cte"


@dataclass
class CausalPairResultV2:
    """Result for one causal pair."""
    spec: CausalPairSpecV2
    te_result: TransferEntropyResultV2
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ----------------------------
# Preprocessing
# ----------------------------

def detrend_series(x: np.ndarray, method: str = "none") -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if method == "none":
        return x.copy()
    if method == "diff1":
        if x.size < 2:
            raise ValueError("diff1 requires at least 2 samples")
        return np.diff(x)
    if method == "diff2":
        if x.size < 3:
            raise ValueError("diff2 requires at least 3 samples")
        return np.diff(np.diff(x))
    raise ValueError(f"Unknown detrend method: {method}")


def preprocess_trials(trials: np.ndarray, method: str = "none") -> np.ndarray:
    trials = np.asarray(trials, dtype=float)
    if trials.ndim != 2:
        raise ValueError("trials must be 2D: (n_trials, n_time)")
    processed = [detrend_series(trial, method) for trial in trials]
    lengths = {len(p) for p in processed}
    if len(lengths) != 1:
        raise ValueError("All trials must have same length after detrending")
    return np.asarray(processed)


def discretize_uniform(x: np.ndarray, n_bins: int = 4) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.asarray([], dtype=np.int64)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError("Non-finite values in series")
    if abs(x_max - x_min) < 1e-30:
        return np.zeros_like(x, dtype=np.int64)
    scaled = (x - x_min) / (x_max - x_min)
    bins = np.floor(scaled * n_bins).astype(np.int64)
    bins[bins == n_bins] = n_bins - 1
    bins = np.clip(bins, 0, n_bins - 1)
    return bins


# ----------------------------
# Core discrete TE / CTE
# ----------------------------

def _iter_cond_rows(cond: Optional[np.ndarray], idx: int) -> Tuple[int, ...]:
    if cond is None:
        return ()
    row = cond[idx]
    if row.ndim == 0:
        return (int(row),)
    return tuple(int(v) for v in row)


def conditional_te_discrete(
    x_d: np.ndarray,
    y_d: np.ndarray,
    lag: int = 1,
    cond_d: Optional[np.ndarray] = None,
) -> float:
    """Discrete conditional TE in bits.

    Computes I(Y_future ; X_past | Y_past, C_past).
    cond_d is shape (N,) or (N, n_cond), assumed aligned with x_d/y_d.
    """
    x_d = np.asarray(x_d, dtype=np.int64)
    y_d = np.asarray(y_d, dtype=np.int64)
    if x_d.shape != y_d.shape:
        raise ValueError("x_d and y_d must have same shape")
    if x_d.ndim != 1:
        raise ValueError("x_d and y_d must be 1D")
    if lag < 1 or lag >= len(x_d):
        raise ValueError("lag must satisfy 1 <= lag < N")

    if cond_d is not None:
        cond_d = np.asarray(cond_d, dtype=np.int64)
        if cond_d.shape[0] != x_d.shape[0]:
            raise ValueError("cond_d must have same length as x_d")
        if cond_d.ndim == 1:
            cond_d = cond_d[:, None]
        elif cond_d.ndim != 2:
            raise ValueError("cond_d must be 1D or 2D")

    n_eff = len(x_d) - lag
    c_joint: Dict[Tuple[int, int, int, Tuple[int, ...]], int] = {}
    c_yxc: Dict[Tuple[int, int, Tuple[int, ...]], int] = {}
    c_yyc: Dict[Tuple[int, int, Tuple[int, ...]], int] = {}
    c_yc: Dict[Tuple[int, Tuple[int, ...]], int] = {}

    for t in range(n_eff):
        yf = int(y_d[t + lag])
        yp = int(y_d[t])
        xp = int(x_d[t])
        cp = _iter_cond_rows(cond_d, t)
        k_joint = (yf, yp, xp, cp)
        k_yxc = (yp, xp, cp)
        k_yyc = (yf, yp, cp)
        k_yc = (yp, cp)
        c_joint[k_joint] = c_joint.get(k_joint, 0) + 1
        c_yxc[k_yxc] = c_yxc.get(k_yxc, 0) + 1
        c_yyc[k_yyc] = c_yyc.get(k_yyc, 0) + 1
        c_yc[k_yc] = c_yc.get(k_yc, 0) + 1

    te = 0.0
    n_eff_f = float(n_eff)
    for (yf, yp, xp, cp), count in c_joint.items():
        p_joint = count / n_eff_f
        p_yc = c_yc[(yp, cp)] / n_eff_f
        p_yxc = c_yxc[(yp, xp, cp)] / n_eff_f
        p_yyc = c_yyc[(yf, yp, cp)] / n_eff_f
        if p_joint <= 0.0 or p_yc <= 0.0 or p_yxc <= 0.0 or p_yyc <= 0.0:
            continue
        ratio = (p_joint * p_yc) / (p_yxc * p_yyc)
        te += p_joint * math.log2(ratio)

    return max(te, 0.0)


def transfer_entropy_discrete(x_d: np.ndarray, y_d: np.ndarray, lag: int = 1) -> float:
    return conditional_te_discrete(x_d=x_d, y_d=y_d, lag=lag, cond_d=None)


def local_te_discrete(
    x_d: np.ndarray,
    y_d: np.ndarray,
    lag: int = 1,
    cond_d: Optional[np.ndarray] = None,
) -> np.ndarray:
    x_d = np.asarray(x_d, dtype=np.int64)
    y_d = np.asarray(y_d, dtype=np.int64)
    if cond_d is not None:
        cond_d = np.asarray(cond_d, dtype=np.int64)
        if cond_d.ndim == 1:
            cond_d = cond_d[:, None]
    n_eff = len(x_d) - lag
    vals = np.zeros(n_eff, dtype=float)

    # build counts once
    c_joint: Dict[Tuple[int, int, int, Tuple[int, ...]], int] = {}
    c_yxc: Dict[Tuple[int, int, Tuple[int, ...]], int] = {}
    c_yyc: Dict[Tuple[int, int, Tuple[int, ...]], int] = {}
    c_yc: Dict[Tuple[int, Tuple[int, ...]], int] = {}
    for t in range(n_eff):
        yf = int(y_d[t + lag]); yp = int(y_d[t]); xp = int(x_d[t]); cp = _iter_cond_rows(cond_d, t)
        c_joint[(yf, yp, xp, cp)] = c_joint.get((yf, yp, xp, cp), 0) + 1
        c_yxc[(yp, xp, cp)] = c_yxc.get((yp, xp, cp), 0) + 1
        c_yyc[(yf, yp, cp)] = c_yyc.get((yf, yp, cp), 0) + 1
        c_yc[(yp, cp)] = c_yc.get((yp, cp), 0) + 1

    n_eff_f = float(n_eff)
    for t in range(n_eff):
        yf = int(y_d[t + lag]); yp = int(y_d[t]); xp = int(x_d[t]); cp = _iter_cond_rows(cond_d, t)
        ratio = (
            (c_joint[(yf, yp, xp, cp)] / n_eff_f) * (c_yc[(yp, cp)] / n_eff_f)
        ) / (
            (c_yxc[(yp, xp, cp)] / n_eff_f) * (c_yyc[(yf, yp, cp)] / n_eff_f)
        )
        vals[t] = math.log2(ratio)
    return vals


# ----------------------------
# Ensemble and significance
# ----------------------------

def flatten_trials(trials: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    trials = np.asarray(trials)
    if trials.ndim != 2:
        raise ValueError("trials must be 2D")
    past = []
    future = []
    for trial in trials:
        past.append(trial[:-lag])
        future.append(trial[lag:])
    return np.concatenate(past), np.concatenate(future)


def ensemble_conditional_te_discrete(
    x_trials_d: np.ndarray,
    y_trials_d: np.ndarray,
    lag: int = 1,
    cond_trials_d: Optional[np.ndarray] = None,
) -> float:
    """Compute CTE over pooled trials, avoiding cross-trial leakage.

    Iterates trial-by-trial to build joint histogram, ensuring
    y_past and y_future always come from the same trial.

    Reference: Wollstadt et al. (2014) PLoS ONE 9:e102833, §2.2
    """
    x_trials_d = np.asarray(x_trials_d, dtype=np.int64)
    y_trials_d = np.asarray(y_trials_d, dtype=np.int64)
    if x_trials_d.shape != y_trials_d.shape:
        raise ValueError("x_trials_d and y_trials_d must have same shape")
    if x_trials_d.ndim != 2:
        raise ValueError("trials must be 2D")
    n_trials, T = x_trials_d.shape
    if lag < 1 or lag >= T:
        raise ValueError("lag must satisfy 1 <= lag < trial_length")
    if cond_trials_d is not None:
        cond_trials_d = np.asarray(cond_trials_d, dtype=np.int64)
        if cond_trials_d.shape[:2] != x_trials_d.shape:
            raise ValueError("cond trials shape mismatch")
        if cond_trials_d.ndim == 2:
            cond_trials_d = cond_trials_d[..., None]

    c_joint: Dict[Tuple[int, int, int, Tuple[int, ...]], int] = {}
    c_yxc: Dict[Tuple[int, int, Tuple[int, ...]], int] = {}
    c_yyc: Dict[Tuple[int, int, Tuple[int, ...]], int] = {}
    c_yc: Dict[Tuple[int, Tuple[int, ...]], int] = {}
    n_eff = 0
    for r in range(n_trials):
        for t in range(T - lag):
            yf = int(y_trials_d[r, t + lag]); yp = int(y_trials_d[r, t]); xp = int(x_trials_d[r, t])
            cp = () if cond_trials_d is None else tuple(int(v) for v in cond_trials_d[r, t])
            c_joint[(yf, yp, xp, cp)] = c_joint.get((yf, yp, xp, cp), 0) + 1
            c_yxc[(yp, xp, cp)] = c_yxc.get((yp, xp, cp), 0) + 1
            c_yyc[(yf, yp, cp)] = c_yyc.get((yf, yp, cp), 0) + 1
            c_yc[(yp, cp)] = c_yc.get((yp, cp), 0) + 1
            n_eff += 1
    n_eff_f = float(n_eff)
    te = 0.0
    for (yf, yp, xp, cp), count in c_joint.items():
        p_joint = count / n_eff_f
        p_yc = c_yc[(yp, cp)] / n_eff_f
        p_yxc = c_yxc[(yp, xp, cp)] / n_eff_f
        p_yyc = c_yyc[(yf, yp, cp)] / n_eff_f
        ratio = (p_joint * p_yc) / (p_yxc * p_yyc)
        te += p_joint * math.log2(ratio)
    return max(te, 0.0)


def trial_shuffle_surrogates(x_trials: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x_trials = np.asarray(x_trials)
    if x_trials.ndim != 2:
        raise ValueError("x_trials must be 2D")
    n_trials = x_trials.shape[0]
    if n_trials < 2:
        raise ValueError("Need at least 2 trials for trial-shuffle surrogate")
    perm = rng.permutation(n_trials)
    return x_trials[perm]


def max_stat_significance(
    observed: float,
    null_values: np.ndarray,
    alpha: float = 0.05,
) -> SignificanceResult:
    null_values = np.asarray(null_values, dtype=float)
    max_thr = float(np.quantile(null_values, 1 - alpha, method="higher")) if null_values.size else math.inf
    p = (1.0 + np.sum(null_values >= observed)) / (1.0 + len(null_values)) if len(null_values) else 1.0
    effect = max(observed - float(np.mean(null_values)) if len(null_values) else observed, 0.0)
    return SignificanceResult(
        observed_bits=float(observed),
        null_bits=null_values,
        p_value=float(p),
        significant=bool(observed > max_thr),
        effect_bits=float(effect),
        alpha=float(alpha),
    )


def select_lag_by_conditional_te(
    x_trials_d: np.ndarray,
    y_trials_d: np.ndarray,
    candidate_lags: Sequence[int],
    cond_trials_d: Optional[np.ndarray] = None,
    criterion: str = "asymmetry",
) -> Tuple[int, Dict[int, float]]:
    """Select optimal lag for CTE.

    Parameters
    ----------
    criterion : str
        "asymmetry" — Wibral (2013): max |CTE(X→Y) - CTE(Y→X)|
            Selects the timescale where DIRECTIONALITY is strongest.
        "max_cte" — max CTE(X→Y) directly.
            Selects the timescale where MAGNITUDE is largest.

    Reference: Wibral M et al. (2013) PLoS Comput Biol 9:e1003196
    """
    scores: Dict[int, float] = {}
    for lag in candidate_lags:
        te_xy = ensemble_conditional_te_discrete(
            x_trials_d, y_trials_d, lag=lag, cond_trials_d=cond_trials_d)
        if criterion == "asymmetry":
            te_yx = ensemble_conditional_te_discrete(
                y_trials_d, x_trials_d, lag=lag, cond_trials_d=cond_trials_d)
            scores[int(lag)] = abs(te_xy - te_yx)
        else:
            scores[int(lag)] = te_xy
    best_lag = max(scores, key=scores.get)
    return best_lag, scores


def conditional_te_bidirectional_ensemble(
    x_trials: np.ndarray,
    y_trials: np.ndarray,
    lag: int,
    n_bins: int = 4,
    cond_trials: Optional[np.ndarray] = None,
    n_surrogates: int = 200,
    surrogate_mode: str = "trial_shuffle",
    random_state: int = 0,
) -> TransferEntropyResultV2:
    x_trials = np.asarray(x_trials, dtype=float)
    y_trials = np.asarray(y_trials, dtype=float)
    if x_trials.shape != y_trials.shape:
        raise ValueError("x_trials and y_trials must have same shape")
    if x_trials.ndim != 2:
        raise ValueError("Trials must be (n_trials, n_time)")
    x_d = np.asarray([discretize_uniform(trial, n_bins=n_bins) for trial in x_trials])
    y_d = np.asarray([discretize_uniform(trial, n_bins=n_bins) for trial in y_trials])

    cond_d = None
    if cond_trials is not None:
        cond_trials = np.asarray(cond_trials, dtype=float)
        if cond_trials.shape[:2] != x_trials.shape:
            raise ValueError("cond_trials leading dims must match")
        if cond_trials.ndim == 2:
            cond_trials = cond_trials[..., None]
        cond_d = np.empty(cond_trials.shape, dtype=np.int64)
        for j in range(cond_trials.shape[2]):
            cond_d[:, :, j] = np.asarray([discretize_uniform(trial, n_bins=n_bins) for trial in cond_trials[:, :, j]])

    te_xy = ensemble_conditional_te_discrete(x_d, y_d, lag=lag, cond_trials_d=cond_d)
    te_yx = ensemble_conditional_te_discrete(y_d, x_d, lag=lag, cond_trials_d=cond_d)

    rng = np.random.default_rng(random_state)
    null_xy = np.zeros(n_surrogates, dtype=float)
    null_yx = np.zeros(n_surrogates, dtype=float)
    for i in range(n_surrogates):
        if surrogate_mode != "trial_shuffle":
            raise NotImplementedError("Only trial_shuffle implemented in reference module")
        xs = trial_shuffle_surrogates(x_d, rng)
        ys = trial_shuffle_surrogates(y_d, rng)
        null_xy[i] = ensemble_conditional_te_discrete(xs, y_d, lag=lag, cond_trials_d=cond_d)
        null_yx[i] = ensemble_conditional_te_discrete(ys, x_d, lag=lag, cond_trials_d=cond_d)

    return TransferEntropyResultV2(
        te_xy_bits=float(te_xy),
        te_yx_bits=float(te_yx),
        lag=int(lag),
        sig_xy=max_stat_significance(te_xy, null_xy),
        sig_yx=max_stat_significance(te_yx, null_yx),
        estimator="ensemble_discrete_cte",
        notes=(
            "trial_shuffle_surrogates preserve within-trial temporal structure",
            "conditioning acts on observed common drivers / mediators only",
        ),
    )


# ----------------------------
# High-level pipeline
# ----------------------------

def compute_causal_flim_generic_v2(
    series_dict: Dict[str, np.ndarray],
    pair_specs: Sequence[CausalPairSpecV2],
) -> List[CausalPairResultV2]:
    results: List[CausalPairResultV2] = []
    for spec in pair_specs:
        x_trials = preprocess_trials(series_dict[spec.source_key], spec.detrend_method)
        y_trials = preprocess_trials(series_dict[spec.target_key], spec.detrend_method)
        x_trials_d = np.asarray([discretize_uniform(tr, spec.n_bins) for tr in x_trials])
        y_trials_d = np.asarray([discretize_uniform(tr, spec.n_bins) for tr in y_trials])

        cond_trials = None
        cond_trials_d = None
        if spec.conditioning_keys:
            cond_list = [preprocess_trials(series_dict[key], spec.detrend_method) for key in spec.conditioning_keys]
            cond_trials = np.stack(cond_list, axis=-1)
            cond_trials_d = np.empty(cond_trials.shape, dtype=np.int64)
            for j in range(cond_trials.shape[2]):
                cond_trials_d[:, :, j] = np.asarray([discretize_uniform(tr, spec.n_bins) for tr in cond_trials[:, :, j]])

        lag, lag_scores = select_lag_by_conditional_te(
            x_trials_d=x_trials_d,
            y_trials_d=y_trials_d,
            candidate_lags=spec.candidate_lags,
            cond_trials_d=cond_trials_d,
            criterion=spec.lag_criterion,
        )
        te_result = conditional_te_bidirectional_ensemble(
            x_trials=x_trials,
            y_trials=y_trials,
            lag=lag,
            n_bins=spec.n_bins,
            cond_trials=cond_trials,
            n_surrogates=spec.n_surrogates,
            surrogate_mode=spec.surrogate_mode,
        )

        # ── Optional ADF diagnostic (not a gate, just info) ──
        adf_info = _adf_diagnostic_trials(x_trials, y_trials)
        te_result.adf_diagnostics = adf_info
        if adf_info.get("warning"):
            te_result.notes = te_result.notes + (adf_info["warning"],)

        results.append(
            CausalPairResultV2(
                spec=spec,
                te_result=te_result,
                diagnostics={"lag_scores": lag_scores, "adf": adf_info},
            )
        )
    return results


# ═══════════════════════════════════════════════════════════════════════
# ADF diagnostic (optional, not a gate)
# ═══════════════════════════════════════════════════════════════════════

def _adf_diagnostic_trials(
    x_trials: np.ndarray,
    y_trials: np.ndarray,
) -> Dict[str, Any]:
    """Run ADF on first trial as a diagnostic.

    NOT a gatekeeper: ensemble trial-shuffle is the primary
    non-stationarity defense. ADF is informational only.

    Reference: Dickey & Fuller (1979) JASA 74:427
    """
    info: Dict[str, Any] = {"available": False, "warning": ""}
    try:
        from statsmodels.tsa.stattools import adfuller
        info["available"] = True
        # Test first trial as representative sample
        x0 = x_trials[0]
        y0 = y_trials[0]
        if len(x0) >= 12:
            x_res = adfuller(x0, maxlag=min(10, len(x0) // 5), autolag='AIC')
            y_res = adfuller(y0, maxlag=min(10, len(y0) // 5), autolag='AIC')
            info["x_adf_stat"] = float(x_res[0])
            info["x_adf_pvalue"] = float(x_res[1])
            info["y_adf_stat"] = float(y_res[0])
            info["y_adf_pvalue"] = float(y_res[1])
            if x_res[1] > 0.05 or y_res[1] > 0.05:
                info["warning"] = (
                    f"ADF diagnostic: possible non-stationarity "
                    f"(X p={x_res[1]:.4f}, Y p={y_res[1]:.4f}). "
                    f"Ensemble trial-shuffle mitigates but does not fully "
                    f"eliminate this risk."
                )
    except ImportError:
        info["warning"] = "statsmodels not available for ADF diagnostic"
    return info


# ═══════════════════════════════════════════════════════════════════════
# Bridge adapter: List[SimulationResult] → series_dict
# ═══════════════════════════════════════════════════════════════════════

def sim_results_to_trials_dict(
    sim_results: list,
    tau_d_ns: float = 4.1,
) -> Dict[str, np.ndarray]:
    """Convert a list of independent SimulationResult replicas into
    a trials dictionary suitable for compute_causal_flim_generic_v2.

    Each SimulationResult contributes ONE trial (row) per series key.
    All replicas must have the same n_frames.

    Parameters
    ----------
    sim_results : list of SimulationResult
        Independent simulation replicas (same config, different seeds).
    tau_d_ns : float
        Unquenched donor lifetime (ns) for FLIM τ extraction.

    Returns
    -------
    series_dict : dict[str, ndarray(n_trials, n_frames)]
        Keys: "n_complexes", "n_bleached", "intensity", "tau_mean",
              "E_mean", "density", "msd"
    """
    if not sim_results:
        raise ValueError("Need at least 1 SimulationResult")

    n_trials = len(sim_results)
    n_frames = sim_results[0].n_frames
    for i, sr in enumerate(sim_results):
        if sr.n_frames != n_frames:
            raise ValueError(
                f"Replica {i} has {sr.n_frames} frames, expected {n_frames}")

    # Pre-allocate trial arrays
    keys = ["n_complexes", "n_bleached", "intensity",
            "tau_mean", "E_mean", "density", "msd"]
    trials = {k: np.zeros((n_trials, n_frames)) for k in keys}

    for r, sr in enumerate(sim_results):
        config = sr.config
        time_range = config.tcspc.time_range
        Lx, Ly = config.membrane.size
        area = Lx * Ly

        # Derive donor channel
        fluoro_map = {f.name: i for i, f in enumerate(config.fluorophores)}
        donor_ch = (fluoro_map[config.fret_pairs[0].donor_type]
                    if config.fret_pairs else 0)

        trials["n_complexes"][r] = sr.get_complex_count_series().astype(float)
        trials["n_bleached"][r] = sr.get_bleaching_series().astype(float)
        trials["intensity"][r] = sr.get_intensity_series().astype(float)

        # τ_mean per frame (from FLIM image — photonic observable)
        # Import here to avoid circular dependency at module level
        from ed_mrdt.analysis import mean_arrival_time
        for fi in range(n_frames):
            stack = sr.get_flim_stack(fi)
            tau_map, intens = mean_arrival_time(
                stack, time_range, channel=donor_ch, min_counts=1)
            valid = ~np.isnan(tau_map) & (intens > 0)
            if valid.any():
                trials["tau_mean"][r, fi] = np.average(
                    tau_map[valid], weights=intens[valid])  # already in ns
            else:
                trials["tau_mean"][r, fi] = tau_d_ns

        # E_mean, density, msd per frame
        for fi, fr in enumerate(sr.frames):
            gt = fr.ground_truth
            # FRET efficiency
            donors = (gt.has_dye & gt.is_alive & ~gt.is_bleached
                      & (gt.dye_type_id == 0))
            if donors.any():
                trials["E_mean"][r, fi] = np.mean(
                    gt.fret_efficiency_exact[donors])
            # Density
            trials["density"][r, fi] = gt.is_alive.sum() / area

        # MSD proxy
        for fi in range(1, n_frames):
            gt_prev = sr.frames[fi - 1].ground_truth
            gt_curr = sr.frames[fi].ground_truth
            alive = gt_prev.is_alive & gt_curr.is_alive
            if alive.any():
                dx = gt_curr.positions[alive] - gt_prev.positions[alive]
                dx[:, 0] -= Lx * np.round(dx[:, 0] / Lx)
                dx[:, 1] -= Ly * np.round(dx[:, 1] / Ly)
                trials["msd"][r, fi] = np.mean(dx[:, 0]**2 + dx[:, 1]**2)
        trials["msd"][r, 0] = trials["msd"][r, 1] if n_frames > 1 else 0.0

    return trials


# ═══════════════════════════════════════════════════════════════════════
# Pixel TE map (exploratory, not primary causal evidence)
# ═══════════════════════════════════════════════════════════════════════

def pixel_te_map_exploratory(
    sim_results: list,
    n_bins: int = 4,
    n_groups: int = 6,
    n_shuffle: int = 30,
    tau_d_ns: float = 4.1,
) -> Dict[str, Any]:
    """Compute exploratory pixel-level TE map from multiple replicas.

    Uses ensemble CTE: groups pixels by mean τ, pools across trials,
    computes CTE([RL]_global → τ_pixel_pooled).

    This is EXPLORATORY — it visualizes spatial distribution of
    causal coupling but is not primary statistical evidence.

    Returns
    -------
    dict with keys:
        "te_map": ndarray[nx, ny] — TE per pixel
        "sig_map": ndarray[nx, ny] — significance per pixel
        "method": str — description
    """
    if not sim_results:
        return {"te_map": None, "sig_map": None, "method": "no data"}

    sr0 = sim_results[0]
    config = sr0.config
    nx = config.scan.pixels_x
    ny = config.scan.pixels_y
    n_frames = sr0.n_frames
    n_trials = len(sim_results)
    time_range = config.tcspc.time_range

    fluoro_map = {f.name: i for i, f in enumerate(config.fluorophores)}
    donor_ch = (fluoro_map[config.fret_pairs[0].donor_type]
                if config.fret_pairs else 0)

    if n_frames < 6:
        return {"te_map": np.full((nx, ny), np.nan),
                "sig_map": np.full((nx, ny), False),
                "method": "insufficient frames"}

    # Extract per-pixel τ across trials: (n_trials, nx, ny, n_frames)
    pixel_tau = np.full((n_trials, nx, ny, n_frames), np.nan)
    complex_series = np.zeros((n_trials, n_frames))

    for r, sr in enumerate(sim_results):
        complex_series[r] = sr.get_complex_count_series().astype(float)
        n_tcspc = sr.get_flim_stack(0).shape[-1]
        bin_centers = ((np.arange(n_tcspc) + 0.5)
                       * (time_range * 1e9 / n_tcspc))
        for fi in range(n_frames):
            stack = sr.get_flim_stack(fi)
            donor_stack = stack[donor_ch]
            total = donor_stack.sum(axis=-1)
            valid_px = total >= 3
            if valid_px.any():
                weighted = np.tensordot(donor_stack, bin_centers,
                                        axes=([-1], [0]))
                pixel_tau[r, :, :, fi] = np.where(
                    valid_px, weighted / np.maximum(total, 1), np.nan)

    # Mean τ per pixel across all trials and frames for grouping
    mean_tau = np.nanmean(pixel_tau, axis=(0, 3))  # (nx, ny)
    valid_pix = ~np.isnan(mean_tau)

    te_map = np.full((nx, ny), np.nan)
    sig_map = np.full((nx, ny), False)

    if valid_pix.sum() < n_groups * 2:
        return {"te_map": te_map, "sig_map": sig_map,
                "method": "too few valid pixels"}

    tau_vals = mean_tau[valid_pix]
    actual_groups = min(n_groups, max(2, valid_pix.sum() // 5))

    try:
        quantiles = np.percentile(
            tau_vals, np.linspace(0, 100, actual_groups + 1))
        pixel_group = np.full((nx, ny), -1, dtype=int)
        for g in range(actual_groups):
            low = quantiles[g]
            high = quantiles[g + 1] + (1e-10 if g == actual_groups - 1 else 0)
            mask = valid_pix & (mean_tau >= low) & (mean_tau < high)
            pixel_group[mask] = g

        for g in range(actual_groups):
            gmask = pixel_group == g
            if gmask.sum() < 2:
                continue

            # Pool pixel series across trials for this group
            # target: (n_trials, n_time_pooled) where n_time_pooled = n_pixels_in_group * n_frames
            target_trials = []
            source_trials = []
            for r in range(n_trials):
                px_series_list = []
                for ix in range(nx):
                    for iy in range(ny):
                        if gmask[ix, iy]:
                            s = pixel_tau[r, ix, iy, :]
                            if np.isnan(s).sum() < n_frames // 2:
                                s_clean = np.nan_to_num(s, nan=np.nanmean(s))
                                px_series_list.append(s_clean)
                if px_series_list:
                    target_trials.append(np.concatenate(px_series_list))
                    # Tile source to match pooled length
                    source_trials.append(
                        np.tile(complex_series[r], len(px_series_list)))

            if len(target_trials) < 2:
                continue

            # Ensure all trials same length
            min_len = min(len(t) for t in target_trials)
            target_arr = np.array([t[:min_len] for t in target_trials])
            source_arr = np.array([s[:min_len] for s in source_trials])

            # Discretize and compute ensemble CTE
            src_d = np.array([discretize_uniform(t, n_bins)
                              for t in source_arr])
            tgt_d = np.array([discretize_uniform(t, n_bins)
                              for t in target_arr])

            te_g = ensemble_conditional_te_discrete(src_d, tgt_d, lag=1)

            # Quick significance via trial shuffle
            rng = np.random.default_rng(g)
            nulls = np.zeros(n_shuffle)
            for i in range(n_shuffle):
                xs = trial_shuffle_surrogates(src_d, rng)
                nulls[i] = ensemble_conditional_te_discrete(xs, tgt_d, lag=1)

            bias = np.mean(nulls)
            te_corr = max(te_g - bias, 0.0)
            sig = te_g > np.percentile(nulls, 95)

            te_map[gmask] = te_corr
            sig_map[gmask] = sig

    except (ValueError, IndexError):
        pass

    return {
        "te_map": te_map,
        "sig_map": sig_map,
        "method": (
            "Exploratory pixel CTE via ensemble trial-shuffle. "
            f"{n_trials} replicas, {actual_groups} τ-groups, "
            f"{n_shuffle} surrogates. NOT primary causal evidence."
        ),
    }
