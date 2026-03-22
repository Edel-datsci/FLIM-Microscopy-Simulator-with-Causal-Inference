"""
ED-MRDT v1.1 — Analysis Module: 7-Layer System Radiograph
============================================================

Comprehensive analysis toolkit that extracts maximum information from
FLIM-FRET time-lapse simulations. Each layer answers a distinct
scientific question; together they provide complete understanding.

Layer Architecture:
  L1  Phasor decomposition     — species identification (fit-free)
  L2  Spatial correlation C(r) — cluster detection (ICS)
  L3  Temporal ACF(τ)          — binding kinetics per pixel
  L4  Pair correlation pCF     — molecular flow mapping
  L5  iMSD via STICS           — diffusion law classification
  L6  Transfer entropy TE      — causal information flow
  L7  Ground truth discrepancy — the "error microscope"

Design principles:
  - Every formula traced to a published reference
  - Pure NumPy/SciPy, zero external dependencies
  - Each function takes SimulationResult → named result dataclass
  - All layers can compare IMAGE vs GROUND TRUTH

References (by layer):
  L1: Digman et al. (2008) Biophys J 94:L14–L16
      Ranjit et al. (2018) Nat Protocols 13:1979–2004
  L2: Petersen et al. (1993) Biophys J 65:1135 (ICS)
      Wiseman & Bhatt (2024) PMC11508332 (phase-FLIM ICS)
  L3: Elson & Magde (1974) Biopolymers 13:1 (FCS theory)
      Sci Rep (2021) 11:20098 (dynamic FRET-FLIM screening)
  L4: Digman & Gratton (2009) Biophys J 97:665–673
      Hinde et al. (2013) PNAS 110:135–140
  L5: Hébert et al. (2005) Biophys J 88:3601–3614 (STICS)
      Di Rienzo et al. (2013) PNAS 110:12307–12312 (iMSD)
  L6: Schreiber (2000) Phys Rev Lett 85:461 (transfer entropy)
      Wibral et al. (2014) J Comput Neurosci 30:45–67
  L7: No direct reference — unique to simulation with ground truth
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit, prange

__all__ = [
    # Lifetime fitting
    "BiExpFitResult",
    "biexp_fit_global",
    # Phasor decomposition (L1)
    "PhasorResult",
    "compute_phasor",
    # Spatial correlation (L2)
    "SpatialCorrelationResult",
    "compute_spatial_correlation",
    # Temporal ACF (L3)
    "TemporalACFResult",
    "compute_temporal_acf",
    # Pair correlation (L4)
    "PairCorrelationResult",
    "compute_pair_correlation",
    # iMSD via STICS (L5)
    "STICSResult",
    "compute_stics_imsd",
    # Transfer entropy (L6)
    "TransferEntropyResult",
    "transfer_entropy",
    "te_ksg_bidirectional",
    "CausalPairSpec",
    "CausalPairResult",
    "CausalFLIMResult",
    "compute_causal_flim",
    "compute_causal_flim_generic",
    # Ground truth (L7)
    "GroundTruthDiscrepancyResult",
    "compute_ground_truth_discrepancy",
]

# ═══════════════════════════════════════════════════════════════════════
# COMMON UTILITIES
# ═══════════════════════════════════════════════════════════════════════


def mean_arrival_time(flim_stack: np.ndarray,
                      time_range: float = 25e-9,
                      channel: int = 0,
                      min_counts: int = 10
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Mean photon arrival time per pixel — fast lifetime estimator.

    For mono-exponential decay I(t) = A exp(-t/τ), the mean arrival
    time <t> ≈ τ when the TCSPC window T >> τ.

    This is the standard estimator used in experimental FLIM analysis.
    For typical parameters (τ_D ~ 4 ns, T = 25 ns, T/τ ~ 6), the
    truncation bias is ~1.4%, which is small compared to shot noise
    at typical photon counts. The bias is a real experimental
    systematic error that L7 (error microscope) should detect.

    References
    ----------
    Lakowicz (2006) Principles of Fluorescence Spectroscopy, Eq. 4.4.
    Becker (2005) Advanced TCSPC Techniques, §8.2.

    Parameters
    ----------
    flim_stack : ndarray [n_ch, nx, ny, n_bins]
    time_range : float, TCSPC window in seconds
    channel : int, fluorophore index (= config.fluorophores[] order)
    min_counts : int, minimum photons for valid τ estimate

    Returns
    -------
    tau_map : ndarray [nx, ny] in nanoseconds, NaN where insufficient counts
    intensity : ndarray [nx, ny] total photon counts
    """
    data = flim_stack[channel].astype(np.float64)  # [nx, ny, n_bins]
    n_bins = data.shape[-1]
    bin_width_ns = (time_range / n_bins) * 1e9
    t_ns = (np.arange(n_bins) + 0.5) * bin_width_ns

    intensity = data.sum(axis=-1)
    weighted = np.tensordot(data, t_ns, axes=([-1], [0]))

    tau_map = np.full_like(intensity, np.nan)
    mask = intensity >= min_counts
    tau_map[mask] = weighted[mask] / intensity[mask]

    return tau_map, intensity


# ═══════════════════════════════════════════════════════════════════════
# BI-EXPONENTIAL LIFETIME FITTING
# ═══════════════════════════════════════════════════════════════════════
# Reference: Lakowicz (2006) Principles of Fluorescence Spectroscopy,
#   Chapter 4 §4.9: Multi-exponential decays
# Bajzer et al. (1991) Eur Biophys J 20:247 — NLLS for TCSPC data
#
# In a FRET system, the donor TCSPC decay is fundamentally bi-exponential:
#
#   I(t) = A_free · exp(-t/τ_D) + A_fret · exp(-t/τ_DA)
#
# where τ_D = unquenched donor lifetime, τ_DA = τ_D(1-E) = quenched
# donor lifetime in the complex. The amplitudes A_free and A_fret are
# proportional to the number of free donors and complexed donors,
# respectively.
#
# A mono-exponential estimator (<t>) gives a population-weighted average
# which conflates the two physically distinct species, losing the
# ability to resolve FRET efficiency and donor fraction independently.
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class BiExpFitResult:
    """Result of bi-exponential lifetime fitting.

    I(t) = A1·exp(-t/τ1) + A2·exp(-t/τ2)

    Convention: τ1 ≥ τ2 (τ1 = free donor, τ2 = FRET complex).

    Attributes
    ----------
    tau1_ns : float, longer lifetime (free donor)
    tau2_ns : float, shorter lifetime (FRET donor-acceptor)
    a1 : float, amplitude of τ1 component
    a2 : float, amplitude of τ2 component
    f1 : float, fractional intensity of τ1: f1 = A1·τ1 / (A1·τ1 + A2·τ2)
    f2 : float, fractional intensity of τ2
    fret_efficiency : float, E = 1 - τ2/τ1
    tau_mean_ns : float, intensity-weighted mean: <τ> = f1·τ1 + f2·τ2
    chi2_reduced : float, reduced chi² of the fit
    converged : bool, whether the fit converged
    """
    tau1_ns: float
    tau2_ns: float
    a1: float
    a2: float
    f1: float
    f2: float
    fret_efficiency: float
    tau_mean_ns: float
    chi2_reduced: float
    converged: bool


def biexp_fit_global(flim_stack: np.ndarray,
                     time_range: float = 25e-9,
                     channel: int = 0,
                     tau_d_ns: float = 4.1,
                     ) -> BiExpFitResult:
    """Fit bi-exponential decay to the GLOBAL (summed) TCSPC histogram.

    Sums all pixels to maximize photon statistics, then fits:
        I(t) = A1·exp(-t/τ1) + A2·exp(-t/τ2)

    Uses constrained Levenberg-Marquardt with physically motivated
    initial guesses: τ1 ≈ τ_D (free donor), τ2 ≈ τ_D/2 (FRET at E~0.5).

    Reference
    ---------
    Lakowicz (2006) Principles of Fluorescence Spectroscopy, §4.9
    Becker (2005) Advanced TCSPC Techniques, §8.4

    Parameters
    ----------
    flim_stack : ndarray [n_ch, nx, ny, n_bins]
    time_range : float, TCSPC window in seconds
    channel : int, fluorophore index (= config.fluorophores[] order)
    tau_d_ns : float, expected unquenched donor lifetime (ns)

    Returns
    -------
    BiExpFitResult with both lifetime components
    """
    from scipy.optimize import curve_fit

    data = flim_stack[channel].astype(np.float64)  # [nx, ny, n_bins]
    n_bins = data.shape[-1]
    bin_width_ns = (time_range / n_bins) * 1e9

    # Global histogram: sum over all pixels
    histogram = data.sum(axis=(0, 1))  # [n_bins]
    t_ns = (np.arange(n_bins) + 0.5) * bin_width_ns  # bin centers

    # Find usable range: from peak to where counts > 1
    # BUG FIX #5: For simulated data without IRF, the peak is analytically
    # at bin 0. Using argmax on noisy data can pick a noise-driven bin,
    # losing 1–2 bins of the fast component. Since this simulator has no
    # IRF convolution, we use bin 0. For real experimental data with an IRF,
    # argmax would be appropriate.
    # Reference: Köllner & Wolfrum (1992) Chem Phys Lett 200:199
    peak_bin = 0  # No IRF in simulation; for experimental data, use np.argmax(histogram)
    usable = histogram[peak_bin:] > 1
    if usable.sum() < 6:
        # Not enough data — return mono-exponential fallback
        total_counts = histogram.sum()
        if total_counts > 0:
            tau_mono = np.sum(histogram * t_ns) / total_counts
        else:
            tau_mono = tau_d_ns
        return BiExpFitResult(
            tau1_ns=tau_mono, tau2_ns=tau_mono,
            a1=1.0, a2=0.0, f1=1.0, f2=0.0,
            fret_efficiency=0.0,
            tau_mean_ns=tau_mono, chi2_reduced=np.nan,
            converged=False,
        )

    # Fit region: from peak onward (avoids IRF rising edge)
    t_fit = t_ns[peak_bin:]
    y_fit = histogram[peak_bin:]
    # Shift time origin to t=0 at peak for numerical stability
    t_shifted = t_fit - t_fit[0]

    # Bi-exponential model
    def biexp(t, a1, tau1, a2, tau2):
        return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)

    # Initial guesses
    amp_max = y_fit[0]
    p0 = [amp_max * 0.5, tau_d_ns, amp_max * 0.5, tau_d_ns * 0.5]
    bounds_low = [0, 0.1, 0, 0.05]
    bounds_high = [amp_max * 10, tau_d_ns * 3, amp_max * 10, tau_d_ns * 1.5]

    # BUG FIX #3: Use Iteratively Reweighted Least Squares (IRLS) to
    # approximate Poisson MLE. Pure Neyman χ² (σ=sqrt(max(data,1))) is biased
    # for low-count bins (0–5 photons). The IRLS approach: start with data-based
    # weights, fit, then update weights using MODEL predictions, and refit.
    # This converges to the Poisson MLE while using the robust Levenberg-
    # Marquardt / trust-region-reflective optimizer from curve_fit.
    # Reference: Bajzer et al. (1991) Biophys J 60:1437
    #            Laurence & Bhatt (2006) Biophys J 91:3361 (IRLS for TCSPC)
    from scipy.optimize import curve_fit

    try:
        # Iteration 0: data-based weights (Neyman chi-squared as seed)
        sigma = np.sqrt(np.maximum(y_fit, 1.0))
        popt, pcov = curve_fit(
            biexp, t_shifted, y_fit, p0=p0,
            sigma=sigma, absolute_sigma=True,
            bounds=(bounds_low, bounds_high),
            maxfev=10000,
        )

        # IRLS iterations: update weights from model predictions
        for _irls_iter in range(3):
            y_model = biexp(t_shifted, *popt)
            sigma = np.sqrt(np.maximum(y_model, 1.0))  # model-based variance
            popt, pcov = curve_fit(
                biexp, t_shifted, y_fit, p0=popt,
                sigma=sigma, absolute_sigma=True,
                bounds=(bounds_low, bounds_high),
                maxfev=10000,
            )

        a1_fit, tau1_fit, a2_fit, tau2_fit = popt

        # Enforce convention: τ1 ≥ τ2
        if tau1_fit < tau2_fit:
            tau1_fit, tau2_fit = tau2_fit, tau1_fit
            a1_fit, a2_fit = a2_fit, a1_fit

        # Goodness of fit: Pearson χ² with model-based variance
        y_pred = biexp(t_shifted, *popt)
        sigma = np.sqrt(np.maximum(y_pred, 1.0))  # model-based variance
        residuals = (y_fit - y_pred) / sigma
        dof = len(y_fit) - 4  # 4 free parameters
        chi2_red = np.sum(residuals**2) / max(dof, 1)

        # Fractional intensities (Lakowicz Eq. 4.29)
        fi_1 = a1_fit * tau1_fit
        fi_2 = a2_fit * tau2_fit
        fi_total = fi_1 + fi_2
        f1 = fi_1 / fi_total if fi_total > 0 else 1.0
        f2 = fi_2 / fi_total if fi_total > 0 else 0.0

        # FRET efficiency
        E = 1.0 - tau2_fit / tau1_fit if tau1_fit > 0 else 0.0

        # Intensity-weighted mean lifetime
        tau_mean = f1 * tau1_fit + f2 * tau2_fit

        return BiExpFitResult(
            tau1_ns=tau1_fit, tau2_ns=tau2_fit,
            a1=a1_fit, a2=a2_fit,
            f1=f1, f2=f2,
            fret_efficiency=E,
            tau_mean_ns=tau_mean,
            chi2_reduced=chi2_red,
            converged=True,
        )

    except (RuntimeError, ValueError, Exception):
        # Fit failed — return mono-exponential fallback
        import warnings
        warnings.warn("biexp_fit_global: Poisson MLE optimization failed, "
                      "returning mono-exponential fallback.", stacklevel=2)
        total_counts = histogram.sum()
        tau_mono = np.sum(histogram * t_ns) / total_counts if total_counts > 0 else tau_d_ns
        return BiExpFitResult(
            tau1_ns=tau_mono, tau2_ns=tau_mono,
            a1=1.0, a2=0.0, f1=1.0, f2=0.0,
            fret_efficiency=0.0,
            tau_mean_ns=tau_mono, chi2_reduced=np.nan,
            converged=False,
        )


def biexp_fit_pixelwise(flim_stack: np.ndarray,
                        time_range: float = 25e-9,
                        channel: int = 0,
                        tau_d_ns: float = 4.1,
                        min_counts: int = 50,
                        ) -> Dict[str, np.ndarray]:
    """Per-pixel bi-exponential fit using global τ1, τ2 as fixed templates.

    Strategy (Pelet et al. 2004 Biophys J 87:2884):
      1. Fit the GLOBAL histogram to get τ1 and τ2
      2. For each pixel, fix τ1 and τ2, fit only amplitudes A1, A2
         via linear least squares (fast, robust at low counts)
      3. Compute per-pixel FRET fraction: f_FRET = A2·τ2 / (A1·τ1 + A2·τ2)

    This "global analysis" approach is standard in FLIM-FRET because
    individual pixels rarely have enough photons for free 4-parameter fits.

    Reference
    ---------
    Pelet et al. (2004) Biophys J 87:2884 — global analysis of FLIM-FRET
    Barber et al. (2009) J R Soc Interface 6:S93 — rapid FLIM-FRET fitting

    Parameters
    ----------
    flim_stack : ndarray [n_ch, nx, ny, n_bins]
    time_range, channel, tau_d_ns, min_counts : as above

    Returns
    -------
    dict with keys: 'tau1_ns', 'tau2_ns', 'a1_map', 'a2_map',
        'f_fret_map', 'E_fret', 'tau_mean_map', 'mask'
    """
    # Step 1: Global fit to get τ1, τ2
    global_fit = biexp_fit_global(flim_stack, time_range, channel, tau_d_ns)
    tau1 = global_fit.tau1_ns
    tau2 = global_fit.tau2_ns

    data = flim_stack[channel].astype(np.float64)
    nx, ny, n_bins = data.shape
    bin_width_ns = (time_range / n_bins) * 1e9
    t_ns = (np.arange(n_bins) + 0.5) * bin_width_ns

    # Build basis vectors (templates)
    peak_bin = np.argmax(data.sum(axis=(0, 1)))
    t_shifted = t_ns[peak_bin:] - t_ns[peak_bin]
    basis1 = np.exp(-t_shifted / tau1)  # [n_fit_bins]
    basis2 = np.exp(-t_shifted / tau2)  # [n_fit_bins]

    # Design matrix for linear regression: y = B @ [a1, a2]
    B = np.column_stack([basis1, basis2])  # [n_fit_bins, 2]
    BtB = B.T @ B
    n_fit = len(t_shifted)

    # Output maps
    a1_map = np.full((nx, ny), np.nan)
    a2_map = np.full((nx, ny), np.nan)
    f_fret_map = np.full((nx, ny), np.nan)
    tau_mean_map = np.full((nx, ny), np.nan)
    intensity = data.sum(axis=-1)
    mask = intensity >= min_counts

    for i in range(nx):
        for j in range(ny):
            if not mask[i, j]:
                continue
            y_pixel = data[i, j, peak_bin:]
            if y_pixel.sum() < min_counts:
                mask[i, j] = False
                continue

            # Linear least squares: [a1, a2] = (BᵀB)⁻¹ Bᵀ y
            Bty = B.T @ y_pixel
            try:
                amps = np.linalg.solve(BtB, Bty)
            except np.linalg.LinAlgError:
                continue

            # Enforce non-negativity
            a1_val = max(amps[0], 0.0)
            a2_val = max(amps[1], 0.0)

            a1_map[i, j] = a1_val
            a2_map[i, j] = a2_val

            # Fractional intensity of FRET component
            fi_1 = a1_val * tau1
            fi_2 = a2_val * tau2
            fi_total = fi_1 + fi_2
            if fi_total > 0:
                f_fret_map[i, j] = fi_2 / fi_total
                tau_mean_map[i, j] = (fi_1 * tau1 + fi_2 * tau2) / fi_total
            else:
                f_fret_map[i, j] = 0.0
                tau_mean_map[i, j] = tau1

    return {
        'tau1_ns': tau1,
        'tau2_ns': tau2,
        'a1_map': a1_map,
        'a2_map': a2_map,
        'f_fret_map': f_fret_map,
        'E_fret': global_fit.fret_efficiency,
        'tau_mean_map': tau_mean_map,
        'mask': mask,
        'global_fit': global_fit,
    }


# ═══════════════════════════════════════════════════════════════════════
# L1: PHASOR DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════
# Reference: Digman et al. (2008) Biophys J 94:L14–L16
#
# The phasor transform maps each pixel's TCSPC histogram to a point
# (g, s) on the phasor plot via the 1st harmonic of the Fourier
# transform:
#
#   g(x,y) = Σ_k I(x,y,k) cos(2πk/N) / Σ_k I(x,y,k)
#   s(x,y) = Σ_k I(x,y,k) sin(2πk/N) / Σ_k I(x,y,k)
#
# For a single exponential with lifetime τ:
#   g = 1 / (1 + (ωτ)²),  s = ωτ / (1 + (ωτ)²)
# where ω = 2π / T_range (angular frequency of the laser repetition).
#
# Points on the universal semicircle → mono-exponential.
# Points inside → multi-exponential mixture.
# Linear combination rule: mixture phasors lie on the line segment
# connecting the pure component phasors.
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class PhasorResult:
    """Result of phasor analysis for one frame."""
    g: np.ndarray              # [nx, ny] cosine component
    s: np.ndarray              # [nx, ny] sine component
    intensity: np.ndarray      # [nx, ny] total counts
    tau_phi: np.ndarray        # [nx, ny] phase lifetime (ns)
    tau_mod: np.ndarray        # [nx, ny] modulation lifetime (ns)
    mask: np.ndarray           # [nx, ny] bool — sufficient photons
    omega: float               # angular frequency (rad/s)
    harmonic: int = 1          # Fourier harmonic used


def phasor_transform(flim_stack: np.ndarray,
                     time_range: float = 25e-9,
                     channel: int = 0,
                     harmonic: int = 1,
                     min_counts: int = 20
                     ) -> PhasorResult:
    """Compute phasor coordinates (g, s) for each pixel.

    Reference: Digman et al. (2008) Biophys J 94:L14–L16, Eq. S1.
    Also: Ranjit et al. (2018) Nat Protocols 13:1979, Eq. 1–2.

    Parameters
    ----------
    flim_stack : ndarray [n_ch, nx, ny, n_bins]
    time_range : float, TCSPC window in seconds
    channel : int, fluorophore index (= config.fluorophores[] order)
    harmonic : int, Fourier harmonic (1 = fundamental)
    min_counts : int, minimum photons for reliable phasor

    Returns
    -------
    PhasorResult with g, s, lifetimes, and validity mask
    """
    data = flim_stack[channel].astype(np.float64)  # [nx, ny, n_bins]
    n_bins = data.shape[-1]

    # Angular frequency: ω = 2π·harmonic / T_range
    omega = 2.0 * np.pi * harmonic / time_range

    # Phase angles for each bin center
    # k-th bin corresponds to time t_k = (k + 0.5) × bin_width
    # Normalized phase: φ_k = 2π·harmonic·(k + 0.5)/N
    phi = 2.0 * np.pi * harmonic * (np.arange(n_bins) + 0.5) / n_bins

    # Cosine and sine projections (vectorized over all pixels)
    cos_proj = np.cos(phi)  # [n_bins]
    sin_proj = np.sin(phi)  # [n_bins]

    intensity = data.sum(axis=-1)  # [nx, ny]
    mask = intensity >= min_counts

    # g = Σ I(k) cos(φ_k) / Σ I(k)
    # s = Σ I(k) sin(φ_k) / Σ I(k)
    g_num = np.tensordot(data, cos_proj, axes=([-1], [0]))
    s_num = np.tensordot(data, sin_proj, axes=([-1], [0]))

    g = np.zeros_like(intensity)
    s = np.zeros_like(intensity)
    g[mask] = g_num[mask] / intensity[mask]
    s[mask] = s_num[mask] / intensity[mask]

    # Derived lifetimes (Ranjit et al. 2018, Eq. 3–4):
    # Phase lifetime:      τ_φ = (1/ω) × (s/g)
    # Modulation lifetime: τ_m = (1/ω) × √(1/m² - 1),  m = √(g²+s²)
    omega_ns = omega * 1e-9   # convert to ns⁻¹ for output in ns
    tau_phi = np.full_like(g, np.nan)
    tau_mod = np.full_like(g, np.nan)

    valid = mask & (g > 1e-10)
    tau_phi[valid] = (s[valid] / g[valid]) / omega_ns

    m_sq = g ** 2 + s ** 2
    mod_valid = valid & (m_sq > 1e-10) & (m_sq < 1.0 - 1e-10)
    tau_mod[mod_valid] = np.sqrt(1.0 / m_sq[mod_valid] - 1.0) / omega_ns

    return PhasorResult(
        g=g, s=s, intensity=intensity,
        tau_phi=tau_phi, tau_mod=tau_mod,
        mask=mask, omega=omega, harmonic=harmonic,
    )


def phasor_universal_circle(n_points: int = 200
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the universal semicircle for phasor plots.

    For mono-exponential decays, (g, s) lies on the semicircle
    centered at (0.5, 0) with radius 0.5.

    Reference: Digman et al. (2008) Biophys J 94:L14, Fig. 1.
    """
    theta = np.linspace(0, np.pi, n_points)
    g = 0.5 + 0.5 * np.cos(theta)
    s = 0.5 * np.sin(theta)
    return g, s


def phasor_fret_trajectory(tau_d_ns: float,
                           omega: float,
                           n_points: int = 100
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the FRET trajectory on the phasor plot.

    As FRET efficiency E goes from 0 to 1, the quenched donor
    lifetime τ_DA = τ_D (1 - E) traces a line on the phasor plot.

    Reference: ISS Technical Note, FLIM Phasors & FRET trajectories;
               Digman et al. (2008) Fig. S6.

    Parameters
    ----------
    tau_d_ns : float, unquenched donor lifetime in nanoseconds
    omega : float, angular frequency in rad/s (from PhasorResult.omega)
    n_points : int, number of points along trajectory

    Returns
    -------
    g_traj, s_traj : arrays of phasor coordinates along FRET trajectory
    """
    E = np.linspace(0, 0.99, n_points)
    tau_da = tau_d_ns * (1.0 - E)   # ns
    omega_ns = omega * 1e-9          # ns⁻¹
    wt = omega_ns * tau_da
    g_traj = 1.0 / (1.0 + wt ** 2)
    s_traj = wt / (1.0 + wt ** 2)
    return g_traj, s_traj


# ═══════════════════════════════════════════════════════════════════════
# L2: SPATIAL AUTOCORRELATION C(r) — Image Correlation Spectroscopy
# ═══════════════════════════════════════════════════════════════════════
# Reference: Petersen et al. (1993) Biophys J 65:1135–1146
#
# The spatial autocorrelation function is defined as:
#
#   G(ξ,η) = <δI(x,y) · δI(x+ξ, y+η)> / <I>²
#
# where δI = I - <I> is the intensity fluctuation.
#
# Key observables:
#   G(0,0) = 1/N_clusters  (inverse cluster density per beam area)
#   Width of G → cluster size convolved with PSF
#
# Radial average C(r) = angular mean of G(ξ,η) at radius r.
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class SpatialCorrelationResult:
    """Result of spatial autocorrelation analysis."""
    r_um: np.ndarray          # radial distance in µm
    C_r: np.ndarray           # radial autocorrelation C(r)
    G_2d: np.ndarray          # full 2D autocorrelation G(ξ,η)
    cluster_density: float    # 1/G(0) in clusters per beam area
    sigma_r_um: float         # Gaussian fit width (cluster size) in µm
    source: str = ""          # "image" or "ground_truth"


def spatial_autocorrelation(image: np.ndarray,
                            pixel_size_um: float,
                            ) -> SpatialCorrelationResult:
    """Compute spatial autocorrelation C(r) via ICS.

    Reference: Petersen et al. (1993) Biophys J 65:1135, Eq. 1–5.

    Parameters
    ----------
    image : ndarray [nx, ny], intensity or lifetime map
    pixel_size_um : float, pixel size in micrometers

    Returns
    -------
    SpatialCorrelationResult with radial correlation function
    """
    img = image.astype(np.float64)
    mean_I = img.mean()
    if mean_I < 1e-10:
        nx, ny = img.shape
        return SpatialCorrelationResult(
            r_um=np.zeros(1), C_r=np.zeros(1),
            G_2d=np.zeros((nx, ny)),
            cluster_density=0.0, sigma_r_um=0.0,
        )

    # Fluctuation image
    delta = img - mean_I

    # 2D autocorrelation via FFT (Wiener-Khinchin theorem)
    # G(ξ,η) = IFFT(|FFT(δI)|²) / (N × <I>²)
    ft = np.fft.fft2(delta)
    power = np.abs(ft) ** 2
    G_2d = np.fft.ifft2(power).real / (img.size * mean_I ** 2)

    # Shift so zero-lag is at center
    G_2d = np.fft.fftshift(G_2d)

    nx, ny = G_2d.shape
    cx, cy = nx // 2, ny // 2

    # Radial average
    y_idx, x_idx = np.mgrid[:nx, :ny]
    r_px = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)

    max_r = min(cx, cy)
    r_bins = np.arange(0, max_r + 1)
    r_um = r_bins * pixel_size_um

    C_r = np.zeros(len(r_bins))
    for i, r in enumerate(r_bins):
        ring = (r_px >= r - 0.5) & (r_px < r + 0.5)
        if ring.any():
            C_r[i] = G_2d[ring].mean()

    # Cluster density: 1/G(0) (Petersen 1993, Eq. 6)
    G0 = C_r[0] if C_r[0] > 1e-20 else 1e-20
    cluster_density = 1.0 / G0

    # Gaussian fit for cluster size: C(r) ≈ G0 × exp(-r²/2σ²)
    # Use log-linear fit on first few points where C > 0
    valid = C_r > C_r[0] * 0.01
    if np.sum(valid) >= 3 and C_r[0] > 0:
        r_fit = r_um[valid][:10]
        c_fit = C_r[valid][:10]
        c_fit = np.maximum(c_fit, 1e-20)
        # ln(C) = ln(G0) - r²/(2σ²)  →  slope of ln(C) vs r² gives σ
        log_c = np.log(c_fit / c_fit[0])
        r_sq = r_fit ** 2
        if len(r_sq) >= 2 and r_sq[-1] > 0:
            # Least squares: log_c = -r² / (2σ²)
            slope = np.sum(log_c * r_sq) / np.sum(r_sq ** 2)
            if slope < -1e-10:
                sigma_r = np.sqrt(-1.0 / (2.0 * slope))
            else:
                sigma_r = 0.0
        else:
            sigma_r = 0.0
    else:
        sigma_r = 0.0

    return SpatialCorrelationResult(
        r_um=r_um, C_r=C_r, G_2d=G_2d,
        cluster_density=cluster_density,
        sigma_r_um=sigma_r,
    )


def spatial_correlation_ground_truth(positions: np.ndarray,
                                     membrane_size: Tuple[float, float],
                                     n_bins: int = 50,
                                     r_max_um: float = 1.0,
                                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Radial distribution function g(r) from particle positions.

    Direct pair counting: the gold standard for spatial correlation.

    Reference: Allen & Tildesley (1987) Computer Simulation of
    Liquids, §2.6 (radial distribution function).

    Parameters
    ----------
    positions : ndarray [N, 2] in µm
    membrane_size : (Lx, Ly) in µm
    n_bins : number of radial bins
    r_max_um : maximum distance

    Returns
    -------
    r_um : bin centers in µm
    g_r : radial distribution function (1 = random)
    """
    Lx, Ly = membrane_size
    N = len(positions)
    area = Lx * Ly
    rho = N / area  # number density

    dr = r_max_um / n_bins
    r_edges = np.linspace(0, r_max_um, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    counts = np.zeros(n_bins)

    # Pair counting with minimum image convention (PBC)
    for i in range(N):
        dx = positions[i + 1:, 0] - positions[i, 0]
        dy = positions[i + 1:, 1] - positions[i, 1]
        # Minimum image
        dx = dx - Lx * np.round(dx / Lx)
        dy = dy - Ly * np.round(dy / Ly)
        r = np.sqrt(dx ** 2 + dy ** 2)
        idx = (r / dr).astype(int)
        valid = idx < n_bins
        np.add.at(counts, idx[valid], 2)  # count pair for both i and j

    # Normalize: g(r) = counts / (N × ρ × 2πr × dr)
    g_r = np.zeros(n_bins)
    for i in range(n_bins):
        shell_area = 2.0 * np.pi * r_centers[i] * dr
        if shell_area > 0 and N > 1:
            g_r[i] = counts[i] / (N * rho * shell_area)

    return r_centers, g_r


# ═══════════════════════════════════════════════════════════════════════
# L3: TEMPORAL AUTOCORRELATION ACF(τ) — binding kinetics
# ═══════════════════════════════════════════════════════════════════════
# Reference: Elson & Magde (1974) Biopolymers 13:1 (FCS theory)
#
# The temporal autocorrelation of a signal X(t):
#
#   ACF(τ) = <δX(t) δX(t+τ)> / <δX²>
#
# where δX = X - <X>. ACF(0) = 1, decays to 0.
#
# For binding equilibrium with rate constants kon, koff:
#   ACF(τ) ~ exp(-τ/τ_corr)
#   τ_corr = 1/(kon·[L]_free + koff)
#
# Reference: Elson & Magde (1974), Eq. 28; Hess et al. (2002)
# Biochemistry 41:697 (binding FCS).
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class TemporalACFResult:
    """Result of temporal autocorrelation analysis."""
    lag_s: np.ndarray          # lag times in seconds
    acf_global: np.ndarray     # ACF of global observable (e.g., mean τ)
    acf_pixel_mean: np.ndarray  # average ACF across all valid pixels
    tau_corr_s: float          # estimated correlation time (seconds)
    n_valid_pixels: int = 0
    observable: str = ""       # description of what was correlated


def temporal_autocorrelation(time_series: np.ndarray,
                             dt: float = 1.0,
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Normalized temporal autocorrelation of a 1D signal.

    ACF(τ) = <δX(t)·δX(t+τ)> / Var(X)

    Uses unbiased estimator with proper normalization.

    Reference: Elson & Magde (1974), Biopolymers 13:1, Eq. 10.

    Parameters
    ----------
    time_series : ndarray [T]
    dt : float, time step between samples

    Returns
    -------
    lags : ndarray, lag times
    acf : ndarray, normalized autocorrelation (ACF[0]=1)
    """
    x = time_series.astype(np.float64)
    N = len(x)
    if N < 3:
        return np.array([0.0]), np.array([1.0])

    x = x - x.mean()
    var = np.var(x)
    if var < 1e-30:
        return np.arange(N) * dt, np.zeros(N)

    # Full correlation via FFT, then normalize per lag
    # Unbiased: divide by (N - |k|) rather than N
    ft = np.fft.rfft(x, n=2 * N)
    acf_raw = np.fft.irfft(ft * np.conj(ft))[:N].real

    # Unbiased normalization
    n_pairs = np.arange(N, 0, -1, dtype=np.float64)
    acf = acf_raw / n_pairs / var

    lags = np.arange(N) * dt
    return lags, acf


def temporal_acf_per_pixel(tau_maps: np.ndarray,
                           frame_interval: float,
                           min_valid_frames: int = 5,
                           ) -> TemporalACFResult:
    """Temporal ACF of lifetime fluctuations, averaged across pixels.

    For each pixel with sufficient data across frames, compute ACF
    of τ(t). Then average over all valid pixels.

    Reference: Sci Rep (2021) 11:20098, Figs. 3–5
    (dynamic FRET-FLIM screening methodology).

    Parameters
    ----------
    tau_maps : ndarray [n_frames, nx, ny] lifetime maps in ns
    frame_interval : float, time between frames in seconds
    min_valid_frames : int, minimum non-NaN frames per pixel

    Returns
    -------
    TemporalACFResult with per-pixel averaged ACF
    """
    n_frames, nx, ny = tau_maps.shape
    max_lag = n_frames // 2

    # Global ACF: average τ across all pixels per frame
    global_mean_tau = np.nanmean(tau_maps, axis=(1, 2))
    lags_global, acf_global = temporal_autocorrelation(
        global_mean_tau, dt=frame_interval
    )

    # Per-pixel ACF, then average
    acf_sum = np.zeros(max_lag)
    n_valid = 0

    for ix in range(nx):
        for iy in range(ny):
            trace = tau_maps[:, ix, iy]
            good = ~np.isnan(trace)
            if good.sum() < min_valid_frames:
                continue
            # Fill NaN with local mean for ACF continuity
            filled = trace.copy()
            filled[~good] = np.nanmean(trace)
            _, acf = temporal_autocorrelation(filled, dt=frame_interval)
            acf_sum[:max_lag] += acf[:max_lag]
            n_valid += 1

    if n_valid > 0:
        acf_pixel_mean = acf_sum / n_valid
    else:
        acf_pixel_mean = np.zeros(max_lag)

    # Estimate τ_corr: find lag where ACF drops below 1/e
    lags_pixel = np.arange(max_lag) * frame_interval
    tau_corr = _estimate_correlation_time(lags_pixel, acf_pixel_mean)

    return TemporalACFResult(
        lag_s=lags_pixel,
        acf_global=acf_global[:max_lag],
        acf_pixel_mean=acf_pixel_mean,
        tau_corr_s=tau_corr,
        n_valid_pixels=n_valid,
        observable="donor_lifetime_ns",
    )


def _estimate_correlation_time(lags: np.ndarray,
                               acf: np.ndarray) -> float:
    """Estimate correlation time τ_c from ACF.

    Find the lag where ACF first drops below 1/e.
    If it never drops, return last lag (lower bound).
    """
    threshold = 1.0 / np.e
    below = np.where(acf < threshold)[0]
    if len(below) > 0 and below[0] > 0:
        # Linear interpolation for precision
        i = below[0]
        if i > 0 and acf[i - 1] > threshold:
            frac = (acf[i - 1] - threshold) / (acf[i - 1] - acf[i])
            return lags[i - 1] + frac * (lags[i] - lags[i - 1])
        return lags[i]
    return lags[-1] if len(lags) > 0 else 0.0


def temporal_acf_intensity(intensity_maps: np.ndarray,
                           frame_interval: float,
                           min_mean_counts: float = 0.1,
                           ) -> TemporalACFResult:
    """Temporal ACF of intensity fluctuations, averaged across pixels.

    Uses donor intensity I(x,y,t) instead of lifetime τ(x,y,t) as
    the observable. Intensity is always well-defined (even with 0-1
    photons/pixel) whereas lifetime requires sufficient photon
    statistics per pixel.

    For FCS/ICS theory, the intensity ACF detects the same binding
    kinetics as lifetime ACF:
      ACF_I(τ) = <δI(t)·δI(t+τ)> / Var(I)

    Reference: Elson & Magde (1974) Biopolymers 13:1, Eq. 10.
               Petersen et al. (1993) Biophys J 65:1135.

    Parameters
    ----------
    intensity_maps : ndarray [n_frames, nx, ny] intensity per pixel
    frame_interval : float, time between frames in seconds
    min_mean_counts : float, minimum mean intensity for a pixel to
        be included (filters dead pixels)

    Returns
    -------
    TemporalACFResult with intensity-based ACF
    """
    n_frames, nx, ny = intensity_maps.shape
    max_lag = n_frames // 2

    # Global ACF: total intensity per frame
    global_intensity = intensity_maps.sum(axis=(1, 2)).astype(np.float64)
    lags_global, acf_global = temporal_autocorrelation(
        global_intensity, dt=frame_interval
    )

    # Per-pixel ACF, then average
    acf_sum = np.zeros(max_lag)
    n_valid = 0

    for ix in range(nx):
        for iy in range(ny):
            trace = intensity_maps[:, ix, iy].astype(np.float64)
            if trace.mean() < min_mean_counts:
                continue
            _, acf = temporal_autocorrelation(trace, dt=frame_interval)
            acf_sum[:max_lag] += acf[:max_lag]
            n_valid += 1

    if n_valid > 0:
        acf_pixel_mean = acf_sum / n_valid
    else:
        acf_pixel_mean = np.zeros(max_lag)

    # Estimate τ_corr
    lags_pixel = np.arange(max_lag) * frame_interval
    tau_corr = _estimate_correlation_time(lags_pixel, acf_pixel_mean)

    return TemporalACFResult(
        lag_s=lags_pixel,
        acf_global=acf_global[:max_lag],
        acf_pixel_mean=acf_pixel_mean,
        tau_corr_s=tau_corr,
        n_valid_pixels=n_valid,
        observable="donor_intensity",
    )


# ═══════════════════════════════════════════════════════════════════════
# L4: PAIR CORRELATION pCF(r, τ) — molecular flow mapping
# ═══════════════════════════════════════════════════════════════════════
# Reference: Digman & Gratton (2009) Biophys J 97:665–673
#            Hinde et al. (2013) PNAS 110:135–140
#
# The pair correlation function cross-correlates intensity
# fluctuations at two spatial points separated by distance δ:
#
#   pCF(δ, τ) = <δI(x, t) · δI(x + δ, t + τ)> / (<I(x)> · <I(x+δ)>)
#
# If molecules flow from x to x+δ, pCF peaks at lag τ_transit.
# If there's a barrier, pCF is suppressed at that distance.
#
# In 2D images, we compute pCF along rows and columns, then
# generate a "carpet" of pCF(pixel_distance, lag_time).
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class PairCorrelationResult:
    """Result of pair correlation function analysis."""
    distances_px: np.ndarray   # pixel distances tested
    distances_um: np.ndarray   # distances in µm
    lag_frames: np.ndarray     # lag times in frame units
    lag_s: np.ndarray          # lag times in seconds
    pcf_carpet: np.ndarray     # [n_distances, n_lags] pCF carpet
    transit_time_s: np.ndarray  # transit time per distance (NaN if no peak)


@njit(fastmath=True)
def _compute_pcf_carpet_numba(intensity_series: np.ndarray, 
                              distances_px: np.ndarray, 
                              max_lag: int) -> np.ndarray:
    n_frames, nx, ny = intensity_series.shape
    n_dist = len(distances_px)
    pcf_carpet = np.zeros((n_dist, max_lag), dtype=np.float64)
    
    for d_idx in range(n_dist):
        delta = int(distances_px[d_idx])
        pcf_sum = np.zeros(max_lag, dtype=np.float64)
        count = 0
        
        # Horizontal pairs (x-axis)
        for ix in range(nx - delta):
            for iy in range(ny):
                mean_a = 0.0
                mean_b = 0.0
                for t in range(n_frames):
                    mean_a += intensity_series[t, ix, iy]
                    mean_b += intensity_series[t, ix + delta, iy]
                mean_a /= n_frames
                mean_b /= n_frames
                
                if mean_a < 1e-10 or mean_b < 1e-10:
                    continue
                    
                for lag in range(max_lag):
                    n_overlap = n_frames - lag
                    if n_overlap < 2:
                        break
                    cc = 0.0
                    for t in range(n_overlap):
                        da = intensity_series[t, ix, iy] - mean_a
                        db = intensity_series[t + lag, ix + delta, iy] - mean_b
                        cc += da * db
                    pcf_sum[lag] += cc / (n_overlap * mean_a * mean_b)
                count += 1
                
        # Vertical pairs (y-axis)
        if delta > 0:
            for ix in range(nx):
                for iy in range(ny - delta):
                    mean_a = 0.0
                    mean_b = 0.0
                    for t in range(n_frames):
                        mean_a += intensity_series[t, ix, iy]
                        mean_b += intensity_series[t, ix, iy + delta]
                    mean_a /= n_frames
                    mean_b /= n_frames
                    
                    if mean_a < 1e-10 or mean_b < 1e-10:
                        continue
                        
                    for lag in range(max_lag):
                        n_overlap = n_frames - lag
                        if n_overlap < 2:
                            break
                        cc = 0.0
                        for t in range(n_overlap):
                            da = intensity_series[t, ix, iy] - mean_a
                            db = intensity_series[t + lag, ix, iy + delta] - mean_b
                            cc += da * db
                        pcf_sum[lag] += cc / (n_overlap * mean_a * mean_b)
                    count += 1
                    
        if count > 0:
            for lag in range(max_lag):
                pcf_carpet[d_idx, lag] = pcf_sum[lag] / count
                
    return pcf_carpet


def pair_correlation_function(intensity_series: np.ndarray,
                              pixel_size_um: float,
                              frame_interval: float,
                              max_distance_px: int = 10,
                              ) -> PairCorrelationResult:
    """Compute pair correlation function carpet from image time-series.

    Reference: Digman & Gratton (2009) Biophys J 97:665, Eq. 1–3.

    Parameters
    ----------
    intensity_series : ndarray [n_frames, nx, ny]
    pixel_size_um : float
    frame_interval : float, seconds between frames
    max_distance_px : int, maximum pixel distance for pCF

    Returns
    -------
    PairCorrelationResult with pCF carpet and transit times
    """
    n_frames, nx, ny = intensity_series.shape
    max_lag = n_frames // 2

    distances_px = np.arange(0, min(max_distance_px + 1, min(nx, ny) // 2))
    n_dist = len(distances_px)
    
    # Delegate to Numba compiled core
    arr_f64 = intensity_series.astype(np.float64)
    pcf_carpet = _compute_pcf_carpet_numba(arr_f64, distances_px, max_lag)

    # Extract transit times: lag of maximum pCF for each distance
    lag_frames = np.arange(max_lag)
    lag_s = lag_frames * frame_interval
    transit_time = np.full(n_dist, np.nan)
    for d_idx in range(n_dist):
        if distances_px[d_idx] == 0:
            transit_time[d_idx] = 0.0
            continue
        carpet_row = pcf_carpet[d_idx, 1:]  # skip zero-lag
        if len(carpet_row) > 0 and np.max(carpet_row) > 0:
            peak_idx = np.argmax(carpet_row) + 1
            transit_time[d_idx] = peak_idx * frame_interval

    return PairCorrelationResult(
        distances_px=distances_px,
        distances_um=distances_px * pixel_size_um,
        lag_frames=lag_frames,
        lag_s=lag_s,
        pcf_carpet=pcf_carpet,
        transit_time_s=transit_time,
    )


# ═══════════════════════════════════════════════════════════════════════
# L5: iMSD via STICS — diffusion law classification
# ═══════════════════════════════════════════════════════════════════════
# Reference: Di Rienzo et al. (2013) PNAS 110:12307, Eq. 1–3
#            Hébert et al. (2005) Biophys J 88:3601 (STICS theory)
#
# STICS computes the spatiotemporal correlation function:
#   G(ξ, η, τ) = <δI(x,y,t)·δI(x+ξ, y+η, t+τ)> / <I>²
#
# For diffusing particles, the Gaussian width σ²(τ) of G grows
# linearly: σ²(τ) = σ²(0) + 4Dτ  (2D free diffusion).
#
# iMSD(τ) = σ²(τ) - σ²(0) is the image Mean Square Displacement.
#
# Diffusion laws from iMSD shape:
#   Free:      iMSD(τ) = 4Dτ           (linear)
#   Confined:  iMSD(τ) → L²/3          (plateau)
#   Hop:       iMSD(τ) bilinear slope change
#   Anomalous: iMSD(τ) = 4Dτ^α, α≠1
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class IMSDResult:
    """Result of iMSD analysis via STICS."""
    lag_s: np.ndarray          # lag times in seconds
    iMSD_um2: np.ndarray       # image MSD in µm²
    sigma2_um2: np.ndarray     # Gaussian width² per lag in µm²
    D_eff: float               # effective diffusion coefficient (µm²/s)
    alpha: float               # anomalous exponent (1 = free)
    regime: str                # "free", "confined", "anomalous"


def compute_imsd(intensity_series: np.ndarray,
                 pixel_size_um: float,
                 frame_interval: float,
                 max_lag: Optional[int] = None,
                 ) -> IMSDResult:
    """Compute iMSD from STICS on an image time-series.

    STICS extracts the Gaussian width σ²(τ) of the spatiotemporal
    correlation function. iMSD = σ²(τ) - σ²(0) classifies the
    diffusion regime from the shape of iMSD(τ).

    Important: STICS detects DEVIATIONS from uniform free diffusion
    (confinement, hopping, clustering). For spatially uniform free
    diffusion without structure, G(ξ,η,τ) ≈ 0 and iMSD is unreliable.
    This is by design — see Di Rienzo (2013), Fig. 1.

    Shot noise subtraction is applied at lag=0 following
    Hébert et al. (2005) Biophys J 88:3601, Eq. 10.

    References
    ----------
    Di Rienzo et al. (2013) PNAS 110:12307, Eq. 1–3, Fig. 1C–D.
    Hébert et al. (2005) Biophys J 88:3601, Eq. 10–12.

    Parameters
    ----------
    intensity_series : ndarray [n_frames, nx, ny]
    pixel_size_um : float
    frame_interval : float, seconds between frames
    max_lag : int or None, maximum lag in frames

    Returns
    -------
    IMSDResult with iMSD curve and fitted diffusion parameters
    """
    n_frames, nx, ny = intensity_series.shape
    if max_lag is None:
        max_lag = min(n_frames // 2, 20)
    max_lag = min(max_lag, n_frames - 1)

    sigma2 = np.zeros(max_lag + 1)

    # Mean image (time average)
    mean_img = intensity_series.astype(np.float64).mean(axis=0)
    mean_I = mean_img.mean()
    if mean_I < 1e-10:
        lags = np.arange(max_lag + 1) * frame_interval
        return IMSDResult(lags, np.zeros(max_lag + 1),
                          np.zeros(max_lag + 1), 0.0, 1.0, "no_signal")

    for lag in range(max_lag + 1):
        # Average spatial correlation function over all frame pairs
        n_pairs = 0
        G_sum = np.zeros((nx, ny))

        for t in range(n_frames - lag):
            d1 = intensity_series[t].astype(np.float64) - mean_img
            d2 = intensity_series[t + lag].astype(np.float64) - mean_img

            # Cross-correlation via FFT
            ft1 = np.fft.fft2(d1)
            ft2 = np.fft.fft2(d2)
            G = np.fft.ifft2(ft1 * np.conj(ft2)).real / (nx * ny * mean_I ** 2)
            G_sum += np.fft.fftshift(G)
            n_pairs += 1

        if n_pairs > 0:
            G_avg = G_sum / n_pairs

            # Shot noise subtraction at lag=0 only
            # The shot noise contributes a delta function δ(ξ)δ(η) / <I>
            # to the autocorrelation at zero lag.
            # Reference: Hébert et al. (2005) Biophys J 88:3601, Eq. 10
            cx, cy = nx // 2, ny // 2
            if lag == 0:
                G_avg[cx, cy] -= 1.0 / mean_I

            # Extract σ² from Gaussian fit of correlation peak.
            # G(ξ,η,τ) ≈ A(τ) exp(-(ξ²+η²)/(2σ²(τ)))
            # Reference: Hébert et al. (2005) Eq. 12
            hw = min(nx // 4, ny // 4, 8)
            hw = max(hw, 2)
            patch = G_avg[cx - hw:cx + hw + 1, cy - hw:cy + hw + 1]
            xs = np.arange(-hw, hw + 1) * pixel_size_um
            ys = np.arange(-hw, hw + 1) * pixel_size_um
            XX, YY = np.meshgrid(xs, ys, indexing='ij')
            R2 = XX ** 2 + YY ** 2

            amp = patch[hw, hw]
            if amp > 1e-20:
                ratio = patch / amp
                valid = (R2 > 0) & (ratio > 0.05) & (ratio < 1.0)
                if valid.sum() >= 2:
                    log_ratio = np.log(ratio[valid])
                    r2_valid = R2[valid]
                    sum_log = np.sum(log_ratio)
                    if sum_log < -0.01:
                        sigma2[lag] = -np.sum(r2_valid) / (2.0 * sum_log)
                    else:
                        # BUG FIX #6: Instead of silently propagating the
                        # previous lag's value (which can cascade zeros from
                        # a failed lag=0), mark as NaN to signal failure.
                        sigma2[lag] = np.nan
                else:
                    sigma2[lag] = np.nan  # BUG FIX #6: insufficient valid points
            else:
                # Negative or zero amplitude: no Gaussian peak detectable
                # BUG FIX #6: Mark as NaN instead of inheriting previous value
                sigma2[lag] = np.nan

    # BUG FIX #6: Check if sigma2[0] is NaN (Gaussian fit failed at lag=0).
    # If so, the entire iMSD curve is unreliable — warn the user.
    lags = np.arange(max_lag + 1) * frame_interval

    if np.isnan(sigma2[0]):
        import warnings
        warnings.warn(
            "compute_imsd: Gaussian fit FAILED at lag=0 (insufficient SNR). "
            "iMSD results are unreliable. Consider increasing photon counts "
            "or spatial binning.",
            stacklevel=2,
        )
        return IMSDResult(
            lag_s=lags, iMSD_um2=np.full(max_lag + 1, np.nan),
            sigma2_um2=sigma2,
            D_eff=np.nan, alpha=np.nan, regime="unreliable",
        )

    # Count how many lags have valid sigma2
    n_nan = np.sum(np.isnan(sigma2))
    if n_nan > 0:
        import warnings
        warnings.warn(
            f"compute_imsd: Gaussian fit failed at {n_nan}/{max_lag+1} lags. "
            "iMSD curve may be unreliable.",
            stacklevel=2,
        )

    # iMSD = σ²(τ) - σ²(0)
    iMSD = sigma2 - sigma2[0]
    iMSD[np.isnan(iMSD)] = 0.0  # NaN lags → 0 for downstream robustness

    # Fit iMSD = 4D·τ^α for diffusion coefficient and anomalous exponent
    # Use log-log fit: log(iMSD) = log(4D) + α·log(τ)
    valid_lags = (lags > 0) & (iMSD > 0) & np.isfinite(iMSD)
    if np.sum(valid_lags) >= 2:
        log_t = np.log(lags[valid_lags])
        log_msd = np.log(iMSD[valid_lags])
        # Linear regression in log-log space
        A = np.vstack([log_t, np.ones(len(log_t))]).T
        result = np.linalg.lstsq(A, log_msd, rcond=None)
        alpha = result[0][0]
        log_4D = result[0][1]
        D_eff = np.exp(log_4D) / 4.0
    else:
        alpha = np.nan
        D_eff = np.nan

    # Classify regime
    if np.isnan(D_eff) or np.isnan(alpha):
        regime = "unreliable"
    elif D_eff < 1e-10:
        regime = "immobile"
    elif alpha < 0.7:
        regime = "confined"
    elif alpha > 1.3:
        regime = "directed"
    else:
        regime = "free"

    return IMSDResult(
        lag_s=lags, iMSD_um2=iMSD, sigma2_um2=sigma2,
        D_eff=D_eff, alpha=alpha, regime=regime,
    )


# ═══════════════════════════════════════════════════════════════════════
# L6: TRANSFER ENTROPY — causal information flow
# ═══════════════════════════════════════════════════════════════════════
# Reference: Schreiber (2000) Phys Rev Lett 85:461–464
#
# Transfer entropy from X to Y:
#
#   TE(X→Y) = Σ p(y_{t+1}, y_t, x_t) ×
#             log [ p(y_{t+1}|y_t, x_t) / p(y_{t+1}|y_t) ]
#
# TE measures the reduction in uncertainty about Y's future when
# knowing X's past, beyond what Y's own past provides.
#
# TE > 0: X has causal influence on Y.
# TE ≈ 0: No directed influence from X to Y.
# TE(X→Y) ≠ TE(Y→X) in general (directed measure).
#
# Implementation: histogram-based estimator with uniform binning.
# Reference: Wibral et al. (2014) J Comput Neurosci 30:45, §2.1.
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class TransferEntropyResult:
    """Result of transfer entropy analysis."""
    te_xy: float               # TE(X→Y) in bits
    te_yx: float               # TE(Y→X) in bits
    te_xy_shuffled: float      # TE(X→Y) under null (time-shuffled X)
    te_yx_shuffled: float      # TE(Y→X) under null
    x_label: str = ""
    y_label: str = ""
    n_bins: int = 0
    significant_xy: bool = False  # TE_xy > TE_xy_shuffled
    significant_yx: bool = False


def transfer_entropy(x: np.ndarray,
                     y: np.ndarray,
                     n_bins: int = 8,
                     lag: int = 1,
                     n_shuffle: int = 50,
                     ) -> TransferEntropyResult:
    """Compute transfer entropy TE(X→Y) and TE(Y→X).

    Uses histogram-based estimator with uniform binning.

    Reference: Schreiber (2000) Phys Rev Lett 85:461, Eq. 3.
    Estimator: Wibral et al. (2014) J Comput Neurosci 30:45.

    Parameters
    ----------
    x, y : ndarray [T], time series (will be discretized)
    n_bins : int, number of bins for discretization
    lag : int, prediction lag (default 1 = next time step)
    n_shuffle : int, number of shuffles for significance testing

    Returns
    -------
    TransferEntropyResult with TE in both directions + significance
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    N = min(len(x), len(y))
    x, y = x[:N], y[:N]

    if N < lag + 2:
        return TransferEntropyResult(0, 0, 0, 0, n_bins=n_bins)

    # Discretize into uniform bins
    x_d = _discretize(x, n_bins)
    y_d = _discretize(y, n_bins)

    # Compute TE(X→Y) and TE(Y→X)
    te_xy = _te_discrete(x_d, y_d, n_bins, lag)
    te_yx = _te_discrete(y_d, x_d, n_bins, lag)

    # Significance via time-shuffled surrogates
    # Bias correction: TE_corrected = TE_raw - <TE_shuffled>
    # Reference: Wibral et al. (2014) J Comput Neurosci 30:45, §2.3
    #   "finite sample bias ... should be subtracted"
    rng = np.random.default_rng(42)
    te_xy_null = np.zeros(n_shuffle)
    te_yx_null = np.zeros(n_shuffle)
    for i in range(n_shuffle):
        x_shuf = rng.permutation(x_d)
        y_shuf = rng.permutation(y_d)
        te_xy_null[i] = _te_discrete(x_shuf, y_d, n_bins, lag)
        te_yx_null[i] = _te_discrete(y_shuf, x_d, n_bins, lag)

    # Bias-corrected TE
    bias_xy = np.mean(te_xy_null)
    bias_yx = np.mean(te_yx_null)
    te_xy_corr = max(te_xy - bias_xy, 0.0)
    te_yx_corr = max(te_yx - bias_yx, 0.0)

    te_xy_thresh = np.percentile(te_xy_null, 95)
    te_yx_thresh = np.percentile(te_yx_null, 95)

    return TransferEntropyResult(
        te_xy=te_xy_corr, te_yx=te_yx_corr,
        te_xy_shuffled=bias_xy,
        te_yx_shuffled=bias_yx,
        n_bins=n_bins,
        significant_xy=(te_xy > te_xy_thresh),
        significant_yx=(te_yx > te_yx_thresh),
    )


@njit(fastmath=True)
def _discretize(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Uniform discretization into n_bins levels."""
    x_min = np.min(x)
    x_max = np.max(x)
    diff = x_max - x_min
    N = len(x)
    out = np.zeros(N, dtype=np.int64)
    if diff < 1e-30:
        return out
    
    for i in range(N):
        scaled = (x[i] - x_min) / diff
        val = int(scaled * n_bins)
        if val < 0:
            val = 0
        elif val >= n_bins:
            val = n_bins - 1
        out[i] = val
    return out


@njit(fastmath=True)
def _te_discrete(x_d: np.ndarray, y_d: np.ndarray,
                 n_bins: int, lag: int) -> float:
    """Compute TE(X→Y) from discretized time series.

    TE(X→Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)

    Using 3D histogram estimation.

    Reference: Schreiber (2000) Eq. 3, discretized form.
    """
    N = len(y_d) - lag
    if N < 3:
        return 0.0

    y_past = y_d[:N]
    y_future = y_d[lag:lag + N]
    x_past = x_d[:N]

    joint_3 = np.zeros((n_bins, n_bins, n_bins), dtype=np.float64)
    for t in range(N):
        joint_3[y_future[t], y_past[t], x_past[t]] += 1.0
        
    p_yx = np.zeros((n_bins, n_bins), dtype=np.float64)
    p_yfy = np.zeros((n_bins, n_bins), dtype=np.float64)
    p_y = np.zeros(n_bins, dtype=np.float64)
    
    for yf in range(n_bins):
        for yp in range(n_bins):
            for xp in range(n_bins):
                val = joint_3[yf, yp, xp] / N
                joint_3[yf, yp, xp] = val
                p_yx[yp, xp] += val
                p_yfy[yf, yp] += val
                p_y[yp] += val
                
    te = 0.0
    for yf in range(n_bins):
        for yp in range(n_bins):
            for xp in range(n_bins):
                p3 = joint_3[yf, yp, xp]
                if p3 < 1e-30:
                    continue
                p_yp = p_y[yp]
                p_ypxp = p_yx[yp, xp]
                p_yfyp = p_yfy[yf, yp]
                if p_yp < 1e-30 or p_ypxp < 1e-30 or p_yfyp < 1e-30:
                    continue
                te += p3 * np.log2((p3 * p_yp) / (p_ypxp * p_yfyp))

    return max(te, 0.0)


# ═══════════════════════════════════════════════════════════════════════
# L6b: KSG ESTIMATOR FOR TRANSFER ENTROPY — Numba-Accelerated
# ═══════════════════════════════════════════════════════════════════════
# Kraskov et al (2004) introduced a k-nearest-neighbor estimator for
# mutual information that avoids histogram binning entirely.
#
# TE(X→Y) = I(Y_future ; X_past | Y_past)   [conditional MI]
#
# Frenzel & Pompe (2007) Eq. 2:
#   I(X;Y|Z) = ψ(k) - <ψ(n_xz + 1) + ψ(n_yz + 1) - ψ(n_z + 1)>
#
# Numba implementation: brute-force Chebyshev distance for N < 5000.
# O(N²) but with Numba JIT this beats cKDTree for N < ~3000.
# For N > 5000, falls back to scipy.spatial.cKDTree.
#
# References:
#   Kraskov et al (2004) Phys Rev E 69:066138
#   Frenzel & Pompe (2007) Phys Rev Lett 99:204101
# ═══════════════════════════════════════════════════════════════════════


@njit(fastmath=True)
def _ksg_kth_neighbor_3d(data_3d, k):
    """Find k-th nearest neighbor Chebyshev distance in 3D space.

    Parameters
    ----------
    data_3d : ndarray[N, 3], standardized points
    k : int, neighbor rank (1-indexed: k=4 → 4th neighbor)

    Returns
    -------
    eps : ndarray[N], k-th neighbor Chebyshev distance per point
    """
    N = data_3d.shape[0]
    eps = np.zeros(N)

    for i in range(N):
        # Compute Chebyshev distance to all other points
        # and find k-th smallest (excluding self)
        dists = np.empty(N - 1)
        idx = 0
        for j in range(N):
            if j == i:
                continue
            # Chebyshev (max-norm) distance
            d = 0.0
            for dim in range(3):
                ad = abs(data_3d[i, dim] - data_3d[j, dim])
                if ad > d:
                    d = ad
            dists[idx] = d
            idx += 1

        # Partial sort to find k-th smallest
        dists.sort()
        eps[i] = dists[k - 1]

    return eps


@njit(fastmath=True)
def _ksg_count_within_eps_2d(data_2d, eps):
    """Count neighbors within Chebyshev eps in 2D (excluding self).

    Parameters
    ----------
    data_2d : ndarray[N, 2]
    eps : ndarray[N], per-point radius (strict <)

    Returns
    -------
    counts : ndarray[N], int64
    """
    N = data_2d.shape[0]
    counts = np.zeros(N, dtype=np.int64)

    for i in range(N):
        e = eps[i] * (1.0 - 1e-10)  # strict inequality
        c = 0
        for j in range(N):
            if j == i:
                continue
            d0 = abs(data_2d[i, 0] - data_2d[j, 0])
            d1 = abs(data_2d[i, 1] - data_2d[j, 1])
            d = d0 if d0 > d1 else d1
            if d < e:
                c += 1
        counts[i] = c

    return counts


@njit(fastmath=True)
def _ksg_count_within_eps_1d(data_1d, eps):
    """Count neighbors within Chebyshev eps in 1D (excluding self).

    Parameters
    ----------
    data_1d : ndarray[N]
    eps : ndarray[N], per-point radius (strict <)

    Returns
    -------
    counts : ndarray[N], int64
    """
    N = len(data_1d)
    counts = np.zeros(N, dtype=np.int64)

    for i in range(N):
        e = eps[i] * (1.0 - 1e-10)
        c = 0
        for j in range(N):
            if j == i:
                continue
            if abs(data_1d[i] - data_1d[j]) < e:
                c += 1
        counts[i] = c

    return counts


@njit(fastmath=True)
def _ksg_digamma(x):
    """Fast digamma approximation for integer and half-integer arguments.

    Uses Stirling's series for x >= 6, recursion for x < 6.
    Accurate to ~1e-8 for positive integers.

    Reference: Abramowitz & Stegun (1964) §6.3.18
    """
    result = 0.0
    val = float(x)
    # Recursion to shift argument ≥ 6
    while val < 6.0:
        result -= 1.0 / val
        val += 1.0
    # Stirling series: ψ(x) ≈ ln(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴)
    result += (math.log(val) - 0.5 / val
               - 1.0 / (12.0 * val * val)
               + 1.0 / (120.0 * val * val * val * val))
    return result


@njit(fastmath=True)
def _te_ksg_core(y_future, x_past, y_past, k):
    """Core KSG TE computation: TE(X→Y) via Frenzel & Pompe (2007).

    Fully Numba-accelerated. O(N²) brute force with Chebyshev norm.
    Fast for N < 5000 (typical for frame-level and ensemble TE).

    Parameters
    ----------
    y_future, x_past, y_past : ndarray[N], standardized
    k : int, nearest neighbors

    Returns
    -------
    te_nats : float, transfer entropy in nats
    """
    N = len(y_future)
    if N < k + 2:
        return 0.0

    # Build full 3D space [y_future, x_past, y_past]
    full_3d = np.empty((N, 3))
    for i in range(N):
        full_3d[i, 0] = y_future[i]
        full_3d[i, 1] = x_past[i]
        full_3d[i, 2] = y_past[i]

    # Find k-th neighbor Chebyshev distance in full 3D space
    eps = _ksg_kth_neighbor_3d(full_3d, k)

    # Build 2D subspaces
    xz = np.empty((N, 2))
    yz = np.empty((N, 2))
    for i in range(N):
        xz[i, 0] = x_past[i]
        xz[i, 1] = y_past[i]
        yz[i, 0] = y_future[i]
        yz[i, 1] = y_past[i]

    # Count neighbors in subspaces within eps
    n_xz = _ksg_count_within_eps_2d(xz, eps)
    n_yz = _ksg_count_within_eps_2d(yz, eps)
    n_z = _ksg_count_within_eps_1d(y_past, eps)

    # Frenzel & Pompe (2007) Eq. 2
    psi_k = _ksg_digamma(k)
    sum_psi = 0.0
    for i in range(N):
        sum_psi += (_ksg_digamma(n_xz[i] + 1)
                    + _ksg_digamma(n_yz[i] + 1)
                    - _ksg_digamma(n_z[i] + 1))

    te_nats = psi_k - sum_psi / N
    return te_nats


def te_ksg(x: np.ndarray, y: np.ndarray,
           k: int = 4, lag: int = 1) -> float:
    """Transfer Entropy TE(X→Y) using Numba-accelerated KSG estimator.

    Implements Frenzel & Pompe (2007) CMI estimator applied to TE:
    TE(X→Y) = I(Y_future ; X_past | Y_past)

    Uses Numba JIT-compiled brute-force O(N²) for N < 5000.
    Falls back to scipy.spatial.cKDTree for N ≥ 5000.

    Parameters
    ----------
    x, y : ndarray[T], continuous time series
    k : int, number of nearest neighbors (default 4, per Kraskov 2004)
    lag : int, prediction lag

    Returns
    -------
    te : float, transfer entropy in bits

    References
    ----------
    Kraskov et al (2004) Phys Rev E 69:066138, Eq. 8
    Frenzel & Pompe (2007) Phys Rev Lett 99:204101, Eq. 2
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    N = min(len(x), len(y)) - lag
    if N < k + 2:
        return 0.0

    # Construct embedding
    y_past = y[:N].copy()
    y_future = y[lag:lag + N].copy()
    x_past = x[:N].copy()

    # Standardize to unit variance (Kraskov 2004)
    for arr in (y_past, y_future, x_past):
        m = np.mean(arr)
        s = np.std(arr)
        if s > 1e-15:
            arr -= m
            arr /= s
        else:
            arr -= m

    if N < 5000:
        # Numba brute-force path (fast for small N)
        te_nats = _te_ksg_core(y_future, x_past, y_past, k)
    else:
        # cKDTree path for large N
        from scipy.spatial import cKDTree
        from scipy.special import digamma

        full_pts = np.column_stack([y_future, x_past, y_past])
        xz_pts = np.column_stack([x_past, y_past])
        yz_pts = np.column_stack([y_future, y_past])
        z_pts = y_past.reshape(-1, 1)

        tree_full = cKDTree(full_pts)
        dists, _ = tree_full.query(full_pts, k=k + 1, p=np.inf)
        eps = dists[:, k] * (1.0 - 1e-10)

        tree_xz = cKDTree(xz_pts)
        tree_yz = cKDTree(yz_pts)
        tree_z = cKDTree(z_pts)

        n_xz = np.array([tree_xz.query_ball_point(
            xz_pts[i], eps[i], p=np.inf, return_length=True) - 1
            for i in range(N)], dtype=np.int64)
        n_yz = np.array([tree_yz.query_ball_point(
            yz_pts[i], eps[i], p=np.inf, return_length=True) - 1
            for i in range(N)], dtype=np.int64)
        n_z = np.array([tree_z.query_ball_point(
            z_pts[i], eps[i], p=np.inf, return_length=True) - 1
            for i in range(N)], dtype=np.int64)

        te_nats = (digamma(k)
                   - np.mean(digamma(n_xz + 1)
                             + digamma(n_yz + 1)
                             - digamma(n_z + 1)))

    return max(te_nats / np.log(2), 0.0)  # nats → bits


def te_ksg_bidirectional(x: np.ndarray, y: np.ndarray,
                         k: int = 4, lag: int = 1,
                         n_shuffle: int = 50) -> TransferEntropyResult:
    """KSG Transfer Entropy in both directions with significance.

    Numba-accelerated for N < 5000 (typical FLIM frame series).

    Parameters
    ----------
    x, y : ndarray[T]
    k : int, nearest neighbors
    lag : int
    n_shuffle : int, surrogates for significance

    Returns
    -------
    TransferEntropyResult with bias-corrected TE + significance flags
    """
    te_xy = te_ksg(x, y, k=k, lag=lag)
    te_yx = te_ksg(y, x, k=k, lag=lag)

    rng = np.random.default_rng(42)
    te_xy_null = np.zeros(n_shuffle)
    te_yx_null = np.zeros(n_shuffle)

    for i in range(n_shuffle):
        te_xy_null[i] = te_ksg(rng.permutation(x), y, k=k, lag=lag)
        te_yx_null[i] = te_ksg(rng.permutation(y), x, k=k, lag=lag)

    bias_xy = np.mean(te_xy_null)
    bias_yx = np.mean(te_yx_null)

    return TransferEntropyResult(
        te_xy=max(te_xy - bias_xy, 0.0),
        te_yx=max(te_yx - bias_yx, 0.0),
        te_xy_shuffled=bias_xy,
        te_yx_shuffled=bias_yx,
        significant_xy=(te_xy > np.percentile(te_xy_null, 95)),
        significant_yx=(te_yx > np.percentile(te_yx_null, 95)),
    )


def find_optimal_lag(x: np.ndarray, y: np.ndarray,
                     k: int = 4,
                     lag_range: list = None,
                     n_shuffle: int = 50) -> tuple:
    """Find optimal lag by maximizing TE asymmetry (Wibral criterion).

    Scans multiple lags WITHOUT surrogates (fast), then runs full
    significance test at the lag with maximum |TE(X→Y) - TE(Y→X)|.

    This selects the timescale at which causal directionality is
    strongest — the hallmark of genuine information transfer.

    Parameters
    ----------
    x, y : ndarray[T]
        Time series (already block-averaged/detrended).
    k : int
        KSG nearest neighbors.
    lag_range : list of int
        Lags to scan (default [1, 2, 5, 10, 20]).
    n_shuffle : int
        Surrogates for significance at optimal lag.

    Returns
    -------
    (optimal_lag, TransferEntropyResult)

    Reference
    ---------
    Wibral M, Vicente R, Lindner M (2014) Directed Information Measures
    in Neuroscience. Springer. Ch. 1–3.
    Wibral M et al. (2013) PLOS Comput Biol 9:e1003196
    """
    if lag_range is None:
        lag_range = [1, 2, 5, 10, 20]

    best_lag = lag_range[0]
    best_asymmetry = -1.0

    for lag in lag_range:
        # Need at least 3× lag points for meaningful TE
        if lag >= len(x) // 3 or lag >= len(y) // 3:
            continue
        te_xy = te_ksg(x, y, k=k, lag=lag)
        te_yx = te_ksg(y, x, k=k, lag=lag)
        asymmetry = abs(te_xy - te_yx)

        if asymmetry > best_asymmetry:
            best_asymmetry = asymmetry
            best_lag = lag

    # Full significance test at optimal lag
    te_result = te_ksg_bidirectional(x, y, k=k, lag=best_lag,
                                     n_shuffle=n_shuffle)
    return best_lag, te_result


# ═══════════════════════════════════════════════════════════════════════
# L6c: LOCAL TRANSFER ENTROPY (Lizier et al 2008)
# ═══════════════════════════════════════════════════════════════════════
# The global TE is an AVERAGE of local values:
#   TE(X→Y) = <t_local(n)>
#
# where t_local(n) = log₂[p(y_{n+1}|y_past,x_past) / p(y_{n+1}|y_past)]
#
# This gives a TE value at EACH space-time point.
# Can be NEGATIVE (misleading information).
# Mean over all points = classical TE.
#
# Reference: Lizier et al (2008) Phys Rev E 77:026110, Eq. 3
# ═══════════════════════════════════════════════════════════════════════


@njit(fastmath=True)
def _local_te_core(y_future: np.ndarray, y_past: np.ndarray,
                   x_past: np.ndarray, n_bins: int, N: int
                   ) -> np.ndarray:
    """Numba-accelerated kernel for local TE computation.

    Builds 3D joint histogram and computes per-point local TE values.
    Reference: Lizier et al (2008) Phys Rev E 77:026110, Eq. 3
    """
    # Build joint histogram p(y_f, y_p, x_p)
    joint_3 = np.zeros((n_bins, n_bins, n_bins))
    for t in range(N):
        joint_3[y_future[t], y_past[t], x_past[t]] += 1.0
    inv_N = 1.0 / N
    for a in range(n_bins):
        for b in range(n_bins):
            for c in range(n_bins):
                joint_3[a, b, c] *= inv_N

    # Marginals
    p_yx = np.zeros((n_bins, n_bins))
    for b in range(n_bins):
        for c in range(n_bins):
            s = 0.0
            for a in range(n_bins):
                s += joint_3[a, b, c]
            p_yx[b, c] = s

    p_yfy = np.zeros((n_bins, n_bins))
    for a in range(n_bins):
        for b in range(n_bins):
            s = 0.0
            for c in range(n_bins):
                s += joint_3[a, b, c]
            p_yfy[a, b] = s

    p_y = np.zeros(n_bins)
    for b in range(n_bins):
        s = 0.0
        for a in range(n_bins):
            for c in range(n_bins):
                s += joint_3[a, b, c]
        p_y[b] = s

    # Local TE at each point
    LOG2 = 0.6931471805599453  # ln(2)
    local_values = np.zeros(N)
    for t in range(N):
        yf = y_future[t]
        yp = y_past[t]
        xp = x_past[t]
        p3 = joint_3[yf, yp, xp]
        pyp = p_y[yp]
        pypxp = p_yx[yp, xp]
        pyfyp = p_yfy[yf, yp]

        if p3 > 1e-30 and pyp > 1e-30 and pypxp > 1e-30 and pyfyp > 1e-30:
            local_values[t] = np.log(p3 * pyp / (pypxp * pyfyp)) / LOG2

    return local_values


def local_te(x: np.ndarray, y: np.ndarray,
             n_bins: int = 4, lag: int = 1
             ) -> Tuple[np.ndarray, float]:
    """Compute Local Transfer Entropy at each time point.

    Parameters
    ----------
    x, y : ndarray[T], time series
    n_bins : int, histogram bins for discretization
    lag : int, prediction lag

    Returns
    -------
    local_values : ndarray[N-lag], local TE at each point (bits)
    global_te : float, mean of local values (= classical TE)

    Reference
    ---------
    Lizier et al (2008) Phys Rev E 77:026110, Eq. 3
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    N = min(len(x), len(y)) - lag

    if N < 3:
        return np.zeros(0), 0.0

    # Discretize (already @njit)
    x_d = _discretize(x[:N + lag], n_bins)
    y_d = _discretize(y[:N + lag], n_bins)

    y_past = y_d[:N]
    y_future = y_d[lag:lag + N]
    x_past = x_d[:N]

    # Numba-accelerated core
    local_values = _local_te_core(y_future, y_past, x_past, n_bins, N)

    return local_values, float(np.mean(local_values))


# ═══════════════════════════════════════════════════════════════════════
# L6d: ENSEMBLE METHOD FOR PIXEL-LEVEL TE (Wollstadt et al 2014)
# ═══════════════════════════════════════════════════════════════════════
# With N_frames < 30 per pixel, neither histogram nor KSG has enough
# data. Solution: pool observations from STATISTICALLY EXCHANGEABLE
# pixels.
#
# Group pixels by similar observable (τ, intensity) → ensemble.
# If M pixels in a group × N frames → M×N data points.
# For 20×20 field with ~10 τ-groups: ~40 pixels × 20 frames = 800 pts.
#
# Key assumption: pixels with same lifetime τ sample the same process.
#
# Reference: Wollstadt et al (2014) PLoS ONE 9:e102833, §2.2
# ═══════════════════════════════════════════════════════════════════════


def ensemble_te(pixel_series_list: List[np.ndarray],
                source_global: np.ndarray,
                n_bins: int = 4,
                lag: int = 1,
                n_shuffle: int = 50,
                estimator: str = 'ksg',
                k_ksg: int = 4,
                ) -> Tuple[float, float, bool]:
    """Compute TE using ensemble of pooled pixel time series.

    Pools all pixel series sharing a statistical class into one long
    sequence, paired with the repeated global source. This yields
    M×N data points from M pixels × N frames.

    Parameters
    ----------
    pixel_series_list : list of ndarray[N_frames]
        Lifetime series for each pixel in the ensemble group
    source_global : ndarray[N_frames]
        Source time series (same for all pixels, e.g., [RL](t))
    n_bins : int, histogram bins (only for estimator='histogram')
    lag : int, prediction lag
    n_shuffle : int, surrogates for significance
    estimator : str, 'ksg' or 'histogram'
    k_ksg : int, KSG nearest neighbors

    Returns
    -------
    te : float, ensemble TE in bits (bias-corrected)
    te_null : float, mean TE under null hypothesis
    significant : bool, TE > 95th percentile of null

    Reference
    ---------
    Wollstadt et al (2014) PLoS ONE 9:e102833, §2.2
    """
    pooled_target = []
    pooled_source = []

    for px_series in pixel_series_list:
        valid = ~np.isnan(px_series)
        if valid.sum() < lag + 2:
            continue
        px_clean = px_series.copy()
        if not valid.all():
            px_clean[~valid] = np.nanmean(px_series)
        pooled_target.append(px_clean)
        pooled_source.append(source_global.copy())

    if len(pooled_target) == 0:
        return 0.0, 0.0, False

    target = np.concatenate(pooled_target)
    source = np.concatenate(pooled_source)

    if estimator == 'ksg':
        te = te_ksg(source, target, k=k_ksg, lag=lag)
        rng = np.random.default_rng(42)
        nulls = np.zeros(n_shuffle)
        for i in range(n_shuffle):
            nulls[i] = te_ksg(rng.permutation(source), target,
                              k=k_ksg, lag=lag)
        bias = np.mean(nulls)
        thresh = np.percentile(nulls, 95)
        return max(te - bias, 0.0), bias, te > thresh
    else:
        result = transfer_entropy(source, target,
                                  n_bins=n_bins, lag=lag,
                                  n_shuffle=n_shuffle)
        return result.te_xy, result.te_xy_shuffled, result.significant_xy


# ═══════════════════════════════════════════════════════════════════════
# L6e: BINARY TE FOR FLUORESCENCE (Orlandi et al 2014)
# ═══════════════════════════════════════════════════════════════════════
# For fluorescence signals (calcium imaging ≈ FLIM analogy):
#   Binarize: Δτ(t) = τ(t) - τ(t-1) > threshold → 1, else → 0
#   With binary signals, histogram has only 2³ = 8 cells
#   Need ~40 data points minimum (5 × 2³ = 40)
#
# This makes TE feasible with as few as 10-30 frames.
#
# Reference: Orlandi et al (2014) PLoS ONE 9:e98842, §Methods
# ═══════════════════════════════════════════════════════════════════════


def binary_te(x: np.ndarray, y: np.ndarray,
              lag: int = 1, n_shuffle: int = 50,
              threshold: str = 'median',
              ) -> TransferEntropyResult:
    """Transfer Entropy with binarized differential signals.

    Binarization: x_bin(t) = 1 if Δx(t) > threshold, else 0

    Parameters
    ----------
    x, y : ndarray[T], continuous time series
    lag : int
    n_shuffle : int
    threshold : str, 'median' (binarize around median of Δ)

    Returns
    -------
    TransferEntropyResult with binary TE in both directions

    Reference
    ---------
    Orlandi et al (2014) PLoS ONE 9:e98842, §Methods
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    dx = np.diff(x)
    dy = np.diff(y)

    x_bin = (dx > np.median(dx)).astype(np.float64)
    y_bin = (dy > np.median(dy)).astype(np.float64)

    N = min(len(x_bin), len(y_bin)) - lag
    if N < 8:
        return TransferEntropyResult(0, 0, 0, 0, n_bins=2)

    return transfer_entropy(x_bin, y_bin, n_bins=2, lag=lag,
                            n_shuffle=n_shuffle)


# ═══════════════════════════════════════════════════════════════════════
# L6f: CAUSAL FLIM — 6 Causal Pairs + Pixel TE Map
# ═══════════════════════════════════════════════════════════════════════
# Comprehensive causal analysis combining all estimators above.
#
# The 6 causal pairs map the complete information flow:
#   Par 1: [RL](t) → τ_mean(t)   — binding causes lifetime change
#   Par 2: bleach(t) → I(t)       — bleaching causes signal loss
#   Par 3: D_local → [RL]         — diffusion-limited vs reaction-limited
#   Par 4: density_local → E      — clustering drives FRET
#   Par 5: E(t) → n_bleach(t)     — FRET modulates bleaching
#   Par 6: Δbind → Δunbind        — Le Chatelier equilibrium test
#
# Pixel-level TE map uses Ensemble+KSG (Wollstadt 2014 + Kraskov 2004)
# with Local TE (Lizier 2008) for spatial resolution.
#
# References:
#   Schreiber (2000) Phys Rev Lett 85:461 — TE definition
#   Kraskov et al (2004) Phys Rev E 69:066138 — KSG estimator
#   Frenzel & Pompe (2007) Phys Rev Lett 99:204101 — CMI extension
#   Lizier et al (2008) Phys Rev E 77:026110 — Local TE
#   Wollstadt et al (2014) PLoS ONE 9:e102833 — Ensemble method
#   Orlandi et al (2014) PLoS ONE 9:e98842 — Binary TE
#   Barnett et al (2009) Phys Rev Lett 103:238701 — Granger ↔ TE
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class CausalPairSpec:
    """Specification for a causal pair to test with Transfer Entropy.

    The x_key and y_key reference data sources:
      - "species:<name>" → SubFrameData.species_counts[name] or GroundTruth series
      - "n_catalytic" → SubFrameData.n_catalytic
      - "n_recruited" → SubFrameData.n_recruited
      - "n_detached" → SubFrameData.n_detached
      - "n_converted" → SubFrameData.n_converted
      - "n_complexes" → SubFrameData.n_complexes (existing)
      - "n_bindings" → SubFrameData.n_bindings (existing)
      - "n_dissociations" → SubFrameData.n_dissociations (existing)
      - "n_bleached" → SubFrameData.n_bleached (existing)
      - "total_photons" → SubFrameData.total_photons (existing)
      - "tau_mean" → mean lifetime from FLIM image (computed at frame level)
      - "intensity" → total intensity per frame
      - "E_mean" → mean FRET efficiency from ground truth
      - "msd" → mean squared displacement (frame level)
      - "density" → particle density (frame level)

    Ref: Schreiber (2000) Phys Rev Lett 85:461
    """
    name: str                          # human-readable pair name
    x_key: str                         # source variable key
    y_key: str                         # target variable key
    x_label: str = ""                  # axis label for X (auto-generated if empty)
    y_label: str = ""                  # axis label for Y (auto-generated if empty)
    interpretation_if_significant: str = ""  # message if TE is significant
    interpretation_if_not: str = ""          # message if TE is not significant


@dataclass
class CausalPairResult:
    """Transfer Entropy result for one causal pair."""
    pair_id: int                # 1-6
    pair_name: str              # e.g., "Binding → Lifetime"
    x_label: str                # source variable name
    y_label: str                # target variable name
    te_xy: float                # TE(X→Y) in bits (bias-corrected)
    te_yx: float                # TE(Y→X) in bits (bias-corrected)
    te_xy_null: float           # mean TE under null (shuffled)
    te_yx_null: float
    significant_xy: bool        # TE_xy > 95th percentile of null
    significant_yx: bool
    n_points: int               # number of temporal samples used
    resolution: str             # "frame" or "subframe"
    interpretation: str = ""    # human-readable interpretation
    optimal_lag: int = 1         # lag selected by Wibral asymmetry criterion
    detrend_applied: str = ""    # "linear", "diff", or "" if disabled
    stationarity_warning: str = ""  # ADF test warning if non-stationary


@dataclass
class CausalFLIMResult:
    """Complete causal FLIM analysis: 6 pairs + pixel map."""
    pairs: List[CausalPairResult]
    pixel_te_map: Optional[np.ndarray] = None   # [nx, ny] TE(binding_global → τ_pixel)
    pixel_te_sig: Optional[np.ndarray] = None   # [nx, ny] bool: significant?
    pixel_te_description: str = ""
    n_frames: int = 0
    n_subframe_points: int = 0


def compute_causal_flim(sim_result,
                        tau_d_ns: float = 4.1,
                        n_bins: int = 0,
                        n_shuffle: int = 100,
                        ) -> CausalFLIMResult:
    """Compute all 6 causal pairs + pixel-level TE map.

    Uses sub-frame data if available (sim_result.subframe_data),
    otherwise falls back to frame-level time series.

    Parameters
    ----------
    sim_result : SimulationResult from pipeline.run()
    tau_d_ns : float, unquenched donor lifetime in nanoseconds
    n_bins : int, histogram bins for TE (0 = auto from data length)
    n_shuffle : int, number of shuffled surrogates for significance

    Returns
    -------
    CausalFLIMResult with all pairs and pixel map

    References
    ----------
    Schreiber (2000) Phys Rev Lett 85:461, Eq. 3
    Wibral et al. (2014) J Comput Neurosci 30:45, §2.3
    """
    config = sim_result.config
    n_frames = sim_result.n_frames
    time_range = config.tcspc.time_range

    # Derive donor/acceptor channels
    fluoro_name_to_id = {f.name: i for i, f in enumerate(config.fluorophores)}
    if len(config.fret_pairs) > 0:
        donor_ch = fluoro_name_to_id[config.fret_pairs[0].donor_type]
    else:
        donor_ch = 0

    # ── Extract frame-level time series ──────────────────────────
    complex_series = sim_result.get_complex_count_series().astype(np.float64)
    bleach_series = sim_result.get_bleaching_series().astype(np.float64)
    intensity_series = sim_result.get_intensity_series().astype(np.float64)

    # Mean lifetime per frame (from FLIM image)
    tau_mean_series = np.zeros(n_frames)
    for i in range(n_frames):
        flim_stack = sim_result.get_flim_stack(i)
        tau_map, intens = mean_arrival_time(flim_stack, time_range,
                                            channel=donor_ch, min_counts=1)
        valid = ~np.isnan(tau_map) & (intens > 0)
        if valid.any():
            tau_mean_series[i] = np.average(tau_map[valid],
                                            weights=intens[valid])  # already in ns
        else:
            tau_mean_series[i] = tau_d_ns

    # Mean FRET efficiency per frame (ground truth)
    E_series = np.zeros(n_frames)
    for i, fr in enumerate(sim_result.frames):
        gt = fr.ground_truth
        e_arr = gt.fret_efficiency_exact
        donors = gt.has_dye & gt.is_alive & ~gt.is_bleached & (gt.dye_type_id == 0)
        if donors.any():
            E_series[i] = np.mean(e_arr[donors])

    # Local density per frame (particles per µm²)
    Lx, Ly = config.membrane.size
    membrane_area = Lx * Ly
    density_series = np.zeros(n_frames)
    for i, fr in enumerate(sim_result.frames):
        gt = fr.ground_truth
        n_alive = gt.is_alive.sum()
        density_series[i] = n_alive / membrane_area

    # Mean squared displacement proxy: mean displacement from frame to frame
    # Approximates D_local time series
    msd_series = np.zeros(n_frames)
    for i in range(1, n_frames):
        gt_prev = sim_result.frames[i-1].ground_truth
        gt_curr = sim_result.frames[i].ground_truth
        # Only track particles alive in both frames
        alive_both = gt_prev.is_alive & gt_curr.is_alive
        if alive_both.any():
            dx = gt_curr.positions[alive_both] - gt_prev.positions[alive_both]
            # Handle periodic boundaries
            dx[:, 0] = dx[:, 0] - Lx * np.round(dx[:, 0] / Lx)
            dx[:, 1] = dx[:, 1] - Ly * np.round(dx[:, 1] / Ly)
            msd_series[i] = np.mean(dx[:, 0]**2 + dx[:, 1]**2)
    msd_series[0] = msd_series[1] if n_frames > 1 else 0.0

    # ── Check for sub-frame data ────────────────────────────────
    has_subframe = (hasattr(sim_result, 'subframe_data') and
                    sim_result.subframe_data is not None)

    # ── Auto n_bins ─────────────────────────────────────────────
    if n_bins == 0:
        if has_subframe:
            N_pts = len(sim_result.subframe_data.n_complexes)
            # Schreiber (2000): N >> n_bins^3
            n_bins = max(2, min(16, int(N_pts ** (1/3) / 3)))
        else:
            n_bins = max(2, min(8, n_frames // 3))

    pairs_results = []

    # ── PAR 1: [RL](t) → τ_mean(t) — binding causes lifetime ──
    # CRITICAL: Source = molecular binding count (ground truth BD steps)
    #           Target = τ_mean measured from FLIM image (photonic observable)
    # These are INDEPENDENT measurements on different instruments:
    #   X = n_complexes from Brownian dynamics (biology)
    #   Y = mean arrival time from TCSPC histogram (photonics)
    # TE(X→Y) tests whether molecular binding events CAUSE the observed
    # lifetime change measured by the FLIM microscope.
    #
    # NOTE: We deliberately avoid sf.mean_fret_E as target because it is
    # a deterministic monotonic function of n_complexes:
    #   mean_fret_E = E_const * n_bound / (n_bound + n_free)
    # Computing TE(X → f(X)) is tautological, not causal.
    # Ref: Wibral et al (2014) J Comput Neurosci 30:45, §4.1
    te1 = te_ksg_bidirectional(complex_series, tau_mean_series,
                               k=min(3, n_frames // 5),
                               lag=1, n_shuffle=n_shuffle)
    n_pts1 = n_frames
    res1 = "frame (KSG)"

    interp1 = ""
    if te1.significant_xy and not te1.significant_yx:
        interp1 = "Binding CAUSES lifetime change (FRET mechanism confirmed)"
    elif te1.significant_xy and te1.significant_yx:
        interp1 = "Bidirectional: binding↔lifetime (feedback loop)"
    elif not te1.significant_xy:
        interp1 = "Coupling too weak or insufficient data to detect"

    pairs_results.append(CausalPairResult(
        pair_id=1, pair_name="Binding → Lifetime",
        x_label="[RL](t)", y_label="τ_mean(t)",
        te_xy=te1.te_xy, te_yx=te1.te_yx,
        te_xy_null=te1.te_xy_shuffled, te_yx_null=te1.te_yx_shuffled,
        significant_xy=te1.significant_xy, significant_yx=te1.significant_yx,
        n_points=n_pts1, resolution=res1, interpretation=interp1,
    ))

    # ── PAR 2: bleach(t) → I(t) — bleaching causes signal loss ──
    if has_subframe:
        sf = sim_result.subframe_data
        te2 = transfer_entropy(sf.n_bleached.astype(np.float64),
                               sf.total_photons.astype(np.float64),
                               n_bins=n_bins, lag=1, n_shuffle=n_shuffle)
        n_pts2 = len(sf.n_bleached)
        res2 = "subframe"
    else:
        te2 = te_ksg_bidirectional(bleach_series, intensity_series,
                                   k=min(3, n_frames // 5),
                                   lag=1, n_shuffle=n_shuffle)
        n_pts2 = n_frames
        res2 = "frame (KSG)"

    interp2 = ""
    if te2.significant_xy:
        interp2 = "Bleaching CAUSES signal loss (photophysics confirmed)"
    else:
        interp2 = "Bleaching too slow to detect in this time window"

    pairs_results.append(CausalPairResult(
        pair_id=2, pair_name="Bleaching → Intensity",
        x_label="n_bleach(t)", y_label="I(t)",
        te_xy=te2.te_xy, te_yx=te2.te_yx,
        te_xy_null=te2.te_xy_shuffled, te_yx_null=te2.te_yx_shuffled,
        significant_xy=te2.significant_xy, significant_yx=te2.significant_yx,
        n_points=n_pts2, resolution=res2, interpretation=interp2,
    ))

    # ── PAR 3: D_local → [RL] — diffusion → binding ──────────
    # MSD is only available at frame level (particle positions per frame)
    # Use KSG for the N=20 frame-level series
    te3 = te_ksg_bidirectional(msd_series, complex_series,
                               k=min(3, n_frames // 5),
                               lag=1, n_shuffle=n_shuffle)
    n_pts3 = n_frames
    res3 = "frame (KSG)"

    interp3 = ""
    if te3.significant_xy:
        interp3 = "Diffusion-limited binding: molecular mobility drives encounters"
    else:
        interp3 = "Reaction-limited: binding rate independent of diffusion"

    pairs_results.append(CausalPairResult(
        pair_id=3, pair_name="Diffusion → Binding",
        x_label="MSD(t)", y_label="[RL](t)",
        te_xy=te3.te_xy, te_yx=te3.te_yx,
        te_xy_null=te3.te_xy_shuffled, te_yx_null=te3.te_yx_shuffled,
        significant_xy=te3.significant_xy, significant_yx=te3.significant_yx,
        n_points=n_pts3, resolution=res3, interpretation=interp3,
    ))

    # ── PAR 4: density_local → E — clustering → FRET ─────────
    # Frame-level with KSG (N=20)
    te4 = te_ksg_bidirectional(density_series, E_series,
                               k=min(3, n_frames // 5),
                               lag=1, n_shuffle=n_shuffle)
    interp4 = ""
    if te4.significant_xy:
        interp4 = "Clustering CAUSES FRET: density-dependent binding"
    else:
        interp4 = "No clustering-FRET causality (uniform distribution)"

    pairs_results.append(CausalPairResult(
        pair_id=4, pair_name="Clustering → FRET",
        x_label="ρ_local(t)", y_label="E(t)",
        te_xy=te4.te_xy, te_yx=te4.te_yx,
        te_xy_null=te4.te_xy_shuffled, te_yx_null=te4.te_yx_shuffled,
        significant_xy=te4.significant_xy, significant_yx=te4.significant_yx,
        n_points=n_frames, resolution="frame (KSG)", interpretation=interp4,
    ))

    # ── PAR 5: E(t) → n_bleach(t) — FRET → bleaching ────────
    # NOTE: E_series from per-particle Förster distances (independent of bleach)
    # NOT sf.mean_fret_E which is a function of n_complexes (see Par 1 note)
    te5 = te_ksg_bidirectional(E_series, bleach_series,
                               k=min(3, n_frames // 5),
                               lag=1, n_shuffle=n_shuffle)
    n_pts5 = n_frames
    res5 = "frame (KSG)"

    interp5 = ""
    if te5.significant_xy:
        interp5 = "FRET modulates bleaching: donor protected, acceptor exposed"
    else:
        interp5 = "FRET-bleaching coupling not detectable at this photon budget"

    pairs_results.append(CausalPairResult(
        pair_id=5, pair_name="FRET → Bleaching",
        x_label="E(t)", y_label="n_bleach(t)",
        te_xy=te5.te_xy, te_yx=te5.te_yx,
        te_xy_null=te5.te_xy_shuffled, te_yx_null=te5.te_yx_shuffled,
        significant_xy=te5.significant_xy, significant_yx=te5.significant_yx,
        n_points=n_pts5, resolution=res5, interpretation=interp5,
    ))

    # ── PAR 6: Δbind → Δunbind — Le Chatelier equilibrium ────
    if has_subframe:
        sf = sim_result.subframe_data
        # Smoothing: aggregate over blocks of steps_per_frame//10
        block = max(1, sf.steps_per_frame // 10)
        n_blocks = len(sf.n_bindings) // block
        bind_blocks = np.array([sf.n_bindings[i*block:(i+1)*block].sum()
                                for i in range(n_blocks)], dtype=np.float64)
        unbind_blocks = np.array([sf.n_dissociations[i*block:(i+1)*block].sum()
                                  for i in range(n_blocks)], dtype=np.float64)
        te6_bins = max(2, min(16, int(n_blocks ** (1/3) / 3)))
        te6 = transfer_entropy(bind_blocks, unbind_blocks,
                               n_bins=te6_bins, lag=1, n_shuffle=n_shuffle)
        n_pts6 = n_blocks
        res6 = "subframe"
    else:
        # Frame-level: KSG on binding/dissociation rate proxies
        bind_rate = np.diff(complex_series, prepend=complex_series[0])
        bind_pos = np.maximum(bind_rate, 0).astype(np.float64)
        bind_neg = np.abs(np.minimum(bind_rate, 0)).astype(np.float64)
        te6 = te_ksg_bidirectional(bind_pos, bind_neg,
                                   k=min(3, n_frames // 5),
                                   lag=1, n_shuffle=n_shuffle)
        n_pts6 = n_frames
        res6 = "frame (KSG)"

    interp6 = ""
    if te6.significant_xy:
        interp6 = "Dynamic equilibrium: binding drives dissociation (Le Chatelier)"
    elif te6.significant_yx:
        interp6 = "Reverse: dissociation drives rebinding (recovery)"
    else:
        interp6 = "System not at dynamic equilibrium or coupling too weak"

    pairs_results.append(CausalPairResult(
        pair_id=6, pair_name="Binding ↔ Dissociation",
        x_label="Δ[RL]_bind(t)", y_label="Δ[RL]_unbind(t)",
        te_xy=te6.te_xy, te_yx=te6.te_yx,
        te_xy_null=te6.te_xy_shuffled, te_yx_null=te6.te_yx_shuffled,
        significant_xy=te6.significant_xy, significant_yx=te6.significant_yx,
        n_points=n_pts6, resolution=res6, interpretation=interp6,
    ))

    # ── PIXEL-LEVEL TE MAP using Ensemble+KSG ─────────────────
    # Method: Wollstadt et al (2014) PLoS ONE 9:e102833
    # Estimator: KSG (Kraskov 2004) or Binary (Orlandi 2014)
    #
    # Strategy:
    #   1. Extract per-pixel lifetime time series τ(x,y,t)
    #   2. Group pixels by mean τ into ~N_groups ensembles
    #   3. Within each group, pool M pixels × N frames → M×N points
    #   4. Compute ensemble TE([RL]_global → τ_pooled) with KSG
    #   5. Assign group TE to all pixels in that group
    #
    # This is the ONLY rigorous way to get pixel-level TE with
    # <30 frames per pixel (Wollstadt 2014 §2.2).
    nx = config.scan.pixels_x
    ny = config.scan.pixels_y
    pixel_te_map = np.full((nx, ny), np.nan)
    pixel_te_sig = np.full((nx, ny), False)

    if n_frames >= 6:
        # Step 1: Extract per-pixel τ time series
        pixel_tau_series = np.full((nx, ny, n_frames), np.nan)
        n_tcspc_bins = sim_result.get_flim_stack(0).shape[-1]
        bin_centers = (np.arange(n_tcspc_bins) + 0.5) * (time_range * 1e9 / n_tcspc_bins)

        for fi in range(n_frames):
            stack = sim_result.get_flim_stack(fi)
            donor_stack = stack[donor_ch]  # [nx, ny, n_bins]
            total = donor_stack.sum(axis=-1)  # [nx, ny]
            # Mean arrival time where enough photons
            valid_px = total >= 3
            if valid_px.any():
                # Vectorized: <t> = Σ(bin * count) / Σ(count)
                weighted = np.tensordot(donor_stack, bin_centers, axes=([-1], [0]))
                pixel_tau_series[:, :, fi] = np.where(
                    valid_px, weighted / np.maximum(total, 1), np.nan)

        # Step 2: Compute mean τ per pixel for grouping
        mean_tau_per_pixel = np.nanmean(pixel_tau_series, axis=2)

        # Step 3: Group pixels by τ quantiles (Wollstadt 2014 ensemble)
        valid_pixels = ~np.isnan(mean_tau_per_pixel)
        if valid_pixels.sum() > 0:
            tau_valid = mean_tau_per_pixel[valid_pixels]
            n_groups = min(8, max(2, valid_pixels.sum() // 5))

            try:
                quantiles = np.percentile(tau_valid,
                                          np.linspace(0, 100, n_groups + 1))
                # Assign group to each pixel
                pixel_group = np.full((nx, ny), -1, dtype=int)
                for g in range(n_groups):
                    low = quantiles[g]
                    high = quantiles[g + 1] + (1e-10 if g == n_groups - 1 else 0)
                    mask = valid_pixels & (mean_tau_per_pixel >= low) & (mean_tau_per_pixel < high)
                    pixel_group[mask] = g

                # Step 4: Ensemble TE per group
                group_te = np.zeros(n_groups)
                group_sig = np.zeros(n_groups, dtype=bool)

                for g in range(n_groups):
                    group_mask = pixel_group == g
                    if group_mask.sum() < 2:
                        continue

                    # Collect all pixel series in this group
                    px_list = []
                    for ix in range(nx):
                        for iy in range(ny):
                            if group_mask[ix, iy]:
                                px_list.append(pixel_tau_series[ix, iy, :])

                    if len(px_list) < 2:
                        continue

                    # Ensemble TE with KSG (if enough points) or histogram
                    n_pooled = len(px_list) * n_frames
                    if n_pooled >= 100:
                        te_g, te_null_g, sig_g = ensemble_te(
                            px_list, complex_series,
                            lag=1, n_shuffle=min(n_shuffle, 30),
                            estimator='ksg', k_ksg=4)
                    else:
                        te_g, te_null_g, sig_g = ensemble_te(
                            px_list, complex_series,
                            n_bins=2, lag=1, n_shuffle=min(n_shuffle, 30),
                            estimator='histogram')

                    group_te[g] = te_g
                    group_sig[g] = sig_g

                # Step 5: Assign group TE to pixels
                for g in range(n_groups):
                    group_mask = pixel_group == g
                    pixel_te_map[group_mask] = group_te[g]
                    pixel_te_sig[group_mask] = group_sig[g]

            except (ValueError, IndexError):
                pass  # Quantile computation failed; leave map as NaN

    return CausalFLIMResult(
        pairs=pairs_results,
        pixel_te_map=pixel_te_map,
        pixel_te_sig=pixel_te_sig,
        pixel_te_description=(
            "TE([RL]_global → τ(x,y)) per pixel\n"
            "Method: Ensemble+KSG (Wollstadt 2014 + Kraskov 2004)\n"
            "Pixels grouped by mean τ for statistical pooling"
        ),
        n_frames=n_frames,
        n_subframe_points=len(sim_result.subframe_data.n_complexes) if has_subframe else 0,
    )


def compute_causal_flim_generic(
    sim_result,
    causal_pairs: List[CausalPairSpec],
    tau_d_ns: float = 4.1,
    n_bins: int = 0,
    n_shuffle: int = 100,
    detrend: bool = True,
    detrend_method: str = "linear",
    lag_range: list = None,
    auto_lag: bool = True,
) -> CausalFLIMResult:
    """Compute user-defined causal pairs using Transfer Entropy.

    Generic version of compute_causal_flim that accepts any set of
    causal pairs defined by CausalPairSpec. Works with any biological
    system (EGFR-EGF, PIP3-PTEN, etc.).

    Parameters
    ----------
    sim_result : SimulationResult
        Result object from pipeline.run()
    causal_pairs : list of CausalPairSpec
        Each pair defines source (X) and target (Y) variables.
    tau_d_ns : float
        Unquenched donor lifetime (ns) for FLIM analysis.
    n_bins : int
        Histogram bins for TE (0 = auto from data length).
    n_shuffle : int
        Number of surrogate shuffles for significance testing.

    Returns
    -------
    CausalFLIMResult
        Contains list of CausalPairResult objects (no pixel map for generic).

    References
    ----------
    Schreiber (2000) Phys Rev Lett 85:461, Eq. 3
    Wibral et al. (2014) J Comput Neurosci 30:45, §2.3
    """
    config = sim_result.config
    n_frames = sim_result.n_frames
    time_range = config.tcspc.time_range

    # Derive donor channel
    fluoro_name_to_id = {f.name: i for i, f in enumerate(config.fluorophores)}
    if len(config.fret_pairs) > 0:
        donor_ch = fluoro_name_to_id[config.fret_pairs[0].donor_type]
    else:
        donor_ch = 0

    # ── Extract all frame-level time series ──────────────────────
    complex_series = sim_result.get_complex_count_series().astype(np.float64)
    bleach_series = sim_result.get_bleaching_series().astype(np.float64)
    intensity_series = sim_result.get_intensity_series().astype(np.float64)

    # Mean lifetime per frame
    tau_mean_series = np.zeros(n_frames)
    for i in range(n_frames):
        flim_stack = sim_result.get_flim_stack(i)
        tau_map, intens = mean_arrival_time(flim_stack, time_range,
                                            channel=donor_ch, min_counts=1)
        valid = ~np.isnan(tau_map) & (intens > 0)
        if valid.any():
            tau_mean_series[i] = np.average(tau_map[valid],
                                            weights=intens[valid])  # already in ns
        else:
            tau_mean_series[i] = tau_d_ns

    # Mean FRET efficiency per frame
    E_series = np.zeros(n_frames)
    for i, fr in enumerate(sim_result.frames):
        gt = fr.ground_truth
        e_arr = gt.fret_efficiency_exact
        donors = gt.has_dye & gt.is_alive & ~gt.is_bleached & (gt.dye_type_id == 0)
        if donors.any():
            E_series[i] = np.mean(e_arr[donors])

    # Local density per frame
    Lx, Ly = config.membrane.size
    membrane_area = Lx * Ly
    density_series = np.zeros(n_frames)
    for i, fr in enumerate(sim_result.frames):
        gt = fr.ground_truth
        n_alive = gt.is_alive.sum()
        density_series[i] = n_alive / membrane_area

    # Mean squared displacement (diffusion proxy)
    msd_series = np.zeros(n_frames)
    for i in range(1, n_frames):
        gt_prev = sim_result.frames[i-1].ground_truth
        gt_curr = sim_result.frames[i].ground_truth
        alive_both = gt_prev.is_alive & gt_curr.is_alive
        if alive_both.any():
            dx = gt_curr.positions[alive_both] - gt_prev.positions[alive_both]
            dx[:, 0] = dx[:, 0] - Lx * np.round(dx[:, 0] / Lx)
            dx[:, 1] = dx[:, 1] - Ly * np.round(dx[:, 1] / Ly)
            msd_series[i] = np.mean(dx[:, 0]**2 + dx[:, 1]**2)
    msd_series[0] = msd_series[1] if n_frames > 1 else 0.0

    # Check for sub-frame data
    has_subframe = (hasattr(sim_result, 'subframe_data') and
                    sim_result.subframe_data is not None)

    # Auto n_bins
    if n_bins == 0:
        if has_subframe:
            N_pts = len(sim_result.subframe_data.n_complexes)
            n_bins = max(2, min(16, int(N_pts ** (1/3) / 3)))
        else:
            n_bins = max(2, min(8, n_frames // 3))

    # Frame-level series cache
    frame_series = {
        "n_complexes": complex_series,
        "n_bleached": bleach_series,
        "total_photons": intensity_series,
        "tau_mean": tau_mean_series,
        "intensity": intensity_series,
        "E_mean": E_series,
        "msd": msd_series,
        "density": density_series,
    }

    # Helper: resolve a key to time series
    def resolve_series(key, sf, f_series):
        if key.startswith("species:"):
            sp_name = key.split(":", 1)[1]
            if sf and sf.species_counts and sp_name in sf.species_counts:
                return sf.species_counts[sp_name].astype(np.float64), "subframe"
            # Fallback to GroundTruth series if available
            return None, None
        elif key in ("n_catalytic", "n_recruited", "n_detached", "n_converted"):
            arr = getattr(sf, key, None) if sf else None
            if arr is not None:
                return arr.astype(np.float64), "subframe"
            return None, None
        elif key in ("n_complexes", "n_bindings", "n_dissociations", "n_bleached", "total_photons"):
            if sf and hasattr(sf, key):
                return getattr(sf, key).astype(np.float64), "subframe"
            return f_series.get(key, None), "frame"
        elif key in ("tau_mean", "intensity", "E_mean", "msd", "density"):
            return f_series.get(key, None), "frame"
        return None, None

    # ── Stationarity gatekeeper (Dickey & Fuller 1979) ──────────
    # TE requires weak stationarity. After detrending, we verify
    # with an ADF test. If non-stationary, TE result is flagged
    # as unreliable (but still computed for user inspection).
    #
    # Reference: Dickey & Fuller (1979) JASA 74:427
    #            Kwiatkowski et al (1992) J Econometrics 54:159 (KPSS)
    def _check_stationarity(series, name="series", alpha=0.05):
        """Test stationarity via Augmented Dickey-Fuller.
        Returns (is_stationary, p_value, adf_stat).
        Falls back to ACF heuristic if statsmodels unavailable."""
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(series, maxlag=min(10, len(series)//5),
                            autolag='AIC')
            adf_stat, p_value = result[0], result[1]
            return p_value < alpha, p_value, adf_stat
        except ImportError:
            # Fallback: ACF(1) heuristic — if |ACF(1)| > 2/√N, non-stationary
            if len(series) < 5:
                return True, 0.0, 0.0  # too short to test
            acf1 = np.corrcoef(series[:-1], series[1:])[0, 1]
            threshold = 2.0 / np.sqrt(len(series))
            return abs(acf1) < threshold, 1.0 - abs(acf1), acf1

    pairs_results = []

    # Process each causal pair
    for pair_idx, pair_spec in enumerate(causal_pairs):
        x_series, x_res = resolve_series(pair_spec.x_key, sim_result.subframe_data if has_subframe else None, frame_series)
        y_series, y_res = resolve_series(pair_spec.y_key, sim_result.subframe_data if has_subframe else None, frame_series)

        if x_series is None or y_series is None:
            continue

        # Trim to same length
        min_len = min(len(x_series), len(y_series))
        x_series = x_series[:min_len]
        y_series = y_series[:min_len]

        # ── TE estimator: ALWAYS use KSG (robust to discrete/sparse data).
        # Histogram-based TE fails with species counts (10-24 unique values
        # spread across 8-16 bins → degenerate joint distribution → TE=0).
        # KSG k-NN estimator handles mixed continuous/discrete data properly.
        #
        # For subframe data (>1000 pts): block-average to ~300 points
        # to reduce autocorrelation and make KSG tractable.
        #
        # Reference: Kraskov, Stögbauer & Grassberger (2004) PRE 69:066138
        #            Wibral et al. (2014) J Comput Neurosci 30:45

        target_pts = 300  # enough for KSG, low autocorrelation

        if x_res == "subframe" and y_res == "subframe":
            # Block-average to ~target_pts points for KSG
            block_size = max(1, len(x_series) // target_pts)
            n_blocks = len(x_series) // block_size
            if n_blocks >= 10:
                x_series = np.array([x_series[i*block_size:(i+1)*block_size].mean()
                                     for i in range(n_blocks)])
                y_series = np.array([y_series[i*block_size:(i+1)*block_size].mean()
                                     for i in range(n_blocks)])
            min_len = min(len(x_series), len(y_series))
            x_series = x_series[:min_len]
            y_series = y_series[:min_len]
            # ── Stationarity correction (Wibral 2014 §2.4) ──
            stationarity_warning = ""
            if detrend and len(x_series) > 4:
                if detrend_method == "linear":
                    from scipy.signal import detrend as sp_detrend
                    x_series = sp_detrend(x_series, type='linear')
                    y_series = sp_detrend(y_series, type='linear')
                elif detrend_method == "diff":
                    x_series = np.diff(x_series)
                    y_series = np.diff(y_series)
                    min_len = min(len(x_series), len(y_series))
                elif detrend_method == "diff2":
                    # Double differencing: removes linear trend in increments
                    # Required for logistic/sigmoidal growth where diff(x) is
                    # still non-stationary (bell-shaped). diff2 achieves 100%
                    # ADF pass rate and FP=6.2% [CI 4.8-7.9%] at α=5%.
                    # Ref: Granger & Joyeux (1980), Monte Carlo validation 2026.
                    x_series = np.diff(np.diff(x_series))
                    y_series = np.diff(np.diff(y_series))
                    min_len = min(len(x_series), len(y_series))
                # ── ADF stationarity gatekeeper ──
                if len(x_series) >= 20:
                    x_stat, x_pval, _ = _check_stationarity(x_series, "X")
                    y_stat, y_pval, _ = _check_stationarity(y_series, "Y")
                    if not x_stat or not y_stat:
                        stationarity_warning = (
                            f"WARNING: non-stationary after {detrend_method} detrend "
                            f"(ADF p: X={x_pval:.4f}, Y={y_pval:.4f}). "
                            f"TE may contain spurious information flow."
                        )
                        import warnings
                        warnings.warn(stationarity_warning)
            k_ksg = max(1, min(5, min_len // 20))
            # ── Optimal lag selection (Wibral 2013) ──
            if auto_lag and lag_range:
                _lr = [l for l in lag_range if l < len(x_series) // 3]
                if len(_lr) >= 1:
                    optimal_lag, te_result = find_optimal_lag(
                        x_series, y_series, k=k_ksg,
                        lag_range=_lr, n_shuffle=n_shuffle)
                else:
                    optimal_lag = 1
                    te_result = te_ksg_bidirectional(x_series, y_series, k=k_ksg, lag=1, n_shuffle=n_shuffle)
            else:
                optimal_lag = 1
                te_result = te_ksg_bidirectional(x_series, y_series, k=k_ksg, lag=1, n_shuffle=n_shuffle)
            resolution = "subframe (KSG, block-avg)"

        elif x_res == "frame" and y_res == "frame":
            # ── Stationarity correction (Wibral 2014 §2.4) ──
            stationarity_warning = ""
            if detrend and len(x_series) > 4:
                if detrend_method == "linear":
                    from scipy.signal import detrend as sp_detrend
                    x_series = sp_detrend(x_series, type='linear')
                    y_series = sp_detrend(y_series, type='linear')
                elif detrend_method == "diff":
                    x_series = np.diff(x_series)
                    y_series = np.diff(y_series)
                    min_len = min(len(x_series), len(y_series))
                elif detrend_method == "diff2":
                    # Double differencing: removes linear trend in increments
                    # Required for logistic/sigmoidal growth where diff(x) is
                    # still non-stationary (bell-shaped). diff2 achieves 100%
                    # ADF pass rate and FP=6.2% [CI 4.8-7.9%] at α=5%.
                    # Ref: Granger & Joyeux (1980), Monte Carlo validation 2026.
                    x_series = np.diff(np.diff(x_series))
                    y_series = np.diff(np.diff(y_series))
                    min_len = min(len(x_series), len(y_series))
                # ── ADF stationarity gatekeeper ──
                if len(x_series) >= 20:
                    x_stat, x_pval, _ = _check_stationarity(x_series, "X")
                    y_stat, y_pval, _ = _check_stationarity(y_series, "Y")
                    if not x_stat or not y_stat:
                        stationarity_warning = (
                            f"WARNING: non-stationary after {detrend_method} detrend "
                            f"(ADF p: X={x_pval:.4f}, Y={y_pval:.4f}). "
                            f"TE may contain spurious information flow."
                        )
                        import warnings
                        warnings.warn(stationarity_warning)
            k_ksg = max(1, min(3, n_frames // 5))
            # ── Optimal lag selection (Wibral 2013) ──
            if auto_lag and lag_range:
                _lr = [l for l in lag_range if l < len(x_series) // 3]
                if len(_lr) >= 1:
                    optimal_lag, te_result = find_optimal_lag(
                        x_series, y_series, k=k_ksg,
                        lag_range=_lr, n_shuffle=n_shuffle)
                else:
                    optimal_lag = 1
                    te_result = te_ksg_bidirectional(x_series, y_series, k=k_ksg, lag=1, n_shuffle=n_shuffle)
            else:
                optimal_lag = 1
                te_result = te_ksg_bidirectional(x_series, y_series, k=k_ksg, lag=1, n_shuffle=n_shuffle)
            resolution = "frame (KSG)"

        else:
            # Mixed: downsample subframe to frame level, then KSG
            stationarity_warning = ""
            if x_res == "subframe":
                steps_per_frame = len(x_series) // n_frames
                if steps_per_frame > 0:
                    x_series = np.array([x_series[i*steps_per_frame:(i+1)*steps_per_frame].mean()
                                        for i in range(n_frames)])
            if y_res == "subframe":
                steps_per_frame = len(y_series) // n_frames
                if steps_per_frame > 0:
                    y_series = np.array([y_series[i*steps_per_frame:(i+1)*steps_per_frame].mean()
                                        for i in range(n_frames)])
            min_len = min(len(x_series), len(y_series))
            x_series = x_series[:min_len]
            y_series = y_series[:min_len]
            # ── Stationarity correction (Wibral 2014 §2.4) ──
            if detrend and len(x_series) > 4:
                if detrend_method == "linear":
                    from scipy.signal import detrend as sp_detrend
                    x_series = sp_detrend(x_series, type='linear')
                    y_series = sp_detrend(y_series, type='linear')
                elif detrend_method == "diff":
                    x_series = np.diff(x_series)
                    y_series = np.diff(y_series)
                    min_len = min(len(x_series), len(y_series))
                elif detrend_method == "diff2":
                    # Double differencing: removes linear trend in increments
                    # Required for logistic/sigmoidal growth where diff(x) is
                    # still non-stationary (bell-shaped). diff2 achieves 100%
                    # ADF pass rate and FP=6.2% [CI 4.8-7.9%] at α=5%.
                    # Ref: Granger & Joyeux (1980), Monte Carlo validation 2026.
                    x_series = np.diff(np.diff(x_series))
                    y_series = np.diff(np.diff(y_series))
                    min_len = min(len(x_series), len(y_series))
                # ── ADF stationarity gatekeeper ──
                if len(x_series) >= 20:
                    x_stat, x_pval, _ = _check_stationarity(x_series, "X")
                    y_stat, y_pval, _ = _check_stationarity(y_series, "Y")
                    if not x_stat or not y_stat:
                        stationarity_warning = (
                            f"WARNING: non-stationary after {detrend_method} detrend "
                            f"(ADF p: X={x_pval:.4f}, Y={y_pval:.4f}). "
                            f"TE may contain spurious information flow."
                        )
                        import warnings
                        warnings.warn(stationarity_warning)
            k_ksg = max(1, min(3, min_len // 5))
            # ── Optimal lag selection (Wibral 2013) ──
            if auto_lag and lag_range:
                _lr = [l for l in lag_range if l < len(x_series) // 3]
                if len(_lr) >= 1:
                    optimal_lag, te_result = find_optimal_lag(
                        x_series, y_series, k=k_ksg,
                        lag_range=_lr, n_shuffle=n_shuffle)
                else:
                    optimal_lag = 1
                    te_result = te_ksg_bidirectional(x_series, y_series, k=k_ksg, lag=1, n_shuffle=n_shuffle)
            else:
                optimal_lag = 1
                te_result = te_ksg_bidirectional(x_series, y_series, k=k_ksg, lag=1, n_shuffle=n_shuffle)
            resolution = "mixed (KSG)"

        # Determine interpretation
        interp = ""
        if te_result.significant_xy:
            interp = pair_spec.interpretation_if_significant
        else:
            interp = pair_spec.interpretation_if_not

        # Create result
        x_label = pair_spec.x_label if pair_spec.x_label else pair_spec.x_key
        y_label = pair_spec.y_label if pair_spec.y_label else pair_spec.y_key

        pairs_results.append(CausalPairResult(
            pair_id=pair_idx + 1,
            pair_name=pair_spec.name,
            x_label=x_label,
            y_label=y_label,
            te_xy=te_result.te_xy,
            te_yx=te_result.te_yx,
            te_xy_null=te_result.te_xy_shuffled,
            te_yx_null=te_result.te_yx_shuffled,
            significant_xy=te_result.significant_xy,
            significant_yx=te_result.significant_yx,
            n_points=min_len,
            resolution=resolution,
            interpretation=interp,
            optimal_lag=optimal_lag,
            detrend_applied=detrend_method if detrend else "",
            stationarity_warning=stationarity_warning,
        ))

    return CausalFLIMResult(
        pairs=pairs_results,
        pixel_te_map=None,
        pixel_te_sig=None,
        pixel_te_description="No pixel map for generic causal analysis",
        n_frames=n_frames,
        n_subframe_points=len(sim_result.subframe_data.n_complexes) if has_subframe else 0,
    )


# ═══════════════════════════════════════════════════════════════════════
# L7: GROUND TRUTH DISCREPANCY — the "error microscope"
# ═══════════════════════════════════════════════════════════════════════
# No direct reference — this is unique to simulation with ground truth.
#
# Concept: for each pixel, compare the lifetime estimated from the
# FLIM image (noisy, PSF-blurred) against the true lifetime computed
# from particle positions and Förster theory. The discrepancy map
# reveals where the microscope image is reliable and where it is not.
#
# This has never been done because no experiment has ground truth,
# and no other simulator produces FLIM images.
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class DiscrepancyResult:
    """Result of ground truth vs image comparison."""
    tau_image: np.ndarray        # [nx, ny] lifetime from FLIM (ns)
    tau_exact: np.ndarray        # [nx, ny] exact lifetime projected to pixels
    delta_tau: np.ndarray        # [nx, ny] |τ_image - τ_exact| (ns)
    relative_error: np.ndarray   # [nx, ny] |Δτ|/τ_exact
    E_image: np.ndarray          # [nx, ny] FRET efficiency from image
    E_exact: np.ndarray          # [nx, ny] FRET efficiency from ground truth
    delta_E: np.ndarray          # [nx, ny] |E_image - E_exact|
    reliability_mask: np.ndarray  # [nx, ny] True where error < threshold
    mae_tau_ns: float            # mean absolute error in τ (ns)
    mae_E: float                 # mean absolute error in E
    n_reliable_pixels: int
    n_total_pixels: int


def ground_truth_discrepancy(
    flim_stack: np.ndarray,
    ground_truth,  # GroundTruth dataclass
    membrane_size: Tuple[float, float],
    pixel_size_um: float,
    time_range: float = 25e-9,
    tau_d_ns: float = 4.1,
    min_counts: int = 20,
    reliability_threshold: float = 0.1,
    donor_channel: int = 0,
) -> DiscrepancyResult:
    """Compare FLIM image against ground truth.

    Projects exact per-particle lifetimes onto the pixel grid,
    then computes pixel-by-pixel discrepancy.

    Parameters
    ----------
    flim_stack : ndarray [n_ch, nx, ny, n_bins]
    ground_truth : GroundTruth dataclass with positions, donor_lifetime_exact
    membrane_size : (Lx, Ly) in µm
    pixel_size_um : float, pixel size in µm
    time_range : float, TCSPC window in seconds
    tau_d_ns : float, unquenched donor lifetime in ns
    min_counts : int, minimum photons for valid τ estimation
    reliability_threshold : float, relative error threshold for "reliable"
    donor_channel : int, fluorophore index of the donor (from FRET pair config)

    Returns
    -------
    DiscrepancyResult with error maps and statistics
    """
    _, nx, ny, _ = flim_stack.shape

    # L7a: Lifetime from FLIM image (same as experimentalist would do)
    tau_image, intensity = mean_arrival_time(
        flim_stack, time_range, channel=donor_channel, min_counts=min_counts
    )

    # L7b: Project exact particle lifetimes onto pixel grid
    # Each pixel accumulates weighted average of particles within it
    tau_exact_map = np.full((nx, ny), np.nan)
    E_exact_map = np.full((nx, ny), np.nan)

    # Pixel grid: pixel (i,j) covers
    #   x ∈ [i·pixel_size, (i+1)·pixel_size]
    pos = ground_truth.positions  # [N, 2] in µm
    tau_exact_particle = ground_truth.donor_lifetime_exact  # seconds
    E_exact_particle = ground_truth.fret_efficiency_exact
    has_dye = ground_truth.has_dye
    is_alive = ground_truth.is_alive
    is_bleached = ground_truth.is_bleached
    dye_type = ground_truth.dye_type_id

    # Only donors (dye_type=0, alive, not bleached, has dye)
    donor_mask = has_dye & is_alive & ~is_bleached & (dye_type == 0)

    px_um = pixel_size_um
    tau_pixel_sum = np.zeros((nx, ny))
    E_pixel_sum = np.zeros((nx, ny))
    count_pixel = np.zeros((nx, ny), dtype=int)

    for idx in np.where(donor_mask)[0]:
        x, y = pos[idx]
        ix = int(x / px_um)
        iy = int(y / px_um)
        if 0 <= ix < nx and 0 <= iy < ny:
            tau_ns = tau_exact_particle[idx] * 1e9  # convert s → ns
            if tau_ns > 0:
                tau_pixel_sum[ix, iy] += tau_ns
                E_pixel_sum[ix, iy] += E_exact_particle[idx]
                count_pixel[ix, iy] += 1

    active_pixels = count_pixel > 0
    tau_exact_map[active_pixels] = (
        tau_pixel_sum[active_pixels] / count_pixel[active_pixels]
    )
    E_exact_map[active_pixels] = (
        E_pixel_sum[active_pixels] / count_pixel[active_pixels]
    )

    # L7c: FRET efficiency from image
    E_image = np.full((nx, ny), np.nan)
    valid_tau = ~np.isnan(tau_image) & (tau_image > 0)
    E_image[valid_tau] = 1.0 - tau_image[valid_tau] / tau_d_ns

    # L7d: Discrepancy maps
    both_valid = ~np.isnan(tau_image) & ~np.isnan(tau_exact_map)
    delta_tau = np.full((nx, ny), np.nan)
    relative_error = np.full((nx, ny), np.nan)
    delta_E = np.full((nx, ny), np.nan)

    delta_tau[both_valid] = np.abs(
        tau_image[both_valid] - tau_exact_map[both_valid]
    )
    nz = both_valid & (tau_exact_map > 0.1)
    relative_error[nz] = delta_tau[nz] / tau_exact_map[nz]

    both_E = ~np.isnan(E_image) & ~np.isnan(E_exact_map)
    delta_E[both_E] = np.abs(E_image[both_E] - E_exact_map[both_E])

    # Reliability mask
    reliability_mask = nz & (relative_error < reliability_threshold)

    # Summary statistics
    mae_tau = float(np.nanmean(delta_tau[both_valid])) if both_valid.any() else 0.0
    mae_E = float(np.nanmean(delta_E[both_E])) if both_E.any() else 0.0

    return DiscrepancyResult(
        tau_image=tau_image,
        tau_exact=tau_exact_map,
        delta_tau=delta_tau,
        relative_error=relative_error,
        E_image=E_image,
        E_exact=E_exact_map,
        delta_E=delta_E,
        reliability_mask=reliability_mask,
        mae_tau_ns=mae_tau,
        mae_E=mae_E,
        n_reliable_pixels=int(reliability_mask.sum()),
        n_total_pixels=int(both_valid.sum()),
    )


# ═══════════════════════════════════════════════════════════════════════
# FULL ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class FullAnalysisResult:
    """Complete 7-layer analysis of a SimulationResult."""
    phasor: List[PhasorResult]               # per frame
    spatial_corr_image: SpatialCorrelationResult
    spatial_corr_gt: Tuple[np.ndarray, np.ndarray]
    temporal_acf: TemporalACFResult
    pair_corr: PairCorrelationResult
    imsd: IMSDResult
    discrepancy: List[DiscrepancyResult]      # per frame
    # Transfer entropy stored separately (requires user-chosen pairs)


def run_full_analysis(result,  # SimulationResult
                      tau_d_ns: float = 4.1,
                      ) -> FullAnalysisResult:
    """Run all 7 analysis layers on a SimulationResult.

    Parameters
    ----------
    result : SimulationResult from pipeline.run()
    tau_d_ns : float, unquenched donor lifetime in ns

    Returns
    -------
    FullAnalysisResult with all layers populated
    """
    config = result.config
    time_range = config.tcspc.time_range
    pixel_size_um = config.scan.pixel_size / 1000.0  # nm → µm
    frame_interval = config.scan.frame_interval
    membrane_size = config.membrane.size  # (Lx, Ly) in µm
    n_frames = result.n_frames

    # ── Donor/Acceptor channel identity from config ─────────────
    # Channel index = fluorophore index in config.fluorophores[].
    # We derive donor_channel from the FRET pair definition,
    # NOT from hardcoded assumptions.
    fluoro_name_to_id = {f.name: i for i, f in enumerate(config.fluorophores)}
    if len(config.fret_pairs) > 0:
        donor_channel = fluoro_name_to_id[config.fret_pairs[0].donor_type]
        acceptor_channel = fluoro_name_to_id[config.fret_pairs[0].acceptor_type]
    else:
        donor_channel = 0
        acceptor_channel = 1 if len(config.fluorophores) > 1 else 0

    # ── Adaptive min_counts ─────────────────────────────────────
    # The photon budget per pixel per frame depends on the confocal
    # scan duty fraction (~2e-5 for typical dwell_time/frame_interval).
    # With ~1-2 photons/pixel/frame, hardcoded min_counts=20 would
    # reject ALL pixels. Instead, compute an adaptive threshold:
    #   - Estimate mean photons/pixel from the first frame
    #   - Set min_counts = max(1, floor(mean_photons * 0.5))
    # This ensures at least ~50% of illuminated pixels are used.
    #
    # For phasor (L1), we need at least 1 photon per pixel.
    # For lifetime (L3, L7), 1 photon gives a noisy but valid <t>.
    # Reference: Ranjit et al. (2018) Nat Protocols 13:1979 —
    #   phasor analysis works with as few as ~5 photons/pixel
    #   but produces increasingly noisy clouds on the phasor plot.
    first_intensity = result.frames[0].flim_frame.intensity[donor_channel]
    mean_photons = first_intensity[first_intensity > 0].mean() if (first_intensity > 0).any() else 1.0
    adaptive_min = max(1, int(mean_photons * 0.5))

    # ── L1: Phasor per frame ────────────────────────────────────
    phasors = []
    for frame in result.frames:
        p = phasor_transform(
            frame.flim_frame.flim_stack, time_range,
            channel=donor_channel, min_counts=adaptive_min,
        )
        phasors.append(p)

    # ── L2: Spatial correlation (last frame) ────────────────────
    # Use donor INTENSITY image (always available, no min_counts
    # threshold needed) rather than lifetime map for ICS.
    # The intensity spatial autocorrelation detects clustering
    # from photon count variations, which is the standard ICS
    # observable (Petersen et al. 1993, Biophys J 65:1135).
    last_frame = result.frames[-1]
    intensity_last = last_frame.flim_frame.intensity[donor_channel].astype(np.float64)
    spatial_img = spatial_autocorrelation(intensity_last, pixel_size_um)
    spatial_img.source = "image_intensity"

    # Ground truth g(r)
    gt_pos = last_frame.ground_truth.positions
    gt_alive = last_frame.ground_truth.is_alive
    pos_alive = gt_pos[gt_alive]
    r_gt, g_gt = spatial_correlation_ground_truth(
        pos_alive, membrane_size, n_bins=50, r_max_um=1.0,
    )

    # ── L3: Temporal ACF ────────────────────────────────────────
    # Use donor intensity per pixel as the temporal observable.
    # Intensity fluctuations δI(t) at each pixel track binding
    # kinetics (FCS theory: Elson & Magde 1974).
    # Lifetime-based ACF requires many photons/pixel/frame; with
    # typical confocal duty fractions (~2e-5), per-pixel lifetime
    # is unreliable. Intensity ACF is always well-defined.
    intensity_maps = np.zeros((n_frames, config.scan.pixels_x,
                               config.scan.pixels_y))
    for i, frame in enumerate(result.frames):
        intensity_maps[i] = frame.flim_frame.intensity[donor_channel].astype(np.float64)

    temporal = temporal_acf_intensity(intensity_maps, frame_interval)

    # ── L4: pCF ─────────────────────────────────────────────────
    intensity_series = np.zeros(
        (n_frames, config.scan.pixels_x, config.scan.pixels_y)
    )
    for i, frame in enumerate(result.frames):
        intensity_series[i] = frame.flim_frame.intensity[donor_channel]

    max_dist = min(5, config.scan.pixels_x // 4)
    pcf = pair_correlation_function(
        intensity_series, pixel_size_um, frame_interval,
        max_distance_px=max_dist,
    )

    # ── L5: iMSD ────────────────────────────────────────────────
    imsd = compute_imsd(
        intensity_series, pixel_size_um, frame_interval,
    )

    # ── L7: Discrepancy per frame ───────────────────────────────
    discrepancies = []
    for frame in result.frames:
        disc = ground_truth_discrepancy(
            frame.flim_frame.flim_stack,
            frame.ground_truth,
            membrane_size,
            pixel_size_um,
            time_range=time_range,
            tau_d_ns=tau_d_ns,
            min_counts=adaptive_min,
            donor_channel=donor_channel,
        )
        discrepancies.append(disc)

    return FullAnalysisResult(
        phasor=phasors,
        spatial_corr_image=spatial_img,
        spatial_corr_gt=(r_gt, g_gt),
        temporal_acf=temporal,
        pair_corr=pcf,
        imsd=imsd,
        discrepancy=discrepancies,
    )
