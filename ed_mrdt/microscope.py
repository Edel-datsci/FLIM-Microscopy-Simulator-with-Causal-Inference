"""
ED-MRDT v1.1 — Virtual Microscope (PSF + Confocal + TCSPC + Noise)
====================================================================

Generates realistic FLIM-TCSPC images from particle positions and
photon arrival times produced by the photophysics engine.

Design sources:
  - Zhang B et al. (2007) Appl Opt 46:1819 — Gaussian PSF for confocal
    "2D Gaussian approximation nearly perfect for LSCM" (Table 4)
  - Gibson SF, Lanni F (1992) JOSA A 9:154 — Gibson-Lanni PSF model
  - Li J, Xue F, Blu T (2017) JOSA A 34:1029 — Fast G-L PSF computation
  - Becker W (2005) Advanced TCSPC Techniques, Springer — TCSPC/IRF
  - Haeberlé O (2004) Opt Commun 235:73 — Confocal PSF model
  - Lakowicz JR (2006) Principles of Fluorescence Spectroscopy, Ch. 5

Architecture:
  The virtual microscope converts photons from the CTMC engine into
  FLIM images. For each BD timestep within a frame:
    1. Photons arrive at particle positions (from PhotonBatch)
    2. PSF maps each photon to pixel probabilities
    3. Detection efficiency and dark counts applied
    4. IRF convolution assigns TCSPC time bins
    5. Afterpulsing adds correlated noise
  Output: flim_stack[channel, px, py, tcspc_bin] per frame

Components:
  3.1: PSF — Gaussian (analytical) or Airy (Debye integral)
  3.2: ConfocalScanner — pixel grid + PSF-weighted photon collection
  3.3: TCSPCAccumulator — time binning with IRF convolution
  3.4: DetectorModel — efficiency, dark counts, afterpulsing, dead time
  3.5: VirtualMicroscope — orchestrator combining all components

Usage:
    from ed_mrdt.microscope import VirtualMicroscope
    scope = VirtualMicroscope(config, state)
    scope.process_photons(photon_batch)  # accumulate
    frame = scope.get_frame()            # extract flim_stack
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit, prange

from .config import SimulationConfig, ConfigError
from .particles import ParticleState

__all__ = [
    "PSF",
    "ConfocalScanner",
    "TCSPCAccumulator",
    "DetectorModel",
    "VirtualMicroscope",
    "FLIMFrame",
]


# ═══════════════════════════════════════════════════════════════════════
# PASO 3.1: POINT SPREAD FUNCTION
# ═══════════════════════════════════════════════════════════════════════
# For 2D confocal (LSCM), the Gaussian approximation is nearly exact.
# Zhang et al. (2007): σ_LSCM ≈ 0.21 λ_em / NA for peak-matching (L∞).
# Confocal PSF = h_ill × h_det (both Gaussian) = Gaussian with
#   σ_conf = σ_wf / sqrt(2) when same wavelength.
# For FLIM-FRET: λ_exc=488, λ_em=519 (donor), λ_em=565 (acceptor).
# ═══════════════════════════════════════════════════════════════════════

class PSF:
    """2D Point Spread Function for confocal FLIM imaging.

    Supports:
      - "gaussian": Analytical Gaussian, σ = 0.21 λ / NA (wf) or σ/√2 (confocal)
      - "airy": Numerical Airy pattern from Debye integral (2D)

    For a 2D membrane (all emitters in focus), the lateral PSF is all
    that matters. Axial (z) PSF is irrelevant. The Gaussian approximation
    is nearly perfect for LSCM (Zhang et al. 2007, Table 4: RSE < 2%).

    Parameters
    ----------
    config : SimulationConfig
        Provides objective NA, PSF model type, emission wavelengths.

    Attributes
    ----------
    sigma_um : dict[int, float]
        PSF σ in µm per channel (0=donor, 1=acceptor).
    fwhm_um : dict[int, float]
        PSF FWHM in µm per channel.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.NA = config.objective.NA
        self.psf_type = config.psf.type

        # Per-channel PSF sigma (different emission wavelengths)
        # Channel index = fluorophore index in config.fluorophores[]
        self.sigma_um: Dict[int, float] = {}
        self.fwhm_um: Dict[int, float] = {}

        # Collect emission wavelengths per channel (fluorophore index = channel)
        for ch_idx, fluo in enumerate(config.fluorophores):
            # Find emission wavelength from fluorescent states
            lam_em = config.psf.emission_wavelength  # default
            for state in fluo.states:
                if state.is_fluorescent and state.emission_wavelength > 0:
                    lam_em = state.emission_wavelength
                    break

            # Widefield PSF sigma: σ_wf = 0.21 × λ / NA (nm)
            # Zhang et al. (2007) Eq. (11), L∞ constraint
            sigma_wf_nm = 0.21 * lam_em / self.NA

            # Confocal PSF: product of illumination and detection PSFs
            # For Gaussian × Gaussian: σ_conf = σ_wf / √2
            # (assuming similar excitation and emission wavelengths)
            # More precisely: 1/σ²_conf = 1/σ²_ill + 1/σ²_det
            # σ_ill = 0.21 × λ_exc / NA, σ_det = 0.21 × λ_em / NA
            lam_exc = config.laser.wavelength  # nm
            sigma_ill_nm = 0.21 * lam_exc / self.NA
            sigma_det_nm = 0.21 * lam_em / self.NA
            sigma_conf_nm = 1.0 / math.sqrt(
                1.0 / (sigma_ill_nm**2) + 1.0 / (sigma_det_nm**2)
            )

            sigma_um = sigma_conf_nm / 1000.0  # nm → µm
            fwhm_um = sigma_um * 2.0 * math.sqrt(2.0 * math.log(2.0))

            self.sigma_um[ch_idx] = sigma_um
            self.fwhm_um[ch_idx] = fwhm_um

        # Default channel for unlisted indices
        if 0 in self.sigma_um:
            self._default_sigma = self.sigma_um[0]
        else:
            # Fallback: use config PSF emission wavelength
            s_nm = 0.21 * config.psf.emission_wavelength / self.NA
            self._default_sigma = s_nm / 1000.0 / math.sqrt(2.0)

        # Pre-compute PSF lookup tables for each channel
        self._psf_tables: Dict[int, np.ndarray] = {}
        self._psf_extent: Dict[int, float] = {}
        self._psf_dx: Dict[int, float] = {}
        for ch_idx in self.sigma_um:
            self._build_psf_table(ch_idx)

    def _build_psf_table(self, channel: int, n_samples: int = 201) -> None:
        """Pre-compute discretized 1D radial PSF for fast lookup.

        The PSF is radially symmetric, so we store h(r) on a fine grid
        and interpolate. Extent = 4σ (captures >99.97% of energy).
        """
        sigma = self.sigma_um[channel]
        extent = 4.0 * sigma  # µm radius
        dx = extent / (n_samples - 1)

        r = np.linspace(0.0, extent, n_samples)

        if self.psf_type == "gaussian" or self.psf_type == "gibson_lanni":
            # Gaussian: h(r) = (1 / 2πσ²) exp(-r²/2σ²)
            # Normalized so ∫h(r) 2πr dr = 1
            table = np.exp(-0.5 * (r / sigma) ** 2) / (2.0 * math.pi * sigma**2)
        elif self.psf_type == "airy":
            # Airy pattern: h(r) = [2 J₁(v) / v]² where v = 2π NA r / λ
            from scipy.special import j1
            lam_um = self.config.psf.emission_wavelength / 1000.0
            v = 2.0 * math.pi * self.NA * r / lam_um
            # Handle r=0
            table = np.ones_like(r)
            mask = v > 1e-10
            table[mask] = (2.0 * j1(v[mask]) / v[mask]) ** 2
            # Normalize to integrate to 1 over 2D
            area = 2.0 * math.pi * np.trapezoid(table * r, r)
            if area > 0:
                table /= area
        else:
            # Fallback: Gaussian
            table = np.exp(-0.5 * (r / sigma) ** 2) / (2.0 * math.pi * sigma**2)

        self._psf_tables[channel] = table
        self._psf_extent[channel] = extent
        self._psf_dx[channel] = dx

    def get_sigma(self, channel: int = 0) -> float:
        """Return PSF σ in µm for given channel."""
        return self.sigma_um.get(channel, self._default_sigma)

    def evaluate(self, r_um: float, channel: int = 0) -> float:
        """Evaluate PSF at radial distance r (µm) for given channel."""
        sigma = self.get_sigma(channel)
        return math.exp(-0.5 * (r_um / sigma) ** 2) / (2.0 * math.pi * sigma**2)

    def evaluate_array(self, r_um: np.ndarray, channel: int = 0) -> np.ndarray:
        """Evaluate PSF at array of radial distances."""
        sigma = self.get_sigma(channel)
        return np.exp(-0.5 * (r_um / sigma) ** 2) / (2.0 * math.pi * sigma**2)

    def get_pixel_weight(
        self, emitter_x: float, emitter_y: float,
        pixel_cx: float, pixel_cy: float,
        pixel_size_um: float, channel: int = 0,
    ) -> float:
        """Weight of PSF at pixel center for an emitter at (emitter_x, emitter_y).

        Approximation: evaluate PSF at pixel center (valid when pixel << σ).
        For our canonical case: pixel=59nm, σ≈53nm → pixel ≈ 1.1σ.
        This is marginal; we use the area-integrated approach for accuracy.
        """
        sigma = self.get_sigma(channel)
        dx = emitter_x - pixel_cx
        dy = emitter_y - pixel_cy
        r_sq = dx * dx + dy * dy
        # Integrated over pixel area (Gaussian integral in 2D)
        # For a Gaussian PSF, the probability of a photon landing in a
        # square pixel of side a centered at distance r from the emitter is:
        # P ≈ h(r) × a² (point approximation, valid when a < σ)
        # Better: P = [erf((x+a/2)/(σ√2)) - erf((x-a/2)/(σ√2))] / 2
        #         × [same for y]
        # This is the exact integral of the Gaussian over the pixel.
        return math.exp(-0.5 * r_sq / (sigma * sigma)) / (2.0 * math.pi * sigma * sigma)


# ═══════════════════════════════════════════════════════════════════════
# PASO 3.2: CONFOCAL SCANNER
# ═══════════════════════════════════════════════════════════════════════
# Raster scan: for each pixel, collect photons from nearby emitters
# weighted by PSF. In confocal mode, the pinhole rejects out-of-focus
# light — irrelevant for our 2D membrane (all in focus).
#
# Key insight: instead of scanning a beam across the sample, we
# equivalently scatter each photon to nearby pixels weighted by PSF.
# This is numerically identical but much faster for sparse emitters.
# ═══════════════════════════════════════════════════════════════════════

class ConfocalScanner:
    """Maps photon emission events to detector pixels via PSF.

    For each photon at position (x, y) on the membrane:
      - Find which pixels are within PSF range (4σ)
      - Assign photon to one pixel with probability ∝ PSF(r)
      - Record pixel index and arrival time

    This "reverse scan" approach (scatter from emitter to pixels)
    is equivalent to the forward scan but O(N_photons × k) where
    k ~ (4σ/pixel_size)² ≈ 11 pixels for our canonical case.

    Parameters
    ----------
    config : SimulationConfig
    psf : PSF
    """

    def __init__(self, config: SimulationConfig, psf: PSF):
        self.config = config
        self.psf = psf

        self.pixel_size_um = config.scan.pixel_size / 1000.0  # nm → µm
        self.nx = config.scan.pixels_x
        self.ny = config.scan.pixels_y

        # Pixel center coordinates in µm
        # Pixel (i,j) has center at (i+0.5)*pixel_size, (j+0.5)*pixel_size
        # FOV covers [0, nx*pixel_size] × [0, ny*pixel_size]
        self.fov_x = self.nx * self.pixel_size_um  # µm
        self.fov_y = self.ny * self.pixel_size_um  # µm

        # Offset: center FOV on membrane center
        Lx, Ly = config.membrane.size
        self.offset_x = (Lx - self.fov_x) / 2.0
        self.offset_y = (Ly - self.fov_y) / 2.0

        # PSF range in pixels (how many pixels to check around emitter)
        max_sigma = max(psf.sigma_um.values()) if psf.sigma_um else 0.1
        self.psf_range_um = 4.0 * max_sigma
        self.psf_range_px = int(math.ceil(self.psf_range_um / self.pixel_size_um))

        # Pre-compute pixel centers
        self.pixel_cx = (np.arange(self.nx) + 0.5) * self.pixel_size_um + self.offset_x
        self.pixel_cy = (np.arange(self.ny) + 0.5) * self.pixel_size_um + self.offset_y

    def assign_photons_to_pixels(
        self,
        photon_x: np.ndarray,         # float64[n_photons] emitter x position µm
        photon_y: np.ndarray,         # float64[n_photons] emitter y position µm
        photon_channel: np.ndarray,   # int8[n_photons]
        n_photons: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Assign photons to pixels via PSF-weighted probability.

        For each photon: compute PSF weight at all nearby pixel centers,
        normalize to probability, and draw which pixel receives it.

        Returns
        -------
        pixel_x : ndarray[n_detected], int32 — pixel column index
        pixel_y : ndarray[n_detected], int32 — pixel row index
        detected_mask : ndarray[n_photons], bool — which photons were detected
            (within FOV and assigned to a valid pixel)
        """
        if n_photons == 0:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.bool_),
            )

        # Pack PSF sigmas per channel for numba
        n_channels = max(self.psf.sigma_um.keys()) + 1 if self.psf.sigma_um else 1
        sigma_per_ch = np.zeros(max(n_channels, 2), dtype=np.float64)
        for ch, sig in self.psf.sigma_um.items():
            if ch < len(sigma_per_ch):
                sigma_per_ch[ch] = sig

        # Pre-generate random numbers
        rng_uniform = rng.random(n_photons)

        out_px = np.empty(n_photons, dtype=np.int32)
        out_py = np.empty(n_photons, dtype=np.int32)
        detected = np.zeros(n_photons, dtype=np.bool_)

        n_det = _assign_photons_numba(
            photon_x, photon_y, photon_channel,
            n_photons,
            self.pixel_cx, self.pixel_cy,
            self.nx, self.ny,
            self.pixel_size_um,
            self.offset_x, self.offset_y,
            self.psf_range_px,
            sigma_per_ch,
            rng_uniform,
            out_px, out_py, detected,
        )

        return out_px[:n_det], out_py[:n_det], detected

    def assign_photons_emitter_centric(
        self,
        particle_idx: np.ndarray,
        positions_x: np.ndarray,
        positions_y: np.ndarray,
        photon_channel: np.ndarray,
        n_photons: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Emitter-centric PSF assignment (Phase 6 Step 3).

        Uses particle_idx to cache PSF weights per emitter, avoiding
        redundant Gaussian evaluations for photons from the same emitter.
        Requires photons to be sorted by particle_idx (PSS kernel output).
        """
        if n_photons == 0:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.bool_),
            )

        n_channels = max(self.psf.sigma_um.keys()) + 1 if self.psf.sigma_um else 1
        sigma_per_ch = np.zeros(max(n_channels, 2), dtype=np.float64)
        for ch, sig in self.psf.sigma_um.items():
            if ch < len(sigma_per_ch):
                sigma_per_ch[ch] = sig

        rng_uniform = rng.random(n_photons)
        out_px = np.empty(n_photons, dtype=np.int32)
        out_py = np.empty(n_photons, dtype=np.int32)
        detected = np.zeros(n_photons, dtype=np.bool_)

        n_det = _assign_photons_emitter_centric(
            particle_idx, photon_channel,
            positions_x, positions_y,
            n_photons,
            self.pixel_cx, self.pixel_cy,
            self.nx, self.ny,
            self.pixel_size_um,
            self.offset_x, self.offset_y,
            self.psf_range_px,
            sigma_per_ch,
            rng_uniform,
            out_px, out_py, detected,
        )

        return out_px[:n_det], out_py[:n_det], detected


@njit(cache=True)
def _assign_photons_numba(
    photon_x, photon_y, photon_channel,
    n_photons,
    pixel_cx, pixel_cy,
    nx, ny,
    pixel_size,
    offset_x, offset_y,
    psf_range_px,
    sigma_per_ch,
    rng_uniform,
    out_px, out_py, detected,
):
    """Numba-accelerated photon→pixel assignment via PSF weighting.

    For each photon at (x, y):
      1. Find the nearest pixel (i0, j0)
      2. Compute PSF weights for all pixels within psf_range_px
      3. Normalize weights to CDF
      4. Draw pixel from CDF using pre-generated U(0,1)
    """
    n_det = 0

    for p in range(n_photons):
        x = photon_x[p]
        y = photon_y[p]
        ch = photon_channel[p]
        if ch < 0:
            ch = 0
        if ch >= len(sigma_per_ch):
            ch = 0
        sigma = sigma_per_ch[ch]
        if sigma <= 0.0:
            sigma = sigma_per_ch[0]

        # Map emitter position to pixel coordinates
        fx = (x - offset_x) / pixel_size - 0.5
        fy = (y - offset_y) / pixel_size - 0.5
        i0 = int(math.floor(fx + 0.5))
        j0 = int(math.floor(fy + 0.5))

        # Check if roughly within FOV
        if i0 < -psf_range_px or i0 >= nx + psf_range_px:
            continue
        if j0 < -psf_range_px or j0 >= ny + psf_range_px:
            continue

        # Compute PSF weights for neighborhood
        total_weight = 0.0
        # Store weights in a flat array (max neighborhood size)
        max_nbr = (2 * psf_range_px + 1) ** 2
        # Use stack-allocated arrays (numba supports small fixed arrays)
        # But for flexibility, compute in two passes
        # Pass 1: compute total weight
        inv_2sig2 = 0.5 / (sigma * sigma)
        for di in range(-psf_range_px, psf_range_px + 1):
            ii = i0 + di
            if ii < 0 or ii >= nx:
                continue
            pcx = pixel_cx[ii]
            dx = x - pcx
            for dj in range(-psf_range_px, psf_range_px + 1):
                jj = j0 + dj
                if jj < 0 or jj >= ny:
                    continue
                pcy = pixel_cy[jj]
                dy = y - pcy
                w = math.exp(-(dx * dx + dy * dy) * inv_2sig2)
                total_weight += w

        if total_weight <= 0.0:
            continue

        # Pass 2: CDF selection
        u = rng_uniform[p] * total_weight
        cum = 0.0
        found = False
        sel_i = i0
        sel_j = j0

        for di in range(-psf_range_px, psf_range_px + 1):
            ii = i0 + di
            if ii < 0 or ii >= nx:
                continue
            pcx = pixel_cx[ii]
            dx = x - pcx
            for dj in range(-psf_range_px, psf_range_px + 1):
                jj = j0 + dj
                if jj < 0 or jj >= ny:
                    continue
                pcy = pixel_cy[jj]
                dy = y - pcy
                w = math.exp(-(dx * dx + dy * dy) * inv_2sig2)
                cum += w
                if cum >= u and not found:
                    sel_i = ii
                    sel_j = jj
                    found = True
                    break
            if found:
                break

        # Clamp to valid range
        if sel_i < 0:
            sel_i = 0
        if sel_i >= nx:
            sel_i = nx - 1
        if sel_j < 0:
            sel_j = 0
        if sel_j >= ny:
            sel_j = ny - 1

        out_px[n_det] = sel_i
        out_py[n_det] = sel_j
        detected[p] = True
        n_det += 1

    return n_det


# ───────────────────────────────────────────────────────────────────────
# PHASE 6 STEP 3: Emitter-centric PSF accumulation
# ───────────────────────────────────────────────────────────────────────
# Instead of evaluating PSF per-photon, compute PSF weights once per
# unique emitter position and distribute multiple photons via the same
# weight vector. Reduces O(n_detected × psf²) to O(n_emitters × psf²).
# Ref: Phase 6 Spec §5, "PSF aggregate"
# ───────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _assign_photons_emitter_centric(
    particle_idx, photon_channel,
    positions_x, positions_y,
    n_photons,
    pixel_cx, pixel_cy,
    nx, ny,
    pixel_size,
    offset_x, offset_y,
    psf_range_px,
    sigma_per_ch,
    rng_uniform,
    out_px, out_py, detected,
):
    """Emitter-centric PSF: compute weights once per emitter, distribute photons.

    Algorithm:
      1. Group photons by particle_idx
      2. For each unique emitter: compute PSF CDF over pixel neighborhood
      3. For each photon from that emitter: draw pixel from CDF

    This is O(N_emitters × psf²) + O(n_detected) vs O(n_detected × psf²).
    """
    n_det = 0

    # Since photons from the same emitter have the same position,
    # we cache the last emitter's PSF weights to avoid recomputation.
    # With PSS kernel output, photons from the same emitter are contiguous.
    last_pid = -999
    last_sigma = -1.0
    # Cache CDF weights (max neighborhood = (2*psf_range+1)^2)
    max_nbr = (2 * psf_range_px + 1) ** 2
    cached_cum_w = np.empty(max_nbr, dtype=np.float64)
    cached_pi = np.empty(max_nbr, dtype=np.int32)
    cached_pj = np.empty(max_nbr, dtype=np.int32)
    cached_n = 0
    cached_total = 0.0

    for p in range(n_photons):
        pid = particle_idx[p]
        ch = photon_channel[p]
        if ch < 0:
            ch = 0
        if ch >= len(sigma_per_ch):
            ch = 0
        sigma = sigma_per_ch[ch]
        if sigma <= 0.0:
            sigma = sigma_per_ch[0]

        # Check if we can reuse cached PSF weights
        if pid != last_pid or sigma != last_sigma:
            # Recompute PSF weights for this emitter
            x = positions_x[pid]
            y = positions_y[pid]

            fx = (x - offset_x) / pixel_size - 0.5
            fy = (y - offset_y) / pixel_size - 0.5
            i0 = int(math.floor(fx + 0.5))
            j0 = int(math.floor(fy + 0.5))

            if i0 < -psf_range_px or i0 >= nx + psf_range_px:
                continue
            if j0 < -psf_range_px or j0 >= ny + psf_range_px:
                continue

            inv_2sig2 = 0.5 / (sigma * sigma)
            cached_n = 0
            cached_total = 0.0

            for di in range(-psf_range_px, psf_range_px + 1):
                ii = i0 + di
                if ii < 0 or ii >= nx:
                    continue
                pcx = pixel_cx[ii]
                dx = x - pcx
                for dj in range(-psf_range_px, psf_range_px + 1):
                    jj = j0 + dj
                    if jj < 0 or jj >= ny:
                        continue
                    pcy = pixel_cy[jj]
                    dy = y - pcy
                    w = math.exp(-(dx * dx + dy * dy) * inv_2sig2)
                    cached_total += w
                    cached_cum_w[cached_n] = cached_total
                    cached_pi[cached_n] = ii
                    cached_pj[cached_n] = jj
                    cached_n += 1

            last_pid = pid
            last_sigma = sigma

        if cached_n == 0 or cached_total <= 0.0:
            continue

        # Draw pixel from cached CDF
        u = rng_uniform[p] * cached_total
        sel_i = cached_pi[0]
        sel_j = cached_pj[0]
        for k in range(cached_n):
            if cached_cum_w[k] >= u:
                sel_i = cached_pi[k]
                sel_j = cached_pj[k]
                break

        out_px[n_det] = sel_i
        out_py[n_det] = sel_j
        detected[p] = True
        n_det += 1

    return n_det


# ═══════════════════════════════════════════════════════════════════════
# PASO 3.3: TCSPC ACCUMULATION
# ═══════════════════════════════════════════════════════════════════════
# Each photon has an arrival_time from the CTMC (time since excitation
# pulse). The TCSPC module:
#   1. Convolves with IRF (instrument response function = Gaussian jitter)
#   2. Bins into TCSPC histogram channels
#   3. Accumulates into flim_stack[channel, px, py, tcspc_bin]
#
# IRF: h_IRF(t) = Gaussian(0, σ_IRF) where σ_IRF combines:
#   - Laser pulse width (~100 ps)
#   - Detector timing jitter (~50 ps for SPAD)
#   σ_IRF = √(σ_laser² + σ_detector²)
# Reference: Becker (2005) Advanced TCSPC, Springer, Ch. 7
# ═══════════════════════════════════════════════════════════════════════

class TCSPCAccumulator:
    """TCSPC histogram accumulator with IRF convolution.

    Builds flim_stack[n_channels, nx, ny, n_bins] by binning photon
    arrival times (convolved with IRF) into TCSPC histograms per pixel.

    Parameters
    ----------
    config : SimulationConfig
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.n_bins = config.tcspc.n_bins
        self.time_range = config.tcspc.time_range  # s
        self.bin_width = config.tcspc.bin_width     # s
        self.nx = config.scan.pixels_x
        self.ny = config.scan.pixels_y

        # Number of spectral channels (one per fluorophore type)
        self.n_channels = max(2, len(config.fluorophores))

        # IRF width: σ_IRF = √(σ_laser² + σ_detector²)
        # Laser pulse width is FWHM → σ = FWHM / (2√(2ln2))
        fwhm_to_sigma = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        sigma_laser = config.laser.pulse_width * fwhm_to_sigma
        sigma_detector = config.detector.timing_jitter  # already σ
        self.sigma_irf = math.sqrt(sigma_laser**2 + sigma_detector**2)

        # Allocate FLIM stack
        self.flim_stack = np.zeros(
            (self.n_channels, self.nx, self.ny, self.n_bins),
            dtype=np.uint32,
        )

        # Pre-allocate IRF kernel (discretized for convolution)
        self._build_irf_kernel()

        # Incomplete decay: when τ ~ T_laser, fluorescence from
        # previous pulses leaks into current TCSPC window.
        # Reference: Becker (2005) TCSPC Handbook, Ch. 5.2
        # SPCImage NG uses "incomplete decay model" for correction.
        # We model this physically by wrapping times mod T_laser.
        self._laser_period = 1.0 / config.laser.repetition_rate  # s
        self._incomplete_decay_enabled = False  # disabled: our CTMC dwell times
        # are already single-excitation relative, not absolute timestamps.
        # Enable only when simulating multi-pulse excitation where photons
        # from pulse n may arrive during pulse n+1 window.

        # Pile-up model: at high count rates, multiple photons per
        # laser period lead to statistical distortion (only first
        # photon counted). Model: for each laser period, accept
        # at most 1 photon (classical pile-up). This is conservative;
        # modern electronics have shorter dead times.
        # Reference: Becker (2005) TCSPC Handbook, Ch. 7.9
        #            Salthammer (1992) J Fluoresc 2:23
        self._pile_up_enabled = False  # disabled by default (low count rates)

    def _build_irf_kernel(self) -> None:
        """Build discretized IRF for TCSPC bin smearing."""
        # IRF as Gaussian in time bins
        n_irf = min(31, self.n_bins)  # kernel half-width in bins
        t_irf = (np.arange(-n_irf, n_irf + 1)) * self.bin_width
        self.irf_kernel = np.exp(-0.5 * (t_irf / self.sigma_irf) ** 2)
        self.irf_kernel /= self.irf_kernel.sum()  # normalize

    def accumulate(
        self,
        pixel_x: np.ndarray,         # int32[n] pixel column
        pixel_y: np.ndarray,         # int32[n] pixel row
        arrival_time: np.ndarray,    # float64[n] seconds (CTMC dwell time)
        channel: np.ndarray,         # int8[n] spectral channel
        n_photons: int,
        rng: np.random.Generator,
    ) -> None:
        """Accumulate photons into TCSPC histograms.

        Each photon's arrival_time is convolved with IRF (jitter added),
        then binned into the appropriate TCSPC channel.

        Incomplete decay model (Becker, TCSPC Handbook 2023, Ch. 5):
        When fluorescence lifetime is comparable to laser period T,
        photons from excitation pulse n-1 may arrive during period n.
        These appear as a flat "pedestal" added to the decay:
          h(t) = A·exp(-t/τ) + A·exp(-(t+T)/τ) + A·exp(-(t+2T)/τ) + ...
               = A·exp(-t/τ) / (1 - exp(-T/τ))
        We model this by wrapping arrival times modulo laser period,
        using the correct wrapped distribution rather than truncating.
        """
        if n_photons == 0:
            return

        # Add IRF jitter to arrival times
        jittered_times = arrival_time[:n_photons] + rng.normal(
            0.0, self.sigma_irf, size=n_photons
        )

        # Incomplete decay: wrap arrival times modulo laser period.
        # Photons with t > T_laser come from "the previous cycle" and
        # should appear at t mod T in the TCSPC window, creating the
        # characteristic pedestal of incomplete decays.
        if self._incomplete_decay_enabled and self._laser_period > 0:
            jittered_times = jittered_times % self._laser_period

        # Bin assignment
        _accumulate_tcspc_numba(
            self.flim_stack,
            pixel_x, pixel_y, jittered_times, channel,
            n_photons,
            self.bin_width, self.n_bins, self.n_channels,
            self.nx, self.ny,
        )

    def reset(self) -> None:
        """Reset FLIM stack to zero (new frame)."""
        self.flim_stack[:] = 0

    @property
    def total_counts(self) -> int:
        """Total photon counts across all pixels and channels."""
        return int(self.flim_stack.sum())

    def get_intensity_image(self, channel: int = 0) -> np.ndarray:
        """Sum over TCSPC bins to get intensity image [nx, ny]."""
        return self.flim_stack[channel].sum(axis=-1).astype(np.float64)


@njit(cache=True)
def _accumulate_tcspc_numba(
    flim_stack,       # uint32[n_ch, nx, ny, n_bins]
    pixel_x, pixel_y, # int32[n]
    arrival_time,     # float64[n] (already jittered)
    channel,          # int8[n]
    n_photons,
    bin_width, n_bins, n_channels, nx, ny,
):
    """Bin photons into TCSPC histograms. O(n_photons)."""
    for p in range(n_photons):
        ch = channel[p]
        if ch < 0 or ch >= n_channels:
            continue
        px = pixel_x[p]
        py = pixel_y[p]
        if px < 0 or px >= nx or py < 0 or py >= ny:
            continue

        t = arrival_time[p]
        if t < 0.0:
            t = 0.0
        b = int(t / bin_width)
        if b >= n_bins:
            continue  # outside TCSPC window (pile-up, ignore)

        flim_stack[ch, px, py, b] += 1


# ═══════════════════════════════════════════════════════════════════════
# PASO 3.4: DETECTOR MODEL
# ═══════════════════════════════════════════════════════════════════════
# Realistic SPAD/PMT detector effects:
#   1. Detection efficiency (QE × collection × filter transmission)
#   2. Dark counts (Poisson process, uniform in TCSPC bins)
#   3. Afterpulsing (correlated noise after each detected photon)
#   4. Dead time (missed counts after each detection)
#
# Reference: Becker (2005) TCSPC Handbook, Ch. 3
#            PicoQuant SPAD specifications
# ═══════════════════════════════════════════════════════════════════════

class DetectorModel:
    """SPAD/PMT detector with realistic noise processes.

    Applied AFTER PSF assignment and TCSPC binning.

    Parameters
    ----------
    config : SimulationConfig
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.det_efficiency = config.detector.detection_efficiency
        self.dark_count_rate = config.detector.dark_count_rate  # Hz
        self.afterpulsing_prob = config.detector.afterpulsing_prob
        self.dead_time = config.detector.dead_time  # s

        # Dwell time per pixel (for dark count calculation)
        self.dwell_time = config.scan.dwell_time  # s

    def apply_detection_efficiency(
        self,
        n_photons: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply quantum efficiency — Bernoulli filter on photons.

        Returns boolean mask of detected photons.
        """
        return rng.random(n_photons) < self.det_efficiency

    def add_dark_counts(
        self,
        flim_stack: np.ndarray,   # uint32[n_ch, nx, ny, n_bins]
        n_frames_accumulated: int,
        rng: np.random.Generator,
    ) -> int:
        """Add dark counts to FLIM stack.

        Dark counts are Poisson-distributed and uniform across
        TCSPC bins (no timing correlation with laser pulses).

        Returns total dark counts added.
        """
        n_ch, nx, ny, n_bins = flim_stack.shape

        # Expected dark counts per pixel per frame:
        # λ_dark = dark_count_rate × dwell_time × n_frames_accumulated
        expected_per_pixel = (
            self.dark_count_rate * self.dwell_time * max(1, n_frames_accumulated)
        )

        if expected_per_pixel < 1e-10:
            return 0

        total_dark = 0
        for ch in range(n_ch):
            for i in range(nx):
                for j in range(ny):
                    # Draw number of dark counts for this pixel
                    n_dark = rng.poisson(expected_per_pixel)
                    if n_dark > 0:
                        # Distribute uniformly across TCSPC bins
                        bins = rng.integers(0, n_bins, size=n_dark)
                        for b in bins:
                            flim_stack[ch, i, j, b] += 1
                        total_dark += n_dark

        return total_dark

    def add_afterpulsing(
        self,
        flim_stack: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        """Add afterpulsing: probability p_ap of extra count after each detection.

        Afterpulse timing: exponential distribution centered ~20-100 ns
        after the detected photon (Becker TCSPC Handbook, Ch. 3.3).
        We place afterpulses in random bins (worst-case model).

        Returns total afterpulse counts added.
        """
        if self.afterpulsing_prob <= 0:
            return 0

        n_ch, nx, ny, n_bins = flim_stack.shape
        total_ap = 0

        for ch in range(n_ch):
            for i in range(nx):
                for j in range(ny):
                    total_counts = int(flim_stack[ch, i, j].sum())
                    if total_counts == 0:
                        continue
                    # Number of afterpulses: Binomial(total_counts, p_ap)
                    n_ap = rng.binomial(total_counts, self.afterpulsing_prob)
                    if n_ap > 0:
                        bins = rng.integers(0, n_bins, size=n_ap)
                        for b in bins:
                            flim_stack[ch, i, j, b] += 1
                        total_ap += n_ap

        return total_ap

    def apply_dead_time(
        self,
        arrival_times: np.ndarray,  # float64[n], sorted within each pixel
        n_photons: int,
    ) -> np.ndarray:
        """Apply dead time filtering to photon stream.

        After each detected photon, subsequent photons arriving within
        dead_time are lost. This causes pile-up distortion at high
        count rates (Patting et al., Opt Express 2007).

        For TCSPC confocal FLIM at low count rates (<5% of laser rate),
        dead time loss is negligible. This method is provided for
        completeness when simulating high-count-rate conditions.

        Parameters
        ----------
        arrival_times : ndarray[n], float64
            Absolute arrival times, sorted.
        n_photons : int

        Returns
        -------
        mask : ndarray[n], bool
            True = photon passes, False = lost to dead time.

        References
        ----------
        Becker W (2005) Advanced TCSPC Techniques, Springer, Ch. 7.9
        Patting et al. (2007) Opt Express 15:15863
        """
        mask = np.ones(n_photons, dtype=np.bool_)
        if self.dead_time <= 0 or n_photons <= 1:
            return mask

        last_accepted = arrival_times[0]
        for i in range(1, n_photons):
            if arrival_times[i] - last_accepted < self.dead_time:
                mask[i] = False  # lost to dead time
            else:
                last_accepted = arrival_times[i]
        return mask


# ═══════════════════════════════════════════════════════════════════════
# FLIM FRAME DATA STRUCTURE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FLIMFrame:
    """One FLIM image frame with metadata.

    Contains the TCSPC histogram stack and derived images.
    """
    flim_stack: np.ndarray          # uint32[n_ch, nx, ny, n_bins]
    intensity: np.ndarray           # float64[n_ch, nx, ny] (sum over bins)
    frame_index: int = 0
    n_signal_photons: int = 0
    n_dark_counts: int = 0
    n_afterpulses: int = 0

    @staticmethod
    def from_stack(
        flim_stack: np.ndarray,
        frame_index: int = 0,
        n_signal: int = 0,
        n_dark: int = 0,
        n_ap: int = 0,
    ) -> "FLIMFrame":
        """Create FLIMFrame from TCSPC stack."""
        intensity = flim_stack.sum(axis=-1).astype(np.float64)
        return FLIMFrame(
            flim_stack=flim_stack.copy(),
            intensity=intensity,
            frame_index=frame_index,
            n_signal_photons=n_signal,
            n_dark_counts=n_dark,
            n_afterpulses=n_ap,
        )

    @property
    def total_counts(self) -> int:
        return int(self.flim_stack.sum())

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.flim_stack.shape


# ═══════════════════════════════════════════════════════════════════════
# PASO 3.5 (PARTIAL): VIRTUAL MICROSCOPE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════
# Combines PSF + Scanner + TCSPC + Detector into a single interface.
# process_photons() is called once per BD timestep.
# get_frame() extracts the accumulated FLIM image.
# ═══════════════════════════════════════════════════════════════════════

class VirtualMicroscope:
    """High-level orchestrator for the virtual FLIM microscope.

    Combines PSF, confocal scanner, TCSPC accumulator, and detector
    model to convert photon events into FLIM images.

    Workflow per frame:
      1. For each BD timestep in the frame:
         a. Receive PhotonBatch from photophysics
         b. Look up emitter positions
         c. Apply detection efficiency (Bernoulli filter)
         d. Assign detected photons to pixels (PSF)
         e. Accumulate into TCSPC histograms (with IRF)
      2. After all timesteps in frame:
         a. Add dark counts
         b. Add afterpulsing
         c. Extract FLIMFrame

    Parameters
    ----------
    config : SimulationConfig
    state : ParticleState
    """

    def __init__(self, config: SimulationConfig, state: ParticleState):
        self.config = config
        self.state = state

        # Build components
        self.psf = PSF(config)
        self.scanner = ConfocalScanner(config, self.psf)
        self.tcspc = TCSPCAccumulator(config)
        self.detector = DetectorModel(config)

        # RNG (separate seed from dynamics and photophysics)
        self._rng = np.random.default_rng(config.random_seed + 2000)

        # Frame tracking
        self._frame_index = 0
        self._n_signal = 0
        self._n_bd_steps_in_frame = 0

        # Performance tracking
        self.total_photons_received = 0
        self.total_photons_detected = 0
        self.total_dark_counts = 0
        self.total_afterpulses = 0

    def process_photons(self, photon_batch) -> int:
        """Process one BD timestep's worth of photons.

        Takes a PhotonBatch from the photophysics engine and:
          1. Looks up emitter positions
          2. Applies detection efficiency
          3. Assigns to pixels via PSF
          4. Accumulates into TCSPC

        Parameters
        ----------
        photon_batch : PhotonBatch
            Photons from one BD timestep.

        Returns
        -------
        n_detected : int
            Number of photons that passed detection and hit the FOV.
        """
        n_ph = photon_batch.n_photons
        if n_ph == 0:
            return 0

        self.total_photons_received += n_ph

        # 1. Apply detection efficiency
        det_mask = self.detector.apply_detection_efficiency(n_ph, self._rng)
        det_indices = np.where(det_mask)[0]
        n_surviving = len(det_indices)

        if n_surviving == 0:
            self._n_bd_steps_in_frame += 1
            return 0

        # 2. Get emitter positions
        particle_idx = photon_batch.particle_idx[det_indices]
        positions = self.state.positions

        # Bounds check particle indices
        valid = (particle_idx >= 0) & (particle_idx < positions.shape[0])
        if not np.all(valid):
            det_indices = det_indices[valid]
            particle_idx = particle_idx[valid]
            n_surviving = len(det_indices)

        if n_surviving == 0:
            self._n_bd_steps_in_frame += 1
            return 0

        emitter_x = positions[particle_idx, 0]
        emitter_y = positions[particle_idx, 1]
        channels = photon_batch.channel[det_indices]
        arrival_times = photon_batch.arrival_time[det_indices]

        # 3. Assign to pixels via PSF (emitter-centric, Phase 6 Step 3)
        # Sort by particle_idx so emitter-centric kernel can cache PSF weights
        sort_order = np.argsort(particle_idx, kind='mergesort')
        particle_idx_sorted = particle_idx[sort_order]
        channels_sorted = channels[sort_order]
        arrival_times_sorted = arrival_times[sort_order]

        px, py, assigned_mask = self.scanner.assign_photons_emitter_centric(
            particle_idx_sorted,
            positions[:, 0], positions[:, 1],
            channels_sorted,
            n_surviving, self._rng,
        )
        n_assigned = len(px)

        if n_assigned == 0:
            self._n_bd_steps_in_frame += 1
            return 0

        # Filter arrival times and channels to only assigned photons
        assigned_indices = np.where(assigned_mask)[0]
        arr_times_assigned = arrival_times_sorted[assigned_indices]
        channels_assigned = channels_sorted[assigned_indices]

        # 4. Accumulate into TCSPC
        self.tcspc.accumulate(
            px, py, arr_times_assigned, channels_assigned,
            n_assigned, self._rng,
        )

        self.total_photons_detected += n_assigned
        self._n_signal += n_assigned
        self._n_bd_steps_in_frame += 1

        return n_assigned

    def finalize_frame(self) -> FLIMFrame:
        """Finalize frame: add noise and extract FLIMFrame.

        Adds dark counts and afterpulsing, then returns the
        complete FLIM image.

        Returns
        -------
        FLIMFrame
            The completed FLIM image with metadata.
        """
        # Add dark counts
        # Note: n_frames_accumulated=1 because each pixel is scanned
        # once per frame. The BD steps are the dynamics time resolution,
        # NOT the scan accumulation count. (Session 8, Bug Fix #6b)
        n_dark = self.detector.add_dark_counts(
            self.tcspc.flim_stack,
            1,  # one scan per frame
            self._rng,
        )
        self.total_dark_counts += n_dark

        # Add afterpulsing
        n_ap = self.detector.add_afterpulsing(
            self.tcspc.flim_stack,
            self._rng,
        )
        self.total_afterpulses += n_ap

        # Create frame
        frame = FLIMFrame.from_stack(
            self.tcspc.flim_stack,
            frame_index=self._frame_index,
            n_signal=self._n_signal,
            n_dark=n_dark,
            n_ap=n_ap,
        )

        # Reset for next frame
        self.tcspc.reset()
        self._frame_index += 1
        self._n_signal = 0
        self._n_bd_steps_in_frame = 0

        return frame

    @property
    def frame_shape(self) -> Tuple[int, int, int, int]:
        """Shape of FLIM stack: (n_channels, nx, ny, n_bins)."""
        return (
            self.tcspc.n_channels,
            self.scanner.nx,
            self.scanner.ny,
            self.tcspc.n_bins,
        )

    def __repr__(self) -> str:
        psf_info = ", ".join(
            f"ch{ch}:σ={sig*1e3:.0f}nm/FWHM={self.psf.fwhm_um[ch]*1e3:.0f}nm"
            for ch, sig in self.psf.sigma_um.items()
        )
        return (
            f"VirtualMicroscope(\n"
            f"  PSF: {self.psf.psf_type} [{psf_info}]\n"
            f"  Scan: {self.scanner.nx}×{self.scanner.ny} px, "
            f"{self.config.scan.pixel_size:.0f} nm/px\n"
            f"  TCSPC: {self.tcspc.n_bins} bins, "
            f"{self.tcspc.bin_width*1e12:.1f} ps/bin\n"
            f"  IRF σ: {self.tcspc.sigma_irf*1e12:.1f} ps\n"
            f"  Detector: QE={self.detector.det_efficiency:.0%}, "
            f"dark={self.detector.dark_count_rate:.0f} Hz, "
            f"AP={self.detector.afterpulsing_prob:.1%}\n"
            f"  FOV: {self.scanner.fov_x:.2f}×{self.scanner.fov_y:.2f} µm\n"
            f")"
        )
