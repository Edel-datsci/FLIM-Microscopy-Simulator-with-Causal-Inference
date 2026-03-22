"""
ED-MRDT v1.1 — Simulation Pipeline (Orchestrator)
====================================================

Connects the three simulation layers into a single execution loop:
  Brownian Dynamics → Photophysics CTMC → Virtual Microscope → FLIM frames

Design sources:
  - Ermak & McCammon (1978): BD timestep is the fundamental clock
  - Becker W (2005): FLIM frame = accumulation of photon events over
    multiple laser pulses during confocal scan time
  - ReaDDy 2 (Hoffmann 2019): Observer pattern for periodic output

Architecture:
  The pipeline is driven by biological time:
    t_bio = 0 → simulation_time (e.g., 60 s)

  Time hierarchy:
    Frame       : frame_interval (e.g., 1.0 s)    — FLIM image output
    BD step     : bd_timestep   (e.g., 10 µs)     — dynamics + photophysics
    Laser pulse : 1/rep_rate    (e.g., 25 ns)     — individual excitation

  Per frame:
    n_bd_steps_per_frame = frame_interval / bd_timestep  (e.g., 100,000)
    Each BD step:
      1. DynamicsEngine.step()           → update positions, bonds
      2. PhotonSimulator.evolve_one_bd_step() → emit photons (CTMC)
      3. VirtualMicroscope.process_photons()  → accumulate into flim_stack
    After all BD steps:
      4. VirtualMicroscope.finalize_frame()   → add noise, extract FLIMFrame
      5. Extract ground truth snapshot

  Ground truth per frame:
    - positions[N, 2]          current particle positions
    - bond_partner[N]          current bonds
    - species_id[N]            current species (after reactions)
    - fret_efficiency[N]       E_exact from k_FRET / (k_FRET + k_rad)
    - donor_lifetime[N]        τ_DA = 1 / (k_rad + k_nr + k_FRET)
    - n_complexes              count of bound complexes
    - n_bleached               count of bleached dyes

Usage:
    from ed_mrdt.pipeline import SimulationPipeline
    pipeline = SimulationPipeline(config)
    results = pipeline.run()

References:
  - Lakowicz JR (2006) Principles of Fluorescence Spectroscopy, Ch. 13
  - Becker W (2005) Advanced TCSPC Techniques, Springer
  - Hoffmann M et al. (2019) PLoS Comput Biol 15:e1006830 (ReaDDy2)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import SimulationConfig, ConfigError
from .particles import ParticleState
from .dynamics import DynamicsEngine
from .photophysics import PhotonSimulator, PhotonBatch, fret_efficiency
from .microscope import VirtualMicroscope, FLIMFrame

__all__ = [
    "SimulationPipeline",
    "FrameResult",
    "SimulationResult",
    "GroundTruth",
    "SubFrameData",
]


# ═══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES FOR RESULTS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GroundTruth:
    """Ground truth snapshot at one frame.

    Contains the exact physical state that the microscope is trying
    to image. For benchmarking image analysis algorithms.

    Attributes
    ----------
    positions : ndarray[N, 2]
        Particle positions in µm at the end of the frame.
    species_id : ndarray[N]
        Species index for each particle.
    bond_partner : ndarray[N]
        Bond partner index (-1 = free).
    is_alive : ndarray[N]
        Whether particle is active.
    has_dye : ndarray[N]
        Whether particle carries a fluorophore.
    dye_type_id : ndarray[N]
        Fluorophore type index.
    is_bleached : ndarray[N]
        Whether dye is bleached.
    fret_efficiency_exact : ndarray[N]
        Exact FRET efficiency from Förster theory (0 for non-donors).
    donor_lifetime_exact : ndarray[N]
        Exact donor lifetime τ_DA in seconds (0 for non-donors).
    n_complexes : int
        Number of bound receptor-ligand complexes.
    n_bleached : int
        Number of bleached dyes.
    n_free_donors : int
        Number of unbound donors.
    species_counts : dict, optional
        Per-species alive counts: {species_name: alive_count}
    """
    positions: np.ndarray
    species_id: np.ndarray
    bond_partner: np.ndarray
    is_alive: np.ndarray
    has_dye: np.ndarray
    dye_type_id: np.ndarray
    is_bleached: np.ndarray
    fret_efficiency_exact: np.ndarray
    donor_lifetime_exact: np.ndarray
    n_complexes: int = 0
    n_bleached: int = 0
    n_free_donors: int = 0
    species_counts: Optional[Dict[str, int]] = None


@dataclass
class SubFrameData:
    """Per-BD-step ground truth for sub-frame causal analysis.

    Records observable time series at BD timestep resolution
    (typically dt_BD = 10 µs, giving ~100K points per 20-frame sim).
    This enables Transfer Entropy estimation with >> n_bins^3 samples.

    Reference: Schreiber (2000) Phys Rev Lett 85:461 — requires
    N >> n_bins^3 for reliable TE; sub-frame data provides this.
    """
    n_complexes: np.ndarray      # [n_steps_total] complex count per BD step
    n_bindings: np.ndarray       # [n_steps_total] binding events this step
    n_dissociations: np.ndarray  # [n_steps_total] dissociation events this step
    n_bleached: np.ndarray       # [n_steps_total] bleached count per step
    total_photons: np.ndarray    # [n_steps_total] photons emitted this step
    mean_fret_E: np.ndarray      # [n_steps_total] mean FRET efficiency of bound donors
    dt_s: float                  # BD timestep in seconds
    steps_per_frame: int         # BD steps per frame
    species_counts: Optional[Dict[str, np.ndarray]] = None  # {name: [n_steps_total]} alive count per species per BD step
    n_catalytic: Optional[np.ndarray] = None      # [n_steps_total] catalytic events this step
    n_recruited: Optional[np.ndarray] = None       # [n_steps_total] recruitment events this step
    n_detached: Optional[np.ndarray] = None        # [n_steps_total] detachment events this step
    n_converted: Optional[np.ndarray] = None       # [n_steps_total] conversion events this step


@dataclass
class FrameResult:
    """Result for one FLIM frame.

    Pairs the microscopy image with its ground truth.
    """
    frame_index: int
    bio_time: float                # biological time at frame end (s)
    flim_frame: FLIMFrame          # FLIM image
    ground_truth: GroundTruth      # exact physical state
    n_photons_emitted: int = 0     # total photons from photophysics
    n_photons_detected: int = 0    # photons in FLIM stack
    wall_time_s: float = 0.0       # wall clock time for this frame

    @property
    def total_counts(self) -> int:
        return self.flim_frame.total_counts


@dataclass
class SimulationResult:
    """Complete simulation output.

    Contains all frames, metadata, and summary statistics.
    """
    config: SimulationConfig
    frames: List[FrameResult]
    total_wall_time_s: float = 0.0
    peak_ram_mb: float = 0.0
    total_photons_emitted: int = 0
    total_photons_detected: int = 0
    total_bindings: int = 0
    total_dissociations: int = 0
    total_bleached: int = 0
    subframe_data: Optional[SubFrameData] = None  # Sub-frame recording (if enabled)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def get_flim_stack(self, frame_idx: int) -> np.ndarray:
        """Get FLIM stack for a specific frame."""
        return self.frames[frame_idx].flim_frame.flim_stack

    def get_intensity_series(self) -> np.ndarray:
        """Get total intensity time series [n_frames]."""
        return np.array([f.total_counts for f in self.frames])

    def get_complex_count_series(self) -> np.ndarray:
        """Get complex count time series [n_frames]."""
        return np.array([f.ground_truth.n_complexes for f in self.frames])

    def get_bleaching_series(self) -> np.ndarray:
        """Get bleached count time series [n_frames]."""
        return np.array([f.ground_truth.n_bleached for f in self.frames])

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"ED-MRDT Simulation Result",
            f"  Config: {self.config.name}",
            f"  Frames: {self.n_frames}",
            f"  Wall time: {self.total_wall_time_s:.1f} s "
            f"({self.total_wall_time_s / 60:.1f} min)",
            f"  Peak RAM: {self.peak_ram_mb:.1f} MB",
            f"  Photons emitted: {self.total_photons_emitted:,}",
            f"  Photons detected: {self.total_photons_detected:,}",
            f"  Bindings: {self.total_bindings:,}",
            f"  Dissociations: {self.total_dissociations:,}",
            f"  Final bleached: {self.total_bleached}",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class SimulationPipeline:
    """Orchestrator: BD → Photophysics → Microscope → FLIM frames.

    Drives the full simulation, managing time scheduling, component
    coupling, ground truth extraction, and progress reporting.

    Parameters
    ----------
    config : SimulationConfig
        Complete simulation configuration (from YAML).
    progress_callback : callable, optional
        Called with (frame_idx, n_frames, msg) for progress updates.
        Default: print to stdout.

    Physics reference for time scheduling:
      In a real FLIM experiment, the laser pulses continuously during
      the confocal scan. Each pixel accumulates photons during dwell_time.
      We simulate the FULL membrane for each BD step, which gives us
      photon events at all emitter positions simultaneously. This is
      equivalent to widefield excitation but the confocal pinhole
      filtering is applied in the VirtualMicroscope (PSF assignment).

      The number of BD steps per frame determines how much dynamics
      occurs between FLIM snapshots:
        n_bd_per_frame = frame_interval / bd_timestep
      For canonical case: 1.0 s / 10 µs = 100,000 steps per frame.
    """

    def __init__(
        self,
        config: SimulationConfig,
        progress_callback: Optional[Callable] = None,
        record_subframe: bool = False,
    ):
        self.config = config
        self.record_subframe = record_subframe

        # ── Validate timing consistency ──────────────────────────────
        dt = config.bd_timestep
        frame_interval = config.scan.frame_interval

        # Prioritize top-level n_frames if defined, else fallback to scan
        n_frames = getattr(config, 'n_frames', config.scan.n_frames)

        if frame_interval <= 0:
            raise ConfigError("frame_interval must be positive")
        if dt <= 0:
            raise ConfigError("bd_timestep must be positive")

        # BD steps per frame (must be integer for reproducibility)
        self.n_bd_per_frame = int(round(frame_interval / dt))
        if self.n_bd_per_frame < 1:
            raise ConfigError(
                f"frame_interval ({frame_interval}s) < bd_timestep ({dt}s)")

        # Actual frame interval after rounding (should match closely)
        actual_frame_interval = self.n_bd_per_frame * dt
        drift = abs(actual_frame_interval - frame_interval) / frame_interval
        if drift > 0.01:
            raise ConfigError(
                f"frame_interval ({frame_interval}s) is not an integer "
                f"multiple of bd_timestep ({dt}s). Drift = {drift:.2%}")

        self.n_frames = n_frames
        self.dt = dt
        self.frame_interval = actual_frame_interval

        # ── Progress callback ────────────────────────────────────────
        if progress_callback is None:
            self._progress = self._default_progress
        else:
            self._progress = progress_callback

        # ── Component state (created at run time) ────────────────────
        self._state: Optional[ParticleState] = None
        self._dynamics: Optional[DynamicsEngine] = None
        self._photon_sim: Optional[PhotonSimulator] = None
        self._microscope: Optional[VirtualMicroscope] = None

    # ─── Initialization ──────────────────────────────────────────────

    def _init_components(self) -> None:
        """Initialize all simulation components."""
        config = self.config

        # 1. Particle state
        self._state = ParticleState(config)

        # 2. Dynamics engine
        self._dynamics = DynamicsEngine(config, self._state)

        # 3. Scan duty fraction: fraction of BD steps during which
        #    each molecule is actually illuminated by the laser.
        #
        #    In confocal scanning, the beam visits each pixel for
        #    dwell_time. A molecule at a fixed position is illuminated
        #    only during ONE pixel dwell per scan, NOT during the
        #    entire raster scan time.
        #
        #    scan_duty_fraction = dwell_time / frame_interval
        #    For canonical case: 20µs / 1.0s = 2×10⁻⁵
        #
        #    This means only 0.002% of BD steps should apply laser
        #    excitation/bleaching. The rest are "dark" steps where
        #    only thermal T1→S0 relaxation occurs.
        #
        #    Ref: Becker W (2005) §3.2; Digman & Gratton (2009)
        scan_duty_fraction = min(
            1.0, config.scan.dwell_time / config.scan.frame_interval)

        # 4. Photon simulator (with scan-fraction correction)
        self._photon_sim = PhotonSimulator(
            config, self._state,
            scan_duty_fraction=scan_duty_fraction,
        )

        # 5. Virtual microscope
        self._microscope = VirtualMicroscope(config, self._state)

    # ─── Ground truth extraction ─────────────────────────────────────

    def _extract_ground_truth(self) -> GroundTruth:
        """Extract exact physical state for current frame.

        Computes exact FRET efficiency and donor lifetime from
        current k_FRET values stored in PhotonSimulator.

        FRET efficiency (exact):
            E = k_FRET / (k_FRET + k_rad + k_nr)
            where k_rad + k_nr = 1/τ_D (donor-only lifetime)

        Donor lifetime with FRET (exact):
            τ_DA = 1 / (k_rad + k_nr + k_FRET)
            = τ_D / (1 + k_FRET × τ_D)

        Reference: Lakowicz (2006) Eq. 13.10, 13.14
        """
        state = self._state
        N = state.N
        config = self.config

        # Exact FRET from photon simulator's k_FRET arrays
        fret_E = np.zeros(N, dtype=np.float64)
        tau_DA = np.zeros(N, dtype=np.float64)

        k_fret_arr = self._photon_sim._k_fret  # from last update_fret_rates()

        # Get donor fluorophore info
        for fp_cfg in config.fret_pairs:
            donor_name = fp_cfg.donor_type
            donor_idx = None
            for fi, f in enumerate(config.fluorophores):
                if f.name == donor_name:
                    donor_idx = fi
                    break
            if donor_idx is None:
                continue

            rm = self._photon_sim.rate_matrices[donor_idx]
            tau_D = rm.get_lifetime(rm.excited_state)  # donor-only lifetime
            if tau_D <= 0 or math.isinf(tau_D):
                continue

            for i in range(N):
                if not state.has_dye[i] or state.is_bleached[i]:
                    continue
                if state.dye_type_id[i] != donor_idx:
                    continue

                k_f = k_fret_arr[i]
                if k_f > 0:
                    # E = k_FRET * τ_D / (1 + k_FRET * τ_D)
                    # τ_DA = τ_D / (1 + k_FRET * τ_D)
                    fret_E[i] = k_f * tau_D / (1.0 + k_f * tau_D)
                    tau_DA[i] = tau_D / (1.0 + k_f * tau_D)
                else:
                    # Donor-only: E=0, τ=τ_D
                    fret_E[i] = 0.0
                    tau_DA[i] = tau_D

        # Count complexes: particles whose species is a reaction product
        # and who have a bond_partner >= 0
        n_complexes = int(np.sum(state.bond_partner >= 0)) // 2  # each bond has 2 partners
        n_bleached = int(np.sum(state.is_bleached))

        # Count free donors (has donor dye, not bound, not bleached)
        n_free_donors = 0
        for fp_cfg in config.fret_pairs:
            donor_name = fp_cfg.donor_type
            for fi, f in enumerate(config.fluorophores):
                if f.name == donor_name:
                    mask = (
                        state.has_dye &
                        (state.dye_type_id == fi) &
                        ~state.is_bleached &
                        (state.bond_partner < 0)
                    )
                    n_free_donors += int(np.sum(mask))

        # Per-species alive counts
        species_counts_dict = {}
        for sp_idx, sp in enumerate(config.species):
            mask = (state.species_id == sp_idx) & state.is_alive
            species_counts_dict[sp.name] = int(np.sum(mask))

        return GroundTruth(
            positions=state.positions.copy(),
            species_id=state.species_id.copy(),
            bond_partner=state.bond_partner.copy(),
            is_alive=state.is_alive.copy(),
            has_dye=state.has_dye.copy(),
            dye_type_id=state.dye_type_id.copy(),
            is_bleached=state.is_bleached.copy(),
            fret_efficiency_exact=fret_E,
            donor_lifetime_exact=tau_DA,
            n_complexes=n_complexes,
            n_bleached=n_bleached,
            n_free_donors=n_free_donors,
            species_counts=species_counts_dict,
        )

    # ─── RAM monitoring ──────────────────────────────────────────────

    @staticmethod
    def _get_ram_mb() -> float:
        """Get current process RSS in MB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1e6
        except ImportError:
            # Fallback: /proc/self/status (Linux)
            try:
                with open("/proc/self/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            return float(line.split()[1]) / 1024.0  # kB → MB
            except (FileNotFoundError, ValueError):
                return 0.0
        return 0.0

    # ─── Progress reporting ──────────────────────────────────────────

    @staticmethod
    def _default_progress(
        frame_idx: int,
        n_frames: int,
        msg: str,
    ) -> None:
        """Default progress callback: print to stdout."""
        print(f"  [{frame_idx + 1}/{n_frames}] {msg}")

    # ─── Main simulation loop ────────────────────────────────────────

    def run(self) -> SimulationResult:
        """Execute the full simulation pipeline.

        Returns
        -------
        SimulationResult
            Complete results with all frames and metadata.
        """
        t_start_total = time.perf_counter()
        peak_ram = self._get_ram_mb()

        # Initialize components
        self._init_components()
        init_ram = self._get_ram_mb()
        peak_ram = max(peak_ram, init_ram)

        self._progress(
            -1, self.n_frames,
            f"Initialized: {self._state.N} particles, "
            f"{self.n_bd_per_frame} BD steps/frame, "
            f"{self.n_frames} frames. RAM: {init_ram:.0f} MB"
        )

        frames: List[FrameResult] = []
        total_photons_emitted = 0
        total_photons_detected = 0

        # Sub-frame recording for causal analysis
        if self.record_subframe:
            total_bd_steps = self.n_frames * self.n_bd_per_frame
            sf_n_complexes = np.zeros(total_bd_steps, dtype=np.int32)
            sf_n_bindings = np.zeros(total_bd_steps, dtype=np.int32)
            sf_n_dissociations = np.zeros(total_bd_steps, dtype=np.int32)
            sf_n_bleached = np.zeros(total_bd_steps, dtype=np.int32)
            sf_total_photons = np.zeros(total_bd_steps, dtype=np.int32)
            sf_mean_fret_E = np.zeros(total_bd_steps, dtype=np.float64)
            sf_step_idx = 0
            prev_bindings = self._dynamics.n_bindings
            prev_dissociations = self._dynamics.n_dissociations

            # Generic species tracking
            species_to_track = [sp.name for sp in self.config.species]
            species_name_to_id = {sp.name: i for i, sp in enumerate(self.config.species)}
            sf_species_counts = {name: np.zeros(total_bd_steps, dtype=np.int32)
                                 for name in species_to_track}

            # New dynamics counters
            has_catalytic = hasattr(self._dynamics, 'n_catalytic')
            sf_n_catalytic = np.zeros(total_bd_steps, dtype=np.int32) if has_catalytic else None
            sf_n_recruited = np.zeros(total_bd_steps, dtype=np.int32) if has_catalytic else None
            sf_n_detached = np.zeros(total_bd_steps, dtype=np.int32) if has_catalytic else None
            sf_n_converted = np.zeros(total_bd_steps, dtype=np.int32) if has_catalytic else None

            prev_catalytic = self._dynamics.n_catalytic if has_catalytic else 0
            prev_recruited = self._dynamics.n_recruited if has_catalytic else 0
            prev_detached = self._dynamics.n_detached if has_catalytic else 0
            prev_converted = self._dynamics.n_converted if has_catalytic else 0

        # ── Frame loop ───────────────────────────────────────────────
        for frame_i in range(self.n_frames):
            t_frame_start = time.perf_counter()
            frame_photons_emitted = 0
            frame_photons_detected = 0

            # ── BD step loop within frame ────────────────────────────
            for bd_step in range(self.n_bd_per_frame):
                # 1. Advance Brownian Dynamics (one timestep)
                self._dynamics.step()

                # 2. Evolve photophysics (one BD step → many laser pulses)
                photon_batch = self._photon_sim.evolve_one_bd_step()
                frame_photons_emitted += photon_batch.n_photons

                # 3. Accumulate photons in microscope
                n_det = self._microscope.process_photons(photon_batch)
                frame_photons_detected += n_det

                # Record sub-frame data
                if self.record_subframe:
                    state = self._state
                    # Complex count: particles with bond_partner >= 0, count each pair once
                    n_bound = np.sum(state.bond_partner[state.is_alive] >= 0) // 2
                    sf_n_complexes[sf_step_idx] = n_bound

                    # Binding/dissociation events THIS step (delta from cumulative)
                    cur_bind = self._dynamics.n_bindings
                    cur_dissoc = self._dynamics.n_dissociations
                    sf_n_bindings[sf_step_idx] = cur_bind - prev_bindings
                    sf_n_dissociations[sf_step_idx] = cur_dissoc - prev_dissociations
                    prev_bindings = cur_bind
                    prev_dissociations = cur_dissoc

                    # Bleached count
                    sf_n_bleached[sf_step_idx] = np.sum(state.is_bleached)

                    # Photons this step
                    sf_total_photons[sf_step_idx] = photon_batch.n_photons

                    # Mean FRET efficiency across all donors
                    # Two modes: bond-mode (E from constant r/R0) and
                    # proximity-mode (E from per-particle k_fret).
                    # Ref: Förster (1948), E = k_fret·τ_D / (1 + k_fret·τ_D)
                    if len(self.config.fret_pairs) > 0:
                        fp = self.config.fret_pairs[0]
                        fret_mode = getattr(fp, 'fret_mode', 'bond')
                        donor_mask = (state.has_dye & state.is_alive
                                      & ~state.is_bleached
                                      & (state.dye_type_id == 0))
                        n_donors_total = int(np.sum(donor_mask))

                        if fret_mode == 'proximity' and n_donors_total > 0:
                            # Proximity mode: compute E per-particle from k_fret
                            k_fret_donors = self._photon_sim._k_fret[donor_mask]
                            tau_D = self._photon_sim._tau_all[0]  # donor tau (s)
                            E_per_donor = (k_fret_donors * tau_D
                                           / (1.0 + k_fret_donors * tau_D))
                            sf_mean_fret_E[sf_step_idx] = float(np.mean(E_per_donor))
                        elif fret_mode != 'proximity' and n_donors_total > 0:
                            # Bond mode: constant E for bound, 0 for free
                            r_nm = fp.intra_complex_distance
                            R0_nm = fp.R0
                            E_const = 1.0 / (1.0 + (r_nm / R0_nm)**6)
                            n_donors_bound = n_bound
                            n_donors_free = n_donors_total - n_donors_bound
                            sf_mean_fret_E[sf_step_idx] = (
                                E_const * n_donors_bound
                                / (n_donors_bound + n_donors_free))
                        else:
                            sf_mean_fret_E[sf_step_idx] = 0.0
                    else:
                        sf_mean_fret_E[sf_step_idx] = 0.0

                    # Per-species alive counts
                    for sp_name in species_to_track:
                        sp_id = species_name_to_id[sp_name]
                        sf_species_counts[sp_name][sf_step_idx] = np.sum(
                            (state.species_id == sp_id) & state.is_alive)

                    # New dynamics event counters
                    if has_catalytic:
                        cur_cat = self._dynamics.n_catalytic
                        cur_rec = self._dynamics.n_recruited
                        cur_det = self._dynamics.n_detached
                        cur_conv = self._dynamics.n_converted
                        sf_n_catalytic[sf_step_idx] = cur_cat - prev_catalytic
                        sf_n_recruited[sf_step_idx] = cur_rec - prev_recruited
                        sf_n_detached[sf_step_idx] = cur_det - prev_detached
                        sf_n_converted[sf_step_idx] = cur_conv - prev_converted
                        prev_catalytic = cur_cat
                        prev_recruited = cur_rec
                        prev_detached = cur_det
                        prev_converted = cur_conv

                    sf_step_idx += 1

            # ── Finalize frame ───────────────────────────────────────
            flim_frame = self._microscope.finalize_frame()

            # ── Extract ground truth ─────────────────────────────────
            # Trigger FRET rate update for accurate ground truth
            self._photon_sim.update_fret_rates()
            gt = self._extract_ground_truth()

            # ── Build frame result ───────────────────────────────────
            bio_time = (frame_i + 1) * self.frame_interval
            wall_time = time.perf_counter() - t_frame_start

            frame_result = FrameResult(
                frame_index=frame_i,
                bio_time=bio_time,
                flim_frame=flim_frame,
                ground_truth=gt,
                n_photons_emitted=frame_photons_emitted,
                n_photons_detected=frame_photons_detected,
                wall_time_s=wall_time,
            )
            frames.append(frame_result)

            total_photons_emitted += frame_photons_emitted
            total_photons_detected += frame_photons_detected

            # ── RAM check ────────────────────────────────────────────
            current_ram = self._get_ram_mb()
            peak_ram = max(peak_ram, current_ram)

            # ── Progress ─────────────────────────────────────────────
            elapsed = time.perf_counter() - t_start_total
            eta = elapsed / (frame_i + 1) * (self.n_frames - frame_i - 1)

            self._progress(
                frame_i, self.n_frames,
                f"t={bio_time:.1f}s | "
                f"photons={frame_photons_detected:,} | "
                f"counts={flim_frame.total_counts:,} | "
                f"complexes={gt.n_complexes} | "
                f"bleached={gt.n_bleached} | "
                f"wall={wall_time:.1f}s | "
                f"RAM={current_ram:.0f}MB | "
                f"ETA={eta:.0f}s"
            )

            # ── RAM limit check ──────────────────────────────────────
            if current_ram > self.config.ram_limit_gb * 1000:
                self._progress(
                    frame_i, self.n_frames,
                    f"⚠️ RAM limit exceeded ({current_ram:.0f} MB > "
                    f"{self.config.ram_limit_gb * 1000:.0f} MB). Stopping."
                )
                break

        # ── Build final result ───────────────────────────────────────
        total_wall = time.perf_counter() - t_start_total

        subframe = None
        if self.record_subframe:
            subframe = SubFrameData(
                n_complexes=sf_n_complexes,
                n_bindings=sf_n_bindings,
                n_dissociations=sf_n_dissociations,
                n_bleached=sf_n_bleached,
                total_photons=sf_total_photons,
                mean_fret_E=sf_mean_fret_E,
                dt_s=self._dynamics.dt,
                steps_per_frame=self.n_bd_per_frame,
                species_counts=sf_species_counts,
                n_catalytic=sf_n_catalytic,
                n_recruited=sf_n_recruited,
                n_detached=sf_n_detached,
                n_converted=sf_n_converted,
            )

        result = SimulationResult(
            config=self.config,
            frames=frames,
            total_wall_time_s=total_wall,
            peak_ram_mb=peak_ram,
            total_photons_emitted=total_photons_emitted,
            total_photons_detected=total_photons_detected,
            total_bindings=self._dynamics.n_bindings,
            total_dissociations=self._dynamics.n_dissociations,
            total_bleached=int(np.sum(self._state.is_bleached)),
            subframe_data=subframe,
        )

        self._progress(
            self.n_frames - 1, self.n_frames,
            f"DONE. {result.summary()}"
        )

        return result

    # ─── Convenience: run mini demo ──────────────────────────────────

    @classmethod
    def run_mini_demo(
        cls,
        config: SimulationConfig,
        n_frames: int = 2,
        n_bd_steps_per_frame: Optional[int] = None,
    ) -> SimulationResult:
        """Run a quick mini simulation for testing/validation.

        Overrides scan.n_frames and optionally frame_interval to
        produce a fast test run.

        Parameters
        ----------
        config : SimulationConfig
            Base configuration.
        n_frames : int
            Number of frames to simulate.
        n_bd_steps_per_frame : int, optional
            If given, override frame_interval = n_bd_steps × dt.
        """
        # Modify config for mini run
        config.scan.n_frames = n_frames
        if n_bd_steps_per_frame is not None:
            config.scan.frame_interval = n_bd_steps_per_frame * config.bd_timestep

        pipeline = cls(config)
        return pipeline.run()

    # ─── Repr ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        total_steps = self.n_bd_per_frame * self.n_frames
        total_bio_time = self.n_frames * self.frame_interval
        return (
            f"SimulationPipeline(\n"
            f"  config: {self.config.name}\n"
            f"  frames: {self.n_frames}\n"
            f"  BD steps/frame: {self.n_bd_per_frame:,}\n"
            f"  total BD steps: {total_steps:,}\n"
            f"  bio time: {total_bio_time:.1f} s\n"
            f"  dt: {self.dt:.1e} s\n"
            f")"
        )
