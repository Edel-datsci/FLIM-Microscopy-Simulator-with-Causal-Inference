"""
ED-MRDT v1.1 — Photophysics Engine (PSS + FRET + Bleaching)
==============================================================

Phase 6 Optimization: Poisson Steady-State (PSS) replaces the
pulse-by-pulse (PbP) CTMC kernel for >100× speedup.

Physics basis (timescale separation):
  τ_S1 (4 ns) ≪ T_laser (25 ns) ≪ dt (10 µs) ≪ τ_bleach (145 ms)

Mathematical formulation:
  1. Per-pulse 3×3 transition matrix T_period {S0, T1, Bleached}
  2. T_step = T_period^n_pulses via matrix power
  3. Expected photons via geometric series:
     S_geo = (I - T_sub^n) @ inv(I - T_sub)
  4. mean_donor_photons = S_geo[start, S0] × p_exc × p_rad × QY_donor
     mean_fret_photons  = S_geo[start, S0] × p_exc × p_fret × QY_acceptor
     (QY correction: not every S1→S0 transition is radiative;
      fraction QY = k_rad/(k_rad+k_nr) actually emit a photon.
      Ref: Lakowicz 2006 Ch.1, Valeur & Berberan-Santos 2012 Ch.3)
  5. n_photons ~ Poisson(mean_photons)
  6. arrival_times ~ Exp(τ_DA) for TCSPC

Validated: PSS/PbP = 1.0026 (10,000 trials, single emitter)

Design sources:
  - Ingargiola (PyBroMo): Poisson photon paradigm
  - Bourgeois (SMIS, 2023): transition probability model
  - Förster T (1948): FRET theory
  - Lakowicz JR (2006): Fluorescence Spectroscopy, 3rd ed
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit, prange

from .config import SimulationConfig, FluorophoreType, FRETpair, ConfigError
from .particles import ParticleState

__all__ = [
    "RateMatrix",
    "PhotonBatch",
    "PhotonSimulator",
    "fret_efficiency",
    "fret_rate",
    "_compute_fret_rates_proximity_numba",
]


# ═══════════════════════════════════════════════════════════════════════
# RATE MATRIX CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════

class RateMatrix:
    """CTMC rate matrix for a single fluorophore type."""

    def __init__(self, fluorophore: FluorophoreType):
        self.fluorophore = fluorophore
        self.n_states = fluorophore.n_states
        self.state_names = [s.name for s in fluorophore.states]
        self._state_map = {s.name: i for i, s in enumerate(fluorophore.states)}

        self.Q = np.zeros((self.n_states, self.n_states), dtype=np.float64)
        for from_name, transitions in fluorophore.thermal_rates.items():
            i = self._state_map[from_name]
            for to_name, rate in transitions.items():
                j = self._state_map[to_name]
                self.Q[i, j] = rate
        self._update_diagonal()

        self.fluorescent_mask = np.array(
            [s.is_fluorescent for s in fluorophore.states], dtype=np.bool_)
        self.terminal_mask = np.array(
            [s.is_terminal for s in fluorophore.states], dtype=np.bool_)
        self.absorbing_mask = np.array(
            [s.is_absorbing for s in fluorophore.states], dtype=np.bool_)

        self.ground_state = 0
        for i, s in enumerate(fluorophore.states):
            if s.is_absorbing and not s.is_fluorescent:
                self.ground_state = i
                break

        self.excited_state = -1
        for i, s in enumerate(fluorophore.states):
            if s.is_fluorescent:
                self.excited_state = i
                break

        self._precompute_branching()

        # Identify T1 and bleached states
        self.t1_state = -1
        self.bleached_state = -1
        for i, s in enumerate(fluorophore.states):
            if not s.is_absorbing and not s.is_fluorescent and not s.is_terminal:
                if self.t1_state < 0:
                    self.t1_state = i
            if s.is_terminal:
                if self.bleached_state < 0:
                    self.bleached_state = i

    def _update_diagonal(self) -> None:
        for i in range(self.n_states):
            self.Q[i, i] = 0.0
            self.Q[i, i] = -np.sum(self.Q[i, :])

    def _precompute_branching(self) -> None:
        self.exit_rates = np.zeros(self.n_states, dtype=np.float64)
        self.branch_probs = np.zeros(
            (self.n_states, self.n_states), dtype=np.float64)
        for i in range(self.n_states):
            total = -self.Q[i, i]
            self.exit_rates[i] = total
            if total > 0:
                for j in range(self.n_states):
                    if j != i:
                        self.branch_probs[i, j] = self.Q[i, j] / total

    def get_exit_rate(self, state: int) -> float:
        return self.exit_rates[state]

    def get_lifetime(self, state: int) -> float:
        k = self.exit_rates[state]
        if k <= 0:
            return float('inf')
        return 1.0 / k

    def copy_with_fret(self, k_fret: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return modified (exit_rates, branch_probs, fret_branch) with FRET."""
        exit_rates = self.exit_rates.copy()
        branch_probs = self.branch_probs.copy()
        fret_branch = np.zeros(self.n_states, dtype=np.float64)
        if self.excited_state < 0 or k_fret <= 0:
            return exit_rates, branch_probs, fret_branch
        s1 = self.excited_state
        s0 = self.ground_state
        old_total = exit_rates[s1]
        new_total = old_total + k_fret
        exit_rates[s1] = new_total
        if new_total > 0:
            fret_branch[s1] = k_fret / new_total
            for j in range(self.n_states):
                if j != s1:
                    branch_probs[s1, j] = self.Q[s1, j] / new_total
        return exit_rates, branch_probs, fret_branch


# ═══════════════════════════════════════════════════════════════════════
# FRET FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def fret_efficiency(r_nm: float, R0_nm: float) -> float:
    """E(r) = 1 / (1 + (r/R0)^6). Ref: Förster 1948, Lakowicz 2006 Eq.13.9"""
    if r_nm <= 0.0:
        return 1.0
    ratio = r_nm / R0_nm
    r6 = ratio * ratio * ratio * ratio * ratio * ratio
    return 1.0 / (1.0 + r6)


@njit(cache=True)
def fret_rate(r_nm: float, R0_nm: float, tau_D: float) -> float:
    """k_FRET = (1/τ_D) * (R0/r)^6. Ref: Lakowicz 2006 Eq.13.12"""
    if r_nm <= 0.0:
        return 1.0e15
    if tau_D <= 0.0:
        return 0.0
    ratio = R0_nm / r_nm
    r6 = ratio * ratio * ratio * ratio * ratio * ratio
    return r6 / tau_D


# ═══════════════════════════════════════════════════════════════════════
# PHOTON BATCH
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PhotonBatch:
    """Collection of photons emitted during one BD timestep."""
    particle_idx: np.ndarray
    arrival_time: np.ndarray
    channel: np.ndarray
    pulse_index: np.ndarray
    is_fret: np.ndarray
    n_photons: int = 0

    @staticmethod
    def empty() -> 'PhotonBatch':
        return PhotonBatch(
            particle_idx=np.empty(0, dtype=np.int32),
            arrival_time=np.empty(0, dtype=np.float64),
            channel=np.empty(0, dtype=np.int8),
            pulse_index=np.empty(0, dtype=np.int32),
            is_fret=np.empty(0, dtype=np.bool_),
            n_photons=0,
        )

    def __repr__(self) -> str:
        if self.n_photons > 0:
            unique_ch, counts_ch = np.unique(self.channel, return_counts=True)
            ch_str = ", ".join(f"ch{c}={n}" for c, n in zip(unique_ch, counts_ch))
        else:
            ch_str = "empty"
        return f"PhotonBatch(n={self.n_photons}, {ch_str})"


# ═══════════════════════════════════════════════════════════════════════
# PSS MATRIX PRECOMPUTATION
# ═══════════════════════════════════════════════════════════════════════

def _build_pss_matrices(rm: RateMatrix, p_exc: float,
                        T_laser: float, n_pulses: int,
                        k_fret: float = 0.0,
                        QY_donor: float = 1.0,
                        QY_acceptor: float = 1.0) -> Dict:
    """Build Poisson Steady-State matrices for one fluorophore type.

    Returns dict with T_step, mean_ph, mean_fret, tau_eff.

    Parameters
    ----------
    rm : RateMatrix
        Donor fluorophore rate matrix.
    p_exc : float
        Single-pulse excitation probability.
    T_laser : float
        Laser period (s).
    n_pulses : int
        Number of laser pulses per BD step.
    k_fret : float
        FRET rate (s⁻¹), 0 if no FRET.
    QY_donor : float
        Quantum yield of the donor fluorophore (0-1).
        Fraction of S1→S0 transitions that emit a photon.
        Ref: Lakowicz (2006) Ch.1 Eq.1.2: QY = k_rad / (k_rad + k_nr)
    QY_acceptor : float
        Quantum yield of the FRET acceptor (0-1).
        Applied to FRET-sensitized acceptor photon counts.

    Ref: Phase 6 Spec §2-3, PyBroMo, SMIS.
    """
    s0, s1 = rm.ground_state, rm.excited_state
    t1, bl = rm.t1_state, rm.bleached_state

    # S1 branching (with optional FRET)
    k_S1_total = rm.exit_rates[s1] + k_fret if (k_fret > 0 and s1 >= 0) else rm.exit_rates[s1]

    if k_S1_total > 0 and s1 >= 0:
        p_rad = rm.Q[s1, s0] / k_S1_total
        p_fret_br = k_fret / k_S1_total if k_fret > 0 else 0.0
        p_isc = rm.Q[s1, t1] / k_S1_total if t1 >= 0 else 0.0
        p_bl_s1 = rm.Q[s1, bl] / k_S1_total if bl >= 0 else 0.0
    else:
        p_rad = p_fret_br = p_isc = p_bl_s1 = 0.0

    p_S0_return = p_rad + p_fret_br

    # T1 per-period transitions
    if t1 >= 0:
        k_T1 = rm.exit_rates[t1]
        p_T1_exit = 1.0 - math.exp(-k_T1 * T_laser) if k_T1 > 0 else 0.0
        p_T1_to_S0 = rm.branch_probs[t1, s0]
        p_T1_to_bl = rm.branch_probs[t1, bl] if bl >= 0 else 0.0
    else:
        p_T1_exit = p_T1_to_S0 = p_T1_to_bl = 0.0

    # 3×3 T_period: {S0, T1, Bleached}
    T_period = np.zeros((3, 3), dtype=np.float64)
    T_period[0, 0] = (1.0 - p_exc) + p_exc * p_S0_return
    T_period[0, 1] = p_exc * p_isc
    T_period[0, 2] = p_exc * p_bl_s1
    T_period[1, 0] = p_T1_exit * p_T1_to_S0
    T_period[1, 1] = 1.0 - p_T1_exit
    T_period[1, 2] = p_T1_exit * p_T1_to_bl
    T_period[2, 2] = 1.0

    # Normalize rows (safety)
    for row in range(3):
        rs = T_period[row].sum()
        if abs(rs - 1.0) > 1e-12:
            T_period[row] /= rs

    T_step = np.linalg.matrix_power(T_period, n_pulses)

    # Geometric series: S_geo = (I - T_sub^n) @ inv(I - T_sub)
    T_sub = T_period[:2, :2].copy()
    I2 = np.eye(2, dtype=np.float64)
    T_sub_n = np.linalg.matrix_power(T_sub, n_pulses)
    try:
        S_geo = (I2 - T_sub_n) @ np.linalg.inv(I2 - T_sub)
    except np.linalg.LinAlgError:
        S_geo = np.zeros((2, 2), dtype=np.float64)
        T_k = I2.copy()
        for _ in range(n_pulses):
            S_geo += T_k
            T_k = T_k @ T_sub

    # Donor photon probability: only fraction QY_donor of S1→S0 events emit
    # p_rad = branching fraction to S0 (includes radiative + non-radiative)
    # p_donor_ph = p_exc × p_rad × QY_donor (only radiative fraction counts)
    # Ref: Lakowicz (2006) Ch.1: QY = Γ/(Γ+k_nr) = k_rad/k_total_S1→S0
    p_donor_ph = p_exc * p_rad * QY_donor
    if p_donor_ph <= 0.0:
        mean_ph = np.zeros(2, dtype=np.float64)
    else:
        mean_ph = np.array([S_geo[0, 0] * p_donor_ph, S_geo[1, 0] * p_donor_ph])

    # FRET acceptor photon probability: each FRET event transfers energy to
    # the acceptor, which emits with its own quantum yield QY_acceptor
    # Ref: Lakowicz (2006) Ch.13 Eq.13.17
    p_fret_ph = p_exc * p_fret_br * QY_acceptor
    if p_fret_ph <= 0.0:
        mean_fret = np.zeros(2, dtype=np.float64)
    else:
        mean_fret = np.array([S_geo[0, 0] * p_fret_ph, S_geo[1, 0] * p_fret_ph])

    tau_eff = 1.0 / k_S1_total if k_S1_total > 0 else 4.1e-9

    return {
        'T_step': T_step,
        'mean_ph': mean_ph,
        'mean_fret': mean_fret,
        'tau_eff': tau_eff,
    }


# ═══════════════════════════════════════════════════════════════════════
# PSS NUMBA KERNEL
# ═══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _evolve_pss_kernel(
    ctmc_state, is_bleached, has_dye, is_alive, dye_type_id,
    k_fret_per_particle, fret_acceptor_idx,
    n_fluoro_types,
    T_step_all, T_step_fret_all,
    mean_ph_all, mean_ph_fret_all,
    mean_fret_all, mean_fret_fret_all,
    tau_all, tau_fret_all,
    k_fret_bound_all,
    ground_state_all, excited_state_all, t1_state_all, bl_state_all,
    out_particle, out_time, out_channel, out_pulse, out_is_fret,
    max_photons, N,
    rng_poisson, rng_transition, rng_arrival,
):
    """PSS kernel: Poisson photon draws + Exponential arrival times.

    Algorithm per emitter i:
      1. Map ctmc_state to PSS state {S0=0, T1=1}
      2. Select normal or FRET matrices (interpolated by actual k_fret)
      3. Draw n_photons ~ Poisson(mean_ph[type][state])
      4. Draw n_fret ~ Poisson(mean_fret[type][state])
      5. For each photon: t = -τ_DA(i) × ln(U) (Exponential, per-particle)
      6. State transition via T_step multinomial

    Per-particle τ_DA fix (Förster 1948):
      τ_DA(i) = τ_D / (1 + k_fret[i] × τ_D)
      This replaces the previous constant tau_fret_all[dtype] which was
      precomputed at E_high only, causing ALL donors to emit with the
      quenched lifetime regardless of actual FRET state.
    """
    n_photons = 0
    arr_idx = 0

    for i in range(N):
        if not has_dye[i] or not is_alive[i] or is_bleached[i]:
            continue
        dtype = dye_type_id[i]
        if dtype < 0 or dtype >= n_fluoro_types:
            continue

        s0 = ground_state_all[dtype]
        s1 = excited_state_all[dtype]
        t1 = t1_state_all[dtype]
        bl = bl_state_all[dtype]

        cur = ctmc_state[i]
        if cur == s0 or cur == s1:
            pss = 0
        elif t1 >= 0 and cur == t1:
            pss = 1
        elif bl >= 0 and cur == bl:
            is_bleached[i] = True
            continue
        else:
            pss = 0

        has_fret = k_fret_per_particle[i] > 0.0
        if has_fret:
            # Per-particle τ_DA from actual k_fret (Förster 1948)
            tau_D = tau_all[dtype]
            k_f_i = k_fret_per_particle[i]
            tau = tau_D / (1.0 + k_f_i * tau_D)

            # Interpolate mean photon counts between no-FRET and FRET
            # using fractional k_fret relative to k_fret_bound
            k_bound = k_fret_bound_all[dtype]
            if k_bound > 0.0:
                frac = min(k_f_i / k_bound, 1.0)
            else:
                frac = 0.0
            mean_d = mean_ph_all[dtype, pss] * (1.0 - frac) + mean_ph_fret_all[dtype, pss] * frac
            mean_f = mean_fret_all[dtype, pss] * (1.0 - frac) + mean_fret_fret_all[dtype, pss] * frac
            T00 = T_step_all[dtype, pss, 0] * (1.0 - frac) + T_step_fret_all[dtype, pss, 0] * frac
            # BUG FIX 2026-03-04: was T_step_all[..., 0] (S0→S0) for no-FRET
            # term — should be T_step_all[..., 1] (S0→T1 ISC probability).
            # Bug had zero impact for binary FRET (frac∈{0,1}) but would cause
            # 1666× ISC overestimate at frac=0.5 for continuous/proximity FRET.
            T01 = T_step_all[dtype, pss, 1] * (1.0 - frac) + T_step_fret_all[dtype, pss, 1] * frac
        else:
            mean_d = mean_ph_all[dtype, pss]
            mean_f = mean_fret_all[dtype, pss]
            tau = tau_all[dtype]
            T00 = T_step_all[dtype, pss, 0]
            T01 = T_step_all[dtype, pss, 1]

        # ── Poisson draw (Knuth / Normal) ──
        n_d = _poisson_draw(mean_d, rng_poisson[i])
        n_f = 0
        if mean_f > 0.0 and has_fret:
            # Use slightly modified seed for FRET draw
            n_f = _poisson_draw(mean_f, rng_poisson[i] * 0.5 + 0.25)

        # ── Donor photons ──
        for j in range(n_d):
            if n_photons >= max_photons:
                break
            if arr_idx < len(rng_arrival):
                u = rng_arrival[arr_idx]
                arr_idx += 1
                if u < 1e-30:
                    u = 1e-30
                t_arr = -tau * math.log(u)
            else:
                t_arr = tau
            out_particle[n_photons] = i
            out_time[n_photons] = t_arr
            out_channel[n_photons] = dtype
            out_pulse[n_photons] = 0
            out_is_fret[n_photons] = False
            n_photons += 1

        # ── FRET acceptor photons ──
        acc = fret_acceptor_idx[i]
        if n_f > 0 and acc >= 0:
            acc_t = dye_type_id[acc]
            tau_a = tau_all[acc_t] if (acc_t >= 0 and acc_t < n_fluoro_types) else tau
            for j in range(n_f):
                if n_photons >= max_photons:
                    break
                if arr_idx < len(rng_arrival):
                    u = rng_arrival[arr_idx]
                    arr_idx += 1
                    if u < 1e-30:
                        u = 1e-30
                    t_arr = -tau_a * math.log(u)
                else:
                    t_arr = tau_a
                out_particle[n_photons] = acc
                out_time[n_photons] = t_arr
                out_channel[n_photons] = acc_t  # fluorophore index, NOT hardcoded
                out_pulse[n_photons] = 0
                out_is_fret[n_photons] = True
                n_photons += 1

        # ── State transition ──
        u_t = rng_transition[i]
        if u_t < T00:
            ctmc_state[i] = s0
        elif u_t < T00 + T01:
            ctmc_state[i] = t1 if t1 >= 0 else s0
        else:
            ctmc_state[i] = bl if bl >= 0 else s0
            is_bleached[i] = True

    return n_photons


@njit(cache=True)
def _poisson_draw(mean: float, u_seed: float) -> int:
    """Draw from Poisson(mean) using Knuth or Normal approximation.

    Uses a single U(0,1) seed to generate a deterministic PRNG stream
    via LCG for the Knuth algorithm.
    """
    if mean <= 0.0:
        return 0
    # Initialize LCG from seed
    s = np.uint64(np.int64(u_seed * 9.007199254740992e15))  # 2^53
    if mean < 30.0:
        # Knuth's algorithm
        L = math.exp(-mean)
        k = 0
        p = 1.0
        while True:
            s = s * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            u = np.float64(s >> np.uint64(11)) / 9.007199254740992e15
            p *= u
            if p <= L:
                break
            k += 1
            if k > 10000:
                break
        return k
    else:
        # Normal approximation
        s = s * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
        u1 = np.float64(s >> np.uint64(11)) / 9.007199254740992e15
        s = s * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
        u2 = np.float64(s >> np.uint64(11)) / 9.007199254740992e15
        if u1 < 1e-30:
            u1 = 1e-30
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(6.283185307179586 * u2)
        return max(0, int(mean + math.sqrt(mean) * z + 0.5))


# ═══════════════════════════════════════════════════════════════════════
# FRET RATE COMPUTATION (Step 2 optimization: bonded-only)
# ═══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _compute_fret_rates_numba(
    k_fret, fret_acceptor_idx,
    positions, species_id, has_dye, is_alive,
    is_bleached, bond_partner, dye_type_id,
    N, R0_nm, tau_D, donor_fid, acceptor_fid,
    intra_complex_dist,
    Lx, Ly, periodic_x, periodic_y,
    intramolecular,
):
    """Compute FRET rates — bonded pairs only (Phase 6 Step 2).

    Free-donor O(N²) search eliminated: at mean inter-particle distance
    82 nm >> 3×R0 = 19.2 nm, FRET efficiency E < 10⁻²⁴.
    Only bonded complexes have significant FRET.

    intra_complex_dist: D-A distance within bound complex (nm),
    a structural parameter from config, NOT the BD inter-particle distance.
    Ref: Ogiso et al. (2002) Cell 110:775 (EGFR-EGF, PDB: 1NQL)

    intramolecular: if True, donor gets FRET when bonded regardless of
    whether partner carries acceptor. Models biosensors (PH-Akt) where
    donor+acceptor are on the same construct and binding triggers
    conformational change that activates FRET.
    """
    for i in range(N):
        if not has_dye[i] or is_bleached[i]:
            continue
        if dye_type_id[i] != donor_fid:
            continue

        partner = bond_partner[i]
        if partner >= 0:
            if intramolecular:
                # Intramolecular FRET: binding event activates FRET
                # within the same construct (D and A on same molecule).
                # Partner doesn't need acceptor — conformational change
                # brings D-A within intra_complex_dist upon binding.
                r_nm = intra_complex_dist
                ratio = R0_nm / r_nm
                r6 = ratio * ratio * ratio * ratio * ratio * ratio
                k_fret[i] = r6 / tau_D
                # BUG FIX 2026-03-04: was `fret_acceptor_idx[i] = i` (self).
                # For intramolecular FRET (e.g. PHAkt mTFP1-Venus), the
                # acceptor (Venus) emits at ~528 nm, outside the donor
                # emission bandpass (~470-510 nm). Setting idx=i caused
                # dye_type_id[i]=mTFP1(0) → FRET photons entered channel 0
                # with τ=2.793 ns (unquenched), contaminating donor TCSPC
                # with +1.05 ns bias per bound molecule.
                # Fix: idx=-1 → PSS kernel skips acceptor photon generation.
                # k_fret[i] is still set correctly above, so donor τ_DA and
                # mean photon count remain accurate via PSS interpolation.
                # Ref: Diagnostic protocol C (TCSPC channel purity audit).
                fret_acceptor_idx[i] = -1  # no acceptor in donor channel
            elif has_dye[partner] and dye_type_id[partner] == acceptor_fid:
                r_nm = intra_complex_dist  # configurable D-A distance
                ratio = R0_nm / r_nm
                r6 = ratio * ratio * ratio * ratio * ratio * ratio
                k_fret[i] = r6 / tau_D
                fret_acceptor_idx[i] = partner
        # Free donors: k_fret stays 0 (set by caller)


@njit(cache=True)
def _compute_fret_rates_proximity_numba(
    k_fret, fret_acceptor_idx,
    positions, species_id, has_dye, is_alive,
    is_bleached, dye_type_id,
    N,
    tau_D,
    donor_fid,
    target_species_id,
    proximity_radius_um,
    k_fret_high,
    k_fret_low,
    Lx, Ly, periodic_x, periodic_y,
):
    """Compute FRET rates for proximity-based sensors (Phase 4, biosensor mode).

    For each donor particle, search for target species within proximity_radius.
    If target found: k_fret[i] = k_fret_high (quenched state)
    If no target:    k_fret[i] = k_fret_low (basal state)

    This biosensor mode (e.g., PH-Akt detecting PIP3) uses membrane recruitment,
    not direct binding, to modulate FRET efficiency.

    Parameters
    ----------
    k_fret : output array, per-particle FRET rates (s⁻¹)
    fret_acceptor_idx : output array, index of acceptor particle (-1 for proximity mode)
    positions : particle positions (um)
    species_id : particle species indices
    has_dye : particle has fluorophore?
    is_alive : particle alive?
    is_bleached : particle bleached?
    dye_type_id : fluorophore type index for each particle
    N : number of particles
    tau_D : donor excited-state lifetime (s)
    donor_fid : fluorophore type index of donor
    target_species_id : species ID that attracts sensor (e.g., PIP3)
    proximity_radius_um : detection radius (μm)
    k_fret_high : FRET rate when target nearby (s⁻¹), from E_high
    k_fret_low : FRET rate when target absent (s⁻¹), from E_low
    Lx, Ly : membrane dimensions (um)
    periodic_x, periodic_y : periodicity flags

    Ref: Nishida et al. (2020) bioRxiv (FRET biosensors);
         Förster (1948), Lakowicz (2006) Ch.13.
    """
    proximity_radius_sq = proximity_radius_um * proximity_radius_um

    for i in range(N):
        if not has_dye[i] or not is_alive[i] or is_bleached[i]:
            continue
        if dye_type_id[i] != donor_fid:
            continue

        # Search for target species within proximity radius
        target_found = False
        for j in range(N):
            if j == i:
                continue
            if not is_alive[j] or species_id[j] != target_species_id:
                continue

            # Compute minimum image distance (periodic boundary conditions)
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]

            if periodic_x:
                if dx > Lx / 2.0:
                    dx -= Lx
                elif dx < -Lx / 2.0:
                    dx += Lx

            if periodic_y:
                if dy > Ly / 2.0:
                    dy -= Ly
                elif dy < -Ly / 2.0:
                    dy += Ly

            dist_sq = dx * dx + dy * dy

            if dist_sq < proximity_radius_sq:
                target_found = True
                break

        # Set FRET rate based on proximity detection
        if target_found:
            k_fret[i] = k_fret_high
        else:
            k_fret[i] = k_fret_low

        # For proximity FRET, no specific acceptor particle
        # (energy is quenched through non-radiative pathways)
        fret_acceptor_idx[i] = -1


# ═══════════════════════════════════════════════════════════════════════
# PHOTON SIMULATOR
# ═══════════════════════════════════════════════════════════════════════

class PhotonSimulator:
    """Orchestrator for photophysics (PSS-optimized, Phase 6)."""

    def __init__(self, config: SimulationConfig, state: ParticleState,
                 scan_duty_fraction: float = 1.0):
        self.config = config
        self.state = state

        self.rate_matrices: List[RateMatrix] = []
        for fluo in config.fluorophores:
            self.rate_matrices.append(RateMatrix(fluo))

        self.n_fluoro_types = len(self.rate_matrices)
        if self.n_fluoro_types == 0:
            self._active = False
            return
        self._active = True

        self.max_states = max(rm.n_states for rm in self.rate_matrices)
        nf = self.n_fluoro_types
        ns = self.max_states

        # Pack arrays for numba (backwards compat)
        self._exit_rates_all = np.zeros((nf, ns), dtype=np.float64)
        self._branch_cdf_all = np.zeros((nf, ns, ns), dtype=np.float64)
        self._fluorescent_mask_all = np.zeros((nf, ns), dtype=np.bool_)
        self._terminal_mask_all = np.zeros((nf, ns), dtype=np.bool_)
        self._absorbing_mask_all = np.zeros((nf, ns), dtype=np.bool_)
        self._ground_state_all = np.zeros(nf, dtype=np.int32)
        self._excited_state_all = np.full(nf, -1, dtype=np.int32)
        self._n_states_all = np.zeros(nf, dtype=np.int32)
        self._t1_state_all = np.full(nf, -1, dtype=np.int32)
        self._bl_state_all = np.full(nf, -1, dtype=np.int32)

        for t, rm in enumerate(self.rate_matrices):
            n = rm.n_states
            self._n_states_all[t] = n
            self._exit_rates_all[t, :n] = rm.exit_rates[:n]
            self._branch_cdf_all[t, :n, :n] = rm.branch_probs[:n, :n]
            self._fluorescent_mask_all[t, :n] = rm.fluorescent_mask[:n]
            self._terminal_mask_all[t, :n] = rm.terminal_mask[:n]
            self._absorbing_mask_all[t, :n] = rm.absorbing_mask[:n]
            self._ground_state_all[t] = rm.ground_state
            self._excited_state_all[t] = rm.excited_state
            self._t1_state_all[t] = rm.t1_state
            self._bl_state_all[t] = rm.bleached_state

        # Laser parameters
        #
        # Scan duty fraction correction (Session 8, Bug Fix #6):
        #   In a confocal microscope, each molecule is illuminated ONLY
        #   during the pixel dwell time when the scanner passes over it,
        #   NOT continuously during the entire frame interval.
        #
        #   real_pulses_per_frame = dwell_time × rep_rate = 800
        #   n_bd_per_frame = frame_interval / bd_timestep = 100,000
        #   illuminated_fraction = real_pulses / (n_pulses_per_step × n_bd) 
        #
        #   We precompute DARK matrices (p_exc=0, only T1 relaxation)
        #   and use them for the majority of BD steps. Only a fraction
        #   scan_duty_fraction of steps use the illuminated matrices.
        #
        #   Ref: Becker W (2005) Advanced TCSPC Techniques, §3.2
        #
        self.scan_duty_fraction = max(0.0, min(1.0, scan_duty_fraction))
        self.laser_period = config.laser.period
        self.n_pulses_per_step = max(
            1, int(round(config.bd_timestep / self.laser_period)))

        # Excitation probability per fluorophore type
        self._p_excitation = np.zeros(nf, dtype=np.float64)
        for t, fluo in enumerate(config.fluorophores):
            self._p_excitation[t] = self._compute_excitation_prob(fluo)

        # FRET setup
        self._fret_pairs = config.fret_pairs
        self._dye_attachments = config.dye_attachments
        self._species_name_to_id = {s.name: i for i, s in enumerate(config.species)}
        self._fluoro_name_to_id = {f.name: i for i, f in enumerate(config.fluorophores)}
        self._fret_maps = self._build_fret_maps()

        # ── D/A channel identity (from FRET pair config) ──────────────
        # These are the fluorophore indices that serve as channels.
        # channel = fluorophore index in config.fluorophores[].
        if len(self._fret_maps) > 0:
            self.donor_channel = self._fret_maps[0]['donor_fluoro_id']
            self.acceptor_channel = self._fret_maps[0]['acceptor_fluoro_id']
        else:
            self.donor_channel = 0
            self.acceptor_channel = 1 if nf > 1 else 0

        N = state.N
        self._k_fret = np.zeros(N, dtype=np.float64)
        self._fret_acceptor_idx = np.full(N, -1, dtype=np.int32)

        # ── PSS precomputation ────────────────────────────────────────
        self._precompute_pss_matrices()

        # Output buffers (smaller than PbP — PSS produces fewer max photons)
        n_labeled = int(np.sum(state.has_dye))
        max_mean = 0.0
        for t in range(nf):
            max_mean = max(max_mean, float(self._mean_ph_all[t, 0]))
        est_max = max(1000, int(n_labeled * max_mean * 3))
        self._max_photons = est_max
        self._out_particle = np.empty(est_max, dtype=np.int32)
        self._out_time = np.empty(est_max, dtype=np.float64)
        self._out_channel = np.empty(est_max, dtype=np.int8)
        self._out_pulse = np.empty(est_max, dtype=np.int32)
        self._out_is_fret = np.empty(est_max, dtype=np.bool_)

        # RNG
        self._rng = np.random.default_rng(config.random_seed + 1000)

        # Statistics
        self.total_donor_photons = 0
        self.total_acceptor_photons = 0
        self.total_fret_events = 0
        self.total_bleached = 0

    def _precompute_pss_matrices(self) -> None:
        """Precompute PSS matrices for all fluorophore types (normal + FRET).

        QY correction (Session 7, Audit Fix #1):
          Each fluorophore type's quantum_yield from its fluorescent state
          is applied to photon emission probabilities. This ensures that
          mean_ph reflects actual emitted photons (not all S1→S0 events).
          Ref: Lakowicz (2006) Ch.1 Eq.1.2
        """
        nf = self.n_fluoro_types
        n_pulses = self.n_pulses_per_step
        T_laser = self.laser_period

        # ── Extract QY per fluorophore type from fluorescent state ────
        self._QY_per_type = np.ones(nf, dtype=np.float64)
        for t, rm in enumerate(self.rate_matrices):
            if rm.excited_state >= 0:
                qy = rm.fluorophore.states[rm.excited_state].quantum_yield
                self._QY_per_type[t] = qy

        # Normal (no FRET)
        self._T_step_all = np.zeros((nf, 3, 3), dtype=np.float64)
        self._mean_ph_all = np.zeros((nf, 2), dtype=np.float64)
        self._mean_fret_all = np.zeros((nf, 2), dtype=np.float64)
        self._tau_all = np.zeros(nf, dtype=np.float64)

        # FRET versions (precomputed for bound-complex k_FRET)
        self._T_step_fret_all = np.zeros((nf, 3, 3), dtype=np.float64)
        self._mean_ph_fret_all = np.zeros((nf, 2), dtype=np.float64)
        self._mean_fret_fret_all = np.zeros((nf, 2), dtype=np.float64)
        self._tau_fret_all = np.zeros(nf, dtype=np.float64)
        self._k_fret_bound_all = np.zeros(nf, dtype=np.float64)  # k_FRET used for precomputing _fret matrices

        for t, rm in enumerate(self.rate_matrices):
            # Scale p_exc by scan_duty_fraction so that total excitation
            # budget over all BD steps matches real confocal exposure.
            # n_pulses_per_step × n_bd_per_frame × p_exc_eff should give
            # the same total excitation as dwell_time × rep_rate × p_exc.
            p_exc = self._p_excitation[t] * self.scan_duty_fraction
            QY_d = float(self._QY_per_type[t])

            # Determine acceptor QY for this donor type's FRET pair
            QY_a = 1.0  # default if no FRET pair or acceptor QY not set
            for fmap in self._fret_maps:
                if fmap['donor_fluoro_id'] == t:
                    acc_id = fmap['acceptor_fluoro_id']
                    if 0 <= acc_id < nf:
                        QY_a = float(self._QY_per_type[acc_id])
                    break

            # Normal matrices (no FRET)
            pss = _build_pss_matrices(rm, p_exc, T_laser, n_pulses,
                                      k_fret=0.0,
                                      QY_donor=QY_d, QY_acceptor=QY_a)
            self._T_step_all[t] = pss['T_step']
            self._mean_ph_all[t] = pss['mean_ph']
            self._mean_fret_all[t] = pss['mean_fret']
            self._tau_all[t] = pss['tau_eff']

            # FRET matrices (precomputed at bound-complex distance)
            # k_FRET_bound = (1/tau_D) * (R0/r_bound)^6
            # Default: r_bound = 5 nm, R0 = 6.4 nm (from FRET pairs)
            k_fret_bound = self._get_bound_fret_rate(t)
            self._k_fret_bound_all[t] = k_fret_bound
            if k_fret_bound > 0:
                pss_f = _build_pss_matrices(rm, p_exc, T_laser, n_pulses,
                                            k_fret=k_fret_bound,
                                            QY_donor=QY_d, QY_acceptor=QY_a)
                self._T_step_fret_all[t] = pss_f['T_step']
                self._mean_ph_fret_all[t] = pss_f['mean_ph']
                self._mean_fret_fret_all[t] = pss_f['mean_fret']
                self._tau_fret_all[t] = pss_f['tau_eff']
            else:
                # Copy normal as fallback
                self._T_step_fret_all[t] = pss['T_step']
                self._mean_ph_fret_all[t] = pss['mean_ph']
                self._mean_fret_fret_all[t] = pss['mean_fret']
                self._tau_fret_all[t] = pss['tau_eff']

    def _get_fret_pair_config(self, fmap: Dict) -> Optional[FRETpair]:
        """Find FRETpair config matching this fret_map.

        Parameters
        ----------
        fmap : dict
            FRET map with 'donor_fluoro_id' and 'acceptor_fluoro_id'

        Returns
        -------
        FRETpair or None
            Config matching the fluorophore pair, or None if not found.
        """
        donor_fid = fmap['donor_fluoro_id']
        acceptor_fid = fmap['acceptor_fluoro_id']
        for fp in self._fret_pairs:
            d_id = self._fluoro_name_to_id.get(fp.donor_type, -1)
            a_id = self._fluoro_name_to_id.get(fp.acceptor_type, -1)
            if d_id == donor_fid and a_id == acceptor_fid:
                return fp
        return None

    def _get_bound_fret_rate(self, fluoro_type_idx: int) -> float:
        """Get the k_FRET for a bound complex or proximity-high donor.

        For proximity mode (biosensors), returns k_FRET from proximity_E_high.
        For bond mode, returns k_FRET from intra_complex_distance.

        Parameters
        ----------
        fluoro_type_idx : int
            Fluorophore type index

        Returns
        -------
        float
            k_FRET rate (s⁻¹) for the high-FRET state
        """
        for fmap in self._fret_maps:
            if fmap['donor_fluoro_id'] == fluoro_type_idx:
                fp_cfg = self._get_fret_pair_config(fmap)
                if fp_cfg and getattr(fp_cfg, 'fret_mode', 'bond') == 'proximity':
                    # Proximity mode: use E_high
                    rm = self.rate_matrices[fluoro_type_idx]
                    tau_D = rm.get_lifetime(rm.excited_state)
                    if tau_D <= 0 or math.isinf(tau_D):
                        return 0.0
                    E_high = fp_cfg.proximity_E_high
                    if E_high >= 1.0:
                        return 1e15
                    return E_high / ((1.0 - E_high) * tau_D)
                else:
                    # Bond mode: existing logic
                    R0_nm = fmap['R0']
                    rm = self.rate_matrices[fluoro_type_idx]
                    tau_D = rm.get_lifetime(rm.excited_state)
                    if tau_D <= 0 or math.isinf(tau_D):
                        return 0.0
                    r_nm = fmap['intra_complex_dist']  # from config
                    ratio = R0_nm / r_nm
                    r6 = ratio ** 6
                    return r6 / tau_D
        return 0.0

    def _compute_excitation_prob(self, fluo: FluorophoreType) -> float:
        """Compute single-pulse excitation probability.

        p_exc = σ_abs(λ_laser) × (n_photons_per_pulse / A_focus)

        σ_abs in config is the PEAK value at excitation_wavelength.
        When laser wavelength differs from the fluorophore's excitation peak,
        we apply a Gaussian spectral correction:
            σ_abs(λ) = σ_abs_peak × exp(-0.5 × ((λ - λ_peak) / σ_spec)²)
        where σ_spec = FWHM / 2.355 and FWHM ≈ 45 nm for typical FPs.

        Ref: Lakowicz 2006 Ch.1, Bourgeois SMIS 2023
        Spectral model: Shaner et al. (2005) Nat Methods 2:905
        """
        h = 6.626e-34
        c = 3.0e8
        laser = self.config.laser
        wavelength_m = laser.wavelength * 1e-9
        E_photon = h * c / wavelength_m
        w0_m = laser.beam_waist * 1e-9
        A_focus = math.pi * w0_m ** 2
        E_per_pulse = laser.power / laser.repetition_rate
        n_photons_per_pulse = E_per_pulse / E_photon
        flux = n_photons_per_pulse / A_focus
        sigma_abs_m2 = fluo.absorption_cross_section * 1e-4

        # Spectral correction: σ_abs in config is peak value at excitation_wavelength.
        # Apply Gaussian model when laser wavelength ≠ fluorophore peak.
        # FWHM ≈ 45 nm is typical for fluorescent proteins (Shaner 2005).
        if fluo.excitation_wavelength > 0:
            delta_lambda = abs(laser.wavelength - fluo.excitation_wavelength)
            if delta_lambda > 1.0:  # >1nm mismatch → apply correction
                fwhm = getattr(fluo, 'absorption_fwhm', 45.0)  # nm
                sigma_spec = fwhm / 2.355
                correction = math.exp(-0.5 * (delta_lambda / sigma_spec) ** 2)
                sigma_abs_m2 *= correction

        p_exc = sigma_abs_m2 * flux
        return min(p_exc, 0.9)

    def _build_fret_maps(self) -> List[Dict]:
        """Build donor→acceptor species mapping from FRET pairs."""
        fret_maps = []
        for fp in self._fret_pairs:
            donor_fluoro_id = self._fluoro_name_to_id.get(fp.donor_type, -1)
            acceptor_fluoro_id = self._fluoro_name_to_id.get(fp.acceptor_type, -1)
            donor_species = set()
            acceptor_species = set()
            for att in self._dye_attachments:
                if att.fluorophore == fp.donor_type:
                    sp_id = self._species_name_to_id.get(att.species, -1)
                    if sp_id >= 0:
                        donor_species.add(sp_id)
                if att.fluorophore == fp.acceptor_type:
                    sp_id = self._species_name_to_id.get(att.species, -1)
                    if sp_id >= 0:
                        acceptor_species.add(sp_id)
            fret_maps.append({
                'R0': fp.R0,
                'kappa2': fp.kappa2,
                'intra_complex_dist': fp.intra_complex_distance,
                'donor_fluoro_id': donor_fluoro_id,
                'acceptor_fluoro_id': acceptor_fluoro_id,
                'donor_species': donor_species,
                'acceptor_species': acceptor_species,
            })
        return fret_maps

    def update_fret_rates(self) -> None:
        """Recompute k_FRET for all donor particles (bonded + proximity, Phase 4-6).

        Phase 6: Bonded-only FRET via _compute_fret_rates_numba (O(bonded) only).
        Phase 4: Proximity-based biosensor FRET via _compute_fret_rates_proximity_numba
                 for sensors like PH-Akt that detect PIP3 via membrane recruitment.
        """
        N = self.state.N
        self._k_fret[:] = 0.0
        self._fret_acceptor_idx[:] = -1

        positions = self.state.positions
        species_id = self.state.species_id
        has_dye = self.state.has_dye
        is_alive = self.state.is_alive
        is_bleached = self.state.is_bleached
        bond_partner = self.state.bond_partner
        dye_type_id = self.state.dye_type_id
        Lx, Ly = self.config.membrane.size
        periodic_x, periodic_y = self.config.membrane.periodic

        # Phase 6: Bonded-pair FRET
        for fmap in self._fret_maps:
            fp_cfg = self._get_fret_pair_config(fmap)
            if fp_cfg and getattr(fp_cfg, 'fret_mode', 'bond') == 'proximity':
                # Skip bond mode for proximity-mode pairs (handled below)
                continue

            R0_nm = fmap['R0']
            donor_fid = fmap['donor_fluoro_id']
            acceptor_fid = fmap['acceptor_fluoro_id']
            if donor_fid < 0 or donor_fid >= self.n_fluoro_types:
                continue
            rm_donor = self.rate_matrices[donor_fid]
            tau_D = rm_donor.get_lifetime(rm_donor.excited_state)
            if tau_D <= 0 or math.isinf(tau_D):
                continue
            # Check if this is intramolecular FRET (biosensor mode)
            is_intramolecular = False
            if fp_cfg is not None:
                is_intramolecular = getattr(fp_cfg, 'intramolecular', False)
            _compute_fret_rates_numba(
                self._k_fret, self._fret_acceptor_idx,
                positions, species_id, has_dye, is_alive,
                is_bleached, bond_partner, dye_type_id,
                N, R0_nm, tau_D, donor_fid, acceptor_fid,
                fmap['intra_complex_dist'],
                Lx, Ly, periodic_x, periodic_y,
                is_intramolecular,
            )

        # Phase 4: Proximity-based FRET (biosensor mode)
        for fp_idx, fp_cfg in enumerate(self._fret_pairs):
            if getattr(fp_cfg, 'fret_mode', 'bond') != 'proximity':
                continue

            # Get donor fluorophore info
            donor_fid = self._fluoro_name_to_id.get(fp_cfg.donor_type, -1)
            if donor_fid < 0 or donor_fid >= self.n_fluoro_types:
                continue

            rm_donor = self.rate_matrices[donor_fid]
            tau_D = rm_donor.get_lifetime(rm_donor.excited_state)
            if tau_D <= 0 or math.isinf(tau_D):
                continue

            # Get target species id
            target_sp_id = -1
            for si, sp in enumerate(self.config.species):
                if sp.name == fp_cfg.proximity_target_species:
                    target_sp_id = si
                    break
            if target_sp_id < 0:
                continue

            # Convert E_high/E_low to k_FRET: k_FRET = E / ((1-E) * tau_D)
            E_high = fp_cfg.proximity_E_high
            E_low = fp_cfg.proximity_E_low
            k_fret_high = (E_high / ((1.0 - E_high) * tau_D)
                           if E_high < 1.0 else 1e15)
            k_fret_low = (E_low / ((1.0 - E_low) * tau_D)
                          if E_low < 1.0 else 1e15)

            # Convert proximity_radius from nm to µm
            proximity_radius_um = fp_cfg.proximity_radius * 1e-3

            _compute_fret_rates_proximity_numba(
                self._k_fret, self._fret_acceptor_idx,
                positions, species_id, has_dye, is_alive,
                is_bleached, dye_type_id,
                N, tau_D, donor_fid, target_sp_id,
                proximity_radius_um, k_fret_high, k_fret_low,
                Lx, Ly, periodic_x, periodic_y,
            )

    def evolve_one_bd_step(self) -> PhotonBatch:
        """Evolve photophysics for one BD step using PSS kernel.

        Scan duty fraction correction (Session 8, Bug Fix #6):
          The excitation probability p_exc is pre-scaled by
          scan_duty_fraction in _precompute_pss_matrices, so that
          the total excitation budget across all BD steps matches
          real confocal exposure (dwell_time × rep_rate pulses).
          No BD steps are skipped — dynamics and state evolution
          proceed on every step.
        """
        if not self._active:
            return PhotonBatch.empty()

        N = self.state.N

        # 1. Update FRET rates
        self.update_fret_rates()

        # 2. Generate RNG (PSS: only N + N + estimated_photons)
        rng_poisson = self._rng.random(N)
        rng_transition = self._rng.random(N)
        # Estimate max photons for arrival time RNG
        max_mean = 0.0
        for t in range(self.n_fluoro_types):
            m = max(float(self._mean_ph_all[t, 0]),
                    float(self._mean_ph_fret_all[t, 0]))
            max_mean = max(max_mean, m)
        est_photons = max(1000, int(N * max_mean * 2.5))
        rng_arrival = self._rng.random(est_photons)

        # 3. Run PSS kernel
        n_photons = _evolve_pss_kernel(
            self.state.ctmc_state,
            self.state.is_bleached,
            self.state.has_dye,
            self.state.is_alive,
            self.state.dye_type_id,
            self._k_fret,
            self._fret_acceptor_idx,
            self.n_fluoro_types,
            self._T_step_all,
            self._T_step_fret_all,
            self._mean_ph_all,
            self._mean_ph_fret_all,
            self._mean_fret_all,
            self._mean_fret_fret_all,
            self._tau_all,
            self._tau_fret_all,
            self._k_fret_bound_all,
            self._ground_state_all,
            self._excited_state_all,
            self._t1_state_all,
            self._bl_state_all,
            self._out_particle,
            self._out_time,
            self._out_channel,
            self._out_pulse,
            self._out_is_fret,
            self._max_photons,
            N,
            rng_poisson, rng_transition, rng_arrival,
        )

        # 4. Package results
        batch = PhotonBatch(
            particle_idx=self._out_particle[:n_photons].copy(),
            arrival_time=self._out_time[:n_photons].copy(),
            channel=self._out_channel[:n_photons].copy(),
            pulse_index=self._out_pulse[:n_photons].copy(),
            is_fret=self._out_is_fret[:n_photons].copy(),
            n_photons=n_photons,
        )

        # 5. Update statistics (use dynamic channel identity, not hardcoded)
        if n_photons > 0:
            self.total_donor_photons += int(np.sum(batch.channel == self.donor_channel))
            self.total_acceptor_photons += int(np.sum(batch.channel == self.acceptor_channel))
            self.total_fret_events += int(np.sum(batch.is_fret))
        self.total_bleached = int(np.sum(self.state.is_bleached))

        return batch

    def get_donor_lifetime_histogram(
        self, photon_batch: PhotonBatch,
        n_bins: int = 256, time_range: float = 25e-9,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build TCSPC histogram from donor photon arrival times."""
        bin_edges = np.linspace(0, time_range, n_bins + 1)
        donor_mask = photon_batch.channel == self.donor_channel
        donor_times = photon_batch.arrival_time[donor_mask]
        counts, _ = np.histogram(donor_times, bins=bin_edges)
        return counts, bin_edges
