"""
ED-MRDT v1.1 — Particle Dynamics Engine
=========================================

Brownian Dynamics integrator with cell-list neighbor search,
contact reactions, and pair potentials.

Design sources:
  - Ermak & McCammon (1978): Overdamped BD step
  - Allen & Tildesley (1987): Cell lists for neighbor search
  - ReaDDy 2 (Hoffmann et al., 2019): Reaction-diffusion pattern
  - Weeks, Chandler, Andersen (1971): WCA potential

Components (built cumulatively, Pasos 1.3–1.6):
  1.3: CellList — spatial hashing for O(N) neighbor queries
  1.4: BD integrator — Ermak-McCammon step + periodic BC
  1.5: Contact reactions — binding/dissociation with species conversion
  1.6: Pair potentials — WCA exclusion, LJ attraction, force computation

Usage:
    from ed_mrdt.dynamics import CellList, bd_step, apply_reactions, compute_forces

All hot-path functions are numba-JIT compiled.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from numba import njit, prange, types
from numba.typed import List as NumbaList

from .config import SimulationConfig, ConfigError
from .particles import ParticleState

__all__ = [
    "CellList",
    "bd_step",
    "apply_binding_reactions",
    "apply_dissociation",
    "compute_forces_wca",
    "DynamicsEngine",
    "apply_catalytic_reaction",
    "apply_catalytic_with_feedback",
    "apply_membrane_recruitment",
    "apply_membrane_recruitment_with_feedback",
    "apply_membrane_recruitment_with_dual_feedback",
    "apply_membrane_detachment",
    "apply_membrane_detachment_with_feedback",
    "apply_species_conversion",
    "apply_species_conversion_with_feedback",
    "compute_density_field",
    "compute_forces_harmonic",
    "apply_diffusion_field_regions_smooth",
    "apply_catalytic_with_feedback_and_adaptation",
    "update_adaptation_field",
    "apply_membrane_recruitment_local",
]


# ═══════════════════════════════════════════════════════════════════════
# PASO 1.3: CELL LIST (Spatial Hashing)
# ═══════════════════════════════════════════════════════════════════════
# Pattern: Allen & Tildesley, "Computer Simulation of Liquids" (1987)
# Grid-based spatial partitioning for O(N) neighbor queries.
# Cell size >= cutoff distance → only check 3×3 neighborhood.
# ═══════════════════════════════════════════════════════════════════════

class CellList:
    """Grid-based spatial hash for fast neighbor queries.
    
    The membrane is divided into cells of size >= cutoff.
    Each particle is assigned to one cell. To find all neighbors
    within cutoff of particle i, only the 3×3 block of cells
    around i's cell needs to be checked.
    
    Parameters
    ----------
    Lx, Ly : float
        Domain size in µm.
    cutoff : float
        Interaction cutoff in µm.
    periodic : tuple[bool, bool]
        Periodic boundary conditions.
    """

    def __init__(
        self,
        Lx: float,
        Ly: float,
        cutoff: float,
        periodic: Tuple[bool, bool] = (True, True),
    ):
        if cutoff <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")

        self.Lx = Lx
        self.Ly = Ly
        self.cutoff = cutoff
        self.periodic_x, self.periodic_y = periodic

        # Number of cells in each dimension (at least 3 for periodic)
        # With the particle-centric pair search, more cells = fewer particles
        # per cell = fewer distance checks. Sweet spot: cell_size ≈ cutoff
        # (the natural choice). For cutoff=12nm on 10µm: ~800 cells, but
        # each particle only checks 9 cells, most empty → fast.
        self.nx = max(3, int(math.floor(Lx / cutoff)))
        self.ny = max(3, int(math.floor(Ly / cutoff)))

        # Actual cell size (≥ cutoff)
        self.cell_sx = Lx / self.nx
        self.cell_sy = Ly / self.ny

        # Storage: head-of-chain + linked list
        self.head = np.full(self.nx * self.ny, -1, dtype=np.int32)
        self.next_particle = np.empty(0, dtype=np.int32)

    def build(self, positions: np.ndarray, N: int) -> None:
        """Assign N particles to cells.
        
        Parameters
        ----------
        positions : ndarray[N, 2], float64
            Particle positions in µm.
        N : int
            Number of active particles.
        """
        if len(self.next_particle) < N:
            self.next_particle = np.empty(N, dtype=np.int32)

        _build_cell_list(
            positions, N,
            self.head, self.next_particle,
            self.nx, self.ny,
            self.cell_sx, self.cell_sy,
        )

    def get_neighbors_brute(
        self, positions: np.ndarray, N: int, cutoff: float
    ) -> List[List[int]]:
        """Brute-force O(N²) neighbor list for validation.
        
        Returns list of (i, j) pairs with i < j and dist < cutoff.
        """
        return _brute_force_pairs(
            positions, N, cutoff,
            self.Lx, self.Ly,
            self.periodic_x, self.periodic_y,
        )

    def get_neighbor_pairs(
        self, positions: np.ndarray, N: int
    ) -> np.ndarray:
        """Return all pairs (i, j) with i < j within cutoff.
        
        Uses cell list for O(N) performance.
        
        Returns
        -------
        ndarray[M, 2], int32
            Array of pair indices.
        """
        return _cell_list_pairs(
            positions, N,
            self.head, self.next_particle,
            self.nx, self.ny,
            self.cell_sx, self.cell_sy,
            self.Lx, self.Ly,
            self.periodic_x, self.periodic_y,
            self.cutoff,
        )


@njit(cache=True)
def _build_cell_list(
    positions, N, head, next_particle, nx, ny, cell_sx, cell_sy
):
    """Build linked-list cell structure. O(N)."""
    head[:] = -1
    for i in range(N):
        cx = int(positions[i, 0] / cell_sx)
        cy = int(positions[i, 1] / cell_sy)
        # Clamp to valid range
        if cx >= nx:
            cx = nx - 1
        if cy >= ny:
            cy = ny - 1
        if cx < 0:
            cx = 0
        if cy < 0:
            cy = 0
        cell_idx = cx * ny + cy
        next_particle[i] = head[cell_idx]
        head[cell_idx] = i


@njit(cache=True)
def _min_image_dist_sq(
    x1, y1, x2, y2, Lx, Ly, periodic_x, periodic_y
):
    """Minimum image distance squared with periodic BC."""
    dx = x1 - x2
    dy = y1 - y2
    if periodic_x:
        if dx > 0.5 * Lx:
            dx -= Lx
        elif dx < -0.5 * Lx:
            dx += Lx
    if periodic_y:
        if dy > 0.5 * Ly:
            dy -= Ly
        elif dy < -0.5 * Ly:
            dy += Ly
    return dx * dx + dy * dy, dx, dy


@njit(cache=True)
def _cell_list_pairs(
    positions, N, head, next_particle,
    nx, ny, cell_sx, cell_sy,
    Lx, Ly, periodic_x, periodic_y, cutoff
):
    """Find all pairs within cutoff using cell list. Returns Mx2 array.
    
    Particle-centric algorithm: for each particle i, check the 3x3 
    neighborhood of its cell. Only form pairs with j > i to avoid 
    double counting. This is O(N * avg_neighbors_per_cell) instead
    of O(n_cells * 9) which is wasteful when cells >> particles.
    """
    cutoff_sq = cutoff * cutoff
    max_pairs = min(N * 30, 500000)
    pairs = np.empty((max_pairs, 2), dtype=np.int32)
    n_pairs = 0

    for i in range(N):
        xi = positions[i, 0]
        yi = positions[i, 1]
        # Which cell is particle i in?
        cxi = int(xi / cell_sx)
        cyi = int(yi / cell_sy)
        if cxi >= nx: cxi = nx - 1
        if cyi >= ny: cyi = ny - 1
        if cxi < 0: cxi = 0
        if cyi < 0: cyi = 0

        # Check 3x3 neighborhood
        for dcx in range(-1, 2):
            for dcy in range(-1, 2):
                ncx = cxi + dcx
                ncy = cyi + dcy

                if periodic_x:
                    ncx = ncx % nx
                elif ncx < 0 or ncx >= nx:
                    continue
                if periodic_y:
                    ncy = ncy % ny
                elif ncy < 0 or ncy >= ny:
                    continue

                neighbor_cell = ncx * ny + ncy
                j = head[neighbor_cell]
                while j >= 0:
                    if j > i:
                        dsq, _, _ = _min_image_dist_sq(
                            xi, yi,
                            positions[j, 0], positions[j, 1],
                            Lx, Ly, periodic_x, periodic_y)
                        if dsq < cutoff_sq:
                            if n_pairs < max_pairs:
                                pairs[n_pairs, 0] = i
                                pairs[n_pairs, 1] = j
                                n_pairs += 1
                    j = next_particle[j]

    return pairs[:n_pairs]


def _brute_force_pairs(
    positions, N, cutoff, Lx, Ly, periodic_x, periodic_y
):
    """Brute force O(N²) pair finding — for validation only."""
    return _brute_force_pairs_numba(
        positions, N, cutoff, Lx, Ly, periodic_x, periodic_y
    )


@njit(cache=True)
def _brute_force_pairs_numba(
    positions, N, cutoff, Lx, Ly, periodic_x, periodic_y
):
    """Numba-accelerated brute force pairs."""
    cutoff_sq = cutoff * cutoff
    max_pairs = N * 20
    pairs = np.empty((max_pairs, 2), dtype=np.int32)
    n_pairs = 0

    for i in range(N):
        for j in range(i + 1, N):
            dsq, _, _ = _min_image_dist_sq(
                positions[i, 0], positions[i, 1],
                positions[j, 0], positions[j, 1],
                Lx, Ly, periodic_x, periodic_y)
            if dsq < cutoff_sq:
                if n_pairs < max_pairs:
                    pairs[n_pairs, 0] = i
                    pairs[n_pairs, 1] = j
                    n_pairs += 1

    return pairs[:n_pairs]


# ═══════════════════════════════════════════════════════════════════════
# PASO 1.4: BROWNIAN DYNAMICS INTEGRATOR
# ═══════════════════════════════════════════════════════════════════════
# Ermak & McCammon (1978): x += sqrt(2*D*dt)*ξ + (D/(kBT))*F*dt
# For overdamped Langevin in 2D, with kBT=1 in reduced units
# when forces are in kBT/length units.
# ═══════════════════════════════════════════════════════════════════════

@njit(cache=True, parallel=True)
def bd_step(
    positions: np.ndarray,       # [N, 2] float64
    D_per_particle: np.ndarray,  # [N] float64
    forces: np.ndarray,          # [N, 2] float64
    grad_D: np.ndarray,          # [N, 2] float64 — ∇D(x) per particle
    dt: float,
    Lx: float,
    Ly: float,
    periodic_x: bool,
    periodic_y: bool,
    N: int,
    noise: np.ndarray,           # [N, 2] float64, pre-generated N(0,1)
    is_alive: np.ndarray,        # [N] bool
    max_displacement: float,     # Safety cap: max |Δx| per step in µm
) -> None:
    """Ermak-McCammon BD step with ∇D correction and displacement cap.
    
    Full Ermak-McCammon equation for position-dependent D(x):
    
        x_i += ∇D_i · dt  +  √(2 D_i dt) · ξ_i  +  D_i · F_i · dt
               ──────────     ────────────────────     ──────────────
               Itô drift       diffusive noise         force drift
               correction
    
    The ∇D·dt term (spurious drift correction) is required for
    thermodynamic consistency when D varies in space. Without it,
    particles accumulate in low-D regions as P_eq ∝ 1/√D(x) instead
    of the correct Boltzmann distribution (Lau & Lubensky 2007).
    
    Displacement cap prevents numerical explosion from divergent
    forces (WCA 1/r¹²) at deep overlaps (Heyes & Melrose 1993).
    With grad_D=0 and max_displacement=∞, reduces to original step.
    
    Forces in kBT/µm, D in µm²/s, dt in s → displacement in µm.
    """
    for i in prange(N):
        if not is_alive[i]:
            continue

        Di = D_per_particle[i]
        sigma_noise = math.sqrt(2.0 * Di * dt)

        # ── Three terms of Ermak-McCammon ───────────────────────
        # 1. Spurious drift correction: ∇D · dt
        dx_total = grad_D[i, 0] * dt
        dy_total = grad_D[i, 1] * dt

        # 2. Diffusive noise: √(2D·dt) · ξ
        dx_total += sigma_noise * noise[i, 0]
        dy_total += sigma_noise * noise[i, 1]

        # 3. Force drift: D · F · dt
        dx_total += Di * forces[i, 0] * dt
        dy_total += Di * forces[i, 1] * dt

        # ── Displacement cap (safety net) ───────────────────────
        dr_sq = dx_total * dx_total + dy_total * dy_total
        if dr_sq > max_displacement * max_displacement:
            dr = math.sqrt(dr_sq)
            scale = max_displacement / dr
            dx_total *= scale
            dy_total *= scale

        # ── Update positions ────────────────────────────────────
        positions[i, 0] += dx_total
        positions[i, 1] += dy_total

        # ── Periodic boundary conditions ────────────────────────
        if periodic_x:
            positions[i, 0] = positions[i, 0] % Lx
        else:
            if positions[i, 0] < 0.0:
                positions[i, 0] = -positions[i, 0]
            elif positions[i, 0] >= Lx:
                positions[i, 0] = 2.0 * Lx - positions[i, 0]

        if periodic_y:
            positions[i, 1] = positions[i, 1] % Ly
        else:
            if positions[i, 1] < 0.0:
                positions[i, 1] = -positions[i, 1]
            elif positions[i, 1] >= Ly:
                positions[i, 1] = 2.0 * Ly - positions[i, 1]



@njit(cache=True)
def apply_diffusion_field_regions(
    D_per_particle: np.ndarray,  # [N] float64 (modified in place)
    D_base: np.ndarray,          # [n_species] float64
    species_id: np.ndarray,      # [N] int32
    positions: np.ndarray,       # [N, 2] float64
    N: int,
    # Region data (arrays for numba compatibility)
    region_cx: np.ndarray,       # [n_regions] float64
    region_cy: np.ndarray,       # [n_regions] float64
    region_r: np.ndarray,        # [n_regions] float64
    region_Dfactor: np.ndarray,  # [n_regions] float64
    n_regions: int,
) -> None:
    """Apply spatially varying D(x) from circular regions."""
    for i in range(N):
        base_D = D_base[species_id[i]]
        factor = 1.0
        x, y = positions[i, 0], positions[i, 1]
        for r in range(n_regions):
            dx = x - region_cx[r]
            dy = y - region_cy[r]
            if dx * dx + dy * dy <= region_r[r] * region_r[r]:
                factor = region_Dfactor[r]
                break  # first matching region
        D_per_particle[i] = base_D * factor




@njit(cache=True)
def apply_diffusion_field_regions_smooth(
    D_per_particle: np.ndarray,  # [N] float64 (modified in place)
    grad_D: np.ndarray,          # [N, 2] float64 (modified in place)
    D_base: np.ndarray,          # [n_species] float64
    species_id: np.ndarray,      # [N] int32
    positions: np.ndarray,       # [N, 2] float64
    N: int,
    region_cx: np.ndarray,       # [n_regions] float64
    region_cy: np.ndarray,       # [n_regions] float64
    region_r: np.ndarray,        # [n_regions] float64
    region_Dfactor: np.ndarray,  # [n_regions] float64
    n_regions: int,
    smoothing_width: float,      # µm (sigmoid transition width)
) -> None:
    """Apply smooth D(x) field with sigmoid transitions and compute ∇D(x).
    
    Sigmoid-smoothed circular regions prevent discontinuous ∇D at
    boundaries. The gradient ∇D is needed for the Ermak-McCammon
    drift correction in bd_step (Stratonovich convention).
    
    For circular region centered at (cx,cy) with radius r_c and factor f:
      sigmoid(d) = 1 / (1 + exp((d - r_c) / w))
      D_factor(d) = 1 + (f - 1) × sigmoid(d)
      ∇D = D_base × (f-1) × d(sigmoid)/dd × r̂
    
    With n_regions=0, D=D_base and ∇D=0 (no-op, backward compatible).
    
    References
    ----------
    Lau & Lubensky (2007) Phys Rev E 76:011123 — spurious drift theory.
    Volpe et al. (2010) Phys Rev Lett 104:170602 — experimental validation.
    """
    for i in range(N):
        x = positions[i, 0]
        y = positions[i, 1]
        base_D = D_base[species_id[i]]

        factor = 1.0
        grad_factor_x = 0.0
        grad_factor_y = 0.0

        for reg in range(n_regions):
            dx_c = x - region_cx[reg]
            dy_c = y - region_cy[reg]
            d_sq = dx_c * dx_c + dy_c * dy_c
            d = math.sqrt(d_sq) if d_sq > 1e-30 else 1e-15

            # Sigmoid: σ(d) = 1 / (1 + exp((d - r_c) / w))
            arg = (d - region_r[reg]) / smoothing_width

            if arg > 100.0:
                sigmoid_val = 0.0
                dsigmoid_dd = 0.0
            elif arg < -100.0:
                sigmoid_val = 1.0
                dsigmoid_dd = 0.0
            else:
                exp_arg = math.exp(arg)
                denom = 1.0 + exp_arg
                sigmoid_val = 1.0 / denom
                # dσ/dd = -exp(arg) / (1+exp(arg))² / w
                dsigmoid_dd = -exp_arg / (denom * denom * smoothing_width)

            local_factor = 1.0 + (region_Dfactor[reg] - 1.0) * sigmoid_val

            # Check if this region is "active" (sigmoid > 0.01 or < 0.99)
            if sigmoid_val > 0.01:
                factor = local_factor

                # ∇factor = (Dfactor - 1) × dσ/dd × r̂
                # r̂ = (dx_c, dy_c) / d
                inv_d = 1.0 / d
                grad_factor_x = (region_Dfactor[reg] - 1.0) * dsigmoid_dd * dx_c * inv_d
                grad_factor_y = (region_Dfactor[reg] - 1.0) * dsigmoid_dd * dy_c * inv_d
                break  # First matching region

        D_per_particle[i] = base_D * factor
        grad_D[i, 0] = base_D * grad_factor_x
        grad_D[i, 1] = base_D * grad_factor_y
# ═══════════════════════════════════════════════════════════════════════
# PASO 1.5: CONTACT REACTIONS
# ═══════════════════════════════════════════════════════════════════════
# Smoluchowski-Collins-Kimball: reaction when particles are within
# contact_radius. Probabilistic acceptance based on kon and dt.
# Dissociation: Poisson process with rate koff.
# ═══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def apply_binding_reactions(
    positions: np.ndarray,       # [N, 2]
    species_id: np.ndarray,      # [N] int32
    bond_partner: np.ndarray,    # [N] int32
    is_alive: np.ndarray,        # [N] bool
    N: int,
    # Reaction parameters
    reactant_a: int,             # species_id of reactant A
    reactant_b: int,             # species_id of reactant B
    product_id: int,             # species_id of product (assigned to A)
    contact_radius_um: float,    # contact radius in µm
    prob_bind: float,            # binding probability per encounter per step
    # Pair list from cell list
    pairs: np.ndarray,           # [M, 2] int32
    n_pairs: int,
    # Random numbers
    rng_uniform: np.ndarray,     # [M] float64, pre-generated U(0,1)
    Lx: float,
    Ly: float,
    periodic_x: bool,
    periodic_y: bool,
) -> int:
    """Attempt binding reactions for all close pairs.
    
    For each pair (i, j) within contact_radius:
      - If species match reactants and both are free: bind with prob_bind.
      - Binding: A becomes product species, B is deactivated (is_alive=False),
        bond_partner links A↔B.
    
    Returns number of new bonds formed.
    """
    n_bound = 0
    contact_sq = contact_radius_um * contact_radius_um

    for p in range(n_pairs):
        i = pairs[p, 0]
        j = pairs[p, 1]

        if not is_alive[i] or not is_alive[j]:
            continue
        if bond_partner[i] >= 0 or bond_partner[j] >= 0:
            continue

        sp_i = species_id[i]
        sp_j = species_id[j]

        # Check if this pair matches reactants (either order)
        match = False
        a_idx, b_idx = -1, -1
        if sp_i == reactant_a and sp_j == reactant_b:
            a_idx, b_idx = i, j
            match = True
        elif sp_i == reactant_b and sp_j == reactant_a:
            a_idx, b_idx = j, i
            match = True

        if not match:
            continue

        # Distance check
        dsq, _, _ = _min_image_dist_sq(
            positions[i, 0], positions[i, 1],
            positions[j, 0], positions[j, 1],
            Lx, Ly, periodic_x, periodic_y)

        if dsq > contact_sq:
            continue

        # Probabilistic acceptance
        if rng_uniform[p] < prob_bind:
            # Form bond: A becomes product, B is consumed
            species_id[a_idx] = product_id
            is_alive[b_idx] = False
            bond_partner[a_idx] = b_idx
            bond_partner[b_idx] = a_idx
            n_bound += 1

    return n_bound


@njit(cache=True)
def apply_dissociation(
    species_id: np.ndarray,      # [N] int32
    bond_partner: np.ndarray,    # [N] int32
    is_alive: np.ndarray,        # [N] bool
    positions: np.ndarray,       # [N, 2]
    N: int,
    product_id: int,             # species_id of the complex
    reactant_a: int,             # species_id back to A
    reactant_b: int,             # species_id back to B
    prob_dissoc: float,          # koff * dt
    rng_uniform: np.ndarray,     # [N] float64
) -> int:
    """Apply dissociation reactions for bound complexes.
    
    For each particle that is the product complex:
      - With probability prob_dissoc, break the bond.
      - Restore both partners to their original species.
    
    Returns number of dissociations.
    """
    n_dissoc = 0

    for i in range(N):
        if not is_alive[i]:
            continue
        if species_id[i] != product_id:
            continue
        if bond_partner[i] < 0:
            continue

        if rng_uniform[i] < prob_dissoc:
            j = bond_partner[i]
            # Restore species
            species_id[i] = reactant_a
            species_id[j] = reactant_b
            is_alive[j] = True
            # Place B at A's position (will diffuse apart)
            positions[j, 0] = positions[i, 0]
            positions[j, 1] = positions[i, 1]
            # Clear bonds
            bond_partner[i] = -1
            bond_partner[j] = -1
            n_dissoc += 1

    return n_dissoc


# ═══════════════════════════════════════════════════════════════════════
# PASO 1.6: PAIR POTENTIALS
# ═══════════════════════════════════════════════════════════════════════
# WCA (Weeks-Chandler-Andersen): purely repulsive, shifted LJ.
# LJ (Lennard-Jones): attraction + repulsion.
# Forces in units of kBT/µm.
# ═══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def compute_forces_wca(
    positions: np.ndarray,        # [N, 2]
    forces: np.ndarray,           # [N, 2] (accumulated, not zeroed)
    species_id: np.ndarray,       # [N]
    is_alive: np.ndarray,         # [N]
    pairs: np.ndarray,            # [M, 2]
    n_pairs: int,
    # Potential parameters per species pair (flattened)
    sigma_matrix: np.ndarray,     # [n_sp, n_sp] float64, in µm
    epsilon_matrix: np.ndarray,   # [n_sp, n_sp] float64, in kBT
    Lx: float,
    Ly: float,
    periodic_x: bool,
    periodic_y: bool,
) -> None:
    """Compute WCA (repulsive LJ) forces for all pairs.
    
    WCA potential: U(r) = 4ε[(σ/r)¹² - (σ/r)⁶] + ε  for r < 2^(1/6)σ
                   U(r) = 0                          for r ≥ 2^(1/6)σ
    
    Force: F(r) = -dU/dr * r̂ = 24ε/r * [2(σ/r)¹² - (σ/r)⁶] * r̂
    
    Units: σ in µm, ε in kBT → Force in kBT/µm.
    """
    for p in range(n_pairs):
        i = pairs[p, 0]
        j = pairs[p, 1]

        if not is_alive[i] or not is_alive[j]:
            continue

        sp_i = species_id[i]
        sp_j = species_id[j]

        sigma = sigma_matrix[sp_i, sp_j]
        epsilon = epsilon_matrix[sp_i, sp_j]

        if sigma <= 0.0 or epsilon <= 0.0:
            continue

        dsq, dx, dy = _min_image_dist_sq(
            positions[i, 0], positions[i, 1],
            positions[j, 0], positions[j, 1],
            Lx, Ly, periodic_x, periodic_y)

        # WCA cutoff: r_cut = 2^(1/6) * sigma
        r_cut_sq = sigma * sigma * 1.2599210498948732  # (2^(1/6))^2

        if dsq >= r_cut_sq or dsq < 1e-30:
            continue

        # r² based computation (avoid sqrt)
        inv_r2 = 1.0 / dsq
        s2_over_r2 = sigma * sigma * inv_r2
        s6 = s2_over_r2 * s2_over_r2 * s2_over_r2
        s12 = s6 * s6

        # Force magnitude / r: F/r = 24*eps*(2*s12 - s6) * inv_r2
        f_over_r = 24.0 * epsilon * (2.0 * s12 - s6) * inv_r2
        # Safety cap: prevent numerical explosion at deep overlaps.
        # At r>0.95σ (normal operation), f_over_r < 100ε/σ (no capping).
        # At r→0, f_over_r → ∞ (capped). Ref: Heyes & Melrose (1993).
        max_f_over_r = 2400.0 * epsilon / (sigma * sigma)
        if f_over_r > max_f_over_r:
            f_over_r = max_f_over_r

        fx = f_over_r * dx
        fy = f_over_r * dy

        # Newton's 3rd law
        forces[i, 0] += fx
        forces[i, 1] += fy
        forces[j, 0] -= fx
        forces[j, 1] -= fy




# ═══════════════════════════════════════════════════════════════════════
# PASO 1.6b: HARMONIC SOFT REPULSION (bounded force alternative to WCA)
# ═══════════════════════════════════════════════════════════════════════
# Pattern: ReaDDy 2 add_harmonic_repulsion(). Bounded force prevents
# numerical explosion for large timesteps.
# Ref: Schöneberg, Noé et al. (2014) BMC Biophysics; Hoffmann et al. (2019)
# ═══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def compute_forces_harmonic(
    positions: np.ndarray,        # [N, 2]
    forces: np.ndarray,           # [N, 2] (accumulated, not zeroed)
    species_id: np.ndarray,       # [N]
    is_alive: np.ndarray,         # [N]
    pairs: np.ndarray,            # [M, 2]
    n_pairs: int,
    sigma_matrix: np.ndarray,     # [n_sp, n_sp] float64, in µm
    k_matrix: np.ndarray,         # [n_sp, n_sp] float64, spring constant kBT/µm²
    Lx: float,
    Ly: float,
    periodic_x: bool,
    periodic_y: bool,
) -> None:
    """Soft harmonic repulsion: V(r) = ½k(σ-r)² for r < σ.
    
    Force: F(r) = k(σ-r) r̂   for r < σ
           F(r) = 0           for r ≥ σ
    
    Maximum force F_max = kσ is BOUNDED, preventing the divergence
    catastrophe of 1/r¹² WCA at small separations.
    
    Units: σ in µm, k in kBT/µm² → Force in kBT/µm.
    """
    for p in range(n_pairs):
        i = pairs[p, 0]
        j = pairs[p, 1]

        if not is_alive[i] or not is_alive[j]:
            continue

        sp_i = species_id[i]
        sp_j = species_id[j]

        sigma = sigma_matrix[sp_i, sp_j]
        k_spring = k_matrix[sp_i, sp_j]

        if sigma <= 0.0 or k_spring <= 0.0:
            continue

        dsq, dx, dy = _min_image_dist_sq(
            positions[i, 0], positions[i, 1],
            positions[j, 0], positions[j, 1],
            Lx, Ly, periodic_x, periodic_y)

        # Sigma squared — cutoff for harmonic (no interaction beyond σ)
        sigma_sq = sigma * sigma
        if dsq >= sigma_sq or dsq < 1e-30:
            continue

        r = math.sqrt(dsq)

        # F(r) = k(σ - r), direction = r̂ = (dx, dy)/r
        f_mag = k_spring * (sigma - r)
        f_over_r = f_mag / r

        fx = f_over_r * dx
        fy = f_over_r * dy

        # Newton's 3rd law
        forces[i, 0] += fx
        forces[i, 1] += fy
        forces[j, 0] -= fx
        forces[j, 1] -= fy
# ═══════════════════════════════════════════════════════════════════════
# PASO 1.7: CATALYTIC REACTIONS (E + S → E + P)
# ═══════════════════════════════════════════════════════════════════════
# Enzymatic catalysis: enzyme untouched, substrate species_id flipped.
# Pattern: Collins-Kimball contact reaction, but NO bond formed.
# Ref: Michaelis & Menten (1913); Andrews & Bray (2004) Phys Biol 1:137
# ═══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def apply_catalytic_reaction(
    positions: np.ndarray,       # [N, 2]
    species_id: np.ndarray,      # [N] int32
    is_alive: np.ndarray,        # [N] bool
    N: int,
    enzyme_id: int,
    substrate_id: int,
    product_id: int,
    contact_radius_um: float,
    prob_catalysis: float,
    pairs: np.ndarray,           # [M, 2] int32
    n_pairs: int,
    rng_uniform: np.ndarray,     # [M] float64
    Lx: float,
    Ly: float,
    periodic_x: bool,
    periodic_y: bool,
) -> int:
    """Enzymatic catalysis: E + S → E + P.

    For each pair (i, j) within contact_radius:
      - If species match enzyme+substrate: flip substrate → product.
      - Enzyme species_id UNCHANGED, no bond formed, no kill.
      - Conservation: total particles of (substrate + product) constant.

    Returns number of catalytic conversions.
    """
    n_converted = 0
    contact_sq = contact_radius_um * contact_radius_um

    for p in range(n_pairs):
        i = pairs[p, 0]
        j = pairs[p, 1]
        if not is_alive[i] or not is_alive[j]:
            continue

        # Check enzyme-substrate match (either order)
        e_idx, s_idx = -1, -1
        if species_id[i] == enzyme_id and species_id[j] == substrate_id:
            e_idx, s_idx = i, j
        elif species_id[j] == enzyme_id and species_id[i] == substrate_id:
            e_idx, s_idx = j, i
        else:
            continue

        # Distance check
        dsq, _, _ = _min_image_dist_sq(
            positions[i, 0], positions[i, 1],
            positions[j, 0], positions[j, 1],
            Lx, Ly, periodic_x, periodic_y)
        if dsq > contact_sq:
            continue

        # Stochastic acceptance
        if rng_uniform[p] < prob_catalysis:
            species_id[s_idx] = product_id
            n_converted += 1

    return n_converted


@njit(cache=True)
def apply_catalytic_with_feedback(
    positions: np.ndarray,
    species_id: np.ndarray,
    is_alive: np.ndarray,
    N: int,
    enzyme_id: int,
    substrate_id: int,
    product_id: int,
    contact_radius_um: float,
    prob_catalysis_base: float,
    pairs: np.ndarray,
    n_pairs: int,
    rng_uniform: np.ndarray,
    Lx: float,
    Ly: float,
    periodic_x: bool,
    periodic_y: bool,
    density_field: np.ndarray,    # [nx_grid, ny_grid] float64
    nx_grid: int,
    ny_grid: int,
    cell_sx: float,
    cell_sy: float,
    K_half: float,
    n_hill: float,
    is_inhibition: bool,
    basal_fraction: float = 0.0,
) -> int:
    """Catalytic reaction with density-dependent rate (Hill feedback).

    k_eff(x) = k_base * hill_factor(ρ(x))
    where hill_factor depends on function_type:
      inhibition: 1 / (1 + (ρ/K_half)^n)
      activation: α + (1-α) * (ρ/K_half)^n / (1 + (ρ/K_half)^n)
        where α = basal_fraction prevents absorbing state at ρ=0.

    Ref: Hill (1910); Fabian et al. (2014) Biophys J
         Knoch et al. (2014) PLOS Comput Biol — basal fraction
    """
    n_converted = 0
    contact_sq = contact_radius_um * contact_radius_um

    for p in range(n_pairs):
        i = pairs[p, 0]
        j = pairs[p, 1]
        if not is_alive[i] or not is_alive[j]:
            continue

        e_idx, s_idx = -1, -1
        if species_id[i] == enzyme_id and species_id[j] == substrate_id:
            e_idx, s_idx = i, j
        elif species_id[j] == enzyme_id and species_id[i] == substrate_id:
            e_idx, s_idx = j, i
        else:
            continue

        dsq, _, _ = _min_image_dist_sq(
            positions[i, 0], positions[i, 1],
            positions[j, 0], positions[j, 1],
            Lx, Ly, periodic_x, periodic_y)
        if dsq > contact_sq:
            continue

        # Local density at substrate position
        sx = positions[s_idx, 0]
        sy = positions[s_idx, 1]
        gx = int(sx / cell_sx)
        gy = int(sy / cell_sy)
        if gx >= nx_grid:
            gx = nx_grid - 1
        if gy >= ny_grid:
            gy = ny_grid - 1
        if gx < 0:
            gx = 0
        if gy < 0:
            gy = 0
        rho = density_field[gx, gy]

        # Hill modulation with basal fraction
        if K_half > 0:
            ratio = rho / K_half
            ratio_n = ratio ** n_hill
            if is_inhibition:
                hill_factor = 1.0 / (1.0 + ratio_n)
            else:
                # activation with basal: α + (1-α) * h(ρ)
                h = ratio_n / (1.0 + ratio_n)
                hill_factor = basal_fraction + (1.0 - basal_fraction) * h
        else:
            hill_factor = 1.0

        prob_eff = prob_catalysis_base * hill_factor
        if rng_uniform[p] < prob_eff:
            species_id[s_idx] = product_id
            n_converted += 1

    return n_converted


@njit(cache=True)
def apply_catalytic_with_feedback_and_adaptation(
    positions: np.ndarray,
    species_id: np.ndarray,
    is_alive: np.ndarray,
    N: int,
    enzyme_id: int,
    substrate_id: int,
    product_id: int,
    contact_radius_um: float,
    prob_catalysis_base: float,
    pairs: np.ndarray,
    n_pairs: int,
    rng_uniform: np.ndarray,
    Lx: float,
    Ly: float,
    periodic_x: bool,
    periodic_y: bool,
    density_field: np.ndarray,
    nx_grid: int,
    ny_grid: int,
    cell_sx: float,
    cell_sy: float,
    K_half: float,
    n_hill: float,
    is_inhibition: bool,
    basal_fraction: float,
    # Adaptation field parameters
    adaptation_field: np.ndarray,   # [nx_grid, ny_grid] float64
    K_adapt: float,                 # adaptation threshold (density units)
    n_adapt: float,                 # adaptation Hill exponent
) -> int:
    """Catalytic reaction with Hill feedback AND adaptation modulation.

    k_eff(x) = k_base * hill_factor(rho(x)) * adapt_factor(w(x))
    where adapt_factor = 1 / (1 + (w/K_adapt)^n_adapt)

    The adaptation field w(x,y) tracks PIP3 density with delay tau_w
    and provides delayed negative feedback (FitzHugh-Nagumo recovery).
    """
    n_converted = 0
    contact_sq = contact_radius_um * contact_radius_um

    for p in range(n_pairs):
        i = pairs[p, 0]
        j = pairs[p, 1]
        if not is_alive[i] or not is_alive[j]:
            continue

        e_idx, s_idx = -1, -1
        if species_id[i] == enzyme_id and species_id[j] == substrate_id:
            e_idx, s_idx = i, j
        elif species_id[j] == enzyme_id and species_id[i] == substrate_id:
            e_idx, s_idx = j, i
        else:
            continue

        dsq, _, _ = _min_image_dist_sq(
            positions[i, 0], positions[i, 1],
            positions[j, 0], positions[j, 1],
            Lx, Ly, periodic_x, periodic_y)
        if dsq > contact_sq:
            continue

        # Local density at substrate position
        sx = positions[s_idx, 0]
        sy = positions[s_idx, 1]
        gx = int(sx / cell_sx)
        gy = int(sy / cell_sy)
        if gx >= nx_grid:
            gx = nx_grid - 1
        if gy >= ny_grid:
            gy = ny_grid - 1
        if gx < 0:
            gx = 0
        if gy < 0:
            gy = 0
        rho = density_field[gx, gy]

        # Hill modulation with basal fraction
        if K_half > 0:
            ratio = rho / K_half
            ratio_n = ratio ** n_hill
            if is_inhibition:
                hill_factor = 1.0 / (1.0 + ratio_n)
            else:
                h = ratio_n / (1.0 + ratio_n)
                hill_factor = basal_fraction + (1.0 - basal_fraction) * h
        else:
            hill_factor = 1.0

        # Adaptation modulation
        w_local = adaptation_field[gx, gy]
        if K_adapt > 0:
            w_ratio = (w_local / K_adapt) ** n_adapt
            adapt_factor = 1.0 / (1.0 + w_ratio)
        else:
            adapt_factor = 1.0

        prob_eff = prob_catalysis_base * hill_factor * adapt_factor
        if rng_uniform[p] < prob_eff:
            species_id[s_idx] = product_id
            n_converted += 1

    return n_converted


@njit(cache=True)
def update_adaptation_field(
    adaptation_field: np.ndarray,    # [nx, ny] float64 — modified in-place
    density_field: np.ndarray,       # [nx, ny] float64 — current PIP3 density
    nx: int,
    ny: int,
    dt: float,
    tau_w: float,
) -> None:
    """Update adaptation field: dw/dt = (rho_PIP3 - w) / tau_w.

    This is the delayed negative feedback (FitzHugh-Nagumo slow variable).
    w tracks PIP3 density with timescale tau_w (typically 60s).
    """
    inv_tau = dt / tau_w
    for ix in range(nx):
        for iy in range(ny):
            adaptation_field[ix, iy] += inv_tau * (
                density_field[ix, iy] - adaptation_field[ix, iy]
            )


@njit(cache=True)
def apply_membrane_recruitment_local(
    species_id: np.ndarray,
    is_alive: np.ndarray,
    positions: np.ndarray,
    N: int,
    target_species_id: int,
    prob_recruit_base: float,
    Lx: float,
    Ly: float,
    rng_accept: np.ndarray,
    rng_offsets: np.ndarray,         # [N, 2] — small random offsets
    max_recruits: int,
    # Density field for choosing recruitment location
    density_field: np.ndarray,       # [nx, ny] PIP3 density
    nx_grid: int,
    ny_grid: int,
    cell_sx: float,
    cell_sy: float,
    K_half: float,
    n_hill: float,
    is_inhibition: bool,
    basal_fraction: float,
    # CDF sampling arrays
    cdf: np.ndarray,                 # [nx*ny] cumulative density
    rng_location: np.ndarray,        # [N] uniform for location sampling
    recruit_sigma: float,            # Gaussian spread around chosen cell (µm)
) -> int:
    """Membrane recruitment BIASED toward high-density regions.

    Instead of random position, uses inverse-CDF sampling of the density
    field to place particles near regions with high PIP3 (for PI3K) or
    high PTEN (for PTEN recruitment). Then applies Hill feedback on rate.

    recruit_sigma adds Gaussian noise around the chosen cell center.
    """
    n_recruited = 0
    n_cells = nx_grid * ny_grid

    for i in range(N):
        if n_recruited >= max_recruits:
            break
        if is_alive[i]:
            continue
        if species_id[i] != target_species_id:
            continue

        # Choose location from density-weighted CDF
        r = rng_location[i]
        # Binary search in CDF
        lo, hi = 0, n_cells - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cdf[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        cell_idx = lo

        # Convert flat index to grid coordinates
        gx = cell_idx // ny_grid
        gy = cell_idx % ny_grid

        # Cell center + random offset
        cx = (gx + 0.5) * cell_sx + rng_offsets[i, 0] * recruit_sigma
        cy = (gy + 0.5) * cell_sy + rng_offsets[i, 1] * recruit_sigma

        # Periodic wrapping
        cx = cx % Lx
        cy = cy % Ly

        # Local density at candidate position for Hill feedback
        gx2 = int(cx / cell_sx)
        gy2 = int(cy / cell_sy)
        if gx2 >= nx_grid:
            gx2 = nx_grid - 1
        if gy2 >= ny_grid:
            gy2 = ny_grid - 1
        if gx2 < 0:
            gx2 = 0
        if gy2 < 0:
            gy2 = 0
        rho = density_field[gx2, gy2]

        if K_half > 0:
            ratio = rho / K_half
            ratio_n = ratio ** n_hill
            if is_inhibition:
                hill_factor = 1.0 / (1.0 + ratio_n)
            else:
                h = ratio_n / (1.0 + ratio_n)
                hill_factor = basal_fraction + (1.0 - basal_fraction) * h
        else:
            hill_factor = 1.0

        prob_eff = prob_recruit_base * hill_factor
        if rng_accept[i] < prob_eff:
            is_alive[i] = True
            positions[i, 0] = cx
            positions[i, 1] = cy
            n_recruited += 1

    return n_recruited


# ═══════════════════════════════════════════════════════════════════════
# PASO 1.8: COMPARTMENT EXCHANGE (membrane ↔ cytosol)
# ═══════════════════════════════════════════════════════════════════════
# Dormant particles (is_alive=False) represent cytosolic reservoir.
# Recruitment: activate dormant → random membrane position.
# Detachment: deactivate active → back to cytosol.
# Ref: Saha et al. (2018) PNAS 115:E4547
# ═══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def apply_membrane_recruitment(
    species_id: np.ndarray,
    is_alive: np.ndarray,
    positions: np.ndarray,
    N: int,
    target_species_id: int,
    prob_recruit: float,
    Lx: float,
    Ly: float,
    rng_accept: np.ndarray,       # [N] float64
    rng_positions: np.ndarray,    # [N, 2] float64  (U(0,1) for x,y)
    max_recruits: int,
) -> int:
    """Activate dormant particles = cytosol → membrane recruitment.

    Scans dormant (is_alive=False) particles of target species.
    Each has probability prob_recruit = k_on * dt of activating.
    Activated particles are placed at random membrane position.

    Returns number of recruited particles.
    """
    n_recruited = 0
    for i in range(N):
        if n_recruited >= max_recruits:
            break
        if is_alive[i]:
            continue
        if species_id[i] != target_species_id:
            continue
        if rng_accept[i] < prob_recruit:
            is_alive[i] = True
            positions[i, 0] = rng_positions[i, 0] * Lx
            positions[i, 1] = rng_positions[i, 1] * Ly
            n_recruited += 1
    return n_recruited


@njit(cache=True)
def apply_membrane_recruitment_with_feedback(
    species_id: np.ndarray,
    is_alive: np.ndarray,
    positions: np.ndarray,
    N: int,
    target_species_id: int,
    prob_recruit_base: float,
    Lx: float,
    Ly: float,
    rng_accept: np.ndarray,
    rng_positions: np.ndarray,
    max_recruits: int,
    density_field: np.ndarray,
    nx_grid: int,
    ny_grid: int,
    cell_sx: float,
    cell_sy: float,
    K_half: float,
    n_hill: float,
    is_inhibition: bool,
    basal_fraction: float = 0.0,
) -> int:
    """Membrane recruitment with density-dependent feedback.

    For inhibition (e.g., PIP3 inhibits PTEN binding):
      k_on_eff(x) = k_on_base / (1 + (ρ_PIP3(x)/K_half)^n)
    For activation with basal fraction α:
      k_on_eff(x) = k_on_base * [α + (1-α) * h(ρ)]

    Recruits to random position, THEN checks local density at that position.
    """
    n_recruited = 0
    for i in range(N):
        if n_recruited >= max_recruits:
            break
        if is_alive[i]:
            continue
        if species_id[i] != target_species_id:
            continue

        # Candidate position
        cx = rng_positions[i, 0] * Lx
        cy = rng_positions[i, 1] * Ly

        # Local density at candidate position
        gx = int(cx / cell_sx)
        gy = int(cy / cell_sy)
        if gx >= nx_grid:
            gx = nx_grid - 1
        if gy >= ny_grid:
            gy = ny_grid - 1
        if gx < 0:
            gx = 0
        if gy < 0:
            gy = 0
        rho = density_field[gx, gy]

        if K_half > 0:
            ratio = rho / K_half
            ratio_n = ratio ** n_hill
            if is_inhibition:
                hill_factor = 1.0 / (1.0 + ratio_n)
            else:
                h = ratio_n / (1.0 + ratio_n)
                hill_factor = basal_fraction + (1.0 - basal_fraction) * h
        else:
            hill_factor = 1.0

        prob_eff = prob_recruit_base * hill_factor
        if rng_accept[i] < prob_eff:
            is_alive[i] = True
            positions[i, 0] = cx
            positions[i, 1] = cy
            n_recruited += 1

    return n_recruited


@njit(cache=True)
def apply_membrane_recruitment_with_dual_feedback(
    species_id: np.ndarray,
    is_alive: np.ndarray,
    positions: np.ndarray,
    N: int,
    target_species_id: int,
    prob_recruit_base: float,
    Lx: float,
    Ly: float,
    rng_accept: np.ndarray,
    rng_positions: np.ndarray,
    max_recruits: int,
    # Primary feedback (e.g., PIP3 inhibition)
    density_field_1: np.ndarray,
    nx_grid_1: int,
    ny_grid_1: int,
    cell_sx_1: float,
    cell_sy_1: float,
    K_half_1: float,
    n_hill_1: float,
    is_inhibition_1: bool,
    basal_fraction_1: float,
    # Secondary feedback (e.g., PTEN self-activation)
    density_field_2: np.ndarray,
    nx_grid_2: int,
    ny_grid_2: int,
    cell_sx_2: float,
    cell_sy_2: float,
    K_half_2: float,
    n_hill_2: float,
    is_inhibition_2: bool,
    basal_fraction_2: float,
) -> int:
    """Membrane recruitment with TWO density-dependent feedbacks.

    Combines two feedback loops multiplicatively:
      k_on_eff(x) = k_on_base * hill_1(ρ_1(x)) * hill_2(ρ_2(x))

    Use case: PTEN recruitment inhibited by PIP3 (hill_1) AND
    promoted by local PTEN_T2 (hill_2) for PTEN bistability.

    Ref: Knoch et al. (2014) PLOS Comput Biol — dual PTEN feedback.
    """
    n_recruited = 0
    for i in range(N):
        if n_recruited >= max_recruits:
            break
        if is_alive[i]:
            continue
        if species_id[i] != target_species_id:
            continue

        # Candidate position
        cx = rng_positions[i, 0] * Lx
        cy = rng_positions[i, 1] * Ly

        # ── Primary feedback (density_field_1) ──
        gx1 = int(cx / cell_sx_1)
        gy1 = int(cy / cell_sy_1)
        if gx1 >= nx_grid_1:
            gx1 = nx_grid_1 - 1
        if gy1 >= ny_grid_1:
            gy1 = ny_grid_1 - 1
        if gx1 < 0:
            gx1 = 0
        if gy1 < 0:
            gy1 = 0
        rho1 = density_field_1[gx1, gy1]

        if K_half_1 > 0:
            ratio1 = rho1 / K_half_1
            ratio_n1 = ratio1 ** n_hill_1
            if is_inhibition_1:
                hill_1 = 1.0 / (1.0 + ratio_n1)
            else:
                h1 = ratio_n1 / (1.0 + ratio_n1)
                hill_1 = basal_fraction_1 + (1.0 - basal_fraction_1) * h1
        else:
            hill_1 = 1.0

        # ── Secondary feedback (density_field_2) ──
        gx2 = int(cx / cell_sx_2)
        gy2 = int(cy / cell_sy_2)
        if gx2 >= nx_grid_2:
            gx2 = nx_grid_2 - 1
        if gy2 >= ny_grid_2:
            gy2 = ny_grid_2 - 1
        if gx2 < 0:
            gx2 = 0
        if gy2 < 0:
            gy2 = 0
        rho2 = density_field_2[gx2, gy2]

        if K_half_2 > 0:
            ratio2 = rho2 / K_half_2
            ratio_n2 = ratio2 ** n_hill_2
            if is_inhibition_2:
                hill_2 = 1.0 / (1.0 + ratio_n2)
            else:
                h2 = ratio_n2 / (1.0 + ratio_n2)
                hill_2 = basal_fraction_2 + (1.0 - basal_fraction_2) * h2
        else:
            hill_2 = 1.0

        prob_eff = prob_recruit_base * hill_1 * hill_2
        if rng_accept[i] < prob_eff:
            is_alive[i] = True
            positions[i, 0] = cx
            positions[i, 1] = cy
            n_recruited += 1

    return n_recruited


@njit(cache=True)
def apply_membrane_detachment(
    species_id: np.ndarray,
    is_alive: np.ndarray,
    N: int,
    target_species_id: int,
    prob_detach: float,
    rng_uniform: np.ndarray,      # [N] float64
) -> int:
    """Deactivate active particles = membrane → cytosol detachment.

    First-order process: P(detach) = k_off * dt.
    Particle becomes dormant (is_alive=False) — invisible to BD, forces, photophysics.

    Returns number of detached particles.
    """
    n_detached = 0
    for i in range(N):
        if not is_alive[i]:
            continue
        if species_id[i] != target_species_id:
            continue
        if rng_uniform[i] < prob_detach:
            is_alive[i] = False
            n_detached += 1
    return n_detached


@njit(cache=True)
def apply_membrane_detachment_with_feedback(
    species_id: np.ndarray,
    is_alive: np.ndarray,
    positions: np.ndarray,
    N: int,
    target_species_id: int,
    prob_detach_base: float,
    rng_uniform: np.ndarray,
    density_field: np.ndarray,
    nx_grid: int,
    ny_grid: int,
    cell_sx: float,
    cell_sy: float,
    K_half: float,
    n_hill: float,
    is_inhibition: bool,
    basal_fraction: float = 0.0,
) -> int:
    """Membrane detachment with density-dependent rate modulation.

    k_off_eff(x) = k_off_base * hill_factor(ρ(x))

    For PIP3-dependent PTEN release (Matsuoka & Ueda 2018):
      activation: k_off increases with local PIP3 (substrate-induced release).
      At PIP3=0: k_off_eff = k_off_base × α (basal detachment)
      At PIP3→∞: k_off_eff = k_off_base × 1.0 (maximum detachment)

    Ref: Matsuoka & Ueda (2018) Nat Commun 9:4481, Fig 5c, Table 1.
    """
    n_detached = 0
    for i in range(N):
        if not is_alive[i]:
            continue
        if species_id[i] != target_species_id:
            continue

        # Local density at particle position
        gx = int(positions[i, 0] / cell_sx)
        gy = int(positions[i, 1] / cell_sy)
        if gx >= nx_grid:
            gx = nx_grid - 1
        if gy >= ny_grid:
            gy = ny_grid - 1
        if gx < 0:
            gx = 0
        if gy < 0:
            gy = 0
        rho = density_field[gx, gy]

        # Hill modulation with basal fraction
        if K_half > 0:
            ratio = rho / K_half
            ratio_n = ratio ** n_hill
            if is_inhibition:
                hill_factor = 1.0 / (1.0 + ratio_n)
            else:
                h = ratio_n / (1.0 + ratio_n)
                hill_factor = basal_fraction + (1.0 - basal_fraction) * h
        else:
            hill_factor = 1.0

        prob_eff = prob_detach_base * hill_factor
        if rng_uniform[i] < prob_eff:
            is_alive[i] = False
            n_detached += 1
    return n_detached


# ═══════════════════════════════════════════════════════════════════════
# PASO 1.9: UNIMOLECULAR SPECIES CONVERSION
# ═══════════════════════════════════════════════════════════════════════
# A → B with rate k. Used for conformational switching.
# Ref: Fabian et al. (2023) Biophys J 122:1
# ═══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def apply_species_conversion(
    species_id: np.ndarray,
    is_alive: np.ndarray,
    N: int,
    from_id: int,
    to_id: int,
    prob_convert: float,
    rng_uniform: np.ndarray,      # [N] float64
) -> int:
    """Unimolecular species conversion: A → B with probability k*dt.

    Scans alive particles with species_id == from_id.
    Each converts to to_id with probability prob_convert.

    Returns number of conversions.
    """
    n_converted = 0
    for i in range(N):
        if not is_alive[i]:
            continue
        if species_id[i] != from_id:
            continue
        if rng_uniform[i] < prob_convert:
            species_id[i] = to_id
            n_converted += 1
    return n_converted


@njit(cache=True)
def apply_species_conversion_with_feedback(
    species_id: np.ndarray,
    is_alive: np.ndarray,
    positions: np.ndarray,
    N: int,
    from_id: int,
    to_id: int,
    prob_convert_base: float,
    rng_uniform: np.ndarray,
    density_field: np.ndarray,
    nx_grid: int,
    ny_grid: int,
    cell_sx: float,
    cell_sy: float,
    K_half: float,
    n_hill: float,
    is_inhibition: bool,
    basal_fraction: float = 0.0,
) -> int:
    """Density-dependent unimolecular conversion: A → B with k_eff(x) = k_base * hill(ρ(x)).

    Used for PIP3-dependent PTEN activation (T1→T2):
    local PIP3 promotes conversion of PTEN_T1 (weak) to PTEN_T2 (strong).

    Ref: Knoch et al. (2014) PLOS Comput Biol — PTEN* → PTEN** model.
    """
    n_converted = 0
    for i in range(N):
        if not is_alive[i]:
            continue
        if species_id[i] != from_id:
            continue

        # Local density at particle position
        gx = int(positions[i, 0] / cell_sx)
        gy = int(positions[i, 1] / cell_sy)
        if gx >= nx_grid:
            gx = nx_grid - 1
        if gy >= ny_grid:
            gy = ny_grid - 1
        if gx < 0:
            gx = 0
        if gy < 0:
            gy = 0
        rho = density_field[gx, gy]

        # Hill modulation with basal fraction
        if K_half > 0:
            ratio = rho / K_half
            ratio_n = ratio ** n_hill
            if is_inhibition:
                hill_factor = 1.0 / (1.0 + ratio_n)
            else:
                h = ratio_n / (1.0 + ratio_n)
                hill_factor = basal_fraction + (1.0 - basal_fraction) * h
        else:
            hill_factor = 1.0

        prob_eff = prob_convert_base * hill_factor
        if rng_uniform[i] < prob_eff:
            species_id[i] = to_id
            n_converted += 1
    return n_converted


# ═══════════════════════════════════════════════════════════════════════
# PASO 1.10: DENSITY FIELD COMPUTATION (for feedback rules)
# ═══════════════════════════════════════════════════════════════════════
# Bins particles of a given species onto a coarse grid.
# Used by Hill feedback functions to compute local density ρ(x).
# ═══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def compute_density_field(
    positions: np.ndarray,
    species_id: np.ndarray,
    is_alive: np.ndarray,
    N: int,
    target_species_id: int,
    nx_grid: int,
    ny_grid: int,
    cell_sx: float,
    cell_sy: float,
    density_out: np.ndarray,      # [nx_grid, ny_grid] float64, zeroed by caller
) -> None:
    """Compute local density field for a species on a coarse grid.

    Each grid cell accumulates count of alive particles of the target species.
    Density = count / cell_area (particles/µm²).
    """
    cell_area = cell_sx * cell_sy
    inv_area = 1.0 / cell_area if cell_area > 0.0 else 0.0

    for i in range(N):
        if not is_alive[i]:
            continue
        if species_id[i] != target_species_id:
            continue
        gx = int(positions[i, 0] / cell_sx)
        gy = int(positions[i, 1] / cell_sy)
        if gx >= nx_grid:
            gx = nx_grid - 1
        if gy >= ny_grid:
            gy = ny_grid - 1
        if gx < 0:
            gx = 0
        if gy < 0:
            gy = 0
        density_out[gx, gy] += inv_area


# ═══════════════════════════════════════════════════════════════════════
# DYNAMICS ENGINE: Orchestrator for one timestep
# ═══════════════════════════════════════════════════════════════════════

class DynamicsEngine:
    """High-level orchestrator for BD + reactions + potentials.
    
    Manages the cell list, force computation, BD integration,
    and reaction processing for each timestep.
    
    Parameters
    ----------
    config : SimulationConfig
        Validated simulation configuration.
    state : ParticleState
        Initialized particle state.
    """

    def __init__(self, config: SimulationConfig, state: ParticleState):
        self.config = config
        self.state = state

        Lx, Ly = config.membrane.size
        self.Lx = Lx
        self.Ly = Ly
        self.periodic_x, self.periodic_y = config.membrane.periodic
        self.dt = config.bd_timestep

        # ── Determine cutoff for cell list ───────────────────────────
        # Maximum of: potential cutoffs (nm→µm), reaction contact radii (nm→µm),
        # and BD step length (σ_eff = max(σ_phys, s_BD) for each reaction).
        # MUST include both config.reactions AND config.catalytic_reactions.
        max_cutoff_nm = config.max_potential_cutoff
        for rxn in config.reactions:
            if rxn.contact_radius > max_cutoff_nm:
                max_cutoff_nm = rxn.contact_radius

        # Include catalytic_reactions contact radii (critical for enzyme-substrate pairs)
        if hasattr(config, 'catalytic_reactions'):
            for cat in config.catalytic_reactions:
                if cat.contact_radius > max_cutoff_nm:
                    max_cutoff_nm = cat.contact_radius

        # Also check BD step length for reaction detection
        max_cutoff_um = max_cutoff_nm / 1000.0
        for rxn in config.reactions:
            if rxn.is_bimolecular and rxn.kon > 0:
                ra_name, rb_name = rxn.reactants[0], rxn.reactants[1]
                ra_id = next(i for i, s in enumerate(config.species) if s.name == ra_name)
                rb_id = next(i for i, s in enumerate(config.species) if s.name == rb_name)
                D_a = config.species[ra_id].diffusion_coefficient
                D_b = config.species[rb_id].diffusion_coefficient
                s_bd = math.sqrt(4.0 * (D_a + D_b) * self.dt)
                sigma_phys = rxn.contact_radius / 1000.0
                sigma_eff = max(sigma_phys, s_bd)
                if sigma_eff > max_cutoff_um:
                    max_cutoff_um = sigma_eff

        # Also check BD step length for catalytic reactions
        if hasattr(config, 'catalytic_reactions'):
            for cat in config.catalytic_reactions:
                enz_name = cat.enzyme
                sub_name = cat.substrate
                enz_id = next(i for i, s in enumerate(config.species) if s.name == enz_name)
                sub_id = next(i for i, s in enumerate(config.species) if s.name == sub_name)
                D_e = config.species[enz_id].diffusion_coefficient
                D_s = config.species[sub_id].diffusion_coefficient
                s_bd = math.sqrt(4.0 * (D_e + D_s) * self.dt)
                sigma_phys = cat.contact_radius / 1000.0
                sigma_eff = max(sigma_phys, s_bd)
                if sigma_eff > max_cutoff_um:
                    max_cutoff_um = sigma_eff

        # Convert to µm and add safety margin
        self.cutoff_um = max(max_cutoff_um, max_cutoff_nm / 1000.0) * 1.1
        if self.cutoff_um < 0.001:
            self.cutoff_um = 0.05  # minimum 50 nm

        # ── Cell list ────────────────────────────────────────────────
        self.cell_list = CellList(
            Lx, Ly, self.cutoff_um,
            periodic=(self.periodic_x, self.periodic_y),
        )

        # ── Build sigma/epsilon/harmonic matrices for potentials ─────
        n_sp = config.n_species
        self.sigma_matrix = np.zeros((n_sp, n_sp), dtype=np.float64)
        self.epsilon_matrix = np.zeros((n_sp, n_sp), dtype=np.float64)
        self.harmonic_matrix = np.zeros((n_sp, n_sp), dtype=np.float64)

        species_name_to_id = {s.name: i for i, s in enumerate(config.species)}

        self._use_wca = False
        self._use_harmonic = False

        for pot in config.potentials:
            if pot.potential_type in ("wca", "lennard_jones"):
                self._use_wca = True
                a = species_name_to_id[pot.species_a]
                b = species_name_to_id[pot.species_b]
                sigma_nm = pot.parameters.get("sigma", 5.0)
                epsilon_kbt = pot.parameters.get("epsilon", 2.5)
                sigma_um = sigma_nm / 1000.0  # nm → µm
                self.sigma_matrix[a, b] = sigma_um
                self.sigma_matrix[b, a] = sigma_um
                self.epsilon_matrix[a, b] = epsilon_kbt
                self.epsilon_matrix[b, a] = epsilon_kbt
            elif pot.potential_type == "harmonic_repulsion":
                self._use_harmonic = True
                a = species_name_to_id[pot.species_a]
                b = species_name_to_id[pot.species_b]
                sigma_nm = pot.parameters.get("sigma", 5.0)
                k_spring = pot.parameters.get("spring_constant",
                                               pot.parameters.get("epsilon", 50.0))
                sigma_um = sigma_nm / 1000.0
                self.sigma_matrix[a, b] = sigma_um
                self.sigma_matrix[b, a] = sigma_um
                self.harmonic_matrix[a, b] = k_spring
                self.harmonic_matrix[b, a] = k_spring

        # ── Displacement cap (safety net) ────────────────────────────
        # Strategy depends on potential type:
        #   - WCA (1/r¹² divergent): cap at fraction × σ_min (strict)
        #   - Harmonic only (bounded F_max=kσ): cap at C × √(2·D_max·dt)
        #     to NEVER suppress normal Brownian motion.
        #   - Mixed: use the tighter WCA cap.
        #
        # Also ensure cap ≥ max(contact_radius) so catalytic encounters
        # are not geometrically suppressed.
        # Reference: Heyes & Melrose (1993) J Non-Newt Fluid Mech 46:1.
        #
        sigma_min = np.inf
        for i in range(n_sp):
            for j in range(n_sp):
                if self.sigma_matrix[i, j] > 0:
                    sigma_min = min(sigma_min, self.sigma_matrix[i, j])

        D_max = max(s.diffusion_coefficient for s in config.species)

        # Contact radius floor: cap must be ≥ any catalytic contact_radius
        contact_max_um = 0.0
        for cat in config.catalytic_reactions:
            contact_max_um = max(contact_max_um, cat.contact_radius / 1000.0)

        if self._use_wca:
            # WCA divergence: strict cap at fraction × σ_min
            cap_base = config.displacement_cap_fraction * sigma_min
            self.max_displacement = max(cap_base, contact_max_um)
        elif self._use_harmonic and not self._use_wca:
            # Harmonic: force bounded → cap at 5 × RMS Brownian step
            # This never suppresses normal diffusion but catches rare
            # numerical anomalies (e.g., runaway grad_D).
            rms_step = np.sqrt(2.0 * D_max * config.bd_timestep)
            self.max_displacement = max(5.0 * rms_step, contact_max_um,
                                        sigma_min if sigma_min < np.inf else 0.1)
        elif sigma_min < np.inf:
            self.max_displacement = config.displacement_cap_fraction * sigma_min
        else:
            self.max_displacement = 0.1  # µm default (no potentials)

        # ── Prepare reaction parameters ──────────────────────────────
        self._reactions = []
        for rxn in config.reactions:
            if rxn.is_bimolecular:
                ra_id = species_name_to_id[rxn.reactants[0]]
                rb_id = species_name_to_id[rxn.reactants[1]]
                # Product: first product species
                prod_id = species_name_to_id[rxn.products[0]]
                contact_um = rxn.contact_radius / 1000.0  # nm → µm

                # ── Rigorous binding probability with BD discretization ──
                #
                # Theory: Andrews & Bray (2004) Phys Biol 1:137
                #         Erban & Chapman (2009) Phys Biol 6:046001
                #         Schöneberg & Noé (2013) PLoS ONE 8:e74261
                #
                # ── Binding probability: Collins-Kimball 2D ──────────
                # References:
                #   Andrews & Bray (2004) Phys. Biol. 1:137 (Smoldyn)
                #   Erban & Chapman (2009) Phys. Biol. 6:046001 (λ-ρ model)
                #   Rice (1985) Diffusion Limited Reactions Ch.3 (2D Smoluchowski)
                #
                # Collins-Kimball decomposition in 2D:
                #   1/k_macro = 1/k_intrinsic + 1/k_D
                #   k_D = 2π(D_A+D_B) / ln(R/σ)  (2D Smoluchowski)
                #   R = min(1/√(π·c_ligand), L/2)  (Torney & McConnell 1983)
                #
                # Then: k_int = (1/k_macro - ln(R/σ)/(2π D))^{-1}
                # And:  P_bind = k_int · dt / (π · σ_eff²)
                #
                # σ_eff = max(σ_phys, s_BD) ensures detection even when
                # BD steps exceed physical contact radius.
                # P_bind is normalized by σ_eff² (detection area).
                #
                D_a = config.species[ra_id].diffusion_coefficient
                D_b = config.species[rb_id].diffusion_coefficient
                D_rel = D_a + D_b
                
                # BD step size for relative motion
                s_bd = math.sqrt(4.0 * D_rel * self.dt)
                
                # Effective detection radius: max(physical, BD step)
                sigma_eff = max(contact_um, s_bd)
                A_eff = math.pi * (sigma_eff ** 2)
                
                if A_eff > 0 and rxn.kon > 0:
                    kon_2D = rxn.kon  # µm²/s (user provides 2D rate)
                    
                    # Collins-Kimball: compute intrinsic rate
                    N_b = config.initial_populations.get(rxn.reactants[1], 0)
                    mem_area = config.membrane.size[0] * config.membrane.size[1]
                    c_b = N_b / mem_area if mem_area > 0 else 0
                    L_half = min(config.membrane.size) / 2.0
                    
                    if c_b > 0 and D_rel > 0 and contact_um > 0:
                        R_ck = min(1.0 / math.sqrt(math.pi * c_b), L_half)
                        ln_ratio = math.log(R_ck / contact_um)
                        
                        if ln_ratio > 0:
                            inv_k_int = 1.0 / kon_2D - ln_ratio / (2.0 * math.pi * D_rel)
                            
                            if inv_k_int > 0:
                                k_int = 1.0 / inv_k_int
                            else:
                                # Diffusion-limited: k_macro ≥ k_D → P = 1
                                k_int = 1e12
                            
                            prob_bind = min(1.0, k_int * self.dt / A_eff)
                        else:
                            # R ≤ σ edge case
                            prob_bind = min(1.0, kon_2D * self.dt / A_eff)
                    else:
                        # No partners or no diffusion
                        prob_bind = min(1.0, kon_2D * self.dt / A_eff)
                else:
                    prob_bind = 0.0

                self._reactions.append({
                    "type": "binding",
                    "ra": ra_id,
                    "rb": rb_id,
                    "product": prod_id,
                    "contact_um": sigma_eff,  # effective radius for detection
                    "prob_bind": prob_bind,
                    "koff": rxn.koff,
                    "prob_dissoc": min(1.0, rxn.koff * self.dt),
                })

        # ── Helper: build feedback info dict from FeedbackRule ──────
        def _build_fb_info(fb):
            """Create feedback info dict from a FeedbackRule object."""
            sensor_id = species_name_to_id[fb.sensor_species]
            nx_g = max(3, int(Lx / fb.grid_resolution))
            ny_g = max(3, int(Ly / fb.grid_resolution))
            return {
                'sensor_id': sensor_id,
                'nx_grid': nx_g,
                'ny_grid': ny_g,
                'cell_sx': Lx / nx_g,
                'cell_sy': Ly / ny_g,
                'K_half': fb.K_half,
                'n_hill': fb.n_hill,
                'is_inhibition': fb.function_type == 'hill_inhibition',
                'basal_fraction': fb.basal_fraction,
                'density_field': np.zeros((nx_g, ny_g), dtype=np.float64),
            }

        # ── Catalytic reactions ───────────────────────────────────────
        self._catalytic_reactions = []
        feedback_map = {fb.name: fb for fb in config.catalytic_reactions[0:0]}  # placeholder
        if hasattr(config, 'feedback_rules'):
            feedback_map = {fb.name: fb for fb in config.feedback_rules}

        if hasattr(config, 'catalytic_reactions'):
            for cat in config.catalytic_reactions:
                enz_id = species_name_to_id[cat.enzyme]
                sub_id = species_name_to_id[cat.substrate]
                prod_id = species_name_to_id[cat.product]

                D_e = config.species[enz_id].diffusion_coefficient
                D_s = config.species[sub_id].diffusion_coefficient
                D_rel = D_e + D_s
                contact_um = cat.contact_radius / 1000.0
                s_bd = math.sqrt(4.0 * D_rel * self.dt)
                sigma_eff = max(contact_um, s_bd)
                A_eff = math.pi * sigma_eff ** 2
                prob_cat = min(1.0, cat.k_cat * self.dt / max(A_eff, 1e-30))

                fb_info = None
                if cat.feedback_rule and cat.feedback_rule in feedback_map:
                    fb_info = _build_fb_info(feedback_map[cat.feedback_rule])

                # Mark PI3K catalysis for adaptation modulation
                is_pi3k = 'PI3K' in cat.enzyme
                self._catalytic_reactions.append({
                    'enzyme': enz_id,
                    'substrate': sub_id,
                    'product': prod_id,
                    'contact_um': sigma_eff,
                    'prob_cat': prob_cat,
                    'feedback': fb_info,
                    'use_adaptation': is_pi3k,  # Only PI3K gets adaptation
                })

        # ── Compartment exchanges ─────────────────────────────────────
        self._compartment_exchanges = []
        if hasattr(config, 'compartment_exchanges'):
            for exch in config.compartment_exchanges:
                sp_id = species_name_to_id[exch.species]
                prob_on = min(1.0, exch.k_on * self.dt)
                prob_off = min(1.0, exch.k_off * self.dt)

                fb_info = None
                if exch.feedback_rule and exch.feedback_rule in feedback_map:
                    fb_info = _build_fb_info(feedback_map[exch.feedback_rule])

                fb_info_2 = None
                if exch.secondary_feedback_rule and exch.secondary_feedback_rule in feedback_map:
                    fb_info_2 = _build_fb_info(feedback_map[exch.secondary_feedback_rule])

                fb_detach = None
                if exch.detachment_feedback_rule and exch.detachment_feedback_rule in feedback_map:
                    fb_detach = _build_fb_info(feedback_map[exch.detachment_feedback_rule])

                # Enable local (density-biased) recruitment for any species
                # that uses PIP3-dependent feedback: PI3K, PHAkt, etc.
                # This places recruited particles WHERE PIP3 is high,
                # not at random positions on the membrane.
                uses_pip3_feedback = (
                    exch.feedback_rule and 'pip3_activation' in exch.feedback_rule
                )
                self._compartment_exchanges.append({
                    'species_id': sp_id,
                    'prob_on': prob_on,
                    'prob_off': prob_off,
                    'max_recruits': exch.max_recruits_per_step,
                    'feedback': fb_info,
                    'feedback_2': fb_info_2,
                    'feedback_detach': fb_detach,
                    'use_local': uses_pip3_feedback,  # Any PIP3-dependent → local
                })

        # ── Unimolecular conversions ──────────────────────────────────
        self._unimolecular_conversions = []
        if hasattr(config, 'unimolecular_conversions'):
            for conv in config.unimolecular_conversions:
                from_id = species_name_to_id[conv.from_species]
                to_id = species_name_to_id[conv.to_species]
                prob = min(1.0, conv.rate * self.dt)

                fb_info = None
                if conv.feedback_rule and conv.feedback_rule in feedback_map:
                    fb_info = _build_fb_info(feedback_map[conv.feedback_rule])

                self._unimolecular_conversions.append({
                    'from_id': from_id,
                    'to_id': to_id,
                    'prob': prob,
                    'feedback': fb_info,
                })

        # ── Extended counters ─────────────────────────────────────────
        self.n_catalytic = 0
        self.n_recruited = 0
        self.n_detached = 0
        self.n_converted = 0

        # ── Adaptation field (delayed negative feedback on PI3K) ─────
        # Implements FitzHugh-Nagumo slow variable:
        #   dw/dt = (rho_PIP3 - w) / tau_w
        #   PI3K catalytic modulation: *= 1/(1 + (w/K_adapt)^n_adapt)
        # Parameters configurable via config.adaptation_* attributes.
        self._use_adaptation = getattr(config, 'use_adaptation', False)
        self._tau_w = getattr(config, 'adaptation_tau_w', 60.0)      # s
        self._K_adapt = getattr(config, 'adaptation_K_w', 10.0)      # /µm²
        self._n_adapt = getattr(config, 'adaptation_n_w', 2.0)

        # Grid for adaptation — use same grid as first catalytic feedback
        self._adapt_nx = max(3, int(Lx / 1.0))   # 1µm grid resolution
        self._adapt_ny = max(3, int(Ly / 1.0))
        self._adapt_sx = Lx / self._adapt_nx
        self._adapt_sy = Ly / self._adapt_ny
        self._adaptation_field = np.zeros(
            (self._adapt_nx, self._adapt_ny), dtype=np.float64
        )
        # PIP3 density field for adaptation tracking
        self._pip3_density_for_adapt = np.zeros(
            (self._adapt_nx, self._adapt_ny), dtype=np.float64
        )
        # Identify PIP3 species ID for density computation
        self._pip3_species_id = species_name_to_id.get('PIP3', -1)

        # ── Local recruitment support ───────────────────────────────
        # CDF array for density-weighted sampling.
        # Size must accommodate the FEEDBACK grid (which can be finer than
        # the adaptation grid). Scan all feedback dicts for max grid size.
        max_fb_cells = self._adapt_nx * self._adapt_ny  # fallback
        for cat_info in self._catalytic_reactions:
            for fb_key in ('fb', 'fb1', 'fb2', 'feedback'):
                fb_info = cat_info.get(fb_key)
                if fb_info and isinstance(fb_info, dict) and 'nx_grid' in fb_info:
                    fb_cells = fb_info['nx_grid'] * fb_info['ny_grid']
                    max_fb_cells = max(max_fb_cells, fb_cells)
        for exch_info in self._compartment_exchanges:
            for fb_key in ('feedback', 'feedback_2', 'feedback_detach'):
                fb_info = exch_info.get(fb_key)
                if fb_info and isinstance(fb_info, dict) and 'nx_grid' in fb_info:
                    fb_cells = fb_info['nx_grid'] * fb_info['ny_grid']
                    max_fb_cells = max(max_fb_cells, fb_cells)
        self._recruit_cdf = np.zeros(max_fb_cells, dtype=np.float64)
        self._use_local_recruitment = getattr(
            config, 'use_local_pi3k_recruitment', False
        )
        self._recruit_sigma = getattr(config, 'recruit_sigma', 0.5)  # µm

        # ── Diffusion field regions ──────────────────────────────────
        df = config.diffusion_field
        self._D_base = np.array(
            [s.diffusion_coefficient for s in config.species], dtype=np.float64
        )
        self._D_per_particle = np.empty(state.N, dtype=np.float64)

        # Extract circular regions for numba
        circles = [r for r in df.regions if r.get("shape") == "circle"]
        self._n_regions = len(circles)
        self._region_cx = np.array([r["center"][0] for r in circles], dtype=np.float64) if circles else np.empty(0, dtype=np.float64)
        self._region_cy = np.array([r["center"][1] for r in circles], dtype=np.float64) if circles else np.empty(0, dtype=np.float64)
        self._region_r = np.array([r["radius"] for r in circles], dtype=np.float64) if circles else np.empty(0, dtype=np.float64)
        self._region_Dfactor = np.array([r.get("D_factor", 1.0) for r in circles], dtype=np.float64) if circles else np.empty(0, dtype=np.float64)
        self._region_smoothing_width = config.region_smoothing_width

        # ── Pre-allocate gradient array for ∇D correction ────────────
        self._grad_D = np.zeros((state.N, 2), dtype=np.float64)

        # ── Pre-allocate force array ─────────────────────────────────
        self.forces = np.zeros((state.N, 2), dtype=np.float64)

        # ── RNG ──────────────────────────────────────────────────────
        self._rng = np.random.default_rng(config.random_seed)

        # ── Statistics ───────────────────────────────────────────────
        self.n_bindings = 0
        self.n_dissociations = 0

    def step(self) -> None:
        """Execute one BD timestep: forces → move → reactions."""
        N = self.state.N
        pos = self.state.positions
        sp_id = self.state.species_id
        alive = self.state.is_alive
        bp = self.state.bond_partner

        # 1. Update diffusion field (smooth sigmoid + ∇D computation)
        apply_diffusion_field_regions_smooth(
            self._D_per_particle, self._grad_D, self._D_base,
            sp_id, pos, N,
            self._region_cx, self._region_cy, self._region_r,
            self._region_Dfactor, self._n_regions,
            self._region_smoothing_width,
        )

        # 2. Build cell list
        self.cell_list.build(pos, N)

        # 3. Get neighbor pairs
        pairs = self.cell_list.get_neighbor_pairs(pos, N)
        n_pairs = pairs.shape[0]

        # 4. Compute forces — dispatch by potential type
        self.forces[:] = 0.0
        if self._use_wca:
            compute_forces_wca(
                pos, self.forces, sp_id, alive,
                pairs, n_pairs,
                self.sigma_matrix, self.epsilon_matrix,
                self.Lx, self.Ly,
                self.periodic_x, self.periodic_y,
            )
        if self._use_harmonic:
            compute_forces_harmonic(
                pos, self.forces, sp_id, alive,
                pairs, n_pairs,
                self.sigma_matrix, self.harmonic_matrix,
                self.Lx, self.Ly,
                self.periodic_x, self.periodic_y,
            )

        # 5. BD step (Ermak-McCammon with ∇D correction + displacement cap)
        noise = self._rng.standard_normal((N, 2))
        bd_step(
            pos, self._D_per_particle, self.forces,
            self._grad_D,
            self.dt, self.Lx, self.Ly,
            self.periodic_x, self.periodic_y,
            N, noise, alive,
            self.max_displacement,
        )

        # 6. Reactions
        for rxn in self._reactions:
            # Binding
            rng_bind = self._rng.random(n_pairs)
            nb = apply_binding_reactions(
                pos, sp_id, bp, alive, N,
                rxn["ra"], rxn["rb"], rxn["product"],
                rxn["contact_um"], rxn["prob_bind"],
                pairs, n_pairs, rng_bind,
                self.Lx, self.Ly,
                self.periodic_x, self.periodic_y,
            )
            self.n_bindings += nb

            # Dissociation
            rng_dissoc = self._rng.random(N)
            nd = apply_dissociation(
                sp_id, bp, alive, pos, N,
                rxn["product"], rxn["ra"], rxn["rb"],
                rxn["prob_dissoc"], rng_dissoc,
            )
            self.n_dissociations += nd

        # 7a. Update adaptation field (tracks PIP3 with delay tau_w)
        if self._use_adaptation and self._pip3_species_id >= 0:
            self._pip3_density_for_adapt[:] = 0.0
            compute_density_field(
                pos, sp_id, alive, N,
                self._pip3_species_id,
                self._adapt_nx, self._adapt_ny,
                self._adapt_sx, self._adapt_sy,
                self._pip3_density_for_adapt,
            )
            update_adaptation_field(
                self._adaptation_field,
                self._pip3_density_for_adapt,
                self._adapt_nx, self._adapt_ny,
                self.dt, self._tau_w,
            )

        # 7b. Catalytic reactions (with optional adaptation modulation)
        for cat in self._catalytic_reactions:
            rng_cat = self._rng.random(n_pairs)
            if cat['feedback'] is not None:
                fb = cat['feedback']
                fb['density_field'][:] = 0.0
                compute_density_field(
                    pos, sp_id, alive, N,
                    fb['sensor_id'],
                    fb['nx_grid'], fb['ny_grid'],
                    fb['cell_sx'], fb['cell_sy'],
                    fb['density_field'],
                )
                # Use adaptation-modulated catalysis for PI3K
                if self._use_adaptation and cat.get('use_adaptation', False):
                    nc = apply_catalytic_with_feedback_and_adaptation(
                        pos, sp_id, alive, N,
                        cat['enzyme'], cat['substrate'], cat['product'],
                        cat['contact_um'], cat['prob_cat'],
                        pairs, n_pairs, rng_cat,
                        self.Lx, self.Ly,
                        self.periodic_x, self.periodic_y,
                        fb['density_field'],
                        fb['nx_grid'], fb['ny_grid'],
                        fb['cell_sx'], fb['cell_sy'],
                        fb['K_half'], fb['n_hill'], fb['is_inhibition'],
                        fb['basal_fraction'],
                        self._adaptation_field,
                        self._K_adapt, self._n_adapt,
                    )
                else:
                    nc = apply_catalytic_with_feedback(
                        pos, sp_id, alive, N,
                        cat['enzyme'], cat['substrate'], cat['product'],
                        cat['contact_um'], cat['prob_cat'],
                        pairs, n_pairs, rng_cat,
                        self.Lx, self.Ly,
                        self.periodic_x, self.periodic_y,
                        fb['density_field'],
                        fb['nx_grid'], fb['ny_grid'],
                        fb['cell_sx'], fb['cell_sy'],
                        fb['K_half'], fb['n_hill'], fb['is_inhibition'],
                        fb['basal_fraction'],
                    )
            else:
                nc = apply_catalytic_reaction(
                    pos, sp_id, alive, N,
                    cat['enzyme'], cat['substrate'], cat['product'],
                    cat['contact_um'], cat['prob_cat'],
                    pairs, n_pairs, rng_cat,
                    self.Lx, self.Ly,
                    self.periodic_x, self.periodic_y,
                )
            self.n_catalytic += nc

        # 8. Compartment exchanges
        for exch in self._compartment_exchanges:
            rng_accept = self._rng.random(N)
            rng_pos = self._rng.random((N, 2))

            if exch['feedback'] is not None and exch['feedback_2'] is not None:
                # ── DUAL FEEDBACK: primary + secondary ──
                fb1 = exch['feedback']
                fb2 = exch['feedback_2']
                fb1['density_field'][:] = 0.0
                fb2['density_field'][:] = 0.0
                compute_density_field(
                    pos, sp_id, alive, N,
                    fb1['sensor_id'],
                    fb1['nx_grid'], fb1['ny_grid'],
                    fb1['cell_sx'], fb1['cell_sy'],
                    fb1['density_field'],
                )
                compute_density_field(
                    pos, sp_id, alive, N,
                    fb2['sensor_id'],
                    fb2['nx_grid'], fb2['ny_grid'],
                    fb2['cell_sx'], fb2['cell_sy'],
                    fb2['density_field'],
                )
                nr = apply_membrane_recruitment_with_dual_feedback(
                    sp_id, alive, pos, N,
                    exch['species_id'], exch['prob_on'],
                    self.Lx, self.Ly,
                    rng_accept, rng_pos, exch['max_recruits'],
                    # Primary feedback
                    fb1['density_field'],
                    fb1['nx_grid'], fb1['ny_grid'],
                    fb1['cell_sx'], fb1['cell_sy'],
                    fb1['K_half'], fb1['n_hill'], fb1['is_inhibition'],
                    fb1['basal_fraction'],
                    # Secondary feedback
                    fb2['density_field'],
                    fb2['nx_grid'], fb2['ny_grid'],
                    fb2['cell_sx'], fb2['cell_sy'],
                    fb2['K_half'], fb2['n_hill'], fb2['is_inhibition'],
                    fb2['basal_fraction'],
                )
            elif exch['feedback'] is not None:
                # ── SINGLE FEEDBACK ──
                fb = exch['feedback']
                fb['density_field'][:] = 0.0
                compute_density_field(
                    pos, sp_id, alive, N,
                    fb['sensor_id'],
                    fb['nx_grid'], fb['ny_grid'],
                    fb['cell_sx'], fb['cell_sy'],
                    fb['density_field'],
                )
                # Use LOCAL recruitment (biased toward PIP3-rich regions)
                if self._use_local_recruitment and exch.get('use_local', False):
                    # Build CDF from density field for density-weighted sampling
                    dens = fb['density_field']
                    nx_f, ny_f = fb['nx_grid'], fb['ny_grid']
                    cdf = self._recruit_cdf
                    total = 0.0
                    for ix in range(nx_f):
                        for iy in range(ny_f):
                            total += max(dens[ix, iy], 0.01)  # floor to avoid zeros
                            cdf[ix * ny_f + iy] = total
                    if total > 0:
                        for k in range(nx_f * ny_f):
                            cdf[k] /= total
                    else:
                        # Uniform fallback
                        for k in range(nx_f * ny_f):
                            cdf[k] = (k + 1.0) / (nx_f * ny_f)

                    rng_loc = self._rng.random(N)
                    rng_off = self._rng.standard_normal((N, 2))
                    nr = apply_membrane_recruitment_local(
                        sp_id, alive, pos, N,
                        exch['species_id'], exch['prob_on'],
                        self.Lx, self.Ly,
                        rng_accept, rng_off, exch['max_recruits'],
                        fb['density_field'],
                        fb['nx_grid'], fb['ny_grid'],
                        fb['cell_sx'], fb['cell_sy'],
                        fb['K_half'], fb['n_hill'], fb['is_inhibition'],
                        fb['basal_fraction'],
                        cdf, rng_loc, self._recruit_sigma,
                    )
                else:
                    nr = apply_membrane_recruitment_with_feedback(
                        sp_id, alive, pos, N,
                        exch['species_id'], exch['prob_on'],
                        self.Lx, self.Ly,
                        rng_accept, rng_pos, exch['max_recruits'],
                        fb['density_field'],
                        fb['nx_grid'], fb['ny_grid'],
                        fb['cell_sx'], fb['cell_sy'],
                        fb['K_half'], fb['n_hill'], fb['is_inhibition'],
                        fb['basal_fraction'],
                    )
            else:
                nr = apply_membrane_recruitment(
                    sp_id, alive, pos, N,
                    exch['species_id'], exch['prob_on'],
                    self.Lx, self.Ly,
                    rng_accept, rng_pos, exch['max_recruits'],
                )
            self.n_recruited += nr

            rng_detach = self._rng.random(N)
            if exch['feedback_detach'] is not None:
                fb_d = exch['feedback_detach']
                fb_d['density_field'][:] = 0.0
                compute_density_field(
                    pos, sp_id, alive, N,
                    fb_d['sensor_id'],
                    fb_d['nx_grid'], fb_d['ny_grid'],
                    fb_d['cell_sx'], fb_d['cell_sy'],
                    fb_d['density_field'],
                )
                nd = apply_membrane_detachment_with_feedback(
                    sp_id, alive, pos, N,
                    exch['species_id'], exch['prob_off'],
                    rng_detach,
                    fb_d['density_field'],
                    fb_d['nx_grid'], fb_d['ny_grid'],
                    fb_d['cell_sx'], fb_d['cell_sy'],
                    fb_d['K_half'], fb_d['n_hill'], fb_d['is_inhibition'],
                    fb_d['basal_fraction'],
                )
            else:
                nd = apply_membrane_detachment(
                    sp_id, alive, N,
                    exch['species_id'], exch['prob_off'],
                    rng_detach,
                )
            self.n_detached += nd

        # 9. Unimolecular conversions (with optional density-dependent feedback)
        for conv in self._unimolecular_conversions:
            rng_conv = self._rng.random(N)
            if conv['feedback'] is not None:
                fb = conv['feedback']
                fb['density_field'][:] = 0.0
                compute_density_field(
                    pos, sp_id, alive, N,
                    fb['sensor_id'],
                    fb['nx_grid'], fb['ny_grid'],
                    fb['cell_sx'], fb['cell_sy'],
                    fb['density_field'],
                )
                nc = apply_species_conversion_with_feedback(
                    sp_id, alive, pos, N,
                    conv['from_id'], conv['to_id'], conv['prob'],
                    rng_conv,
                    fb['density_field'],
                    fb['nx_grid'], fb['ny_grid'],
                    fb['cell_sx'], fb['cell_sy'],
                    fb['K_half'], fb['n_hill'], fb['is_inhibition'],
                    fb['basal_fraction'],
                )
            else:
                nc = apply_species_conversion(
                    sp_id, alive, N,
                    conv['from_id'], conv['to_id'], conv['prob'],
                    rng_conv,
                )
            self.n_converted += nc

    def run(self, n_steps: int, report_every: int = 0) -> None:
        """Run multiple BD steps.
        
        Parameters
        ----------
        n_steps : int
            Number of timesteps to execute.
        report_every : int
            Print progress every N steps (0 = no reporting).
        """
        for step_i in range(n_steps):
            self.step()
            if report_every > 0 and (step_i + 1) % report_every == 0:
                t = (step_i + 1) * self.dt
                n_prod = int(np.sum(
                    self.state.species_id == self._reactions[0]["product"]
                )) if self._reactions else 0
                print(
                    f"  t={t:.4f}s | bound={n_prod} "
                    f"| Σbind={self.n_bindings} Σdissoc={self.n_dissociations}"
                )
