"""
ED-MRDT v1.1 — Particle State (Structure of Arrays)
=====================================================

SoA layout for all particle data, optimized for numba JIT compilation.

Design sources:
  - ReaDDy 2 (Hoffmann et al., 2019): per-particle type + state tracking
  - PROJECT_MASTER.md §2: SoA runtime state specification
  - Ermak-McCammon BD: positions, forces, diffusion coefficients

Usage:
    from ed_mrdt.config import load_config
    from ed_mrdt.particles import ParticleState

    config = load_config("examples/egfr_egf.yaml")
    state = ParticleState(config)
    # state.positions  → float64[N, 2]
    # state.species_id → int32[N]
    # state.has_dye    → bool[N]
    # ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import SimulationConfig, ConfigError

__all__ = ["ParticleState"]


class ParticleState:
    """Structure-of-Arrays container for all particle data.
    
    Every array has length N (total particles). Arrays use dtypes
    that are directly compatible with numba @njit functions:
      - float64 for positions, forces
      - int32   for species/state/bond indices
      - int8    for CTMC photophysical state
      - bool    for flags (has_dye, is_bleached)
    
    All arrays are C-contiguous for cache efficiency.
    
    Attributes
    ----------
    N : int
        Total number of particles.
    positions : ndarray[N, 2], float64
        (x, y) positions in µm on the membrane.
    species_id : ndarray[N], int32
        Index into config.species for each particle.
    state_id : ndarray[N], int32
        Conformational state index (0 = first state of species).
    bond_partner : ndarray[N], int32
        Index of bonded partner particle (-1 = free).
    has_dye : ndarray[N], bool
        Whether particle carries a fluorophore.
    dye_type_id : ndarray[N], int32
        Index into config.fluorophores (-1 = no dye).
    ctmc_state : ndarray[N], int8
        CTMC photophysical state (S0=0, S1=1, T1=2, ...).
    is_bleached : ndarray[N], bool
        Whether dye is permanently bleached.
    is_alive : ndarray[N], bool
        Whether particle is active (False = consumed by reaction).
    
    Per-species bookkeeping
    -----------------------
    species_names : list[str]
        Species name at each species_id index.
    species_ranges : dict[str, tuple[int, int]]
        (start, end) indices for each species in the initial layout.
        NOTE: After reactions modify species, use masks instead.
    """

    def __init__(
        self,
        config: SimulationConfig,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize particle state from simulation configuration.
        
        Parameters
        ----------
        config : SimulationConfig
            Validated simulation configuration.
        rng : np.random.Generator, optional
            Random number generator. If None, uses config.random_seed.
        """
        if rng is None:
            rng = np.random.default_rng(config.random_seed)
        self._rng = rng
        self._config = config

        # ── Build species map ────────────────────────────────────────
        self.species_names: List[str] = [s.name for s in config.species]
        self._species_name_to_id: Dict[str, int] = {
            name: i for i, name in enumerate(self.species_names)
        }

        # ── Count total particles ────────────────────────────────────
        self.N: int = 0
        species_counts: List[Tuple[int, int]] = []  # (species_idx, count)
        for sp in config.species:
            count = config.initial_populations.get(sp.name, 0)
            species_counts.append((self._species_name_to_id[sp.name], count))
            self.N += count

        # Dormant pool: cytosolic reservoir particles (is_alive=False)
        # These represent the cytosolic compartment for membrane ↔ cytosol exchange.
        # Pattern: ReaDDy 2 (Hoffmann 2019) reservoir particles.
        self._dormant_pools: List[Tuple[int, int, int]] = []  # (species_idx, start, count)
        if hasattr(config, 'compartment_exchanges'):
            for exch in config.compartment_exchanges:
                if exch.cytosol_pool > 0:
                    sp_id = self._species_name_to_id.get(exch.species, -1)
                    if sp_id >= 0:
                        self._dormant_pools.append((sp_id, self.N, exch.cytosol_pool))
                        self.N += exch.cytosol_pool

        if self.N == 0:
            raise ConfigError(
                "Total particle count is 0. Check initial_populations in config."
            )

        # ── Allocate SoA arrays ──────────────────────────────────────
        self.positions = np.zeros((self.N, 2), dtype=np.float64)
        self.species_id = np.zeros(self.N, dtype=np.int32)
        self.state_id = np.zeros(self.N, dtype=np.int32)
        self.bond_partner = np.full(self.N, -1, dtype=np.int32)
        self.has_dye = np.zeros(self.N, dtype=np.bool_)
        self.dye_type_id = np.full(self.N, -1, dtype=np.int32)
        self.ctmc_state = np.zeros(self.N, dtype=np.int8)
        self.is_bleached = np.zeros(self.N, dtype=np.bool_)
        self.is_alive = np.ones(self.N, dtype=np.bool_)

        # ── Fill species_id and positions ────────────────────────────
        self.species_ranges: Dict[str, Tuple[int, int]] = {}
        offset = 0
        Lx, Ly = config.membrane.size

        for sp_idx, count in species_counts:
            sp_name = self.species_names[sp_idx]
            start = offset
            end = offset + count

            # Species ID
            self.species_id[start:end] = sp_idx

            # State ID = 0 (first conformational state)
            self.state_id[start:end] = 0

            # Random uniform placement on membrane
            self.positions[start:end, 0] = rng.uniform(0.0, Lx, size=count)
            self.positions[start:end, 1] = rng.uniform(0.0, Ly, size=count)

            self.species_ranges[sp_name] = (start, end)
            offset = end

        # ── Initialize dormant pool particles ─────────────────────────
        # Dormant particles are placed at random positions but set is_alive=False.
        # They become visible only when recruited by apply_membrane_recruitment.
        # NOTE: Dyes ARE assigned to dormant particles (see _assign_dyes below).
        # Genetically encoded biosensors (e.g. PHAkt-mTFP1/Venus) carry their
        # fluorophore regardless of subcellular compartment.  The photophysics
        # module correctly skips is_alive=False particles during scanning, so
        # dormant-with-dye particles don't emit until recruited.
        for sp_idx, start, count in self._dormant_pools:
            end = start + count
            self.species_id[start:end] = sp_idx
            self.state_id[start:end] = 0
            self.positions[start:end, 0] = rng.uniform(0.0, Lx, size=count)
            self.positions[start:end, 1] = rng.uniform(0.0, Ly, size=count)
            self.is_alive[start:end] = False  # dormant = cytosolic
            offset = end

        # ── Assign dyes (labeling) ───────────────────────────────────
        self._assign_dyes(config)

        # ── Build diffusion coefficient lookup (per species) ─────────
        self._D_array = np.zeros(len(config.species), dtype=np.float64)
        for i, sp in enumerate(config.species):
            self._D_array[i] = sp.diffusion_coefficient

    # ─── Dye assignment ──────────────────────────────────────────────

    def _assign_dyes(self, config: SimulationConfig) -> None:
        """Assign fluorophores to particles based on dye_attachments.

        Respects labeling_efficiency: a fraction of particles of each
        species will carry the dye. Uses Bernoulli trials per particle.

        IMPORTANT: Assigns dyes to ALL particles of the target species,
        including dormant (cytosolic) pool particles.  Genetically encoded
        biosensors carry their fluorophore in every subcellular compartment.
        The photophysics module skips is_alive=False particles during
        scanning, so dormant particles with dyes don't emit until recruited.

        Bug-fix 2026-03-04: Previously used self.species_ranges which only
        covered initial (active) population, leaving 96%+ of cytosol-pool
        particles unlabeled.  Now uses species_id matching to cover all.
        Ref: Diagnostic protocol B (fluorophore labeling audit).
        """
        # Build fluorophore name → index map
        fluoro_name_to_id = {f.name: i for i, f in enumerate(config.fluorophores)}

        for attachment in config.dye_attachments:
            sp_name = attachment.species
            fluoro_name = attachment.fluorophore
            eff = attachment.labeling_efficiency

            # Resolve species index
            sp_idx = self._species_name_to_id.get(sp_name, -1)
            if sp_idx < 0:
                # Species not in config — skip silently
                continue

            fluoro_idx = fluoro_name_to_id.get(fluoro_name)
            if fluoro_idx is None:
                raise ConfigError(
                    f"DyeAttachment references fluorophore '{fluoro_name}' "
                    f"not found in config.fluorophores"
                )

            # Find ALL particles of this species (active + dormant)
            all_indices = np.where(self.species_id == sp_idx)[0]
            n_particles = len(all_indices)

            if n_particles == 0:
                continue

            # Bernoulli labeling
            labeled_mask = self._rng.random(n_particles) < eff
            indices = all_indices[labeled_mask]

            self.has_dye[indices] = True
            self.dye_type_id[indices] = fluoro_idx
            # CTMC state 0 = ground state (S0)
            self.ctmc_state[indices] = 0

    # ─── Query methods ───────────────────────────────────────────────

    def get_positions_of(self, species_name: str) -> np.ndarray:
        """Return positions of all particles of a given species.
        
        Parameters
        ----------
        species_name : str
            Name of the species.
            
        Returns
        -------
        ndarray[M, 2], float64
            Positions of the M particles of that species.
        """
        sp_id = self._species_name_to_id.get(species_name)
        if sp_id is None:
            raise ConfigError(f"Species '{species_name}' not found")
        mask = self.species_id == sp_id
        return self.positions[mask]

    def get_alive_mask(self) -> np.ndarray:
        """Return boolean mask of alive (active) particles."""
        return self.is_alive

    def get_labeled_mask(self, fluorophore_name: Optional[str] = None) -> np.ndarray:
        """Return mask of particles carrying a specific (or any) dye.
        
        Parameters
        ----------
        fluorophore_name : str, optional
            If given, filter by specific fluorophore. If None, any dye.
        """
        if fluorophore_name is None:
            return self.has_dye.copy()
        fluoro_id = None
        for i, f in enumerate(self._config.fluorophores):
            if f.name == fluorophore_name:
                fluoro_id = i
                break
        if fluoro_id is None:
            raise ConfigError(f"Fluorophore '{fluorophore_name}' not found")
        return self.has_dye & (self.dye_type_id == fluoro_id)

    def get_free_mask(self) -> np.ndarray:
        """Return mask of unbound particles."""
        return self.bond_partner == -1

    def get_bound_mask(self) -> np.ndarray:
        """Return mask of bound particles."""
        return self.bond_partner >= 0

    def count_species(self, species_name: str) -> int:
        """Count alive particles of a given species."""
        sp_id = self._species_name_to_id.get(species_name)
        if sp_id is None:
            raise ConfigError(f"Species '{species_name}' not found")
        return int(np.sum((self.species_id == sp_id) & self.is_alive))

    def get_D(self, idx: int) -> float:
        """Get diffusion coefficient for particle idx (base, no field)."""
        return float(self._D_array[self.species_id[idx]])

    def get_D_array(self) -> np.ndarray:
        """Return per-particle diffusion coefficients.
        
        Returns
        -------
        ndarray[N], float64
            D in µm²/s for each particle, based on species.
        """
        return self._D_array[self.species_id]

    # ─── Numba-friendly raw arrays ───────────────────────────────────

    def get_numba_arrays(self) -> dict:
        """Return dict of raw numpy arrays for passing to @njit functions.
        
        All arrays are contiguous and use numba-compatible dtypes.
        This avoids passing the Python object into numba.
        """
        return {
            "positions": self.positions,
            "species_id": self.species_id,
            "state_id": self.state_id,
            "bond_partner": self.bond_partner,
            "has_dye": self.has_dye,
            "dye_type_id": self.dye_type_id,
            "ctmc_state": self.ctmc_state,
            "is_bleached": self.is_bleached,
            "is_alive": self.is_alive,
            "D_per_species": self._D_array,
        }

    # ─── Memory ──────────────────────────────────────────────────────

    @property
    def ram_bytes(self) -> int:
        """Estimated RAM usage of all SoA arrays in bytes."""
        total = 0
        total += self.positions.nbytes          # N * 2 * 8
        total += self.species_id.nbytes         # N * 4
        total += self.state_id.nbytes           # N * 4
        total += self.bond_partner.nbytes       # N * 4
        total += self.has_dye.nbytes            # N * 1
        total += self.dye_type_id.nbytes        # N * 4
        total += self.ctmc_state.nbytes         # N * 1
        total += self.is_bleached.nbytes        # N * 1
        total += self.is_alive.nbytes           # N * 1
        return total

    # ─── Repr ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        lines = [f"ParticleState(N={self.N})"]
        for sp_name, (start, end) in self.species_ranges.items():
            count = end - start
            n_labeled = int(np.sum(self.has_dye[start:end]))
            lines.append(f"  {sp_name}: {count} particles, {n_labeled} labeled")
        for sp_idx, start, count in self._dormant_pools:
            sp_name = self.species_names[sp_idx]
            lines.append(f"  {sp_name} (dormant cytosol): {count} particles")
        lines.append(f"  RAM: {self.ram_bytes / 1e6:.2f} MB")
        return "\n".join(lines)
