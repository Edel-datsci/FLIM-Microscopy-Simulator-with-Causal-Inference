"""
ED-MRDT v1.1 — Configuration System
====================================

Dataclasses for simulation configuration + YAML loader with validation.

Design sources:
  - ReaDDy 2 (Hoffmann et al., PLoS Comp Biol 2019): Species/Reaction/Potential pattern
  - SMIS (Bourgeois, Commun Biol 2023): Fluorophore photophysical state model
  - Photon-HDF5 / readPTU_FLIM: TCSPC/FLIM output conventions

Usage:
    config = load_config("examples/egfr_egf.yaml")
    config.validate()  # raises ConfigError if invalid
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

__all__ = [
    "Species",
    "Reaction",
    "PairPotential",
    "DiffusionField",
    "CatalyticReaction",
    "CompartmentExchange",
    "UnimolecularConversion",
    "FeedbackRule",
    "MembraneGeometry",
    "PhotoState",
    "FluorophoreType",
    "FRETpair",
    "DyeAttachment",
    "ObjectiveLens",
    "PSFmodel",
    "Detector",
    "LaserSource",
    "ScanParameters",
    "TCSPCsettings",
    "SimulationConfig",
    "load_config",
    "ConfigError",
]


# ═══════════════════════════════════════════════════════════════════════
# Exceptions
# ═══════════════════════════════════════════════════════════════════════

class ConfigError(Exception):
    """Raised when configuration is invalid."""
    pass


# ═══════════════════════════════════════════════════════════════════════
# CAPA 1: Reaction-Diffusion System (Pattern: ReaDDy)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Species:
    """Particle type definition.
    
    Pattern: ReaDDy system.add_species(name, diffusion_constant).
    Extension: conformational states with state-dependent D.
    """
    name: str
    diffusion_coefficient: float       # µm²/s
    radius: float                      # nm (contact detection)
    color: str = "#FFFFFF"
    states: List[str] = field(default_factory=lambda: ["default"])
    state_transitions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    D_per_state: Optional[Dict[str, float]] = None

    def get_D(self, state: str = "default") -> float:
        """Return diffusion coefficient for a given conformational state."""
        if self.D_per_state and state in self.D_per_state:
            return self.D_per_state[state]
        return self.diffusion_coefficient


@dataclass
class Reaction:
    """Bimolecular or unimolecular reaction.
    
    Pattern: ReaDDy 'myfusion: A +(radius) B -> C' with rate.
    
    Three modes for specifying binding rate (user provides ONE):
      Mode 1: kon directly in µm²/s (kon_mode='direct')
      Mode 2: Keq in µm² + koff (kon_mode='Keq')  → kon = Keq × koff
      Mode 3: f_bound fraction + koff + densities (kon_mode='f_bound')
              → Keq derived from Langmuir isotherm, then kon = Keq × koff
    
    References:
      Andrews & Bray (2004) Phys. Biol. 1:137 (Smoldyn)
      Erban & Chapman (2009) Phys. Biol. 6:046001 (λ-ρ model)
      Rice (1985) Diffusion Limited Reactions, Ch. 3 (2D Smoluchowski)
    """
    name: str
    reactants: List[str]
    products: List[str]
    kon: float = 0.0                   # µm²/s (2D macroscopic on-rate)
    koff: float = 0.0                  # s⁻¹ (dissociation)
    contact_radius: float = 5.0        # nm
    Keq: Optional[float] = None        # µm² (equilibrium constant, mode 2)
    f_bound: Optional[float] = None    # fraction bound at eq (mode 3)
    required_state: Optional[Dict[str, str]] = None
    product_state: Optional[Dict[str, str]] = None

    @property
    def is_bimolecular(self) -> bool:
        return len(self.reactants) == 2

    @property
    def is_reversible(self) -> bool:
        return self.koff > 0.0


@dataclass
class PairPotential:
    """Pair interaction potential.
    
    Pattern: ReaDDy system.potentials.add_harmonic_repulsion().
    """
    species_a: str
    species_b: str
    potential_type: str                # "wca", "lennard_jones", "harmonic_repulsion"
    parameters: Dict[str, float] = field(default_factory=dict)

    def get_cutoff(self) -> float:
        """Return interaction cutoff distance in nm."""
        if self.potential_type == "wca":
            sigma = self.parameters.get("sigma", 5.0)
            return sigma * 2.0**(1.0/6.0)  # WCA cutoff = 2^(1/6) * sigma
        elif self.potential_type == "lennard_jones":
            sigma = self.parameters.get("sigma", 5.0)
            return sigma * 2.5  # standard LJ cutoff
        elif self.potential_type == "harmonic_repulsion":
            return self.parameters.get("interaction_distance", 5.0)
        return 10.0  # default


@dataclass
class DiffusionField:
    """Spatially varying diffusion coefficient field.
    
    Models crowding and lipid domains as effective medium D(x).
    """
    type: str = "uniform"              # "uniform", "regions", "image"
    default_D_factor: float = 1.0
    regions: List[Dict[str, Any]] = field(default_factory=list)

    def get_D_factor(self, x: float, y: float) -> float:
        """Return local D multiplier at position (x, y) in µm."""
        if self.type == "uniform":
            return self.default_D_factor
        for region in self.regions:
            if region.get("shape") == "circle":
                cx, cy = region["center"]
                r = region["radius"]
                if (x - cx)**2 + (y - cy)**2 <= r**2:
                    return region.get("D_factor", self.default_D_factor)
            elif region.get("shape") == "rectangle":
                x0, y0 = region["origin"]
                w, h = region["size"]
                if x0 <= x <= x0 + w and y0 <= y <= y0 + h:
                    return region.get("D_factor", self.default_D_factor)
        return self.default_D_factor


@dataclass
class CatalyticReaction:
    """Enzymatic catalysis: E + S → E + P (enzyme unchanged, substrate converted).

    In particle-based BD, the enzyme and substrate must be within contact_radius
    for a catalytic event to occur. The probability per encounter per step is
    derived from k_cat and the effective detection area (Collins-Kimball pattern).

    References:
        Michaelis & Menten (1913) Biochem Z 49:333
        Andrews & Bray (2004) Phys Biol 1:137
    """
    name: str
    enzyme: str                        # species name of enzyme
    substrate: str                     # species name of substrate
    product: str                       # species name of product
    k_cat: float = 1.0                 # catalytic rate constant (s⁻¹)
    contact_radius: float = 5.0        # nm (enzyme-substrate detection)
    feedback_rule: Optional[str] = None  # name of FeedbackRule (if density-dependent)


@dataclass
class CompartmentExchange:
    """Membrane ↔ cytosol exchange via dormant particle pool.

    Models recruitment of cytosolic proteins to the membrane (birth)
    and detachment from membrane back to cytosol (death).
    Dormant particles (is_alive=False) represent the cytosolic reservoir.

    References:
        Saha et al. (2018) PNAS 115:E4547
        Luo et al. (2015) J Biol Chem 290:23467
    """
    name: str
    species: str                       # species name that exchanges
    k_on: float = 0.01                 # membrane recruitment rate (s⁻¹)
    k_off: float = 0.01               # membrane detachment rate (s⁻¹)
    cytosol_pool: int = 100            # number of dormant (cytosolic) particles
    max_recruits_per_step: int = 10    # cap per BD step for stability
    feedback_rule: Optional[str] = None  # name of FeedbackRule (inhibition by local density)
    secondary_feedback_rule: Optional[str] = None  # second feedback (e.g., PTEN self-activation)
    detachment_feedback_rule: Optional[str] = None  # feedback on k_off (e.g., PIP3 accelerates PTEN release)


@dataclass
class UnimolecularConversion:
    """Unimolecular species conversion: A → B with rate k.

    Used for conformational switching (e.g., PTEN T1 ↔ T2),
    spontaneous activation/deactivation, or any first-order process.

    When feedback_rule is specified, the conversion rate is modulated
    by local density: k_eff(x) = k_base * hill_factor(ρ(x)).
    This enables PIP3-dependent PTEN activation (Knoch et al. 2014).

    Reference:
        Fabian et al. (2023) Biophys J 122:1
        Knoch et al. (2014) PLOS Comput Biol — PTEN* → PTEN** activation
    """
    name: str
    from_species: str
    to_species: str
    rate: float = 0.1                  # s⁻¹
    feedback_rule: Optional[str] = None  # name of FeedbackRule (density-dependent rate)


@dataclass
class FeedbackRule:
    """Density-dependent rate modulation via Hill function.

    Computes local density of sensor_species on a coarse grid,
    then modulates the associated reaction rate:
      hill_inhibition:  k_eff = k_base / (1 + (ρ/K_half)^n_hill)
      hill_activation:  k_eff = k_base * [α + (1-α) * (ρ/K_half)^n / (1 + (ρ/K_half)^n)]

    where α = basal_fraction ∈ [0, 1].
    At ρ=0: hill_factor = α (not zero → prevents absorbing state).
    At ρ→∞: hill_factor → 1.0 (full activation).

    References:
        Hill AV (1910) J Physiol 40:iv–vii
        Fabian et al. (2014) Biophys J — PIP3/PTEN bistability
        Knoch et al. (2014) PLOS Comput Biol — PTEN activation model
    """
    name: str
    sensor_species: str                # species whose density modulates rate
    function_type: str = "hill_inhibition"  # "hill_inhibition" | "hill_activation"
    K_half: float = 100.0              # half-saturation density (particles/µm²)
    n_hill: float = 3.0                # Hill coefficient (2-4 for bistability)
    grid_resolution: float = 0.2       # µm (cell size for local density computation)
    basal_fraction: float = 0.0        # α: minimum Hill factor at ρ=0 (prevents absorbing state)


@dataclass
class MembraneGeometry:
    """2D simulation domain (membrane patch).

    Pattern: ReaDDy ReactionDiffusionSystem(box_size, periodic_bc).
    """
    size: Tuple[float, float] = (2.0, 2.0)       # µm
    periodic: Tuple[bool, bool] = (True, True)
    geometry_type: str = "planar"

    @property
    def area(self) -> float:
        """Membrane area in µm²."""
        return self.size[0] * self.size[1]


# ═══════════════════════════════════════════════════════════════════════
# CAPA 2: Molecular Photophysics (Pattern: SMIS)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PhotoState:
    """Individual photophysical state of a fluorophore."""
    name: str
    is_fluorescent: bool = False
    is_absorbing: bool = True
    is_terminal: bool = False
    quantum_yield: float = 1.0
    emission_wavelength: float = 0.0   # nm


@dataclass
class FluorophoreType:
    """Complete fluorophore definition with photophysical model.
    
    Pattern: SMIS transition matrices for thermally-induced and
    light-induced transformations (Bourgeois 2023).
    Extension: lifetime can depend on local environment.
    """
    name: str
    states: List[PhotoState] = field(default_factory=list)
    thermal_rates: Dict[str, Dict[str, float]] = field(default_factory=dict)
    photo_yields: Dict[str, Dict[str, float]] = field(default_factory=dict)
    absorption_cross_section: float = 0.0   # cm²
    excitation_wavelength: float = 0.0      # nm
    extinction_coeff: float = 0.0           # M⁻¹cm⁻¹
    tau_rad: float = 4.0e-9                 # s
    tau_nr_base: float = 1.0e-8             # s
    environment_sensitivity: str = "none"   # "none", "viscosity", "pH", "polarity"

    def get_state_index(self, state_name: str) -> int:
        """Map state name to integer index for SoA arrays."""
        for i, s in enumerate(self.states):
            if s.name == state_name:
                return i
        raise ConfigError(f"State '{state_name}' not found in fluorophore '{self.name}'")

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def fluorescent_states(self) -> List[int]:
        """Indices of fluorescent states."""
        return [i for i, s in enumerate(self.states) if s.is_fluorescent]

    @property
    def terminal_states(self) -> List[int]:
        """Indices of terminal (bleached) states."""
        return [i for i, s in enumerate(self.states) if s.is_terminal]


@dataclass
class FRETpair:
    """FRET donor-acceptor pair. Förster theory: E(r) = 1/(1+(r/R0)^6).

    The intra_complex_distance is the D-A separation within a bound
    complex (nm). This is a structural parameter set by the protein
    geometry, NOT the BD inter-particle distance (which diverges for
    the particle-absorption binding model used by the dynamics engine).
    Ref: Ogiso et al. (2002) Cell 110:775 (EGFR-EGF, PDB: 1NQL)
    """
    donor_type: str
    acceptor_type: str
    R0: float                          # nm (Förster distance)
    kappa2: float = 2.0/3.0            # orientation factor (isotropic)
    intra_complex_distance: float = 5.0  # nm (D-A distance in bound complex)
    fret_mode: str = "bond"                  # "bond" (existing) | "proximity"
    intramolecular: bool = False             # True: FRET within same particle (PH-Akt biosensor)
    # When intramolecular=True and fret_mode="bond": donor gets FRET when
    # bond_partner >= 0 (binding triggers conformational FRET), regardless
    # of whether the partner carries an acceptor. Models biosensors where
    # donor+acceptor are on the same construct and binding changes D-A distance.
    proximity_radius: float = 10.0           # nm, used only if fret_mode="proximity"
    proximity_E_high: float = 0.5            # E when sensor near target species
    proximity_E_low: float = 0.02            # E when sensor far from target
    proximity_target_species: str = ""       # species that attracts sensor (e.g., "PIP3")

    def efficiency(self, r_nm: float) -> float:
        """FRET efficiency at distance r (nm)."""
        if r_nm <= 0:
            return 1.0
        return 1.0 / (1.0 + (r_nm / self.R0)**6)


@dataclass
class DyeAttachment:
    """How a fluorophore is attached to a particle species."""
    species: str
    fluorophore: str
    labeling_efficiency: float = 1.0
    linker_length: float = 1.0         # nm
    position_on_species: str = "N-term"


# ═══════════════════════════════════════════════════════════════════════
# CAPA 3: Virtual Microscope (Pattern: SMIS + Becker&Hickl)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ObjectiveLens:
    """Microscope objective. Pattern: SMIS 'Objective and PSF parameters'."""
    NA: float = 1.4
    magnification: float = 100.0
    immersion: str = "oil"
    n_immersion: float = 1.515

    def airy_radius(self, wavelength_nm: float) -> float:
        """First Airy disk radius in nm: 0.61 * λ / NA."""
        return 0.61 * wavelength_nm / self.NA


@dataclass
class PSFmodel:
    """Point Spread Function model.
    
    Gibson-Lanni (rigorous) or Gaussian (fast fallback).
    """
    type: str = "gibson_lanni"
    emission_wavelength: float = 520.0  # nm
    sigma_xy: float = 0.0              # nm (auto-calculated if 0)
    sigma_z: float = 0.0               # nm

    def get_sigma_xy(self, NA: float) -> float:
        """PSF sigma in nm. If not set, approximate from diffraction limit."""
        if self.sigma_xy > 0:
            return self.sigma_xy
        # Gaussian approximation: σ ≈ 0.21 * λ / NA (Airy → Gaussian)
        return 0.21 * self.emission_wavelength / NA


@dataclass
class Detector:
    """TCSPC detector (SPAD/PMT). Pattern: Becker&Hickl TCSPC handbook."""
    type: str = "SPAD"
    dark_count_rate: float = 100.0     # Hz
    detection_efficiency: float = 0.2
    timing_jitter: float = 50e-12      # s (contributes to IRF)
    dead_time: float = 100e-9          # s
    afterpulsing_prob: float = 0.005


@dataclass
class LaserSource:
    """Pulsed laser source. Pattern: SMIS 'Setup Laser'."""
    wavelength: float = 488.0          # nm
    repetition_rate: float = 40e6      # Hz
    pulse_width: float = 100e-12       # s
    power: float = 1e-6               # W
    beam_profile: str = "gaussian"
    beam_waist: float = 250.0          # nm

    @property
    def period(self) -> float:
        """Laser pulse period in seconds."""
        return 1.0 / self.repetition_rate


@dataclass
class ScanParameters:
    """Confocal scan configuration. Pattern: readPTU_FLIM conventions."""
    mode: str = "confocal"
    pixel_size: float = 50.0           # nm
    pixels_x: int = 256
    pixels_y: int = 256
    dwell_time: float = 10e-6          # s
    n_frames: int = 1
    frame_interval: float = 1.0        # s
    pinhole_diameter: float = 1.0      # Airy units

    @property
    def fov_x(self) -> float:
        """Field of view in µm (x)."""
        return self.pixels_x * self.pixel_size / 1000.0

    @property
    def fov_y(self) -> float:
        """Field of view in µm (y)."""
        return self.pixels_y * self.pixel_size / 1000.0

    @property
    def frame_time(self) -> float:
        """Total time per frame in seconds."""
        return self.pixels_x * self.pixels_y * self.dwell_time


@dataclass
class TCSPCsettings:
    """TCSPC histogram settings. Pattern: readPTU_FLIM."""
    n_bins: int = 256
    time_range: float = 25e-9          # s

    @property
    def bin_width(self) -> float:
        """Time per TCSPC bin in seconds."""
        return self.time_range / self.n_bins


# ═══════════════════════════════════════════════════════════════════════
# CAPA 4: Complete Configuration (Top-level)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SimulationConfig:
    """Complete ED-MRDT simulation configuration.
    
    One YAML file = one complete reproducible experiment.
    The scientist edits the YAML, never the code.
    """
    # Metadata
    name: str = "Unnamed simulation"
    version: str = "1.1.0"

    # Domain
    membrane: MembraneGeometry = field(
        default_factory=lambda: MembraneGeometry(size=(2.0, 2.0)))
    diffusion_field: DiffusionField = field(default_factory=DiffusionField)

    # Biology (Pattern: ReaDDy)
    species: List[Species] = field(default_factory=list)
    reactions: List[Reaction] = field(default_factory=list)
    potentials: List[PairPotential] = field(default_factory=list)
    initial_populations: Dict[str, int] = field(default_factory=dict)
    catalytic_reactions: List[CatalyticReaction] = field(default_factory=list)
    compartment_exchanges: List[CompartmentExchange] = field(default_factory=list)
    unimolecular_conversions: List[UnimolecularConversion] = field(default_factory=list)
    feedback_rules: List[FeedbackRule] = field(default_factory=list)

    # Photophysics (Pattern: SMIS)
    fluorophores: List[FluorophoreType] = field(default_factory=list)
    fret_pairs: List[FRETpair] = field(default_factory=list)
    dye_attachments: List[DyeAttachment] = field(default_factory=list)

    # Virtual Microscope
    objective: ObjectiveLens = field(default_factory=ObjectiveLens)
    laser: LaserSource = field(default_factory=LaserSource)
    detector: Detector = field(default_factory=Detector)
    scan: ScanParameters = field(default_factory=ScanParameters)
    tcspc: TCSPCsettings = field(default_factory=TCSPCsettings)
    psf: PSFmodel = field(default_factory=PSFmodel)

    # Temporal
    simulation_time: float = 60.0      # s
    bd_timestep: float = 1e-5          # s
    random_seed: int = 42
    pss_mode: bool = True              # True=PSS (default, fast), False=PbP (off)
    n_threads: int = 8
    ram_limit_gb: float = 12.0
    displacement_cap_fraction: float = 0.5    # Safety: max displacement = fraction × min(σ)
    region_smoothing_width: float = 0.05      # µm (sigmoid width for D-field transitions)

    # ─── Derived lookups (built by validate()) ───────────────────────
    _species_map: Dict[str, int] = field(default_factory=dict, repr=False)
    _fluoro_map: Dict[str, int] = field(default_factory=dict, repr=False)

    # ─── Convenience lookups ─────────────────────────────────────────

    def species_by_name(self, name: str) -> Species:
        """Get species by name."""
        idx = self._species_map.get(name)
        if idx is None:
            raise ConfigError(f"Species '{name}' not found")
        return self.species[idx]

    def fluorophore_by_name(self, name: str) -> FluorophoreType:
        """Get fluorophore by name."""
        idx = self._fluoro_map.get(name)
        if idx is None:
            raise ConfigError(f"Fluorophore '{name}' not found")
        return self.fluorophores[idx]

    @property
    def total_particles(self) -> int:
        """Total initial particle count."""
        return sum(self.initial_populations.values())

    @property
    def n_species(self) -> int:
        return len(self.species)

    @property
    def max_potential_cutoff(self) -> float:
        """Maximum interaction cutoff across all potentials (nm)."""
        if not self.potentials:
            return 0.0
        return max(p.get_cutoff() for p in self.potentials)

    # ─── Validation ──────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """Validate configuration consistency. Returns list of warnings.
        Raises ConfigError for fatal errors."""
        warnings = []
        errors = []

        # Build lookup maps
        self._species_map = {s.name: i for i, s in enumerate(self.species)}
        self._fluoro_map = {f.name: i for i, f in enumerate(self.fluorophores)}

        # 1. Species names must be unique
        names = [s.name for s in self.species]
        if len(names) != len(set(names)):
            errors.append("Duplicate species names detected")

        # 2. Reaction references must exist
        for rxn in self.reactions:
            for r in rxn.reactants:
                if r not in self._species_map:
                    errors.append(f"Reaction '{rxn.name}': reactant '{r}' not in species list")
            for p in rxn.products:
                if p not in self._species_map:
                    errors.append(f"Reaction '{rxn.name}': product '{p}' not in species list")

        # 3. Potential references must exist
        for pot in self.potentials:
            if pot.species_a not in self._species_map:
                errors.append(f"Potential: species_a '{pot.species_a}' not in species list")
            if pot.species_b not in self._species_map:
                errors.append(f"Potential: species_b '{pot.species_b}' not in species list")

        # 4. Initial populations must reference existing species
        for sp_name in self.initial_populations:
            if sp_name not in self._species_map:
                errors.append(f"initial_populations: species '{sp_name}' not in species list")

        # 5. Dye attachments must reference existing species and fluorophores
        for att in self.dye_attachments:
            if att.species not in self._species_map:
                errors.append(f"DyeAttachment: species '{att.species}' not in species list")
            if att.fluorophore not in self._fluoro_map:
                errors.append(f"DyeAttachment: fluorophore '{att.fluorophore}' not in fluorophores")

        # 6. FRET pairs must reference existing fluorophores
        for fp in self.fret_pairs:
            if fp.donor_type not in self._fluoro_map:
                errors.append(f"FRETpair: donor '{fp.donor_type}' not in fluorophores")
            if fp.acceptor_type not in self._fluoro_map:
                errors.append(f"FRETpair: acceptor '{fp.acceptor_type}' not in fluorophores")

        # 7. Fluorophore states consistency
        for fluo in self.fluorophores:
            state_names = {s.name for s in fluo.states}
            for from_state, transitions in fluo.thermal_rates.items():
                if from_state not in state_names:
                    errors.append(
                        f"Fluorophore '{fluo.name}': thermal_rate from "
                        f"state '{from_state}' not in states")
                for to_state in transitions:
                    if to_state not in state_names:
                        errors.append(
                            f"Fluorophore '{fluo.name}': thermal_rate to "
                            f"state '{to_state}' not in states")

        # 8. TCSPC time range vs laser period
        if self.tcspc.time_range > self.laser.period * 1.01:
            errors.append(
                f"TCSPC time_range ({self.tcspc.time_range:.2e}s) > "
                f"laser period ({self.laser.period:.2e}s)")

        # 9. Scan FOV vs membrane size
        fov_x, fov_y = self.scan.fov_x, self.scan.fov_y
        mem_x, mem_y = self.membrane.size
        if fov_x > mem_x * 1.01 or fov_y > mem_y * 1.01:
            warnings.append(
                f"Scan FOV ({fov_x:.1f}×{fov_y:.1f} µm) exceeds "
                f"membrane ({mem_x:.1f}×{mem_y:.1f} µm)")

        # 10. Physical sanity checks
        for sp in self.species:
            if sp.diffusion_coefficient < 0:
                errors.append(f"Species '{sp.name}': negative D")
            if sp.radius <= 0:
                errors.append(f"Species '{sp.name}': non-positive radius")

        if self.bd_timestep <= 0:
            errors.append("bd_timestep must be positive")
        if self.simulation_time <= 0:
            errors.append("simulation_time must be positive")

        # 11. RAM estimate
        N = self.total_particles
        Px, Py = self.scan.pixels_x, self.scan.pixels_y
        Tb = self.tcspc.n_bins
        ram_bytes = (N * 50 +                     # particle SoA
                     N * 2 * 8 * 2 +              # cell lists
                     Px * Py * Tb * 4 +            # flim_stack
                     Px * Py * 4 +                 # intensity
                     N * 50)                       # ground truth
        ram_gb = ram_bytes / 1e9
        if ram_gb > self.ram_limit_gb:
            errors.append(
                f"Estimated RAM ({ram_gb:.1f} GB) exceeds limit ({self.ram_limit_gb} GB)")
        elif ram_gb > self.ram_limit_gb * 0.8:
            warnings.append(
                f"Estimated RAM ({ram_gb:.1f} GB) is close to limit ({self.ram_limit_gb} GB)")

        # 12. Catalytic reaction references
        for cat in self.catalytic_reactions:
            if cat.enzyme not in self._species_map:
                errors.append(f"CatalyticReaction '{cat.name}': enzyme '{cat.enzyme}' not in species")
            if cat.substrate not in self._species_map:
                errors.append(f"CatalyticReaction '{cat.name}': substrate '{cat.substrate}' not in species")
            if cat.product not in self._species_map:
                errors.append(f"CatalyticReaction '{cat.name}': product '{cat.product}' not in species")
            if cat.feedback_rule and cat.feedback_rule not in {fb.name for fb in self.feedback_rules}:
                errors.append(f"CatalyticReaction '{cat.name}': feedback_rule '{cat.feedback_rule}' not found")

        # 13. Compartment exchange references
        fb_names = {fb.name for fb in self.feedback_rules}
        for exch in self.compartment_exchanges:
            if exch.species not in self._species_map:
                errors.append(f"CompartmentExchange '{exch.name}': species '{exch.species}' not in species")
            if exch.feedback_rule and exch.feedback_rule not in fb_names:
                errors.append(f"CompartmentExchange '{exch.name}': feedback_rule '{exch.feedback_rule}' not found")
            if exch.secondary_feedback_rule and exch.secondary_feedback_rule not in fb_names:
                errors.append(f"CompartmentExchange '{exch.name}': secondary_feedback_rule '{exch.secondary_feedback_rule}' not found")
            if exch.detachment_feedback_rule and exch.detachment_feedback_rule not in fb_names:
                errors.append(f"CompartmentExchange '{exch.name}': detachment_feedback_rule '{exch.detachment_feedback_rule}' not found")

        # 14. Unimolecular conversion references
        for conv in self.unimolecular_conversions:
            if conv.from_species not in self._species_map:
                errors.append(f"UnimolecularConversion '{conv.name}': from '{conv.from_species}' not in species")
            if conv.to_species not in self._species_map:
                errors.append(f"UnimolecularConversion '{conv.name}': to '{conv.to_species}' not in species")
            if conv.feedback_rule and conv.feedback_rule not in fb_names:
                errors.append(f"UnimolecularConversion '{conv.name}': feedback_rule '{conv.feedback_rule}' not found")

        # 15. Feedback rule references
        for fb in self.feedback_rules:
            if fb.sensor_species not in self._species_map:
                errors.append(f"FeedbackRule '{fb.name}': sensor_species '{fb.sensor_species}' not in species")
            if fb.function_type not in ("hill_inhibition", "hill_activation"):
                errors.append(f"FeedbackRule '{fb.name}': function_type must be 'hill_inhibition' or 'hill_activation'")
            if not (0.0 <= fb.basal_fraction <= 1.0):
                errors.append(f"FeedbackRule '{fb.name}': basal_fraction must be in [0, 1], got {fb.basal_fraction}")

        # 16. FRET pair proximity mode references
        for fp in self.fret_pairs:
            if fp.fret_mode == "proximity":
                if not fp.proximity_target_species:
                    errors.append(f"FRETpair {fp.donor_type}/{fp.acceptor_type}: proximity mode requires proximity_target_species")
                elif fp.proximity_target_species not in self._species_map:
                    errors.append(f"FRETpair: proximity_target_species '{fp.proximity_target_species}' not in species")

        # 17. BD timestep stability vs potential length scales
        D_max = max((s.diffusion_coefficient for s in self.species), default=0.0)
        sigma_min = float('inf')
        for pot in self.potentials:
            if pot.potential_type in ("wca", "lennard_jones"):
                s_nm = pot.parameters.get("sigma", 5.0)
                sigma_min = min(sigma_min, s_nm)
        if sigma_min < float('inf') and D_max > 0:
            sigma_um = sigma_min / 1000.0
            dt_stable = sigma_um ** 2 / (4.0 * D_max)
            if self.bd_timestep > dt_stable:
                import warnings
                warnings.warn(
                    f"BD timestep ({self.bd_timestep:.1e}s) exceeds WCA stability "
                    f"limit σ²/(4D) = {dt_stable:.1e}s for σ_min={sigma_min:.1f}nm, "
                    f"D_max={D_max:.2f}µm²/s. Consider displacement_cap or harmonic potential.",
                    stacklevel=2,
                )

        # 18. Smoothing width bounds
        if not (0.001 <= self.region_smoothing_width <= 0.5):
            errors.append(
                f"region_smoothing_width ({self.region_smoothing_width}) must be "
                f"in [0.001, 0.5] µm"
            )

        if errors:
            raise ConfigError(
                f"Configuration has {len(errors)} error(s):\n" +
                "\n".join(f"  • {e}" for e in errors))

        return warnings


# ═══════════════════════════════════════════════════════════════════════
# YAML Loading
# ═══════════════════════════════════════════════════════════════════════

def _build_photo_state(d: Dict[str, Any]) -> PhotoState:
    """Build PhotoState from YAML dict."""
    return PhotoState(
        name=d["name"],
        is_fluorescent=d.get("is_fluorescent", False),
        is_absorbing=d.get("is_absorbing", True),
        is_terminal=d.get("is_terminal", False),
        quantum_yield=d.get("quantum_yield", 1.0),
        emission_wavelength=d.get("emission_wavelength", 0.0),
    )


def _build_fluorophore(d: Dict[str, Any]) -> FluorophoreType:
    """Build FluorophoreType from YAML dict."""
    states = [_build_photo_state(s) for s in d.get("states", [])]
    return FluorophoreType(
        name=d["name"],
        states=states,
        thermal_rates=d.get("thermal_rates", {}),
        photo_yields=d.get("photo_yields", {}),
        absorption_cross_section=d.get("absorption_cross_section", 0.0),
        excitation_wavelength=d.get("excitation_wavelength", 0.0),
        extinction_coeff=d.get("extinction_coeff", 0.0),
        tau_rad=d.get("tau_rad", 4.0e-9),
        tau_nr_base=d.get("tau_nr_base", 1.0e-8),
        environment_sensitivity=d.get("environment_sensitivity", "none"),
    )


def _build_species(d: Dict[str, Any]) -> Species:
    """Build Species from YAML dict."""
    return Species(
        name=d["name"],
        diffusion_coefficient=d["diffusion_coefficient"],
        radius=d["radius"],
        color=d.get("color", "#FFFFFF"),
        states=d.get("states", ["default"]),
        state_transitions=d.get("state_transitions", {}),
        D_per_state=d.get("D_per_state", None),
    )


def _build_reaction(d: Dict[str, Any]) -> Reaction:
    """Build Reaction from YAML dict.
    
    Supports three modes:
      Mode 1: kon given directly (µm²/s)
      Mode 2: Keq given (µm²) → kon = Keq × koff
      Mode 3: f_bound given (0-1) → Keq from Langmuir → kon = Keq × koff
    """
    kon = d.get("kon", 0.0)
    koff = d.get("koff", 0.0)
    Keq = d.get("Keq", None)
    f_bound = d.get("f_bound", None)
    
    # Mode 3: f_bound → Keq (Langmuir isotherm)
    # f = [L]/(Kd + [L])  →  Kd = [L]·(1-f)/f  →  Keq = 1/Kd
    # Note: requires ligand_density in the YAML reaction block
    if f_bound is not None and Keq is None and kon == 0.0:
        if not (0.0 < f_bound < 1.0):
            raise ConfigError(
                f"Reaction '{d['name']}': f_bound must be in (0,1), got {f_bound}")
        ligand_density = d.get("ligand_density", None)
        if ligand_density is None:
            raise ConfigError(
                f"Reaction '{d['name']}': f_bound mode requires 'ligand_density' "
                f"(particles/µm²) to compute Keq via Langmuir isotherm")
        Kd = ligand_density * (1.0 - f_bound) / f_bound  # particles/µm²
        Keq = 1.0 / Kd  # µm²
    
    # Mode 2: Keq → kon
    if Keq is not None and kon == 0.0:
        if koff <= 0.0:
            raise ConfigError(
                f"Reaction '{d['name']}': Keq mode requires koff > 0")
        kon = Keq * koff  # µm²/s
    
    return Reaction(
        name=d["name"],
        reactants=d["reactants"],
        products=d["products"],
        kon=kon,
        koff=koff,
        contact_radius=d.get("contact_radius", 5.0),
        Keq=Keq,
        f_bound=f_bound,
        required_state=d.get("required_state", None),
        product_state=d.get("product_state", None),
    )


def _build_potential(d: Dict[str, Any]) -> PairPotential:
    """Build PairPotential from YAML dict."""
    return PairPotential(
        species_a=d["species_a"],
        species_b=d["species_b"],
        potential_type=d["potential_type"],
        parameters=d.get("parameters", {}),
    )


def _build_catalytic_reaction(d: Dict[str, Any]) -> CatalyticReaction:
    return CatalyticReaction(
        name=d["name"],
        enzyme=d["enzyme"],
        substrate=d["substrate"],
        product=d["product"],
        k_cat=d.get("k_cat", 1.0),
        contact_radius=d.get("contact_radius", 5.0),
        feedback_rule=d.get("feedback_rule", None),
    )


def _build_compartment_exchange(d: Dict[str, Any]) -> CompartmentExchange:
    return CompartmentExchange(
        name=d["name"],
        species=d["species"],
        k_on=d.get("k_on", 0.01),
        k_off=d.get("k_off", 0.01),
        cytosol_pool=d.get("cytosol_pool", 100),
        max_recruits_per_step=d.get("max_recruits_per_step", 10),
        feedback_rule=d.get("feedback_rule", None),
        secondary_feedback_rule=d.get("secondary_feedback_rule", None),
        detachment_feedback_rule=d.get("detachment_feedback_rule", None),
    )


def _build_unimolecular_conversion(d: Dict[str, Any]) -> UnimolecularConversion:
    return UnimolecularConversion(
        name=d["name"],
        from_species=d["from_species"],
        to_species=d["to_species"],
        rate=d.get("rate", 0.1),
        feedback_rule=d.get("feedback_rule", None),
    )


def _build_feedback_rule(d: Dict[str, Any]) -> FeedbackRule:
    return FeedbackRule(
        name=d["name"],
        sensor_species=d["sensor_species"],
        function_type=d.get("function_type", "hill_inhibition"),
        K_half=d.get("K_half", 100.0),
        n_hill=d.get("n_hill", 3.0),
        grid_resolution=d.get("grid_resolution", 0.2),
        basal_fraction=d.get("basal_fraction", 0.0),
    )


def _build_membrane(d: Dict[str, Any]) -> MembraneGeometry:
    """Build MembraneGeometry from YAML dict."""
    size = d.get("size", [2.0, 2.0])
    periodic = d.get("periodic", [True, True])
    return MembraneGeometry(
        size=tuple(size),
        periodic=tuple(periodic),
        geometry_type=d.get("geometry_type", "planar"),
    )


def _build_scan(d: Dict[str, Any]) -> ScanParameters:
    """Build ScanParameters from YAML dict."""
    return ScanParameters(
        mode=d.get("mode", "confocal"),
        pixel_size=d.get("pixel_size", 50.0),
        pixels_x=d.get("pixels_x", 256),
        pixels_y=d.get("pixels_y", 256),
        dwell_time=d.get("dwell_time", 10e-6),
        n_frames=d.get("n_frames", 1),
        frame_interval=d.get("frame_interval", 1.0),
        pinhole_diameter=d.get("pinhole_diameter", 1.0),
    )


def load_config(path: Union[str, Path]) -> SimulationConfig:
    """Load and validate a SimulationConfig from a YAML file.
    
    Args:
        path: Path to YAML configuration file.
        
    Returns:
        Validated SimulationConfig instance.
        
    Raises:
        ConfigError: If configuration is invalid.
        FileNotFoundError: If YAML file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError(f"YAML root must be a dict, got {type(raw).__name__}")

    # Build sub-objects
    membrane = _build_membrane(raw.get("membrane", {}))

    df_raw = raw.get("diffusion_field", {})
    diffusion_field = DiffusionField(
        type=df_raw.get("type", "uniform"),
        default_D_factor=df_raw.get("default_D_factor", 1.0),
        regions=df_raw.get("regions", []),
    )

    species = [_build_species(s) for s in raw.get("species", [])]
    reactions = [_build_reaction(r) for r in raw.get("reactions", [])]
    potentials = [_build_potential(p) for p in raw.get("potentials", [])]
    catalytic_reactions = [_build_catalytic_reaction(r) for r in raw.get("catalytic_reactions", [])]
    compartment_exchanges = [_build_compartment_exchange(e) for e in raw.get("compartment_exchanges", [])]
    unimolecular_conversions = [_build_unimolecular_conversion(u) for u in raw.get("unimolecular_conversions", [])]
    feedback_rules = [_build_feedback_rule(fb) for fb in raw.get("feedback_rules", [])]
    fluorophores = [_build_fluorophore(f) for f in raw.get("fluorophores", [])]

    fret_pairs = [
        FRETpair(
            donor_type=fp["donor_type"],
            acceptor_type=fp["acceptor_type"],
            R0=fp["R0"],
            kappa2=fp.get("kappa2", 2.0/3.0),
            intra_complex_distance=fp.get("intra_complex_distance", 5.0),
            fret_mode=fp.get("fret_mode", "bond"),
            intramolecular=fp.get("intramolecular", False),
            proximity_radius=fp.get("proximity_radius", 10.0),
            proximity_E_high=fp.get("proximity_E_high", 0.5),
            proximity_E_low=fp.get("proximity_E_low", 0.02),
            proximity_target_species=fp.get("proximity_target_species", ""),
        )
        for fp in raw.get("fret_pairs", [])
    ]

    dye_attachments = [
        DyeAttachment(
            species=da["species"],
            fluorophore=da["fluorophore"],
            labeling_efficiency=da.get("labeling_efficiency", 1.0),
            linker_length=da.get("linker_length", 1.0),
            position_on_species=da.get("position_on_species", "N-term"),
        )
        for da in raw.get("dye_attachments", [])
    ]

    obj_raw = raw.get("objective", {})
    objective = ObjectiveLens(
        NA=obj_raw.get("NA", 1.4),
        magnification=obj_raw.get("magnification", 100.0),
        immersion=obj_raw.get("immersion", "oil"),
        n_immersion=obj_raw.get("n_immersion", 1.515),
    )

    las_raw = raw.get("laser", {})
    laser = LaserSource(
        wavelength=las_raw.get("wavelength", 488.0),
        repetition_rate=las_raw.get("repetition_rate", 40e6),
        pulse_width=las_raw.get("pulse_width", 100e-12),
        power=las_raw.get("power", 1e-6),
        beam_profile=las_raw.get("beam_profile", "gaussian"),
        beam_waist=las_raw.get("beam_waist", 250.0),
    )

    det_raw = raw.get("detector", {})
    detector = Detector(
        type=det_raw.get("type", "SPAD"),
        dark_count_rate=det_raw.get("dark_count_rate", 100.0),
        detection_efficiency=det_raw.get("detection_efficiency", 0.2),
        timing_jitter=det_raw.get("timing_jitter", 50e-12),
        dead_time=det_raw.get("dead_time", 100e-9),
        afterpulsing_prob=det_raw.get("afterpulsing_prob", 0.005),
    )

    scan = _build_scan(raw.get("scan", {}))

    tcspc_raw = raw.get("tcspc", {})
    tcspc = TCSPCsettings(
        n_bins=tcspc_raw.get("n_bins", 256),
        time_range=tcspc_raw.get("time_range", 25e-9),
    )

    psf_raw = raw.get("psf", {})
    psf = PSFmodel(
        type=psf_raw.get("type", "gibson_lanni"),
        emission_wavelength=psf_raw.get("emission_wavelength", 520.0),
        sigma_xy=psf_raw.get("sigma_xy", 0.0),
        sigma_z=psf_raw.get("sigma_z", 0.0),
    )

    config = SimulationConfig(
        name=raw.get("name", "Unnamed simulation"),
        version=raw.get("version", "1.1.0"),
        membrane=membrane,
        diffusion_field=diffusion_field,
        species=species,
        reactions=reactions,
        potentials=potentials,
        initial_populations=raw.get("initial_populations", {}),
        catalytic_reactions=catalytic_reactions,
        compartment_exchanges=compartment_exchanges,
        unimolecular_conversions=unimolecular_conversions,
        feedback_rules=feedback_rules,
        fluorophores=fluorophores,
        fret_pairs=fret_pairs,
        dye_attachments=dye_attachments,
        objective=objective,
        laser=laser,
        detector=detector,
        scan=scan,
        tcspc=tcspc,
        psf=psf,
        simulation_time=raw.get("simulation_time", 60.0),
        bd_timestep=raw.get("bd_timestep", 1e-5),
        random_seed=raw.get("random_seed", 42),
        pss_mode=raw.get("pss_mode", True),
        n_threads=raw.get("n_threads", 8),
        ram_limit_gb=raw.get("ram_limit_gb", 12.0),
    )

    # Validate
    warnings = config.validate()

    return config
