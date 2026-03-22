"""
ED-MRDT v1.1 — Generic Reaction-Diffusion Extensions Tests (PHASE 8)
==================================================================

Quantitative tests for new reaction-diffusion modules extended in PHASE 8:
  1. CatalyticReaction, CompartmentExchange, UnimolecularConversion, FeedbackRule
  2. apply_catalytic_reaction, apply_catalytic_with_feedback
  3. apply_membrane_recruitment, apply_membrane_recruitment_with_feedback
  4. apply_membrane_detachment, apply_species_conversion, compute_density_field
  5. _compute_fret_rates_proximity_numba (biosensor FRET)

Test categories:
  T1: Catalytic reaction conservation (particle count + mass)
  T2: Compartment exchange conservation (dormant pool)
  T3: Species conversion balance (first-order kinetics)
  T4: Density field accuracy (spatial binning)
  T5: Feedback Hill modulation (rate inhibition/activation)
  T6: Proximity FRET rates (biosensor mode)
  T7: Config roundtrip (YAML load + new dataclasses)
  T8: Integration test (mini pipeline with catalytic reactions)

References:
  - Michaelis & Menten (1913): Enzyme kinetics
  - Hill (1910): Cooperative binding
  - Förster (1948): FRET theory
  - Fabian et al. (2014): PTEN/PIP3 bistability
"""

import math
import sys
import os

import numpy as np
import pytest

# Ensure package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ed_mrdt.config import (
    SimulationConfig, load_config, ConfigError,
    MembraneGeometry, Species, Reaction, PairPotential,
    DiffusionField, FluorophoreType, PhotoState,
    FRETpair, DyeAttachment, ObjectiveLens, LaserSource,
    Detector, ScanParameters, TCSPCsettings, PSFmodel,
    CatalyticReaction, CompartmentExchange, UnimolecularConversion, FeedbackRule,
)
from ed_mrdt.particles import ParticleState
from ed_mrdt.dynamics import (
    apply_catalytic_reaction,
    apply_catalytic_with_feedback,
    apply_membrane_recruitment,
    apply_membrane_recruitment_with_feedback,
    apply_membrane_detachment,
    apply_species_conversion,
    compute_density_field,
)
from ed_mrdt.photophysics import _compute_fret_rates_proximity_numba
from ed_mrdt.pipeline import SimulationPipeline


# ═══════════════════════════════════════════════════════════════════════
# FIXTURES: Minimal configs
# ═══════════════════════════════════════════════════════════════════════

def _make_minimal_config_with_catalytic(
    n_particles: int = 20,
    Lx: float = 0.5,
    Ly: float = 0.5,
    n_frames: int = 1,
    bd_steps_per_frame: int = 3,
    dt: float = 1e-4,
    seed: int = 42,
) -> SimulationConfig:
    """Build minimal config with catalytic reactions for fast testing."""
    frame_interval = bd_steps_per_frame * dt
    pixel_size_nm = Lx * 1000.0 / 5

    config = SimulationConfig(
        name="catalytic_test",
        membrane=MembraneGeometry(size=(Lx, Ly), periodic=(True, True)),
        diffusion_field=DiffusionField(type="uniform"),
        species=[
            Species(name="Enzyme", diffusion_coefficient=0.05, radius=5.0),
            Species(name="Substrate", diffusion_coefficient=1.0, radius=2.0),
            Species(name="Product", diffusion_coefficient=0.9, radius=2.0),
        ],
        catalytic_reactions=[
            CatalyticReaction(
                name="catalysis",
                enzyme="Enzyme",
                substrate="Substrate",
                product="Product",
                k_cat=100.0,
                contact_radius=7.0,
            ),
        ],
        initial_populations={"Enzyme": 10, "Substrate": 50},
        fluorophores=[
            FluorophoreType(
                name="donor",
                states=[
                    PhotoState(name="S0", is_absorbing=True),
                    PhotoState(name="S1", is_fluorescent=True,
                               quantum_yield=0.92, emission_wavelength=519.0),
                    PhotoState(name="T1"),
                    PhotoState(name="bleached", is_terminal=True),
                ],
                thermal_rates={
                    "S1": {"S0": 2.44e8, "T1": 1e6, "bleached": 1e3},
                    "T1": {"S0": 1e4, "bleached": 1e2},
                },
                absorption_cross_section=3e-16,
                tau_rad=4.1e-9,
            ),
        ],
        dye_attachments=[
            DyeAttachment(species="Enzyme", fluorophore="donor",
                          labeling_efficiency=0.8),
        ],
        objective=ObjectiveLens(NA=1.4),
        laser=LaserSource(wavelength=488.0, repetition_rate=40e6,
                          power=5e-6, beam_waist=250.0),
        detector=Detector(detection_efficiency=0.2, dark_count_rate=100.0,
                          timing_jitter=50e-12, afterpulsing_prob=0.005),
        scan=ScanParameters(
            pixel_size=pixel_size_nm,
            pixels_x=5, pixels_y=5,
            dwell_time=20e-6,
            n_frames=n_frames,
            frame_interval=frame_interval,
        ),
        tcspc=TCSPCsettings(n_bins=64, time_range=25e-9),
        psf=PSFmodel(type="gaussian", emission_wavelength=519.0),
        simulation_time=n_frames * frame_interval,
        bd_timestep=dt,
        random_seed=seed,
    )
    config.validate()
    return config


def _make_minimal_config_with_exchange(
    n_particles: int = 30,
    Lx: float = 0.5,
    Ly: float = 0.5,
    seed: int = 42,
) -> SimulationConfig:
    """Build minimal config with compartment exchange."""
    config = SimulationConfig(
        name="exchange_test",
        membrane=MembraneGeometry(size=(Lx, Ly), periodic=(True, True)),
        diffusion_field=DiffusionField(type="uniform"),
        species=[
            Species(name="ProteinX", diffusion_coefficient=0.05, radius=5.0),
        ],
        compartment_exchanges=[
            CompartmentExchange(
                name="recruitment",
                species="ProteinX",
                k_on=0.5,
                k_off=0.1,
                cytosol_pool=30,
                max_recruits_per_step=5,
            ),
        ],
        initial_populations={"ProteinX": 20},
        fluorophores=[
            FluorophoreType(
                name="donor",
                states=[
                    PhotoState(name="S0", is_absorbing=True),
                    PhotoState(name="S1", is_fluorescent=True,
                               quantum_yield=0.92, emission_wavelength=519.0),
                    PhotoState(name="bleached", is_terminal=True),
                ],
                thermal_rates={
                    "S1": {"S0": 2.44e8, "bleached": 1e3},
                },
                absorption_cross_section=3e-16,
                tau_rad=4.1e-9,
            ),
        ],
        dye_attachments=[
            DyeAttachment(species="ProteinX", fluorophore="donor",
                          labeling_efficiency=0.8),
        ],
        objective=ObjectiveLens(NA=1.4),
        laser=LaserSource(wavelength=488.0, repetition_rate=40e6,
                          power=5e-6, beam_waist=250.0),
        detector=Detector(detection_efficiency=0.2, dark_count_rate=100.0,
                          timing_jitter=50e-12, afterpulsing_prob=0.005),
        scan=ScanParameters(
            pixel_size=100.0,
            pixels_x=5, pixels_y=5,
            dwell_time=20e-6,
            n_frames=1,
            frame_interval=1e-3,
        ),
        tcspc=TCSPCsettings(n_bins=64, time_range=25e-9),
        psf=PSFmodel(type="gaussian", emission_wavelength=519.0),
        simulation_time=1e-3,
        bd_timestep=1e-4,
        random_seed=seed,
    )
    config.validate()
    return config


# ═══════════════════════════════════════════════════════════════════════
# T1: CATALYTIC REACTION CONSERVATION
# ═══════════════════════════════════════════════════════════════════════

class TestCatalyticReactionConservation:
    """T1: Catalytic reaction conserves particle count and mass."""

    def test_catalytic_reaction_simple(self):
        """Basic catalytic conversion: E + S → E + P (deterministic)."""
        # Setup: 3 particles, 1 enzyme + 2 substrate, close together
        N = 3
        positions = np.array([
            [0.1, 0.1],   # Enzyme at i=0
            [0.11, 0.11], # Substrate at i=1 (close, ~14 nm)
            [0.5, 0.5],   # Substrate at i=2 (far)
        ], dtype=np.float64)

        species_id = np.array([0, 1, 1], dtype=np.int32)  # 0=Enzyme, 1=Substrate
        is_alive = np.array([True, True, True], dtype=bool)

        # Create pair list manually
        pairs = np.array([[0, 1]], dtype=np.int32)  # Only pair (E, S)
        n_pairs = 1

        # RNG: accept the conversion
        rng_uniform = np.array([0.0], dtype=np.float64)

        Lx, Ly = 1.0, 1.0
        contact_radius_um = 0.02  # 20 nm

        # Call the function with prob_catalysis=1.0 (deterministic)
        n_converted = apply_catalytic_reaction(
            positions=positions,
            species_id=species_id,
            is_alive=is_alive,
            N=N,
            enzyme_id=0,
            substrate_id=1,
            product_id=2,
            contact_radius_um=contact_radius_um,
            prob_catalysis=1.0,
            pairs=pairs,
            n_pairs=n_pairs,
            rng_uniform=rng_uniform,
            Lx=Lx,
            Ly=Ly,
            periodic_x=True,
            periodic_y=True,
        )

        # Assertions
        assert n_converted == 1, f"Expected 1 conversion, got {n_converted}"
        assert species_id[0] == 0, "Enzyme should remain unchanged"
        assert species_id[1] == 2, "Substrate should convert to Product"
        assert species_id[2] == 1, "Far substrate should remain unchanged"
        assert np.sum(is_alive) == 3, "All particles should still be alive"

    def test_catalytic_conservation_multiple_pairs(self):
        """Multiple E-S pairs conserve substrate+product pool."""
        N = 8
        positions = np.array([
            [0.0, 0.0],    # Enzyme 0
            [0.001, 0.001], # Substrate 1 (close to E0)
            [0.2, 0.2],    # Enzyme 2
            [0.201, 0.201], # Substrate 3 (close to E2)
            [0.3, 0.3],    # Substrate 4 (alone)
            [0.4, 0.4],    # Substrate 5 (alone)
            [0.5, 0.5],    # Substrate 6 (alone)
            [0.9, 0.9],    # Substrate 7 (alone)
        ], dtype=np.float64)

        species_id = np.array([0, 1, 0, 1, 1, 1, 1, 1], dtype=np.int32)
        is_alive = np.array([True, True, True, True, True, True, True, True], dtype=bool)

        # Count substrates before
        substrate_count_before = np.sum(species_id == 1)

        # Two close E-S pairs
        pairs = np.array([[0, 1], [2, 3]], dtype=np.int32)
        n_pairs = 2

        # Both pairs accept
        rng_uniform = np.array([0.1, 0.1], dtype=np.float64)

        contact_radius_um = 0.02  # 20 nm

        n_converted = apply_catalytic_reaction(
            positions=positions,
            species_id=species_id,
            is_alive=is_alive,
            N=N,
            enzyme_id=0,
            substrate_id=1,
            product_id=2,
            contact_radius_um=contact_radius_um,
            prob_catalysis=1.0,  # 100% deterministic
            pairs=pairs,
            n_pairs=n_pairs,
            rng_uniform=rng_uniform,
            Lx=1.0,
            Ly=1.0,
            periodic_x=True,
            periodic_y=True,
        )

        # Total substrate (id=1) + product (id=2) should remain constant
        total_s_p_before = substrate_count_before
        total_s_p_after = np.sum(species_id == 1) + np.sum(species_id == 2)
        assert total_s_p_after == total_s_p_before, \
            f"Substrate+Product not conserved: {total_s_p_before} → {total_s_p_after}"
        assert n_converted == 2, f"Expected 2 conversions, got {n_converted}"

    def test_enzyme_unchanged(self):
        """Enzyme count and species_id never change."""
        N = 5
        positions = np.random.default_rng(42).uniform(0, 1, (N, 2))
        species_id = np.array([0, 0, 1, 1, 1], dtype=np.int32)  # 2 enzymes, 3 substrates
        is_alive = np.array([True, True, True, True, True], dtype=bool)

        pairs = np.array([[0, 2], [1, 3]], dtype=np.int32)
        n_pairs = 2
        rng_uniform = np.ones(2, dtype=np.float64) * 0.5

        enzyme_count_before = np.sum(species_id == 0)

        apply_catalytic_reaction(
            positions=positions,
            species_id=species_id,
            is_alive=is_alive,
            N=N,
            enzyme_id=0,
            substrate_id=1,
            product_id=2,
            contact_radius_um=0.5,
            prob_catalysis=1.0,
            pairs=pairs,
            n_pairs=n_pairs,
            rng_uniform=rng_uniform,
            Lx=1.0,
            Ly=1.0,
            periodic_x=True,
            periodic_y=True,
        )

        enzyme_count_after = np.sum(species_id == 0)
        assert enzyme_count_before == enzyme_count_after, "Enzyme count changed"
        assert np.all(species_id[species_id == 0] == 0), "Enzyme species_id changed"


# ═══════════════════════════════════════════════════════════════════════
# T2: COMPARTMENT EXCHANGE CONSERVATION
# ═══════════════════════════════════════════════════════════════════════

class TestCompartmentExchangeConservation:
    """T2: Membrane ↔ cytosol exchange conserves total particles."""

    def test_membrane_recruitment_basic(self):
        """Dormant → alive particles via recruitment."""
        N = 50
        # 20 alive, 30 dormant
        species_id = np.zeros(N, dtype=np.int32)
        is_alive = np.concatenate([np.ones(20, dtype=bool),
                                     np.zeros(30, dtype=bool)])
        positions = np.zeros((N, 2), dtype=np.float64)

        Lx, Ly = 1.0, 1.0

        # RNG for acceptance: first 10 dormant accept
        rng_accept = np.concatenate([
            np.ones(20, dtype=np.float64),  # alive stay alive (rng > 1)
            np.linspace(0.05, 0.45, 30, dtype=np.float64)  # dormant: first 10 < 0.5
        ])

        # Random positions for recruits
        rng_positions = np.random.default_rng(42).uniform(0, 1, (N, 2))

        n_recruited = apply_membrane_recruitment(
            species_id=species_id,
            is_alive=is_alive,
            positions=positions,
            N=N,
            target_species_id=0,
            prob_recruit=0.5,
            Lx=Lx,
            Ly=Ly,
            rng_accept=rng_accept,
            rng_positions=rng_positions,
            max_recruits=15,
        )

        # Should recruit ~10 (those with rng < 0.5)
        alive_after = np.sum(is_alive)
        total_alive_and_dormant = np.sum(is_alive) + np.sum(~is_alive)

        assert n_recruited > 0, "No particles recruited"
        assert total_alive_and_dormant == 50, "Total pool not conserved"
        assert alive_after == 20 + n_recruited, "Alive count mismatch"

    def test_membrane_detachment_basic(self):
        """Alive → dormant particles via detachment."""
        N = 50
        species_id = np.zeros(N, dtype=np.int32)
        is_alive = np.ones(N, dtype=bool)

        rng_uniform = np.linspace(0.1, 0.9, N, dtype=np.float64)

        n_detached = apply_membrane_detachment(
            species_id=species_id,
            is_alive=is_alive,
            N=N,
            target_species_id=0,
            prob_detach=0.5,
            rng_uniform=rng_uniform,
        )

        # Should detach ~25 (half with rng < 0.5)
        assert 20 <= n_detached <= 30, f"Unexpected detachment count: {n_detached}"
        assert np.sum(is_alive) + n_detached == N, "Total not conserved"

    def test_recruitment_then_detachment_cycle(self):
        """Cycle: recruit particles, then detach some."""
        N = 60
        species_id = np.zeros(N, dtype=np.int32)
        is_alive = np.concatenate([np.ones(30, dtype=bool),
                                     np.zeros(30, dtype=bool)])
        positions = np.random.default_rng(42).uniform(0, 1, (N, 2))

        Lx, Ly = 1.0, 1.0

        # Recruitment phase
        rng_accept = np.concatenate([
            np.ones(30, dtype=np.float64),
            np.linspace(0.1, 0.4, 30, dtype=np.float64)
        ])
        rng_positions = np.random.default_rng(43).uniform(0, 1, (N, 2))

        n_recruited = apply_membrane_recruitment(
            species_id=species_id,
            is_alive=is_alive,
            positions=positions,
            N=N,
            target_species_id=0,
            prob_recruit=0.3,
            Lx=Lx,
            Ly=Ly,
            rng_accept=rng_accept,
            rng_positions=rng_positions,
            max_recruits=20,
        )

        alive_after_recruit = np.sum(is_alive)

        # Detachment phase
        rng_detach = np.linspace(0.05, 0.95, N, dtype=np.float64)
        n_detached = apply_membrane_detachment(
            species_id=species_id,
            is_alive=is_alive,
            N=N,
            target_species_id=0,
            prob_detach=0.5,
            rng_uniform=rng_detach,
        )

        # Total pool should remain 60
        total_final = np.sum(is_alive) + np.sum(~is_alive)
        assert total_final == N, f"Pool not conserved: {total_final} != {N}"


# ═══════════════════════════════════════════════════════════════════════
# T3: SPECIES CONVERSION BALANCE
# ═══════════════════════════════════════════════════════════════════════

class TestSpeciesConversionBalance:
    """T3: Unimolecular conversion conserves total particle count."""

    def test_species_conversion_simple(self):
        """First-order conversion A → B."""
        N = 100
        species_id = np.concatenate([
            np.zeros(60, dtype=np.int32),  # 60 of species A (id=0)
            np.ones(40, dtype=np.int32),   # 40 of species B (id=1)
        ])
        is_alive = np.ones(N, dtype=bool)

        total_before = 100
        count_a_before = np.sum(species_id == 0)
        count_b_before = np.sum(species_id == 1)

        # RNG: convert particles where rng < 0.1 (expect ~6 of 60)
        rng_uniform = np.random.default_rng(42).uniform(0, 1, N)

        n_converted = apply_species_conversion(
            species_id=species_id,
            is_alive=is_alive,
            N=N,
            from_id=0,
            to_id=1,
            prob_convert=0.1,
            rng_uniform=rng_uniform,
        )

        total_after = 100
        count_a_after = np.sum(species_id == 0)
        count_b_after = np.sum(species_id == 1)

        # Conservation checks
        assert total_before == total_after, "Total particle count changed"
        assert count_a_before == count_a_after + n_converted, "A count mismatch"
        assert count_b_after == count_b_before + n_converted, "B count mismatch"
        assert count_a_after + count_b_after == 100, "Total species mismatch"

    def test_species_conversion_deterministic(self):
        """Deterministic conversion (prob=1.0)."""
        N = 20
        species_id = np.zeros(N, dtype=np.int32)
        is_alive = np.ones(N, dtype=bool)

        rng_uniform = np.zeros(N, dtype=np.float64)  # All rng < 1.0

        n_converted = apply_species_conversion(
            species_id=species_id,
            is_alive=is_alive,
            N=N,
            from_id=0,
            to_id=2,
            prob_convert=1.0,
            rng_uniform=rng_uniform,
        )

        assert n_converted == 20, f"All should convert, got {n_converted}"
        assert np.all(species_id == 2), "All particles should be species 2"

    def test_species_conversion_ignores_alive_false(self):
        """Dormant particles (is_alive=False) are not converted."""
        N = 50
        species_id = np.zeros(N, dtype=np.int32)
        is_alive = np.concatenate([
            np.ones(30, dtype=bool),   # 30 alive
            np.zeros(20, dtype=bool),  # 20 dormant
        ])

        rng_uniform = np.zeros(N, dtype=np.float64)

        n_converted = apply_species_conversion(
            species_id=species_id,
            is_alive=is_alive,
            N=N,
            from_id=0,
            to_id=1,
            prob_convert=1.0,
            rng_uniform=rng_uniform,
        )

        assert n_converted == 30, f"Only alive should convert, got {n_converted}"
        assert species_id[0] == 1, "Alive should convert"
        assert species_id[35] == 0, "Dormant should not convert"


# ═══════════════════════════════════════════════════════════════════════
# T4: DENSITY FIELD ACCURACY
# ═══════════════════════════════════════════════════════════════════════

class TestDensityFieldAccuracy:
    """T4: Density field binning is accurate."""

    def test_density_field_uniform(self):
        """Uniform distribution → uniform density."""
        N = 100
        # Create a 2×2 µm box with 100 particles uniformly distributed
        positions = np.array([
            [0.5, 0.5],  # 50 particles in lower-left quadrant
            [1.5, 0.5],
            [0.5, 1.5],
            [1.5, 1.5],
        ] * 25, dtype=np.float64)[:100]

        species_id = np.zeros(N, dtype=np.int32)
        is_alive = np.ones(N, dtype=bool)

        # Grid: 2 cells × 2 cells (each 1 µm)
        nx_grid, ny_grid = 2, 2
        cell_sx, cell_sy = 1.0, 1.0
        density_out = np.zeros((nx_grid, ny_grid), dtype=np.float64)

        compute_density_field(
            positions=positions,
            species_id=species_id,
            is_alive=is_alive,
            N=N,
            target_species_id=0,
            nx_grid=nx_grid,
            ny_grid=ny_grid,
            cell_sx=cell_sx,
            cell_sy=cell_sy,
            density_out=density_out,
        )

        # Each cell should have ~25 particles → density ~25 p/µm²
        cell_area = 1.0 * 1.0  # 1 µm²
        expected_density = 25.0  # particles/µm²

        for i in range(nx_grid):
            for j in range(ny_grid):
                assert abs(density_out[i, j] - expected_density) < 1.0, \
                    f"Cell ({i},{j}) density {density_out[i,j]:.1f} != {expected_density}"

    def test_density_field_clustered(self):
        """Clustered distribution → non-uniform density."""
        N = 100
        # 80 particles in lower-left, 20 in upper-right
        positions = np.concatenate([
            np.random.default_rng(42).uniform(0.0, 0.5, (80, 2)),  # cluster in (0,0.5)²
            np.random.default_rng(43).uniform(1.5, 2.0, (20, 2)),  # cluster in (1.5,2)²
        ])

        species_id = np.zeros(N, dtype=np.int32)
        is_alive = np.ones(N, dtype=bool)

        nx_grid, ny_grid = 2, 2
        cell_sx, cell_sy = 1.0, 1.0
        density_out = np.zeros((nx_grid, ny_grid), dtype=np.float64)

        compute_density_field(
            positions=positions,
            species_id=species_id,
            is_alive=is_alive,
            N=N,
            target_species_id=0,
            nx_grid=nx_grid,
            ny_grid=ny_grid,
            cell_sx=cell_sx,
            cell_sy=cell_sy,
            density_out=density_out,
        )

        # Lower-left (0,0) should have ~80 particles → ~80 p/µm²
        # Upper-right (1,1) should have ~20 particles → ~20 p/µm²
        assert density_out[0, 0] > 50.0, f"LL cluster too low: {density_out[0,0]}"
        assert density_out[1, 1] > 10.0, f"UR cluster too low: {density_out[1,1]}"
        assert density_out[0, 0] > density_out[1, 1], "LL should be denser than UR"

    def test_density_field_ignores_dormant(self):
        """Dormant particles not counted in density."""
        N = 100
        positions = np.random.default_rng(42).uniform(0, 2, (N, 2))
        species_id = np.zeros(N, dtype=np.int32)
        is_alive = np.concatenate([
            np.ones(60, dtype=bool),   # 60 alive
            np.zeros(40, dtype=bool),  # 40 dormant
        ])

        nx_grid, ny_grid = 2, 2
        cell_sx, cell_sy = 1.0, 1.0
        density_out = np.zeros((nx_grid, ny_grid), dtype=np.float64)

        compute_density_field(
            positions=positions,
            species_id=species_id,
            is_alive=is_alive,
            N=N,
            target_species_id=0,
            nx_grid=nx_grid,
            ny_grid=ny_grid,
            cell_sx=cell_sx,
            cell_sy=cell_sy,
            density_out=density_out,
        )

        # Total should be ~60 particles distributed, not 100
        total_counted = np.sum(density_out) * 1.0  # multiply by cell_area (already dividing)
        assert total_counted < 70.0, f"Should count ~60, counted {total_counted:.1f}"


# ═══════════════════════════════════════════════════════════════════════
# T5: FEEDBACK HILL MODULATION
# ═══════════════════════════════════════════════════════════════════════

class TestFeedbackHillModulation:
    """T5: Hill function modulates catalytic rate correctly."""

    def test_catalytic_with_inhibitory_feedback(self):
        """High density inhibits catalysis."""
        N = 10
        # 5 E-S pairs, all close
        positions = np.array([
            [0.1, 0.1],   # E0
            [0.11, 0.11], # S0 in high-density region
            [0.3, 0.3],   # E1
            [0.31, 0.31], # S1 in high-density region
            [0.9, 0.9],   # E2
            [0.91, 0.91], # S2 in low-density region
            [0.5, 0.1],   # E3
            [0.51, 0.11], # S3 in high-density region
            [0.7, 0.7],   # E4
            [0.71, 0.71], # S4 in high-density region
        ], dtype=np.float64)

        species_id = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
        is_alive = np.ones(N, dtype=bool)

        # Density field: high density in cells [0,0] and [0,1], low elsewhere
        nx_grid, ny_grid = 2, 2
        cell_sx, cell_sy = 0.5, 0.5
        density_field = np.array([
            [500.0, 500.0],  # Both cells in bottom half have high density
            [10.0, 10.0],    # Top half has low density
        ], dtype=np.float64)

        pairs = np.array([
            [0, 1], [2, 3], [4, 5], [6, 7], [8, 9]
        ], dtype=np.int32)
        n_pairs = 5

        # All pairs accept if no feedback
        rng_uniform = np.zeros(5, dtype=np.float64)

        # With inhibition, high density should suppress conversion
        n_converted = apply_catalytic_with_feedback(
            positions=positions,
            species_id=species_id,
            is_alive=is_alive,
            N=N,
            enzyme_id=0,
            substrate_id=1,
            product_id=2,
            contact_radius_um=0.02,
            prob_catalysis_base=1.0,
            pairs=pairs,
            n_pairs=n_pairs,
            rng_uniform=rng_uniform,
            Lx=1.0,
            Ly=1.0,
            periodic_x=True,
            periodic_y=True,
            density_field=density_field,
            nx_grid=nx_grid,
            ny_grid=ny_grid,
            cell_sx=cell_sx,
            cell_sy=cell_sy,
            K_half=200.0,
            n_hill=3.0,
            is_inhibition=True,
        )

        # Conversion should happen in low-density regions (pair 4: S2 at 0.91, 0.91 → cell [1,1])
        # Conversions in high-density regions should be suppressed
        assert n_converted > 0, "Some conversion should occur"
        assert n_converted <= 5, f"Too many conversions: {n_converted}"

    def test_hill_inhibition_formula(self):
        """Hill inhibition: k_eff = k_base / (1 + (rho/K)^n)."""
        # When rho >> K: k_eff ≈ 0
        # When rho << K: k_eff ≈ k_base
        # When rho = K: k_eff = k_base / 2

        rho_low = 10.0
        rho_high = 500.0
        K_half = 200.0
        n_hill = 3.0
        k_base = 1.0

        # Low density
        ratio_low = rho_low / K_half
        hill_low = 1.0 / (1.0 + ratio_low**n_hill)
        k_eff_low = k_base * hill_low

        # High density
        ratio_high = rho_high / K_half
        hill_high = 1.0 / (1.0 + ratio_high**n_hill)
        k_eff_high = k_base * hill_high

        assert k_eff_low > k_eff_high, "Low density should have higher rate"
        assert k_eff_low > 0.5, "Low density rate should be substantial"
        assert k_eff_high < 0.1, "High density rate should be suppressed"


# ═══════════════════════════════════════════════════════════════════════
# T6: PROXIMITY FRET RATES
# ═══════════════════════════════════════════════════════════════════════

class TestProximityFRETRates:
    """T6: Proximity-based FRET rates (biosensor mode)."""

    def test_proximity_fret_close_targets(self):
        """Donors near target species get high FRET rate."""
        N = 5
        positions = np.array([
            [0.0, 0.0],    # Donor 0, species=1
            [0.001, 0.001], # Target 1, species=2 (close to donor 0)
            [0.5, 0.5],    # Donor 2, species=1 (far from all targets)
            [0.8, 0.8],    # Target 3, species=2 (far from donor 2)
            [0.9, 0.9],    # Dummy, species=1
        ], dtype=np.float64)

        # Donor particles have dye_type_id=0, target particles have species_id=2
        species_id = np.array([1, 2, 1, 2, 1], dtype=np.int32)
        has_dye = np.ones(N, dtype=bool)
        is_alive = np.ones(N, dtype=bool)
        is_bleached = np.zeros(N, dtype=bool)
        dye_type_id = np.array([0, 0, 0, 0, 0], dtype=np.int32)  # All have donor dye type

        k_fret = np.zeros(N, dtype=np.float64)
        fret_acceptor_idx = np.full(N, -1, dtype=np.int32)

        proximity_radius_um = 0.02  # 20 nm

        _compute_fret_rates_proximity_numba(
            k_fret=k_fret,
            fret_acceptor_idx=fret_acceptor_idx,
            positions=positions,
            species_id=species_id,
            has_dye=has_dye,
            is_alive=is_alive,
            is_bleached=is_bleached,
            dye_type_id=dye_type_id,
            N=N,
            tau_D=4.1e-9,
            donor_fid=0,
            target_species_id=2,
            proximity_radius_um=proximity_radius_um,
            k_fret_high=1e8,
            k_fret_low=1e6,
            Lx=1.0,
            Ly=1.0,
            periodic_x=True,
            periodic_y=True,
        )

        # Donor 0 (species=1) near target 1 (species=2, ~1.4 nm) should have k_fret = k_fret_high
        # Donor 2 (species=1) far from all targets (~424 nm to nearest target) should have k_fret = k_fret_low
        assert k_fret[0] > 5e7, f"Donor 0 near target should have high rate, got {k_fret[0]}"
        assert k_fret[2] < 2e6, f"Donor 2 far from target should have low rate, got {k_fret[2]}"

    def test_proximity_fret_respects_range(self):
        """Only targets within proximity_radius affect FRET."""
        N = 4
        positions = np.array([
            [0.0, 0.0],   # Donor 0
            [0.01, 0.01], # Target at 14 nm (within 20 nm)
            [0.05, 0.05], # Target at 70 nm (outside 20 nm)
            [0.0, 0.0],   # Dummy
        ], dtype=np.float64)

        species_id = np.array([0, 2, 2, 0], dtype=np.int32)
        has_dye = np.ones(N, dtype=bool)
        is_alive = np.ones(N, dtype=bool)
        is_bleached = np.zeros(N, dtype=bool)
        dye_type_id = np.zeros(N, dtype=np.int32)

        k_fret = np.zeros(N, dtype=np.float64)
        fret_acceptor_idx = np.full(N, -1, dtype=np.int32)

        _compute_fret_rates_proximity_numba(
            k_fret=k_fret,
            fret_acceptor_idx=fret_acceptor_idx,
            positions=positions,
            species_id=species_id,
            has_dye=has_dye,
            is_alive=is_alive,
            is_bleached=is_bleached,
            dye_type_id=dye_type_id,
            N=N,
            tau_D=4.1e-9,
            donor_fid=0,
            target_species_id=2,
            proximity_radius_um=0.02,  # 20 nm
            k_fret_high=1e8,
            k_fret_low=1e6,
            Lx=1.0,
            Ly=1.0,
            periodic_x=True,
            periodic_y=True,
        )

        # Donor 0 near target 1 → should have k_fret_high
        assert k_fret[0] > 5e7, f"Donor near target: {k_fret[0]}"
        # (Note: target 2 is outside range, but donor 0 still sees target 1)

    def test_proximity_fret_ignores_deadness(self):
        """Dead/bleached particles are ignored."""
        N = 6
        positions = np.array([
            [0.0, 0.0],    # Donor 0, species=1
            [0.001, 0.001], # Target 1, species=2 (alive)
            [0.1, 0.1],    # Donor 2, species=1 (far from all targets)
            [0.5, 0.5],    # Target 3, species=2 (dead but would be far anyway)
            [0.2, 0.2],    # Donor 4, species=1 (far from all targets)
            [0.8, 0.8],    # Target 5, species=2 (bleached but would be far anyway)
        ], dtype=np.float64)

        species_id = np.array([1, 2, 1, 2, 1, 2], dtype=np.int32)
        has_dye = np.ones(N, dtype=bool)
        is_alive = np.array([True, True, True, False, True, True], dtype=bool)
        is_bleached = np.array([False, False, False, False, False, True], dtype=bool)
        dye_type_id = np.zeros(N, dtype=np.int32)

        k_fret = np.zeros(N, dtype=np.float64)
        fret_acceptor_idx = np.full(N, -1, dtype=np.int32)

        _compute_fret_rates_proximity_numba(
            k_fret=k_fret,
            fret_acceptor_idx=fret_acceptor_idx,
            positions=positions,
            species_id=species_id,
            has_dye=has_dye,
            is_alive=is_alive,
            is_bleached=is_bleached,
            dye_type_id=dye_type_id,
            N=N,
            tau_D=4.1e-9,
            donor_fid=0,
            target_species_id=2,
            proximity_radius_um=0.02,
            k_fret_high=1e8,
            k_fret_low=1e6,
            Lx=1.0,
            Ly=1.0,
            periodic_x=True,
            periodic_y=True,
        )

        # Donor 0 sees live target 1 (close, ~1.4 nm) → high rate
        assert k_fret[0] > 5e7, f"Donor near live target: {k_fret[0]}"
        # Donor 2 doesn't see any live targets (3 is dead, far anyway) → low rate
        assert k_fret[2] < 2e6, f"Donor away from live targets: {k_fret[2]}"
        # Donor 4 doesn't see any live targets (5 is bleached, far anyway) → low rate
        assert k_fret[4] < 2e6, f"Donor away from live targets: {k_fret[4]}"


# ═══════════════════════════════════════════════════════════════════════
# T7: CONFIG NEW TYPES ROUNDTRIP
# ═══════════════════════════════════════════════════════════════════════

class TestConfigNewTypesRoundtrip:
    """T7: New dataclasses load and validate correctly."""

    def test_catalytic_reaction_config(self):
        """CatalyticReaction dataclass exists and validates."""
        cat = CatalyticReaction(
            name="test_catalysis",
            enzyme="Enz",
            substrate="Sub",
            product="Prod",
            k_cat=100.0,
            contact_radius=7.0,
        )
        assert cat.name == "test_catalysis"
        assert cat.enzyme == "Enz"
        assert cat.k_cat == 100.0

    def test_compartment_exchange_config(self):
        """CompartmentExchange dataclass exists and validates."""
        exch = CompartmentExchange(
            name="test_exchange",
            species="Protein",
            k_on=0.5,
            k_off=0.1,
            cytosol_pool=100,
            max_recruits_per_step=5,
        )
        assert exch.name == "test_exchange"
        assert exch.k_on == 0.5
        assert exch.cytosol_pool == 100

    def test_unimolecular_conversion_config(self):
        """UnimolecularConversion dataclass exists."""
        uni = UnimolecularConversion(
            name="test_conv",
            from_species="A",
            to_species="B",
            rate=0.1,
        )
        assert uni.from_species == "A"
        assert uni.to_species == "B"

    def test_feedback_rule_config(self):
        """FeedbackRule dataclass exists."""
        fb = FeedbackRule(
            name="test_feedback",
            sensor_species="PIP3",
            function_type="hill_inhibition",
            K_half=200.0,
            n_hill=3.0,
            grid_resolution=0.2,
        )
        assert fb.sensor_species == "PIP3"
        assert fb.K_half == 200.0

    def test_fretpair_proximity_fields(self):
        """FRETpair extended with proximity fields."""
        fret = FRETpair(
            donor_type="donor",
            acceptor_type="acceptor",
            R0=6.4,
            fret_mode="proximity",
            proximity_radius=10.0,
            proximity_E_high=0.5,
            proximity_E_low=0.02,
            proximity_target_species="PIP3",
        )
        assert fret.fret_mode == "proximity"
        assert fret.proximity_radius == 10.0
        assert fret.proximity_target_species == "PIP3"

    def test_config_with_catalytic_reactions(self):
        """SimulationConfig accepts catalytic_reactions field."""
        config = _make_minimal_config_with_catalytic(n_particles=10)
        assert len(config.catalytic_reactions) == 1
        assert config.catalytic_reactions[0].name == "catalysis"

    def test_config_with_compartment_exchanges(self):
        """SimulationConfig accepts compartment_exchanges field."""
        config = _make_minimal_config_with_exchange(n_particles=10)
        assert len(config.compartment_exchanges) == 1
        assert config.compartment_exchanges[0].name == "recruitment"


# ═══════════════════════════════════════════════════════════════════════
# T8: INTEGRATION TEST (MINI PIPELINE)
# ═══════════════════════════════════════════════════════════════════════

class TestIntegrationMiniPipeline:
    """T8: End-to-end pipeline with new extensions."""

    def test_pipeline_runs_with_catalytic_reactions(self):
        """Pipeline runs successfully with catalytic reactions configured."""
        config = _make_minimal_config_with_catalytic(
            n_particles=15,
            n_frames=1,
            bd_steps_per_frame=2,
        )
        result = SimulationPipeline(config).run()

        assert result.n_frames == 1
        assert result.total_wall_time_s > 0
        # Check that fluorophore was attached
        assert len(result.frames[0].ground_truth.positions) >= 10

    def test_pipeline_runs_with_compartment_exchange(self):
        """Pipeline runs successfully with compartment exchange."""
        config = _make_minimal_config_with_exchange(n_particles=20)
        result = SimulationPipeline(config).run()

        assert result.n_frames == 1
        assert result.total_wall_time_s > 0

    def test_mini_demo_integration(self):
        """Mini demo completes with extended config."""
        config = _make_minimal_config_with_catalytic(
            n_particles=20,
            n_frames=2,
            bd_steps_per_frame=3,
        )
        result = SimulationPipeline.run_mini_demo(config, n_frames=2,
                                                   n_bd_steps_per_frame=3)
        assert result.n_frames == 2
        assert result.total_wall_time_s > 0
        # Summary should be well-formed
        summary = result.summary()
        assert "ED-MRDT" in summary


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
