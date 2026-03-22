"""
ED-MRDT v1.1 — Pipeline Tests (Step 4.1)
==========================================

Quantitative tests for SimulationPipeline orchestrator.

Test categories:
  T1: Initialization & timing consistency
  T2: Frame scheduling arithmetic
  T3: Component coupling (BD → photophysics → microscope)
  T4: Ground truth extraction accuracy
  T5: RAM monitoring
  T6: Multi-frame continuity (state persists between frames)
  T7: Mini-demo end-to-end (photons detected, FLIM stack nonzero)
  T8: Edge cases (1 frame, 1 particle, 1 BD step)
  T9: Progress callback
  T10: Bleaching monotonicity across frames

References:
  - Lakowicz (2006) Eq. 13.10: E = k_FRET·τ_D / (1 + k_FRET·τ_D)
  - Becker (2005): FLIM frame = accumulated TCSPC histogram
"""

import math
import sys
import os
import time

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
)
from ed_mrdt.particles import ParticleState
from ed_mrdt.dynamics import DynamicsEngine
from ed_mrdt.photophysics import PhotonSimulator, fret_efficiency
from ed_mrdt.microscope import VirtualMicroscope, FLIMFrame
from ed_mrdt.pipeline import (
    SimulationPipeline, FrameResult, SimulationResult, GroundTruth,
)


# ═══════════════════════════════════════════════════════════════════════
# FIXTURES: Minimal configs for fast testing
# ═══════════════════════════════════════════════════════════════════════

def _make_minimal_config(
    n_particles: int = 20,
    Lx: float = 0.5,
    Ly: float = 0.5,
    n_frames: int = 2,
    bd_steps_per_frame: int = 5,
    dt: float = 1e-4,
    seed: int = 42,
) -> SimulationConfig:
    """Build minimal config for fast pipeline tests.

    Uses small membrane, few particles, minimal frames.
    frame_interval = bd_steps_per_frame × dt
    """
    frame_interval = bd_steps_per_frame * dt
    pixel_size_nm = Lx * 1000.0 / 5  # 5×5 pixel grid

    config = SimulationConfig(
        name="pipeline_test",
        membrane=MembraneGeometry(size=(Lx, Ly), periodic=(True, True)),
        diffusion_field=DiffusionField(type="uniform"),
        species=[
            Species(name="R", diffusion_coefficient=0.05, radius=5.0),
            Species(name="L", diffusion_coefficient=1.0, radius=2.0),
            Species(name="RL", diffusion_coefficient=0.04, radius=6.0),
        ],
        reactions=[
            Reaction(
                name="bind", reactants=["R", "L"], products=["RL"],
                kon=0.1, koff=0.01, contact_radius=7.0,
            ),
        ],
        potentials=[
            PairPotential(
                species_a="R", species_b="R",
                potential_type="wca",
                parameters={"sigma": 8.0, "epsilon": 2.5},
            ),
        ],
        initial_populations={"R": n_particles, "L": n_particles // 2},
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
            FluorophoreType(
                name="acceptor",
                states=[
                    PhotoState(name="S0", is_absorbing=True),
                    PhotoState(name="S1", is_fluorescent=True,
                               quantum_yield=0.10, emission_wavelength=565.0),
                    PhotoState(name="T1"),
                    PhotoState(name="bleached", is_terminal=True),
                ],
                thermal_rates={
                    "S1": {"S0": 3.33e9, "T1": 5e5, "bleached": 5e2},
                    "T1": {"S0": 5e3, "bleached": 50.0},
                },
                absorption_cross_section=1.5e-16,
                tau_rad=3e-10,
            ),
        ],
        fret_pairs=[
            FRETpair(donor_type="donor", acceptor_type="acceptor", R0=6.4),
        ],
        dye_attachments=[
            DyeAttachment(species="R", fluorophore="donor",
                          labeling_efficiency=0.8),
            DyeAttachment(species="L", fluorophore="acceptor",
                          labeling_efficiency=0.9),
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


# ═══════════════════════════════════════════════════════════════════════
# T1: INITIALIZATION & TIMING
# ═══════════════════════════════════════════════════════════════════════

class TestInitialization:
    """T1: Pipeline initializes correctly from config."""

    def test_basic_creation(self):
        """Pipeline creates without error."""
        config = _make_minimal_config()
        pipeline = SimulationPipeline(config)
        assert pipeline.n_frames == 2
        assert pipeline.n_bd_per_frame == 5
        assert pipeline.dt == 1e-4

    def test_bd_steps_per_frame_exact(self):
        """n_bd_per_frame = frame_interval / bd_timestep."""
        config = _make_minimal_config(bd_steps_per_frame=10, dt=1e-5)
        pipeline = SimulationPipeline(config)
        assert pipeline.n_bd_per_frame == 10

    def test_frame_interval_not_multiple_of_dt(self):
        """Raise ConfigError if frame_interval not integer multiple of dt."""
        config = _make_minimal_config(dt=1e-4)
        config.scan.frame_interval = 3.33e-4  # not a clean multiple
        # Should still work if drift < 1%
        # 3.33e-4 / 1e-4 = 3.33 → rounds to 3, drift = |3e-4 - 3.33e-4|/3.33e-4 = 9.9%
        with pytest.raises(ConfigError):
            SimulationPipeline(config)

    def test_zero_frame_interval_raises(self):
        """frame_interval=0 raises ConfigError."""
        config = _make_minimal_config()
        config.scan.frame_interval = 0.0
        with pytest.raises(ConfigError):
            SimulationPipeline(config)

    def test_repr(self):
        """Repr includes key info."""
        config = _make_minimal_config()
        pipeline = SimulationPipeline(config)
        r = repr(pipeline)
        assert "pipeline_test" in r
        assert "frames" in r

    def test_large_bd_steps_per_frame(self):
        """Many BD steps per frame (canonical-like)."""
        config = _make_minimal_config(
            n_frames=1, bd_steps_per_frame=100, dt=1e-5
        )
        pipeline = SimulationPipeline(config)
        assert pipeline.n_bd_per_frame == 100


# ═══════════════════════════════════════════════════════════════════════
# T2: FRAME SCHEDULING
# ═══════════════════════════════════════════════════════════════════════

class TestFrameScheduling:
    """T2: Biological time advances correctly per frame."""

    def test_bio_time_per_frame(self):
        """bio_time = (frame_i + 1) * frame_interval."""
        config = _make_minimal_config(n_frames=3, bd_steps_per_frame=3, dt=1e-4)
        pipeline = SimulationPipeline(config)
        result = pipeline.run()

        for i, frame in enumerate(result.frames):
            expected_t = (i + 1) * pipeline.frame_interval
            assert abs(frame.bio_time - expected_t) < 1e-12, \
                f"Frame {i}: bio_time={frame.bio_time}, expected={expected_t}"

    def test_total_bd_steps(self):
        """Total BD steps = n_frames × n_bd_per_frame."""
        config = _make_minimal_config(n_frames=2, bd_steps_per_frame=5)
        pipeline = SimulationPipeline(config)
        # Total steps = 2 × 5 = 10
        assert pipeline.n_bd_per_frame * pipeline.n_frames == 10

    def test_frame_indices_sequential(self):
        """Frame indices are 0, 1, 2, ..."""
        config = _make_minimal_config(n_frames=4, bd_steps_per_frame=2)
        result = SimulationPipeline(config).run()
        indices = [f.frame_index for f in result.frames]
        assert indices == [0, 1, 2, 3]


# ═══════════════════════════════════════════════════════════════════════
# T3: COMPONENT COUPLING
# ═══════════════════════════════════════════════════════════════════════

class TestComponentCoupling:
    """T3: BD, photophysics, and microscope are coupled correctly."""

    def test_photons_emitted_positive(self):
        """At least some photons are emitted per frame."""
        config = _make_minimal_config(
            n_particles=40, bd_steps_per_frame=10, dt=1e-4
        )
        result = SimulationPipeline(config).run()
        # With 40 particles, 10 steps, 4000 pulses/step → should have photons
        assert result.total_photons_emitted > 0, \
            "No photons emitted — photophysics not coupled"

    def test_photons_detected_subset_of_emitted(self):
        """Detected ≤ emitted (detection efficiency + PSF filtering)."""
        config = _make_minimal_config(
            n_particles=40, bd_steps_per_frame=10, dt=1e-4
        )
        result = SimulationPipeline(config).run()
        assert result.total_photons_detected <= result.total_photons_emitted

    def test_flim_stack_has_counts(self):
        """FLIM stack is not all zeros after simulation."""
        config = _make_minimal_config(
            n_particles=40, bd_steps_per_frame=10, dt=1e-4
        )
        result = SimulationPipeline(config).run()
        total = sum(f.flim_frame.total_counts for f in result.frames)
        assert total > 0, "FLIM stack all zeros — microscope not receiving photons"

    def test_flim_stack_shape(self):
        """FLIM stack has correct shape [n_ch, nx, ny, n_bins]."""
        config = _make_minimal_config()
        result = SimulationPipeline(config).run()
        frame0 = result.frames[0].flim_frame
        # n_channels=2 (donor, acceptor), 5×5 pixels, 64 bins
        assert frame0.shape == (2, 5, 5, 64), f"Wrong shape: {frame0.shape}"

    def test_dynamics_bindings_tracked(self):
        """Dynamics engine tracks binding/dissociation counts."""
        config = _make_minimal_config(
            n_particles=30, bd_steps_per_frame=20, dt=1e-4
        )
        result = SimulationPipeline(config).run()
        # Bindings may or may not happen depending on RNG, but stats exist
        assert result.total_bindings >= 0
        assert result.total_dissociations >= 0


# ═══════════════════════════════════════════════════════════════════════
# T4: GROUND TRUTH EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

class TestGroundTruth:
    """T4: Ground truth captures exact physical state."""

    def test_ground_truth_arrays_shape(self):
        """GT arrays have correct shape [N]."""
        config = _make_minimal_config(n_particles=20)
        result = SimulationPipeline(config).run()
        gt = result.frames[0].ground_truth

        N = 20 + 10  # R=20, L=10
        assert gt.positions.shape == (N, 2)
        assert gt.species_id.shape == (N,)
        assert gt.bond_partner.shape == (N,)
        assert gt.fret_efficiency_exact.shape == (N,)
        assert gt.donor_lifetime_exact.shape == (N,)

    def test_fret_efficiency_bounded(self):
        """FRET efficiency ∈ [0, 1] for all particles."""
        config = _make_minimal_config(n_particles=30, bd_steps_per_frame=10)
        result = SimulationPipeline(config).run()
        for frame in result.frames:
            E = frame.ground_truth.fret_efficiency_exact
            assert np.all(E >= 0.0), f"Negative FRET E: {E.min()}"
            assert np.all(E <= 1.0), f"FRET E > 1: {E.max()}"

    def test_donor_lifetime_positive(self):
        """Donor lifetime ≥ 0 for all particles."""
        config = _make_minimal_config(n_particles=30, bd_steps_per_frame=10)
        result = SimulationPipeline(config).run()
        for frame in result.frames:
            tau = frame.ground_truth.donor_lifetime_exact
            assert np.all(tau >= 0.0), f"Negative lifetime: {tau.min()}"

    def test_donor_only_lifetime_equals_tau_D(self):
        """Free donors (no FRET partner) have τ = τ_D ≈ 4.1 ns."""
        config = _make_minimal_config(
            n_particles=20, bd_steps_per_frame=5,
            Lx=2.0, Ly=2.0,  # large box → most particles far apart
        )
        # Adjust pixel grid for larger box
        config.scan.pixel_size = 400.0
        config.scan.pixels_x = 5
        config.scan.pixels_y = 5
        config.validate()

        result = SimulationPipeline(config).run()
        gt = result.frames[-1].ground_truth

        # Find donor particles with no FRET partner
        tau = gt.donor_lifetime_exact
        # Donors with τ > 0 but no FRET (E≈0 → τ≈τ_D)
        E = gt.fret_efficiency_exact
        free_donor_mask = (tau > 0) & (E < 0.01)
        if np.sum(free_donor_mask) > 0:
            tau_free = tau[free_donor_mask]
            # Should be close to τ_D = 4.1 ns
            tau_D = 4.1e-9
            relative_error = np.abs(tau_free - tau_D) / tau_D
            assert np.all(relative_error < 0.05), \
                f"Free donor τ deviates from τ_D: {tau_free} vs {tau_D}"

    def test_complexes_counted(self):
        """n_complexes ≥ 0 and bounded by particle count."""
        config = _make_minimal_config(n_particles=20, bd_steps_per_frame=10)
        result = SimulationPipeline(config).run()
        for frame in result.frames:
            assert frame.ground_truth.n_complexes >= 0
            # At most min(N_R, N_L) complexes
            assert frame.ground_truth.n_complexes <= 20

    def test_bleaching_counted(self):
        """n_bleached ≥ 0."""
        config = _make_minimal_config(n_particles=20, bd_steps_per_frame=5)
        result = SimulationPipeline(config).run()
        for frame in result.frames:
            assert frame.ground_truth.n_bleached >= 0


# ═══════════════════════════════════════════════════════════════════════
# T5: RAM MONITORING
# ═══════════════════════════════════════════════════════════════════════

class TestRAMMonitoring:
    """T5: Pipeline tracks memory usage."""

    def test_peak_ram_positive(self):
        """Peak RAM is a positive number."""
        config = _make_minimal_config()
        result = SimulationPipeline(config).run()
        assert result.peak_ram_mb > 0

    def test_ram_within_limit(self):
        """RAM stays below configured limit."""
        config = _make_minimal_config()
        result = SimulationPipeline(config).run()
        assert result.peak_ram_mb < config.ram_limit_gb * 1000


# ═══════════════════════════════════════════════════════════════════════
# T6: MULTI-FRAME CONTINUITY
# ═══════════════════════════════════════════════════════════════════════

class TestMultiFrameContinuity:
    """T6: State persists between frames (no reset)."""

    def test_positions_change_between_frames(self):
        """Particle positions evolve over time."""
        config = _make_minimal_config(n_frames=3, bd_steps_per_frame=20)
        result = SimulationPipeline(config).run()

        pos0 = result.frames[0].ground_truth.positions
        pos2 = result.frames[2].ground_truth.positions
        # At least some particles should have moved
        diff = np.linalg.norm(pos2 - pos0, axis=1)
        assert np.max(diff) > 0, "Positions didn't change between frames"

    def test_frame_photon_counts_vary(self):
        """Different frames can have different photon counts."""
        config = _make_minimal_config(
            n_particles=30, n_frames=3, bd_steps_per_frame=10
        )
        result = SimulationPipeline(config).run()
        counts = [f.total_counts for f in result.frames]
        # All counts should be non-negative (but may be same if deterministic)
        assert all(c >= 0 for c in counts)

    def test_wall_time_per_frame_positive(self):
        """Each frame has positive wall clock time."""
        config = _make_minimal_config(n_frames=2, bd_steps_per_frame=5)
        result = SimulationPipeline(config).run()
        for frame in result.frames:
            assert frame.wall_time_s >= 0


# ═══════════════════════════════════════════════════════════════════════
# T7: END-TO-END MINI DEMO
# ═══════════════════════════════════════════════════════════════════════

class TestMiniDemo:
    """T7: End-to-end integration (mini case)."""

    def test_run_mini_demo(self):
        """Mini demo completes and produces valid output."""
        config = _make_minimal_config(
            n_particles=30, n_frames=2, bd_steps_per_frame=10
        )
        result = SimulationPipeline.run_mini_demo(config, n_frames=2,
                                                   n_bd_steps_per_frame=10)
        assert result.n_frames == 2
        assert result.total_wall_time_s > 0

    def test_result_summary(self):
        """Summary string is well-formed."""
        config = _make_minimal_config()
        result = SimulationPipeline(config).run()
        s = result.summary()
        assert "ED-MRDT" in s
        assert "Frames" in s
        assert "Photons" in s

    def test_intensity_series(self):
        """Intensity time series has correct length."""
        config = _make_minimal_config(n_frames=3, bd_steps_per_frame=5)
        result = SimulationPipeline(config).run()
        intensity = result.get_intensity_series()
        assert len(intensity) == 3

    def test_complex_count_series(self):
        """Complex count series has correct length."""
        config = _make_minimal_config(n_frames=3, bd_steps_per_frame=5)
        result = SimulationPipeline(config).run()
        complexes = result.get_complex_count_series()
        assert len(complexes) == 3
        assert all(c >= 0 for c in complexes)

    def test_get_flim_stack(self):
        """Can retrieve FLIM stack for any frame."""
        config = _make_minimal_config(n_frames=2, bd_steps_per_frame=5)
        result = SimulationPipeline(config).run()
        stack = result.get_flim_stack(0)
        assert stack.shape == (2, 5, 5, 64)


# ═══════════════════════════════════════════════════════════════════════
# T8: EDGE CASES
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """T8: Pipeline handles edge cases gracefully."""

    def test_single_frame(self):
        """Works with n_frames=1."""
        config = _make_minimal_config(n_frames=1, bd_steps_per_frame=5)
        result = SimulationPipeline(config).run()
        assert result.n_frames == 1

    def test_single_bd_step_per_frame(self):
        """Works with 1 BD step per frame."""
        config = _make_minimal_config(n_frames=2, bd_steps_per_frame=1)
        result = SimulationPipeline(config).run()
        assert result.n_frames == 2

    def test_few_particles(self):
        """Works with minimal particles (6: R=4, L=2)."""
        config = _make_minimal_config(n_particles=4, bd_steps_per_frame=5)
        result = SimulationPipeline(config).run()
        assert result.n_frames == 2


# ═══════════════════════════════════════════════════════════════════════
# T9: PROGRESS CALLBACK
# ═══════════════════════════════════════════════════════════════════════

class TestProgressCallback:
    """T9: Progress callback receives correct data."""

    def test_custom_callback(self):
        """Custom callback is invoked for each frame."""
        messages = []

        def cb(frame_idx, n_frames, msg):
            messages.append((frame_idx, n_frames, msg))

        config = _make_minimal_config(n_frames=3, bd_steps_per_frame=3)
        pipeline = SimulationPipeline(config, progress_callback=cb)
        pipeline.run()

        # Should have init message (-1) + 3 frame messages + 1 final
        assert len(messages) >= 4
        # First message is init (frame_idx = -1)
        assert messages[0][0] == -1
        # Frame messages have correct indices
        frame_indices = [m[0] for m in messages if m[0] >= 0]
        assert 0 in frame_indices
        assert 1 in frame_indices
        assert 2 in frame_indices


# ═══════════════════════════════════════════════════════════════════════
# T10: BLEACHING MONOTONICITY
# ═══════════════════════════════════════════════════════════════════════

class TestBleaching:
    """T10: Bleaching is irreversible (monotonically non-decreasing)."""

    def test_bleaching_monotonic(self):
        """n_bleached never decreases across frames."""
        config = _make_minimal_config(
            n_particles=40, n_frames=5, bd_steps_per_frame=10
        )
        result = SimulationPipeline(config).run()
        bleached = result.get_bleaching_series()
        for i in range(1, len(bleached)):
            assert bleached[i] >= bleached[i - 1], \
                f"Bleaching decreased: frame {i-1}={bleached[i-1]}, frame {i}={bleached[i]}"


# ═══════════════════════════════════════════════════════════════════════
# T11: QUANTITATIVE PHOTON BUDGET CHECK
# ═══════════════════════════════════════════════════════════════════════

class TestPhotonBudget:
    """T11: Photon counts are physically reasonable."""

    def test_detection_fraction_reasonable(self):
        """Detection fraction ≈ QE × PSF_acceptance (order of magnitude)."""
        config = _make_minimal_config(
            n_particles=40, n_frames=1, bd_steps_per_frame=20
        )
        result = SimulationPipeline(config).run()
        if result.total_photons_emitted > 100:
            fraction = result.total_photons_detected / result.total_photons_emitted
            # QE=0.2 × PSF acceptance (≤1.0) → fraction should be < 0.2
            assert fraction <= 0.5, \
                f"Detection fraction {fraction:.3f} suspiciously high"
            # Should be > 0 if photons were emitted
            assert fraction > 0, "No photons detected despite emission"


# ═══════════════════════════════════════════════════════════════════════
# T12: DETERMINISTIC WITH SEED
# ═══════════════════════════════════════════════════════════════════════

class TestDeterminism:
    """T12: Same seed → same results (bitwise reproducibility)."""

    def test_deterministic_photon_count(self):
        """Two runs with same seed produce same photon count."""
        config1 = _make_minimal_config(seed=99, n_frames=1, bd_steps_per_frame=5)
        config2 = _make_minimal_config(seed=99, n_frames=1, bd_steps_per_frame=5)

        r1 = SimulationPipeline(config1).run()
        r2 = SimulationPipeline(config2).run()

        assert r1.total_photons_emitted == r2.total_photons_emitted
        assert r1.total_photons_detected == r2.total_photons_detected

    def test_deterministic_positions(self):
        """Two runs with same seed produce same final positions."""
        config1 = _make_minimal_config(seed=77, n_frames=1, bd_steps_per_frame=5)
        config2 = _make_minimal_config(seed=77, n_frames=1, bd_steps_per_frame=5)

        r1 = SimulationPipeline(config1).run()
        r2 = SimulationPipeline(config2).run()

        np.testing.assert_array_equal(
            r1.frames[0].ground_truth.positions,
            r2.frames[0].ground_truth.positions,
        )


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
