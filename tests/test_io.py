"""
ED-MRDT v1.1 — I/O Tests (Step 4.2)
======================================

Quantitative tests for HDF5 streaming writer and readers.

Test categories:
  T1: Write + read-back roundtrip (data integrity)
  T2: Streaming write (frame-by-frame)
  T3: Compression (gzip reduces file size)
  T4: Metadata correctness
  T5: Ground truth read-back matches original
  T6: FLIM stack read-back matches original
  T7: Statistics read-back
  T8: Multiple frames sequential consistency
  T9: Context manager (with statement)
  T10: Convenience functions (write_results, read_*)
  T11: Edge cases (1 frame, 1 particle)
  T12: File size reasonable for canonical dimensions
"""

import math
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ed_mrdt.config import (
    SimulationConfig, MembraneGeometry, Species, Reaction,
    PairPotential, DiffusionField, FluorophoreType, PhotoState,
    FRETpair, DyeAttachment, ObjectiveLens, LaserSource,
    Detector, ScanParameters, TCSPCsettings, PSFmodel,
)
from ed_mrdt.pipeline import (
    SimulationPipeline, SimulationResult, FrameResult, GroundTruth,
)
from ed_mrdt.io import (
    HDF5Writer, write_results,
    read_metadata, read_flim_stack, read_ground_truth_frame,
    read_intensity_series, read_statistics,
)


# ═══════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════

def _make_config(n_particles=20, n_frames=3, bd_per_frame=5, dt=1e-4):
    """Minimal config for fast I/O tests."""
    frame_interval = bd_per_frame * dt
    config = SimulationConfig(
        name="io_test",
        membrane=MembraneGeometry(size=(0.5, 0.5), periodic=(True, True)),
        diffusion_field=DiffusionField(type="uniform"),
        species=[
            Species(name="R", diffusion_coefficient=0.05, radius=5.0),
            Species(name="L", diffusion_coefficient=1.0, radius=2.0),
            Species(name="RL", diffusion_coefficient=0.04, radius=6.0),
        ],
        reactions=[
            Reaction(name="bind", reactants=["R", "L"], products=["RL"],
                     kon=0.1, koff=0.01, contact_radius=7.0),
        ],
        potentials=[
            PairPotential(species_a="R", species_b="R",
                          potential_type="wca",
                          parameters={"sigma": 8.0, "epsilon": 2.5}),
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
                absorption_cross_section=3e-16, tau_rad=4.1e-9,
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
                absorption_cross_section=1.5e-16, tau_rad=3e-10,
            ),
        ],
        fret_pairs=[FRETpair(donor_type="donor", acceptor_type="acceptor", R0=6.4)],
        dye_attachments=[
            DyeAttachment(species="R", fluorophore="donor", labeling_efficiency=0.8),
            DyeAttachment(species="L", fluorophore="acceptor", labeling_efficiency=0.9),
        ],
        objective=ObjectiveLens(NA=1.4),
        laser=LaserSource(wavelength=488.0, repetition_rate=40e6,
                          power=5e-6, beam_waist=250.0),
        detector=Detector(detection_efficiency=0.2, dark_count_rate=100.0,
                          timing_jitter=50e-12, afterpulsing_prob=0.005),
        scan=ScanParameters(pixel_size=100.0, pixels_x=5, pixels_y=5,
                            dwell_time=20e-6, n_frames=n_frames,
                            frame_interval=frame_interval),
        tcspc=TCSPCsettings(n_bins=64, time_range=25e-9),
        psf=PSFmodel(type="gaussian", emission_wavelength=519.0),
        simulation_time=n_frames * frame_interval,
        bd_timestep=dt, random_seed=42,
    )
    config.validate()
    return config


def _run_simulation(config):
    """Run pipeline and return result."""
    return SimulationPipeline(config).run()


@pytest.fixture
def sim_result():
    """Pre-computed simulation result for tests."""
    config = _make_config(n_particles=20, n_frames=3, bd_per_frame=5)
    return _run_simulation(config)


@pytest.fixture
def h5_path(tmp_path):
    """Temporary HDF5 file path."""
    return tmp_path / "test_output.h5"


# ═══════════════════════════════════════════════════════════════════════
# T1: ROUNDTRIP (WRITE → READ)
# ═══════════════════════════════════════════════════════════════════════

class TestRoundtrip:
    """T1: Write results and read them back, verify data integrity."""

    def test_write_and_read_metadata(self, sim_result, h5_path):
        """Metadata round-trips correctly."""
        write_results(sim_result, h5_path)
        meta = read_metadata(h5_path)

        assert meta["sim_name"] == "io_test"
        assert meta["version"] == "1.1.0"
        assert meta["random_seed"] == 42
        assert meta["n_frames"] == 3
        assert meta["n_particles"] == 30  # 20 + 10
        assert meta["frames_written"] == 3

    def test_write_and_read_flim_stack(self, sim_result, h5_path):
        """FLIM stack round-trips with exact equality."""
        write_results(sim_result, h5_path)

        for i, frame in enumerate(sim_result.frames):
            stack_read = read_flim_stack(h5_path, i)
            np.testing.assert_array_equal(
                stack_read, frame.flim_frame.flim_stack,
                err_msg=f"FLIM stack mismatch at frame {i}",
            )

    def test_write_and_read_ground_truth(self, sim_result, h5_path):
        """Ground truth arrays round-trip with exact equality."""
        write_results(sim_result, h5_path)

        for i, frame in enumerate(sim_result.frames):
            gt = read_ground_truth_frame(h5_path, i)
            gt_orig = frame.ground_truth

            np.testing.assert_array_equal(
                gt["positions"], gt_orig.positions)
            np.testing.assert_array_equal(
                gt["species_id"], gt_orig.species_id)
            np.testing.assert_array_equal(
                gt["bond_partner"], gt_orig.bond_partner)
            np.testing.assert_array_equal(
                gt["fret_efficiency"], gt_orig.fret_efficiency_exact)
            np.testing.assert_array_equal(
                gt["donor_lifetime"], gt_orig.donor_lifetime_exact)
            assert gt["n_complexes"] == gt_orig.n_complexes
            assert gt["n_bleached"] == gt_orig.n_bleached

    def test_write_and_read_statistics(self, sim_result, h5_path):
        """Statistics round-trip correctly."""
        write_results(sim_result, h5_path)
        stats = read_statistics(h5_path)

        assert stats["total_wall_time"] == pytest.approx(
            sim_result.total_wall_time_s, rel=1e-6)
        assert stats["total_bindings"] == sim_result.total_bindings
        assert stats["total_dissociations"] == sim_result.total_dissociations

        for i, frame in enumerate(sim_result.frames):
            assert stats["photons_emitted"][i] == frame.n_photons_emitted
            assert stats["photons_detected"][i] == frame.n_photons_detected


# ═══════════════════════════════════════════════════════════════════════
# T2: STREAMING WRITE
# ═══════════════════════════════════════════════════════════════════════

class TestStreamingWrite:
    """T2: Frame-by-frame streaming write works correctly."""

    def test_streaming_matches_batch(self, sim_result, tmp_path):
        """Streaming write produces same file as batch write."""
        path_batch = tmp_path / "batch.h5"
        path_stream = tmp_path / "stream.h5"

        # Batch
        write_results(sim_result, path_batch)

        # Streaming
        N = sim_result.frames[0].ground_truth.positions.shape[0]
        writer = HDF5Writer(path_stream, sim_result.config, N)
        writer.open()
        for frame in sim_result.frames:
            writer.write_frame(frame)
        writer.finalize(
            total_wall_time=sim_result.total_wall_time_s,
            peak_ram_mb=sim_result.peak_ram_mb,
            total_bindings=sim_result.total_bindings,
            total_dissociations=sim_result.total_dissociations,
        )
        writer.close()

        # Compare FLIM stacks
        for i in range(sim_result.n_frames):
            s1 = read_flim_stack(path_batch, i)
            s2 = read_flim_stack(path_stream, i)
            np.testing.assert_array_equal(s1, s2)

    def test_frames_written_counter(self, sim_result, h5_path):
        """frames_written tracks the correct count."""
        N = sim_result.frames[0].ground_truth.positions.shape[0]
        writer = HDF5Writer(h5_path, sim_result.config, N)
        writer.open()
        assert writer.frames_written == 0
        writer.write_frame(sim_result.frames[0])
        assert writer.frames_written == 1
        writer.write_frame(sim_result.frames[1])
        assert writer.frames_written == 2
        writer.finalize()
        writer.close()


# ═══════════════════════════════════════════════════════════════════════
# T3: COMPRESSION
# ═══════════════════════════════════════════════════════════════════════

class TestCompression:
    """T3: gzip compression reduces file size."""

    def test_gzip_smaller_than_uncompressed(self, sim_result, tmp_path):
        """gzip file is smaller than uncompressed."""
        path_gz = tmp_path / "compressed.h5"
        path_raw = tmp_path / "raw.h5"

        write_results(sim_result, path_gz, compression="gzip",
                      compression_level=4)
        write_results(sim_result, path_raw, compression=None)

        size_gz = os.path.getsize(path_gz)
        size_raw = os.path.getsize(path_raw)

        assert size_gz < size_raw, \
            f"Compressed ({size_gz}) not smaller than raw ({size_raw})"

    def test_compressed_data_intact(self, sim_result, tmp_path):
        """Data reads back correctly after compression."""
        path = tmp_path / "comp.h5"
        write_results(sim_result, path, compression="gzip",
                      compression_level=9)

        for i in range(sim_result.n_frames):
            stack = read_flim_stack(path, i)
            np.testing.assert_array_equal(
                stack, sim_result.frames[i].flim_frame.flim_stack)


# ═══════════════════════════════════════════════════════════════════════
# T4: METADATA
# ═══════════════════════════════════════════════════════════════════════

class TestMetadata:
    """T4: All metadata fields written and readable."""

    def test_species_names(self, sim_result, h5_path):
        write_results(sim_result, h5_path)
        meta = read_metadata(h5_path)
        assert meta["species_names"] == ["R", "L", "RL"]

    def test_fluorophore_names(self, sim_result, h5_path):
        write_results(sim_result, h5_path)
        meta = read_metadata(h5_path)
        assert meta["fluorophore_names"] == ["donor", "acceptor"]

    def test_membrane_size(self, sim_result, h5_path):
        write_results(sim_result, h5_path)
        meta = read_metadata(h5_path)
        assert meta["membrane_size"] == [0.5, 0.5]

    def test_config_yaml_parseable(self, sim_result, h5_path):
        """Stored YAML can be parsed back."""
        write_results(sim_result, h5_path)
        meta = read_metadata(h5_path)
        import yaml
        parsed = yaml.safe_load(meta["config_yaml"])
        assert parsed["name"] == "io_test"
        assert parsed["bd_timestep"] == 1e-4

    def test_bd_timestep(self, sim_result, h5_path):
        write_results(sim_result, h5_path)
        meta = read_metadata(h5_path)
        assert meta["bd_timestep"] == pytest.approx(1e-4)


# ═══════════════════════════════════════════════════════════════════════
# T5: GROUND TRUTH DETAIL
# ═══════════════════════════════════════════════════════════════════════

class TestGroundTruthDetail:
    """T5: Ground truth arrays have correct shapes and values."""

    def test_positions_shape(self, sim_result, h5_path):
        write_results(sim_result, h5_path)
        gt = read_ground_truth_frame(h5_path, 0)
        N = sim_result.frames[0].ground_truth.positions.shape[0]
        assert gt["positions"].shape == (N, 2)

    def test_boolean_arrays(self, sim_result, h5_path):
        write_results(sim_result, h5_path)
        gt = read_ground_truth_frame(h5_path, 0)
        assert gt["is_alive"].dtype == np.bool_
        assert gt["has_dye"].dtype == np.bool_
        assert gt["is_bleached"].dtype == np.bool_

    def test_fret_bounded(self, sim_result, h5_path):
        write_results(sim_result, h5_path)
        for i in range(sim_result.n_frames):
            gt = read_ground_truth_frame(h5_path, i)
            assert np.all(gt["fret_efficiency"] >= 0)
            assert np.all(gt["fret_efficiency"] <= 1)


# ═══════════════════════════════════════════════════════════════════════
# T6: FLIM STACK DETAIL
# ═══════════════════════════════════════════════════════════════════════

class TestFLIMStack:
    """T6: FLIM stack arrays correct."""

    def test_flim_stack_shape(self, sim_result, h5_path):
        write_results(sim_result, h5_path)
        stack = read_flim_stack(h5_path, 0)
        assert stack.shape == (2, 5, 5, 64)  # C, Px, Py, Tb

    def test_flim_stack_dtype(self, sim_result, h5_path):
        write_results(sim_result, h5_path)
        stack = read_flim_stack(h5_path, 0)
        assert stack.dtype == np.uint32

    def test_intensity_series(self, sim_result, h5_path):
        write_results(sim_result, h5_path)
        intensity = read_intensity_series(h5_path)
        assert len(intensity) == 3
        assert all(i >= 0 for i in intensity)


# ═══════════════════════════════════════════════════════════════════════
# T7: STATISTICS
# ═══════════════════════════════════════════════════════════════════════

class TestStatistics:
    """T7: Statistics arrays correct."""

    def test_photon_arrays_length(self, sim_result, h5_path):
        write_results(sim_result, h5_path)
        stats = read_statistics(h5_path)
        assert len(stats["photons_emitted"]) == 3
        assert len(stats["photons_detected"]) == 3

    def test_photons_emitted_positive(self, sim_result, h5_path):
        write_results(sim_result, h5_path)
        stats = read_statistics(h5_path)
        assert all(p >= 0 for p in stats["photons_emitted"])

    def test_peak_ram_positive(self, sim_result, h5_path):
        write_results(sim_result, h5_path)
        stats = read_statistics(h5_path)
        assert stats["peak_ram_mb"] > 0


# ═══════════════════════════════════════════════════════════════════════
# T8: MULTI-FRAME CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════

class TestMultiFrame:
    """T8: Multiple frames are independent and sequential."""

    def test_bio_time_increasing(self, sim_result, h5_path):
        """Bio times are strictly increasing across frames."""
        write_results(sim_result, h5_path)
        import h5py
        with h5py.File(str(h5_path), "r") as f:
            times = f["microscopy/bio_time"][()]
        for i in range(1, len(times)):
            assert times[i] > times[i - 1]

    def test_different_frames_different_data(self, sim_result, h5_path):
        """Frames contain different position data (dynamics evolve)."""
        write_results(sim_result, h5_path)
        gt0 = read_ground_truth_frame(h5_path, 0)
        gt2 = read_ground_truth_frame(h5_path, 2)
        # Positions should differ (particles moved)
        assert not np.array_equal(gt0["positions"], gt2["positions"])


# ═══════════════════════════════════════════════════════════════════════
# T9: CONTEXT MANAGER
# ═══════════════════════════════════════════════════════════════════════

class TestContextManager:
    """T9: With-statement properly opens and closes."""

    def test_with_statement(self, sim_result, h5_path):
        N = sim_result.frames[0].ground_truth.positions.shape[0]
        with HDF5Writer(h5_path, sim_result.config, N) as w:
            w.write_frame(sim_result.frames[0])
            w.finalize()
        # File should be closed and readable
        meta = read_metadata(h5_path)
        assert meta["sim_name"] == "io_test"


# ═══════════════════════════════════════════════════════════════════════
# T10: EDGE CASES
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """T10: Edge cases handled gracefully."""

    def test_single_frame(self, tmp_path):
        config = _make_config(n_particles=10, n_frames=1, bd_per_frame=3)
        result = _run_simulation(config)
        path = tmp_path / "single.h5"
        write_results(result, path)
        meta = read_metadata(path)
        assert meta["frames_written"] == 1

    def test_no_compression(self, sim_result, tmp_path):
        path = tmp_path / "nocomp.h5"
        write_results(sim_result, path, compression=None)
        stack = read_flim_stack(path, 0)
        np.testing.assert_array_equal(
            stack, sim_result.frames[0].flim_frame.flim_stack)

    def test_empty_result_raises(self, tmp_path):
        """Writing empty result raises ValueError."""
        config = _make_config()
        empty_result = SimulationResult(config=config, frames=[])
        with pytest.raises(ValueError):
            write_results(empty_result, tmp_path / "empty.h5")


# ═══════════════════════════════════════════════════════════════════════
# T11: FILE SIZE
# ═══════════════════════════════════════════════════════════════════════

class TestFileSize:
    """T11: File size is reasonable."""

    def test_compressed_file_size(self, sim_result, h5_path):
        """File < 1 MB for mini simulation (5×5×64 TCSPC, 3 frames)."""
        write_results(sim_result, h5_path)
        size_bytes = os.path.getsize(h5_path)
        size_kb = size_bytes / 1024
        assert size_kb < 1024, f"File too large: {size_kb:.0f} KB"
        print(f"  File size: {size_kb:.1f} KB")

    def test_canonical_size_estimate(self):
        """Estimate canonical case file size: 60 frames × 21×21×256 × uint32.

        Raw data per frame:
          flim_stack: 2 × 21 × 21 × 256 × 4 bytes = 903 KB
          positions:  600 × 2 × 8 = 9.4 KB
          ground_truth scalars: ~15 KB
        Per frame: ~928 KB raw
        60 frames: ~54 MB raw
        With gzip ~10:1 on sparse TCSPC: ~5.4 MB
        Should be < 100 MB easily.
        """
        frame_raw = 2 * 21 * 21 * 256 * 4 + 600 * 2 * 8 + 600 * 50
        total_raw = 60 * frame_raw
        assert total_raw < 200e6, f"Raw estimate too large: {total_raw/1e6:.0f} MB"
        print(f"  Canonical raw estimate: {total_raw/1e6:.1f} MB")
        print(f"  Expected with gzip: ~{total_raw/1e6/5:.1f} MB")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
