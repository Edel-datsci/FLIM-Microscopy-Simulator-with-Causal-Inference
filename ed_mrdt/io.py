"""
ED-MRDT v1.1 — I/O Module (HDF5 Writer + Reader)
===================================================

Streaming HDF5 writer for simulation results. Writes frame-by-frame
so RAM never holds more than one frame at a time.

Design sources:
  - Photon-HDF5 (Ingargiola et al., 2016): FLIM data conventions
  - readPTU_FLIM (Kapusta, PicoQuant): TCSPC data layout
  - HDF5 best practices (The HDF Group): chunking, gzip compression

HDF5 Layout:
  simulation.h5
  ├── /metadata
  │   ├── config_yaml          (string — full YAML config)
  │   ├── random_seed          (int)
  │   ├── version              (string)
  │   ├── sim_name             (string)
  │   ├── n_particles          (int)
  │   ├── n_frames             (int)
  │   ├── bd_timestep          (float64)
  │   ├── frame_interval       (float64)
  │   ├── membrane_size        (float64[2])
  │   ├── species_names        (string array)
  │   └── fluorophore_names    (string array)
  │
  ├── /microscopy
  │   ├── flim_stack           (uint32[F, C, Px, Py, Tb])  — gzip
  │   ├── intensity            (float64[F, C, Px, Py])
  │   ├── bio_time             (float64[F])
  │   ├── n_signal_photons     (int32[F])
  │   ├── n_dark_counts        (int32[F])
  │   └── n_afterpulses        (int32[F])
  │
  ├── /ground_truth
  │   ├── positions            (float64[F, N, 2])
  │   ├── species_id           (int32[F, N])
  │   ├── bond_partner         (int32[F, N])
  │   ├── is_alive             (bool[F, N])
  │   ├── has_dye              (bool[F, N])
  │   ├── dye_type_id          (int32[F, N])
  │   ├── is_bleached          (bool[F, N])
  │   ├── fret_efficiency      (float64[F, N])
  │   ├── donor_lifetime       (float64[F, N])
  │   ├── n_complexes          (int32[F])
  │   ├── n_bleached           (int32[F])
  │   └── n_free_donors        (int32[F])
  │
  └── /statistics
      ├── photons_emitted      (int64[F])
      ├── photons_detected     (int64[F])
      ├── wall_time_per_frame  (float64[F])
      ├── total_wall_time      (float64)
      ├── peak_ram_mb          (float64)
      ├── total_bindings       (int64)
      └── total_dissociations  (int64)

  F = n_frames, C = n_channels, Px/Py = pixels, Tb = TCSPC bins, N = particles

Compression:
  - flim_stack: gzip level 4 (sparse uint32 histograms compress ~10:1)
  - positions: gzip level 1 (float64 compresses less)
  - scalars: no compression (tiny)

Chunking:
  - flim_stack: (1, C, Px, Py, Tb) — one frame per chunk for streaming
  - positions: (1, N, 2) — one frame per chunk

References:
  - Ingargiola A et al. (2016) Biophys J 110:26a — Photon-HDF5
  - HDF5 Group: Best Practices for Data Compression
  - Becker W (2005) Advanced TCSPC Techniques, Springer
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import yaml

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from .config import SimulationConfig
from .pipeline import SimulationResult, FrameResult, GroundTruth

__all__ = [
    "HDF5Writer",
    "write_results",
    "read_metadata",
    "read_flim_stack",
    "read_ground_truth_frame",
]


# ═══════════════════════════════════════════════════════════════════════
# HDF5 STREAMING WRITER
# ═══════════════════════════════════════════════════════════════════════

class HDF5Writer:
    """Streaming HDF5 writer for ED-MRDT simulation output.

    Writes frame-by-frame to avoid holding all frames in RAM.
    Uses gzip compression for FLIM stacks (sparse histograms).

    Usage (streaming, frame-by-frame):
        writer = HDF5Writer("output.h5", config, n_particles=600)
        writer.open()
        for frame_result in simulation:
            writer.write_frame(frame_result)
        writer.finalize(total_wall_time, peak_ram, ...)
        writer.close()

    Usage (batch, from SimulationResult):
        write_results(result, "output.h5")

    Parameters
    ----------
    path : str or Path
        Output HDF5 file path.
    config : SimulationConfig
        Simulation configuration (written to /metadata).
    n_particles : int
        Total number of particles (for pre-allocating datasets).
    compression : str
        Compression algorithm ("gzip" or None).
    compression_level : int
        Compression level (1-9, default 4).
    """

    def __init__(
        self,
        path: Union[str, Path],
        config: SimulationConfig,
        n_particles: int,
        compression: Optional[str] = "gzip",
        compression_level: int = 4,
    ):
        if not HAS_H5PY:
            raise ImportError(
                "h5py is required for HDF5 output. "
                "Install with: pip install h5py"
            )

        self.path = Path(path)
        self.config = config
        self.n_particles = n_particles
        self.compression = compression
        self.compression_level = compression_level

        self._file: Optional[h5py.File] = None
        self._frame_idx = 0
        self._is_open = False

    # ─── Open / Close ────────────────────────────────────────────────

    def open(self) -> "HDF5Writer":
        """Open HDF5 file and create dataset structure."""
        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._file = h5py.File(str(self.path), "w")
        self._is_open = True

        self._write_metadata()
        self._create_datasets()

        return self

    def close(self) -> None:
        """Close HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None
        self._is_open = False

    def __enter__(self) -> "HDF5Writer":
        return self.open()

    def __exit__(self, *args) -> None:
        self.close()

    # ─── Metadata ────────────────────────────────────────────────────

    def _write_metadata(self) -> None:
        """Write /metadata group with config and simulation parameters."""
        f = self._file
        cfg = self.config
        meta = f.create_group("metadata")

        # Serialize config to YAML string
        # We rebuild the dict from config fields for clean serialization
        config_dict = self._config_to_dict(cfg)
        config_yaml = yaml.dump(config_dict, default_flow_style=False,
                                allow_unicode=True)
        meta.create_dataset("config_yaml",
                            data=config_yaml,
                            dtype=h5py.string_dtype())

        meta.create_dataset("random_seed", data=cfg.random_seed)
        meta.create_dataset("version", data=cfg.version,
                            dtype=h5py.string_dtype())
        meta.create_dataset("sim_name", data=cfg.name,
                            dtype=h5py.string_dtype())
        meta.create_dataset("n_particles", data=self.n_particles)
        meta.create_dataset("n_frames", data=cfg.scan.n_frames)
        meta.create_dataset("bd_timestep", data=cfg.bd_timestep)
        meta.create_dataset("frame_interval", data=cfg.scan.frame_interval)
        meta.create_dataset("membrane_size",
                            data=np.array(cfg.membrane.size, dtype=np.float64))

        # Species and fluorophore names as string arrays
        sp_names = [s.name for s in cfg.species]
        dt_str = h5py.string_dtype()
        ds_sp = meta.create_dataset("species_names", shape=(len(sp_names),),
                                     dtype=dt_str)
        for i, name in enumerate(sp_names):
            ds_sp[i] = name

        fl_names = [fl.name for fl in cfg.fluorophores]
        ds_fl = meta.create_dataset("fluorophore_names", shape=(len(fl_names),),
                                     dtype=dt_str)
        for i, name in enumerate(fl_names):
            ds_fl[i] = name

    @staticmethod
    def _config_to_dict(cfg: SimulationConfig) -> dict:
        """Convert SimulationConfig to serializable dict.

        Only includes the essential fields that would appear in YAML.
        """
        d = {
            "name": cfg.name,
            "version": cfg.version,
            "membrane": {
                "size": list(cfg.membrane.size),
                "periodic": list(cfg.membrane.periodic),
                "geometry_type": cfg.membrane.geometry_type,
            },
            "simulation_time": cfg.simulation_time,
            "bd_timestep": cfg.bd_timestep,
            "random_seed": cfg.random_seed,
            "n_threads": cfg.n_threads,
            "ram_limit_gb": cfg.ram_limit_gb,
            "initial_populations": dict(cfg.initial_populations),
        }
        # Add scan parameters
        s = cfg.scan
        d["scan"] = {
            "mode": s.mode,
            "pixel_size": s.pixel_size,
            "pixels_x": s.pixels_x,
            "pixels_y": s.pixels_y,
            "dwell_time": s.dwell_time,
            "n_frames": s.n_frames,
            "frame_interval": s.frame_interval,
        }
        # Add TCSPC
        d["tcspc"] = {
            "n_bins": cfg.tcspc.n_bins,
            "time_range": cfg.tcspc.time_range,
        }
        # Add laser
        la = cfg.laser
        d["laser"] = {
            "wavelength": la.wavelength,
            "repetition_rate": la.repetition_rate,
            "power": la.power,
        }
        # Add detector
        det = cfg.detector
        d["detector"] = {
            "type": det.type,
            "detection_efficiency": det.detection_efficiency,
            "dark_count_rate": det.dark_count_rate,
        }
        return d

    # ─── Dataset creation ────────────────────────────────────────────

    def _create_datasets(self) -> None:
        """Pre-allocate resizable HDF5 datasets for streaming write."""
        f = self._file
        F = self.config.scan.n_frames
        N = self.n_particles
        C = 2  # donor + acceptor channels
        Px = self.config.scan.pixels_x
        Py = self.config.scan.pixels_y
        Tb = self.config.tcspc.n_bins

        comp = self.compression
        clev = self.compression_level

        # Compression kwargs
        ckw_heavy = dict(compression=comp, compression_opts=clev) if comp else {}
        ckw_light = dict(compression=comp, compression_opts=1) if comp else {}

        # ── /microscopy ──────────────────────────────────────────────
        mic = f.create_group("microscopy")

        mic.create_dataset(
            "flim_stack",
            shape=(F, C, Px, Py, Tb),
            dtype=np.uint32,
            chunks=(1, C, Px, Py, Tb),
            **ckw_heavy,
        )
        mic.create_dataset(
            "intensity",
            shape=(F, C, Px, Py),
            dtype=np.float64,
            chunks=(1, C, Px, Py),
            **ckw_light,
        )
        mic.create_dataset("bio_time", shape=(F,), dtype=np.float64)
        mic.create_dataset("n_signal_photons", shape=(F,), dtype=np.int32)
        mic.create_dataset("n_dark_counts", shape=(F,), dtype=np.int32)
        mic.create_dataset("n_afterpulses", shape=(F,), dtype=np.int32)

        # ── /ground_truth ────────────────────────────────────────────
        gt = f.create_group("ground_truth")

        gt.create_dataset(
            "positions",
            shape=(F, N, 2),
            dtype=np.float64,
            chunks=(1, N, 2),
            **ckw_light,
        )
        gt.create_dataset(
            "species_id",
            shape=(F, N),
            dtype=np.int32,
            chunks=(1, N),
        )
        gt.create_dataset(
            "bond_partner",
            shape=(F, N),
            dtype=np.int32,
            chunks=(1, N),
        )
        gt.create_dataset("is_alive", shape=(F, N), dtype=np.bool_,
                          chunks=(1, N))
        gt.create_dataset("has_dye", shape=(F, N), dtype=np.bool_,
                          chunks=(1, N))
        gt.create_dataset("dye_type_id", shape=(F, N), dtype=np.int32,
                          chunks=(1, N))
        gt.create_dataset("is_bleached", shape=(F, N), dtype=np.bool_,
                          chunks=(1, N))
        gt.create_dataset(
            "fret_efficiency",
            shape=(F, N),
            dtype=np.float64,
            chunks=(1, N),
            **ckw_light,
        )
        gt.create_dataset(
            "donor_lifetime",
            shape=(F, N),
            dtype=np.float64,
            chunks=(1, N),
            **ckw_light,
        )
        gt.create_dataset("n_complexes", shape=(F,), dtype=np.int32)
        gt.create_dataset("n_bleached", shape=(F,), dtype=np.int32)
        gt.create_dataset("n_free_donors", shape=(F,), dtype=np.int32)

        # ── /statistics ──────────────────────────────────────────────
        stats = f.create_group("statistics")

        stats.create_dataset("photons_emitted", shape=(F,), dtype=np.int64)
        stats.create_dataset("photons_detected", shape=(F,), dtype=np.int64)
        stats.create_dataset("wall_time_per_frame", shape=(F,), dtype=np.float64)

        # Scalars (written at finalize)
        stats.create_dataset("total_wall_time", data=0.0)
        stats.create_dataset("peak_ram_mb", data=0.0)
        stats.create_dataset("total_bindings", data=np.int64(0))
        stats.create_dataset("total_dissociations", data=np.int64(0))

    # ─── Frame write ─────────────────────────────────────────────────

    def write_frame(self, frame: FrameResult) -> None:
        """Write one frame to HDF5 (streaming).

        Parameters
        ----------
        frame : FrameResult
            Frame from SimulationPipeline.
        """
        if not self._is_open:
            raise RuntimeError("HDF5Writer is not open. Call open() first.")

        i = self._frame_idx
        f = self._file

        # ── Microscopy ───────────────────────────────────────────────
        fl = frame.flim_frame
        f["microscopy/flim_stack"][i] = fl.flim_stack
        f["microscopy/intensity"][i] = fl.intensity
        f["microscopy/bio_time"][i] = frame.bio_time
        f["microscopy/n_signal_photons"][i] = fl.n_signal_photons
        f["microscopy/n_dark_counts"][i] = fl.n_dark_counts
        f["microscopy/n_afterpulses"][i] = fl.n_afterpulses

        # ── Ground truth ─────────────────────────────────────────────
        gt = frame.ground_truth
        f["ground_truth/positions"][i] = gt.positions
        f["ground_truth/species_id"][i] = gt.species_id
        f["ground_truth/bond_partner"][i] = gt.bond_partner
        f["ground_truth/is_alive"][i] = gt.is_alive
        f["ground_truth/has_dye"][i] = gt.has_dye
        f["ground_truth/dye_type_id"][i] = gt.dye_type_id
        f["ground_truth/is_bleached"][i] = gt.is_bleached
        f["ground_truth/fret_efficiency"][i] = gt.fret_efficiency_exact
        f["ground_truth/donor_lifetime"][i] = gt.donor_lifetime_exact
        f["ground_truth/n_complexes"][i] = gt.n_complexes
        f["ground_truth/n_bleached"][i] = gt.n_bleached
        f["ground_truth/n_free_donors"][i] = gt.n_free_donors

        # ── Statistics ───────────────────────────────────────────────
        f["statistics/photons_emitted"][i] = frame.n_photons_emitted
        f["statistics/photons_detected"][i] = frame.n_photons_detected
        f["statistics/wall_time_per_frame"][i] = frame.wall_time_s

        # Flush to disk (ensures data survives crashes)
        f.flush()

        self._frame_idx += 1

    # ─── Finalize ────────────────────────────────────────────────────

    def finalize(
        self,
        total_wall_time: float = 0.0,
        peak_ram_mb: float = 0.0,
        total_bindings: int = 0,
        total_dissociations: int = 0,
    ) -> None:
        """Write final summary statistics.

        Parameters
        ----------
        total_wall_time : float
            Total wall clock time in seconds.
        peak_ram_mb : float
            Peak RAM usage in MB.
        total_bindings : int
            Total binding events across all frames.
        total_dissociations : int
            Total dissociation events.
        """
        if not self._is_open:
            return

        f = self._file
        f["statistics/total_wall_time"][()] = total_wall_time
        f["statistics/peak_ram_mb"][()] = peak_ram_mb
        f["statistics/total_bindings"][()] = np.int64(total_bindings)
        f["statistics/total_dissociations"][()] = np.int64(total_dissociations)

        # Write actual number of frames written (may differ from planned
        # if simulation was interrupted)
        if "frames_written" not in f["metadata"]:
            f["metadata"].create_dataset("frames_written",
                                          data=self._frame_idx)
        else:
            f["metadata/frames_written"][()] = self._frame_idx

        f.flush()

    @property
    def frames_written(self) -> int:
        return self._frame_idx


# ═══════════════════════════════════════════════════════════════════════
# BATCH WRITE CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def write_results(
    result: SimulationResult,
    path: Union[str, Path],
    compression: Optional[str] = "gzip",
    compression_level: int = 4,
) -> Path:
    """Write a complete SimulationResult to HDF5.

    Convenience function for batch writing (all frames at once).

    Parameters
    ----------
    result : SimulationResult
        Complete simulation output from pipeline.run().
    path : str or Path
        Output HDF5 file path.
    compression : str, optional
        Compression algorithm ("gzip" or None).
    compression_level : int
        gzip level (1-9).

    Returns
    -------
    Path
        Path to written file.
    """
    if not result.frames:
        raise ValueError("SimulationResult has no frames to write")

    N = result.frames[0].ground_truth.positions.shape[0]
    path = Path(path)

    writer = HDF5Writer(
        path, result.config, N,
        compression=compression,
        compression_level=compression_level,
    )

    with writer:
        for frame in result.frames:
            writer.write_frame(frame)
        writer.finalize(
            total_wall_time=result.total_wall_time_s,
            peak_ram_mb=result.peak_ram_mb,
            total_bindings=result.total_bindings,
            total_dissociations=result.total_dissociations,
        )

    return path


# ═══════════════════════════════════════════════════════════════════════
# READ-BACK FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def read_metadata(path: Union[str, Path]) -> dict:
    """Read metadata from an ED-MRDT HDF5 file.

    Returns
    -------
    dict
        Metadata including config_yaml, seed, version, etc.
    """
    if not HAS_H5PY:
        raise ImportError("h5py required")

    info = {}
    with h5py.File(str(path), "r") as f:
        meta = f["metadata"]
        info["config_yaml"] = meta["config_yaml"][()].decode()
        info["random_seed"] = int(meta["random_seed"][()])
        info["version"] = meta["version"][()].decode()
        info["sim_name"] = meta["sim_name"][()].decode()
        info["n_particles"] = int(meta["n_particles"][()])
        info["n_frames"] = int(meta["n_frames"][()])
        info["bd_timestep"] = float(meta["bd_timestep"][()])
        info["frame_interval"] = float(meta["frame_interval"][()])
        info["membrane_size"] = meta["membrane_size"][()].tolist()
        info["species_names"] = [s.decode() for s in meta["species_names"][()]]
        info["fluorophore_names"] = [s.decode() for s in meta["fluorophore_names"][()]]
        if "frames_written" in meta:
            info["frames_written"] = int(meta["frames_written"][()])

        # Statistics summary
        if "statistics" in f:
            stats = f["statistics"]
            info["total_wall_time"] = float(stats["total_wall_time"][()])
            info["peak_ram_mb"] = float(stats["peak_ram_mb"][()])
            info["total_bindings"] = int(stats["total_bindings"][()])
            info["total_dissociations"] = int(stats["total_dissociations"][()])

    return info


def read_flim_stack(
    path: Union[str, Path],
    frame_idx: int,
) -> np.ndarray:
    """Read FLIM stack for one frame.

    Parameters
    ----------
    path : str or Path
        HDF5 file path.
    frame_idx : int
        Frame index.

    Returns
    -------
    ndarray[C, Px, Py, Tb]
        FLIM TCSPC histogram stack.
    """
    if not HAS_H5PY:
        raise ImportError("h5py required")

    with h5py.File(str(path), "r") as f:
        return f["microscopy/flim_stack"][frame_idx].copy()


def read_ground_truth_frame(
    path: Union[str, Path],
    frame_idx: int,
) -> dict:
    """Read ground truth for one frame.

    Returns
    -------
    dict
        Ground truth arrays: positions, species_id, bond_partner,
        fret_efficiency, donor_lifetime, n_complexes, etc.
    """
    if not HAS_H5PY:
        raise ImportError("h5py required")

    gt = {}
    with h5py.File(str(path), "r") as f:
        g = f["ground_truth"]
        gt["positions"] = g["positions"][frame_idx].copy()
        gt["species_id"] = g["species_id"][frame_idx].copy()
        gt["bond_partner"] = g["bond_partner"][frame_idx].copy()
        gt["is_alive"] = g["is_alive"][frame_idx].copy()
        gt["has_dye"] = g["has_dye"][frame_idx].copy()
        gt["dye_type_id"] = g["dye_type_id"][frame_idx].copy()
        gt["is_bleached"] = g["is_bleached"][frame_idx].copy()
        gt["fret_efficiency"] = g["fret_efficiency"][frame_idx].copy()
        gt["donor_lifetime"] = g["donor_lifetime"][frame_idx].copy()
        gt["n_complexes"] = int(g["n_complexes"][frame_idx])
        gt["n_bleached"] = int(g["n_bleached"][frame_idx])
        gt["n_free_donors"] = int(g["n_free_donors"][frame_idx])

    return gt


def read_intensity_series(path: Union[str, Path]) -> np.ndarray:
    """Read total intensity per frame.

    Returns
    -------
    ndarray[F]
        Total photon counts per frame.
    """
    if not HAS_H5PY:
        raise ImportError("h5py required")

    with h5py.File(str(path), "r") as f:
        intensity = f["microscopy/intensity"][()]
        return intensity.sum(axis=(1, 2, 3))  # sum over channels, px, py


def read_statistics(path: Union[str, Path]) -> dict:
    """Read per-frame statistics.

    Returns
    -------
    dict
        Keys: photons_emitted, photons_detected, wall_time_per_frame, etc.
    """
    if not HAS_H5PY:
        raise ImportError("h5py required")

    stats = {}
    with h5py.File(str(path), "r") as f:
        g = f["statistics"]
        stats["photons_emitted"] = g["photons_emitted"][()].copy()
        stats["photons_detected"] = g["photons_detected"][()].copy()
        stats["wall_time_per_frame"] = g["wall_time_per_frame"][()].copy()
        stats["total_wall_time"] = float(g["total_wall_time"][()])
        stats["peak_ram_mb"] = float(g["peak_ram_mb"][()])
        stats["total_bindings"] = int(g["total_bindings"][()])
        stats["total_dissociations"] = int(g["total_dissociations"][()])

    return stats
