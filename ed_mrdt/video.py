"""
ED-MRDT Video Generator
========================
Generates animated videos from FLIM simulation results.

Two modalities per replica:
  1. Fluorescence Intensity: total photon counts per pixel per frame
  2. FLIM Lifetime (tau): mean arrival time per pixel per frame

Output formats:
  - GIF (always available via Pillow)
  - MP4 (requires ffmpeg)

Usage:
    from ed_mrdt.video import generate_replica_videos
    paths = generate_replica_videos(sim_result, replica_idx=0, output_dir="videos/")
    # Returns dict with paths: {"intensity": "...", "flim": "..."}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter

# Try ffmpeg for MP4
try:
    from matplotlib.animation import FFMpegWriter
    _HAS_FFMPEG = FFMpegWriter.isAvailable()
except Exception:
    _HAS_FFMPEG = False


# ═══════════════════════════════════════════════════════════════════════
# Core frame extraction
# ═══════════════════════════════════════════════════════════════════════

def extract_intensity_frames(sim_result, donor_channel: int = 0) -> np.ndarray:
    """Extract intensity map per frame: sum over TCSPC bins.

    Returns
    -------
    intensity_stack : ndarray[n_frames, nx, ny]
        Total photon counts per pixel per frame.
    """
    n_frames = sim_result.n_frames
    stack0 = sim_result.get_flim_stack(0)
    nx, ny = stack0.shape[1], stack0.shape[2]
    intensity = np.zeros((n_frames, nx, ny))

    for fi in range(n_frames):
        flim_stack = sim_result.get_flim_stack(fi)  # [n_ch, nx, ny, n_bins]
        intensity[fi] = flim_stack[donor_channel].sum(axis=-1)

    return intensity


def extract_lifetime_frames(
    sim_result,
    donor_channel: int = 0,
    min_counts: int = 3,
) -> Tuple[np.ndarray, float]:
    """Extract lifetime (tau) map per frame: mean arrival time.

    Returns
    -------
    tau_stack : ndarray[n_frames, nx, ny]
        Mean arrival time in nanoseconds. NaN where counts < min_counts.
    time_range_ns : float
        TCSPC time range in nanoseconds.
    """
    n_frames = sim_result.n_frames
    config = sim_result.config
    time_range = config.tcspc.time_range  # seconds
    time_range_ns = time_range * 1e9

    stack0 = sim_result.get_flim_stack(0)
    nx, ny = stack0.shape[1], stack0.shape[2]
    n_bins = stack0.shape[3]
    bin_centers_ns = (np.arange(n_bins) + 0.5) * (time_range_ns / n_bins)

    tau_stack = np.full((n_frames, nx, ny), np.nan)

    for fi in range(n_frames):
        flim_stack = sim_result.get_flim_stack(fi)
        donor_stack = flim_stack[donor_channel]  # [nx, ny, n_bins]
        total = donor_stack.sum(axis=-1)  # [nx, ny]
        valid = total >= min_counts
        if valid.any():
            weighted = np.tensordot(donor_stack, bin_centers_ns, axes=([-1], [0]))
            tau_stack[fi] = np.where(valid, weighted / np.maximum(total, 1), np.nan)

    return tau_stack, time_range_ns


# ═══════════════════════════════════════════════════════════════════════
# Video rendering
# ═══════════════════════════════════════════════════════════════════════

def _make_animation(
    frames: np.ndarray,
    title_template: str,
    cmap: str,
    vmin: float,
    vmax: float,
    cbar_label: str,
    frame_interval_s: float,
    pixel_size_um: float,
    fps: int = 4,
    dpi: int = 150,
    figsize: Tuple[float, float] = (5.5, 5.0),
) -> FuncAnimation:
    """Create matplotlib animation from a stack of 2D frames."""
    n_frames, nx, ny = frames.shape
    extent_um = [0, ny * pixel_size_um, nx * pixel_size_um, 0]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.subplots_adjust(right=0.85, top=0.92, bottom=0.08)

    im = ax.imshow(frames[0], cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent_um, interpolation='nearest', aspect='equal')

    ax.set_xlabel('x (um)', fontsize=9)
    ax.set_ylabel('y (um)', fontsize=9)
    ax.tick_params(labelsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    title = ax.set_title('', fontsize=10, fontweight='bold')

    def update(fi):
        im.set_data(frames[fi])
        bio_time = (fi + 1) * frame_interval_s
        title.set_text(title_template.format(frame=fi + 1, n=n_frames,
                                              time=bio_time))
        return [im, title]

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 // fps, blit=False)
    return anim, fig


def render_intensity_video(
    sim_result,
    output_path: str,
    donor_channel: int = 0,
    fps: int = 4,
    dpi: int = 150,
) -> str:
    """Render fluorescence intensity video.

    Parameters
    ----------
    sim_result : SimulationResult
    output_path : str
        Output file path (.gif or .mp4)
    donor_channel : int
        Channel index for donor fluorophore.
    fps : int
        Frames per second in output video.
    dpi : int
        Resolution.

    Returns
    -------
    path : str
        Absolute path to saved video file.
    """
    intensity = extract_intensity_frames(sim_result, donor_channel)
    config = sim_result.config
    pixel_size_um = config.scan.pixel_size * 1e-3  # nm -> um
    frame_interval = config.scan.frame_interval

    # Auto scale: 0 to 95th percentile
    vmax = np.percentile(intensity[intensity > 0], 95) if (intensity > 0).any() else 1.0
    vmin = 0.0

    anim, fig = _make_animation(
        frames=intensity,
        title_template="Fluorescence Intensity | Frame {frame}/{n} | t={time:.2f}s",
        cmap='hot',
        vmin=vmin,
        vmax=vmax,
        cbar_label='Photon counts',
        frame_interval_s=frame_interval,
        pixel_size_um=pixel_size_um,
        fps=fps,
        dpi=dpi,
    )

    _save_animation(anim, output_path, fps, dpi)
    plt.close(fig)
    return os.path.abspath(output_path)


def render_flim_video(
    sim_result,
    output_path: str,
    donor_channel: int = 0,
    tau_d_ns: float = 4.1,
    min_counts: int = 3,
    fps: int = 4,
    dpi: int = 150,
) -> str:
    """Render FLIM lifetime video.

    Parameters
    ----------
    sim_result : SimulationResult
    output_path : str
        Output file path (.gif or .mp4)
    donor_channel : int
        Channel index for donor fluorophore.
    tau_d_ns : float
        Unquenched donor lifetime (ns) for color scale reference.
    min_counts : int
        Minimum photon counts per pixel to compute tau.
    fps : int
    dpi : int

    Returns
    -------
    path : str
        Absolute path to saved video file.
    """
    tau_stack, time_range_ns = extract_lifetime_frames(
        sim_result, donor_channel, min_counts)
    config = sim_result.config
    pixel_size_um = config.scan.pixel_size * 1e-3
    frame_interval = config.scan.frame_interval

    # Replace NaN with 0 for display (will show as black)
    tau_display = np.nan_to_num(tau_stack, nan=0.0)

    # Color scale: 0 to tau_d (unquenched lifetime)
    vmin = 0.0
    vmax = tau_d_ns * 1.2  # slight headroom

    # Custom colormap: jet-like but with black for zero (no data)
    cmap = plt.cm.jet.copy()
    cmap.set_under('black')

    anim, fig = _make_animation(
        frames=tau_display,
        title_template="FLIM Lifetime | Frame {frame}/{n} | t={time:.2f}s",
        cmap=cmap,
        vmin=0.01,  # just above 0 to make 0 = "under" = black
        vmax=vmax,
        cbar_label='tau (ns)',
        frame_interval_s=frame_interval,
        pixel_size_um=pixel_size_um,
        fps=fps,
        dpi=dpi,
    )

    _save_animation(anim, output_path, fps, dpi)
    plt.close(fig)
    return os.path.abspath(output_path)


def _save_animation(anim: FuncAnimation, path: str, fps: int, dpi: int):
    """Save animation to file, choosing writer by extension."""
    ext = Path(path).suffix.lower()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)

    if ext == '.mp4' and _HAS_FFMPEG:
        writer = FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(path, writer=writer, dpi=dpi)
    elif ext == '.gif' or not _HAS_FFMPEG:
        # Fallback to GIF
        if ext == '.mp4':
            path = path.replace('.mp4', '.gif')
        writer = PillowWriter(fps=fps)
        anim.save(path, writer=writer, dpi=dpi)
    else:
        writer = PillowWriter(fps=fps)
        anim.save(path, writer=writer, dpi=dpi)


# ═══════════════════════════════════════════════════════════════════════
# High-level: generate both videos for one replica
# ═══════════════════════════════════════════════════════════════════════

def generate_replica_videos(
    sim_result,
    replica_idx: int = 0,
    output_dir: str = "videos",
    prefix: str = "replica",
    fmt: str = "gif",
    donor_channel: int = 0,
    tau_d_ns: float = 4.1,
    fps: int = 4,
    dpi: int = 150,
) -> Dict[str, str]:
    """Generate intensity and FLIM videos for one simulation replica.

    Parameters
    ----------
    sim_result : SimulationResult
        Result from one pipeline run.
    replica_idx : int
        Replica number (for filename).
    output_dir : str
        Directory to save videos.
    prefix : str
        Filename prefix.
    fmt : str
        "gif" or "mp4" (mp4 requires ffmpeg).
    donor_channel : int
        Donor fluorophore channel.
    tau_d_ns : float
        Unquenched donor lifetime (ns).
    fps : int
        Frames per second.
    dpi : int
        Resolution.

    Returns
    -------
    paths : dict
        {"intensity": "/abs/path/to/intensity.gif",
         "flim": "/abs/path/to/flim.gif"}
    """
    os.makedirs(output_dir, exist_ok=True)
    tag = f"{prefix}_{replica_idx:03d}"

    intensity_path = os.path.join(output_dir, f"{tag}_intensity.{fmt}")
    flim_path = os.path.join(output_dir, f"{tag}_flim.{fmt}")

    p_int = render_intensity_video(
        sim_result, intensity_path,
        donor_channel=donor_channel, fps=fps, dpi=dpi)

    p_flim = render_flim_video(
        sim_result, flim_path,
        donor_channel=donor_channel, tau_d_ns=tau_d_ns,
        fps=fps, dpi=dpi)

    return {"intensity": p_int, "flim": p_flim}


def _build_density_grid(positions, species_id, is_alive, species_indices,
                        Lx, Ly, grid_res=0.5):
    """Build 2D density map for selected species on a spatial grid.

    Parameters
    ----------
    positions : ndarray[N, 2] in um
    species_id : ndarray[N]
    is_alive : ndarray[N] bool
    species_indices : list of int, which species IDs to count
    Lx, Ly : float, membrane size in um
    grid_res : float, grid cell size in um

    Returns
    -------
    density : ndarray[ny, nx], particle count per cell
    """
    nx_grid = max(1, int(np.ceil(Lx / grid_res)))
    ny_grid = max(1, int(np.ceil(Ly / grid_res)))
    density = np.zeros((ny_grid, nx_grid))

    mask = is_alive.copy()
    for sid in species_indices:
        mask_sp = mask & (species_id == sid)
        pos = positions[mask_sp]
        if len(pos) == 0:
            continue
        ix = np.clip((pos[:, 0] / Lx * nx_grid).astype(int), 0, nx_grid - 1)
        iy = np.clip((pos[:, 1] / Ly * ny_grid).astype(int), 0, ny_grid - 1)
        for i in range(len(ix)):
            density[iy[i], ix[i]] += 1

    return density


def render_bd_density_video(
    sim_result,
    output_path: str,
    species_rgb: Optional[Dict[str, str]] = None,
    grid_res: float = 0.0,
    sigma_smooth: float = 0.0,
    fps: int = 4,
    dpi: int = 150,
) -> str:
    """Render BD particle density video as RGB heatmap.

    Each species group maps to one color channel (R, G, B), producing
    a composite density image that reveals spatial patterns, nucleation
    domains, and species co-localization at realistic scale.

    Default channel assignment (auto-detected):
      EGFR-EGF system:  R=EGF, G=EGFR_EGF (complex), B=EGFR
      PIP3-PTEN system: R=PIP3, G=PTEN_T1, B=PTEN_T2

    Parameters
    ----------
    sim_result : SimulationResult
    output_path : str
    species_rgb : dict, optional
        Maps 'R', 'G', 'B' to species name(s). Example:
        {'R': 'PIP3', 'G': 'PTEN_T1', 'B': 'PTEN_T2'}
        If None, auto-detected from species list.
    grid_res : float
        Spatial grid resolution in um (default 0.5 um).
    sigma_smooth : float
        Gaussian smoothing sigma in grid cells (0 = no smoothing).
    fps, dpi : int
    """
    from scipy.ndimage import gaussian_filter

    config = sim_result.config
    Lx, Ly = config.membrane.size
    n_frames = sim_result.n_frames
    frame_interval = config.scan.frame_interval

    # Auto-scale grid resolution and smoothing based on domain size.
    # Target: ~60-80 grid cells per axis for good visual resolution.
    # Small membranes (1-2 um) need finer grid; large (10 um) need coarser.
    if grid_res <= 0:
        target_cells = 67  # ~67 cells per axis
        grid_res = max(Lx, Ly) / target_cells
    if sigma_smooth <= 0:
        # Smoothing: ~2 cells for small domains, ~1.2 for large
        sigma_smooth = max(1.2, 2.5 - 0.15 * max(Lx, Ly))

    species_names = [sp.name for sp in config.species]
    species_name_to_id = {sp.name: i for i, sp in enumerate(config.species)}

    # Auto-detect RGB channel assignment
    if species_rgb is None:
        # Heuristic: find key species for common systems
        if 'PIP3' in species_names and 'PTEN_T1' in species_names:
            # PIP3-PTEN phosphoinositide system
            species_rgb = {'R': ['PIP3'], 'G': ['PTEN_T1'], 'B': ['PTEN_T2']}
        elif 'EGFR' in species_names and 'EGF' in species_names:
            # EGFR-EGF receptor system
            complex_names = []
            for rxn in config.reactions:
                if hasattr(rxn, 'products'):
                    complex_names.extend(rxn.products)
            species_rgb = {
                'R': ['EGF'],
                'G': [n for n in complex_names if n in species_names] or ['EGFR'],
                'B': ['EGFR'],
            }
        else:
            # Generic: first 3 species to R, G, B
            species_rgb = {
                'R': [species_names[0]] if len(species_names) > 0 else [],
                'G': [species_names[1]] if len(species_names) > 1 else [],
                'B': [species_names[2]] if len(species_names) > 2 else [],
            }

    # Resolve species names to IDs
    channel_ids = {}
    channel_labels = {}
    for ch in ('R', 'G', 'B'):
        names = species_rgb.get(ch, [])
        if isinstance(names, str):
            names = [names]
        ids = [species_name_to_id[n] for n in names if n in species_name_to_id]
        channel_ids[ch] = ids
        channel_labels[ch] = '+'.join(names) if names else 'none'

    # Grid dimensions
    nx_grid = max(1, int(np.ceil(Lx / grid_res)))
    ny_grid = max(1, int(np.ceil(Ly / grid_res)))

    # Pre-compute all frames
    rgb_frames = np.zeros((n_frames, ny_grid, nx_grid, 3))

    for fi in range(n_frames):
        gt = sim_result.frames[fi].ground_truth
        for ci, ch in enumerate(('R', 'G', 'B')):
            if channel_ids[ch]:
                density = _build_density_grid(
                    gt.positions, gt.species_id, gt.is_alive,
                    channel_ids[ch], Lx, Ly, grid_res)
                if sigma_smooth > 0:
                    density = gaussian_filter(density, sigma=sigma_smooth)
                rgb_frames[fi, :, :, ci] = density

    # Per-frame percentile normalization for maximum contrast.
    # Each channel is scaled so that its 2nd-98th percentile maps to [0, 1].
    # This ensures that spatial VARIATION is visible even when absolute
    # densities differ by orders of magnitude between species.
    for fi in range(n_frames):
        for ci in range(3):
            ch = rgb_frames[fi, :, :, ci]
            if ch.max() > 0:
                p_lo = np.percentile(ch, 2)
                p_hi = np.percentile(ch, 98)
                if p_hi > p_lo:
                    ch = (ch - p_lo) / (p_hi - p_lo)
                else:
                    ch = ch / ch.max() if ch.max() > 0 else ch
                rgb_frames[fi, :, :, ci] = np.clip(ch, 0, 1)

    # Gamma correction for perceptual contrast on dark background
    gamma = 0.45
    rgb_frames = np.clip(rgb_frames ** gamma, 0, 1)

    # Build animation with dark scientific aesthetic
    extent = [0, Lx, Ly, 0]
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 6.0))
    fig.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.10)
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    im = ax.imshow(rgb_frames[0], extent=extent,
                   interpolation='gaussian', aspect='equal')

    ax.set_xlabel('x (um)', fontsize=10, color='white')
    ax.set_ylabel('y (um)', fontsize=10, color='white')
    ax.tick_params(labelsize=8, colors='white')
    for spine in ax.spines.values():
        spine.set_color('#333333')

    title = ax.set_title('', fontsize=10, fontweight='bold', color='white')

    # RGB legend with dark background
    legend_handles = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#FF4444',
                    markersize=10, label=f'R: {channel_labels["R"]}'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#44FF44',
                    markersize=10, label=f'G: {channel_labels["G"]}'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#4488FF',
                    markersize=10, label=f'B: {channel_labels["B"]}'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7,
              framealpha=0.7, facecolor='#1a1a1a', edgecolor='#444444',
              labelcolor='white')

    def update(fi):
        im.set_data(rgb_frames[fi])
        bio_time = (fi + 1) * frame_interval
        gt = sim_result.frames[fi].ground_truth
        title.set_text(
            f"BD Density | Frame {fi+1}/{n_frames} | t={bio_time:.2f}s\n"
            f"alive={gt.is_alive.sum()}  complexes={gt.n_complexes}  "
            f"bleached={gt.n_bleached}")
        return [im, title]

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 // fps, blit=False)

    _save_animation(anim, output_path, fps, dpi)
    plt.close(fig)
    return os.path.abspath(output_path)


def generate_replica_videos(
    sim_result,
    replica_idx: int = 0,
    output_dir: str = "videos",
    prefix: str = "replica",
    fmt: str = "gif",
    donor_channel: int = 0,
    tau_d_ns: float = 4.1,
    fps: int = 4,
    dpi: int = 150,
) -> Dict[str, str]:
    """Generate intensity, FLIM, and BD particle videos for one replica.

    Returns
    -------
    paths : dict
        {"intensity": "...", "flim": "...", "bd_particles": "..."}
    """
    os.makedirs(output_dir, exist_ok=True)
    tag = f"{prefix}_{replica_idx:03d}"

    intensity_path = os.path.join(output_dir, f"{tag}_intensity.{fmt}")
    flim_path = os.path.join(output_dir, f"{tag}_flim.{fmt}")
    bd_path = os.path.join(output_dir, f"{tag}_bd_particles.{fmt}")

    p_int = render_intensity_video(
        sim_result, intensity_path,
        donor_channel=donor_channel, fps=fps, dpi=dpi)

    p_flim = render_flim_video(
        sim_result, flim_path,
        donor_channel=donor_channel, tau_d_ns=tau_d_ns,
        fps=fps, dpi=dpi)

    p_bd = render_bd_density_video(
        sim_result, bd_path, fps=fps, dpi=dpi)

    return {"intensity": p_int, "flim": p_flim, "bd_particles": p_bd}


def generate_all_replica_videos(
    sim_results: list,
    output_dir: str = "videos",
    prefix: str = "replica",
    fmt: str = "gif",
    tau_d_ns: float = 4.1,
    fps: int = 4,
    dpi: int = 150,
) -> List[Dict[str, str]]:
    """Generate videos for ALL replicas.

    Returns
    -------
    all_paths : list of dict
        One dict per replica with "intensity", "flim", and "bd_particles" paths.
    """
    all_paths = []
    for i, sr in enumerate(sim_results):
        paths = generate_replica_videos(
            sr, replica_idx=i, output_dir=output_dir,
            prefix=prefix, fmt=fmt, tau_d_ns=tau_d_ns,
            fps=fps, dpi=dpi)
        all_paths.append(paths)
        print(f"  Replica {i}: {os.path.basename(paths['intensity'])}, "
              f"{os.path.basename(paths['flim'])}, "
              f"{os.path.basename(paths['bd_particles'])}")
    return all_paths
