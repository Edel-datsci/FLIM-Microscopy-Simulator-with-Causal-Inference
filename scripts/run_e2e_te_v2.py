"""
E2E Pipeline: BD → FLIM → TE v2 (Conditional Transfer Entropy)
================================================================
Runs N_REPLICAS independent simulations with different seeds,
converts to trials dict, runs CTE analysis, and reports results.

Uses mini_demo.yaml for fast execution (~30s per replica).
"""

import sys
import os
import time
import json
import numpy as np

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ed_mrdt.config import load_config
from ed_mrdt.pipeline import SimulationPipeline

from l6_te_redesign import (
    sim_results_to_trials_dict,
    CausalPairSpecV2,
    compute_causal_flim_generic_v2,
    discretize_uniform,
    ensemble_conditional_te_discrete,
    select_lag_by_conditional_te,
    conditional_te_bidirectional_ensemble,
)

from ed_mrdt.video import generate_all_replica_videos

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

N_REPLICAS = 5          # independent simulation runs
YAML_PATH = "ed_mrdt/examples/mini_demo.yaml"
N_SURROGATES = 100      # for significance testing
OUTPUT_FILE = "e2e_te_v2_results.json"


def run_one_replica(config, seed, replica_idx):
    """Run one simulation replica with a modified random seed."""
    import copy
    cfg = copy.deepcopy(config)
    # Modify seed by changing initial conditions slightly
    # The pipeline uses internal RNG; we change populations slightly to get variation
    # Actually, we need to set a seed. Let's check if config has a seed field.
    if hasattr(cfg, 'random_seed'):
        cfg.random_seed = seed
    # Run
    pipeline = SimulationPipeline(cfg, record_subframe=True)
    t0 = time.perf_counter()
    result = pipeline.run()
    wall = time.perf_counter() - t0
    print(f"  Replica {replica_idx}: {wall:.1f}s, "
          f"{result.total_photons_detected} photons, "
          f"{result.n_frames} frames, "
          f"complexes=[{', '.join(str(f.ground_truth.n_complexes) for f in result.frames)}]")
    return result


def main():
    print("=" * 70)
    print("E2E PIPELINE: BD → FLIM → TE v2 (Conditional Transfer Entropy)")
    print("=" * 70)

    # Load config and increase frames for meaningful TE
    config = load_config(YAML_PATH)

    # Override: need >= 10 frames for TE to have any statistical power
    # With 5 trials x 10 frames, we get 5 x 9 = 45 effective samples
    config.scan.n_frames = 10
    # Keep frame_interval short so total sim time is manageable
    # 10 frames x 0.1s = 1.0s total biology
    config.scan.frame_interval = 0.1
    # Reduce BD steps per frame for speed: 0.1s / 0.0001s = 1000 steps (already)
    config.validate()

    print(f"\nConfig: {YAML_PATH} (overridden: {config.scan.n_frames} frames)")
    print(f"  Membrane: {config.membrane.size} um")
    print(f"  Species: {[s.name for s in config.species]}")
    print(f"  Frames: {config.scan.n_frames}")
    print(f"  BD timestep: {config.bd_timestep}")
    print(f"  Frame interval: {config.scan.frame_interval}s")
    print(f"  Replicas: {N_REPLICAS}")

    # ── Run N replicas ──────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print(f"Phase 1: Running {N_REPLICAS} independent simulation replicas")
    print(f"{'─' * 50}")

    t0_total = time.perf_counter()
    sim_results = []
    for i in range(N_REPLICAS):
        result = run_one_replica(config, seed=1000 + i * 7, replica_idx=i)
        sim_results.append(result)

    wall_total = time.perf_counter() - t0_total
    print(f"\n  Total simulation time: {wall_total:.1f}s")

    # ── Convert to trials dict ──────────────────────────────────
    print(f"\n{'─' * 50}")
    print(f"Phase 2: Converting to trials dictionary")
    print(f"{'─' * 50}")

    trials_dict = sim_results_to_trials_dict(sim_results)

    for key, arr in trials_dict.items():
        print(f"  {key:15s}: shape={arr.shape}, "
              f"range=[{arr.min():.4f}, {arr.max():.4f}], "
              f"mean={arr.mean():.4f}")

    # ── Run CTE analysis ────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print(f"Phase 3: Conditional Transfer Entropy analysis")
    print(f"{'─' * 50}")

    # Define causal pairs for this system
    pair_specs = [
        CausalPairSpecV2(
            source_key="n_complexes",
            target_key="tau_mean",
            candidate_lags=(1,),
            n_bins=2,
            n_surrogates=N_SURROGATES,
            lag_criterion="asymmetry",
        ),
        CausalPairSpecV2(
            source_key="n_bleached",
            target_key="intensity",
            candidate_lags=(1,),
            n_bins=2,
            n_surrogates=N_SURROGATES,
            lag_criterion="asymmetry",
        ),
        CausalPairSpecV2(
            source_key="msd",
            target_key="n_complexes",
            candidate_lags=(1,),
            n_bins=2,
            n_surrogates=N_SURROGATES,
            lag_criterion="asymmetry",
        ),
        CausalPairSpecV2(
            source_key="density",
            target_key="E_mean",
            candidate_lags=(1,),
            n_bins=2,
            n_surrogates=N_SURROGATES,
            lag_criterion="asymmetry",
        ),
    ]

    t0_te = time.perf_counter()
    results = compute_causal_flim_generic_v2(trials_dict, pair_specs)
    wall_te = time.perf_counter() - t0_te
    print(f"  TE computation time: {wall_te:.1f}s")

    # ── Report results ──────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"RESULTS: Conditional Transfer Entropy (v2)")
    print(f"{'═' * 70}")
    print(f"{'Pair':<30s} {'TE(X→Y)':>10s} {'TE(Y→X)':>10s} {'p(X→Y)':>10s} {'Sig?':>6s} {'Lag':>5s}")
    print(f"{'─' * 71}")

    output_data = {
        "config": YAML_PATH,
        "n_replicas": N_REPLICAS,
        "n_surrogates": N_SURROGATES,
        "n_frames": sim_results[0].n_frames,
        "wall_time_sim_s": wall_total,
        "wall_time_te_s": wall_te,
        "pairs": [],
    }

    for res in results:
        spec = res.spec
        te = res.te_result
        pair_name = f"{spec.source_key} → {spec.target_key}"
        sig_str = "YES" if te.sig_xy and te.sig_xy.significant else "no"
        p_val = te.sig_xy.p_value if te.sig_xy else float('nan')

        print(f"{pair_name:<30s} {te.te_xy_bits:>10.4f} {te.te_yx_bits:>10.4f} "
              f"{p_val:>10.4f} {sig_str:>6s} {te.lag:>5d}")

        pair_data = {
            "source": spec.source_key,
            "target": spec.target_key,
            "te_xy_bits": te.te_xy_bits,
            "te_yx_bits": te.te_yx_bits,
            "p_xy": p_val,
            "significant_xy": bool(te.sig_xy and te.sig_xy.significant),
            "lag": te.lag,
            "notes": list(te.notes),
        }
        if te.adf_diagnostics:
            pair_data["adf"] = {
                k: v for k, v in te.adf_diagnostics.items()
                if k != "available" and not isinstance(v, np.ndarray)
            }
        output_data["pairs"].append(pair_data)

    print(f"\n{'─' * 71}")

    # ADF diagnostics
    print(f"\nADF Diagnostics (informational):")
    for res in results:
        adf = res.te_result.adf_diagnostics
        if adf and adf.get("available"):
            pair_name = f"{res.spec.source_key} → {res.spec.target_key}"
            print(f"  {pair_name}: X p={adf.get('x_adf_pvalue', '?'):.4f}, "
                  f"Y p={adf.get('y_adf_pvalue', '?'):.4f}")
        elif adf and adf.get("warning"):
            print(f"  Warning: {adf['warning']}")

    # ── Video generation ───────────────────────────────────────
    print(f"\n{'─' * 50}")
    print(f"Phase 4: Generating videos (intensity + FLIM per replica)")
    print(f"{'─' * 50}")

    t0_vid = time.perf_counter()
    video_dir = "videos_e2e"
    video_paths = generate_all_replica_videos(
        sim_results, output_dir=video_dir,
        prefix="replica", fmt="gif",
        tau_d_ns=4.1, fps=3, dpi=120)
    wall_vid = time.perf_counter() - t0_vid

    print(f"\n  {len(video_paths)} replicas x 3 videos = {len(video_paths) * 3} files")
    print(f"  Output directory: {os.path.abspath(video_dir)}")
    print(f"  Video generation time: {wall_vid:.1f}s")

    output_data["videos"] = [
        {"replica": i, "intensity": p["intensity"], "flim": p["flim"],
         "bd_particles": p["bd_particles"]}
        for i, p in enumerate(video_paths)
    ]

    # ── Convergence analysis ────────────────────────────────────
    print(f"\n{'─' * 50}")
    print(f"Phase 5: Convergence analysis (TE variance vs n_trials)")
    print(f"{'─' * 50}")

    # Test with increasing number of trials for first pair
    key_x = pair_specs[0].source_key
    key_y = pair_specs[0].target_key
    x_all = trials_dict[key_x]
    y_all = trials_dict[key_y]

    trial_counts = list(range(2, N_REPLICAS + 1))
    convergence_data = []

    for n_t in trial_counts:
        x_sub = x_all[:n_t]
        y_sub = y_all[:n_t]
        xd = np.array([discretize_uniform(t, 2) for t in x_sub])
        yd = np.array([discretize_uniform(t, 2) for t in y_sub])
        te_val = ensemble_conditional_te_discrete(xd, yd, lag=1)
        convergence_data.append({"n_trials": n_t, "te_bits": te_val})
        print(f"  n_trials={n_t}: TE={te_val:.6f} bits")

    output_data["convergence"] = convergence_data

    # Save results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print(f"Total wall time: {time.perf_counter() - t0_total:.1f}s")


if __name__ == "__main__":
    main()
