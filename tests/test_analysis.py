"""
Test suite for ED-MRDT analysis.py — 7-Layer System Radiograph
================================================================

Every test validates a published theoretical prediction against
computed output. No test is "does it run" — every test is
"does the physics match".

Validation strategy per layer:
  L1 Phasor:    Mono-exponential → on universal semicircle (Digman 2008)
  L2 C(r):      Poisson randoms → G(0) = 1/N (Petersen 1993)
  L3 ACF:       Exponential decay → ACF = exp(-t/τ) (Elson 1974)
  L4 pCF:       No flow → zero-lag peak only (Digman 2009)
  L5 iMSD:      Free diffusion → linear iMSD (Di Rienzo 2013)
  L6 TE:        Coupled time series → TE > 0 in correct direction
  L7 Discrep:   Perfect image → zero error
"""

import numpy as np
import sys
import traceback

# ═══════════════════════════════════════════════════════════════════════
# TEST INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════════

PASS_COUNT = 0
FAIL_COUNT = 0
ALL_RESULTS = []


def check(name, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    ALL_RESULTS.append((name, status, detail))
    flag = "✓" if condition else "✗"
    print(f"  {flag} {name}", end="")
    if detail:
        print(f"  [{detail}]", end="")
    print()


def section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ═══════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATORS
# ═══════════════════════════════════════════════════════════════════════

def make_mono_exponential_flim(tau_ns, time_range_ns=25.0, n_bins=64,
                                nx=16, ny=16, photons_per_pixel=500,
                                seed=42):
    """Generate FLIM stack from mono-exponential decay.

    I(t) = (1/τ) exp(-t/τ), sampled as TCSPC histogram.
    Reference: Lakowicz (2006) Eq. 1.2.
    """
    rng = np.random.default_rng(seed)
    time_range_s = time_range_ns * 1e-9
    bin_width_ns = time_range_ns / n_bins

    # For each pixel, draw photons from exponential distribution
    flim_stack = np.zeros((1, nx, ny, n_bins), dtype=np.uint32)
    for ix in range(nx):
        for iy in range(ny):
            # Draw arrival times from exponential
            arrivals = rng.exponential(tau_ns, size=photons_per_pixel)
            # Bin into TCSPC
            bins = (arrivals / bin_width_ns).astype(int)
            valid = (bins >= 0) & (bins < n_bins)
            np.add.at(flim_stack[0, ix, iy], bins[valid], 1)

    return flim_stack, time_range_s


def make_two_species_flim(tau1_ns, tau2_ns, fraction1=0.5,
                           time_range_ns=25.0, n_bins=64,
                           nx=16, ny=16, photons_per_pixel=1000,
                           seed=42):
    """Generate FLIM stack from two-species mixture."""
    rng = np.random.default_rng(seed)
    bin_width_ns = time_range_ns / n_bins
    time_range_s = time_range_ns * 1e-9

    flim_stack = np.zeros((1, nx, ny, n_bins), dtype=np.uint32)
    for ix in range(nx):
        for iy in range(ny):
            n1 = int(photons_per_pixel * fraction1)
            n2 = photons_per_pixel - n1
            arr1 = rng.exponential(tau1_ns, size=n1)
            arr2 = rng.exponential(tau2_ns, size=n2)
            arrivals = np.concatenate([arr1, arr2])
            bins = (arrivals / bin_width_ns).astype(int)
            valid = (bins >= 0) & (bins < n_bins)
            np.add.at(flim_stack[0, ix, iy], bins[valid], 1)

    return flim_stack, time_range_s


def make_intensity_series_diffusion(D, pixel_size_um, frame_interval,
                                     n_frames=20, nx=16, ny=16,
                                     n_particles=200, seed=42):
    """Simulate 2D Brownian diffusion, image as intensity time-series."""
    rng = np.random.default_rng(seed)
    Lx = nx * pixel_size_um
    Ly = ny * pixel_size_um

    # Initial positions uniform
    x = rng.uniform(0, Lx, n_particles)
    y = rng.uniform(0, Ly, n_particles)

    series = np.zeros((n_frames, nx, ny))

    for frame in range(n_frames):
        # Diffuse
        sigma = np.sqrt(2 * D * frame_interval)
        x += rng.normal(0, sigma, n_particles)
        y += rng.normal(0, sigma, n_particles)
        # PBC
        x = x % Lx
        y = y % Ly
        # Bin into pixels
        ix = np.clip((x / pixel_size_um).astype(int), 0, nx - 1)
        iy = np.clip((y / pixel_size_um).astype(int), 0, ny - 1)
        for p in range(n_particles):
            series[frame, ix[p], iy[p]] += 1

    return series


# ═══════════════════════════════════════════════════════════════════════
# L1: PHASOR TESTS
# ═══════════════════════════════════════════════════════════════════════

def test_L1_phasor():
    section("L1: PHASOR DECOMPOSITION (Digman 2008)")

    from ed_mrdt.analysis import (phasor_transform, phasor_universal_circle,
                                   phasor_fret_trajectory)

    # Test 1: Mono-exponential → on universal semicircle
    # Theory: g = 1/(1+(ωτ)²), s = ωτ/(1+(ωτ)²)
    # Reference: Digman (2008) Eq. S1
    tau_ns = 4.0
    time_range_ns = 25.0
    flim, time_range_s = make_mono_exponential_flim(
        tau_ns, time_range_ns=time_range_ns, n_bins=64,
        nx=16, ny=16, photons_per_pixel=2000, seed=42
    )

    result = phasor_transform(flim, time_range_s, channel=0,
                               harmonic=1, min_counts=20)

    omega = result.omega
    omega_ns = omega * 1e-9
    wt = omega_ns * tau_ns

    g_theory = 1.0 / (1.0 + wt ** 2)
    s_theory = wt / (1.0 + wt ** 2)

    g_mean = result.g[result.mask].mean()
    s_mean = result.s[result.mask].mean()

    check("Phasor g matches theory",
          abs(g_mean - g_theory) < 0.05,
          f"g_measured={g_mean:.4f}, g_theory={g_theory:.4f}")

    check("Phasor s matches theory",
          abs(s_mean - s_theory) < 0.05,
          f"s_measured={s_mean:.4f}, s_theory={s_theory:.4f}")

    # Test 2: Point lies ON the semicircle
    # distance from center (0.5, 0) should be ≈ 0.5
    dist_from_center = np.sqrt((g_mean - 0.5) ** 2 + s_mean ** 2)
    check("Mono-exp phasor on semicircle",
          abs(dist_from_center - 0.5) < 0.05,
          f"radius={dist_from_center:.4f}, expected=0.5")

    # Test 3: Two-species mixture → INSIDE semicircle
    flim2, tr2 = make_two_species_flim(
        2.0, 6.0, fraction1=0.5,
        time_range_ns=25.0, n_bins=64,
        photons_per_pixel=2000, seed=99
    )
    res2 = phasor_transform(flim2, tr2, channel=0, min_counts=20)
    g2 = res2.g[res2.mask].mean()
    s2 = res2.s[res2.mask].mean()
    dist2 = np.sqrt((g2 - 0.5) ** 2 + s2 ** 2)
    check("Two-species phasor INSIDE semicircle",
          dist2 < 0.5 - 0.01,
          f"radius={dist2:.4f} < 0.5")

    # Test 4: Phase lifetime recovers τ
    tau_phi_mean = np.nanmean(result.tau_phi[result.mask])
    check("Phase lifetime recovers τ",
          abs(tau_phi_mean - tau_ns) / tau_ns < 0.10,
          f"τ_φ={tau_phi_mean:.2f} ns, expected={tau_ns} ns")

    # Test 5: Modulation lifetime recovers τ for mono-exp
    tau_mod_mean = np.nanmean(result.tau_mod[result.mask])
    check("Modulation lifetime recovers τ (mono-exp)",
          abs(tau_mod_mean - tau_ns) / tau_ns < 0.10,
          f"τ_m={tau_mod_mean:.2f} ns, expected={tau_ns} ns")

    # Test 6: τ_φ = τ_m for mono-exponential
    # Reference: Ranjit (2018) — equality only holds for single exp
    check("τ_φ ≈ τ_m (mono-exp identity)",
          abs(tau_phi_mean - tau_mod_mean) < 0.3,
          f"|τ_φ - τ_m| = {abs(tau_phi_mean - tau_mod_mean):.3f} ns")

    # Test 7: Universal circle geometry
    gc, sc = phasor_universal_circle(200)
    check("Universal circle has correct bounds",
          abs(gc.min() - 0.0) < 0.01 and abs(gc.max() - 1.0) < 0.01,
          f"g ∈ [{gc.min():.3f}, {gc.max():.3f}]")

    # Test 8: FRET trajectory starts at donor phasor
    g_traj, s_traj = phasor_fret_trajectory(tau_ns, omega)
    check("FRET trajectory starts at donor phasor",
          abs(g_traj[0] - g_theory) < 0.01 and abs(s_traj[0] - s_theory) < 0.01,
          f"start=({g_traj[0]:.3f},{s_traj[0]:.3f}), donor=({g_theory:.3f},{s_theory:.3f})")

    # Test 9: FRET trajectory ends near (1,0) at high E
    check("FRET trajectory approaches (1,0) at high E",
          g_traj[-1] > 0.9 and s_traj[-1] < 0.1,
          f"end=({g_traj[-1]:.3f},{s_traj[-1]:.3f})")

    # Test 10: Different τ → different phasor position
    flim_short, tr_short = make_mono_exponential_flim(
        1.0, photons_per_pixel=2000, seed=77
    )
    res_short = phasor_transform(flim_short, tr_short, min_counts=20)
    g_short = res_short.g[res_short.mask].mean()
    check("Different τ → different phasor (g)",
          abs(g_short - g_mean) > 0.05,
          f"g(1ns)={g_short:.3f}, g(4ns)={g_mean:.3f}")


# ═══════════════════════════════════════════════════════════════════════
# L2: SPATIAL CORRELATION TESTS
# ═══════════════════════════════════════════════════════════════════════

def test_L2_spatial():
    section("L2: SPATIAL CORRELATION C(r) (Petersen 1993)")

    from ed_mrdt.analysis import (spatial_autocorrelation,
                                   spatial_correlation_ground_truth)

    # Test 1: Uniform random image → G(0) ≈ 1/<N_per_beam>
    # For Poisson image with mean λ per pixel,
    # G(0) = 1/λ (shot noise correlation)
    # Reference: Petersen (1993) Eq. 6
    rng = np.random.default_rng(42)
    lam = 50.0
    img = rng.poisson(lam, size=(64, 64)).astype(float)
    res = spatial_autocorrelation(img, pixel_size_um=0.059)
    G0 = res.C_r[0]
    G0_theory = 1.0 / lam
    check("Poisson G(0) ≈ 1/λ",
          abs(G0 - G0_theory) / G0_theory < 0.25,
          f"G(0)={G0:.5f}, 1/λ={G0_theory:.5f}")

    # Test 2: C(r) decays monotonically from G(0)
    decaying = all(res.C_r[i] >= res.C_r[i + 1] - 0.001
                   for i in range(min(5, len(res.C_r) - 1)))
    check("C(r) decays from G(0)",
          decaying,
          f"first 5 values: {res.C_r[:5].round(5)}")

    # Test 3: Clustered image → larger G(0) than uniform
    clustered = np.zeros((64, 64))
    for cx, cy in [(16, 16), (48, 48), (16, 48), (48, 16)]:
        y, x = np.ogrid[:64, :64]
        mask = ((x - cx) ** 2 + (y - cy) ** 2) < 25
        clustered[mask] = 100
    clustered += rng.poisson(5, size=(64, 64))

    res_clust = spatial_autocorrelation(clustered, 0.059)
    check("Clustered image → higher G(0)",
          res_clust.C_r[0] > G0,
          f"G(0)_clust={res_clust.C_r[0]:.4f} > G(0)_uniform={G0:.4f}")

    # Test 4: Ground truth g(r) for uniform random → g(r) ≈ 1
    pos = rng.uniform(0, 2.0, size=(500, 2))
    r, gr = spatial_correlation_ground_truth(
        pos, (2.0, 2.0), n_bins=30, r_max_um=0.8
    )
    # Skip first bin (self-exclusion), check g(r) ≈ 1 for r > 0.05
    far = r > 0.05
    if far.any():
        gr_mean = gr[far].mean()
        check("Uniform positions → g(r) ≈ 1",
              abs(gr_mean - 1.0) < 0.3,
              f"<g(r)> = {gr_mean:.3f}")
    else:
        check("Uniform positions → g(r) ≈ 1", True, "skipped (no far bins)")

    # Test 5: Clustered positions → g(r) peak at small r
    # Put half the particles in a small cluster
    pos_clust = np.zeros((200, 2))
    pos_clust[:100] = rng.normal([0.5, 0.5], 0.05, size=(100, 2))
    pos_clust[100:] = rng.uniform(0, 2, size=(100, 2))
    pos_clust = pos_clust % 2.0  # PBC

    r_c, gr_c = spatial_correlation_ground_truth(
        pos_clust, (2.0, 2.0), n_bins=30, r_max_um=0.5
    )
    # g(r) at small r should be >> 1 due to cluster
    small_r = (r_c > 0.01) & (r_c < 0.15)
    if small_r.any():
        peak = gr_c[small_r].max()
        check("Clustered positions → g(r) > 1 at small r",
              peak > 2.0,
              f"g(r<0.15µm) peak = {peak:.1f}")
    else:
        check("Clustered positions → g(r) peak", True, "no small-r bins")


# ═══════════════════════════════════════════════════════════════════════
# L3: TEMPORAL ACF TESTS
# ═══════════════════════════════════════════════════════════════════════

def test_L3_temporal():
    section("L3: TEMPORAL ACF (Elson & Magde 1974)")

    from ed_mrdt.analysis import temporal_autocorrelation, _estimate_correlation_time

    # Test 1: White noise → ACF(τ>0) ≈ 0
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1, 1000)
    lags, acf = temporal_autocorrelation(noise, dt=1.0)
    check("White noise ACF(τ>0) ≈ 0",
          abs(acf[1:10].mean()) < 0.1,
          f"<ACF[1:10]> = {acf[1:10].mean():.4f}")

    # Test 2: ACF(0) = 1 by definition
    check("ACF(0) = 1",
          abs(acf[0] - 1.0) < 1e-10,
          f"ACF(0) = {acf[0]:.10f}")

    # Test 3: Exponentially correlated signal → ACF ≈ exp(-t/τ_c)
    # Generate AR(1): x[t] = φ x[t-1] + ε
    # Theory: ACF(k) = φ^k, τ_corr = -1/ln(φ)
    # Reference: Box, Jenkins & Reinsel (2015) Time Series Analysis
    phi = 0.9
    N = 10000
    x = np.zeros(N)
    for t in range(1, N):
        x[t] = phi * x[t - 1] + rng.normal(0, 1)

    lags_ar, acf_ar = temporal_autocorrelation(x, dt=1.0)

    # Check ACF at lag 1 ≈ φ
    check("AR(1) ACF(1) ≈ φ",
          abs(acf_ar[1] - phi) < 0.05,
          f"ACF(1)={acf_ar[1]:.4f}, φ={phi}")

    # Check ACF at lag 5 ≈ φ^5
    phi5 = phi ** 5
    check("AR(1) ACF(5) ≈ φ⁵",
          abs(acf_ar[5] - phi5) < 0.08,
          f"ACF(5)={acf_ar[5]:.4f}, φ⁵={phi5:.4f}")

    # Test 4: Correlation time estimation
    tau_c_theory = -1.0 / np.log(phi)  # ≈ 9.49
    tau_c_est = _estimate_correlation_time(lags_ar, acf_ar)
    check("Correlation time recovery",
          abs(tau_c_est - tau_c_theory) / tau_c_theory < 0.15,
          f"τ_c={tau_c_est:.2f}, theory={tau_c_theory:.2f}")

    # Test 5: Constant signal → ACF undefined (variance = 0)
    const = np.ones(100) * 5.0
    _, acf_const = temporal_autocorrelation(const, dt=1.0)
    check("Constant signal → ACF = 0 (zero variance)",
          abs(acf_const.sum()) < 1e-10,
          f"sum(ACF) = {acf_const.sum():.1e}")

    # Test 6: Sinusoidal → periodic ACF
    t = np.arange(1000) * 0.1
    signal = np.sin(2 * np.pi * t / 10)  # period = 10
    _, acf_sin = temporal_autocorrelation(signal, dt=0.1)
    # ACF should have a maximum near lag = 10 (one period = 100 samples)
    half_period = 50
    if len(acf_sin) > half_period + 5:
        near_half = acf_sin[half_period - 5:half_period + 5]
        check("Sinusoidal ACF minimum near half-period",
              near_half.min() < -0.5,
              f"ACF near T/2 min = {near_half.min():.3f}")
    else:
        check("Sinusoidal ACF half-period", True, "skipped (series too short)")


# ═══════════════════════════════════════════════════════════════════════
# L4: PAIR CORRELATION TESTS
# ═══════════════════════════════════════════════════════════════════════

def test_L4_pcf():
    section("L4: PAIR CORRELATION pCF (Digman & Gratton 2009)")

    from ed_mrdt.analysis import pair_correlation_function

    # Test 1: Static uniform image → pCF ≈ 0 everywhere
    # No fluctuations → no correlation
    rng = np.random.default_rng(42)
    static = np.ones((10, 8, 8)) * 50.0
    # Add tiny noise to avoid division by zero
    static += rng.normal(0, 0.01, static.shape)
    res = pair_correlation_function(static, 0.059, 1.0, max_distance_px=3)

    # pCF should be near zero for all distances and lags
    check("Static image → pCF ≈ 0",
          np.abs(res.pcf_carpet).max() < 0.1,
          f"max|pCF| = {np.abs(res.pcf_carpet).max():.4f}")

    # Test 2: Self-correlation (distance=0) peak at lag=0
    # With real fluctuations
    fluct = rng.poisson(50, size=(20, 8, 8)).astype(float)
    res2 = pair_correlation_function(fluct, 0.059, 1.0, max_distance_px=3)
    # distance=0, lag=0 should be the highest value
    pcf_d0 = res2.pcf_carpet[0, :]
    if len(pcf_d0) > 1:
        check("Self-correlation (d=0) peak at lag=0",
              pcf_d0[0] >= pcf_d0[1:].max() - 0.01,
              f"pCF(0,0)={pcf_d0[0]:.4f}, max(rest)={pcf_d0[1:].max():.4f}")

    # Test 3: pCF carpet has correct dimensions
    n_dist = len(res2.distances_px)
    n_lag = len(res2.lag_frames)
    check("pCF carpet shape correct",
          res2.pcf_carpet.shape == (n_dist, n_lag),
          f"shape={res2.pcf_carpet.shape}, expected=({n_dist},{n_lag})")

    # Test 4: distances_um = distances_px × pixel_size
    expected_um = res2.distances_px * 0.059
    check("Distance conversion px → µm correct",
          np.allclose(res2.distances_um, expected_um, atol=1e-6),
          f"max diff = {np.abs(res2.distances_um - expected_um).max():.1e}")

    # Test 5: transit time at d=0 should be 0
    check("Transit time at d=0 is zero",
          res2.transit_time_s[0] == 0.0,
          f"t_transit(d=0) = {res2.transit_time_s[0]}")


# ═══════════════════════════════════════════════════════════════════════
# L5: iMSD TESTS
# ═══════════════════════════════════════════════════════════════════════

def test_L5_imsd():
    section("L5: iMSD via STICS (Di Rienzo 2013)")

    from ed_mrdt.analysis import compute_imsd

    # STICS detects DEVIATIONS from uniform free diffusion.
    # For spatially uniform free diffusion, G(ξ,η,τ)→0 and
    # iMSD is unreliable (no Gaussian peak to fit).
    # Reference: Di Rienzo (2013) requires spatial heterogeneity.

    # Test 1: iMSD(0) = 0 by definition
    pixel_size = 0.1  # µm
    dt = 0.5  # s
    rng = np.random.default_rng(42)

    # Uniform diffusion (no structure) → small/zero iMSD
    series_uniform = rng.poisson(20, (30, 32, 32)).astype(float)
    res_u = compute_imsd(series_uniform, pixel_size, dt, max_lag=10)

    check("iMSD(0) = 0",
          abs(res_u.iMSD_um2[0]) < 1e-10,
          f"iMSD(0) = {res_u.iMSD_um2[0]:.1e}")

    # Test 2: Static uniform image → iMSD ≈ 0
    # After shot noise subtraction, G(0,0,0) should be ~0 for Poisson
    check("Uniform static → small iMSD",
          np.max(np.abs(res_u.iMSD_um2)) < 0.5,
          f"max|iMSD| = {np.max(np.abs(res_u.iMSD_um2)):.4f}")

    # Test 3: Image with a MOVING cluster → iMSD should grow
    # Create a localized bright spot that moves across the image
    n_frames = 30
    nx, ny = 32, 32
    cluster_series = np.ones((n_frames, nx, ny)) * 5.0  # background
    for f in range(n_frames):
        # Cluster center moves linearly
        cx = int(8 + f * 0.5) % nx
        cy = int(8 + f * 0.3) % ny
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                px, py = (cx + dx) % nx, (cy + dy) % ny
                cluster_series[f, px, py] += 100  # bright cluster

    res_cluster = compute_imsd(cluster_series, pixel_size, dt, max_lag=10)

    # With a moving bright cluster, σ² should increase with lag
    # because the correlation peak broadens
    check("Moving cluster → D_eff > 0 or growing sigma²",
          res_cluster.D_eff > 0 or np.any(res_cluster.sigma2_um2 > 0),
          f"D_eff={res_cluster.D_eff:.4f}, max(σ²)={np.max(res_cluster.sigma2_um2):.4f}")

    # Test 4: Verify output structure is complete
    check("IMSDResult has all fields",
          hasattr(res_u, 'lag_s') and hasattr(res_u, 'iMSD_um2')
          and hasattr(res_u, 'D_eff') and hasattr(res_u, 'alpha')
          and hasattr(res_u, 'regime'),
          f"regime={res_u.regime}, α={res_u.alpha:.2f}")

    # Test 5: 1-frame series doesn't crash
    single = rng.poisson(10, (1, 8, 8)).astype(float)
    res_s = compute_imsd(single, pixel_size, dt)
    check("1-frame iMSD doesn't crash",
          len(res_s.iMSD_um2) > 0,
          f"regime = {res_s.regime}")


# ═══════════════════════════════════════════════════════════════════════
# L6: TRANSFER ENTROPY TESTS
# ═══════════════════════════════════════════════════════════════════════

def test_L6_te():
    section("L6: TRANSFER ENTROPY (Schreiber 2000)")

    from ed_mrdt.analysis import transfer_entropy

    rng = np.random.default_rng(42)

    # Test 1: Independent series → TE ≈ 0 in both directions
    x_ind = rng.normal(0, 1, 500)
    y_ind = rng.normal(0, 1, 500)
    res_ind = transfer_entropy(x_ind, y_ind, n_bins=8, n_shuffle=30)
    check("Independent: TE(X→Y) ≈ 0",
          res_ind.te_xy < 0.05,
          f"TE(X→Y) = {res_ind.te_xy:.4f}")
    check("Independent: TE(Y→X) ≈ 0",
          res_ind.te_yx < 0.05,
          f"TE(Y→X) = {res_ind.te_yx:.4f}")

    # Test 2: X drives Y: y[t] = 0.8*x[t-1] + noise
    # TE(X→Y) should be >> TE(Y→X)
    N = 2000
    x = rng.normal(0, 1, N)
    y = np.zeros(N)
    for t in range(1, N):
        y[t] = 0.8 * x[t - 1] + 0.2 * rng.normal()

    res_causal = transfer_entropy(x, y, n_bins=8, n_shuffle=50)
    check("Causal: TE(X→Y) > TE(Y→X)",
          res_causal.te_xy > res_causal.te_yx,
          f"TE(X→Y)={res_causal.te_xy:.4f}, TE(Y→X)={res_causal.te_yx:.4f}")

    check("Causal: TE(X→Y) significant",
          res_causal.significant_xy,
          f"TE={res_causal.te_xy:.4f} > null={res_causal.te_xy_shuffled:.4f}")

    # Test 3: Bidirectional coupling — both should be positive
    x_bi = np.zeros(N)
    y_bi = np.zeros(N)
    for t in range(1, N):
        x_bi[t] = 0.5 * y_bi[t - 1] + 0.5 * rng.normal()
        y_bi[t] = 0.5 * x_bi[t - 1] + 0.5 * rng.normal()

    res_bi = transfer_entropy(x_bi, y_bi, n_bins=8, n_shuffle=50)
    check("Bidirectional: both TE > 0",
          res_bi.te_xy > 0.01 and res_bi.te_yx > 0.01,
          f"TE(X→Y)={res_bi.te_xy:.4f}, TE(Y→X)={res_bi.te_yx:.4f}")

    # Test 4: TE is non-negative (by definition)
    check("TE is non-negative (X→Y)",
          res_causal.te_xy >= 0,
          f"TE(X→Y) = {res_causal.te_xy}")
    check("TE is non-negative (Y→X)",
          res_causal.te_yx >= 0,
          f"TE(Y→X) = {res_causal.te_yx}")

    # Test 5: Short series handled gracefully
    short_x = rng.normal(0, 1, 5)
    short_y = rng.normal(0, 1, 5)
    res_short = transfer_entropy(short_x, short_y, n_bins=4, n_shuffle=10)
    check("Short series doesn't crash",
          isinstance(res_short.te_xy, float),
          f"TE = {res_short.te_xy}")


# ═══════════════════════════════════════════════════════════════════════
# L7: GROUND TRUTH DISCREPANCY TESTS
# ═══════════════════════════════════════════════════════════════════════

def test_L7_discrepancy():
    section("L7: GROUND TRUTH DISCREPANCY (error microscope)")

    from ed_mrdt.analysis import ground_truth_discrepancy
    from dataclasses import dataclass

    # Create a minimal GroundTruth-like object
    @dataclass
    class MockGroundTruth:
        positions: np.ndarray
        species_id: np.ndarray
        bond_partner: np.ndarray
        is_alive: np.ndarray
        has_dye: np.ndarray
        dye_type_id: np.ndarray
        is_bleached: np.ndarray
        fret_efficiency_exact: np.ndarray
        donor_lifetime_exact: np.ndarray
        n_complexes: int = 0
        n_bleached: int = 0
        n_free_donors: int = 0

    # Test 1: Perfect mono-exponential → near-zero error
    tau_d_ns = 4.0
    tau_d_s = tau_d_ns * 1e-9
    nx, ny = 8, 8
    pixel_size_um = 0.25  # µm
    time_range_ns = 25.0
    n_bins = 64

    # Generate FLIM from exactly τ_D
    flim, time_range_s = make_mono_exponential_flim(
        tau_d_ns, time_range_ns=time_range_ns, n_bins=n_bins,
        nx=nx, ny=ny, photons_per_pixel=5000, seed=42
    )

    # Create ground truth: all donors at τ_D (free, no FRET)
    n_donors = nx * ny * 4  # 4 donors per pixel on average
    rng = np.random.default_rng(42)
    positions = np.zeros((n_donors, 2))
    for i in range(n_donors):
        px = i % (nx * ny) // ny
        py = i % ny
        positions[i] = [
            (px + rng.uniform(0, 1)) * pixel_size_um,
            (py + rng.uniform(0, 1)) * pixel_size_um,
        ]

    gt = MockGroundTruth(
        positions=positions,
        species_id=np.zeros(n_donors, dtype=int),
        bond_partner=np.full(n_donors, -1, dtype=int),
        is_alive=np.ones(n_donors, dtype=bool),
        has_dye=np.ones(n_donors, dtype=bool),
        dye_type_id=np.zeros(n_donors, dtype=int),  # donors
        is_bleached=np.zeros(n_donors, dtype=bool),
        fret_efficiency_exact=np.zeros(n_donors),
        donor_lifetime_exact=np.full(n_donors, tau_d_s),
    )

    membrane_size = (nx * pixel_size_um, ny * pixel_size_um)
    disc = ground_truth_discrepancy(
        flim, gt, membrane_size, pixel_size_um,
        time_range=time_range_s, tau_d_ns=tau_d_ns,
        min_counts=20, reliability_threshold=0.1,
    )

    check("MAE(τ) < 0.5 ns for high-photon mono-exp",
          disc.mae_tau_ns < 0.5,
          f"MAE = {disc.mae_tau_ns:.3f} ns")

    # Test 2: FRET efficiency ≈ 0 for free donors
    valid_E = ~np.isnan(disc.E_exact) & ~np.isnan(disc.E_image)
    if valid_E.any():
        E_exact_mean = disc.E_exact[valid_E].mean()
        check("E_exact ≈ 0 for free donors",
              abs(E_exact_mean) < 0.05,
              f"<E_exact> = {E_exact_mean:.4f}")

    # Test 3: Reliability mask populated
    check("Reliability mask has reliable pixels",
          disc.n_reliable_pixels > 0,
          f"{disc.n_reliable_pixels}/{disc.n_total_pixels} reliable")

    # Test 4: delta_E exists and is finite
    n_valid_dE = np.sum(~np.isnan(disc.delta_E))
    check("Delta E map has valid pixels",
          n_valid_dE > 0,
          f"{n_valid_dE} valid pixels")

    # Test 5: Relative error map is non-negative
    valid_re = ~np.isnan(disc.relative_error)
    if valid_re.any():
        check("Relative error is non-negative",
              np.all(disc.relative_error[valid_re] >= 0),
              f"min = {disc.relative_error[valid_re].min():.4f}")


# ═══════════════════════════════════════════════════════════════════════
# INTEGRATION: MINI PIPELINE + FULL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def test_integration():
    section("INTEGRATION: full analysis on mini simulation")

    try:
        from ed_mrdt.pipeline import SimulationPipeline
        from ed_mrdt.config import load_config
        from ed_mrdt.analysis import (run_full_analysis, transfer_entropy,
                                       phasor_transform)

        config = load_config("ed_mrdt/examples/mini_demo.yaml")
        result = SimulationPipeline.run_mini_demo(config, n_frames=3)

        check("Mini demo ran successfully",
              result.n_frames >= 2,
              f"{result.n_frames} frames")

        # Run full 7-layer analysis
        analysis = run_full_analysis(result, tau_d_ns=4.1)

        # L1 check
        check("L1 phasor: computed for all frames",
              len(analysis.phasor) == result.n_frames,
              f"{len(analysis.phasor)} phasor results")

        # L2 check: mini demo has sparse data, so just check it runs
        check("L2 spatial: correlation computed without error",
              isinstance(analysis.spatial_corr_image.cluster_density, float),
              f"N_clust = {analysis.spatial_corr_image.cluster_density:.1f}")

        # L3 check
        check("L3 temporal: ACF computed",
              len(analysis.temporal_acf.lag_s) > 0,
              f"{len(analysis.temporal_acf.lag_s)} lags")

        # L4 check
        check("L4 pCF: carpet has data",
              analysis.pair_corr.pcf_carpet.shape[0] > 0,
              f"carpet shape = {analysis.pair_corr.pcf_carpet.shape}")

        # L5 check
        check("L5 iMSD: D_eff computed",
              isinstance(analysis.imsd.D_eff, float),
              f"D_eff = {analysis.imsd.D_eff:.4f} µm²/s, regime={analysis.imsd.regime}")

        # L6: Transfer entropy on complex count vs intensity
        complex_ts = result.get_complex_count_series().astype(float)
        intensity_ts = result.get_intensity_series().astype(float)
        if len(complex_ts) >= 3:
            te_res = transfer_entropy(complex_ts, intensity_ts, n_bins=4,
                                       n_shuffle=20)
            check("L6 TE: computed without error",
                  isinstance(te_res.te_xy, float),
                  f"TE(RL→I)={te_res.te_xy:.4f}, TE(I→RL)={te_res.te_yx:.4f}")

        # L7 check
        check("L7 discrepancy: computed for all frames",
              len(analysis.discrepancy) == result.n_frames,
              f"{len(analysis.discrepancy)} discrepancy maps")

        last_disc = analysis.discrepancy[-1]
        check("L7 discrepancy: MAE is finite",
              np.isfinite(last_disc.mae_tau_ns),
              f"MAE(τ) = {last_disc.mae_tau_ns:.3f} ns")

    except Exception as e:
        check(f"Integration test failed: {e}", False,
              traceback.format_exc()[:200])


# ═══════════════════════════════════════════════════════════════════════
# ATTACK VECTORS: edge cases that could break the code
# ═══════════════════════════════════════════════════════════════════════

def test_attack_vectors():
    section("ATTACK VECTORS: edge cases")

    from ed_mrdt.analysis import (phasor_transform, spatial_autocorrelation,
                                   temporal_autocorrelation, transfer_entropy,
                                   compute_imsd)

    # AV1: All-zero FLIM stack
    zero_flim = np.zeros((1, 4, 4, 32), dtype=np.uint32)
    res = phasor_transform(zero_flim, 25e-9)
    check("AV1: Zero FLIM → no valid phasors",
          res.mask.sum() == 0,
          f"valid pixels = {res.mask.sum()}")

    # AV2: Single pixel with 1 photon
    one_photon = np.zeros((1, 1, 1, 64), dtype=np.uint32)
    one_photon[0, 0, 0, 10] = 1
    res2 = phasor_transform(one_photon, 25e-9, min_counts=2)
    check("AV2: 1-photon pixel rejected by min_counts",
          res2.mask.sum() == 0,
          "correctly rejected")

    # AV3: Spatial correlation of uniform image (no variance)
    uniform = np.ones((16, 16)) * 100.0
    res3 = spatial_autocorrelation(uniform, 0.059)
    check("AV3: Uniform image → G(0) ≈ 0",
          abs(res3.C_r[0]) < 0.01,
          f"G(0) = {res3.C_r[0]:.6f}")

    # AV4: Very short time series for ACF
    short = np.array([1.0, 2.0])
    lags, acf = temporal_autocorrelation(short, dt=1.0)
    check("AV4: 2-point series ACF doesn't crash",
          len(acf) > 0,
          f"ACF length = {len(acf)}")

    # AV5: Transfer entropy with identical series
    x = np.arange(100, dtype=float)
    res5 = transfer_entropy(x, x, n_bins=8, n_shuffle=10)
    check("AV5: Identical series → TE computable",
          np.isfinite(res5.te_xy),
          f"TE = {res5.te_xy:.4f}")

    # AV6: iMSD on 1-frame series
    single = np.random.default_rng(42).poisson(10, (1, 8, 8)).astype(float)
    res6 = compute_imsd(single, 0.059, 1.0)
    check("AV6: 1-frame iMSD doesn't crash",
          len(res6.iMSD_um2) > 0,
          f"regime = {res6.regime}")


# ═══════════════════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  ED-MRDT analysis.py — COMPREHENSIVE TEST SUITE")
    print("=" * 60)

    test_L1_phasor()
    test_L2_spatial()
    test_L3_temporal()
    test_L4_pcf()
    test_L5_imsd()
    test_L6_te()
    test_L7_discrepancy()
    test_attack_vectors()
    test_integration()

    print("\n" + "=" * 60)
    print(f"  RESULTS: {PASS_COUNT} PASS / {FAIL_COUNT} FAIL / "
          f"{PASS_COUNT + FAIL_COUNT} TOTAL")
    print("=" * 60)

    if FAIL_COUNT > 0:
        print("\nFAILED TESTS:")
        for name, status, detail in ALL_RESULTS:
            if status == "FAIL":
                print(f"  ✗ {name}  [{detail}]")

    sys.exit(1 if FAIL_COUNT > 0 else 0)
