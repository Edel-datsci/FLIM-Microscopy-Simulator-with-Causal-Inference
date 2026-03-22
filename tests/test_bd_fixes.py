#!/usr/bin/env python3
"""
Tests for BD integrator fixes:
  1. Harmonic repulsion force correctness
  2. Harmonic bounded vs WCA divergence
  3. Displacement cap prevents explosion
  4. Smooth D-field gradient (sigmoid, not step)
  5. Equilibrium uniform density with ∇D correction
  6. Backward-compat MSD = 4Dt ± 5%

References:
  Ermak & McCammon (1978) J Chem Phys 69:1352
  Lau & Lubensky (2007) PRE 76:011123
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ed_mrdt.dynamics import (
    bd_step,
    compute_forces_wca,
    compute_forces_harmonic,
    apply_diffusion_field_regions_smooth,
)

passed = 0
failed = 0
total = 0

def check(name, condition, detail=""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  ✅ {name}" + (f"  [{detail}]" if detail else ""))
    else:
        failed += 1
        print(f"  ✗ {name}" + (f"  [{detail}]" if detail else ""))


# ═══════════════════════════════════════════════════════════════
print("─── Test 1: Harmonic force correctness ───")
# F = k*(sigma - r) * r_hat  for r < sigma
# ═══════════════════════════════════════════════════════════════
N_h = 2
positions_h = np.array([[0.0, 0.0], [0.006, 0.0]], dtype=np.float64)  # r = 6nm = 0.006 µm
forces_h = np.zeros((N_h, 2), dtype=np.float64)
species_h = np.zeros(N_h, dtype=np.int32)
alive_h = np.ones(N_h, dtype=np.bool_)
pairs_h = np.array([[0, 1]], dtype=np.int32)

sigma_h = 0.008  # 8nm in µm
k_spring = 50.0  # kBT/µm²  (tuned)
sigma_mat_h = np.array([[sigma_h]], dtype=np.float64)
k_mat_h = np.array([[k_spring]], dtype=np.float64)

compute_forces_harmonic(
    positions_h, forces_h, species_h, alive_h,
    pairs_h, 1,
    sigma_mat_h, k_mat_h,
    100.0, 100.0, True, True,
)

r_test = 0.006  # µm
expected_F = k_spring * (sigma_h - r_test)  # = 50 * 0.002 = 0.1 kBT/µm
F_on_0 = forces_h[0, 0]  # particle 0 pushed left (negative x)
F_on_1 = forces_h[1, 0]  # particle 1 pushed right (positive x)

check("F magnitude correct",
      abs(abs(F_on_1) - expected_F) / expected_F < 0.01,
      f"F={F_on_1:.6f}, expected={expected_F:.6f}")
check("Newton 3rd law: F0 = -F1",
      abs(F_on_0 + F_on_1) < 1e-12,
      f"F0={F_on_0:.6f}, F1={F_on_1:.6f}")
check("Direction: particle 1 pushed right",
      F_on_1 > 0,
      f"F1_x={F_on_1:.6f}")
check("No force in y",
      abs(forces_h[0, 1]) < 1e-12 and abs(forces_h[1, 1]) < 1e-12)


# ═══════════════════════════════════════════════════════════════
print("\n─── Test 2: Harmonic bounded vs WCA divergence ───")
# At r = 0.1*sigma, harmonic gives finite F; WCA explodes
# ═══════════════════════════════════════════════════════════════
r_overlap = 0.1 * sigma_h  # deep overlap
pos_ov = np.array([[0.0, 0.0], [r_overlap, 0.0]], dtype=np.float64)

# Harmonic
f_harm = np.zeros((2, 2), dtype=np.float64)
compute_forces_harmonic(
    pos_ov, f_harm, species_h, alive_h,
    pairs_h, 1, sigma_mat_h, k_mat_h,
    100.0, 100.0, True, True,
)
F_harmonic_mag = abs(f_harm[1, 0])
F_harmonic_max = k_spring * sigma_h  # theoretical maximum
check("Harmonic force bounded",
      F_harmonic_mag <= F_harmonic_max * 1.01,
      f"|F_harm|={F_harmonic_mag:.4f} <= F_max={F_harmonic_max:.4f}")

# WCA (with force cap) — should be capped, not exploding
eps_h = 2.5
eps_mat_h = np.array([[eps_h]], dtype=np.float64)
f_wca = np.zeros((2, 2), dtype=np.float64)
compute_forces_wca(
    pos_ov, f_wca, species_h, alive_h,
    pairs_h, 1, sigma_mat_h, eps_mat_h,
    100.0, 100.0, True, True,
)
F_wca_mag = abs(f_wca[1, 0])
# With our force cap, WCA should be capped at 2400*eps/sigma^2
wca_cap = 2400.0 * eps_h / (sigma_h * sigma_h)  # force cap as f_over_r
# Actual force = f_over_r * dx where dx = r_overlap
F_wca_expected_cap = wca_cap * r_overlap
check("WCA force capped (not infinite)",
      F_wca_mag < 1e12,
      f"|F_wca|={F_wca_mag:.2e}")
check("Harmonic << WCA at deep overlap",
      F_harmonic_mag < F_wca_mag,
      f"harm={F_harmonic_mag:.4f} vs wca={F_wca_mag:.2e}")


# ═══════════════════════════════════════════════════════════════
print("\n─── Test 3: Displacement cap prevents explosion ───")
# ═══════════════════════════════════════════════════════════════
N_cap = 1
pos_cap = np.array([[50.0, 50.0]], dtype=np.float64)
pos_before = pos_cap.copy()
D_cap = np.array([1.0], dtype=np.float64)
# Extreme force that would send particle flying
forces_extreme = np.array([[1e8, 0.0]], dtype=np.float64)
grad_D_zero = np.zeros((1, 2), dtype=np.float64)
noise_cap = np.zeros((1, 2), dtype=np.float64)
alive_cap = np.ones(1, dtype=np.bool_)
max_disp = 0.004  # 4nm cap

bd_step(
    pos_cap, D_cap, forces_extreme, grad_D_zero, 1e-3,
    100.0, 100.0, True, True,
    N_cap, noise_cap, alive_cap, max_disp,
)

displacement = np.sqrt((pos_cap[0, 0] - pos_before[0, 0])**2 +
                        (pos_cap[0, 1] - pos_before[0, 1])**2)
check("Displacement capped",
      displacement <= max_disp * 1.001,
      f"|Δx|={displacement:.6f} <= cap={max_disp:.6f}")
check("Particle moved (not zero)",
      displacement > 0.0,
      f"|Δx|={displacement:.6f}")


# ═══════════════════════════════════════════════════════════════
print("\n─── Test 4: Smooth D-field gradient ───")
# Sigmoid transition — ∇D should be smooth, not infinite
# ═══════════════════════════════════════════════════════════════
N_df = 100
D_base_scalar = 1.0
# Particles spread radially from a region center
region_cx = np.array([50.0], dtype=np.float64)
region_cy = np.array([50.0], dtype=np.float64)
region_r = np.array([10.0], dtype=np.float64)  # 10 µm radius
region_Df = np.array([0.3], dtype=np.float64)  # D inside = 0.3

# Line of particles along x through region center
xs = np.linspace(35.0, 65.0, N_df)
positions_df = np.column_stack([xs, np.full(N_df, 50.0)]).astype(np.float64)
species_df = np.zeros(N_df, dtype=np.int32)
D_base_arr = np.array([D_base_scalar], dtype=np.float64)  # per-species array
D_per = np.full(N_df, D_base_scalar, dtype=np.float64)
grad_D = np.zeros((N_df, 2), dtype=np.float64)

smoothing_w = 0.5  # 0.5 µm = 500nm

apply_diffusion_field_regions_smooth(
    D_per, grad_D, D_base_arr, species_df, positions_df, N_df,
    region_cx, region_cy, region_r, region_Df, 1,
    smoothing_w,
)

# Check D transitions smoothly
D_inside = D_per[N_df // 2]  # center
D_outside = D_per[0]         # far left
check("D inside region ≈ 0.3",
      abs(D_inside - 0.3) < 0.05,
      f"D_center={D_inside:.4f}")
check("D outside region ≈ 1.0",
      abs(D_outside - 1.0) < 0.05,
      f"D_edge={D_outside:.4f}")

# ∇D should be finite everywhere (no infinities)
max_gradD = np.max(np.abs(grad_D))
check("∇D finite everywhere",
      np.all(np.isfinite(grad_D)),
      f"max|∇D|={max_gradD:.4f}")

# ∇D should be non-zero near boundary (transition region)
# Boundary is at d = region_r = 10 µm from center
# Particles near boundary: those at distance ~10 from center
boundary_mask = np.abs(np.sqrt((xs - 50)**2) - 10.0) < 2.0
gradD_at_boundary = np.abs(grad_D[boundary_mask, 0])
check("∇D non-zero at boundary",
      np.max(gradD_at_boundary) > 0.01,
      f"max|∇D_x| at boundary = {np.max(gradD_at_boundary):.4f}")

# ∇D should be ≈0 far from boundary
center_mask = np.abs(xs - 50.0) < 3.0
gradD_center = np.abs(grad_D[center_mask, 0])
check("∇D ≈ 0 far from boundary (center)",
      np.max(gradD_center) < 0.01,
      f"max|∇D_x| at center = {np.max(gradD_center):.6f}")


# ═══════════════════════════════════════════════════════════════
print("\n─── Test 5: Equilibrium uniform with ∇D correction ───")
# With Ermak-McCammon ∇D correction, particles should NOT
# accumulate in low-D regions. Without it, P(x) ∝ 1/√D(x).
# ═══════════════════════════════════════════════════════════════
N_eq = 500
Lx_eq = 20.0
Ly_eq = 20.0
rng = np.random.default_rng(42)

# Region at center: D_inside = 0.2, D_outside = 1.0
region_cx_eq = np.array([Lx_eq / 2], dtype=np.float64)
region_cy_eq = np.array([Ly_eq / 2], dtype=np.float64)
region_r_eq = np.array([5.0], dtype=np.float64)
region_Df_eq = np.array([0.2], dtype=np.float64)
D_base_eq = 1.0
D_base_eq_arr = np.array([D_base_eq], dtype=np.float64)
sm_w = 0.3

species_eq = np.zeros(N_eq, dtype=np.int32)
alive_eq = np.ones(N_eq, dtype=np.bool_)
forces_eq = np.zeros((N_eq, 2), dtype=np.float64)

# Start uniformly distributed
pos_eq = np.column_stack([
    rng.uniform(0, Lx_eq, N_eq),
    rng.uniform(0, Ly_eq, N_eq),
]).astype(np.float64)

D_eq = np.full(N_eq, D_base_eq, dtype=np.float64)
gD_eq = np.zeros((N_eq, 2), dtype=np.float64)

dt_eq = 1e-4
n_steps_eq = 5000

for _ in range(n_steps_eq):
    # Compute D(x) and ∇D(x) with smooth field
    D_eq[:] = D_base_eq
    gD_eq[:] = 0.0
    apply_diffusion_field_regions_smooth(
        D_eq, gD_eq, D_base_eq_arr, species_eq, pos_eq, N_eq,
        region_cx_eq, region_cy_eq, region_r_eq, region_Df_eq, 1,
        sm_w,
    )
    noise_eq = rng.standard_normal((N_eq, 2))
    bd_step(
        pos_eq, D_eq, forces_eq, gD_eq, dt_eq,
        Lx_eq, Ly_eq, True, True,
        N_eq, noise_eq, alive_eq, 1.0,
    )

# Count particles inside vs outside region
dist_from_center = np.sqrt((pos_eq[:, 0] - Lx_eq/2)**2 +
                            (pos_eq[:, 1] - Ly_eq/2)**2)
n_inside = np.sum(dist_from_center < 5.0)
area_inside = np.pi * 5.0**2
area_outside = Lx_eq * Ly_eq - area_inside
density_inside = n_inside / area_inside
density_outside = (N_eq - n_inside) / area_outside

# With ∇D correction: density should be roughly uniform
# Ratio should be close to 1.0 (within statistical fluctuation)
density_ratio = density_inside / density_outside if density_outside > 0 else float('inf')
check("Density ratio ≈ 1 (uniform equilibrium)",
      0.5 < density_ratio < 2.0,
      f"ρ_in/ρ_out = {density_ratio:.3f}")


# ═══════════════════════════════════════════════════════════════
print("\n─── Test 6: Backward-compat MSD = 4Dt ± 5% ───")
# With default parameters (grad_D=0, max_disp=∞), new bd_step
# must reproduce the standard free-diffusion result.
# ═══════════════════════════════════════════════════════════════
N_bc = 2000
D_bc = 0.1
dt_bc = 1e-4
n_steps_bc = 1000
t_total_bc = dt_bc * n_steps_bc  # 0.1 s

rng_bc = np.random.default_rng(999)
pos_bc = np.column_stack([
    rng_bc.uniform(10, 90, N_bc),
    rng_bc.uniform(10, 90, N_bc),
]).astype(np.float64)
pos_init_bc = pos_bc.copy()

D_arr_bc = np.full(N_bc, D_bc, dtype=np.float64)
forces_bc = np.zeros((N_bc, 2), dtype=np.float64)
gD_bc = np.zeros((N_bc, 2), dtype=np.float64)
alive_bc = np.ones(N_bc, dtype=np.bool_)
Lx_bc = 100.0
Ly_bc = 100.0

for _ in range(n_steps_bc):
    noise_bc = rng_bc.standard_normal((N_bc, 2))
    bd_step(
        pos_bc, D_arr_bc, forces_bc, gD_bc, dt_bc,
        Lx_bc, Ly_bc, True, True,
        N_bc, noise_bc, alive_bc, 1e6,  # huge cap = no cap
    )

# Unwrap PBC
disp_bc = pos_bc - pos_init_bc
for i in range(N_bc):
    if disp_bc[i, 0] > Lx_bc / 2: disp_bc[i, 0] -= Lx_bc
    elif disp_bc[i, 0] < -Lx_bc / 2: disp_bc[i, 0] += Lx_bc
    if disp_bc[i, 1] > Ly_bc / 2: disp_bc[i, 1] -= Ly_bc
    elif disp_bc[i, 1] < -Ly_bc / 2: disp_bc[i, 1] += Ly_bc

msd_bc = np.mean(disp_bc[:, 0]**2 + disp_bc[:, 1]**2)
msd_expected = 4.0 * D_bc * t_total_bc
msd_error = abs(msd_bc - msd_expected) / msd_expected * 100

check("MSD = 4Dt ± 5%",
      msd_error < 5.0,
      f"MSD={msd_bc:.6f}, expected={msd_expected:.6f}, err={msd_error:.2f}%")


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"RESULTS: {passed}/{total} passed, {failed} failed")
print(f"{'='*70}")
if failed == 0:
    print("🎉 ALL BD-FIX TESTS PASSED")
else:
    print(f"⚠️  {failed} test(s) failed")
