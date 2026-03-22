#!/usr/bin/env python3
"""
ED-MRDT v1.1 — PHASE 1, Step 1.2: Test for particles.py

Tests:
  1. ParticleState creates correct number of particles per species
  2. Positions are within membrane bounds
  3. SoA dtypes are numba-compatible
  4. Dye labeling respects labeling_efficiency (statistical)
  5. Species queries return correct subsets
  6. Bond partner initialized to -1 (free)
  7. RAM estimate matches expected
  8. C-contiguity of arrays
  9. Deterministic with fixed seed
  10. Edge cases (zero population species)
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ed_mrdt.config import load_config, SimulationConfig, Species, ConfigError
from ed_mrdt.particles import ParticleState

passed = 0
failed = 0
total = 0

def check(name, condition, detail=""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}")
        if detail:
            print(f"       → {detail}")

print("=" * 70)
print("TEST SUITE: particles.py (Step 1.2)")
print("=" * 70)

# Load canonical config
config = load_config(os.path.join(os.path.dirname(__file__), "..", "ed_mrdt", "examples", "egfr_egf.yaml"))

# ================================================================
# TEST GROUP 1: Basic Creation
# ================================================================
print("\n─── Group 1: Basic Creation ───")

state = ParticleState(config)

check("ParticleState created", state is not None)
check("N = 600", state.N == 600, f"got {state.N}")
check("positions shape", state.positions.shape == (600, 2),
     f"got {state.positions.shape}")
check("species_id shape", state.species_id.shape == (600,))
check("state_id shape", state.state_id.shape == (600,))
check("bond_partner shape", state.bond_partner.shape == (600,))
check("has_dye shape", state.has_dye.shape == (600,))
check("dye_type_id shape", state.dye_type_id.shape == (600,))
check("ctmc_state shape", state.ctmc_state.shape == (600,))
check("is_bleached shape", state.is_bleached.shape == (600,))
check("is_alive shape", state.is_alive.shape == (600,))

# ================================================================
# TEST GROUP 2: Species Populations
# ================================================================
print("\n─── Group 2: Species Populations ───")

n_egfr = np.sum(state.species_id == 0)
n_egf = np.sum(state.species_id == 1)
n_complex = np.sum(state.species_id == 2)

check("EGFR count = 400", n_egfr == 400, f"got {n_egfr}")
check("EGF count = 200", n_egf == 200, f"got {n_egf}")
check("EGFR_EGF count = 0", n_complex == 0, f"got {n_complex}")

# Species ranges
check("EGFR range", state.species_ranges["EGFR"] == (0, 400),
     f"got {state.species_ranges['EGFR']}")
check("EGF range", state.species_ranges["EGF"] == (400, 600),
     f"got {state.species_ranges['EGF']}")

# count_species method
check("count_species EGFR", state.count_species("EGFR") == 400)
check("count_species EGF", state.count_species("EGF") == 200)
check("count_species EGFR_EGF", state.count_species("EGFR_EGF") == 0)

# ================================================================
# TEST GROUP 3: Position Bounds
# ================================================================
print("\n─── Group 3: Position Bounds ───")

Lx, Ly = config.membrane.size
check("All x >= 0", np.all(state.positions[:, 0] >= 0.0))
check("All x < Lx", np.all(state.positions[:, 0] < Lx),
     f"max x = {state.positions[:, 0].max()}")
check("All y >= 0", np.all(state.positions[:, 1] >= 0.0))
check("All y < Ly", np.all(state.positions[:, 1] < Ly),
     f"max y = {state.positions[:, 1].max()}")

# Check spatial uniformity (chi-squared test on 10x10 grid)
nx, ny = 10, 10
hist, _, _ = np.histogram2d(
    state.positions[:, 0], state.positions[:, 1],
    bins=[nx, ny], range=[[0, Lx], [0, Ly]]
)
expected_per_cell = state.N / (nx * ny)
chi2 = np.sum((hist - expected_per_cell)**2 / expected_per_cell)
# With 99 DOF, chi2 < 150 is very conservative p > 0.001
check("Spatial uniformity (chi2)", chi2 < 150,
     f"chi2 = {chi2:.1f}, expected < 150 for 99 DOF")

# ================================================================
# TEST GROUP 4: Initial State Values
# ================================================================
print("\n─── Group 4: Initial States ───")

check("All state_id = 0", np.all(state.state_id == 0))
check("All bond_partner = -1", np.all(state.bond_partner == -1))
check("All is_alive = True", np.all(state.is_alive))
check("All is_bleached = False", np.all(~state.is_bleached))

# CTMC state: should be 0 for labeled, 0 for unlabeled (default)
check("All ctmc_state = 0", np.all(state.ctmc_state == 0))

# ================================================================
# TEST GROUP 5: Dye Labeling
# ================================================================
print("\n─── Group 5: Dye Labeling ───")

# EGFR → Alexa488 (labeling_efficiency = 0.8)
# EGF → Alexa555 (labeling_efficiency = 0.9)
# EGFR_EGF → no attachment (0 particles anyway)

egfr_start, egfr_end = state.species_ranges["EGFR"]
egf_start, egf_end = state.species_ranges["EGF"]

n_egfr_labeled = int(np.sum(state.has_dye[egfr_start:egfr_end]))
n_egf_labeled = int(np.sum(state.has_dye[egf_start:egf_end]))

# Expected: 400 * 0.8 = 320, 200 * 0.9 = 180
# Allow ±5% (statistical fluctuation for Bernoulli)
check("EGFR labeling ~80%",
     abs(n_egfr_labeled - 320) < 50,
     f"got {n_egfr_labeled}, expected ~312 (78% of 400)")
check("EGF labeling ~90%",
     abs(n_egf_labeled - 180) < 40,
     f"got {n_egf_labeled}, expected ~180 (90% of 200)")

# Check dye_type_id for labeled EGFR = 0 (Alexa488)
egfr_labeled_mask = state.has_dye[egfr_start:egfr_end]
egfr_dye_ids = state.dye_type_id[egfr_start:egfr_end][egfr_labeled_mask]
check("EGFR dye = Alexa488 (id=0)", np.all(egfr_dye_ids == 0),
     f"unique ids: {np.unique(egfr_dye_ids)}")

# Check dye_type_id for labeled EGF = 1 (Alexa555)
egf_labeled_mask = state.has_dye[egf_start:egf_end]
egf_dye_ids = state.dye_type_id[egf_start:egf_end][egf_labeled_mask]
check("EGF dye = Alexa555 (id=1)", np.all(egf_dye_ids == 1),
     f"unique ids: {np.unique(egf_dye_ids)}")

# Unlabeled particles should have dye_type_id = -1
egfr_unlabeled = state.dye_type_id[egfr_start:egfr_end][~egfr_labeled_mask]
check("Unlabeled EGFR dye_id = -1", np.all(egfr_unlabeled == -1))

# get_labeled_mask methods
mask_any = state.get_labeled_mask()
check("get_labeled_mask (any)", np.sum(mask_any) == n_egfr_labeled + n_egf_labeled)

mask_488 = state.get_labeled_mask("Alexa488")
check("get_labeled_mask Alexa488", np.sum(mask_488) == n_egfr_labeled,
     f"got {np.sum(mask_488)}")

mask_555 = state.get_labeled_mask("Alexa555")
check("get_labeled_mask Alexa555", np.sum(mask_555) == n_egf_labeled,
     f"got {np.sum(mask_555)}")

# ================================================================
# TEST GROUP 6: Query Methods
# ================================================================
print("\n─── Group 6: Query Methods ───")

pos_egfr = state.get_positions_of("EGFR")
check("get_positions_of EGFR shape", pos_egfr.shape == (400, 2),
     f"got {pos_egfr.shape}")

pos_egf = state.get_positions_of("EGF")
check("get_positions_of EGF shape", pos_egf.shape == (200, 2),
     f"got {pos_egf.shape}")

pos_complex = state.get_positions_of("EGFR_EGF")
check("get_positions_of EGFR_EGF empty", pos_complex.shape == (0, 2),
     f"got {pos_complex.shape}")

check("get_free_mask all free", np.all(state.get_free_mask()))
check("get_bound_mask none bound", np.sum(state.get_bound_mask()) == 0)

# Diffusion coefficient array
D_arr = state.get_D_array()
check("D array shape", D_arr.shape == (600,))
check("D EGFR = 0.05", np.allclose(D_arr[:400], 0.05))
check("D EGF = 10.0", np.allclose(D_arr[400:600], 10.0))
check("get_D single", abs(state.get_D(0) - 0.05) < 1e-15)
check("get_D single EGF", abs(state.get_D(400) - 10.0) < 1e-15)

# ================================================================
# TEST GROUP 7: Data Types (numba-compatible)
# ================================================================
print("\n─── Group 7: Data Types ───")

check("positions dtype float64", state.positions.dtype == np.float64)
check("species_id dtype int32", state.species_id.dtype == np.int32)
check("state_id dtype int32", state.state_id.dtype == np.int32)
check("bond_partner dtype int32", state.bond_partner.dtype == np.int32)
check("has_dye dtype bool", state.has_dye.dtype == np.bool_)
check("dye_type_id dtype int32", state.dye_type_id.dtype == np.int32)
check("ctmc_state dtype int8", state.ctmc_state.dtype == np.int8)
check("is_bleached dtype bool", state.is_bleached.dtype == np.bool_)
check("is_alive dtype bool", state.is_alive.dtype == np.bool_)

# C-contiguity
check("positions C-contiguous", state.positions.flags["C_CONTIGUOUS"])
check("species_id contiguous", state.species_id.flags["C_CONTIGUOUS"])

# ================================================================
# TEST GROUP 8: RAM Estimate
# ================================================================
print("\n─── Group 8: RAM ───")

ram = state.ram_bytes
# N=600: 600*(2*8 + 4 + 4 + 4 + 1 + 4 + 1 + 1 + 1) = 600 * 28 = 420000
expected = 600 * (16 + 4 + 4 + 4 + 1 + 4 + 1 + 1 + 1)  # 36 bytes/particle
# Actually: positions=16, species_id=4, state_id=4, bond_partner=4,
#           has_dye=1, dye_type_id=4, ctmc_state=1, is_bleached=1, is_alive=1 = 36
check(f"RAM = {ram} bytes", ram == 600 * 36,
     f"got {ram}, expected {600 * 36}")
check(f"RAM < 1 MB", ram < 1e6,
     f"got {ram/1e6:.2f} MB")

# ================================================================
# TEST GROUP 9: Deterministic with fixed seed
# ================================================================
print("\n─── Group 9: Determinism ───")

state_a = ParticleState(config, rng=np.random.default_rng(42))
state_b = ParticleState(config, rng=np.random.default_rng(42))

check("Same seed → same positions", np.array_equal(state_a.positions, state_b.positions))
check("Same seed → same dye labels", np.array_equal(state_a.has_dye, state_b.has_dye))

state_c = ParticleState(config, rng=np.random.default_rng(99))
check("Different seed → different positions",
     not np.array_equal(state_a.positions, state_c.positions))

# ================================================================
# TEST GROUP 10: numba arrays dict
# ================================================================
print("\n─── Group 10: Numba Interface ───")

arrays = state.get_numba_arrays()
check("numba dict has positions", "positions" in arrays)
check("numba dict has D_per_species", "D_per_species" in arrays)
check("D_per_species shape", arrays["D_per_species"].shape == (3,),
     f"got {arrays['D_per_species'].shape}")
check("D_per_species values", np.allclose(arrays["D_per_species"], [0.05, 10.0, 0.04]))

# ================================================================
# TEST GROUP 11: __repr__
# ================================================================
print("\n─── Group 11: Repr ───")

r = repr(state)
check("repr contains ParticleState", "ParticleState" in r)
check("repr contains EGFR", "EGFR" in r)
check("repr contains RAM", "MB" in r)

# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 70)
print(f"RESULTS: {passed}/{total} passed, {failed} failed")
print("=" * 70)

if failed == 0:
    print("🎉 ALL TESTS PASSED — Step 1.2 COMPLETED")
else:
    print(f"⚠️  {failed} test(s) failed — needs fixing")
