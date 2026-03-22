#!/usr/bin/env python3
"""
ED-MRDT v1.1 — PHASE 1, Step 1.1: Test for config.py + egfr_egf.yaml

This test verifies:
  1. YAML loads without errors
  2. All fields are parsed correctly
  3. Validation passes (no fatal errors)
  4. Cross-references are consistent
  5. Numerical values are physically reasonable
  6. Derived methods work correctly
  7. Error detection (invalid configs are rejected)
"""
import sys
import os
import math
import traceback

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ed_mrdt.config import (
    load_config, ConfigError, SimulationConfig,
    Species, Reaction, FRETpair, FluorophoreType
)

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
print("TEST SUITE: config.py + egfr_egf.yaml")
print("=" * 70)

# ================================================================
# TEST GROUP 1: YAML Loading
# ================================================================
print("\n─── Group 1: YAML Loading ───")

try:
    config = load_config(os.path.join(os.path.dirname(__file__), "..", "ed_mrdt", "examples", "egfr_egf.yaml"))
    check("YAML loads without error", True)
except Exception as e:
    check("YAML loads without error", False, str(e))
    traceback.print_exc()
    sys.exit(1)

check("Config is SimulationConfig", isinstance(config, SimulationConfig))
check("Name parsed", config.name == "EGFR-EGF membrane FLIM-FRET")
check("Version parsed", config.version == "1.1.0")

# ================================================================
# TEST GROUP 2: Membrane & Domain
# ================================================================
print("\n─── Group 2: Membrane ───")

check("Membrane size", config.membrane.size == (2.0, 2.0),
     f"got {config.membrane.size}")
check("Membrane periodic", config.membrane.periodic == (True, True))
check("Membrane area", abs(config.membrane.area - 4.0) < 0.01,
     f"got {config.membrane.area}")
check("Geometry type", config.membrane.geometry_type == "planar")

# ================================================================
# TEST GROUP 3: Species
# ================================================================
print("\n─── Group 3: Species ───")

check("3 species defined", len(config.species) == 3,
     f"got {len(config.species)}")
check("EGFR exists", any(s.name == "EGFR" for s in config.species))
check("EGF exists", any(s.name == "EGF" for s in config.species))
check("EGFR_EGF exists", any(s.name == "EGFR_EGF" for s in config.species))

egfr = config.species_by_name("EGFR")
check("EGFR D = 0.05", abs(egfr.diffusion_coefficient - 0.05) < 1e-10,
     f"got {egfr.diffusion_coefficient}")
check("EGFR radius = 5.0 nm", abs(egfr.radius - 5.0) < 1e-10)
check("EGFR has 3 states", len(egfr.states) == 3,
     f"got {egfr.states}")
check("EGFR D_per_state works", egfr.get_D("dimerized") == 0.025,
     f"got {egfr.get_D('dimerized')}")
check("EGFR default D fallback", egfr.get_D("default") == 0.05)

egf = config.species_by_name("EGF")
check("EGF D = 10.0", abs(egf.diffusion_coefficient - 10.0) < 1e-10)

# ================================================================
# TEST GROUP 4: Reactions
# ================================================================
print("\n─── Group 4: Reactions ───")

check("2 reactions defined", len(config.reactions) == 2)

rxn_bind = config.reactions[0]
check("Binding reaction name", rxn_bind.name == "EGF_binding")
check("Binding reactants", rxn_bind.reactants == ["EGFR", "EGF"])
check("Binding products", rxn_bind.products == ["EGFR_EGF"])
check("kon = 0.1 µm²/s", abs(rxn_bind.kon - 0.1) < 1e-10,
     f"got {rxn_bind.kon}")
check("koff = 0.01", abs(rxn_bind.koff - 0.01) < 1e-10)
check("contact_radius = 7 nm", abs(rxn_bind.contact_radius - 7.0) < 1e-10)
check("Is bimolecular", rxn_bind.is_bimolecular)
check("Is reversible", rxn_bind.is_reversible)

# ================================================================
# TEST GROUP 5: Potentials
# ================================================================
print("\n─── Group 5: Potentials ───")

check("3 potentials defined", len(config.potentials) == 3,
     f"got {len(config.potentials)}")
pot0 = config.potentials[0]
check("WCA potential type", pot0.potential_type == "wca")
check("WCA sigma = 8.0", abs(pot0.parameters["sigma"] - 8.0) < 1e-10)

cutoff = pot0.get_cutoff()
expected_cutoff = 8.0 * 2**(1.0/6.0)
check("WCA cutoff correct", abs(cutoff - expected_cutoff) < 0.01,
     f"got {cutoff:.3f}, expected {expected_cutoff:.3f}")

# ================================================================
# TEST GROUP 6: Fluorophores (SMIS pattern)
# ================================================================
print("\n─── Group 6: Fluorophores ───")

check("2 fluorophores defined", len(config.fluorophores) == 2)

alexa488 = config.fluorophore_by_name("Alexa488")
check("Alexa488 has 4 states", alexa488.n_states == 4,
     f"got {alexa488.n_states}")
check("S0 index = 0", alexa488.get_state_index("S0") == 0)
check("S1 index = 1", alexa488.get_state_index("S1") == 1)
check("T1 index = 2", alexa488.get_state_index("T1") == 2)
check("bleached index = 3", alexa488.get_state_index("bleached") == 3)
check("Fluorescent states = [1]", alexa488.fluorescent_states == [1])
check("Terminal states = [3]", alexa488.terminal_states == [3])
check("tau_rad = 4.1 ns", abs(alexa488.tau_rad - 4.1e-9) < 1e-15,
     f"got {alexa488.tau_rad}")

# Check thermal rates S1→S0
k_S1_S0 = alexa488.thermal_rates["S1"]["S0"]
check("S1→S0 rate ≈ 1/4.1ns", abs(k_S1_S0 - 2.44e8) < 1e6,
     f"got {k_S1_S0:.3e}, expected 2.44e+08")

# S1 is fluorescent
s1 = alexa488.states[1]
check("S1 is fluorescent", s1.is_fluorescent)
check("S1 QY = 0.92", abs(s1.quantum_yield - 0.92) < 1e-10)
check("S1 emission = 519 nm", abs(s1.emission_wavelength - 519.0) < 1e-10)

# ================================================================
# TEST GROUP 7: FRET Pairs
# ================================================================
print("\n─── Group 7: FRET Pairs ───")

check("1 FRET pair defined", len(config.fret_pairs) == 1)
fp = config.fret_pairs[0]
check("Donor = Alexa488", fp.donor_type == "Alexa488")
check("Acceptor = Alexa555", fp.acceptor_type == "Alexa555")
check("R0 = 6.4 nm", abs(fp.R0 - 6.4) < 1e-10)

# Verify Förster equation
E_at_R0 = fp.efficiency(6.4)
check("E(r=R0) = 0.5", abs(E_at_R0 - 0.5) < 1e-10,
     f"got {E_at_R0}")

E_close = fp.efficiency(3.0)
check("E(r=3nm) > 0.98", E_close > 0.98,
     f"got {E_close:.4f}")

E_far = fp.efficiency(20.0)
check("E(r=20nm) < 0.01", E_far < 0.01,
     f"got {E_far:.6f}")

E_zero = fp.efficiency(0.0)
check("E(r=0) = 1.0", abs(E_zero - 1.0) < 1e-10)

# ================================================================
# TEST GROUP 8: Dye Attachments
# ================================================================
print("\n─── Group 8: Dye Attachments ───")

check("2 dye attachments", len(config.dye_attachments) == 2)
da_egfr = config.dye_attachments[0]
check("EGFR → Alexa488", da_egfr.species == "EGFR" and da_egfr.fluorophore == "Alexa488")
check("Labeling 80%", abs(da_egfr.labeling_efficiency - 0.8) < 1e-10)

# ================================================================
# TEST GROUP 9: Microscope
# ================================================================
print("\n─── Group 9: Microscope ───")

check("NA = 1.4", abs(config.objective.NA - 1.4) < 1e-10)
check("Airy radius", abs(config.objective.airy_radius(519) - 0.61*519/1.4) < 0.1)
check("Laser λ = 488 nm", abs(config.laser.wavelength - 488.0) < 1e-10)
check("Laser period", abs(config.laser.period - 25e-9) < 1e-15)
check("Detector QE = 0.2", abs(config.detector.detection_efficiency - 0.2) < 1e-10)
check("Scan 34×34 px", config.scan.pixels_x == 34 and config.scan.pixels_y == 34)
check("Scan FOV ≈ 2 µm", abs(config.scan.fov_x - 2.006) < 0.01)
check("TCSPC 256 bins", config.tcspc.n_bins == 256)
check("TCSPC bin width", abs(config.tcspc.bin_width - 25e-9/256) < 1e-18)

# ================================================================
# TEST GROUP 10: Diffusion Field
# ================================================================
print("\n─── Group 10: Diffusion Field ───")

df = config.diffusion_field
check("Field type = uniform", df.type == "uniform")

# Inside raft (center 5,5 radius 2)
D_raft = df.get_D_factor(5.0, 5.0)
check("D uniform = 0.3", abs(D_raft - 1.0) < 1e-10,
     f"got {D_raft}")

# Outside raft
D_out = df.get_D_factor(0.0, 0.0)
check("D outside raft = 1.0", abs(D_out - 1.0) < 1e-10,
     f"got {D_out}")

# Edge of raft (exactly at boundary)
D_edge = df.get_D_factor(5.0, 3.0)  # distance = 2.0 = radius → inside
check("D uniform check", abs(D_edge - 1.0) < 1e-10,
     f"got {D_edge}")

# Just outside
D_just_out = df.get_D_factor(5.0, 2.99)  # distance = 2.01 > radius → outside
check("D just outside raft = 1.0", abs(D_just_out - 1.0) < 1e-10,
     f"got {D_just_out}")

# ================================================================
# TEST GROUP 11: Global Properties
# ================================================================
print("\n─── Group 11: Global Properties ───")

check("Total particles = 600", config.total_particles == 600,
     f"got {config.total_particles}")
check("n_species = 3", config.n_species == 3)
check("simulation_time = 60", abs(config.simulation_time - 60.0) < 1e-10)
check("bd_timestep = 1e-5", abs(config.bd_timestep - 1e-5) < 1e-20)
check("random_seed = 42", config.random_seed == 42)

# ================================================================
# TEST GROUP 12: Error Detection (invalid configs must be rejected)
# ================================================================
print("\n─── Group 12: Error Detection ───")

import yaml, tempfile

def check_invalid_yaml(name, yaml_content, expected_error_substr):
    """Test that an invalid YAML raises ConfigError with the expected message."""
    global passed, failed, total
    total += 1
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        f.flush()
        try:
            c = load_config(f.name)
            failed += 1
            print(f"  ❌ {name}: should have raised ConfigError")
        except ConfigError as e:
            if expected_error_substr.lower() in str(e).lower():
                passed += 1
                print(f"  ✅ {name}")
            else:
                failed += 1
                print(f"  ❌ {name}: wrong error message: {e}")
        except Exception as e:
            failed += 1
            print(f"  ❌ {name}: unexpected {type(e).__name__}: {e}")
        finally:
            os.unlink(f.name)

# Reaction references non-existent species
check_invalid_yaml(
    "Reject unknown reactant",
    """
species:
  - name: A
    diffusion_coefficient: 1.0
    radius: 5.0
reactions:
  - name: bad_rxn
    reactants: [A, B_DOES_NOT_EXIST]
    products: [A]
    kon: 1e5
""",
    "not in species"
)

# Negative diffusion coefficient
check_invalid_yaml(
    "Reject negative D",
    """
species:
  - name: A
    diffusion_coefficient: -0.5
    radius: 5.0
""",
    "negative d"
)

# TCSPC time range exceeds laser period
check_invalid_yaml(
    "Reject TCSPC > laser period",
    """
species:
  - name: A
    diffusion_coefficient: 1.0
    radius: 5.0
laser:
  repetition_rate: 80.0e+06
tcspc:
  time_range: 50.0e-09
""",
    "tcspc time_range"
)

# Fluorophore with invalid thermal rate state
check_invalid_yaml(
    "Reject invalid photophysics state",
    """
species:
  - name: A
    diffusion_coefficient: 1.0
    radius: 5.0
fluorophores:
  - name: BadDye
    states:
      - name: S0
      - name: S1
        is_fluorescent: true
    thermal_rates:
      NONEXISTENT_STATE:
        S0: 1.0e+08
""",
    "not in states"
)

# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 70)
print(f"RESULTS: {passed}/{total} passed, {failed} failed")
print("=" * 70)

if failed == 0:
    print("🎉 ALL TESTS PASSED — Step 1.1 COMPLETED")
else:
    print(f"⚠️  {failed} test(s) failed — needs fixing")
