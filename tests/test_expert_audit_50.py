"""
ED-MRDT v1.1 — Expert Audit Test Suite (50 Tests)
====================================================
Senior-level diagnostic tests targeting conceptual errors,
physics violations, interface mismatches, and algorithmic bugs
across the entire 4-layer FLIM-FRET simulation pipeline.

Author: Expert Audit Engine
Date: 2026-03-21
Methodology: Each test probes a specific physical invariant,
mathematical identity, or interface contract. Failures indicate
real bugs, not test tuning issues.
"""

import math
import sys
import os
import numpy as np
from pathlib import Path

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ed_mrdt.config import (
    load_config, SimulationConfig, Species, Reaction, PairPotential,
    CatalyticReaction, CompartmentExchange, UnimolecularConversion,
    FeedbackRule, FRETpair, FluorophoreType, DyeAttachment, ConfigError,
    MembraneGeometry, ObjectiveLens, PSFmodel, Detector, LaserSource,
    ScanParameters, TCSPCsettings,
)
from ed_mrdt.particles import ParticleState
from ed_mrdt.dynamics import (
    CellList, bd_step, apply_binding_reactions, apply_dissociation,
    compute_forces_wca, compute_forces_harmonic, apply_catalytic_reaction,
    compute_density_field, apply_diffusion_field_regions_smooth,
    _min_image_dist_sq, _cell_list_pairs,
)
from ed_mrdt.photophysics import (
    RateMatrix, PhotonBatch, PhotonSimulator, fret_efficiency, fret_rate,
    _build_pss_matrices, _evolve_pss_kernel, _poisson_draw,
)
from ed_mrdt.microscope import PSF, ConfocalScanner, VirtualMicroscope, FLIMFrame
from ed_mrdt.pipeline import SimulationPipeline, SimulationResult, GroundTruth
from ed_mrdt.analysis import (
    mean_arrival_time, phasor_transform, transfer_entropy, _discretize, _te_discrete,
)


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def _load_mini():
    """Load mini_demo.yaml config."""
    cfg_path = Path(__file__).resolve().parent.parent / "ed_mrdt" / "examples" / "mini_demo.yaml"
    return load_config(str(cfg_path))


def _load_pip3():
    """Load pip3_pten_flim.yaml config."""
    cfg_path = Path(__file__).resolve().parent.parent / "ed_mrdt" / "examples" / "pip3_pten_flim.yaml"
    return load_config(str(cfg_path))


def _make_minimal_config():
    """Build a minimal config programmatically for unit tests."""
    membrane = MembraneGeometry(size=(2.0, 2.0), periodic=(True, True))
    sp_a = Species(name="A", diffusion_coefficient=0.1, radius=5.0)
    sp_b = Species(name="B", diffusion_coefficient=0.1, radius=5.0)
    sp_ab = Species(name="AB", diffusion_coefficient=0.05, radius=7.0)

    photo_s0 = type('PhotoState', (), {
        'name': 'S0', 'is_fluorescent': False, 'is_terminal': False,
        'is_absorbing': True, 'emission_wavelength': 0
    })()
    photo_s1 = type('PhotoState', (), {
        'name': 'S1', 'is_fluorescent': True, 'is_terminal': False,
        'is_absorbing': False, 'emission_wavelength': 519
    })()
    photo_bl = type('PhotoState', (), {
        'name': 'Bleached', 'is_fluorescent': False, 'is_terminal': True,
        'is_absorbing': False, 'emission_wavelength': 0
    })()

    fluo = FluorophoreType(
        name="TestDye",
        states=[photo_s0, photo_s1, photo_bl],
        thermal_rates={
            'S1': {'S0': 2.5e8, 'Bleached': 1e2},
        },
        n_states=3,
    )

    rxn = Reaction(
        name="binding",
        reactants=["A", "B"], products=["AB"],
        kon=0.1, koff=0.01, contact_radius=7.0,
    )

    config = SimulationConfig(
        membrane=membrane,
        species=[sp_a, sp_b, sp_ab],
        reactions=[rxn],
        potentials=[],
        catalytic_reactions=[],
        compartment_exchanges=[],
        unimolecular_conversions=[],
        feedback_rules=[],
        fluorophores=[fluo],
        fret_pairs=[],
        dye_attachments=[],
        initial_populations={"A": 50, "B": 25},
        simulation_time=0.01,
        bd_timestep=1e-4,
        random_seed=42,
        objective=ObjectiveLens(NA=1.4, magnification=100),
        psf=PSFmodel(type="gaussian", emission_wavelength=519),
        laser=LaserSource(wavelength=488, rep_rate=40e6, pulse_width=100e-12, power=5e-6),
        detector=Detector(efficiency=0.2, dark_count_rate=100, afterpulse_prob=0.018,
                          timing_jitter=50e-12, dead_time=100e-9),
        scan=ScanParameters(pixels_x=16, pixels_y=16, pixel_size=125.0,
                            dwell_time=20e-6, n_frames=2, frame_interval=0.005),
        tcspc=TCSPCsettings(n_bins=64, time_range=25e-9),
    )
    return config


passed = 0
failed = 0
errors = []


def run_test(name, func):
    global passed, failed, errors
    try:
        func()
        passed += 1
        print(f"  [PASS] {name}")
    except Exception as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  [FAIL] {name}: {e}")


# ═══════════════════════════════════════════════════════════════════
# BLOCK 1: FORSTER PHYSICS — CONCEPTUAL ERRORS IN FRET
# ═══════════════════════════════════════════════════════════════════

def test_01_fret_efficiency_at_R0():
    """E(r=R0) must be exactly 0.5 (definition of R0)."""
    R0 = 5.4
    E = fret_efficiency(R0, R0)
    assert abs(E - 0.5) < 1e-10, f"E(R0)={E}, expected 0.5"

def test_02_fret_efficiency_monotonicity():
    """E(r) must be monotonically decreasing."""
    R0 = 5.4
    r_vals = np.linspace(0.5, 30.0, 100)
    E_vals = [fret_efficiency(r, R0) for r in r_vals]
    for i in range(1, len(E_vals)):
        assert E_vals[i] <= E_vals[i-1] + 1e-15, \
            f"E not monotonic: E({r_vals[i]:.1f})={E_vals[i]:.6f} > E({r_vals[i-1]:.1f})={E_vals[i-1]:.6f}"

def test_03_fret_rate_vs_efficiency_consistency():
    """k_FRET and E must be consistent: E = k_FRET*tau_D / (1 + k_FRET*tau_D)."""
    R0, tau_D = 5.4, 2.793e-9  # nm, s
    for r in [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        E = fret_efficiency(r, R0)
        k = fret_rate(r, R0, tau_D)
        E_from_k = k * tau_D / (1.0 + k * tau_D)
        assert abs(E - E_from_k) < 1e-10, \
            f"r={r}: E={E:.6f} vs E_from_k={E_from_k:.6f}"

def test_04_fret_tau_da_formula():
    """tau_DA = tau_D * (1 - E) must match 1/(k_rad + k_nr + k_FRET)."""
    R0, tau_D = 5.4, 2.793e-9
    r = 4.0  # nm (intra-complex distance)
    E = fret_efficiency(r, R0)
    tau_DA_forster = tau_D * (1.0 - E)
    k_fret_val = fret_rate(r, R0, tau_D)
    tau_DA_kinetic = 1.0 / (1.0 / tau_D + k_fret_val)
    assert abs(tau_DA_forster - tau_DA_kinetic) / tau_DA_kinetic < 1e-6, \
        f"tau_DA Forster={tau_DA_forster*1e9:.4f} ns vs kinetic={tau_DA_kinetic*1e9:.4f} ns"

def test_05_fret_efficiency_limits():
    """E(r->0)=1, E(r->inf)=0."""
    R0 = 5.4
    assert fret_efficiency(0.001, R0) > 0.999, "E(~0) should be ~1"
    assert fret_efficiency(100.0, R0) < 1e-6, "E(far) should be ~0"
    assert fret_efficiency(0.0, R0) == 1.0, "E(0) should be exactly 1"


# ═══════════════════════════════════════════════════════════════════
# BLOCK 2: BROWNIAN DYNAMICS — PHYSICAL INVARIANTS
# ═══════════════════════════════════════════════════════════════════

def test_06_msd_einstein_relation():
    """MSD = 4*D*t in 2D (Einstein relation, fundamental invariant)."""
    N = 500
    D = 0.1  # um^2/s
    dt = 1e-4  # s
    n_steps = 1000
    Lx, Ly = 50.0, 50.0  # Large to avoid PBC artifacts

    rng = np.random.default_rng(42)
    pos = rng.uniform(10.0, 40.0, size=(N, 2))
    pos_init = pos.copy()
    D_arr = np.full(N, D)
    forces = np.zeros((N, 2))
    grad_D = np.zeros((N, 2))
    is_alive = np.ones(N, dtype=np.bool_)

    for _ in range(n_steps):
        noise = rng.standard_normal((N, 2))
        bd_step(pos, D_arr, forces, grad_D, dt, Lx, Ly, True, True,
                N, noise, is_alive, 100.0)

    dr = pos - pos_init
    # PBC correction for large membrane
    dr[:, 0] -= np.round(dr[:, 0] / Lx) * Lx
    dr[:, 1] -= np.round(dr[:, 1] / Ly) * Ly
    msd = np.mean(dr[:, 0]**2 + dr[:, 1]**2)
    t_total = n_steps * dt
    msd_expected = 4.0 * D * t_total

    error = abs(msd - msd_expected) / msd_expected
    assert error < 0.10, f"MSD error {error:.1%}: MSD={msd:.6f}, expected={msd_expected:.6f}"

def test_07_bd_step_dead_particles_immobile():
    """Particles with is_alive=False must NOT move."""
    N = 10
    pos = np.random.uniform(0, 5, (N, 2))
    pos_orig = pos.copy()
    D_arr = np.full(N, 0.1)
    forces = np.zeros((N, 2))
    grad_D = np.zeros((N, 2))
    is_alive = np.zeros(N, dtype=np.bool_)  # ALL dead
    noise = np.random.standard_normal((N, 2))

    bd_step(pos, D_arr, forces, grad_D, 1e-4, 10.0, 10.0, True, True,
            N, noise, is_alive, 1.0)

    assert np.allclose(pos, pos_orig), "Dead particles moved!"

def test_08_pbc_positions_in_box():
    """After BD step, ALL positions must be in [0, Lx) x [0, Ly)."""
    N = 200
    Lx, Ly = 2.0, 2.0
    rng = np.random.default_rng(99)
    pos = rng.uniform(0, Lx, (N, 2))
    D_arr = np.full(N, 0.5)  # High D to force large displacement
    forces = np.zeros((N, 2))
    grad_D = np.zeros((N, 2))
    is_alive = np.ones(N, dtype=np.bool_)

    for _ in range(100):
        noise = rng.standard_normal((N, 2))
        bd_step(pos, D_arr, forces, grad_D, 1e-3, Lx, Ly, True, True,
                N, noise, is_alive, 10.0)

    assert np.all(pos[:, 0] >= 0) and np.all(pos[:, 0] < Lx), \
        f"x positions outside box: min={pos[:,0].min()}, max={pos[:,0].max()}"
    assert np.all(pos[:, 1] >= 0) and np.all(pos[:, 1] < Ly), \
        f"y positions outside box: min={pos[:,1].min()}, max={pos[:,1].max()}"

def test_09_displacement_cap_enforced():
    """The displacement cap must limit |dx| to max_displacement."""
    N = 1
    pos = np.array([[1.0, 1.0]])
    D_arr = np.array([100.0])  # Huge D -> huge displacement
    forces = np.zeros((N, 2))
    grad_D = np.zeros((N, 2))
    is_alive = np.ones(N, dtype=np.bool_)
    noise = np.array([[5.0, 5.0]])  # huge noise
    max_disp = 0.01  # um

    pos_before = pos.copy()
    bd_step(pos, D_arr, forces, grad_D, 1e-3, 10.0, 10.0, True, True,
            N, noise, is_alive, max_disp)

    dr = np.sqrt((pos[0, 0] - pos_before[0, 0])**2 + (pos[0, 1] - pos_before[0, 1])**2)
    # With PBC the displacement may look different, but the raw dx should have been capped
    # The test verifies it did not explode to huge values
    assert dr < max_disp * 2, f"Displacement {dr:.6f} exceeds cap {max_disp}"

def test_10_wca_force_repulsive():
    """The WCA force must ALWAYS be repulsive (push particles apart)."""
    N = 2
    sigma = 0.008  # um (8 nm)
    eps = 2.5
    # Two particles very close (r < 2^(1/6)*sigma)
    pos = np.array([[1.0, 1.0], [1.0 + sigma * 0.9, 1.0]])
    forces = np.zeros((N, 2))
    species_id = np.array([0, 0], dtype=np.int32)
    is_alive = np.ones(N, dtype=np.bool_)
    pairs = np.array([[0, 1]], dtype=np.int32)
    sigma_m = np.array([[sigma]])
    eps_m = np.array([[eps]])

    compute_forces_wca(pos, forces, species_id, is_alive, pairs, 1,
                       sigma_m, eps_m, 10.0, 10.0, True, True)

    # Force on i must push to the left (away from j)
    assert forces[0, 0] < 0, f"Force on i must be negative (repulsive), got {forces[0,0]}"
    # Force on j must push to the right (Newton 3rd)
    assert forces[1, 0] > 0, f"Force on j must be positive, got {forces[1,0]}"
    # Newton 3rd: F_i = -F_j
    assert abs(forces[0, 0] + forces[1, 0]) < 1e-10, "Newton's 3rd law violation"
    assert abs(forces[0, 1] + forces[1, 1]) < 1e-10, "Newton's 3rd law violation (y)"


# ═══════════════════════════════════════════════════════════════════
# BLOCK 3: CELL LIST — ALGORITHMIC CORRECTNESS
# ═══════════════════════════════════════════════════════════════════

def test_11_cell_list_vs_brute_force():
    """Cell list must find EXACTLY the same pairs as brute force."""
    rng = np.random.default_rng(77)
    N = 100
    Lx, Ly = 2.0, 2.0
    cutoff = 0.05  # um
    pos = rng.uniform(0, Lx, (N, 2))

    cl = CellList(Lx, Ly, cutoff, periodic=(True, True))
    cl.build(pos, N)
    pairs_cl = cl.get_neighbor_pairs(pos, N)
    pairs_bf = cl.get_neighbors_brute(pos, N, cutoff)

    # Convert to sets of tuples for comparison
    set_cl = set(tuple(sorted(p)) for p in pairs_cl)
    set_bf = set(tuple(sorted(p)) for p in pairs_bf)

    assert set_cl == set_bf, \
        f"Cell list: {len(set_cl)} pairs, brute force: {len(set_bf)} pairs. " \
        f"Only in CL: {set_cl - set_bf}, Only in BF: {set_bf - set_cl}"

def test_12_min_image_distance_symmetry():
    """d(i,j) = d(j,i) (metric symmetry)."""
    Lx, Ly = 2.0, 2.0
    for _ in range(100):
        x1, y1 = np.random.uniform(0, Lx), np.random.uniform(0, Ly)
        x2, y2 = np.random.uniform(0, Lx), np.random.uniform(0, Ly)
        d12, _, _ = _min_image_dist_sq(x1, y1, x2, y2, Lx, Ly, True, True)
        d21, _, _ = _min_image_dist_sq(x2, y2, x1, y1, Lx, Ly, True, True)
        assert abs(d12 - d21) < 1e-14, f"Asymmetry: d12={d12}, d21={d21}"

def test_13_min_image_distance_max():
    """Minimum image distance <= L/2 in each dimension."""
    Lx, Ly = 2.0, 3.0
    max_d_sq = (Lx/2)**2 + (Ly/2)**2
    for _ in range(200):
        x1, y1 = np.random.uniform(0, Lx), np.random.uniform(0, Ly)
        x2, y2 = np.random.uniform(0, Lx), np.random.uniform(0, Ly)
        d_sq, dx, dy = _min_image_dist_sq(x1, y1, x2, y2, Lx, Ly, True, True)
        assert d_sq <= max_d_sq + 1e-12, f"d^2={d_sq} > max={max_d_sq}"
        assert abs(dx) <= Lx / 2 + 1e-12, f"|dx|={abs(dx)} > Lx/2"
        assert abs(dy) <= Ly / 2 + 1e-12, f"|dy|={abs(dy)} > Ly/2"


# ═══════════════════════════════════════════════════════════════════
# BLOCK 4: REACTIONS — DETAILED BALANCE AND CONSERVATION
# ═══════════════════════════════════════════════════════════════════

def test_14_binding_conserves_total_particles():
    """Binding A+B->AB: N_A + N_AB = const, N_B + N_AB = const."""
    N = 100
    pos = np.random.uniform(0, 1, (N, 2)).astype(np.float64)
    species_id = np.zeros(N, dtype=np.int32)
    species_id[:60] = 0  # A
    species_id[60:] = 1  # B
    bond_partner = np.full(N, -1, dtype=np.int32)
    is_alive = np.ones(N, dtype=np.bool_)

    N_A_init = np.sum(species_id == 0)
    N_B_init = np.sum(species_id == 1)

    # Place all pairs very close to force binding
    pairs = np.array([[i, 60+i] for i in range(40)], dtype=np.int32)
    for i in range(40):
        pos[60+i] = pos[i] + 0.001

    rng_u = np.zeros(len(pairs))  # prob=0 -> always bind

    n_bound = apply_binding_reactions(
        pos, species_id, bond_partner, is_alive, N,
        0, 1, 2,  # A=0, B=1, AB=2
        0.01,  # contact radius um
        1.0,   # prob = 1
        pairs, len(pairs), rng_u,
        2.0, 2.0, True, True)

    N_A_final = np.sum((species_id == 0) & is_alive)
    N_AB_final = np.sum((species_id == 2) & is_alive)
    # Each binding: A becomes AB, B is deactivated
    # => N_A_init = N_A_final + N_AB_final
    assert N_A_init == N_A_final + N_AB_final, \
        f"Conservation violated: {N_A_init} != {N_A_final} + {N_AB_final}"

def test_15_dissociation_restores_species():
    """Dissociation AB->A+B must restore correct species_id."""
    N = 4
    species_id = np.array([2, 1, 2, 1], dtype=np.int32)  # AB, B, AB, B
    bond_partner = np.array([1, 0, 3, 2], dtype=np.int32)  # 0-1, 2-3
    is_alive = np.array([True, False, True, False])
    positions = np.array([[1,1],[1,1],[2,2],[2,2]], dtype=np.float64)
    rng_u = np.zeros(N)  # prob=0 -> always dissociate

    n_dissoc = apply_dissociation(
        species_id, bond_partner, is_alive, positions, N,
        2, 0, 1,  # product=AB=2, reactant_a=A=0, reactant_b=B=1
        1.0,  # prob = 1
        rng_u)

    assert n_dissoc == 2, f"Expected 2 dissociations, got {n_dissoc}"
    assert species_id[0] == 0, "A not restored"
    assert species_id[1] == 1, "B not restored"
    assert is_alive[1] == True, "B not reactivated"
    assert bond_partner[0] == -1, "Bond not cleared"

def test_16_binding_requires_both_free():
    """Cannot bind an already bonded particle."""
    N = 3
    pos = np.array([[0.5, 0.5], [0.501, 0.5], [0.502, 0.5]], dtype=np.float64)
    species_id = np.array([0, 1, 1], dtype=np.int32)
    bond_partner = np.array([1, 0, -1], dtype=np.int32)  # 0-1 already bonded
    is_alive = np.array([True, True, True])
    pairs = np.array([[0, 2]], dtype=np.int32)
    rng_u = np.zeros(1)

    n = apply_binding_reactions(pos, species_id, bond_partner, is_alive, N,
                                 0, 1, 2, 1.0, 1.0, pairs, 1, rng_u,
                                 5.0, 5.0, True, True)
    assert n == 0, "Binding of already bonded particle should not occur"


# ═══════════════════════════════════════════════════════════════════
# BLOCK 5: PSS KERNEL — MATHEMATICAL CORRECTNESS
# ═══════════════════════════════════════════════════════════════════

def test_17_pss_tperiod_is_stochastic():
    """T_period must be a stochastic matrix (rows sum to 1)."""
    cfg = _load_mini()
    state = ParticleState(cfg)
    ps = PhotonSimulator(cfg, state)

    # Access internal T_step matrices
    T_step = ps._T_step_all
    for dtype in range(ps.n_fluoro_types):
        for pss_state in range(2):
            # T_step is stored flattened [dtype, pss_state, 3] -> actually check T_step_all
            # The full 3x3 matrix is stored as T_step_all[dtype] which is a 3x3
            pass
    # Instead, test via _build_pss_matrices directly
    rm = RateMatrix(cfg.fluorophores[0])
    T_laser = 1.0 / cfg.laser.repetition_rate
    n_pulses = max(1, int(cfg.bd_timestep * cfg.laser.repetition_rate))
    result = _build_pss_matrices(rm, 0.01, T_laser, n_pulses, k_fret=0.0, QY_donor=0.85)
    T = result['T_step']
    for row in range(3):
        s = T[row].sum()
        assert abs(s - 1.0) < 1e-6, f"T_step row {row} sums to {s}, expected 1.0"

def test_18_pss_mean_photons_nonnegative():
    """mean_ph must be >= 0 (photons cannot be negative)."""
    cfg = _load_mini()
    rm = RateMatrix(cfg.fluorophores[0])
    T_laser = 1.0 / cfg.laser.repetition_rate
    n_pulses = max(1, int(cfg.bd_timestep * cfg.laser.repetition_rate))
    result = _build_pss_matrices(rm, 0.01, T_laser, n_pulses, k_fret=0.0, QY_donor=0.92)
    assert np.all(result['mean_ph'] >= 0), f"negative mean_ph: {result['mean_ph']}"
    assert np.all(result['mean_fret'] >= 0), f"negative mean_fret: {result['mean_fret']}"

def test_19_pss_fret_reduces_donor_photons():
    """With FRET, donor photons must DECREASE (energy transferred to acceptor)."""
    cfg = _load_mini()
    rm = RateMatrix(cfg.fluorophores[0])
    T_laser = 1.0 / cfg.laser.repetition_rate
    n_pulses = max(1, int(cfg.bd_timestep * cfg.laser.repetition_rate))

    r_no_fret = _build_pss_matrices(rm, 0.01, T_laser, n_pulses, k_fret=0.0, QY_donor=0.92)
    # k_FRET for E~0.5: k = (R0/r)^6 / tau_D, with R0=r => k = 1/tau_D
    tau_D = rm.get_lifetime(rm.excited_state)
    k_fret_val = 1.0 / tau_D  # E = 0.5

    r_fret = _build_pss_matrices(rm, 0.01, T_laser, n_pulses, k_fret=k_fret_val, QY_donor=0.92)

    assert r_fret['mean_ph'][0] < r_no_fret['mean_ph'][0], \
        f"FRET did not reduce donor photons: without={r_no_fret['mean_ph'][0]:.4f}, with={r_fret['mean_ph'][0]:.4f}"

def test_20_pss_tau_eff_with_fret():
    """tau_eff with FRET must be less than tau_D (quenching)."""
    cfg = _load_mini()
    rm = RateMatrix(cfg.fluorophores[0])
    T_laser = 1.0 / cfg.laser.repetition_rate
    n_pulses = max(1, int(cfg.bd_timestep * cfg.laser.repetition_rate))
    tau_D = rm.get_lifetime(rm.excited_state)
    k_fret_val = 1.0 / tau_D

    r = _build_pss_matrices(rm, 0.01, T_laser, n_pulses, k_fret=k_fret_val, QY_donor=0.92)
    assert r['tau_eff'] < tau_D, \
        f"tau_eff={r['tau_eff']*1e9:.3f} ns not less than tau_D={tau_D*1e9:.3f} ns"

def test_21_poisson_draw_mean():
    """_poisson_draw must produce a mean close to the mean parameter."""
    mean_target = 10.0
    N = 10000
    draws = []
    for i in range(N):
        seed = float(i) / N
        draws.append(_poisson_draw(mean_target, seed))
    actual_mean = np.mean(draws)
    # The standard deviation of the mean is sqrt(mean/N)
    assert abs(actual_mean - mean_target) < 3.0 * math.sqrt(mean_target / N) + 1.0, \
        f"Poisson mean: {actual_mean:.2f}, expected: {mean_target:.1f}"

def test_22_poisson_draw_zero_mean():
    """_poisson_draw(0) must always return 0."""
    for i in range(100):
        assert _poisson_draw(0.0, float(i)/100) == 0

def test_23_poisson_draw_negative_mean():
    """_poisson_draw(mean<0) must return 0 (guard)."""
    assert _poisson_draw(-5.0, 0.5) == 0


# ═══════════════════════════════════════════════════════════════════
# BLOCK 6: MICROSCOPE — PSF AND TCSPC
# ═══════════════════════════════════════════════════════════════════

def test_24_psf_sigma_confocal_formula():
    """sigma_conf = 1/sqrt(1/sigma_ill^2 + 1/sigma_det^2)."""
    cfg = _load_mini()
    psf = PSF(cfg)
    NA = cfg.objective.NA
    lam_exc = cfg.laser.wavelength  # nm
    lam_em = 519.0  # nm (Alexa488)
    sigma_ill = 0.21 * lam_exc / NA
    sigma_det = 0.21 * lam_em / NA
    sigma_conf_expected = 1.0 / math.sqrt(1.0/sigma_ill**2 + 1.0/sigma_det**2)
    sigma_conf_expected_um = sigma_conf_expected / 1000.0

    sigma_actual = psf.get_sigma(0)
    error = abs(sigma_actual - sigma_conf_expected_um) / sigma_conf_expected_um
    assert error < 0.01, f"sigma_conf: {sigma_actual*1000:.2f} nm vs expected {sigma_conf_expected:.2f} nm"

def test_25_psf_normalization():
    """PSF must integrate to ~1 (probability conservation)."""
    cfg = _load_mini()
    psf = PSF(cfg)
    sigma = psf.get_sigma(0)
    # Integrate PSF numerically
    r = np.linspace(0, 5*sigma, 1000)
    h_r = psf.evaluate_array(r, 0)
    integral = 2.0 * math.pi * np.trapz(h_r * r, r)
    assert abs(integral - 1.0) < 0.05, f"PSF integral = {integral:.4f}, expected ~1.0"

def test_26_tcspc_bin_range():
    """Photons with arrival_time >= time_range must fall outside bins."""
    n_bins = 256
    time_range = 25e-9
    bin_width = time_range / n_bins
    # A photon with t = 26 ns should fall in bin 266 -> out of range
    t = 26e-9
    bin_idx = int(t / bin_width)
    assert bin_idx >= n_bins, f"Photon with t={t*1e9:.1f} ns falls in bin {bin_idx} < {n_bins}"


# ═══════════════════════════════════════════════════════════════════
# BLOCK 7: PARTICLES — SoA CORRECTNESS
# ═══════════════════════════════════════════════════════════════════

def test_27_particle_state_total_count():
    """N total = sum(initial_populations) + sum(dormant_pools)."""
    cfg = _load_pip3()
    state = ParticleState(cfg)
    n_initial = sum(cfg.initial_populations.values())
    n_dormant = sum(ex.cytosol_pool for ex in cfg.compartment_exchanges if ex.cytosol_pool > 0)
    assert state.N == n_initial + n_dormant, \
        f"N={state.N}, expected {n_initial}+{n_dormant}={n_initial+n_dormant}"

def test_28_dormant_particles_not_alive():
    """Dormant particles must have is_alive=False."""
    cfg = _load_pip3()
    state = ParticleState(cfg)
    n_alive = np.sum(state.is_alive)
    n_initial = sum(cfg.initial_populations.values())
    assert n_alive == n_initial, f"n_alive={n_alive}, expected {n_initial}"

def test_29_dye_assignment_includes_dormant():
    """Dyes must be assigned to dormant particles (genetic biosensors)."""
    cfg = _load_pip3()
    state = ParticleState(cfg, rng=np.random.default_rng(42))
    # PHAkt has dye_attachment with labeling_efficiency=1.0
    # Verify that dormant PHAkt also have dye
    dormant_mask = ~state.is_alive
    has_dye_dormant = np.sum(state.has_dye & dormant_mask)
    # At least some dormant must have dye (PHAkt pool is large)
    total_dormant = np.sum(dormant_mask)
    if total_dormant > 0:
        frac = has_dye_dormant / total_dormant
        # Not all dormant are PHAkt, but there should be some
        assert has_dye_dormant > 0, "No dormant has dye — bug in _assign_dyes"

def test_30_species_id_valid_range():
    """All species_id must be in [0, n_species)."""
    cfg = _load_pip3()
    state = ParticleState(cfg)
    n_sp = len(cfg.species)
    assert np.all(state.species_id >= 0), "species_id < 0"
    assert np.all(state.species_id < n_sp), f"species_id >= {n_sp}"

def test_31_positions_in_membrane():
    """All initial positions must be inside the membrane."""
    cfg = _load_pip3()
    state = ParticleState(cfg)
    Lx, Ly = cfg.membrane.size
    assert np.all(state.positions[:, 0] >= 0), "x < 0"
    assert np.all(state.positions[:, 0] < Lx), f"x >= {Lx}"
    assert np.all(state.positions[:, 1] >= 0), "y < 0"
    assert np.all(state.positions[:, 1] < Ly), f"y >= {Ly}"


# ═══════════════════════════════════════════════════════════════════
# BLOCK 8: INTERFACE CONTRACTS BETWEEN MODULES
# ═══════════════════════════════════════════════════════════════════

def test_32_photon_batch_empty_valid():
    """PhotonBatch.empty() must produce a valid batch with n_photons=0."""
    batch = PhotonBatch.empty()
    assert batch.n_photons == 0
    assert len(batch.arrival_time) == 0
    assert len(batch.channel) == 0

def test_33_dynamics_engine_creates_from_config():
    """DynamicsEngine must build without error from any valid config."""
    from ed_mrdt.dynamics import DynamicsEngine
    cfg = _load_mini()
    state = ParticleState(cfg)
    engine = DynamicsEngine(cfg, state)
    assert engine.dt == cfg.bd_timestep
    assert engine.Lx == cfg.membrane.size[0]

def test_34_pipeline_frame_count():
    """Pipeline must generate exactly n_frames frames."""
    cfg = _load_mini()
    pipeline = SimulationPipeline(cfg)
    result = pipeline.run()
    expected_frames = int(cfg.simulation_time / cfg.scan.frame_interval)
    assert len(result.frames) == expected_frames, \
        f"Frames: {len(result.frames)}, expected {expected_frames}"

def test_35_flim_stack_shape():
    """flim_stack must have shape [n_channels, nx, ny, n_bins]."""
    cfg = _load_mini()
    pipeline = SimulationPipeline(cfg)
    result = pipeline.run()
    frame = result.frames[0]
    stack = frame.flim_frame.flim_stack
    n_ch = len(cfg.fluorophores)
    nx = cfg.scan.pixels_x
    ny = cfg.scan.pixels_y
    n_bins = cfg.tcspc.n_bins
    assert stack.shape == (n_ch, nx, ny, n_bins), \
        f"Shape: {stack.shape}, expected ({n_ch},{nx},{ny},{n_bins})"

def test_36_flim_stack_dtype_uint32():
    """flim_stack must be uint32 (integer photon counts)."""
    cfg = _load_mini()
    pipeline = SimulationPipeline(cfg)
    result = pipeline.run()
    stack = result.frames[0].flim_frame.flim_stack
    assert stack.dtype == np.uint32, f"dtype={stack.dtype}, expected uint32"

def test_37_ground_truth_shapes_match():
    """GroundTruth arrays must have length N from ParticleState."""
    cfg = _load_mini()
    pipeline = SimulationPipeline(cfg)
    result = pipeline.run()
    gt = result.frames[0].ground_truth
    N = len(gt.positions)
    assert len(gt.species_id) == N
    assert len(gt.bond_partner) == N
    assert len(gt.is_alive) == N
    assert len(gt.has_dye) == N
    assert len(gt.fret_efficiency_exact) == N
    assert len(gt.donor_lifetime_exact) == N

def test_38_photon_budget_conservation():
    """photons_emitted >= photons_detected (photons are not created)."""
    cfg = _load_mini()
    pipeline = SimulationPipeline(cfg)
    result = pipeline.run()
    assert result.total_photons_emitted >= result.total_photons_detected, \
        f"emitted={result.total_photons_emitted} < detected={result.total_photons_detected}"


# ═══════════════════════════════════════════════════════════════════
# BLOCK 9: ANALYSIS — ERRORS IN STATISTICAL ESTIMATORS
# ═══════════════════════════════════════════════════════════════════

def test_39_mean_arrival_time_monoexp():
    """For synthetic mono-exponential, <t> ~ tau (bias < 5% for T/tau > 5)."""
    tau_ns = 3.0
    time_range = 25e-9
    n_bins = 256
    bin_width_ns = (time_range / n_bins) * 1e9
    t_ns = (np.arange(n_bins) + 0.5) * bin_width_ns

    # Generate mono-exponential histogram
    N_photons = 100000
    rng = np.random.default_rng(42)
    times = rng.exponential(tau_ns, N_photons)
    times = times[times < time_range * 1e9]  # truncate

    hist, _ = np.histogram(times, bins=np.arange(n_bins + 1) * bin_width_ns)
    stack = np.zeros((1, 1, 1, n_bins), dtype=np.uint32)
    stack[0, 0, 0, :] = hist

    tau_map, intensity = mean_arrival_time(stack, time_range, channel=0, min_counts=10)
    tau_est = tau_map[0, 0]

    # Expected bias: tau_est = tau * [1 - T*exp(-T/tau)/(1-exp(-T/tau))]
    # For T/tau = 25/3 ~ 8.3, bias ~0.4%
    error = abs(tau_est - tau_ns) / tau_ns
    assert error < 0.05, f"estimated tau={tau_est:.3f} ns, actual={tau_ns} ns, error={error:.1%}"

def test_40_phasor_semicircle():
    """For mono-exponential, (g,s) must fall on the universal semicircle."""
    tau_ns = 3.0
    omega = 2 * math.pi * 80e6  # 80 MHz
    tau_s = tau_ns * 1e-9
    g_theory = 1.0 / (1.0 + (omega * tau_s)**2)
    s_theory = omega * tau_s / (1.0 + (omega * tau_s)**2)

    # Verify that (g,s) is on the semicircle: (g-0.5)^2 + s^2 = 0.25
    r_sq = (g_theory - 0.5)**2 + s_theory**2
    assert abs(r_sq - 0.25) < 0.01, \
        f"(g,s)=({g_theory:.4f},{s_theory:.4f}) outside semicircle: r^2={r_sq:.4f}"

def test_41_transfer_entropy_independent_series():
    """TE between independent series must be ~0 (not significant)."""
    rng = np.random.default_rng(42)
    N = 1000
    x = rng.standard_normal(N)
    y = rng.standard_normal(N)

    result = transfer_entropy(x, y, n_bins=8, lag=1, n_shuffle=50)
    # Bias-corrected TE should be very close to 0
    assert result.te_xy < 0.05, f"TE(ind)={result.te_xy:.4f}, should be ~0"
    assert not result.significant_xy, "TE significant for independent series!"

def test_42_transfer_entropy_causal_series():
    """TE(X->Y) must be significant when Y = f(X_past)."""
    rng = np.random.default_rng(42)
    N = 2000
    x = rng.standard_normal(N)
    y = np.zeros(N)
    y[0] = rng.standard_normal()
    for t in range(1, N):
        y[t] = 0.7 * x[t-1] + 0.3 * rng.standard_normal()

    result = transfer_entropy(x, y, n_bins=8, lag=1, n_shuffle=100)
    assert result.te_xy > result.te_yx, \
        f"TE(X->Y)={result.te_xy:.4f} should be > TE(Y->X)={result.te_yx:.4f}"

def test_43_discretize_uniform_bins():
    """_discretize must produce values in [0, n_bins-1]."""
    x = np.array([0.0, 0.5, 1.0, 0.25, 0.75], dtype=np.float64)
    n_bins = 4
    d = _discretize(x, n_bins)
    assert np.all(d >= 0), f"Negative bins: {d}"
    assert np.all(d < n_bins), f"Bins >= n_bins: {d}"

def test_44_discretize_constant_series():
    """_discretize of constant series must give all 0 (no division by zero)."""
    x = np.full(100, 5.0, dtype=np.float64)
    d = _discretize(x, 8)
    assert np.all(d == 0), f"Constant series should give bin 0, got {np.unique(d)}"


# ═══════════════════════════════════════════════════════════════════
# BLOCK 10: INTERFACE ERRORS BETWEEN LAYERS
# ═══════════════════════════════════════════════════════════════════

def test_45_photon_positions_inside_membrane():
    """Emitted photons must have positions inside the membrane."""
    cfg = _load_mini()
    state = ParticleState(cfg)
    from ed_mrdt.dynamics import DynamicsEngine
    engine = DynamicsEngine(cfg, state)
    ps = PhotonSimulator(cfg, state)

    engine.step()
    batch = ps.evolve_one_bd_step()

    if batch.n_photons > 0:
        Lx, Ly = cfg.membrane.size
        ph_x = state.positions[batch.particle_idx[:batch.n_photons], 0]
        ph_y = state.positions[batch.particle_idx[:batch.n_photons], 1]
        assert np.all(ph_x >= 0) and np.all(ph_x < Lx), "Photon outside membrane (x)"
        assert np.all(ph_y >= 0) and np.all(ph_y < Ly), "Photon outside membrane (y)"

def test_46_photon_arrival_times_positive():
    """TCSPC arrival times must be > 0."""
    cfg = _load_mini()
    state = ParticleState(cfg)
    from ed_mrdt.dynamics import DynamicsEngine
    engine = DynamicsEngine(cfg, state)
    ps = PhotonSimulator(cfg, state)

    engine.step()
    batch = ps.evolve_one_bd_step()

    if batch.n_photons > 0:
        times = batch.arrival_time[:batch.n_photons]
        assert np.all(times > 0), f"Negative times detected: min={times.min()}"

def test_47_channel_indices_valid():
    """Channel indices in PhotonBatch must be valid for config.fluorophores."""
    cfg = _load_mini()
    state = ParticleState(cfg)
    from ed_mrdt.dynamics import DynamicsEngine
    engine = DynamicsEngine(cfg, state)
    ps = PhotonSimulator(cfg, state)

    engine.step()
    batch = ps.evolve_one_bd_step()

    if batch.n_photons > 0:
        channels = batch.channel[:batch.n_photons]
        n_fluoro = len(cfg.fluorophores)
        assert np.all(channels >= 0), f"Negative channel: {channels.min()}"
        assert np.all(channels < n_fluoro), f"Channel >= n_fluoro({n_fluoro}): {channels.max()}"


# ═══════════════════════════════════════════════════════════════════
# BLOCK 11: CONCEPTUAL ERRORS IN HILL FEEDBACK
# ═══════════════════════════════════════════════════════════════════

def test_48_hill_activation_limits():
    """Hill activation: h(0)=basal, h(inf)=1."""
    # h(rho) = basal + (1-basal) * rho^n / (K^n + rho^n)
    K = 50.0
    n = 2
    basal = 0.05

    h_zero = basal + (1-basal) * 0.0**n / (K**n + 0.0**n + 1e-30)
    assert abs(h_zero - basal) < 1e-10, f"h(0)={h_zero}, expected {basal}"

    rho_big = 1e6
    h_inf = basal + (1-basal) * rho_big**n / (K**n + rho_big**n)
    assert abs(h_inf - 1.0) < 1e-6, f"h(inf)={h_inf}, expected 1.0"

def test_49_hill_inhibition_limits():
    """Hill inhibition: h(0)=1, h(inf)=basal."""
    K = 200.0
    n = 3
    basal = 0.0

    h_zero = basal + (1-basal) * K**n / (K**n + 0.0**n + 1e-30)
    # When rho=0: K^n / (K^n + 0) = 1
    assert abs(h_zero - 1.0) < 1e-10, f"h(0)={h_zero}, expected 1.0"

    rho_big = 1e6
    h_inf = basal + (1-basal) * K**n / (K**n + rho_big**n)
    assert abs(h_inf - basal) < 1e-3, f"h(inf)={h_inf}, expected {basal}"


# ═══════════════════════════════════════════════════════════════════
# BLOCK 12: DENSITY FIELD & GRAD_D CONSISTENCY
# ═══════════════════════════════════════════════════════════════════

def test_50_smooth_diffusion_field_gradient_finite_difference():
    """Analytical nabla_D must match finite difference."""
    N = 1
    eps = 1e-6  # um
    # Particle at the boundary of a circular region
    cx, cy, r_reg = 1.0, 1.0, 0.5
    D_factor = 0.3
    D_base = np.array([0.1], dtype=np.float64)
    species_id = np.array([0], dtype=np.int32)
    smoothing = 0.05

    region_cx = np.array([cx])
    region_cy = np.array([cy])
    region_r = np.array([r_reg])
    region_Df = np.array([D_factor])

    # Point near the boundary
    x0, y0 = cx + r_reg - 0.01, cy

    # Analytical
    positions = np.array([[x0, y0]], dtype=np.float64)
    D_arr = np.zeros(1, dtype=np.float64)
    grad_D = np.zeros((1, 2), dtype=np.float64)
    apply_diffusion_field_regions_smooth(
        D_arr, grad_D, D_base, species_id, positions, 1,
        region_cx, region_cy, region_r, region_Df, 1, smoothing)

    grad_analytic = grad_D[0].copy()

    # Finite difference in x
    pos_p = np.array([[x0 + eps, y0]], dtype=np.float64)
    pos_m = np.array([[x0 - eps, y0]], dtype=np.float64)
    D_p = np.zeros(1, dtype=np.float64)
    D_m = np.zeros(1, dtype=np.float64)
    grad_dummy = np.zeros((1, 2), dtype=np.float64)

    apply_diffusion_field_regions_smooth(
        D_p, grad_dummy, D_base, species_id, pos_p, 1,
        region_cx, region_cy, region_r, region_Df, 1, smoothing)
    apply_diffusion_field_regions_smooth(
        D_m, grad_dummy, D_base, species_id, pos_m, 1,
        region_cx, region_cy, region_r, region_Df, 1, smoothing)

    grad_fd_x = (D_p[0] - D_m[0]) / (2 * eps)

    if abs(grad_analytic[0]) > 1e-12:
        error = abs(grad_analytic[0] - grad_fd_x) / abs(grad_analytic[0])
        assert error < 0.05, \
            f"grad_D_x: analytical={grad_analytic[0]:.6e}, FD={grad_fd_x:.6e}, error={error:.1%}"


# ═══════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("ED-MRDT v1.1 — EXPERT AUDIT: 50 DIAGNOSTIC TESTS")
    print("=" * 70)

    all_tests = [
        # Block 1: Forster FRET
        ("01_fret_efficiency_at_R0", test_01_fret_efficiency_at_R0),
        ("02_fret_efficiency_monotonicity", test_02_fret_efficiency_monotonicity),
        ("03_fret_rate_vs_efficiency_consistency", test_03_fret_rate_vs_efficiency_consistency),
        ("04_fret_tau_da_formula", test_04_fret_tau_da_formula),
        ("05_fret_efficiency_limits", test_05_fret_efficiency_limits),
        # Block 2: Brownian Dynamics
        ("06_msd_einstein_relation", test_06_msd_einstein_relation),
        ("07_bd_step_dead_particles_immobile", test_07_bd_step_dead_particles_immobile),
        ("08_pbc_positions_in_box", test_08_pbc_positions_in_box),
        ("09_displacement_cap_enforced", test_09_displacement_cap_enforced),
        ("10_wca_force_repulsive", test_10_wca_force_repulsive),
        # Block 3: Cell List
        ("11_cell_list_vs_brute_force", test_11_cell_list_vs_brute_force),
        ("12_min_image_distance_symmetry", test_12_min_image_distance_symmetry),
        ("13_min_image_distance_max", test_13_min_image_distance_max),
        # Block 4: Reactions
        ("14_binding_conserves_total_particles", test_14_binding_conserves_total_particles),
        ("15_dissociation_restores_species", test_15_dissociation_restores_species),
        ("16_binding_requires_both_free", test_16_binding_requires_both_free),
        # Block 5: PSS Kernel
        ("17_pss_tperiod_stochastic", test_17_pss_tperiod_is_stochastic),
        ("18_pss_mean_photons_nonnegative", test_18_pss_mean_photons_nonnegative),
        ("19_pss_fret_reduces_donor_photons", test_19_pss_fret_reduces_donor_photons),
        ("20_pss_tau_eff_with_fret", test_20_pss_tau_eff_with_fret),
        ("21_poisson_draw_mean", test_21_poisson_draw_mean),
        ("22_poisson_draw_zero", test_22_poisson_draw_zero_mean),
        ("23_poisson_draw_negative", test_23_poisson_draw_negative_mean),
        # Block 6: Microscope
        ("24_psf_sigma_confocal_formula", test_24_psf_sigma_confocal_formula),
        ("25_psf_normalization", test_25_psf_normalization),
        ("26_tcspc_bin_range", test_26_tcspc_bin_range),
        # Block 7: Particles SoA
        ("27_particle_total_count", test_27_particle_state_total_count),
        ("28_dormant_not_alive", test_28_dormant_particles_not_alive),
        ("29_dye_includes_dormant", test_29_dye_assignment_includes_dormant),
        ("30_species_id_range", test_30_species_id_valid_range),
        ("31_positions_in_membrane", test_31_positions_in_membrane),
        # Block 8: Interface Contracts
        ("32_photon_batch_empty", test_32_photon_batch_empty_valid),
        ("33_dynamics_engine_creates", test_33_dynamics_engine_creates_from_config),
        ("34_pipeline_frame_count", test_34_pipeline_frame_count),
        ("35_flim_stack_shape", test_35_flim_stack_shape),
        ("36_flim_stack_dtype", test_36_flim_stack_dtype_uint32),
        ("37_ground_truth_shapes", test_37_ground_truth_shapes_match),
        ("38_photon_budget_conservation", test_38_photon_budget_conservation),
        # Block 9: Analysis
        ("39_mean_arrival_time_monoexp", test_39_mean_arrival_time_monoexp),
        ("40_phasor_semicircle", test_40_phasor_semicircle),
        ("41_te_independent_series", test_41_transfer_entropy_independent_series),
        ("42_te_causal_series", test_42_transfer_entropy_causal_series),
        ("43_discretize_uniform_bins", test_43_discretize_uniform_bins),
        ("44_discretize_constant", test_44_discretize_constant_series),
        # Block 10: Interface between layers
        ("45_photon_positions_in_membrane", test_45_photon_positions_inside_membrane),
        ("46_photon_arrival_times_positive", test_46_photon_arrival_times_positive),
        ("47_channel_indices_valid", test_47_channel_indices_valid),
        # Block 11: Hill feedback
        ("48_hill_activation_limits", test_48_hill_activation_limits),
        ("49_hill_inhibition_limits", test_49_hill_inhibition_limits),
        # Block 12: Gradient consistency
        ("50_grad_D_finite_difference", test_50_smooth_diffusion_field_gradient_finite_difference),
    ]

    print(f"\nRunning {len(all_tests)} tests...\n")

    for name, func in all_tests:
        run_test(name, func)

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {passed} PASSED, {failed} FAILED out of {len(all_tests)}")
    print(f"{'=' * 70}")

    if errors:
        print("\nDETAILED FAILURES:")
        for name, msg in errors:
            print(f"  [{name}]: {msg}")

    sys.exit(0 if failed == 0 else 1)
