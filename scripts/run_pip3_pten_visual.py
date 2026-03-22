#!/usr/bin/env python3
"""
ED-MRDT v1.1 — PIP3/PTEN pipeline: run + generate result PNGs
===============================================================
Produces 11 publication-quality figures in the output directory.
"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm

from scipy.ndimage import convolve

from ed_mrdt.config import load_config
from ed_mrdt.pipeline import SimulationPipeline
from ed_mrdt.analysis import (
    mean_arrival_time, phasor_transform,
    compute_causal_flim_generic, CausalPairSpec,
)

# ── FLIM spatial binning (standard post-processing) ────────────
# Becker (2005) Advanced TCSPC Techniques §5.2:
# "Spatial binning ... sums the TCSPC histograms of neighboring pixels
#  to increase the photon count per analysis unit."
# SPCImage default = 3×3. This improves SNR by √9 = 3× while
# reducing spatial resolution from 312nm to 937nm — still sufficient
# to resolve PIP3 domains (~4 µm = 4.3 superpixels).
FLIM_BIN_SIZE = 3  # 3×3 pixel neighborhood

def spatial_bin_flim(flim_stack, bin_size=FLIM_BIN_SIZE):
    """Sum TCSPC histograms over bin_size × bin_size spatial neighborhoods.

    Parameters
    ----------
    flim_stack : ndarray [n_ch, nx, ny, n_bins]
    bin_size : int (default 3 → 3×3 = 9 pixels)

    Returns
    -------
    binned : ndarray, same shape (each pixel contains summed histogram
             from its bin_size×bin_size neighborhood)
    """
    kernel = np.ones((bin_size, bin_size), dtype=np.float64)
    binned = np.empty_like(flim_stack, dtype=np.float64)
    for ch in range(flim_stack.shape[0]):
        for t in range(flim_stack.shape[3]):
            binned[ch, :, :, t] = convolve(
                flim_stack[ch, :, :, t].astype(np.float64),
                kernel, mode='reflect'
            )
    return binned

# ── Output directory ─────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wave_v4_bond_fret")
os.makedirs(OUT, exist_ok=True)

# ═════════════════════════════════════════════════════════════════
# 1. LOAD + CONFIGURE
# ═════════════════════════════════════════════════════════════════
yaml_path = os.path.join(os.path.dirname(__file__), "ed_mrdt", "examples", "pip3_pten_flim.yaml")
config = load_config(yaml_path)

# ── MEMBRANE & SCANNING: 10×10 µm (CORRECTED from paso5 analysis) ──
# 10×10 µm² = 100 µm²: λ_crit ≈ 6.3µm → domain 10µm fits ~1.6 wavelengths.
# Previous 5µm < λ_crit → NO wave could fit (only polarity).
# 10µm is marginal but sufficient for one complete wave cycle.
config.bd_timestep = 2e-3          # 2ms BD timestep
n_frames = 10                      # 10 frames (est. ~2.3 min wall)
n_bd_per_frame = 1000              # 1000 × 2ms = 2s/frame → 20s total biology
config.scan.n_frames = n_frames
config.scan.frame_interval = n_bd_per_frame * config.bd_timestep  # 2s per frame
config.scan.dwell_time = 5.0e-3    # 5ms dwell (CORRECTED: was 1.5ms, 3.3× more photons/pixel)
config.membrane.size = [10.0, 10.0]  # 10×10 µm² (CORRECTED: was 5×5)
config.scan.pixels_x = 32           # 312 nm/pixel
config.scan.pixels_y = 32
# CRITICAL FIX (Directive 3, root cause): pixel_size must match domain/pixels.
# YAML default = 100nm → FOV = 32×100nm = 3.2µm → only 10% of 10µm membrane!
# Fix: 10µm / 32 = 312.5nm → FOV = 10µm = full membrane.
config.scan.pixel_size = 312.5       # nm (= 10µm / 32 pixels)

# Raft centered at membrane center, scaled radius for 10µm domain
for reg in config.diffusion_field.regions:
    if reg.get("shape") == "circle":
        reg["center"] = [5.0, 5.0]  # center of 10×10 µm membrane
        reg["radius"] = 2.0         # scaled 2× from 5µm domain

# ── POPULATIONS: scaled for 10×10µm = 100µm² (CORRECTED) ──
# Total particles scale with area: 100/25 = 4× more than original.
# Start at LOW state: PIP3 ≈ 0.4/µm², PTEN ≈ 20/µm² (from PDE analysis).
# Wave nucleation requires PTEN-dominated membrane with stochastic holes.
area_um2 = 10.0 * 10.0  # 100 µm²
config.initial_populations["PIP2"] = int(100.0 * area_um2)   # 100/µm² × 100 = 10000
config.initial_populations["PIP3"] = int(0.4 * area_um2)     # 0.4/µm² (LOW state) = 40
config.initial_populations["PI3K"] = int(5.0 * area_um2)     # 5/µm² × 100 = 500
config.initial_populations["PTEN_T1"] = int(20.0 * area_um2) # 20/µm² (dominant) = 2000
config.initial_populations["PTEN_T2"] = int(5.0 * area_um2)  # 5/µm² (stable) = 500
# ── PH-Akt BIOSENSOR: recruited from cytosol by PIP3 ──
# Start at LOW membrane (PIP3 is low → biosensor mostly in cytosol).
# As PIP3 nucleates → PHAkt translocates to membrane → INTENSITY increases locally.
# PLUS: bound PHAkt_PIP3 has intramolecular FRET → τ drops. Double contrast.
# Reference: Varnai & Balla 1998, Iijima & Devreotes 2002.
config.initial_populations["PHAkt"] = 50                        # Low expression: passive biosensor (seq < 10%)
config.initial_populations["PHAkt_PIP3"] = 0                    # start unbound

# ── DIFFUSION COEFFICIENTS: CORRECTED from PDE analysis (paso5) ──
# PDE waves require D_PIP3/D_PTEN = 3.3 (fire-diffuse-fire mechanism).
# D_PIP3 = 0.10 µm²/s (Arai 2010; slightly below FRAP bulk value)
# D_PTEN = 0.03 µm²/s (Matsuoka 2018; slow membrane-bound protein)
# Ratio 3.3× ensures front nucleation in bistable system.
# PI3K: D=0.10 µm²/s (kinase, intermediate — unchanged)
for sp in config.species:
    if sp.name in ("PIP2", "PIP3"):
        sp.diffusion_coefficient = 0.10    # µm²/s (CORRECTED: was 0.15)
    elif sp.name == "PI3K":
        sp.diffusion_coefficient = 0.10    # µm²/s (unchanged)
    elif sp.name in ("PTEN_T1", "PTEN_T2"):
        sp.diffusion_coefficient = 0.03    # µm²/s (CORRECTED: was 0.08)
    elif sp.name == "PHAkt":
        sp.diffusion_coefficient = 0.08    # µm²/s (peripheral membrane protein)
    elif sp.name == "PHAkt_PIP3":
        sp.diffusion_coefficient = 0.06    # µm²/s (biosensor-lipid complex)

# ── BLEACH RATE REDUCTION ─────────────────────────────────────────
# 1024 pixels at 5ms dwell, 10 frames → moderate total exposure.
# Reduce bleach by 500× to keep <3% total bleach (Becker, 2005).
for fluor in config.fluorophores:
    for state, transitions in fluor.thermal_rates.items():
        if 'bleached' in transitions:
            transitions['bleached'] /= 500.0

# ── FRET: BOND mode — PHAkt binds PIP3 transiently ──────────────
# intra_complex_distance = 5.0 nm, R₀ = 5.4 nm → E = 1/(1+(5.0/5.4)⁶) = 0.613
#
# PHAkt binding kinetics — optimized for FRET visibility + low sequestration.
# Previous: koff=20 s⁻¹ gave f_bound=9.8% but seq=29.5% (too many PHAkt).
# Current:  koff=3.0 with N_PHAkt=300 gives f_bound≈42%, seq≈7.7%.
# The higher f_bound makes the bi-exponential FRET decay visible in phasor
# space (SNR_elong=8.0), while seq<10% ensures passive biosensor behavior.
# Reference: Varnai & Balla 1998 — PH domains Kd ≈ 1-10 µM;
# Milo & Phillips 2015 Cell Biology by the Numbers §3.5 — biosensor <10% seq.
for rxn in config.reactions:
    if "PHAkt" in rxn.name:
        rxn.koff = 3.0     # s⁻¹ (τ=333ms, f_bound≈42% with λ_bind≈2.2)
        rxn.kon = 0.05      # µm²/s (unchanged — diffusion-limited encounter)

# ── CATALYTIC RATES: CORRECTED from PDE→BD mapping (paso5) ───────
# CRITICAL FIX: PI3K k_cat increased ×18 (0.0083 → 0.15) and catalytic
# feedback RESTORED. PDE analysis proved that without autoactivation
# (J[0,0] > 0), bistability requires much higher basal production rate.
# The "absorbing trap" concern is addressed by basal_fraction α=0.05:
#   At PIP3=0: hill = 0.05 → k_eff = 0.15 × 0.05 = 0.0075 (NOT zero)
#   At PIP3>>K: hill → 1.0 → k_eff = 0.15 (full autoactivation)
#
# PTEN_T1 k_cat increased ×10 (0.05 → 0.50): PTEN high-activity form.
# Consistent with Matsuoka 2018, Iijima 2004 range [0.05, 5.0] s⁻¹.
#
# contact_radius = 150nm: electrostatic capture radius (Saha 2014).
for cat in config.catalytic_reactions:
    if "PI3K" in cat.name:
        cat.k_cat = 0.15              # s⁻¹ (CORRECTED: was 0.0083, ×18 increase)
        cat.feedback_rule = "pip3_activation"  # RESTORED: positive feedback on catalysis
        # This is THE autoactivation driver: PIP3 ↑ → PI3K works faster → PIP3 ↑↑
        # Combined with recruitment feedback → strong bistability + wave nucleation
    if "PTEN_T1" in cat.name:
        cat.k_cat = 0.50              # s⁻¹ (CORRECTED: was 0.05, ×10 increase)
    elif "PTEN_T2" in cat.name:
        cat.k_cat = 0.05              # s⁻¹ (unchanged: T2 is slow/stable form)
    cat.contact_radius = 150.0  # nm — electrostatic capture radius

# ── COMPARTMENT EXCHANGES: CORRECTED from PDE→BD mapping (paso5) ──
# CRITICAL FIX: PTEN_T1 exchange rates reduced ÷12 and ÷25.
# PDE analysis showed: separation of timescales requires SLOW PTEN
# exchange (τ_PTEN >> τ_wave). Previous values k_on=0.60, k_off=0.50
# gave τ_PTEN ≈ 1s, but waves have τ ≈ 500s → PTEN is 500× too fast.
# Corrected: k_on=0.05, k_off=0.02 → τ_PTEN ≈ 50s (within wave timescale).
#
# Reference: Matsuoka & Ueda (2018) Nat Commun 9:4481 Table 1
for exch in config.compartment_exchanges:
    if "PI3K" in exch.name:
        # PI3K kinetics ×80 from YAML: τ_PI3K ≈ 2.5s (adiabatic, OK)
        exch.k_on *= 80.0      # YAML 0.002 → 0.16 s⁻¹
        exch.k_off *= 80.0     # YAML 0.005 → 0.40 s⁻¹ (τ=2.5s)
    elif "PTEN_T1" in exch.name:
        # CORRECTED: k_on ÷12 (0.60→0.05), k_off ÷25 (0.50→0.02)
        # This creates the timescale separation needed for wave dynamics.
        # PTEN binds SLOWLY and stays LONG on membrane → effective inhibition.
        exch.k_on = 0.05       # s⁻¹ (CORRECTED: was 0.60, ÷12)
        exch.k_off = 0.02      # s⁻¹ (CORRECTED: was 0.50, ÷25)
        # pten_self_activation ACTIVE via secondary_feedback_rule (YAML)
    elif "PTEN_T2" in exch.name:
        pass  # k_on=0, k_off=1.1 — both at Matsuoka values, no boost
    elif "PHAkt" in exch.name:
        # PH-Akt biosensor: PIP3-dependent translocation from cytosol.
        # Total pool = 300 (50 initial + 250 cytosol), low expression for
        # passive biosensor behavior (sequestration <10%).
        # k_on=5.0, k_off=0.3 → f_bound≈50%, seq≈10% at steady state.
        # Ref: Varnai & Balla 1998 — PH domains; Milo & Phillips 2015 §3.5.
        exch.k_on = 5.0             # s⁻¹ (strong recruitment, PIP3-modulated)
        exch.k_off = 0.3            # s⁻¹ (membrane residence ~3s)
        exch.cytosol_pool = 250     # Low expression: total PHAkt≈300 (passive biosensor)
    # Scale max_recruits for larger domain (225µm² vs 25µm²)
    exch.max_recruits_per_step = 20  # was 5, scaled for 9× area

# ── UNIMOLECULAR CONVERSIONS: T1→T2 calibrated ───────────────────
# k_12_base = 0.35 s⁻¹ with pip3_activation_pten (K=15, factor=0.66)
# → effective k_12 = 0.23 s⁻¹ → T2_ss ≈ 15 (40% of total PTEN).
# T2→T1 reversion kept at 0.015 (slow spontaneous deactivation).
# Reference: Knoch et al. (2014) PLOS Comput Biol — PTEN activation.
for conv in config.unimolecular_conversions:
    if conv.name == "T1_to_T2":
        conv.rate = 0.35  # s⁻¹ (base, amplified by pip3_activation_pten)
    # T2→T1 stays at 0.015 (spontaneous reversion)

# ── FEEDBACK RULES: tuned for bifurcation at basal PIP3 ──────────
# pip3_inhibition: K_half=40 with n=4 → at PIP3=20: factor=0.86 (14% inh.)
#   At PIP3=40: factor=0.50 → PTEN drops to half → mutual exclusion.
#   K_half=15 was too aggressive (76% inhibition at basal → PTEN collapse).
#
# pip3_activation: PIP3 recruits more PI3K → positive feedback.
#   K_half = 20/µm² (YAML): at basal, Hill factor = 0.5 → bifurcation.
#   n_hill = 2 (YAML): moderate cooperativity for Ras-PI3K axis.
#
# Reference: Goryachev & Leda (2017) Mol Biol Cell 28:370.
# Feedback tuning: numerically verified bistability (see analysis above).
# At PIP3 ≈ 20/µm²: near equilibrium (bifurcation threshold).
# At PIP3 > 25/µm²: production > clearance → PIP3 wave grows.
# At PIP3 < 15/µm²: clearance > production → zone decays.
# This creates the excitable dynamics needed for traveling waves.
#
# CRITICAL FIX: grid_resolution = 1.0 µm (was 0.2 µm in YAML).
# At 0.2µm cells: area=0.04µm², mean 0.8 PIP3/cell → CV=112%
# → feedback operates on NOISE, not on real spatial patterns.
# At 1.0µm cells: area=1.0µm², mean 20 PIP3/cell → CV=22%
# → feedback detects coherent density fluctuations.
# Predicted wave wavelength λ ≈ 4.5 µm → grid must be ≤ λ/4.
for fb in config.feedback_rules:
    if fb.name == "pip3_inhibition":
        # CORRECTED: K_half=5.0 (was 40), n_hill=3 (was 4)
        # PDE: g(u) = 0.1 + 0.9/(1+(u/5)³) → K_i=5/µm², n=3
        # basal_fraction=0.10 (g_basal): PTEN can ALWAYS bind at 10% rate
        # This prevents the "absorbing HIGH state" where PTEN never returns.
        fb.K_half = 5.0            # /µm² (CORRECTED: was 40, ÷8)
        fb.n_hill = 3.0            # (CORRECTED: was 4)
        fb.basal_fraction = 0.10   # g_basal: PTEN always has 10% binding (NEW)
        fb.grid_resolution = 1.0   # µm
    elif fb.name == "pip3_activation":
        # CORRECTED: K_half=3.0 (was 15), n_hill=3 (was 2)
        # PDE: f(u) = 0.05 + 0.95*u³/(3³+u³) → K_a=3/µm², n=3
        # This is BOTH recruitment AND catalytic feedback now.
        fb.K_half = 3.0            # /µm² (CORRECTED: was 15, ÷5)
        fb.n_hill = 3.0            # (CORRECTED: was 2)
        fb.basal_fraction = 0.05   # α=0.05: prevents absorbing state at PIP3=0
        fb.grid_resolution = 1.0   # µm
    elif fb.name == "pip3_activation_pten":
        fb.K_half = 15.0           # /µm² — match basal PIP3 for T1→T2 conversion
        fb.grid_resolution = 1.0   # µm
    elif fb.name == "pip3_detachment_boost":
        fb.grid_resolution = 1.0   # µm
    elif fb.name == "pten_self_activation":
        fb.grid_resolution = 1.0   # µm

# ── ADAPTATION VARIABLE (FitzHugh-Nagumo slow recovery) ──────────
# This is THE missing ingredient for oscillatory waves.
# w(x,y) tracks PIP3 with delay τ_w=60s and inhibits PI3K production.
# Without this, system is bistable (only fronts/polarity, no waves).
# With this, system becomes excitable (fire-diffuse-fire).
# Ref: Xiong 2010, Huang 2013 — adaptation timescales in Dictyostelium.
config.use_adaptation = True
config.adaptation_tau_w = 60.0     # s — adaptation timescale
config.adaptation_K_w = 10.0       # /µm² — inhibition threshold
config.adaptation_n_w = 2.0        # Hill exponent for adaptation

# ── LOCAL PI3K RECRUITMENT (biased toward PIP3-rich regions) ──────
# CRITICAL FIX: YAML recruitment places PI3K at RANDOM positions.
# PDE assumes LOCAL feedback: PI3K recruits where PIP3 is already high.
# Without locality, feedback gain ÷25 (diluted over entire domain).
# This uses density-weighted inverse-CDF sampling for PI3K position.
config.use_local_pi3k_recruitment = True
config.recruit_sigma = 0.5         # µm — Gaussian spread around target

config.validate()

# ═════════════════════════════════════════════════════════════════
# 2. RUN PIPELINE
# ═════════════════════════════════════════════════════════════════
print("Running PIP3/PTEN simulation...")
t0 = time.perf_counter()
pipeline = SimulationPipeline(config, record_subframe=True)
result = pipeline.run()
wall = time.perf_counter() - t0
print(f"Done in {wall:.1f}s — {result.n_frames} frames, {result.total_photons_emitted:,} photons")

sf = result.subframe_data
time_range = config.tcspc.time_range
bio_times = [result.frames[i].bio_time for i in range(n_frames)]

# ═════════════════════════════════════════════════════════════════
# FIGURE 1: Species dynamics time series
# ═════════════════════════════════════════════════════════════════
print("Generating Figure 1: Species Dynamics...")
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# Frame-level
t_frames = np.array(bio_times)
pip2_f = [result.frames[i].ground_truth.species_counts.get("PIP2", 0) for i in range(n_frames)]
pip3_f = [result.frames[i].ground_truth.species_counts.get("PIP3", 0) for i in range(n_frames)]
pi3k_f = [result.frames[i].ground_truth.species_counts.get("PI3K", 0) for i in range(n_frames)]
t1_f = [result.frames[i].ground_truth.species_counts.get("PTEN_T1", 0) for i in range(n_frames)]
t2_f = [result.frames[i].ground_truth.species_counts.get("PTEN_T2", 0) for i in range(n_frames)]

ax = axes[0]
ax.plot(t_frames, pip2_f, 'b-o', label='PIP2', markersize=4, linewidth=2)
ax.plot(t_frames, pip3_f, 'r-s', label='PIP3', markersize=4, linewidth=2)
ax.set_ylabel('Particle count')
ax.legend(loc='right')
ax.set_title('Lipid Species: PIP2 ↔ PIP3 Interconversion', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(t_frames, t1_f, 'g-o', label='PTEN_T1 (strong)', markersize=4, linewidth=2)
ax.plot(t_frames, t2_f, 'lime', marker='s', label='PTEN_T2 (weak)', markersize=4, linewidth=2)
ax.set_ylabel('Particle count')
ax.legend(loc='right')
ax.set_title('PTEN Isoforms: T1 ↔ T2 Conformational Switching', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(t_frames, pi3k_f, color='orange', marker='D', label='PI3K (enzyme)', markersize=4, linewidth=2)
ltot = [p2 + p3 for p2, p3 in zip(pip2_f, pip3_f)]
ax.plot(t_frames, ltot, 'k--', label='PIP2+PIP3 total', linewidth=2)
ax.set_ylabel('Particle count')
ax.set_xlabel('Biological time (s)')
ax.legend(loc='right')
ax.set_title('Conservation: PI3K invariant, Lipid total constant', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

fig.suptitle('PIP3/PTEN Membrane Signaling — Species Dynamics', fontsize=14, fontweight='bold', y=0.98)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig1_species_dynamics.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ═════════════════════════════════════════════════════════════════
# FIGURE 2: Subframe dynamics (BD-step resolution)
# ═════════════════════════════════════════════════════════════════
print("Generating Figure 2: Subframe Dynamics...")
if sf and sf.species_counts:
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    t_sf = np.arange(len(sf.n_complexes)) * sf.dt_s

    ax = axes[0]
    ax.plot(t_sf, sf.species_counts["PIP3"], 'r-', linewidth=0.5, alpha=0.8)
    ax.set_ylabel('PIP3 count')
    ax.set_title('PIP3(t) — Subframe Resolution (every BD step)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_sf, sf.species_counts["PTEN_T1"], 'g-', linewidth=0.5, alpha=0.8, label='T1')
    ax.plot(t_sf, sf.species_counts["PTEN_T2"], color='lime', linewidth=0.5, alpha=0.8, label='T2')
    ax.set_ylabel('Count')
    ax.set_title('PTEN T1/T2 Membrane Population — Switching Dynamics', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    # Smooth catalytic events over 100-step blocks
    block = 100
    n_blocks = len(sf.n_catalytic) // block
    t_blocks = np.arange(n_blocks) * block * sf.dt_s
    cat_blocks = np.array([sf.n_catalytic[i*block:(i+1)*block].sum() for i in range(n_blocks)])
    rec_blocks = np.array([sf.n_recruited[i*block:(i+1)*block].sum() for i in range(n_blocks)])
    det_blocks = np.array([sf.n_detached[i*block:(i+1)*block].sum() for i in range(n_blocks)])
    ax.bar(t_blocks, cat_blocks, width=block*sf.dt_s*0.8, color='purple', alpha=0.7, label='Catalytic')
    ax.bar(t_blocks, rec_blocks, width=block*sf.dt_s*0.4, color='teal', alpha=0.7, label='Recruited')
    ax.bar(t_blocks, -det_blocks, width=block*sf.dt_s*0.4, color='salmon', alpha=0.7, label='Detached')
    ax.set_ylabel('Events / block')
    ax.set_title(f'Reaction Events (binned {block} steps = {block*sf.dt_s:.2f}s)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    conv_blocks = np.array([sf.n_converted[i*block:(i+1)*block].sum() for i in range(n_blocks)])
    ax.bar(t_blocks, conv_blocks, width=block*sf.dt_s*0.8, color='goldenrod', alpha=0.7)
    ax.set_ylabel('Conversions / block')
    ax.set_xlabel('Biological time (s)')
    ax.set_title('T1 ↔ T2 Conformational Conversions', fontweight='bold')
    ax.grid(True, alpha=0.3)

    fig.suptitle('PIP3/PTEN — Subframe Dynamics (BD-step resolution)', fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig2_subframe_dynamics.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

# ═════════════════════════════════════════════════════════════════
# FIGURE 3: FLIM Lifetime maps (τ per pixel per frame)
# ═════════════════════════════════════════════════════════════════
print("Generating Figure 3: FLIM Lifetime Maps...")
Lx, Ly = config.membrane.size  # needed for extent in imshow
n_show = min(n_frames, 6)
fig, axes = plt.subplots(2, n_show, figsize=(3.5*n_show, 7))

for fi in range(n_show):
    flim_stack = result.get_flim_stack(fi)
    # 3×3 spatial binning: standard FLIM post-processing (Becker 2005 §5.2).
    # Sums TCSPC histograms from 9 neighboring pixels → SNR ×3.
    flim_stack_binned = spatial_bin_flim(flim_stack)
    # min_counts=10: at <10 photons the mean_arrival_time estimator
    # degenerates to ~t_max/2 (uniform noise).  Lakowicz (2006) §4.4.
    tau_map, intensity = mean_arrival_time(flim_stack_binned, time_range, channel=0, min_counts=10)

    # Lifetime map — mask insufficient pixels as gray
    ax = axes[0, fi]
    tau_display = np.ma.masked_invalid(tau_map.copy())
    tau_display = np.ma.clip(tau_display, 0.8, 3.5)
    cmap_tau = plt.cm.RdYlBu.copy()
    cmap_tau.set_bad(color='0.85')  # gray for masked/NaN pixels
    # Colorbar calibrated for FRET-corrected donor TCSPC (no acceptor contamination):
    # τ_D=2.793ns (free), τ_DA=1.080ns (bound). At f_bound≈50%: τ_pixel≈1.94ns.
    # Range [1.0, 3.0] captures full FRET dynamic range including pure-bound pixels.
    # Ref: Lakowicz 2006 §13.2 — FRET lifetime imaging display conventions.
    im = ax.imshow(tau_display.T, origin='lower', cmap=cmap_tau,
                   extent=[0, Lx, 0, Ly], vmin=1.00, vmax=3.00)
    ax.set_title(f't={bio_times[fi]:.1f}s', fontsize=10)
    if fi == 0:
        ax.set_ylabel('τ map (ns)\ny (µm)')
    ax.set_xlabel('x (µm)')
    plt.colorbar(im, ax=ax, shrink=0.8, label='τ (ns)')

    # Intensity map
    ax = axes[1, fi]
    im2 = ax.imshow(intensity.T, origin='lower', cmap='hot',
                    extent=[0, Lx, 0, Ly])
    if fi == 0:
        ax.set_ylabel('Intensity\ny (µm)')
    ax.set_xlabel('x (µm)')
    plt.colorbar(im2, ax=ax, shrink=0.8, label='counts')

fig.suptitle('PIP3/PTEN FLIM: Lifetime Maps [3×3 binned] (top) & Intensity (bottom)', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig3_flim_lifetime_maps.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ═════════════════════════════════════════════════════════════════
# FIGURE 4: Phasor plot
# ═════════════════════════════════════════════════════════════════
print("Generating Figure 4: Phasor Plot...")
fig, ax = plt.subplots(1, 1, figsize=(7, 7))

# Universal semicircle
theta = np.linspace(0, np.pi, 200)
ax.plot(0.5 + 0.5*np.cos(theta), 0.5*np.sin(theta), 'k-', linewidth=1.5, label='Universal circle')

colors_phasor = cm.viridis(np.linspace(0, 1, n_frames))
for fi in range(n_frames):
    flim_stack = result.get_flim_stack(fi)
    phasor = phasor_transform(flim_stack, time_range, harmonic=1, channel=0)
    valid = ~np.isnan(phasor.g) & ~np.isnan(phasor.s) & phasor.mask
    if valid.any():
        G = phasor.g[valid]
        S = phasor.s[valid]
        ax.scatter(G, S, s=5, alpha=0.4, color=colors_phasor[fi],
                   label=f't={bio_times[fi]:.0f}s' if fi % 3 == 0 else None)

ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.05, 0.65)
ax.set_xlabel('G (real)', fontsize=12)
ax.set_ylabel('S (imaginary)', fontsize=12)
ax.set_title('Phasor Plot — PIP3/PTEN FLIM-FRET Biosensor', fontsize=13, fontweight='bold')
ax.set_aspect('equal')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Mark expected lifetimes
# τ_D = 1/(k_S1→S0 + k_ISC + k_bl) = 1/(3.57e8 + 1e6 + 1e3) ≈ 2.793 ns
# τ_DA = τ_D / (1 + (R0/r)^6) = 2.793 / (1 + (5.4/5.0)^6) = 1.080 ns
# Ref: Förster (1948), Lakowicz (2006) §13.2
tau_D_ns = 2.793   # mTFP1 donor-only lifetime (from YAML thermal_rates)
R0, r_fret = 5.4, 5.0  # nm (from YAML fret_pairs)
tau_DA_ns = tau_D_ns / (1.0 + (R0 / r_fret)**6)  # FRET-quenched lifetime
omega = 2 * np.pi / (time_range * 1e9)  # rad/ns
for tau, label, color in [(tau_D_ns, f'τ_D={tau_D_ns:.2f}ns', 'blue'),
                           (tau_DA_ns, f'τ_DA={tau_DA_ns:.2f}ns', 'red')]:
    G_exp = 1 / (1 + (omega * tau)**2)
    S_exp = omega * tau / (1 + (omega * tau)**2)
    ax.plot(G_exp, S_exp, '*', color=color, markersize=15, markeredgecolor='k', label=label)

# Draw FRET trajectory line (τ_D ↔ τ_DA)
G_D = 1 / (1 + (omega * tau_D_ns)**2)
S_D = omega * tau_D_ns / (1 + (omega * tau_D_ns)**2)
G_DA = 1 / (1 + (omega * tau_DA_ns)**2)
S_DA = omega * tau_DA_ns / (1 + (omega * tau_DA_ns)**2)
ax.plot([G_D, G_DA], [S_D, S_DA], '--', color='gray', linewidth=1.2,
        alpha=0.7, label='FRET trajectory')

ax.legend(fontsize=8, ncol=2, loc='upper right')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig4_phasor_plot.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ═════════════════════════════════════════════════════════════════
# FIGURE 5: Photon statistics + bleaching
# ═════════════════════════════════════════════════════════════════
print("Generating Figure 5: Photon Statistics...")
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Photons per frame
em_per_frame = [result.frames[i].n_photons_emitted for i in range(n_frames)]
det_per_frame = [result.frames[i].n_photons_detected for i in range(n_frames)]
ax = axes[0, 0]
ax.plot(t_frames, em_per_frame, 'b-o', label='Emitted', linewidth=2, markersize=4)
ax.plot(t_frames, det_per_frame, 'r-s', label='Detected', linewidth=2, markersize=4)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Photons')
ax.set_title('Photons per Frame', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Bleaching
bleach_f = [result.frames[i].ground_truth.n_bleached for i in range(n_frames)]
ax = axes[0, 1]
ax.plot(t_frames, bleach_f, 'k-o', linewidth=2, markersize=4)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Bleached dyes')
ax.set_title('Photobleaching Progression', fontweight='bold')
ax.grid(True, alpha=0.3)

# Mean lifetime per frame (3×3 binned for consistent SNR)
tau_means = []
for fi in range(n_frames):
    flim_stack = result.get_flim_stack(fi)
    flim_stack_binned = spatial_bin_flim(flim_stack)
    tau_map, intensity = mean_arrival_time(flim_stack_binned, time_range, channel=0, min_counts=5)
    valid = ~np.isnan(tau_map) & (intensity > 0)
    if valid.any():
        tau_means.append(np.average(tau_map[valid], weights=intensity[valid]))
    else:
        tau_means.append(np.nan)

ax = axes[1, 0]
ax.plot(t_frames, tau_means, 'purple', marker='o', linewidth=2, markersize=5)
ax.axhline(y=2.793, color='blue', linestyle='--', alpha=0.5, label='τ_D = 2.793 ns (mTFP1)')
ax.axhline(y=1.080, color='red', linestyle='--', alpha=0.5, label='τ_DA = 1.080 ns (FRET)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('<τ> (ns)')
ax.set_title('Mean Lifetime Evolution', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Counts per pixel histogram (frame 0)
ax = axes[1, 1]
flim_0 = result.get_flim_stack(0)
total_per_pixel = flim_0[0].sum(axis=-1).flatten()
total_per_pixel = total_per_pixel[total_per_pixel > 0]
ax.hist(total_per_pixel, bins=30, color='darkcyan', edgecolor='k', alpha=0.7)
ax.set_xlabel('Photons per pixel')
ax.set_ylabel('Number of pixels')
ax.set_title(f'Photon Count Distribution (Frame 0)', fontweight='bold')
ax.axvline(x=np.mean(total_per_pixel), color='red', linestyle='--',
           label=f'mean={np.mean(total_per_pixel):.0f}')
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle('PIP3/PTEN FLIM — Photon Statistics & Bleaching', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig5_photon_statistics.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ═════════════════════════════════════════════════════════════════
# FIGURE 6: Conservation laws + event summary
# ═════════════════════════════════════════════════════════════════
print("Generating Figure 6: Conservation & Events Summary...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Conservation
ax = axes[0]
ltot = [pip2_f[i] + pip3_f[i] for i in range(n_frames)]
ax.plot(t_frames, ltot, 'k-o', linewidth=2.5, markersize=6, label='PIP2 + PIP3')
ax.set_ylim([min(ltot) - 5, max(ltot) + 5])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Total lipid count')
ax.set_title('Lipid Conservation Law: PIP2 + PIP3 = const', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Event counts per frame
ax = axes[1]
spf = sf.steps_per_frame if sf else n_bd_per_frame
cat_per_f = [sf.n_catalytic[i*spf:(i+1)*spf].sum() for i in range(n_frames)] if sf else []
rec_per_f = [sf.n_recruited[i*spf:(i+1)*spf].sum() for i in range(n_frames)] if sf else []
det_per_f = [sf.n_detached[i*spf:(i+1)*spf].sum() for i in range(n_frames)] if sf else []
conv_per_f = [sf.n_converted[i*spf:(i+1)*spf].sum() for i in range(n_frames)] if sf else []

x = np.arange(n_frames)
width = 0.2
ax.bar(x - 1.5*width, cat_per_f, width, label='Catalytic', color='purple')
ax.bar(x - 0.5*width, rec_per_f, width, label='Recruited', color='teal')
ax.bar(x + 0.5*width, det_per_f, width, label='Detached', color='salmon')
ax.bar(x + 1.5*width, conv_per_f, width, label='T1↔T2', color='goldenrod')
ax.set_xticks(x)
ax.set_xticklabels([f'{t:.0f}s' for t in t_frames])
ax.set_xlabel('Frame (bio time)')
ax.set_ylabel('Events per frame')
ax.set_title('Reaction Events per Frame', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('PIP3/PTEN — Conservation & Reaction Events', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig6_conservation_events.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ═════════════════════════════════════════════════════════════════
# FIGURE 7: Causal-FLIM — 7 non-trivial causal pairs
# ═════════════════════════════════════════════════════════════════
# References:
#   Schreiber (2000) Phys Rev Lett 85:461 — Transfer Entropy
#   Wibral et al. (2014) J Comput Neurosci 30:45 — Estimation
#
# 5 independent bidirectional pairs (consolidated from 7):
#   te_ksg_bidirectional computes BOTH TE(X→Y) and TE(Y→X), so
#   pairs 3/5 and 4/6 (which were reverses) are now single entries.
#
#   Q1. PIP3 ↔ Recruitment — Hill feedback active?
#   Q2. PTEN_T1 ↔ PIP3 — Catalysis confirmed? (KEY)
#   Q3. PTEN_T1 ↔ τ — FLIM detects activity?
#   Q4. PTEN_T2 ↔ τ — Or only density?
#   Q5. Intensity → τ — Control (should be ≈0)
#
# Corrections applied (Wibral 2013, 2014):
#   • Linear detrending for stationarity
#   • Optimal lag selection by max asymmetry criterion
# ═════════════════════════════════════════════════════════════════
print("Generating Figure 7: Causal-FLIM (5 independent bidirectional pairs)...")

causal_pairs = [
    # ── Pair 1: PIP3 ↔ Recruitment (Hill feedback) ──
    CausalPairSpec(
        name="PIP3 ↔ Recruitment",
        x_key="species:PIP3",
        y_key="n_recruited",
        x_label="PIP3(t)",
        y_label="n_recruited(t)",
        interpretation_if_significant=(
            "PIP3 causally drives PTEN recruitment — Hill feedback active."
        ),
        interpretation_if_not=(
            "No causal PIP3 → recruitment: feedback may be too weak or delayed."
        ),
    ),
    # ── Pair 2: PTEN_T1 ↔ PIP3 (catalytic dephosphorylation — KEY) ──
    CausalPairSpec(
        name="PTEN_T1 ↔ PIP3",
        x_key="species:PTEN_T1",
        y_key="species:PIP3",
        x_label="PTEN_T1(t)",
        y_label="PIP3(t)",
        interpretation_if_significant=(
            "PTEN_T1 causally clears PIP3 — catalytic dephosphorylation confirmed."
        ),
        interpretation_if_not=(
            "No causal PTEN_T1 → PIP3: catalysis may be rate-limited."
        ),
    ),
    # ── Pair 3: PTEN_T1 ↔ τ (biosensor vs enzymatic activity) ──
    CausalPairSpec(
        name="PTEN_T1 ↔ τ",
        x_key="species:PTEN_T1",
        y_key="tau_mean",
        x_label="PTEN_T1(t)",
        y_label="⟨τ⟩ (ns)",
        interpretation_if_significant=(
            "PTEN_T1 drives τ: FLIM responds to enzymatic activity."
        ),
        interpretation_if_not=(
            "PTEN_T1 does not drive τ: FLIM may reflect density, not activity."
        ),
    ),
    # ── Pair 4: PTEN_T2 ↔ τ (biosensor vs weak isoform density) ──
    CausalPairSpec(
        name="PTEN_T2 ↔ τ",
        x_key="species:PTEN_T2",
        y_key="tau_mean",
        x_label="PTEN_T2(t)",
        y_label="⟨τ⟩ (ns)",
        interpretation_if_significant=(
            "PTEN_T2 drives τ: FLIM reflects total PTEN density (T1+T2)."
        ),
        interpretation_if_not=(
            "PTEN_T2 does not drive τ: only active T1 modulates FLIM."
        ),
    ),
    # ── Pair 5: Intensity → τ (control — should be ≈0) ──
    CausalPairSpec(
        name="Intensity → τ (control)",
        x_key="intensity",
        y_key="tau_mean",
        x_label="Intensity(t)",
        y_label="⟨τ⟩ (ns)",
        interpretation_if_significant=(
            "WARNING: Intensity causally drives τ — photon statistics artifact."
        ),
        interpretation_if_not=(
            "GOOD: Intensity does not drive τ — FLIM readout is unbiased."
        ),
    ),
]

causal_result = compute_causal_flim_generic(
    result,
    causal_pairs,
    tau_d_ns=2.793,     # mTFP1 unquenched lifetime (1/3.58e8 s⁻¹ from YAML)
    n_shuffle=200,      # 200 surrogates for robust significance
    detrend=True,
    detrend_method="diff2",  # BUG FIX 2026-03-04: was "linear" — epistemically invalid
    # for non-linear growth (logistic PIP3 accumulation). Linear detrend
    # leaves ACF(1)=0.976 → spurious TE (FP=30%). Single diff also fails
    # (FP=22%) because diff(logistic)=bell curve is still non-stationary.
    # Double differencing (diff2) achieves:
    #   - ADF stationarity: 100% pass rate
    #   - FP rate: 6.2% [CI 4.8-7.9%], well-calibrated at α=5%
    #   - Power: 43% at coupling=0.7 (correct scaling)
    # Monte Carlo validated: 1000 independent logistic pairs, binomial p=0.094.
    # Ref: Granger & Joyeux (1980) JRSS-B; Schreiber (2000) PRL 85:461
    lag_range=[1, 2, 5, 10, 20],
    auto_lag=True,
)

# ── Visualization: horizontal bar chart + interpretation table ──
n_pairs_plot = len(causal_result.pairs)
fig, axes = plt.subplots(1, 2, figsize=(16, max(6, 0.9 * n_pairs_plot)),
                         gridspec_kw={'width_ratios': [2, 3]})

# Left panel: TE bar chart (X→Y and Y→X side by side)
ax = axes[0]
pair_names = [p.pair_name for p in causal_result.pairs]
te_xy = np.array([p.te_xy for p in causal_result.pairs])
te_yx = np.array([p.te_yx for p in causal_result.pairs])
te_xy_null = np.array([p.te_xy_null for p in causal_result.pairs])
te_yx_null = np.array([p.te_yx_null for p in causal_result.pairs])
sig_xy = [p.significant_xy for p in causal_result.pairs]
sig_yx = [p.significant_yx for p in causal_result.pairs]

y_pos = np.arange(n_pairs_plot)
bar_h = 0.35

# TE(X→Y) bars
colors_xy = ['#2196F3' if s else '#BBDEFB' for s in sig_xy]
ax.barh(y_pos + bar_h/2, te_xy, bar_h, color=colors_xy, edgecolor='#1565C0',
        linewidth=0.5, label='TE(X→Y)')
# TE(Y→X) bars
colors_yx = ['#F44336' if s else '#FFCDD2' for s in sig_yx]
ax.barh(y_pos - bar_h/2, te_yx, bar_h, color=colors_yx, edgecolor='#B71C1C',
        linewidth=0.5, label='TE(Y→X)')

# Null thresholds
for i, (nxy, nyx) in enumerate(zip(te_xy_null, te_yx_null)):
    ax.plot(nxy, i + bar_h/2, '|', color='navy', markersize=10, markeredgewidth=2)
    ax.plot(nyx, i - bar_h/2, '|', color='darkred', markersize=10, markeredgewidth=2)

ax.set_yticks(y_pos)
ax.set_yticklabels(pair_names, fontsize=9)
ax.set_xlabel('Transfer Entropy (bits)', fontsize=11)
ax.set_title('Causal-FLIM: Transfer Entropy', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# Right panel: Interpretation table
ax2 = axes[1]
ax2.axis('off')

table_data = []
for p in causal_result.pairs:
    sig_mark = "✓ *" if p.significant_xy else "✗"
    table_data.append([
        p.pair_name,
        f"{p.te_xy:.4f}",
        f"{p.te_xy_null:.4f}",
        sig_mark,
        f"lag*={p.optimal_lag}",
        p.interpretation[:70] if p.interpretation else "",
    ])

col_labels = ["Pair", "TE(X→Y)", "Null", "Sig?", "Lag*", "Interpretation"]
table = ax2.table(
    cellText=table_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='left',
    colWidths=[0.16, 0.09, 0.09, 0.05, 0.06, 0.55],
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)

# Color significant rows
for row_idx, p in enumerate(causal_result.pairs):
    for col_idx in range(6):
        cell = table[row_idx + 1, col_idx]
        if p.significant_xy:
            cell.set_facecolor('#E3F2FD')
        if col_idx == 3:
            cell.set_text_props(fontweight='bold',
                                color='green' if p.significant_xy else 'gray')

# Header styling
for col_idx in range(6):
    table[0, col_idx].set_facecolor('#1565C0')
    table[0, col_idx].set_text_props(color='white', fontweight='bold')

fig.suptitle(
    f'Causal-FLIM: 5 Independent Bidirectional Pairs — PIP3/PTEN System\n'
    f'({causal_result.n_subframe_points:,} subframe pts, '
    f'{causal_result.n_frames} frames, n_shuffle=200, detrend=linear, auto_lag)',
    fontsize=12, fontweight='bold', y=0.98,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT, 'fig7_causal_flim.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ═════════════════════════════════════════════════════════════════
# Print Causal-FLIM summary to console
# ═════════════════════════════════════════════════════════════════
print("\n── Causal-FLIM Results (detrend=linear, auto_lag) ──")
for p in causal_result.pairs:
    sig_xy = "✓" if p.significant_xy else "✗"
    sig_yx = "✓" if p.significant_yx else "✗"
    dt_tag = f" [detrend={p.detrend_applied}]" if p.detrend_applied else ""
    print(f"  {p.pair_name:25s} TE(X→Y)={p.te_xy:.4f} TE(Y→X)={p.te_yx:.4f} "
          f"lag*={p.optimal_lag} sig={sig_xy}/{sig_yx}{dt_tag}")
    if p.interpretation:
        print(f"    → {p.interpretation}")

# ═════════════════════════════════════════════════════════════════
# DONE
# ═════════════════════════════════════════════════════════════════
# FIGURE 8: Radial Kymograph — Species density vs (radius, time)
#
# For each frame, compute radial density profile of each species
# from the membrane center (1.0, 1.0 µm). Plot as heatmap:
#   X-axis = radial distance from center (0 → 1.0 µm)
#   Y-axis = biological time (0 → 20s)
# Diagonal bands reveal propagating excitable waves.
#
# Reference: Goryachev & Leda (2017) Mol Biol Cell 28:370
# ═════════════════════════════════════════════════════════════════
print("Generating Figure 8: Radial Kymograph (3 key species)...")

from scipy.ndimage import uniform_filter

membrane_center = np.array([config.membrane.size[0] / 2,
                            config.membrane.size[1] / 2])
max_radius = min(config.membrane.size) / 2  # µm
n_radial_bins = 15  # fewer bins → less shot noise per bin
radial_edges = np.linspace(0, max_radius, n_radial_bins + 1)
radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])

# Focus on 3 key signal species (PIP2 is background, PI3K is sparse)
kymo_species = ["PIP3", "PTEN_T1", "PTEN_T2"]
kymo_cmaps = {"PIP3": "Reds", "PTEN_T1": "Greens", "PTEN_T2": "YlGn"}
species_names_map = {sp.name: idx for idx, sp in enumerate(config.species)}

# Build kymograph matrices: [n_frames, n_radial_bins]
kymo = {sp: np.zeros((n_frames, n_radial_bins)) for sp in kymo_species}

for fi in range(n_frames):
    gt = result.frames[fi].ground_truth
    pos = gt.positions
    sid = gt.species_id
    alive = gt.is_alive
    r = np.sqrt(((pos - membrane_center) ** 2).sum(axis=1))

    for sp in kymo_species:
        sp_idx = species_names_map.get(sp, -1)
        if sp_idx < 0:
            continue
        mask = (sid == sp_idx) & alive
        if mask.sum() == 0:
            continue
        counts, _ = np.histogram(r[mask], bins=radial_edges)
        areas = np.pi * (radial_edges[1:] ** 2 - radial_edges[:-1] ** 2)
        areas = np.maximum(areas, 1e-6)
        kymo[sp][fi, :] = counts / areas

# Apply temporal smoothing (3-frame moving average) to reveal structure
for sp in kymo_species:
    kymo[sp] = uniform_filter(kymo[sp], size=(3, 1), mode='nearest')

# Plot: 3 subplots for key species + 1 for PIP3/PTEN anticorrelation
t_bio = np.array([result.frames[i].bio_time for i in range(n_frames)])
fig_kymo, axes_k = plt.subplots(1, 4, figsize=(22, 7), sharey=True,
                                gridspec_kw={'width_ratios': [1, 1, 1, 1.2]})

for ax, sp in zip(axes_k[:3], kymo_species):
    data = kymo[sp]
    vmax = np.percentile(data, 97) if data.max() > 0 else 1.0
    im = ax.pcolormesh(radial_centers, t_bio, data,
                       cmap=kymo_cmaps[sp], vmin=0, vmax=vmax,
                       shading='gouraud')  # smooth interpolation
    ax.set_xlabel('r (µm)', fontsize=10)
    ax.set_title(sp, fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='ρ (µm⁻²)', shrink=0.8)

# 4th panel: PIP3 − PTEN_T1 difference (anticorrelation reveals waves)
ax_diff = axes_k[3]
# Normalize each to [0,1] then compute difference
pip3_norm = kymo["PIP3"] / (kymo["PIP3"].max() + 1e-10)
pten_norm = kymo["PTEN_T1"] / (kymo["PTEN_T1"].max() + 1e-10)
diff_map = pip3_norm - pten_norm  # +1 = PIP3 dominant, -1 = PTEN dominant
vabs = max(abs(diff_map.min()), abs(diff_map.max()), 0.5)
im_d = ax_diff.pcolormesh(radial_centers, t_bio, diff_map,
                           cmap='RdBu_r', vmin=-vabs, vmax=vabs,
                           shading='gouraud')
ax_diff.set_xlabel('r (µm)', fontsize=10)
ax_diff.set_title('PIP3 − PTEN (norm)', fontsize=13, fontweight='bold')
plt.colorbar(im_d, ax=ax_diff, label='Δρ (red=PIP3, blue=PTEN)', shrink=0.8)

axes_k[0].set_ylabel('Biological time (s)', fontsize=11)
fig_kymo.suptitle(
    'Radial Kymograph — Excitable Wave Signatures\n'
    f'Membrane {config.membrane.size[0]:.0f}×{config.membrane.size[1]:.0f} µm², '
    f'{n_frames} frames, {n_radial_bins} radial bins, 3-frame smoothing',
    fontsize=14, fontweight='bold', y=1.02
)
fig_kymo.tight_layout()
fig_kymo.savefig(os.path.join(OUT, 'fig8_radial_kymograph.png'), dpi=150, bbox_inches='tight')
plt.close(fig_kymo)

# ═════════════════════════════════════════════════════════════════
# FIGURE 9: Triple-Panel Evolutivo — BD density | τ | Intensity
#
# 8 rows × 3 columns. BD column shows 2D density heatmap (not
# individual points) for PIP3 and PTEN, smoothed with Gaussian
# kernel. This reveals spatial patterns much better than scatter.
# ═════════════════════════════════════════════════════════════════
print("Generating Figure 9: Triple-Panel Evolutivo (BD density | τ | Intensity)...")

from scipy.ndimage import gaussian_filter

n_show_tp = 8
frame_indices_tp = np.linspace(0, n_frames - 1, n_show_tp, dtype=int)

Lx, Ly = config.membrane.size
npx_density = 32  # bins for 2D density histogram (matches pixel grid: CORRECTED from 64)
x_edges = np.linspace(0, Lx, npx_density + 1)
y_edges = np.linspace(0, Ly, npx_density + 1)

fig_tp, axes_tp = plt.subplots(n_show_tp, 3, figsize=(14, n_show_tp * 2.8))

for row, fi in enumerate(frame_indices_tp):
    gt = result.frames[fi].ground_truth
    pos = gt.positions
    sid = gt.species_id
    alive = gt.is_alive
    bio_t = result.frames[fi].bio_time

    # ── Left: BD 2D density (PIP3 red channel, PTEN green channel) ──
    ax_bd = axes_tp[row, 0]

    # Build RGB density image: R=PIP3, G=PTEN_T1, B=PTEN_T2
    rgb_img = np.zeros((npx_density, npx_density, 3))
    for sp, ch in [("PIP3", 0), ("PTEN_T1", 1), ("PTEN_T2", 2)]:
        sp_idx = species_names_map.get(sp, -1)
        if sp_idx < 0:
            continue
        mask = (sid == sp_idx) & alive
        if mask.sum() == 0:
            continue
        h, _, _ = np.histogram2d(pos[mask, 0], pos[mask, 1],
                                 bins=[x_edges, y_edges])
        h = gaussian_filter(h, sigma=1.5)  # smooth for visual clarity
        rgb_img[:, :, ch] = h

    # Normalize each channel to [0,1]
    for ch in range(3):
        mx = rgb_img[:, :, ch].max()
        if mx > 0:
            rgb_img[:, :, ch] /= mx

    ax_bd.imshow(rgb_img.transpose(1, 0, 2), origin='lower',
                 extent=[0, Lx, 0, Ly], aspect='equal', interpolation='bicubic')
    ax_bd.set_ylabel(f't={bio_t:.0f}s', fontsize=10, fontweight='bold')
    if row == 0:
        ax_bd.set_title('BD Density (R=PIP3 G=PTEN₁ B=PTEN₂)',
                        fontsize=9, fontweight='bold')
    if row < n_show_tp - 1:
        ax_bd.set_xticklabels([])

    # ── Center: FLIM τ map (3×3 binned, NaN shown as gray) ──
    ax_tau = axes_tp[row, 1]
    flim_stack_fi = result.get_flim_stack(fi)
    flim_stack_fi_binned = spatial_bin_flim(flim_stack_fi)
    tau_map_fi, int_map_fi = mean_arrival_time(
        flim_stack_fi_binned, time_range, channel=0, min_counts=5
    )
    # 3×3 binning (Becker 2005 §5.2) + min_counts=5: SNR improves ×3.
    # Use 'nearest' interpolation: does NOT propagate NaN to neighbors.
    cmap_tau = plt.cm.RdYlBu.copy()
    cmap_tau.set_bad(color='#404040')  # dark gray = "no signal"
    # Colorbar [1.0, 3.0] ns: full FRET range after channel-purity fix.
    # τ_DA=1.080ns (bound), τ_D=2.793ns (free), f_bound≈50% → τ_pixel≈1.94ns.
    im_tau = ax_tau.imshow(tau_map_fi.T, origin='lower', cmap=cmap_tau,
                           extent=[0, Lx, 0, Ly],
                           vmin=1.00, vmax=3.00, aspect='equal',
                           interpolation='nearest')
    if row == 0:
        ax_tau.set_title('FLIM τ (ns)', fontsize=11, fontweight='bold')
    if row < n_show_tp - 1:
        ax_tau.set_xticklabels([])

    # ── Right: Intensity map (bicubic interpolation) ──
    ax_int = axes_tp[row, 2]
    im_int = ax_int.imshow(int_map_fi.T, origin='lower', cmap='hot',
                           extent=[0, Lx, 0, Ly],
                           aspect='equal', interpolation='bicubic')
    if row == 0:
        ax_int.set_title('Intensity (counts)', fontsize=11, fontweight='bold')
    if row < n_show_tp - 1:
        ax_int.set_xticklabels([])

fig_tp.suptitle(
    'BD ↔ FLIM ↔ Intensity: Evolution over 20s\n'
    f'{n_frames} frames, {config.scan.pixels_x}×{config.scan.pixels_y} px, '
    f'{Lx:.0f}×{Ly:.0f} µm²',
    fontsize=13, fontweight='bold', y=1.02
)
fig_tp.tight_layout(rect=[0, 0, 0.92, 0.97])
# Colorbars in narrow strips at the right margin
cax_tau = fig_tp.add_axes([0.93, 0.52, 0.015, 0.40])
fig_tp.colorbar(im_tau, cax=cax_tau, label='τ (ns)')
cax_int = fig_tp.add_axes([0.93, 0.05, 0.015, 0.40])
fig_tp.colorbar(im_int, cax=cax_int, label='Counts')
fig_tp.savefig(os.path.join(OUT, 'fig9_triple_panel_evolution.png'), dpi=150, bbox_inches='tight')
plt.close(fig_tp)

# ═════════════════════════════════════════════════════════════════
# FIGURE 10: Overlay τ map + BD particles (6 timepoints)
#
# FLIM τ map (bicubic-smoothed) as background, BD particles
# overlaid with larger markers. 6 panels for finer temporal
# sampling of wave evolution.
# ═════════════════════════════════════════════════════════════════
print("Generating Figure 10: Overlay τ + BD particles...")

n_show_ov = 6
frame_indices_ov = np.linspace(0, n_frames - 1, n_show_ov, dtype=int)

fig_ov, axes_ov = plt.subplots(2, 3, figsize=(18, 12))
axes_ov_flat = axes_ov.flatten()

species_scatter_colors = {
    "PIP3": "#FF1744",    # bright red
    "PTEN_T1": "#00E676", # bright green
    "PTEN_T2": "#76FF03", # lime
}

for idx, fi in enumerate(frame_indices_ov):
    ax = axes_ov_flat[idx]
    gt = result.frames[fi].ground_truth
    pos = gt.positions
    sid = gt.species_id
    alive = gt.is_alive
    bio_t = result.frames[fi].bio_time

    # τ map as background (3×3 binned, NaN as gray, no invented data)
    flim_stack_fi = result.get_flim_stack(fi)
    flim_stack_fi_binned = spatial_bin_flim(flim_stack_fi)
    tau_map_fi, _ = mean_arrival_time(
        flim_stack_fi_binned, time_range, channel=0, min_counts=5
    )
    # 3×3 spatial binning (Becker 2005 §5.2): SNR ×3, resolution 937nm
    cmap_ov = plt.cm.RdYlBu.copy()
    cmap_ov.set_bad(color='#404040')  # dark gray = no signal
    # Colorbar [1.0, 3.0] ns: full FRET range after channel-purity fix
    im = ax.imshow(tau_map_fi.T, origin='lower', cmap=cmap_ov,
                   extent=[0, Lx, 0, Ly],
                   vmin=1.00, vmax=3.00, aspect='equal',
                   interpolation='nearest', alpha=0.85)

    # Overlay BD particles — only signal species, large markers
    for sp in ["PIP3", "PTEN_T1", "PTEN_T2"]:
        sp_idx = species_names_map.get(sp, -1)
        if sp_idx < 0:
            continue
        mask = (sid == sp_idx) & alive
        if mask.sum() == 0:
            continue
        marker = '^' if sp == "PIP3" else 'o'
        edge = 'white'
        sz = 12 if sp == "PIP3" else 10  # smaller for 10×10 µm (many particles)
        ax.scatter(pos[mask, 0], pos[mask, 1],
                   c=species_scatter_colors[sp],
                   s=sz, alpha=0.85,
                   label=sp if idx == 0 else None,
                   edgecolors=edge, linewidths=0.3, marker=marker,
                   zorder=5)

    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_title(f't = {bio_t:.0f}s', fontsize=12, fontweight='bold')
    if idx == 0:
        ax.legend(fontsize=9, loc='upper left', framealpha=0.9,
                  markerscale=2.0, fancybox=True)

# Colorbar: manual axes to avoid overlap with panels
cax_ov = fig_ov.add_axes([0.93, 0.15, 0.015, 0.70])
fig_ov.colorbar(im, cax=cax_ov, label='τ (ns)')
fig_ov.suptitle(
    'FLIM τ Map [3×3 binned] + BD Particle Overlay — PIP3/PTEN Spatial Dynamics\n'
    f'Background: FLIM lifetime (3×3 binned, gray=no signal). ▲ PIP3   ● PTEN_T1   ● PTEN_T2',
    fontsize=14, fontweight='bold', y=1.02
)
fig_ov.subplots_adjust(right=0.91, wspace=0.25, hspace=0.25)
fig_ov.savefig(os.path.join(OUT, 'fig10_tau_bd_overlay.png'), dpi=200, bbox_inches='tight')
plt.close(fig_ov)

# ═════════════════════════════════════════════════════════════════
# FIGURE 11: Quantitative FLIM↔BD Correlation Analysis
# ═════════════════════════════════════════════════════════════════
print("Computing FLIM↔BD correlation...")
from scipy import stats as sp_stats

species_names_list = [sp.name for sp in config.species]
pip3_id = species_names_list.index("PIP3")

corr_results = []
for fi in range(n_frames):
    frame = result.frames[fi]
    gt = frame.ground_truth
    pos_f = gt.positions
    sid_f = gt.species_id
    alv_f = gt.is_alive

    pip3_mask = (sid_f == pip3_id) & alv_f
    pip3_pos = pos_f[pip3_mask]

    pip3_dens, _, _ = np.histogram2d(
        pip3_pos[:, 0], pip3_pos[:, 1],
        bins=[np.linspace(0, Lx, npx_density+1), np.linspace(0, Ly, npx_density+1)]
    )

    flim_stack_fi = result.get_flim_stack(fi)
    flim_stack_fi_binned = spatial_bin_flim(flim_stack_fi)
    tau_fi, int_fi = mean_arrival_time(flim_stack_fi_binned, time_range, channel=0, min_counts=5)

    valid = ~np.isnan(tau_fi) & (int_fi >= 5)
    n_valid = valid.sum()
    coverage = n_valid / (npx_density * npx_density)

    if n_valid < 10:
        corr_results.append((frame.bio_time, pip3_mask.sum(), n_valid, coverage, np.nan, np.nan, np.nan, np.nan, np.nan))
        continue

    tau_v = tau_fi[valid]
    pip3_v = pip3_dens[valid]
    r, p = sp_stats.pearsonr(pip3_v, tau_v)
    rho, p_s = sp_stats.spearmanr(pip3_v, tau_v)
    med = np.median(pip3_v)
    tau_h = np.mean(tau_v[pip3_v > med]) if (pip3_v > med).sum() > 0 else np.nan
    tau_l = np.mean(tau_v[pip3_v <= med]) if (pip3_v <= med).sum() > 0 else np.nan
    delta = tau_l - tau_h

    corr_results.append((frame.bio_time, pip3_mask.sum(), n_valid, coverage, r, p, rho, delta, np.mean(int_fi[valid])))

    print(f"  t={frame.bio_time:.0f}s: PIP3={pip3_mask.sum()}, "
          f"valid={n_valid}/{npx_density**2} ({coverage:.0%}), "
          f"r={r:+.3f} (p={p:.2e}), Δτ={delta:+.3f}ns, "
          f"<I>={np.mean(int_fi[valid]):.0f}")

# Plot correlation evolution
fig_corr, axes_corr = plt.subplots(1, 3, figsize=(15, 4))
times = [c[0] for c in corr_results]
rs = [c[4] for c in corr_results]
deltas = [c[7] for c in corr_results]
coverages = [c[3] for c in corr_results]

axes_corr[0].plot(times, rs, 'bo-', linewidth=2, markersize=8)
axes_corr[0].axhline(0, color='gray', linestyle='--')
axes_corr[0].axhline(-0.1, color='red', linestyle=':', alpha=0.5, label='weak threshold')
axes_corr[0].set_xlabel('Time (s)')
axes_corr[0].set_ylabel('Pearson r (PIP3 density, τ)')
axes_corr[0].set_title('Correlation: PIP3 density vs τ')
axes_corr[0].set_ylim(-0.5, 0.5)
axes_corr[0].legend()

axes_corr[1].plot(times, deltas, 'rs-', linewidth=2, markersize=8)
axes_corr[1].axhline(0, color='gray', linestyle='--')
axes_corr[1].set_xlabel('Time (s)')
axes_corr[1].set_ylabel('Δτ (LOW−HIGH PIP3) (ns)')
axes_corr[1].set_title('FRET Signal: τ shift')

axes_corr[2].plot(times, coverages, 'g^-', linewidth=2, markersize=8)
axes_corr[2].set_xlabel('Time (s)')
axes_corr[2].set_ylabel('Pixel coverage (min_counts≥5)')
axes_corr[2].set_title('Signal Coverage')
axes_corr[2].set_ylim(0, 1)

fig_corr.suptitle('FLIM↔BD Correlation Verification', fontsize=14, fontweight='bold')
fig_corr.tight_layout()
fig_corr.savefig(os.path.join(OUT, 'fig11_correlation_analysis.png'), dpi=150, bbox_inches='tight')
plt.close(fig_corr)
print("Figure 11 saved.")

# ═════════════════════════════════════════════════════════════════
# DONE
# ═════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"11 FIGURES saved to: {OUT}/")
for fn in sorted(os.listdir(OUT)):
    sz = os.path.getsize(os.path.join(OUT, fn))
    print(f"  {fn} ({sz/1024:.0f} KB)")
print(f"{'='*60}")
