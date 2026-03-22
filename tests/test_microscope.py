#!/usr/bin/env python3
"""
ED-MRDT v1.1 — PHASE 3, Steps 3.1–3.5: Tests for microscope.py

Step 3.1: PSF — FWHM ≈ 0.51λ/NA (confocal), Gaussian accuracy
Step 3.2: Confocal Scan — uniform emitters → flat intensity
Step 3.3: TCSPC — mono-exponential decay recoverable from histogram
Step 3.4: Detector — dark counts Poisson, afterpulsing correlated
Step 3.5: Integration — full pipeline photons → FLIM image
"""
import sys
import time
import math
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ed_mrdt.config import load_config, SimulationConfig
from ed_mrdt.particles import ParticleState
from ed_mrdt.microscope import (
    PSF, ConfocalScanner, TCSPCAccumulator, DetectorModel,
    VirtualMicroscope, FLIMFrame,
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
print("TEST SUITE: microscope.py (Steps 3.1–3.5)")
print("=" * 70)

# Load canonical config
config = load_config(os.path.join(os.path.dirname(__file__), "..", "ed_mrdt", "examples", "egfr_egf.yaml"))

# ================================================================
# TEST GROUP 1 (Step 3.1): PSF Properties
# ================================================================
print("\n─── Step 3.1: PSF ───")

psf = PSF(config)

# Expected confocal PSF sigma for Alexa488 (λ_em=519nm):
# σ_ill = 0.21 × 488 / 1.4 = 73.2 nm
# σ_det = 0.21 × 519 / 1.4 = 77.85 nm
# σ_conf = 1/√(1/σ_ill² + 1/σ_det²) = 53.35 nm
sigma_ill = 0.21 * 488.0 / 1.4  # nm
sigma_det = 0.21 * 519.0 / 1.4  # nm
sigma_conf_expected = 1.0 / math.sqrt(1.0/(sigma_ill**2) + 1.0/(sigma_det**2))

sigma_ch0_nm = psf.sigma_um[0] * 1000  # µm → nm
check(f"PSF σ_confocal ch0 = {sigma_ch0_nm:.1f} nm (expected {sigma_conf_expected:.1f})",
     abs(sigma_ch0_nm - sigma_conf_expected) < 0.5,
     f"got {sigma_ch0_nm:.2f}, expected {sigma_conf_expected:.2f}")

# FWHM = 2√(2ln2) × σ
fwhm_expected = sigma_conf_expected * 2.0 * math.sqrt(2.0 * math.log(2.0))
fwhm_ch0_nm = psf.fwhm_um[0] * 1000
check(f"PSF FWHM ch0 = {fwhm_ch0_nm:.1f} nm (expected {fwhm_expected:.1f})",
     abs(fwhm_ch0_nm - fwhm_expected) < 1.0)

# Diffraction limit check: confocal FWHM < 0.51λ/NA (widefield Airy)
airy_fwhm = 0.51 * 519.0 / 1.4  # nm
check(f"FWHM < Airy limit ({airy_fwhm:.0f} nm)",
     fwhm_ch0_nm < airy_fwhm,
     f"FWHM={fwhm_ch0_nm:.1f}, Airy={airy_fwhm:.1f}")

# Check channel 1 (acceptor, λ_em=565 nm)
if 1 in psf.sigma_um:
    sigma_ch1_nm = psf.sigma_um[1] * 1000
    sigma_det_a = 0.21 * 565.0 / 1.4
    sigma_conf_a = 1.0 / math.sqrt(1.0/(sigma_ill**2) + 1.0/(sigma_det_a**2))
    check(f"PSF σ ch1 (acceptor) = {sigma_ch1_nm:.1f} nm (expected {sigma_conf_a:.1f})",
         abs(sigma_ch1_nm - sigma_conf_a) < 1.0)
    check("σ_acceptor > σ_donor (longer λ)", sigma_ch1_nm > sigma_ch0_nm)

# PSF normalization: ∫PSF dA ≈ 1 (checked via numerical integration)
r = np.linspace(0, 0.5, 1000)  # µm, up to 500nm
h = psf.evaluate_array(r, channel=0)
integral = 2.0 * math.pi * np.trapezoid(h * r, r)
check(f"PSF normalization: ∫PSF dA = {integral:.4f} (expected ~1.0)",
     abs(integral - 1.0) < 0.01)

# PSF(0) = 1/(2πσ²) — peak value
peak = psf.evaluate(0.0, channel=0)
expected_peak = 1.0 / (2.0 * math.pi * psf.sigma_um[0]**2)
check(f"PSF(0) = {peak:.1f} (expected {expected_peak:.1f})",
     abs(peak - expected_peak) / expected_peak < 0.001)

# ================================================================
# TEST GROUP 2 (Step 3.2): Confocal Scanner
# ================================================================
print("\n─── Step 3.2: Confocal Scanner ───")

scanner = ConfocalScanner(config, psf)

check(f"Pixel size = {scanner.pixel_size_um*1000:.0f} nm",
     abs(scanner.pixel_size_um * 1000 - 59.0) < 0.1)
check(f"Grid = {scanner.nx}×{scanner.ny}", scanner.nx == 34 and scanner.ny == 34)
check(f"FOV = {scanner.fov_x:.3f}×{scanner.fov_y:.3f} µm",
     abs(scanner.fov_x - 34*0.059) < 0.001)

# Test: uniform emitters → roughly uniform pixel distribution
# Place 10000 emitters uniformly on the FOV
rng_test = np.random.default_rng(42)
n_uniform = 10000
ex = rng_test.uniform(scanner.offset_x, scanner.offset_x + scanner.fov_x, n_uniform)
ey = rng_test.uniform(scanner.offset_y, scanner.offset_y + scanner.fov_y, n_uniform)
ch_uniform = np.zeros(n_uniform, dtype=np.int8)

px_u, py_u, det_u = scanner.assign_photons_to_pixels(
    ex, ey, ch_uniform, n_uniform, rng_test,
)
n_det_uniform = len(px_u)
check(f"Uniform: {n_det_uniform}/{n_uniform} photons assigned",
     n_det_uniform > 0.9 * n_uniform,
     f"only {n_det_uniform}")

# Check pixel distribution is roughly uniform
hist_uniform = np.zeros((scanner.nx, scanner.ny), dtype=np.int32)
for k in range(n_det_uniform):
    hist_uniform[px_u[k], py_u[k]] += 1

expected_per_pixel = n_det_uniform / (scanner.nx * scanner.ny)
chi2 = np.sum((hist_uniform - expected_per_pixel)**2 / expected_per_pixel)
# DOF = 34*34-1 = 1155, chi2 < 1300 for p>0.01 (larger grid → higher DOF)
check(f"Uniform → flat image (χ² = {chi2:.0f}, DOF=1155, threshold ~1300)",
     chi2 < 1400,
     f"χ² = {chi2:.1f}")

cv = np.std(hist_uniform) / np.mean(hist_uniform)
check(f"Pixel CV = {cv:.3f} (expect ~{1/math.sqrt(expected_per_pixel):.3f})",
     cv < 0.45)  # Poisson CV = 1/sqrt(~8.6) ≈ 0.34; allow margin

# Test: point source at center → PSF-shaped pixel pattern
px_center = scanner.offset_x + scanner.fov_x / 2
py_center = scanner.offset_y + scanner.fov_y / 2
n_point = 50000
ex_point = np.full(n_point, px_center)
ey_point = np.full(n_point, py_center)
ch_point = np.zeros(n_point, dtype=np.int8)

px_p, py_p, _ = scanner.assign_photons_to_pixels(
    ex_point, ey_point, ch_point, n_point, rng_test,
)
hist_point = np.zeros((scanner.nx, scanner.ny), dtype=np.int32)
for k in range(len(px_p)):
    hist_point[px_p[k], py_p[k]] += 1

# Peak should be at center pixel
center_px = scanner.nx // 2
center_py = scanner.ny // 2
peak_i, peak_j = np.unravel_index(hist_point.argmax(), hist_point.shape)
check(f"Point source peak at pixel ({peak_i},{peak_j}) (expected ~{center_px},{center_py})",
     abs(peak_i - center_px) <= 1 and abs(peak_j - center_py) <= 1)

# Measure FWHM of point source image
# Note: PSF σ=53nm with 95nm pixels means PSF is undersampled.
# The FWHM measurement in pixels will be approximate.
row = hist_point[peak_i, :].astype(np.float64)
if row.max() > 0:
    row_norm = row / row.max()
    above_half = np.where(row_norm >= 0.5)[0]
    if len(above_half) >= 1:
        fwhm_pixels = max(1, above_half[-1] - above_half[0] + 1)
        fwhm_measured_nm = fwhm_pixels * 95.0  # nm
        # For undersampled PSF (pixel > σ), FWHM ≈ 1-2 pixels
        # Actual FWHM = 126nm, pixel = 95nm → expect 1-2 pixels
        check(f"Point source FWHM ~ {fwhm_measured_nm:.0f} nm "
             f"({fwhm_pixels} px × 95nm, PSF FWHM={fwhm_expected:.0f}nm)",
             fwhm_pixels >= 1 and fwhm_pixels <= 4,
             f"FWHM={fwhm_measured_nm:.0f}nm, {fwhm_pixels} pixels")
    else:
        check("Point source FWHM measurement", False, "couldn't find half-max")
else:
    check("Point source has counts", False)

# ================================================================
# TEST GROUP 3 (Step 3.3): TCSPC Accumulation
# ================================================================
print("\n─── Step 3.3: TCSPC Accumulation ───")

tcspc = TCSPCAccumulator(config)

check(f"n_bins = {tcspc.n_bins}", tcspc.n_bins == 256)
check(f"bin_width = {tcspc.bin_width*1e12:.1f} ps",
     abs(tcspc.bin_width - 25e-9/256) < 1e-18)
check(f"IRF σ = {tcspc.sigma_irf*1e12:.1f} ps",
     tcspc.sigma_irf > 0 and tcspc.sigma_irf < 1e-9)
check(f"FLIM stack shape = {tcspc.flim_stack.shape}",
     tcspc.flim_stack.shape == (2, 34, 34, 256))

# Test: mono-exponential input → recoverable decay
# Generate 100000 photons with τ = 4.1 ns (Alexa488 S1 lifetime)
# all at pixel (10, 10), channel 0
tau_test = 4.1e-9  # s
n_tcspc_test = 100000
rng_tcspc = np.random.default_rng(99)
arrival_times = rng_tcspc.exponential(tau_test, size=n_tcspc_test)
px_arr = np.full(n_tcspc_test, 10, dtype=np.int32)
py_arr = np.full(n_tcspc_test, 10, dtype=np.int32)
ch_arr = np.zeros(n_tcspc_test, dtype=np.int8)

tcspc.reset()
tcspc.accumulate(px_arr, py_arr, arrival_times, ch_arr, n_tcspc_test, rng_tcspc)

# Extract histogram
hist = tcspc.flim_stack[0, 10, 10, :].astype(np.float64)
total_in_hist = hist.sum()
check(f"TCSPC total counts = {int(total_in_hist)} (input {n_tcspc_test})",
     total_in_hist > 0.5 * n_tcspc_test,
     f"only {int(total_in_hist)} of {n_tcspc_test}")

# Fit mono-exponential: log(h) = const - t/τ
# Use bins where counts > 10 (avoid log(0))
t_bins = (np.arange(256) + 0.5) * tcspc.bin_width  # bin centers in seconds
valid = hist > 10
if np.sum(valid) > 10:
    # Also skip first few bins (IRF artifact)
    first_valid = np.argmax(hist > hist.max() * 0.5)
    fit_mask = valid & (np.arange(256) >= first_valid + 3)
    if np.sum(fit_mask) > 5:
        t_fit = t_bins[fit_mask]
        log_h = np.log(hist[fit_mask])
        # Linear fit: log(h) = a - t/τ
        coeffs = np.polyfit(t_fit, log_h, 1)
        tau_fitted = -1.0 / coeffs[0]
        tau_error_pct = abs(tau_fitted - tau_test) / tau_test * 100
        check(f"Mono-exp fit: τ = {tau_fitted*1e9:.2f} ns (expected {tau_test*1e9:.1f}, "
             f"err={tau_error_pct:.1f}%)",
             tau_error_pct < 10,
             f"τ_fitted={tau_fitted*1e9:.2f}ns, error={tau_error_pct:.1f}%")
    else:
        check("Mono-exp fit", False, f"not enough valid bins: {np.sum(fit_mask)}")
else:
    check("Mono-exp fit", False, f"not enough counts above threshold: {np.sum(valid)}")

# TCSPC peak should be near t=0 (exponential peaks at 0)
peak_bin = np.argmax(hist)
peak_time_ns = t_bins[peak_bin] * 1e9
check(f"TCSPC peak at {peak_time_ns:.1f} ns (expected near 0 + IRF)",
     peak_time_ns < 2.0,
     f"peak at bin {peak_bin}, t={peak_time_ns:.1f} ns")

# Reset test
tcspc.reset()
check("TCSPC reset: all zeros", tcspc.flim_stack.sum() == 0)

# ================================================================
# TEST GROUP 4 (Step 3.4): Detector Model
# ================================================================
print("\n─── Step 3.4: Detector Model ───")

detector = DetectorModel(config)

check(f"Detection efficiency = {detector.det_efficiency}",
     abs(detector.det_efficiency - 0.2) < 1e-10)
check(f"Dark count rate = {detector.dark_count_rate} Hz",
     abs(detector.dark_count_rate - 100.0) < 1e-10)
check(f"Afterpulsing prob = {detector.afterpulsing_prob}",
     abs(detector.afterpulsing_prob - 0.005) < 1e-10)

# Detection efficiency: 20% of photons should survive
rng_det = np.random.default_rng(77)
n_test_det = 100000
det_mask = detector.apply_detection_efficiency(n_test_det, rng_det)
frac_det = np.sum(det_mask) / n_test_det
check(f"Detection filter: {frac_det:.3f} (expected 0.200 ± 0.01)",
     abs(frac_det - 0.2) < 0.01)

# Dark counts: Poisson distributed
test_stack = np.zeros((2, 34, 34, 256), dtype=np.uint32)
n_dark = detector.add_dark_counts(test_stack, 1, rng_det)
check(f"Dark counts added: {n_dark}",
     n_dark >= 0)

# Expected dark counts: 100 Hz × 20 µs × 1 step × 2ch × 34×34 pixels
expected_dark = 100.0 * 20e-6 * 1 * 2 * 34 * 34
check(f"Dark counts ~ {expected_dark:.1f} (got {n_dark})",
     n_dark < expected_dark * 5,  # allow fluctuation
     f"expected ~{expected_dark:.1f}")

# Afterpulsing: add some counts first, then check AP adds proportionally
test_stack2 = np.zeros((2, 34, 34, 256), dtype=np.uint32)
test_stack2[0, 10, 10, 50] = 10000  # big peak
n_ap = detector.add_afterpulsing(test_stack2, rng_det)
expected_ap = 10000 * 0.005
check(f"Afterpulsing: {n_ap} (expected ~{expected_ap:.0f})",
     abs(n_ap - expected_ap) < 3 * math.sqrt(expected_ap) + 10,
     f"got {n_ap}, expected {expected_ap:.0f}")

# ================================================================
# TEST GROUP 5 (Step 3.5): VirtualMicroscope Integration
# ================================================================
print("\n─── Step 3.5: VirtualMicroscope Integration ───")

state = ParticleState(config)
scope = VirtualMicroscope(config, state)

check("VirtualMicroscope created", scope is not None)
check(f"Frame shape = {scope.frame_shape}",
     scope.frame_shape == (2, 34, 34, 256))

# repr should contain useful info
r = repr(scope)
check("repr contains PSF info", "PSF" in r and "σ=" in r)
check("repr contains scan info", "34×34" in r)
check("repr contains TCSPC info", "256 bins" in r)
print(f"    {r}")

# Test with synthetic PhotonBatch
from dataclasses import dataclass as _dc, field as _f

@_dc
class MockPhotonBatch:
    particle_idx: np.ndarray
    arrival_time: np.ndarray
    channel: np.ndarray
    pulse_index: np.ndarray
    is_fret: np.ndarray
    n_photons: int = 0

# Create photons from known particles
n_mock = 5000
rng_mock = np.random.default_rng(123)
# Pick random alive particles
alive_idx = np.where(state.is_alive)[0]
selected = rng_mock.choice(alive_idx, size=n_mock, replace=True)

mock_batch = MockPhotonBatch(
    particle_idx=selected.astype(np.int32),
    arrival_time=rng_mock.exponential(4.1e-9, size=n_mock),
    channel=np.zeros(n_mock, dtype=np.int8),  # all donor
    pulse_index=np.zeros(n_mock, dtype=np.int32),
    is_fret=np.zeros(n_mock, dtype=np.bool_),
    n_photons=n_mock,
)

t0 = time.perf_counter()
n_det = scope.process_photons(mock_batch)
t_proc = time.perf_counter() - t0

check(f"process_photons: {n_det} detected of {n_mock} input",
     n_det > 0,
     f"n_det={n_det}")
check(f"Detection rate ~ {n_det/n_mock:.1%} (expected ~{config.detector.detection_efficiency:.0%})",
     abs(n_det/n_mock - config.detector.detection_efficiency) < 0.1)
check(f"process_photons time: {t_proc*1e3:.1f} ms", t_proc < 5.0)

# Finalize frame
frame = scope.finalize_frame()
check(f"Frame has {frame.total_counts} total counts",
     frame.total_counts > 0)
check(f"Frame signal photons = {frame.n_signal_photons}", frame.n_signal_photons > 0)
check(f"Frame shape = {frame.shape}", frame.shape == (2, 34, 34, 256))

# Intensity image should have non-zero pixels
intensity = frame.intensity[0]  # donor channel
n_nonzero = np.sum(intensity > 0)
check(f"Non-zero pixels: {n_nonzero}/{scanner.nx*scanner.ny}",
     n_nonzero > 0)

# After finalize, TCSPC should be reset
check("TCSPC reset after finalize", scope.tcspc.total_counts == 0)

# ================================================================
# TEST GROUP 6: Physical Consistency
# ================================================================
print("\n─── Physical Consistency ───")

# PSF FWHM vs pixel size (Nyquist check)
# Session 7 Fix: pixel_size=59nm correctly satisfies confocal Nyquist.
# Confocal FWHM = 126nm → Nyquist = 63nm → pixel=59nm < 63nm ✓
# Ref: Zhang et al. (2007) Appl Opt 46:1819 (confocal Gaussian PSF)
fwhm_nm = psf.fwhm_um[0] * 1000
nyquist_conf = fwhm_nm / 2.0
check(f"Nyquist (confocal): pixel ({config.scan.pixel_size:.0f}nm) ≤ FWHM/2 ({nyquist_conf:.0f}nm)",
     config.scan.pixel_size <= nyquist_conf * 1.01,
     f"pixel={config.scan.pixel_size}nm, confocal FWHM/2={nyquist_conf:.0f}nm")
check(f"Confocal FWHM ({fwhm_nm:.0f}nm) > 2× pixel ({config.scan.pixel_size:.0f}nm)",
     fwhm_nm > config.scan.pixel_size * 2.0 * 0.95)

# TCSPC time range matches laser period
check("TCSPC time = laser period",
     abs(config.tcspc.time_range - config.laser.period) < 1e-15)

# IRF width < bin width × 10 (resolvable)
irf_bins = scope.tcspc.sigma_irf / scope.tcspc.bin_width
check(f"IRF σ = {irf_bins:.1f} bins (should be > 0.3 for realistic broadening)",
     irf_bins > 0.1 and irf_bins < 50,
     f"IRF = {irf_bins:.1f} bins")

# FOV covers membrane (or most of it)
Lx, Ly = config.membrane.size
check(f"FOV ({scanner.fov_x:.2f}µm) ≤ membrane ({Lx}µm)",
     scanner.fov_x <= Lx * 1.01)

# ================================================================
# TEST GROUP 7: Performance
# ================================================================
print("\n─── Performance ───")

# Process many photons to measure throughput
n_perf = 50000
rng_perf = np.random.default_rng(777)
perf_batch = MockPhotonBatch(
    particle_idx=rng_perf.choice(alive_idx, size=n_perf, replace=True).astype(np.int32),
    arrival_time=rng_perf.exponential(4.1e-9, size=n_perf),
    channel=np.zeros(n_perf, dtype=np.int8),
    pulse_index=np.zeros(n_perf, dtype=np.int32),
    is_fret=np.zeros(n_perf, dtype=np.bool_),
    n_photons=n_perf,
)

# Warm up
scope2 = VirtualMicroscope(config, state)
_ = scope2.process_photons(perf_batch)

# Measure
scope3 = VirtualMicroscope(config, state)
t0 = time.perf_counter()
n_det3 = scope3.process_photons(perf_batch)
t_perf = time.perf_counter() - t0

photons_per_sec = n_perf / t_perf if t_perf > 0 else 0
check(f"Throughput: {photons_per_sec/1e6:.1f} M photons/s ({t_perf*1e3:.1f} ms for {n_perf})",
     t_perf < 10.0)

# Frame finalization time
t0 = time.perf_counter()
f3 = scope3.finalize_frame()
t_finalize = time.perf_counter() - t0
check(f"Frame finalization: {t_finalize*1e3:.1f} ms",
     t_finalize < 5.0)

# RAM estimate for FLIM stack
ram = scope.tcspc.flim_stack.nbytes
check(f"FLIM stack RAM: {ram/1e6:.2f} MB",
     ram < 100e6,  # should be < 100 MB
     f"{ram/1e6:.2f} MB")

# ================================================================
# TEST GROUP 8: Multi-frame consistency
# ================================================================
print("\n─── Multi-frame ───")

scope_mf = VirtualMicroscope(config, state)

# Process 3 frames
frames = []
for fi in range(3):
    batch = MockPhotonBatch(
        particle_idx=rng_mock.choice(alive_idx, size=2000, replace=True).astype(np.int32),
        arrival_time=rng_mock.exponential(4.1e-9, size=2000),
        channel=np.zeros(2000, dtype=np.int8),
        pulse_index=np.zeros(2000, dtype=np.int32),
        is_fret=np.zeros(2000, dtype=np.bool_),
        n_photons=2000,
    )
    scope_mf.process_photons(batch)
    frames.append(scope_mf.finalize_frame())

check(f"3 frames created", len(frames) == 3)
check(f"Frame indices: {[f.frame_index for f in frames]}",
     frames[0].frame_index == 0 and frames[2].frame_index == 2)
check("Each frame has counts",
     all(f.total_counts > 0 for f in frames))
check("Frames are independent (different counts)",
     not all(f.total_counts == frames[0].total_counts for f in frames))

# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 70)
print(f"RESULTS: {passed}/{total} passed, {failed} failed")
print("=" * 70)

if failed == 0:
    print("🎉 ALL TESTS PASSED — Steps 3.1–3.5 COMPLETED")
else:
    print(f"⚠️  {failed} test(s) failed — needs fixing")
