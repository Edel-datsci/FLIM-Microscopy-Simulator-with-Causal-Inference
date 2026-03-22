# Manuscript Plan — Scientific Reports

## Title (max 20 words)
"Causal inference from synthetic FLIM-FRET images reveals directed information flow in PIP3-PTEN signaling"

## Target
Scientific Reports (Nature), max 4500 words main text, max 8 figures

## Story Arc
1. PROBLEM: FLIM-FRET images show WHERE signaling happens, but not WHAT CAUSES WHAT
2. GAP: No tool exists that connects molecular dynamics to FLIM observables with causal analysis
3. SOLUTION: ED-MRDT — first generative simulator: BD → FLIM → CTE
4. KEY RESULT: P6 significant (p=0.005) — biosensor binding CAUSES lifetime change
5. SIGNIFICANCE: Ground truth channel enables validation impossible with experiments alone

## Structure

### Abstract (~200 words)
- Problem: FLIM-FRET widely used but correlation ≠ causation
- What we built: generative particle simulator + virtual microscope + CTE
- Key result: P6 significant, negative control P10 not significant
- Impact: first tool bridging molecular dynamics to photonic observables with causal inference

### Introduction (~800 words)
- Para 1: FLIM-FRET is the gold standard for protein interactions in living cells
- Para 2: But images only show correlations. We need causality.
- Para 3: Transfer entropy (Schreiber 2000) measures directed information flow
- Para 4: Gap: no simulator connects BD → FLIM → TE. We fill this gap.
- Para 5: We present ED-MRDT and demonstrate it on PIP3-PTEN waves

### Results (~1500 words)
- R1: Pipeline architecture (Fig 1 — the TikZ figure)
- R2: PIP3-PTEN wave dynamics reproduce experimental patterns (Fig 2 — BD density video frames)
- R3: Virtual FLIM images match experimental characteristics (Fig 3 — intensity + FLIM)
- R4: CTE identifies causal pair P6 at p=0.005 (Fig 4 — TE results table + convergence)
- R5: Negative control P10 confirms specificity (part of Fig 4)
- R6: 10 orthogonal pairs map the causal network (Fig 5 — causal network diagram)

### Discussion (~1000 words)
- What P6 means: molecular binding CAUSES the photonic signal
- Why other pairs are not significant (yet): sampling, weak coupling
- Comparison with Matsuoka 2018 and Knoch 2014
- Limitations: bivariate CTE, 2D membrane, biosensor idealization
- Future: multivariate TE, 3D, experimental FLIM data

### Methods (~1500 words, not counted)
- M1: Brownian dynamics engine (Ermak-McCammon, 5 reaction types)
- M2: Photophysics (CTMC, PSS kernel)
- M3: Virtual microscope (PSF, TCSPC, noise)
- M4: Conditional Transfer Entropy v2
- M5: Simulation parameters (Table 1 — all from literature)
- M6: Statistical analysis (trial-shuffle, Wibral criterion)

## Figures Plan
1. Fig 1: Pipeline diagram (TikZ — already made)
2. Fig 2: BD density + Fluorescence + FLIM side-by-side (from videos)
3. Fig 3: Species dynamics time series (PIP3, PTEN, PI3K over 40s)
4. Fig 4: CTE results — 10 pairs table + convergence plot + P6 highlight
5. Fig 5: Causal network inferred from CTE (nodes = species, edges = significant pairs)

## Key References
- Schreiber (2000) PRL 85:461 — TE definition
- Matsuoka & Ueda (2018) Nat Commun 9:4481 — PIP3-PTEN mutual inhibition
- Knoch et al. (2014) — PIP3-PTEN wave model
- Frenzel & Pompe (2007) PRL 99:204101 — CMI/CTE
- Wibral et al. (2014) — Directed information measures
- Wollstadt et al. (2014) PLoS ONE — ensemble TE
- Digman et al. (2008) Biophys J — phasor FLIM
- Arai et al. (2010) PNAS — PIP3 waves in Dictyostelium

## Writing Rules
- First person plural: "we found", "we developed"
- No filler words, no inflated claims
- Technical terms kept, but plain English otherwise
- Every claim backed by data or citation
- Personality and conviction in the writing
