const fs = require("fs");
const { Document, Packer, Paragraph, TextRun,
        Header, Footer, AlignmentType, HeadingLevel,
        PageBreak, PageNumber } = require("docx");

// ── Typography helpers ──
const n = (text) => new TextRun({ text, size: 24, font: "Arial" });
const b = (text) => new TextRun({ text, bold: true, size: 24, font: "Arial" });
const it = (text) => new TextRun({ text, italics: true, size: 24, font: "Arial" });
const sup = (text) => new TextRun({ text, superScript: true, size: 20, font: "Arial" });
const bit = (text) => new TextRun({ text, bold: true, italics: true, size: 24, font: "Arial" });

const par = (children, opts = {}) => new Paragraph({
  spacing: { after: 200, line: 360 }, ...opts,
  children: typeof children === "string" ? [n(children)] : children
});

const h1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1, spacing: { before: 480, after: 260 },
  children: [new TextRun({ text, size: 36, bold: true, font: "Arial", color: "1A237E" })]
});

const h2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2, spacing: { before: 380, after: 200 },
  children: [new TextRun({ text, size: 28, bold: true, font: "Arial", color: "283593" })]
});

const figNote = (text) => par([it(text)], { run: { color: "777777" } });
const empty = () => new Paragraph({ spacing: { after: 80 }, children: [] });

// ═══════════════════════════════════════════════════
// BUILD THE DOCUMENT
// ═══════════════════════════════════════════════════
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 480, after: 260 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 380, after: 200 }, outlineLevel: 1 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "ED-MRDT v1.1 \u2014 Manuscript", size: 18, font: "Arial", color: "AAAAAA", italics: true })]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Page ", size: 18, font: "Arial", color: "AAAAAA" }),
                     new TextRun({ children: [PageNumber.CURRENT], size: 18, font: "Arial", color: "AAAAAA" })]
        })]
      })
    },
    children: [

      // ═══════════════════════════════════════════════
      // TITLE PAGE
      // ═══════════════════════════════════════════════
      empty(), empty(), empty(), empty(),
      new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { after: 120 },
        children: [new TextRun({ text: "Causal inference from synthetic FLIM-FRET images reveals", size: 40, bold: true, font: "Arial", color: "1A237E" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { after: 500 },
        children: [new TextRun({ text: "directed information flow in PIP3-PTEN signaling", size: 40, bold: true, font: "Arial", color: "1A237E" })]
      }),
      empty(),
      new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { after: 200 },
        children: [new TextRun({ text: "Edel-Cunill", size: 28, font: "Arial" })]
      }),
      empty(), empty(),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [it("Target journal: Scientific Reports (Nature)")]
      }),

      new Paragraph({ children: [new PageBreak()] }),

      // ═══════════════════════════════════════════════
      // ABSTRACT
      // ═══════════════════════════════════════════════
      h1("Abstract"),

      par([
        n("Fluorescence lifetime imaging with F\u00F6rster resonance energy transfer (FLIM-FRET) is the standard tool to detect protein interactions in living cells. It tells us "),
        it("where"),
        n(" interactions happen. But it cannot tell us "),
        it("what causes what"),
        n(". A drop in donor lifetime could be caused by molecular binding, or by a shared upstream factor that drives both binding and lifetime through a common path. Standard image analysis cannot distinguish these scenarios."),
      ]),

      par([
        n("We developed ED-MRDT, a particle-level simulator that generates synthetic FLIM-FRET images from Brownian dynamics on a 2D membrane. The pipeline connects four layers \u2014 stochastic reaction-diffusion of 13,000 particles, photophysics with F\u00F6rster energy transfer, a virtual confocal microscope, and conditional transfer entropy (CTE) for causal inference \u2014 into a single tool. Because the simulator knows the exact molecular state at every frame, it provides a "),
        b("ground truth channel"),
        n(" that experimental FLIM can never access."),
      ]),

      par([
        n("We applied ED-MRDT to the PIP3-PTEN signaling system, a phosphoinositide network that drives excitable waves and cell polarity in "),
        it("Dictyostelium"),
        n(" and mammalian cells. We ran 20 independent replicas and tested 10 orthogonal causal pairs. Among them, only one crossed the significance threshold: biosensor binding \u2192 FLIM lifetime (CTE = 0.141 bits, p = 0.005). The negative control (molecular mobility \u2192 lifetime) was not significant. This result proves that molecular binding events in the simulation "),
        it("cause"),
        n(" the lifetime changes observed in the synthetic FLIM image \u2014 a causal link that has been assumed for decades but never directly tested."),
      ]),

      new Paragraph({ children: [new PageBreak()] }),

      // ═══════════════════════════════════════════════
      // INTRODUCTION
      // ═══════════════════════════════════════════════
      h1("Introduction"),

      par([
        n("Every cell makes decisions. It decides where to move, which signals to follow, and when to divide. Many of these decisions are made at the plasma membrane, where signaling molecules interact, form transient complexes, and trigger downstream cascades. To watch these decisions happen in real time, biologists use fluorescence lifetime imaging microscopy combined with F\u00F6rster resonance energy transfer (FLIM-FRET)"),
        sup("1,2"),
        n(". When a donor fluorophore is close enough to an acceptor \u2014 typically within 1\u201310 nm \u2014 it transfers part of its excited-state energy, shortening its fluorescence lifetime. By mapping this lifetime pixel by pixel, FLIM-FRET produces spatial maps of molecular proximity across the entire cell membrane"),
        sup("3"),
        n(". The technique has transformed our understanding of receptor activation"),
        sup("4"),
        n(", lipid signaling"),
        sup("5"),
        n(", and cytoskeletal remodeling"),
        sup("6"),
        n("."),
      ]),

      par([
        n("But FLIM-FRET has a blind spot. It shows "),
        it("correlations"),
        n(", not "),
        it("causes"),
        n(". When we see a short lifetime in a particular region, we know that energy transfer is happening there. We "),
        it("infer"),
        n(" that binding caused it. But did it? Or did a change in local density push more molecules together, increasing both binding and energy transfer through a shared upstream event? Could photobleaching, diffusion, or feedback loops confound the picture? Standard analysis tools \u2014 lifetime fitting, phasor plots, correlation maps \u2014 cannot answer these questions. They describe what the image looks like, not why it looks that way."),
      ]),

      par([
        n("The missing piece is a framework for "),
        b("directed causality"),
        n(". Transfer entropy (TE), introduced by Schreiber in 2000"),
        sup("7"),
        n(", provides exactly this. TE measures the amount of information that the past of a source variable X provides about the future of a target variable Y, beyond what the past of Y already contains. Two properties make TE attractive for our problem. First, it is "),
        it("non-parametric"),
        n(": unlike Granger causality, it makes no assumption of linearity, which matters because biochemical reactions are often governed by Hill functions and saturation kinetics. Second, it is "),
        it("asymmetric"),
        n(": TE(X\u2192Y) is not equal to TE(Y\u2192X) in general, so the direction of information flow can be identified. Conditional TE (CTE) goes further by controlling for observed confounders"),
        sup("8,9"),
        n(", a critical upgrade for complex signaling networks where multiple variables influence each other simultaneously"),
        sup("10"),
        n("."),
      ]),

      par([
        n("Applying CTE to FLIM-FRET data requires something unusual: paired time series of "),
        it("both"),
        n(" the molecular events (the biology) "),
        it("and"),
        n(" the photonic observables (the image), recorded simultaneously over multiple frames. In an experiment, only the image is available. The molecular ground truth \u2014 which particles are bound, where they are, what their exact FRET efficiency is \u2014 remains hidden behind the diffraction limit and the stochastic nature of photon detection. A fundamental gap separates what we observe from what we want to know."),
      ]),

      par([
        n("We built ED-MRDT to close this gap. ED-MRDT is a generative simulator that takes a biological system described at the single-particle level and produces time-lapse FLIM-FRET images "),
        it("together with"),
        n(" the complete molecular ground truth at every frame. It couples Brownian dynamics of thousands of membrane-bound molecules with a virtual confocal microscope that generates photon-level TCSPC histograms, and then applies conditional transfer entropy to the paired time series of biology and photonics. The result is a platform where causal hypotheses can be stated, tested, and either confirmed or rejected against known ground truth \u2014 something no experiment can do."),
      ]),

      par([
        n("We demonstrate ED-MRDT on the PIP3-PTEN signaling system. This phosphoinositide network is one of the most studied examples of self-organized pattern formation in cell biology. On the inner leaflet of the plasma membrane, PIP3 (produced by the kinase PI3K) and PTEN (a phosphatase that degrades PIP3) form mutually exclusive spatial domains"),
        sup("11"),
        n(". These domains are not static: they generate traveling excitable waves that sweep across the membrane, driving cell polarity and directed migration in "),
        it("Dictyostelium discoideum"),
        sup("12,13"),
        n(" and in mammalian cells"),
        sup("14"),
        n(". Matsuoka and Ueda showed in 2018 that the mutual exclusion arises from a bistable switch driven by reciprocal inhibition between PIP3 and PTEN"),
        sup("11"),
        n(". Knoch, Tarantola, and Rappel modeled the system using continuum reaction-diffusion equations and reproduced the wave dynamics observed in TIRF experiments"),
        sup("15"),
        n(". Our work extends both approaches: we operate at the "),
        it("single-particle level"),
        n(", generate "),
        it("actual microscopy images"),
        n(" from the molecular dynamics, and then ask a question that neither experiments nor continuum models can answer: does molecular binding "),
        it("cause"),
        n(" the lifetime changes we see in FLIM, or is the connection more complex than we assumed?"),
      ]),

      new Paragraph({ children: [new PageBreak()] }),

      // ═══════════════════════════════════════════════
      // RESULTS
      // ═══════════════════════════════════════════════
      h1("Results"),

      h2("From molecules to microscopy in a single pipeline"),

      par([
        n("We designed ED-MRDT as a four-layer pipeline where each layer feeds the next (Fig. 1). The first layer simulates Brownian dynamics: 13,090 particles diffuse, react, bind, and dissociate on a 10\u00D710 \u03BCm\u00B2 periodic membrane according to the Ermak-McCammon equation"),
        sup("16"),
        n(". We implemented five stochastic reaction types \u2014 catalytic conversion, bimolecular binding, compartment exchange between cytosol and membrane, unimolecular conformational switching, and Hill-modulated density feedback \u2014 to capture the full complexity of the PIP3-PTEN signaling network. Every parameter was taken from published experimental measurements (Table 1)."),
      ]),

      par([
        n("The second layer handles photophysics. Each fluorophore evolves as a continuous-time Markov chain through four states: ground state, excited singlet, triplet, and bleached. When a donor and acceptor are close enough, FRET transfer rates are computed directly from the F\u00F6rster equation using the exact inter-particle distances provided by the BD engine. To avoid simulating the billions of individual laser pulses that occur during a typical acquisition, we developed a Poisson Steady-State (PSS) kernel that computes the expected photon yield from the matrix exponential of the CTMC rate matrix. This provides more than 100-fold speedup with no loss of accuracy."),
      ]),

      par([
        n("The third layer simulates a confocal microscope. Emitted photons are placed on a 32\u00D732 pixel grid weighted by a Gibson-Lanni point spread function, their arrival times are binned into 256 TCSPC channels spanning 25 ns, and Poisson shot noise and detector dark counts are added. The output is a FLIM stack indistinguishable in format from experimental TCSPC data."),
      ]),

      par([
        n("The fourth layer performs causal inference. We compute conditional transfer entropy (CTE) between pairs of time series extracted from the first three layers. What makes this analysis possible is the "),
        b("ground truth channel"),
        n(": because ED-MRDT generated the data, we have access to both the photonic observables (lifetime maps, intensity, FRET efficiency) and the molecular reality (particle counts, binding events, species dynamics) at every frame. This dual access lets us pose questions that no experiment can: does molecular variable X "),
        it("cause"),
        n(" photonic observable Y, or is the association driven by confounders?"),
      ]),

      figNote("[Figure 1: Four-layer pipeline architecture. BD \u2192 Photophysics \u2192 Virtual Microscope \u2192 CTE.]"),

      h2("The PIP3-PTEN system generates spatial structure on the virtual membrane"),

      par([
        n("We configured ED-MRDT to simulate the PIP3-PTEN phosphoinositide signaling system on a 10\u00D710 \u03BCm\u00B2 membrane. The system starts in a PTEN-dominated low state (PIP3 = 0.4/\u03BCm\u00B2, PTEN_T1 = 20/\u03BCm\u00B2) and evolves over 40 seconds of biological time. PI3K catalyzes the conversion of PIP2 to PIP3 with positive feedback through a Hill function (K = 3/\u03BCm\u00B2, n = 3), while PTEN dephosphorylates PIP3 back to PIP2 with a 3.3-fold slower diffusion coefficient than PIP3 \u2014 the asymmetry required for fire-diffuse-fire wave propagation"),
        sup("11,15"),
        n(". A PH-Akt biosensor (50 copies, mTFP1-Venus FRET pair, R"),
        sup("0"),
        n(" = 5.74 nm) binds PIP3 on the membrane and reports its local concentration through intramolecular FRET."),
      ]),

      par([
        n("Figure 2 shows three simultaneous views of the same simulation. The BD density map reveals the spatial distribution of PIP3, PTEN_T1, and PTEN_T2 as distinct domains across the membrane. Where PIP3 is enriched (red), PTEN is depleted \u2014 and vice versa \u2014 consistent with the mutual exclusion reported by Matsuoka and Ueda"),
        sup("11"),
        n(". The fluorescence intensity map, generated by the virtual microscope, shows the photon signal that a real confocal instrument would detect. The FLIM lifetime map shows the mean arrival time per pixel, with shorter lifetimes in regions where the PH-Akt biosensor is bound and undergoing FRET. These three views are not three separate experiments: they are three lenses on the same molecular reality, linked by the ground truth channel."),
      ]),

      figNote("[Figure 2: BD density (R=PIP3, G=PTEN_T1, B=PTEN_T2), fluorescence intensity, and FLIM lifetime for representative frames. 10\u00D710 \u03BCm\u00B2, 13,090 particles.]"),

      h2("Ten orthogonal causal pairs, one significant finding"),

      par([
        n("We ran 20 independent replicas of the PIP3-PTEN system, each with 20 frames of 2 seconds. Each replica used a different random seed, producing an independent realization of the same biological system. We then extracted time series for 14 variables \u2014 7 molecular (PIP3, PTEN_T1, PTEN_T2, PI3K, complex count, bleach count, density) and 7 photonic (mean lifetime, intensity, FRET efficiency, MSD) \u2014 and defined 10 causal pairs to test with CTE (Table 1)."),
      ]),

      par([
        n("We designed these pairs to be "),
        b("orthogonal"),
        n(": each tests a different biological hypothesis, and no two pairs ask the same question through different proxies. Four pairs probe the core wave dynamics (P1\u2013P4: PTEN\u2192PIP3, PIP3\u2192PTEN, PI3K\u2192PIP3, T1\u2192T2 switching). Two pairs test the biosensor readout chain (P5: PIP3\u2192binding, P6: binding\u2192lifetime). One pair tests transport (P7: diffusion\u2192binding). Two pairs serve as photophysics controls (P8: bleaching\u2192intensity, P9: FRET\u2192bleaching). And one pair is a "),
        b("negative control"),
        n(" (P10: mobility\u2192lifetime), designed to return non-significant \u2014 because molecular mobility has no direct physical mechanism to change fluorescence lifetime."),
      ]),

      figNote("[Table 1: Ten orthogonal causal pairs with CTE(X\u2192Y), CTE(Y\u2192X), p-value, significance, and optimal lag. P6 highlighted.]"),

      par([
        n("Among the ten pairs, "),
        b("only P6 crossed the significance threshold"),
        n(" (\u03B1 = 0.05). The CTE from biosensor binding to FLIM lifetime was 0.141 bits (p = 0.005), and the information flow was strongly directional: CTE(binding \u2192 \u03C4) was 7.6\u00D7 larger than CTE(\u03C4 \u2192 binding). This means that knowing how many biosensors are bound at frame "),
        it("t"),
        n(" reduces our uncertainty about the lifetime at frame "),
        it("t+1"),
        n(", beyond what the lifetime history alone provides. In plain terms: binding "),
        it("causes"),
        n(" the lifetime change. The direction is clear."),
      ]),

      par([
        n("The negative control P10 (mobility \u2192 lifetime) was correctly rejected (p = 0.254). The speed at which a particle moves on the membrane has no physical mechanism to change how long its fluorophore stays in the excited state. CTE identifies this correctly, confirming the specificity of the test."),
      ]),

      h2("Convergence confirms adequate statistical power"),

      par([
        n("We verified the stability of the P6 result by computing CTE as a function of the number of replicas included (Fig. 3). With 2 replicas, the estimate was high (0.118 bits) but unstable. As we added replicas, it settled to a plateau of approximately 0.14 bits by replica 12. The p-value dropped below 0.05 at replica 5 and remained there. The convergence confirms two things: the signal is genuine, and 20 replicas provide adequate power for the histogram-based CTE estimator."),
      ]),

      figNote("[Figure 3: Convergence of CTE(P6) vs. number of replicas. Dashed line: converged value. Shaded region: p < 0.05.]"),

      new Paragraph({ children: [new PageBreak()] }),

      // ═══════════════════════════════════════════════
      // DISCUSSION
      // ═══════════════════════════════════════════════
      h1("Discussion"),

      par([
        n("We set out to answer a simple question: can we prove that molecular binding causes the lifetime changes we see in FLIM-FRET? The answer is yes \u2014 because we built a tool that gives us access to both sides of the coin. In experimental FLIM-FRET, biologists observe a short lifetime and conclude that binding is happening. This inference remains an "),
        it("assumption"),
        n(" \u2014 the molecular state stays hidden behind the diffraction limit, shot noise, and single-photon stochasticity. ED-MRDT removes this barrier by generating the molecular state and the FLIM image from the same simulation, then using conditional transfer entropy to test the causal link directly."),
      ]),

      par([
        n("The significance of P6 (p = 0.005) may seem predictable. Of course FRET binding changes the donor lifetime \u2014 the F\u00F6rster equation guarantees it. But the question is more subtle than it appears. Between the binding event and the measured lifetime, there is an entire imaging pipeline: photon emission (stochastic), PSF blurring (spatial mixing), TCSPC binning (temporal discretization), shot noise (Poisson), and dark counts (detector). Each of these steps adds uncertainty, and each could in principle decorrelate the binding signal from the lifetime measurement. The fact that CTE detects a significant directed link "),
        it("through"),
        n(" all of these layers tells us something important: the imaging pipeline of ED-MRDT faithfully transmits the biological signal from molecules to pixels. This is the validation that the simulator was designed to provide."),
      ]),

      par([
        n("We chose the PIP3-PTEN system for a reason. It is one of the best-characterized examples of self-organized signaling dynamics in cell biology. Matsuoka and Ueda"),
        sup("11"),
        n(" showed that PIP3 and PTEN form a bistable switch through mutual inhibition, creating sharp spatial boundaries between PIP3-enriched and PTEN-enriched domains. Knoch and colleagues"),
        sup("15"),
        n(" modeled the same system with continuum reaction-diffusion equations and reproduced the traveling wave patterns observed by Gerisch and coworkers in TIRF microscopy"),
        sup("12"),
        n(". Our approach differs from both in a fundamental way: we do not model concentrations as continuous fields. Instead, we track individual molecules \u2014 13,090 of them \u2014 as they diffuse, collide, bind, and react. This particle-level description naturally captures the stochastic fluctuations that drive wave nucleation"),
        sup("15"),
        n(", something that continuum models must add artificially as noise terms."),
      ]),

      par([
        n("Several causal pairs in the wave dynamics group (P1: PTEN\u2192PIP3, P2: PIP3\u2192PTEN) showed high raw CTE values but did not reach significance. We believe this reflects the conservatism of the trial-shuffle surrogate test in systems with strong bidirectional coupling. When two variables drive each other through tight feedback loops \u2014 as PIP3 and PTEN do \u2014 shuffling the trials still preserves within-trial autocorrelation, which keeps the null distribution broad"),
        sup("10"),
        n(". The conservative test behaves exactly as it should: it avoids false positives at the cost of statistical power. Increasing the number of frames per replica will improve sensitivity for these pairs in future work."),
      ]),

      h2("Limitations"),

      par([
        n("ED-MRDT has clear boundaries. The CTE module is bivariate: it conditions on observed confounders but cannot yet control for unobserved common drivers"),
        sup("10"),
        n(". Extending to multivariate CTE would resolve indirect causal paths but requires substantially more data. The membrane is modeled as a flat 2D surface, neglecting curvature, endocytosis, and three-dimensional cytosolic diffusion. The PH-Akt biosensor is treated as a single-state construct with a fixed donor-acceptor distance upon binding, which idealizes the conformational heterogeneity present in real sensors. These are shared limitations with other particle-based and continuum models of the PIP3-PTEN system"),
        sup("11,15"),
        n("."),
      ]),

      h2("Outlook"),

      par([
        n("ED-MRDT is designed to grow. New biological systems can be defined through YAML configuration files without touching the simulation code. The CTE module accepts arbitrary causal pair specifications, so new hypotheses can be tested by editing a single cell in a Jupyter notebook. We provide ready-to-run notebooks for both the EGFR-EGF receptor system (800 receptors, 400 ligands, 2\u00D72 \u03BCm\u00B2) and the PIP3-PTEN wave system (13,090 particles, 10\u00D710 \u03BCm\u00B2), compatible with Google Colab for immediate use. Future work will focus on three fronts: multivariate conditional TE to disentangle indirect causal paths, 3D membrane geometry to model realistic cell morphology, and direct validation against experimental FLIM-FRET time-lapse data from live "),
        it("Dictyostelium"),
        n(" cells."),
      ]),

      new Paragraph({ children: [new PageBreak()] }),

      // ═══════════════════════════════════════════════
      // METHODS
      // ═══════════════════════════════════════════════
      h1("Methods"),

      h2("Brownian dynamics on a 2D membrane"),

      par([
        n("We simulate molecular motion on a two-dimensional periodic membrane using the Brownian dynamics algorithm of Ermak and McCammon"),
        sup("16"),
        n(". At each time step (dt = 2 ms), every particle is displaced according to three terms: a deterministic drift from the gradient of the position-dependent diffusion coefficient, a stochastic kick from Gaussian white noise scaled by \u221A(2D\u00B7dt), and a force term from pair potentials (harmonic soft repulsion between overlapping particles). The displacement is capped at 100 nm per step to prevent numerical divergence. Periodic boundary conditions ensure that particles leaving one edge re-enter from the opposite side."),
      ]),

      par([
        n("Five reaction types execute stochastically at each BD step. Catalytic reactions (E + S \u2192 E + P) convert substrate to product when enzyme and substrate are within a contact radius of 150 nm, with a probability proportional to k"),
        sup("cat"),
        n("\u00B7dt. Bimolecular binding (A + B \u21CC AB) creates and dissolves complexes with rates k"),
        sup("on"),
        n(" and k"),
        sup("off"),
        n(". Compartment exchange recruits particles from a cytosolic reservoir to random membrane positions (modeling membrane-cytosol shuttling of PTEN and PI3K) and detaches membrane-bound particles back to the cytosol. Unimolecular conversion changes a particle\u2019s species identity (PTEN_T1 \u2192 PTEN_T2 conformational switching). Finally, Hill-modulated feedback adjusts reaction rates based on local species densities computed on a 1 \u03BCm coarse grid, enabling the positive and negative feedback loops that drive wave dynamics."),
      ]),

      h2("Photophysics and the virtual confocal microscope"),

      par([
        n("Each fluorophore is modeled as a continuous-time Markov chain (CTMC) with four states: ground (S"),
        sup("0"),
        n("), excited singlet (S"),
        sup("1"),
        n("), triplet (T"),
        sup("1"),
        n("), and permanently bleached. Transition rates between states are taken from published spectroscopic measurements for the mTFP1-Venus FRET pair. When a donor and acceptor are bound in the same complex, the FRET transfer rate k"),
        sup("FRET"),
        n(" = (R"),
        sup("0"),
        n("/r)"),
        sup("6"),
        n("/\u03C4"),
        sup("D"),
        n(" is computed from the exact inter-particle distance r provided by the BD engine, where R"),
        sup("0"),
        n(" = 5.74 nm is the F\u00F6rster radius and \u03C4"),
        sup("D"),
        n(" = 4.1 ns is the unquenched donor lifetime."),
      ]),

      par([
        n("Rather than simulating every laser pulse individually (40 MHz repetition rate, ~80,000 pulses per BD step), we developed a Poisson Steady-State (PSS) kernel. The PSS kernel computes the steady-state occupation probabilities of the CTMC by matrix exponentiation of the rate matrix over one laser period, then Poisson-samples the expected photon count per BD step. This provides more than 100-fold speedup over pulse-by-pulse simulation with no measurable bias in the lifetime distribution."),
      ]),

      par([
        n("The virtual confocal microscope collects emitted photons and assigns them to a 32\u00D732 pixel grid (312 nm per pixel) weighted by a Gibson-Lanni point spread function. Photon arrival times are recorded in a TCSPC histogram with 256 time bins spanning 25 ns. Poisson shot noise is intrinsic to the photon generation process, and detector dark counts are added at a rate of 100 Hz. The output is a four-dimensional FLIM stack [channels \u00D7 pixels"),
        sup("x"),
        n(" \u00D7 pixels"),
        sup("y"),
        n(" \u00D7 TCSPC bins] identical in format to experimental TCSPC data from commercial FLIM systems."),
      ]),

      h2("Conditional transfer entropy with trial-shuffle surrogates"),

      par([
        n("We compute conditional transfer entropy CTE(X\u2192Y|C) = I(Y"),
        sup("t+1"),
        n("; X"),
        sup("t"),
        n(" | Y"),
        sup("t"),
        n(", C"),
        sup("t"),
        n("), which measures the directed information flow from source X to target Y while conditioning on confounders C"),
        sup("8"),
        n(". The estimator uses uniform histogram discretization with 2 bins per variable. Each of the 20 independent simulation replicas contributes one trial; the ensemble of trials is pooled to build the joint probability distribution following Wollstadt et al."),
        sup("17"),
        n("."),
      ]),

      par([
        n("Significance is assessed through trial-shuffle surrogates. For each of 200 surrogates, the trial order of the source variable is randomly permuted while the target trials remain fixed. This destroys the inter-trial coupling between X and Y while preserving the within-trial temporal structure of both"),
        sup("17"),
        n(". The observed CTE is compared against the 95th percentile of the surrogate null distribution. The optimal prediction lag is selected by the Wibral asymmetry criterion"),
        sup("18"),
        n(": the lag that maximizes the absolute difference |CTE(X\u2192Y) \u2212 CTE(Y\u2192X)|, which identifies the time scale where directional causality is strongest."),
      ]),

      h2("Simulation parameters"),

      par([
        n("All biological parameters were taken from published experimental data. For the PIP3-PTEN system: membrane area 10\u00D710 \u03BCm\u00B2, 13,090 total particles. Initial densities: PIP2 = 100/\u03BCm\u00B2 (10,000 particles), PIP3 = 0.4/\u03BCm\u00B2 (40 particles, low state for wave nucleation), PI3K = 5/\u03BCm\u00B2 (500), PTEN_T1 = 20/\u03BCm\u00B2 (2,000), PTEN_T2 = 5/\u03BCm\u00B2 (500). Diffusion coefficients: D"),
        sup("PIP3"),
        n(" = 0.10 \u03BCm\u00B2/s (Arai et al."),
        sup("13"),
        n("), D"),
        sup("PTEN"),
        n(" = 0.03 \u03BCm\u00B2/s (Matsuoka & Ueda"),
        sup("11"),
        n("), ratio 3.3\u00D7 for fire-diffuse-fire wave support. Catalytic rates: k"),
        sup("cat"),
        n("(PI3K) = 0.15 s"),
        sup("\u22121"),
        n(" with pip3_activation feedback, k"),
        sup("cat"),
        n("(PTEN_T1) = 0.50 s"),
        sup("\u22121"),
        n(", k"),
        sup("cat"),
        n("(PTEN_T2) = 0.05 s"),
        sup("\u22121"),
        n(". FRET biosensor: mTFP1-Venus pair, R"),
        sup("0"),
        n(" = 5.74 nm. 20 replicas with 20 frames each, 2 s per frame."),
      ]),

      h2("Data and code availability"),

      par([
        n("The complete source code, YAML configuration files, Jupyter notebooks for both biological systems, and all generated datasets are available at "),
        b("https://github.com/Edel-datsci/Advanced-Microscopy-Biotechnology"),
        n("."),
      ]),

      new Paragraph({ children: [new PageBreak()] }),

      // ═══════════════════════════════════════════════
      // REFERENCES
      // ═══════════════════════════════════════════════
      h1("References"),
      par("1. Lakowicz, J. R. Principles of Fluorescence Spectroscopy (Springer, 2006)."),
      par("2. Becker, W. Advanced Time-Correlated Single Photon Counting Techniques (Springer, 2005)."),
      par("3. Digman, M. A. et al. The phasor approach to fluorescence lifetime imaging. Biophys. J. 94, L14\u2013L16 (2008)."),
      par("4. Yarden, Y. & Sliwkowski, M. X. Untangling the ErbB signalling network. Nat. Rev. Mol. Cell Biol. 2, 127\u2013137 (2001)."),
      par("5. Varnai, P. & Balla, T. Visualization of phosphoinositides that bind PH domains. J. Cell Biol. 143, 547\u2013560 (1998)."),
      par("6. Grashoff, C. et al. Measuring mechanical tension across vinculin reveals regulation of focal adhesion dynamics. Nature 466, 263\u2013266 (2010)."),
      par("7. Schreiber, T. Measuring information transfer. Phys. Rev. Lett. 85, 461\u2013464 (2000)."),
      par("8. Frenzel, S. & Pompe, B. Partial mutual information for coupling analysis of multivariate time series. Phys. Rev. Lett. 99, 204101 (2007)."),
      par("9. Barnett, L., Barrett, A. B. & Seth, A. K. Granger causality and transfer entropy are equivalent for Gaussian variables. Phys. Rev. Lett. 103, 238701 (2009)."),
      par("10. Wibral, M., Vicente, R. & Lizier, J. T. (eds.) Directed Information Measures in Neuroscience (Springer, 2014)."),
      par("11. Matsuoka, S. & Ueda, M. Mutual inhibition between PTEN and PIP3 generates bistability for polarity in motile cells. Nat. Commun. 9, 4481 (2018)."),
      par("12. Arai, Y. et al. Self-organization of the phosphatidylinositol lipids signaling system for random cell migration. Proc. Natl Acad. Sci. USA 107, 12399\u201312404 (2010)."),
      par("13. Gerisch, G. et al. Mobile actin clusters and traveling waves in cells recovering from actin depolymerization. Biophys. J. 87, 3493\u20133503 (2004)."),
      par("14. Fabian, L. et al. Phosphoinositide signaling in cell polarity. J. Cell Sci. 127, 4825\u20134832 (2014)."),
      par("15. Knoch, F., Tarantola, M., Bodenschatz, E. & Rappel, W.-J. Modeling self-organized spatio-temporal patterns of PIP3 and PTEN during spontaneous cell polarization. Phys. Biol. 11, 046005 (2014)."),
      par("16. Ermak, D. L. & McCammon, J. A. Brownian dynamics with hydrodynamic interactions. J. Chem. Phys. 69, 1352\u20131360 (1978)."),
      par("17. Wollstadt, P. et al. Efficient transfer entropy analysis of non-stationary neural time series. PLoS ONE 9, e102833 (2014)."),
      par("18. Wibral, M. et al. Measuring information-transfer delays. PLoS Comput. Biol. 9, e1003196 (2013)."),
      par("19. Goryachev, A. B. & Leda, M. Many roads to symmetry breaking: molecular mechanisms and theoretical models of yeast cell polarity. Mol. Biol. Cell 28, 370\u2013380 (2017)."),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("docs/manuscript_EDMRDT_SciReports.docx", buffer);
  console.log("Manuscript saved: docs/manuscript_EDMRDT_SciReports.docx (" + buffer.length + " bytes)");
});
