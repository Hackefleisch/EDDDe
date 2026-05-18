# Project Plan: Electron-Density Molecular Similarity Benchmark

## 1. Project Context

### 1.1 Goal
Empirically evaluate whether a novel electron-density-based molecular embedding (hereafter **MUT** — "method under test") produces distances that correlate with molecular functional similarity better than established baselines.

### 1.2 Core Hypothesis
Distances between MUT embeddings (coefficients for basis functions fit to electron density) reflect functional similarity better than topological fingerprints with Tanimoto similarity, because electron densities encode more QM-relevant information than connectivity alone.

### 1.3 What "Better" Means Operationally
Performance across six experiments that together span: embedding smoothness on chemical series, sensitivity to electronic effects, activity-based retrieval, activity-cliff sensitivity, bioisostere recognition, scaffold hopping.

### 1.4 Non-Goals
- Developing or training the MUT itself (assumed to exist and produce vectors).
- Property prediction tasks (future work).
- Prospective virtual screening.
- Runtime optimization of any method.

---

## 2. Glossary

| Term | Definition |
|------|------------|
| **MUT** | Method Under Test: the electron-density-based embedding + its distance function. Treated as a black-box `embed(SMILES) -> vector` and `distance(v1, v2) -> float`. |
| **Baseline** | Any non-MUT similarity/distance method listed in §3.1. |
| **Similarity / Distance** | For methods producing similarities in [0,1] (fingerprints with Tanimoto), distance is `1 - similarity`. For vector embeddings, distance is specified per method. All metrics framed as: smaller distance = more similar. |
| **Query** | A molecule used as reference; other molecules are ranked by ascending distance to the query. |
| **Active / Decoy / Inactive** | Active = confirmed binding/activity against target. Decoy = property-matched computationally generated non-binder. Inactive = experimentally verified non-binder. |
| **MMP** | Matched Molecular Pair: two molecules differing only at a single site. |
| **Activity Cliff** | An MMP with large potency difference (|ΔpKi| ≥ 2, i.e. ≥100×). |
| **Bioisostere** | A structural replacement that preserves biological activity despite changed connectivity. |
| **Scaffold (Bemis-Murcko)** | The ring system and linkers of a molecule with substituents removed. |

---

## 3. Methods

### 3.1 Baselines

All baselines must be computed on the same molecule sets as MUT. When applicable, use RDKit (≥2024.03) as the canonical implementation.

| ID | Method | Category | Vector / Similarity | Reference |
|----|--------|----------|---------------------|-----------|
| B1 | ECFP4 (Morgan r=2, 2048 bits) | Topological FP | bitvector / Tanimoto | [Rogers & Hahn 2010](https://doi.org/10.1021/ci100050t) |
| B2 | ECFP6 (Morgan r=3, 2048 bits) | Topological FP | bitvector / Tanimoto | Same as B1 |
| B3 | FCFP4 (feature-based Morgan r=2) | Topological FP | bitvector / Tanimoto | Same as B1, `useFeatures=True` |
| B4 | MACCS keys (166 bits) | Topological FP | bitvector / Tanimoto | RDKit `MACCSkeys` |
| B5 | Atom Pair fingerprint | Topological FP | bitvector / Tanimoto | Carhart et al. 1985 |
| B6 | Topological Torsion fingerprint | Topological FP | bitvector / Tanimoto | Nilakantan et al. 1987 |
| B7 | RDKit 2D descriptors (~200 descriptors) | Physicochemical | vector / cosine, Euclidean | RDKit `Descriptors` |
| B8 | Gaussian shape + color Tanimoto (ROCS-equivalent) | 3D shape+field | combo Tanimoto from overlay | RDKit `rdShapeAlign` (PAPER algorithm, [Grant et al. 1996](https://doi.org/10.1021/jp951631s); BSD-licensed open-source port of the OpenEye ROCS approach) |
| B9 | USR (12-d) | 3D shape | vector / inverse-Manhattan | Ballester & Richards 2007 |
| B10 | USRCAT (60-d, pharmacophoric) | 3D shape+pharmacophore | vector / inverse-Manhattan | [Schreyer & Blundell 2012](https://doi.org/10.1186/1758-2946-4-27) |
| B11 | eSim (electrostatic field + shape) — ships as two variants `B11-shape` (shape-driven pose search via `rdShapeAlign.AlignMol`) and `B11-o3a` (atom-correspondence alignment via `rdMolAlign.GetCrippenO3A`) | 3D field | combo from overlay: `1 - 0.5 * (espsim.GetShapeSim + espsim.GetEspSim)` with MMFF charges (renormalised Carbo) | Cites [Jain 2020](https://doi.org/10.1007/s10822-019-00236-6) (proprietary). Implementation uses `espsim` ([Bolcato et al. JCIM 2022](https://doi.org/10.1021/acs.jcim.1c01535), MIT, [hesther/espsim](https://github.com/hesther/espsim)) — same OSS-substitute pattern as B8. B11-o3a uses **Crippen-O3A** (LogP/MR atomic contributions) rather than MMFF-O3A; the choice is load-bearing on two dimensions: (1) **speed** — Crippen is ~1.4× faster on small-molecule series (EXP-1/2) and ~1.8× faster on drug-like libraries (EXP-3a/3b), because MMFF-O3A's correspondence search scales poorly with atom count and pre-caching MMFF properties saves nothing (the bottleneck is inside the C++ optimiser); (2) **EXP-2 conjugation-specificity** — switching MMFF→Crippen widens the S6−S7 silhouette gap from −0.079 (anti-specific: S6 0.292, S7 0.371) to +0.206 (properly specific: S6 0.213, S7 0.007), making B11-o3a the cleanest control-aware result in the panel; this is genuine signal, not just a numerical shift. Identical to MMFF on substituent-conserved series (e.g. Hammett benzoic acids, Δ ≤ 0.001), Spearman 0.89 on diverse drug-like pairs (retrieval ranks may shift a place or two). Both variants are per-pair (asymmetric); experiments that build self-matrices must symmetrise. B11-o3a's atom-correspondence search can pick a symmetry-mapped mapping for symmetric molecules, producing a non-zero self-distance (up to ~0.4 for monosubstituted benzene/cyclohexane analogues under Crippen typing); this affects symmetrised self-matrices only at the diagonal (zeroed before MDS in EXP-2) and is intrinsic to atom-correspondence aligners. |
| B12 | Mol2vec (300-d) | Learned | vector / cosine | [Jaeger et al. 2018](https://doi.org/10.1021/acs.jcim.7b00616), [repo](https://github.com/samoturk/mol2vec) |
| B13 | Uni-Mol embedding | Learned 3D | vector / cosine | [Zhou et al. ICLR 2023](https://openreview.net/forum?id=6K2RM6wVqKu), [repo](https://github.com/deepmodeling/Uni-Mol) |
| B14 | Chemprop D-MPNN hidden state | Learned 2D | vector / cosine | [Yang et al. 2019](https://doi.org/10.1021/acs.jcim.9b00237), [repo](https://github.com/chemprop/chemprop) |
| B15 | Coulomb Matrix | QM-inspired | matrix / Frobenius or kernel | Rupp et al. 2012 |
| B16 | SOAP | QM-inspired | vector / kernel | Bartók et al. 2013 (use DScribe lib) |
| B17 | ACSF (Behler-Parrinello) | QM-inspired | vector / Euclidean | Behler 2011 (use DScribe lib) |
| B18 | BCL::Mol2D atom-environment descriptor (574-d count vector, atom-type encoding, height=1 — published defaults of BCL's `UMol2D` invocation) | Physicochemical / 2D | vector / cosine | [Vu et al. JCAMD 2019](https://doi.org/10.1007/s10822-019-00199-8). Implementation invokes BCL's `bcl.exe molecule:Properties -tabulate UMol2D` per dataset via subprocess (one batched call, no per-mol overhead). BCL is a closed-source C++ binary whose [license](https://github.com/BCLCommons/bcl/blob/master/LICENSE) forbids redistribution — the path is read from `BCL_BIN` in `eddde/local_settings.py` (gitignored; copy from [`local_settings.example.py`](eddde/local_settings.example.py)). When unset, B18 silently does not register; every other method is unaffected. See [CLAUDE.md](CLAUDE.md) §Environment for the install + config sequence. **TODO cite (Spinnaker)**: in-preparation paper, leave as placeholder until publication. |

**Baseline groupings for analysis:** Topological (B1–B6), Physicochemical (B7, B18), 3D-structural (B8–B11), Learned (B12–B14), QM-descriptor (B15–B17).

**Rationale for QM-descriptor baselines:** Critical for isolating whether any MUT advantage stems from electron density specifically, vs. QM information in general.

**Conformer handling:** All 3D methods (B8–B11, B13, B15–B17, and MUT) use a single shared conformer per molecule: the lowest MMFF94 energy geometry found across 5 ETKDGv3-sampled starting conformers (`pruneRmsThresh=0.5`). Each molecule's pkl entry contains exactly one conformer. Potential follow-up: add a best-of-ensemble variant using a larger conformer set to test whether flexibility adds signal.

### 3.2 Methods Under Test (MUTs)

MUTs condense the `(n_atoms, 127)` ElektroNN coefficient matrix into a fixed-size embedding paired with a distance function. All variants share the same upstream stage (`Stage.ELEKTRONN_COEFFS`) and are registered in `eddde/methods/muts/`. Each variant gets its own stable `id` so benchmark outputs can distinguish condensing schemes.

| ID | Condensing scheme | Distance | Status | Rationale |
|----|-------------------|----------|--------|-----------|
| MUT-mean | Mean over atoms → `(127,)` | Euclidean (L2) | implemented | Simplest sensible baseline. Coefficients are homogeneous (same basis), so L2 respects magnitude as real chemical signal; mean over atoms already normalises for molecule size. |
| MUT-mean-cosine | Mean over atoms → `(127,)` | Cosine | planned | A/B against MUT-mean to test whether magnitude carries signal or washes out. If cosine matches or beats L2, the direction of the coefficient vector is what matters, not its length. |
| MUT-mean-irrep-weighted | Mean over atoms → `(127,)`, then weighted L2 with per-irrep weights | Weighted Euclidean | planned | The 127 dims are **not** equivalent: they split as 14 scalars (0e) + 42 vectors (14×3, 1o) + 25 d (5×5, 2e) + 28 f (4×7, 3o) + 18 g (2×9, 4e). Plain L2 treats every dim equally; per-irrep weights (uniform, equal-per-channel, or tuned on a validation split) test whether certain angular-momentum channels carry disproportionate similarity signal. |
| MUT-mean-mahalanobis | Mean over atoms → `(127,)`, then Mahalanobis with inverse covariance from training-set statistics | Mahalanobis | planned | Accounts for dimension-wise scale differences and between-dimension correlations in the pooled coefficient space. The covariance is a learned object — fit once on a training split, reused at inference. More principled than fixed per-irrep weights when correlations matter. |

**Training-data fairness constraint:** Any learned component — whether on the embedding side (attention weights, pooled GNN parameters) or the distance side (Mahalanobis covariance, tuned irrep weights) — must be fit only on a held-out training split. It must **never** see the evaluation datasets (D3–D9). This keeps the comparison with baselines fair and rules out leakage as an explanation for MUT performance. Trainable MUTs are explicitly allowed as the variant family matures; early variants (MUT-mean and the three listed above) keep the embedding non-trainable on purpose, so we can first characterise the raw signal in the coefficients before adding representation learning.

**Future directions (not yet scoped):** attention-pooled condensing (learned per-atom weights), graph-pooled condensing (aggregate using `adjacencies` / `distances`, possibly via a small GNN), per-atom-type pooled embedding (separate mean per element, concatenated). These are candidates for trainable-embedding MUTs once the non-trainable baseline is established.

---

## 4. Datasets

| ID | Dataset | Purpose | Source |
|----|---------|---------|--------|
| D1 | Custom homologous series | EXP-1 | Generated from SMILES; see §5.1 |
| D2 | Custom functional-group substitution sets | EXP-2 | Generated from SMILES; see §5.2 |
| D3 | WelQrate (9 targets) | EXP-3a | [welqrate.org](https://welqrate.org) / [arXiv:2411.09820](https://arxiv.org/abs/2411.09820) |
| D4 | MUV (17 targets) | EXP-3b | [Riniker-Landrum benchmarking platform](https://github.com/rdkit/benchmarking_platform); original [Rohrer & Baumann 2009](https://doi.org/10.1021/ci8002649) |
| D5 | DUD-E (102 targets) — **deferred, see §5.5** | EXP-3c | [dude.docking.org](http://dude.docking.org) / [Mysinger et al. 2012](https://doi.org/10.1021/jm300687e) |
| D6 | Activity cliff pairs (30 ChEMBL targets) | EXP-4 | **Primary**: MoleculeACE pre-curated data — [github.com/molML/MoleculeACE/tree/main/MoleculeACE/Data/benchmark_data](https://github.com/molML/MoleculeACE/tree/main/MoleculeACE/Data/benchmark_data) / [van Tilborg et al. 2022](https://doi.org/10.1021/acs.jcim.2c01073). **Fallback** (if cliff definition needs to be MMP-only): current ChEMBL release + `mmpdb`; method: [Hu et al. 2012](https://doi.org/10.1021/ci3001138) |
| D7 | SwissBioisostere replacements | EXP-5a | [swissbioisostere.ch](http://www.swissbioisostere.ch) / [Isert et al. 2022](https://doi.org/10.1093/nar/gkab1047) |
| D8 | Curated classical bioisostere pairs | EXP-5b | Literature-based list, ~50–100 pairs; seed list from [Meanwell 2011](https://doi.org/10.1021/jm1013693) |
| D9 | Riniker-Landrum 88-target benchmark | EXP-6 | [github.com/rdkit/benchmarking_platform](https://github.com/rdkit/benchmarking_platform) / [Riniker & Landrum 2013](https://doi.org/10.1186/1758-2946-5-26) |

**Known dataset issues to document in results:**
- D5 (DUD-E) is **deferred** — see EXP-3c §5.5 for rationale. The known analog bias and property shortcuts ([Chen et al. 2019](https://doi.org/10.1371/journal.pone.0220113)) make D3+D4 a more rigorous test of the same hypothesis at ~25 % of the compute cost. If D5 is later run, report both the full 102-target set and the bias-reduced ~47-target subset from [Lagarde et al. 2015](https://doi.org/10.1021/acs.jcim.5b00090), and weight results below D3 and D4 in final conclusions.

---

## 5. Experiments

All experiments produce MUT results and corresponding results for every applicable baseline. Output artifact expectations are in §7.

### 5.1 EXP-1: Homologous Series Smoothness

- **Type**: internal
- **Question**: Does distance grow smoothly and monotonically with position along a chemical series?
- **Molecule sets** (each generated from SMILES; effective sizes after the project-wide minimum-heavy-atom filter, MIN_HEAVY_ATOMS=3 — see CLAUDE.md):
  - S1: n-alkanes C1–C12 (**10**, methane and ethane dropped)
  - S2: n-alkanols, methanol to 1-dodecanol (**11**, methanol dropped)
  - S3: n-alkanoic acids, formic to dodecanoic (12)
  - S4: n-alkylamines, methylamine to dodecylamine (**11**, methylamine dropped)
  - S5: polyethylene glycols HO-(CH₂CH₂O)ₙ-H, n=1–10 (10)
- **Metrics** (see §6): M-MONO, M-SMOOTH, M-LIN
- **Expected behavior**:
  - MUT: smooth monotonic growth.
  - Topological FPs (B1–B6): plateau/saturation for long chains as local neighborhoods become self-similar.
  - QM baselines (B15–B17): expected similarly smooth.
- **Success criterion**: MUT achieves M-MONO ≥ 0.95 on all series; M-SMOOTH is comparable to or better than B15–B17.

### 5.2 EXP-2: Functional Group Substitution Sensitivity

- **Type**: internal
- **Question**: Does the embedding reflect electronic effects of substituents?
- **Molecule sets** (effective set after the project-wide ElektroNN element filter — see CLAUDE.md):
  - S6: monosubstituted benzenes (**11**, Br dropped), substituents: -H, -CH₃, -OH, -NH₂, -F, -Cl, -NO₂, -COOH, -CHO, -COCH₃, -OCH₃
  - S7: monosubstituted cyclohexanes (10), same substituent set minus -Br and -NO₂
  - S8: para-substituted benzoic acids (Hammett series, **9**, Br dropped), X ∈ {-NH₂, -OCH₃, -CH₃, -H, -F, -Cl, -CF₃, -CN, -NO₂}. Each compound has a known Hammett σ_para value tabulated in the dataset.
- **Metrics**:
  - M-HAMMETT-PAIR (S8) — primary. Spearman ρ between |σᵢ − σⱼ| and d(Xᵢ, Xⱼ) over all pairs. Symmetric in both sides so a U-shape around H (chemically correct) doesn't get penalised. 36 pairs for n=9 compounds.
  - M-HAMMETT-ABS (S8) — secondary. Spearman ρ between |σ_para| and d(X, H-compound). Sanity check; n=9.
  - M-SILHOUETTE-S6 (S6) — silhouette of 2D MDS using donor/acceptor/neutral labels. Conjugated scaffold; higher is better.
  - M-SILHOUETTE-S7 (S7) — same score on aliphatic cyclohexane control. Electronic effects shouldn't propagate on sp³ scaffolds, so **lower is better** here — this is the conjugation-specificity control and must not be averaged with S6.
  - ~~M-HAMMETT~~ (signed σ vs d(X, H)) dropped: σ is signed, distance is unsigned, so a correct U-shape scores low. Replaced by M-HAMMETT-PAIR.
  - ~~M-HALO-ORDER~~ dropped — requires Br, which is outside ElektroNN's supported element set.
- **Expected behavior**: MUT shows strong M-HAMMETT-PAIR correlation because σ values reflect electronic effects. B1–B6 perform poorly because they encode connectivity, not electronics. B15–B17 also expected to perform well. M-SILHOUETTE-S6 should exceed M-SILHOUETTE-S7 for MUTs that are genuinely conjugation-sensitive.
- **Success criterion**: MUT achieves Spearman ρ ≥ 0.7 on M-HAMMETT-PAIR, significantly higher than all B1–B6 (p < 0.01 via permutation test). Threshold relaxed from the old ≥ 0.8 because the pair metric operates on 36 noisier data points.

### 5.3 EXP-3a: WelQrate Retrieval

- **Type**: external
- **Question**: Does MUT rank actives closer to query actives than decoys/inactives?
- **Data**: D3. Nine datasets (AID435008, AID1798, AID435034, AID1843, AID2258, AID463087, AID488997, AID2689, AID485290). Use scaffold split provided by WelQrate (prevents analog-bias leakage).
- **Scaffold split interpretation**: The Bemis-Murcko scaffold of a molecule is its ring system with all side chains stripped. A scaffold split groups molecules sharing the same scaffold and assigns whole scaffold groups to train/valid/test, so no scaffold seen at test time appears in train. WelQrate provides five seeds — five independent random shuffles of the scaffold-group ordering before the 3:1:1 partitioning. Each seed yields a different but equally valid partition; the molecule's scaffold never changes, only which partition it lands in. For EXP-3a the **test** split supplies the queries; results are averaged across all five seeds to reduce sensitivity to any single partition.
- **Pool composition**: **valid + test** (all non-train molecules, minus the query itself). Train is excluded to reduce computation (it is 60 % of the data and contributes no unique signal for non-learned methods). Valid molecules have different scaffolds from the query by construction; test molecules include scaffold-mates of the query, preserving a realistic retrieval setting. Scaffold-diversity analysis is delegated to EXP-6.
- **Protocol**:
  - For each target and each scaffold seed: draw query actives from the test split.
  - For each query, rank the valid+test pool by distance; record active ranks (compact). Compute M-LOGAUC, M-BEDROC20, M-EF1, M-DCG100; average over queries and seeds per target.
- **Metrics**: M-LOGAUC, M-BEDROC20, M-EF1, M-DCG100
- **Expected behavior**: Strong methods achieve BEDROC20 > 0.3. MUT expected competitive with learned methods (B12–B14) and better than pure topology on scaffold-split.
- **Success criterion**: MUT's median LogAUC across 9 targets ≥ that of ECFP4 (B1) and ≥ Mol2vec (B12).

### 5.4 EXP-3b: MUV Retrieval

- **Type**: external
- **Question**: Same as EXP-3a, on a benchmark specifically constructed to defeat analog-based similarity.
- **Data**: D4. 17 targets. MUV design removes analog bias and artificial enrichment.
- **Protocol**: Same retrieval protocol as EXP-3a; 5 random query selections per target.
- **Metrics**: M-AUCROC, M-BEDROC20, M-EF1
- **Expected behavior**: All topological FPs struggle (this is MUV's design purpose). MUT advantage, if real, should show most clearly here.
- **Success criterion**: MUT's mean AUCROC > 0.7 on majority of targets AND statistically better than best topological FP (Wilcoxon signed-rank, p < 0.05 across 17 targets).

### 5.5 EXP-3c: DUD-E Retrieval — **DEFERRED**

> **Status: deferred.** Not implemented in the current iteration. Two open actions:
> 1. **Re-evaluate at implementation time** — revisit only if EXP-3a (WelQrate) and EXP-3b (MUV) results are ambiguous and need a third retrieval data point, or if a specific reviewer/collaborator asks for DUD-E numbers.
> 2. **Discuss at writing time** — the paper must explicitly justify the omission (or post-hoc inclusion). The "why we did not run DUD-E" paragraph belongs in the methods or limitations section, citing Chen et al. 2019 (analog bias) and Lagarde et al. 2015 (property shortcut) and pointing at EXP-3b as the better-designed replacement.
>
> **Deferral rationale.**
> - D5 has well-documented biases: analog bias ([Chen et al. 2019](https://doi.org/10.1371/journal.pone.0220113)) and property-matching shortcuts that let trivial classifiers succeed.
> - EXP-3a (curated scaffold-split retrieval) and EXP-3b (analog-bias-free retrieval by design) already test the same hypothesis more rigorously; D5 contributes mostly "literature comparability", a publication-strategy argument rather than a scientific one.
> - Compute cost is prohibitive: ~1.15 M molecules for the full 102-target set (~530 k for the Lagarde bias-reduced 47-target subset) — 2–4× of D3+D4 combined, dominated by ElektroNN inference.
>
> The specification below describes what EXP-3c *would* look like if implemented. Keep it intact so a future contributor (or future-us) can resume without redesign.

- **Type**: external (with caveats)
- **Question**: Literature-comparable retrieval performance.
- **Data**: D5. 102 targets, each with ~224 actives and 50× property-matched decoys.
- **Protocol**: Per-target retrieval with one active as query; rank remaining actives + decoys. Report on full 102-target set AND on the ~47-target bias-reduced subset.
- **Metrics**: M-AUCROC, M-BEDROC20, M-EF1, M-LOGAUC
- **Expected behavior**: All methods perform well on full DUD-E due to property-matching bias. Bias-reduced subset is more informative.
- **Success criterion**: MUT performance on bias-reduced subset ≥ B8 (Gaussian shape + color Tanimoto) on median AUCROC.
- **Weighting**: Results here weighted BELOW EXP-3a and EXP-3b due to known biases.

### 5.6 EXP-4: Activity Cliff Analysis

- **Type**: external
- **Question**: Does MUT distance track potency difference across MMPs?
- **Data**: D6. Two construction options — pick one and document:
  - **(Preferred) MoleculeACE benchmark** ([van Tilborg et al. 2022](https://doi.org/10.1021/acs.jcim.2c01073)): 30 ChEMBL targets, 35,632 unique molecules, pre-curated with cliff labels, train/test splits, and per-pair similarity metrics. Activity cliff = pair with similarity ≥ 0.9 in any of {ECFP-Tanimoto, scaffold-ECFP-Tanimoto, scaled-Levenshtein on SMILES} AND ≥10× difference in Ki or EC50. Apply project-wide ElektroNN element filter (drop molecules with unsupported atoms) before computing metrics. Data: [github.com/molML/MoleculeACE/tree/main/MoleculeACE/Data/benchmark_data](https://github.com/molML/MoleculeACE/tree/main/MoleculeACE/Data/benchmark_data).
  - **(Fallback) MMP-only** if a strict matched-molecular-pair definition is required: extract ChEMBL Ki against human targets, generate MMPs via Hussain-Rea (`mmpdb`, transformation size ≤ 13 heavy atoms), label cliff if |ΔpKi| ≥ 2 and non-cliff if |ΔpKi| < 0.5, sample ~5,000 of each class matched for scaffold diversity.
- **Metrics**: M-SALI-DIST (distribution comparison), M-CLIFF-AUC, M-DIST-POTENCY-RHO
- **Expected behavior**: Topological FPs assign very high similarity (low distance) to both cliffs and non-cliffs because transformations are small by construction — poor cliff discrimination. MUT should assign higher distance to cliff pairs if it captures electronic perturbation from small structural changes.
- **Success criterion**: MUT's Spearman ρ (distance vs. |ΔpKi|) > 0.3 AND significantly higher than all B1–B6 (p < 0.01).
- **Sub-analysis**: Stratify M-DIST-POTENCY-RHO by transformation type: R-group / ring / linker changes.
- **Optional upgrade — decide at implementation time**: cliffs have multiple causes (electronic, steric, receptor-specific induced fit) and a ligand-only method cannot resolve receptor-specific cases. To make claims sharper, stratify pairs into easy / medium / hard subsets by transformation class:
  - **Easy / electronic**: charge-state change, H-bond donor/acceptor swap, strong π-perturbation (e.g. -OMe → -CN). MUT *should* clearly beat topological FPs.
  - **Easy / steric**: Δheavy-atom-count ≥ 3, branching change (Me → tBu), ring fusion. MUT *should* beat topological FPs moderately.
  - **Medium**: halogen swap (F↔Cl), single H↔Me, polar-to-polar same-class swap. Calibration zone.
  - **Hard / receptor-required**: bioisosteric swap, single-atom change in flexible region, subtle conformational lock. Parity expected — no ligand-only method should win.
  - Add **diagnostic tests** that bypass the receptor entirely (verify the embedding is sensitive to known-important features): among non-cliff pairs, MUT distance for charge-changing / H-bond-changing / large-Δvolume pairs should exceed MUT distance for inert pairs, and the gap should be larger than for ECFP. If MUT fails these, EXP-4 is dead before the cliff AUC matters.
  - **Reframes the headline claim** from "MUT beats baselines at cliff detection" to "MUT beats baselines on ligand-explainable cliffs and matches them on receptor-specific cliffs" — weaker but truer, and dovetails with EXP-2 (Hammett) and EXP-5 (bioisosteres).
  - **Cost**: writing the transformation classifier is non-trivial (a few hundred LOC of cheminformatics). Decide at EXP-4 implementation time whether to pay it.

### 5.7 EXP-5: Bioisostere Recognition

- **Type**: external
- **Question**: Does MUT place functionally-equivalent-but-topologically-different fragments closer than fingerprints do?

**EXP-5a: SwissBioisostere-derived test**
- **Data**: D7. Construct three pair sets:
  - Set BIO+ (bioisosteric, activity-preserving): SwissBioisostere MMPs with bioisosteric replacement AND |ΔpActivity| < 0.3 on same target. Target ~2000 pairs.
  - Set MMP-match (control, activity-preserving non-bioisostere): random MMPs with |ΔpActivity| < 0.3. Target ~2000 pairs.
  - Set MMP-nomatch (negative, activity-disrupted): random MMPs with |ΔpActivity| > 1.5. Target ~2000 pairs.
- **Metrics**: M-BIO-AUC (BIO+ vs. MMP-match), M-BIO-RANK

**EXP-5b: Classical bioisostere pairs**
- **Data**: D8. For each classical pair (e.g., COOH↔tetrazole, phenyl↔thienyl, amide↔oxadiazole, ester↔reversed amide, pyridine↔pyrimidine, OH↔SH, H↔F in aromatic context, ~50–100 pairs total):
  - Pull from ChEMBL: ≥20 compounds containing fragment A + ≥20 matched compounds with fragment B on same scaffold/target with comparable activity.
- **Metrics**: M-CROSS-WITHIN-RATIO

- **Expected behavior**: Topological FPs fail by construction — bioisosteres have different atoms. 3D shape methods partially succeed via volume similarity. MUT expected to excel since bioisosterism is fundamentally about equivalent electron distributions.
- **Success criterion (CRITICAL)**: MUT ranks top-2 among all methods on M-BIO-AUC. **If MUT does not substantially outperform topological FPs here, the core hypothesis is weakened regardless of other results.**

### 5.8 EXP-6: Scaffold Hopping

- **Type**: external
- **Question**: Does MUT retrieve actives with diverse scaffolds, not just analogs of the query?
- **Data**: D9. 88 targets filtered for scaffold-hopping evaluation.
- **Protocol**:
  - For each target, select query; rank all compounds by distance.
  - Measure both overall enrichment and scaffold-level enrichment at 1%, 5%, 10% cutoffs.
  - 5 random queries per target.
  - **Additional**: leave-one-scaffold-out retrieval on the 20 targets with ≥5 distinct active scaffolds. Use all actives from one scaffold as queries; measure retrieval of actives from held-out scaffolds only.
- **Metrics**: M-EF5, M-SCAFEF5, M-SCAFRATIO, M-RECALL-HELDOUT-SCAFFOLD
- **Expected behavior**: Topological FPs (especially B1, B2) retrieve analogs → high EF, low ScafEF/EF ratio. FCFP4 (B3) hops scaffolds better via feature abstraction. 3D methods (B8, B10) show better scaffold diversity. MUT should show the best ScafEF/EF ratio.
- **Success criterion**: MUT's median M-SCAFRATIO across 88 targets > ECFP4 (B1), p < 0.05 via Wilcoxon signed-rank.

---

## 6. Metrics (Central Definitions)

Any metric referenced by ID in §5 is defined here.

### 6.1 Internal-Experiment Metrics

| ID | Name | Definition |
|----|------|------------|
| M-MONO | Monotonicity score | Spearman ρ between chain-length difference \|i−j\| and d(mol_i, mol_j), over all pairs within a series. Range [-1, 1]; 1 = perfect. |
| M-SMOOTH | Smoothness score | Standard deviation of consecutive-distance ratios d(k, k+1) / d(k-1, k) across a series. Lower is smoother. |
| M-LIN | Linearity R² | Linear regression of d(mol_1, mol_k) vs. k. |
| M-HAMMETT-PAIR | Hammett pair correlation (primary) | Spearman ρ between \|σᵢ − σⱼ\| and d(Xᵢ, Xⱼ) over all C(n,2) pairs in S8. Higher = better. |
| M-HAMMETT-ABS | Hammett magnitude correlation (secondary) | Spearman ρ between \|σ_para\| and d(X, H-compound) on S8. Higher = better. |
| M-SILHOUETTE-S6 | Donor/acceptor clustering on conjugated scaffold | Silhouette score of 2D MDS projection of S6 distance matrix, using donor/acceptor/neutral labels. Higher = better. |
| M-SILHOUETTE-S7 | Donor/acceptor clustering on aliphatic control | Same score on S7 (cyclohexane). **Lower = better** — aliphatic scaffolds shouldn't cluster by electronic class. Do not average with S6. |
| ~~M-HAMMETT~~ | ~~Signed σ vs d(X, H)~~ | *Dropped: σ is signed and distance is unsigned, so a chemically-correct U-shape scores poorly. Replaced by M-HAMMETT-PAIR.* |
| ~~M-HALO-ORDER~~ | ~~Halogen ordering correctness~~ | *Dropped: Br is outside ElektroNN's supported basis set (see CLAUDE.md §"SMILES-stage element filter").* |

### 6.2 Retrieval Metrics

| ID | Name | Definition / Reference |
|----|------|------------------------|
| M-AUCROC | Area under ROC | Standard; positives = actives, negatives = decoys/inactives. |
| M-LOGAUC | Log-scaled AUC | Emphasizes early enrichment; per Mysinger et al. 2012. |
| M-BEDROC20 | Boltzmann-enhanced discrimination of ROC, α=20 | Truchon & Bayly 2007. |
| M-EF1 | Enrichment factor at 1% | (fraction of actives in top 1%) / (fraction of actives overall) |
| M-EF5 | Enrichment factor at 5% | As M-EF1 but at 5%. |
| M-DCG100 | Discounted cumulative gain at rank 100 | Per WelQrate. |
| M-SCAFEF5 | Scaffold enrichment factor at 5% | Number of distinct Bemis-Murcko scaffolds among actives in top 5%, normalized by expectation under random ranking. |
| M-SCAFRATIO | Scaffold-EF / EF ratio | M-SCAFEF5 / M-EF5. Values closer to 1.0 indicate scaffold-diverse retrieval. |
| M-RECALL-HELDOUT-SCAFFOLD | Recall on held-out scaffolds | Fraction of held-out-scaffold actives retrieved within top k%. |

### 6.3 Activity-Cliff Metrics

| ID | Name | Definition |
|----|------|------------|
| M-SALI-DIST | SALI distribution difference | SALI(A,B) = \|ΔpKi\| / (1 − sim(A,B)). Compare distributions between cliff and non-cliff pairs (KS test). |
| M-CLIFF-AUC | Cliff-detection AUC | Rank all pairs by SALI; compute AUC for classifying cliffs vs non-cliffs. |
| M-DIST-POTENCY-RHO | Distance–potency correlation | Spearman ρ between d(A,B) and \|ΔpKi\| across all MMP pairs. |

### 6.4 Bioisostere Metrics

| ID | Name | Definition |
|----|------|------------|
| M-BIO-AUC | Bioisostere detection AUC | Positives = BIO+ set, Negatives = MMP-match set. Rank by similarity (higher = more similar). |
| M-BIO-RANK | Bioisostere rank | For each BIO+ pair (A, B), rank B among all other molecules in the dataset by similarity to A. Lower is better. Report median rank. |
| M-CROSS-WITHIN-RATIO | Cross-set vs within-set similarity | (mean similarity between fragment-A-compounds and fragment-B-compounds) / (mean within-set similarity). Closer to 1.0 = better bioisostere recognition. |

---

## 7. Output Artifacts

Each experiment must produce the following, saved to `results/EXP-X/`:

- `raw/` — per-target, per-method, per-query raw results (one row per query per target)
- `aggregated.csv` — per-target, per-method means with standard errors
- `summary.csv` — per-method aggregated across targets, with all §6 metrics
- `statistics.csv` — pairwise significance tests (MUT vs each baseline), Wilcoxon signed-rank with Bonferroni correction
- `plots/` — per-experiment visualizations (see §8)
- `metadata.json` — conformer settings used, software versions, random seeds, compute time

**Embeddings** (intermediate products) are saved to `cache/embeddings/{method_id}/{dataset_id}.pkl` and indexed by manifest for staleness tracking. Every artifact sidecar records `{version, inputs, output_hash, compute_time, upstream_compute_time, timestamp, dataset_size, compute_time_per_mol}`. The aggregate `results/SUMMARY.md` table includes an `s/mol` column (end-to-end chain time per molecule, averaged across datasets).

### 7.1 Ranking Semantics in `results/SUMMARY.md`

The summary writer aggregates ranks in two stages:

1. **Within an experiment.** For each metric, methods are ranked by mean value across the metric's applicable datasets (lower rank = better, with `metric_direction` controlling sort order). Each method's per-experiment "Avg rank" column is the mean of its metric ranks within that experiment.
2. **Across experiments.** Each experiment contributes exactly one scalar — its within-experiment Avg rank — to the cross-experiment average. The number of metrics inside an experiment does **not** affect cross-experiment weight: a 4-metric experiment and a 3-metric experiment count equally.

**Deferred design decision — headline-metric selection (revisit when EXP-4/5/6 land).**

Several experiments report multiple metrics that capture closely-related signal:

- **EXP-3a**: M-LOGAUC, M-BEDROC20, M-EF1, M-DCG100 are all early-enrichment retrieval scores with different weighting curves; they typically agree on method ranking. Effective independent signal ≈ 1.3 of 4.
- **EXP-3b**: M-AUCROC is genuinely distinct from M-BEDROC20 + M-EF1, but the latter two cluster together.
- **EXP-2**: M-HAMMETT-PAIR (all O(n²) pairs) and M-HAMMETT-ABS (just the H-reference row) measure the same Hammett-correlation capability with different aggregations.
- **EXP-1**: M-MONO (Spearman) and M-LIN (R²) overlap on monotonic-distance growth; M-SMOOTH is genuinely orthogonal.
- **EXP-4 (planned)**: M-CLIFF-AUC, M-SALI-DIST, M-DIST-POTENCY-RHO all probe "does distance grow with potency difference" at different aggregations.
- **EXP-5 (planned)**: M-BIO-AUC and M-BIO-RANK both score bioisostere separation.
- **EXP-6 (planned)**: M-SCAFRATIO is literally `M-SCAFEF5 / M-EF5` — including all three triple-counts the same ranking.

Within-experiment redundancy is mostly fine for the cross-experiment ranking (each experiment still contributes one collapsed value), but it distorts the per-experiment "Avg rank" column shown in each experiment's table: a method's per-experiment rank reflects a consensus across redundant metrics rather than a balanced view.

**Proposed mechanism (not yet implemented):** Add an optional `rank_metrics: tuple[str, ...]` on the `Experiment` protocol. The SUMMARY writer would use this subset for both the within-experiment Avg rank and the cross-experiment aggregation, while the per-experiment table continues to show every metric (rank_metrics + diagnostic metrics) for transparency. If `rank_metrics` is unset, behavior falls back to today's "all metrics".

**Candidate headline picks** (decide at implementation time, not now):

| Experiment | Candidate `rank_metrics` | Demoted to diagnostic |
|----|----|----|
| EXP-1 | `("M-MONO", "M-SMOOTH")` | M-LIN |
| EXP-2 | `("M-HAMMETT-PAIR", "M-SILHOUETTE-S6", "M-SILHOUETTE-S7")` | M-HAMMETT-ABS |
| EXP-3a | `("M-LOGAUC",)` *or* `("M-BEDROC20",)` | The other three (LogAUC is WelQrate's canonical; BEDROC20 is the literature default) |
| EXP-3b | `("M-AUCROC", "M-BEDROC20")` | M-EF1 |
| EXP-4 | `("M-CLIFF-AUC",)` | M-SALI-DIST, M-DIST-POTENCY-RHO |
| EXP-5 | `("M-BIO-AUC",)` | M-BIO-RANK, M-CROSS-WITHIN-RATIO |
| EXP-6 | `("M-SCAFRATIO", "M-RECALL-HELDOUT-SCAFFOLD")` | M-EF5, M-SCAFEF5 |

These picks are judgement calls and benefit from being made when writing the paper (with a full set of cross-method numbers visible), not pre-emptively now. The mechanism is cheap to add; the picks are the load-bearing decision.

---

## 8. Required Plots

- EXP-1: One subplot per series per method; x = chain length, y = d(mol_1, mol_k). Methods overlaid for comparison series.
- EXP-2: MDS projection of S6 with donor/acceptor/neutral coloring. Scatter plot of Hammett σ vs MUT distance for S8, overlaid with baseline scatters.
- EXP-3 (a/b/c): Method × target heatmap, colored by BEDROC20. Box plots per method across targets.
- EXP-4: Distribution of distances for cliff vs non-cliff pairs per method. Distance vs. |ΔpKi| scatter per method with fitted line.
- EXP-5: Bar chart of M-BIO-AUC per method. Per-bioisostere-type breakdown (one bar per pair type).
- EXP-6: M-SCAFRATIO distribution per method (violin plot). Critical difference diagram.
- Aggregated: Critical difference diagram across all external benchmarks (EXP-3 through EXP-6), average-rank-based, with Nemenyi post-hoc.

---

## 9. Overall Success Criteria

MUT is considered an improvement if **at least three** of the following are met:

1. **Electronic sensitivity (EXP-2)**: Significantly better M-HAMMETT-PAIR than all topological FPs (p < 0.01).
2. **Bioisostere recognition (EXP-5)**: Top-2 among all methods on M-BIO-AUC.
3. **Scaffold hopping (EXP-6)**: Significantly higher M-SCAFRATIO than ECFP4 (p < 0.05).
4. **Retrieval on curated data (EXP-3a & 3b)**: Non-inferior on WelQrate vs best FP baseline AND superior on MUV (p < 0.05).
5. **Activity-cliff sensitivity (EXP-4)**: Significantly higher M-DIST-POTENCY-RHO than all topological FPs (p < 0.01).

**Hypothesis-failure condition**: If MUT fails EXP-5 specifically, the core hypothesis is weakened regardless of other results; re-examine basis-function choice and distance metric before further benchmarking.

**Niche-success condition**: If MUT excels on EXP-2, EXP-5, EXP-6 but underperforms on EXP-3, this indicates a specialized advantage (scaffold hopping / bioisostere applications) rather than general-purpose superiority. This is still a publishable finding but changes positioning.

---

## 10. Cross-Cutting Requirements

- **Reproducibility**: Fix random seeds everywhere. Record software versions.
- **Method fairness**: All 3D methods use the identical conformer ensemble (§3).
- **Statistical reporting**: Always report effect size alongside p-values (Cliff's delta or matched-pairs rank-biserial).
- **Never report a single cherry-picked metric**: Full §6 metric suite for each experiment.
- **Always report scaffold-split vs random-split**: For EXP-3 (where applicable), to quantify analog-bias effects.
- **Compute cost**: Record per-molecule embedding time and per-pair distance time for all methods for a fair cost/quality discussion.
