# Experimental Plan: Benchmarking Electron Density-Based Molecular Similarity

## 1. Overview and Objectives

This document describes a systematic experimental plan to evaluate an electron density-based molecular embedding and its associated distance function against established molecular similarity methods. The central hypothesis is that electron density coefficient vectors, grounded in quantum mechanical theory, capture functional molecular similarity better than topology-based fingerprints.

The evaluation is structured in two phases: **internal consistency validation** (Experiments 1–2) to verify the embedding space behaves chemically sensibly, and **external benchmarking** (Experiments 3–6) to quantify performance against established datasets and baselines.

---

## 2. Baseline Methods

A comprehensive set of baselines is critical. The following methods span five distinct paradigms—topological fingerprints, descriptor-based, 3D shape/field, learned representations, and quantum-chemical descriptors—ensuring the comparison is not biased toward any single approach.

### 2.1 Topological Fingerprints (2D)

All fingerprint-based baselines use Tanimoto similarity unless stated otherwise.

| Method | Description | Tool / Reference |
|--------|-------------|------------------|
| **ECFP4 (Morgan r=2, 2048 bits)** | Circular fingerprint encoding atom neighborhoods up to radius 2. The most widely used fingerprint in modern cheminformatics. | RDKit `GetMorganFingerprintAsBitVect`. Rogers & Hahn, *J. Chem. Inf. Model.* 2010, 50, 742–754. DOI: [10.1021/ci100050t](https://doi.org/10.1021/ci100050t) |
| **ECFP6 (Morgan r=3, 2048 bits)** | Larger radius variant capturing more extended neighborhoods. | Same as above. |
| **FCFP4 (Morgan feature-based, r=2)** | Pharmacophoric variant of ECFP4 using feature invariants (H-bond donor/acceptor, aromatic, etc.) instead of element-based atom types. Expected to perform better on scaffold hopping. | RDKit `GetMorganFingerprintAsBitVect` with `useFeatures=True`. |
| **MACCS Keys (166 bits)** | Dictionary-based structural keys. Simple baseline representing predefined substructural patterns. | RDKit `MACCSkeys.GenMACCSKeys`. |
| **Atom Pair fingerprint** | Encodes pairs of atoms and their topological distance. | RDKit `GetAtomPairFingerprint`. Carhart et al., *J. Chem. Inf. Comput. Sci.* 1985, 25, 64–73. |
| **Topological Torsion fingerprint** | Encodes sequences of four consecutively bonded non-hydrogen atoms. | RDKit `GetTopologicalTorsionFingerprint`. Nilakantan et al., *J. Chem. Inf. Comput. Sci.* 1987, 27, 82–85. |

### 2.2 Physicochemical Descriptor-Based

| Method | Description | Tool / Reference |
|--------|-------------|------------------|
| **RDKit 2D descriptors (200 descriptors) + Euclidean/cosine distance** | Molecular weight, logP, TPSA, ring counts, hydrogen bond donors/acceptors, etc. Represents classical QSAR descriptors. | RDKit `Descriptors` module. |

### 2.3 3D Shape and Field Similarity

| Method | Description | Tool / Reference |
|--------|-------------|------------------|
| **ROCS (Shape + Color Tanimoto)** | Gaussian-based molecular volume overlay. Industry standard for 3D shape comparison. Requires conformer generation. Commercial but widely reported. | OpenEye ROCS. Grant et al., *J. Phys. Chem.* 1996, 100, 11775. Documentation: [docs.eyesopen.com/applications/rocs](https://docs.eyesopen.com/applications/rocs/rocs/rocs_overview.html) |
| **USR (Ultrafast Shape Recognition)** | Alignment-free shape descriptor using distributions of atom distances from four reference points. Produces a 12-dimensional vector per molecule. Fast, open-source alternative to ROCS. | Ballester & Richards, *J. Comput. Chem.* 2007, 28, 1711–1723. Implementation available in RDKit `Chem.Descriptors3D.GetUSR`. |
| **USRCAT** | Extension of USR incorporating pharmacophoric atom-type information (hydrogen bond donors, acceptors, hydrophobic atoms, aromatic atoms). 60-dimensional vector. | Schreyer & Blundell, *J. Cheminform.* 2012, 4, 27. RDKit `Chem.Descriptors3D.GetUSRCAT`. |
| **Electrostatic similarity (eSim)** | Combines electrostatic field comparison with molecular shape. Relevant since your electron density method is also fundamentally electrostatic. | Jain, *J. Comput. Aided Mol. Des.* 2020, 34, 129–150. DOI: [10.1007/s10822-019-00236-6](https://doi.org/10.1007/s10822-019-00236-6) |

### 2.4 Learned Molecular Representations

| Method | Description | Tool / Reference |
|--------|-------------|------------------|
| **Mol2vec (300-d vectors)** | Unsupervised Word2vec-style embedding of Morgan substructures. Pre-trained model available. Similarity via cosine distance. | Jaeger et al., *J. Chem. Inf. Model.* 2018, 58, 27–35. DOI: [10.1021/acs.jcim.7b00616](https://doi.org/10.1021/acs.jcim.7b00616). GitHub: [github.com/samoturk/mol2vec](https://github.com/samoturk/mol2vec) |
| **Uni-Mol embeddings** | 3D-aware transformer pretrained on 209M conformations. State-of-the-art learned representation. Extract final-layer representations and compute cosine similarity. | Zhou et al., ICLR 2023. GitHub: [github.com/deepmodeling/Uni-Mol](https://github.com/deepmodeling/Uni-Mol) |
| **Chemprop D-MPNN fingerprints** | Directed message-passing neural network. Extract the learned molecular representation (hidden state) before the final prediction layer. | Yang et al., *J. Chem. Inf. Model.* 2019, 59, 3370–3388. DOI: [10.1021/acs.jcim.9b00237](https://doi.org/10.1021/acs.jcim.9b00237). GitHub: [github.com/chemprop/chemprop](https://github.com/chemprop/chemprop) |

### 2.5 Quantum-Chemical Descriptors

These are the most relevant comparisons since your method also derives from QM. They help isolate whether any advantage comes from the electron density basis specifically or from including QM information in general.

| Method | Description | Tool / Reference |
|--------|-------------|------------------|
| **Coulomb Matrix + kernel distance** | Encodes nuclear charges and internuclear distances. The original QM-inspired molecular descriptor for ML. | Rupp et al., *Phys. Rev. Lett.* 2012, 108, 058301. |
| **SOAP (Smooth Overlap of Atomic Positions)** | Encodes local atomic environments as power spectra of spherical harmonics. Can be compared via a kernel. | Bartók et al., *Phys. Rev. B* 2013, 87, 184115. |
| **ACSF (Atom-Centered Symmetry Functions)** | Encodes radial and angular distributions of neighbors around each atom. Aggregated to molecular level. | Behler, *J. Chem. Phys.* 2011, 134, 074106. |

---

## 3. Internal Validation Experiments

### Experiment 1: Homologous Series Smoothness

**Objective:** Verify that the embedding space produces smooth, monotonic trajectories for structurally related series, confirming the distance function captures gradual chemical changes.

**Molecule sets (construct all from SMILES using RDKit):**

1. **n-Alkanes:** methane through n-dodecane (C1–C12). 12 molecules.
2. **n-Alkanols:** methanol through 1-dodecanol. 12 molecules.
3. **n-Alkanoic acids:** formic acid through dodecanoic acid. 12 molecules.
4. **n-Alkylamines:** methylamine through dodecylamine. 12 molecules.
5. **Polyethylene glycols:** HO-(CH₂CH₂O)ₙ-H, n = 1–10. 10 molecules.

**Procedure:**

1. Compute electron density embedding vectors for all molecules in each series.
2. Compute pairwise distance matrices within each series using your distance function.
3. For each series, extract the sequence of consecutive distances: d(C₁,C₂), d(C₂,C₃), ..., d(Cₙ₋₁,Cₙ).
4. Repeat steps 1–3 for all baseline methods.

**Measurements:**

- **Monotonicity score:** Spearman rank correlation between chain length difference |i−j| and embedding distance d(Cᵢ,Cⱼ) for all pairs within a series. Perfect monotonicity = 1.0.
- **Smoothness score:** Standard deviation of consecutive distance ratios d(Cₖ,Cₖ₊₁)/d(Cₖ₋₁,Cₖ). Lower values indicate smoother trajectories.
- **Linearity (R²):** Linear regression of d(C₁,Cₖ) vs. k. High R² indicates the embedding captures the regular incremental change of adding CH₂ groups.

**Visualization:** Plot embedding distance from the first member vs. chain length for each series. Each method gets one subplot. Lines should be approximately linear and monotonically increasing.

**Expected outcomes:** All methods should show reasonable monotonicity for simple alkane series. The electron density method should show smoother trajectories (lower smoothness score variance) because it captures the continuous change in electron distribution, whereas fingerprint methods may show step-function behavior when substructural features appear or disappear at certain chain lengths. Topological fingerprints will likely saturate (distances plateau) for longer chains since the local neighborhoods become self-similar. The QM baselines (Coulomb matrix, SOAP) should also show smooth behavior.

---

### Experiment 2: Functional Group Substitution Sensitivity

**Objective:** Verify that the embedding captures the chemical effect of functional group changes on a fixed molecular scaffold, and that the relative distances reflect known chemical intuition (e.g., carboxylic acid is closer to ester than to alkane).

**Design:** Use a common scaffold and systematically vary one substituent. Two scaffold sets:

**Set A — Monosubstituted benzenes (12 molecules):**

Benzene with substituents: -H, -CH₃, -OH, -NH₂, -F, -Cl, -Br, -NO₂, -COOH, -CHO, -COCH₃, -OCH₃.

**Set B — Substituted cyclohexanes (10 molecules):**

Cyclohexane with substituents: -H, -CH₃, -OH, -NH₂, -F, -Cl, -COOH, -CHO, -COCH₃, -OCH₃.

**Set C — Para-substituted benzoic acids (Hammett series, 10 molecules):**

4-X-benzoic acid where X = -NH₂, -OCH₃, -CH₃, -H, -F, -Cl, -Br, -CF₃, -CN, -NO₂. These are ordered by Hammett σ_para constant, providing a physicochemical reference ordering.

**Procedure:**

1. Compute embeddings and pairwise distance matrices for each set.
2. Perform multidimensional scaling (MDS, 2D) or UMAP projection on the distance matrix.
3. For Set C, compute rank correlation between Hammett σ values and distance from the unsubstituted compound (X = H).

**Measurements:**

- **Chemical clustering:** In the MDS plot of Set A, do electron-donating groups (OH, NH₂, OCH₃) cluster separately from electron-withdrawing groups (NO₂, COOH, CHO)? Evaluate by silhouette score after labeling compounds as donors/acceptors/neutral.
- **Hammett correlation (Set C only):** Spearman ρ between σ_para and d(X, H) across all methods. A high ρ indicates the similarity metric tracks electronic effects.
- **Halogen ordering:** Within the halogen substituents (F, Cl, Br), verify that d(F,Cl) < d(F,Br) for methods that capture polarizability/electronegativity trends.

**Expected outcomes:** Topological fingerprints (ECFP, MACCS) will largely fail the Hammett correlation test because they encode connectivity, not electronic effects. ROCS/USR will partially capture the effect through shape differences. The electron density method should show the strongest Hammett correlation because σ values are themselves derived from electronic effects. QM baselines (Coulomb matrix, SOAP) should also perform well. Mol2vec may capture some trends from statistical co-occurrence but without physical grounding.

---

## 4. External Validation Experiments

### Experiment 3: Activity-Based Retrospective Benchmarks

**Objective:** Evaluate whether the electron density distance function ranks active compounds closer to each other (and farther from inactives/decoys) compared to baselines.

#### 3a. WelQrate

**Data source:** 9 curated datasets across 5 therapeutic target classes from WelQrate.

- Paper: Liu et al., *NeurIPS 2024 Datasets and Benchmarks Track*. arXiv: [2411.09820](https://arxiv.org/abs/2411.09820)
- Data: [welqrate.org](https://welqrate.org)
- Key advantage over MoleculeNet: hierarchical curation with confirmatory and counter screens, PAINS filtering, defined stereochemistry.

**Datasets:** AID1798 (5-HT1a), AID435008 (Rab9), AID435034 (ALDH1a), AID1259247 (KAT2A), AID2689 (hTDP1), AID624202 (ADRB2), AID449739 (Cav3 T-type), AID488997 (OPRK1), AID652065 (FEN1).

**Procedure:**

1. For each dataset and each method, compute the embedding for every compound.
2. Use scaffold split (provided by WelQrate) to prevent data leakage from analog enrichment.
3. For the similarity-based retrieval task: for each active compound as query, rank all other compounds by similarity (ascending distance) and compute retrieval metrics.
4. For the classification task: use a k-NN classifier (k = 5, 10, 20) based on the embedding distance. Predict active vs. inactive.

**Metrics (as recommended by WelQrate):**

- **LogAUC** (log-scaled area under the ROC curve, emphasizing early enrichment)
- **BEDROC(α=20)** (Boltzmann-Enhanced Discrimination of ROC, early enrichment metric)
- **EF₁%** (enrichment factor at 1% of ranked list)
- **DCG₁₀₀** (Discounted Cumulative Gain at rank 100)
- **AUC-ROC** (overall discrimination)

Report mean ± standard error across 5 random query selections per target.

#### 3b. MUV (Maximum Unbiased Validation)

**Data source:** 17 datasets from PubChem BioAssay, specifically designed to be free of analog bias and artificial enrichment.

- Paper: Rohrer & Baumann, *J. Chem. Inf. Model.* 2009, 49, 169–184. DOI: [10.1021/ci8002649](https://doi.org/10.1021/ci8002649)
- Data: Available via the Riniker & Landrum benchmarking platform at [github.com/rdkit/benchmarking_platform](https://github.com/rdkit/benchmarking_platform)
- ~15,000 decoys and 30 actives per target.

**Procedure:** Same retrieval protocol as WelQrate.

**Metrics:** AUC-ROC, BEDROC(α=20), EF₁%.

**Why MUV matters:** MUV datasets are specifically constructed to be challenging for topological similarity methods, with actives that are not structural analogs of each other. If the electron density method succeeds where fingerprints fail, this is strong evidence for capturing functional rather than structural similarity.

#### 3c. DUD-E

**Data source:** 102 protein targets with 22,886 clustered ligands from ChEMBL, each with 50 property-matched decoys from ZINC.

- Paper: Mysinger et al., *J. Med. Chem.* 2012, 55, 6582–6594. DOI: [10.1021/jm300687e](https://doi.org/10.1021/jm300687e)
- Data: [dude.docking.org](http://dude.docking.org)

**Important caveat:** DUD-E has known biases, particularly analog bias and property-based shortcuts that allow trivial classifiers to succeed (Chen et al., *PLOS One* 2019, DOI: [10.1371/journal.pone.0220113](https://doi.org/10.1371/journal.pone.0220113)). It is included for comparability with published literature, but MUV and WelQrate results should be weighted more heavily.

**Procedure:** For each target, use one active as query. Rank remaining actives + decoys by similarity. Compute per-target metrics and average across targets.

**Metrics:** AUC-ROC, BEDROC(α=20), EF₁%, LogAUC.

**Reporting:** Report results both on the full 102-target set (for literature comparability) and on the bias-reduced subset of ~47 targets identified by Lagarde et al. (*J. Chem. Inf. Model.* 2015, 55, 1297–1307).

---

### Experiment 4: Activity Cliff Analysis

**Objective:** Determine whether the electron density distance function better reflects potency changes associated with small structural modifications than topological methods.

**Data source:** MMP-cliffs from ChEMBL, as defined by the Bajorath group.

- Primary reference: Hu et al., *J. Chem. Inf. Model.* 2012, 52, 1138–1145. DOI: [10.1021/ci3001138](https://doi.org/10.1021/ci3001138)
- Activity cliff concept review: Stumpfe & Bajorath, *J. Med. Chem.* 2012, 55, 2932–2942. DOI: [10.1021/jm201706b](https://doi.org/10.1021/jm201706b)
- Data construction: Bajorath et al., *F1000Research* 2014, 3, 36. DOI: [10.12688/f1000research.3-36.v2](https://doi.org/10.12688/f1000research.3-36.v2)

**Activity cliff definition used:** Matched molecular pairs (MMPs) with transformation size ≤ 13 heavy atoms and |ΔpKi| ≥ 2 (i.e., ≥100-fold potency difference). Use Ki values only to ensure assay-independent comparisons.

**Data construction procedure:**

1. Extract from ChEMBL (latest release) all compounds with Ki values against human targets.
2. Generate MMPs using the Hussain–Rea algorithm (RDKit `rdFMCS` or `mmpdb` tool, [github.com/rdkit/mmpdb](https://github.com/rdkit/mmpdb)).
3. For each MMP, compute |ΔpKi|.
4. Label pairs with |ΔpKi| ≥ 2 as activity cliffs; pairs with |ΔpKi| < 0.5 as non-cliffs.
5. Sample ~5,000 cliff pairs and ~5,000 non-cliff pairs (matched for scaffold diversity).

**Measurements:**

- **SALI (Structure-Activity Landscape Index):** For each pair, SALI = |ΔpKi| / (1 − sim(A,B)), where sim is the similarity. Higher SALI = sharper cliff. A good similarity metric should yield high SALI for cliff pairs and low SALI for non-cliff pairs.
- **Cliff detection AUC:** Treat cliff vs. non-cliff classification as a binary task. For each method, rank pairs by SALI and compute AUC-ROC for distinguishing cliffs from non-cliffs. A method that assigns high similarity to cliff pairs (making high SALI) and low similarity to non-cliff pairs detects cliffs better.
- **Distance–potency correlation:** Spearman ρ between d(A,B) and |ΔpKi| across all MMP pairs. A strong correlation means the distance function tracks activity changes. Compute this overall and stratified by transformation type (R-group change, ring change, linker change).

**Expected outcomes:** Topological fingerprints (especially ECFP4 with Tanimoto) will yield the classic activity cliff signature: very high similarity (low distance) but large potency differences, resulting in high SALI but poor cliff-vs-non-cliff discrimination because high similarity is assigned indiscriminately. The electron density method should show greater distance for cliff pairs than fingerprints do (because the electronic perturbation from the structural change is captured), leading to better distance–potency correlation. The key test is whether ρ(distance, ΔpKi) is significantly higher for the electron density method.

---

### Experiment 5: Bioisostere Recognition

**Objective:** Test whether the electron density embedding places known bioisosteric replacements closer together than topological fingerprints do, since bioisosteres are defined by functional equivalence despite structural dissimilarity.

**Data sources:**

1. **SwissBioisostere database:** >25 million molecular replacements with bioactivity annotations.
   - Wirth et al., *Nucleic Acids Res.* 2013, 41, D1137–D1143. DOI: [10.1093/nar/gks1059](https://doi.org/10.1093/nar/gks1059)
   - Updated: Isert et al., *Nucleic Acids Res.* 2022, 50, D1382–D1390. DOI: [10.1093/nar/gkab1047](https://doi.org/10.1093/nar/gkab1047)
   - Web: [swissbioisostere.ch](http://www.swissbioisostere.ch)

2. **Classical bioisostere pairs (curated test set):** Construct a set of 50–100 bioisosteric fragment pairs from the literature. Key examples:
   - Carboxylic acid ↔ tetrazole
   - Phenyl ↔ thienyl
   - Amide ↔ 1,2,4-oxadiazole
   - Ester ↔ reversed amide
   - Pyridine ↔ pyrimidine
   - Fluorine ↔ hydrogen (in aromatic contexts)
   - Hydroxyl ↔ thiol
   - Reference: Meanwell, *J. Med. Chem.* 2011, 54, 2529–2591. DOI: [10.1021/jm1013693](https://doi.org/10.1021/jm1013693)

**Procedure:**

**Sub-experiment 5a — Fragment-level test:**

1. From SwissBioisostere, extract matched molecular pairs where the replacement is a known bioisosteric exchange and both compounds retain >80% activity (|ΔpActivity| < 0.3) against the same target. These represent confirmed functional equivalence.
2. Extract an equal number of control pairs: random MMPs where the replacement is not bioisosteric and activity is retained.
3. Extract an equal number of negative pairs: random MMPs with large activity changes (|ΔpActivity| > 1.5).
4. For each method, compute similarity for all three sets.

**Sub-experiment 5b — Whole-molecule test using classical pairs:**

1. For each classical bioisostere pair (e.g., COOH ↔ tetrazole), find 20 compounds in ChEMBL containing fragment A and 20 matched compounds containing fragment B (same scaffold context, same target, comparable activity).
2. Compute cross-set similarity: average similarity between fragment-A molecules and fragment-B molecules.
3. Compare to within-set similarity as baseline.

**Metrics:**

- **Bioisostere recognition rate:** Fraction of bioisosteric pairs where similarity > median similarity of random pairs. Report at multiple thresholds.
- **AUC for bioisostere detection:** Treat bioisostere pairs as positives, non-bioisosteric MMP pairs as negatives. Score by similarity. Compute AUC-ROC.
- **Cross-set / within-set similarity ratio (Sub-experiment 5b):** Ratio of mean cross-set similarity to mean within-set similarity. A ratio approaching 1.0 means the method treats bioisosteric molecules as equivalent.

**Expected outcomes:** Topological fingerprints should perform poorly because bioisosteres by definition have different connectivity (COOH and tetrazole have completely different atom patterns). ECFP4 Tanimoto for COOH↔tetrazole pairs will likely be < 0.3. 3D shape methods (ROCS, USR) should do better since bioisosteres often occupy similar volumes. The electron density method should excel because bioisosteres are characterized by similar electron distributions—this is arguably the most theoretically well-motivated test for the approach. If the electron density method does not substantially outperform fingerprints here, the fundamental hypothesis is weakened.

---

### Experiment 6: Scaffold Hopping

**Objective:** Evaluate the ability to identify molecules with different Bemis-Murcko scaffolds that are active against the same target.

**Data source and framework:** Use the Riniker & Landrum open-source benchmarking platform.

- Paper: Riniker & Landrum, *J. Cheminform.* 2013, 5, 26. DOI: [10.1186/1758-2946-5-26](https://doi.org/10.1186/1758-2946-5-26)
- Platform: [github.com/rdkit/benchmarking_platform](https://github.com/rdkit/benchmarking_platform)
- Contains compound sets from three data sources: MUV, DUD, ChEMBL, with 88 targets after filtering.
- Includes scaffold enrichment metrics specifically designed for scaffold hopping assessment.

**Procedure:**

1. For each of the 88 targets and each method:
   a. Select a query molecule from the active set.
   b. Rank all other compounds (actives + decoys) by similarity.
   c. At each rank threshold (1%, 5%, 10%), count both the total number of actives found and the number of distinct Bemis-Murcko scaffolds represented.

2. Repeat with 5 random query selections per target. Report mean ± standard error.

**Metrics:**

- **EF₅%** (enrichment factor at 5%): Standard VS performance.
- **Scaffold-EF₅%** (scaffold enrichment factor at 5%): Number of distinct active scaffolds found in top 5%, normalized by expected number under random ranking. This directly measures scaffold hopping ability.
- **Scaffold-EF / EF ratio:** Ratio of scaffold enrichment to raw enrichment. A ratio close to 1.0 means nearly every retrieved active represents a new scaffold (maximal diversity). Methods that retrieve analogs of the query will show Scaffold-EF/EF << 1.

**Additional scaffold hopping analysis:**

1. For a subset of 20 targets with ≥5 distinct active scaffolds, perform leave-one-scaffold-out retrieval: use all actives from one scaffold as queries and measure how well actives from other scaffolds are retrieved. This removes any within-scaffold advantage.
2. Compute recall@k% for scaffold-diverse actives only.

**Expected outcomes:** Topological fingerprints will show high EF but relatively low Scaffold-EF because they preferentially retrieve structural analogs. FCFP4 (feature-based) should hop scaffolds better than ECFP4. 3D methods (ROCS) should show better Scaffold-EF/EF ratios. The electron density method should show the best Scaffold-EF/EF ratio if it captures functional similarity independent of scaffold identity.

---

## 5. Statistical Analysis

### 5.1 Per-Experiment Statistics

- For all metrics computed across multiple targets/datasets, report **mean ± standard error** and **median** across targets.
- Use the **Wilcoxon signed-rank test** (paired, non-parametric) to compare each baseline to the electron density method across targets. Bonferroni-correct for the number of baselines tested.
- Report **effect sizes** (Cliff's delta or matched-pairs rank-biserial correlation) alongside p-values.

### 5.2 Aggregated Comparison

- Construct a **critical difference diagram** (Demšar, *JMLR* 2006, 7, 1–30) showing average rank of each method across all external benchmarks, with Nemenyi post-hoc test for pairwise significance.
- Present a **heatmap** showing method × benchmark performance, normalized to the best method per benchmark, to identify strengths and weaknesses.

### 5.3 Reporting Best Practices

- Never pick a single favorable metric. Report the full metric suite for each experiment.
- Report per-target results (or at least distributions) in supplementary material, not just averages.
- For WelQrate and DUD-E, compare scaffold split vs. random split results to quantify analog bias effects.
- Report computational cost (time per molecule for embedding computation) to contextualize any accuracy gains.

---

## 6. Computational Considerations

### 6.1 Conformer Generation

For all 3D methods (ROCS, USR, USRCAT, eSim, Coulomb matrix, SOAP, ACSF, and the electron density method itself), use a single shared conformer per molecule:

- Generate 20 starting geometries with RDKit ETKDGv3, `pruneRmsThresh=0.5`.
- Minimize all 20 with MMFF94, keep only the lowest-energy result.
- Every method then operates on this single conformer — no per-method conformer selection logic.

Rationale: a single minimum-energy conformer is the natural input for QM-derived representations (ElektroNN produces coefficients for a fixed geometry) and avoids inflated compute for methods that would otherwise iterate over an ensemble. If results suggest conformational flexibility matters, a best-of-ensemble follow-up can be added as a separate method variant for all 3D methods uniformly.

### 6.2 Software Stack

| Component | Tool |
|-----------|------|
| Cheminformatics core | RDKit (≥ 2024.03) |
| Conformer generation | RDKit ETKDG |
| MMP analysis | mmpdb or RDKit |
| Fingerprints (all types) | RDKit |
| Mol2vec | github.com/samoturk/mol2vec |
| Uni-Mol | github.com/deepmodeling/Uni-Mol |
| Chemprop | github.com/chemprop/chemprop |
| ROCS | OpenEye toolkit (commercial license required) |
| Statistical analysis | SciPy, scikit-learn |
| Visualization | matplotlib, seaborn |

### 6.3 Reproducibility

- Fix all random seeds.
- Publish all SMILES lists, embedding vectors, and distance matrices.
- Provide a single Snakemake or Nextflow workflow file to reproduce all experiments end-to-end.

---

## 7. Summary Table of Experiments

| # | Experiment | Type | Datasets | Key Metric(s) | What It Tests |
|---|-----------|------|----------|---------------|---------------|
| 1 | Homologous series | Internal | Custom (5 series, ~56 mols) | Monotonicity ρ, smoothness σ, linearity R² | Embedding continuity and chemical sensibility |
| 2 | Functional group substitution | Internal | Custom (3 sets, ~32 mols) | Hammett correlation ρ, donor/acceptor clustering silhouette | Electronic effect sensitivity |
| 3a | WelQrate retrieval | External | 9 targets | LogAUC, BEDROC(20), EF₁%, DCG₁₀₀ | Virtual screening on curated data |
| 3b | MUV retrieval | External | 17 targets | AUC-ROC, BEDROC(20), EF₁% | Performance without analog bias |
| 3c | DUD-E retrieval | External | 102 targets | AUC-ROC, BEDROC(20), EF₁% | Literature-comparable VS benchmark |
| 4 | Activity cliff analysis | External | ~10,000 MMP pairs from ChEMBL | SALI, cliff detection AUC, distance–ΔpKi ρ | Sensitivity to potency-relevant changes |
| 5 | Bioisostere recognition | External | SwissBioisostere + classical pairs | Bioisostere detection AUC, cross/within ratio | Functional equivalence beyond topology |
| 6 | Scaffold hopping | External | 88 targets (Riniker & Landrum) | Scaffold-EF₅%, Scaffold-EF/EF ratio | Chemotype diversity in retrieval |

---

## 8. Decision Framework for Interpreting Results

The electron density method should be considered an improvement if **at least three** of the following criteria are met:

1. **Internal consistency:** Significantly better Hammett correlation (Exp. 2, Set C) than all topological fingerprints (p < 0.01).
2. **Bioisostere recognition:** Top-2 performer (among all methods) in bioisostere detection AUC (Exp. 5).
3. **Scaffold hopping superiority:** Significantly higher Scaffold-EF/EF ratio than ECFP4 on the Riniker-Landrum platform (Exp. 6, p < 0.05).
4. **Activity-based retrieval:** Non-inferior AUC-ROC on WelQrate (Exp. 3a) compared to the best fingerprint baseline, AND superior performance on MUV (Exp. 3b, p < 0.05).
5. **Activity cliff sensitivity:** Higher distance–ΔpKi Spearman ρ than all topological baselines (Exp. 4, p < 0.01).

If the method excels on internal tests and bioisostere/scaffold hopping but underperforms on general retrieval (WelQrate/DUD-E), this indicates a niche advantage for scaffold hopping applications rather than general-purpose superiority—which is still a valuable and publishable finding.

If the method underperforms on bioisostere recognition (Exp. 5), the core hypothesis (electron density captures functional similarity) is weakened regardless of other results, and the choice of basis functions or distance metric should be revisited before further benchmarking.
