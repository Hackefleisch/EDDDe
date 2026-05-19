# Strain D — Topological / persistence

Strain D of the MUT family. Canonical spec: [PROJECT_PLAN.md §3.2.5](../PROJECT_PLAN.md). Narrative rationale: [experimental_plan.md §2.6.5](../experimental_plan.md).

This file preserves the context that informed the strain's spec — collaborator code (since removed) and design decisions the plan tables cannot fit. The companion PDF [geo_topo_baseline.pdf](../geo_topo_baseline.pdf) sketched the broader "geometric and topological representations" direction; this document and the plan sections together replace it for Strain-D-specific decisions.

## 1. What the collaborator scripts did

Two scripts existed at the repo root before being removed: `umap_eucldist.py` ("barycenter + Euclidean + UMAP") and `umap_wassdist.py` ("persistence diagram + Wasserstein + UMAP"). Both were exploratory and labelled by the user as "from earlier iterations with less refined project goals" — not source-of-truth, but useful to nail down what the PDF's prose actually meant.

### 1.1 "Barycenter" (umap_eucldist.py)

The script defined the barycenter as a per-atom mean of the (n_atoms, 127) coefficient matrix:

```python
def compute_barycenter(molecule_features):
    return np.mean(molecule_features, axis=0)  # (n_atoms, 127) -> (127,)
```

Pairwise distances were Euclidean (`scipy.spatial.distance.cdist(..., metric='euclidean')`); UMAP with `metric='precomputed'` for 2D visualisation against `dipol_mom`.

**Decision: this is MUT-mean, not a distinct strain.** The PDF used "barycentric coordinates" in the geometric sense (weighted combinations of vertices with weights summing to 1) — a genuine concept — but the code computes a centroid in feature space. No separate `MUT-barycentric` variant is introduced; Strain A's `MUT-mean` (post-scalarisation per [PROJECT_PLAN.md §3.2.2](../PROJECT_PLAN.md)) already covers this recipe.

### 1.2 Persistence-diagram Wasserstein (umap_wassdist.py)

The script treated each molecule's (n_atoms, 127) feature matrix as a point cloud in 127-d feature space, ran Vietoris-Rips persistence via `ripser`, kept only H1, and compared molecules via `persim.wasserstein` on the H1 diagrams:

```python
from ripser import ripser
from persim import wasserstein

result = ripser(points, maxdim=1)                # points: (n_atoms, 127)
diag_h1 = result["dgms"][1]                       # H1 only — H0 dropped
diag_h1 = diag_h1[np.isfinite(diag_h1[:, 1])]    # strip infinite-death points

dist = wasserstein(dgm_i, dgm_j)                  # 1-Wasserstein on H1
```

NaN distances were imputed with the matrix max. Distance matrix fed into precomputed-distance UMAP, coloured by LUMO.

## 2. Audit issues with the as-implemented approach

Direct adoption of `umap_wassdist.py` would fail two of the §3.2.1 contract tests:

1. **Rotation leakage.** Euclidean distance in raw 127-d feature space is not rotation-invariant — the l>0 blocks of the ElektroNN features transform by Wigner D-matrices under molecular rotation, so the VR filtration shifts and the PD shifts with it. Two orientations of the same conformer get different persistence diagrams. Same root cause as the pre-refactor MUT-mean bug.
2. **H0 silently dropped.** The script discards `result["dgms"][0]`. H0 (connected components, equivalent to single-linkage clustering of the point cloud) typically carries most of the persistence signal for small drug-like molecules — H1 is frequently empty after stripping infinite-death entries, leaving the Wasserstein distance degenerate.

A third minor issue: NaN-to-max imputation papers over numerical failures that should be investigated rather than hidden.

## 3. Adopted variants

See [PROJECT_PLAN.md §3.2.5](../PROJECT_PLAN.md) for the canonical table. Briefly:

| ID | Filtration | Distance |
|----|-----------|----------|
| MUT-TDA-PD-featinv | VR on 39-d scalarised per-atom features (Strain A scalarisation) | Wasserstein on H0+H1 PDs |
| MUT-TDA-PD-pos | VR on 3D atom positions | Wasserstein on H0+H1 PDs |
| MUT-TDA-PD-poswfeat | VR on `(3D pos, λ · 39-d scalarised features)` stacked | Wasserstein on H0+H1 PDs |
| MUT-TDA-PL-featinv | as PD-featinv | L1 on H0+H1 persistence landscapes |
| MUT-TDA-PL-pos | as PD-pos | L1 on H0+H1 persistence landscapes |

The collaborator's exact recipe (VR on raw 127-d features + H1-only) is **dropped** rather than kept as a labelled-broken reference — same logic that replaced raw 127-d MUT-mean with the scalarised version (Strain A §3.2.2).

## 4. Implementation decisions left open

The plan tables don't fix these. Recording defaults and alternatives here so the implementation PR doesn't re-derive them.

- **Wasserstein order.** Default `p=1` (matches collaborator code; less sensitive to long-lifetime outlier features). `persim.wasserstein` accepts an order argument; switch to `p=2` if results suggest the smooth norm fits the task better.
- **H0+H1 aggregation for PD variants.** Default: sum of per-dimension Wasserstein distances. Alternative: report per-dim separately (creates per-dim ablations). Sum is the simpler default; switch only if a dimension carries clearly distinct signal.
- **Persistence landscape parameters.** `persim.PersistenceLandscaper` defaults (`num_landscapes=5`, `resolution=100`) are a reasonable starting point. Concatenate H0 and H1 landscape vectors, then L1.
- **Empty-diagram handling.** If H1 is empty (common at low atom counts), its contribution to the summed Wasserstein is 0 — do not impute. Empty H0 cannot occur for non-empty point clouds.
- **λ for the hybrid filtration.** Tuned on a held-out training split (fairness rule). Initial sweep `λ ∈ {0.1, 0.5, 1.0, 2.0, 5.0}`; pick by held-out ranking correlation against the chosen training-task target.
- **`maxdim` for ripser.** `maxdim=1` (H0 + H1). H2 is rare for drug-like molecules and `ripser`'s cost grows quickly with homology dimension; defer.
- **Infinite-death points.** Drop (matches collaborator). Lifetime is undefined for the global component.
- **Point-cloud size limit.** `ripser` is `O(n²)` in memory; for 39-d scalarised features × ≤50 atoms, well within limits. For position-only filtrations (3 d × ≤50 atoms), trivially fast.

## 5. Dependencies

Add to `pyproject.toml` when the strain is implemented:
- `ripser` — Vietoris-Rips persistence.
- `persim` — PD Wasserstein, persistence landscapes.

Both are pip-installable and have no special build requirements on Linux.
