# Strain B — Equivariant central-atom pooling (e3nn)

Strain B of the MUT family. Canonical spec: [PROJECT_PLAN.md §3.2.3](../PROJECT_PLAN.md). Narrative rationale: [experimental_plan.md §2.6.3](../experimental_plan.md).

This file locks down implementation details that don't fit in the plan tables and that are silent-failure-prone (SH conventions, centroid edge cases, output ordering). Without these pinned, a first implementer who guesses any of them differently will get a method that *looks* like it works but silently breaks SO(3) invariance — and that will not be caught unless the §3.2.1 rotation audit infrastructure is already in place.

## 1. The construction in concrete form

Inputs per molecule (from `Stage.ELEKTRONN_COEFFS` and `Stage.CONFORMERS`):
- `positions`: `(N, 3)` array of atom coordinates.
- `features`: `(N, 127)` array of per-atom irrep coefficients, laid out as concatenated blocks for `14x0e + 14x1o + 5x2e + 4x3o + 2x4e` (in this order — see §3 for why this ordering is load-bearing).

Algorithm:

```
1. centre c = pick_centre(positions, features)               # see §4
2. for each atom j:
       r_jc = positions[j] - c
       d_j  = ||r_jc||
       Ω_j  = r_jc / d_j                                     # undefined if d_j ≈ 0; see §4
3. for each l in [0, 1, 2, 3, 4]:
       sh_l[j] = spherical_harmonics(l, Ω_j)                 # shape (2l+1,)
4. for each radial channel k in [0, K):
       for each l, for each multiplicity m of irrep l:
           out[k, l, m] = sum_j R_k(d_j) · ⟨features[j, block(l, m)], sh_l[j]⟩
5. return flatten(out)                                       # canonical ordering, see §5
```

Step 4 is one numpy dot product per (atom, l, m, k). No `e3nn.o3.TensorProduct` layer is needed — the equivariant projection collapses to a contraction of two equally-transforming (2l+1)-vectors, which gives a scalar (l=0 irrep) by construction. e3nn is used only as the source-of-truth implementation of `spherical_harmonics(l, direction)` and as the reference for the convention check in §3.

## 2. Why this is rotation-invariant

For a rotation R ∈ SO(3):
- `positions → R · positions`, so `Ω_j → R · Ω_j`.
- `features[j, block(l, m)] → D^(l)(R) · features[j, block(l, m)]` (irrep features transform by the Wigner D-matrix).
- `sh_l[j] → D^(l)(R) · sh_l[j]` (spherical harmonics at the rotated direction transform by the same D-matrix).
- The dot product `⟨D^(l)(R) · x, D^(l)(R) · y⟩ = ⟨x, y⟩` because `D^(l)(R)` is orthogonal.

So step 4's per-atom contributions are invariant; the sum over atoms is permutation-invariant; the centre is translation-invariant by construction. **All three contract tests in [PROJECT_PLAN.md §3.2.1](../PROJECT_PLAN.md) are satisfied by the recipe — provided the convention check in §3 passes.**

## 3. SH convention pitfalls — the critical section

The dot product in step 4 only produces an invariant if ElektroNN's irrep features and e3nn's spherical harmonics use **the same convention**. Mismatches that have caused real silent failures elsewhere:

- **Real vs complex SHs.** Both ElektroNN and e3nn use real SHs, but the real-from-complex transform itself has sign conventions (Condon-Shortley phase included or not).
- **Ordering of `m` within an irrep `l`.** For `l=1`, e3nn orders components as `(y, z, x)` — *not* `(x, y, z)` (physics) or `(z, x, y)` (some chemistry codes). For higher `l`, the differences are less intuitive.
- **Normalisation.** e3nn supports `"component"` (default, each component has unit RMS over the sphere) or `"norm"` (each `l`-block has unit L2 norm). ElektroNN's coefficients implicitly assume one of these.

Without a verified match, the dot product is *some* linear combination of `(2l+1)` features that is not rotation-invariant.

**Verification recipe (run once, before any Strain B variant is implemented):**

```python
import e3nn.o3 as o3
import numpy as np

# 1. Pick a fixed reference molecule (any from the conformers cache).
mol, features_orig = load_reference_molecule()
positions_orig = mol.GetConformer().GetPositions()                 # (N, 3)

# 2. Pick a random rotation.
R = o3.rand_matrix().numpy()                                       # (3, 3)

# 3. Path A: rotate the molecule's positions, re-run ElektroNN.
positions_rot = positions_orig @ R.T
features_rerun = elektronn_predict(mol_with_positions(positions_rot))

# 4. Path B: keep the original ElektroNN output, apply e3nn's Wigner D-matrix.
irreps = o3.Irreps("14x0e + 14x1o + 5x2e + 4x3o + 2x4e")
D = irreps.D_from_matrix(o3.matrix_to_angles(R)).numpy()           # (127, 127)
features_predicted = features_orig @ D.T

# 5. They should match within numerical noise.
assert np.allclose(features_rerun, features_predicted, atol=1e-3)
```

If the assertion fails, derive the per-`l` basis transform `T_l ∈ R^{(2l+1)×(2l+1)}` (almost always a permutation × diagonal-sign matrix) and apply once to the cached coefficients before any Strain B variant runs. Stash the transform in `eddde/methods/muts/elektronn_to_e3nn_basis.npy` with a sidecar JSON documenting its derivation, mirroring the pattern planned for the GW-R5 overlap matrix.

If the assertion passes, no transform is needed; document the verification result in the same JSON for reproducibility.

The same `D_from_matrix` trick is the right vehicle for the §3.2.1 rotation audit of any individual Strain B variant — see §8 below.

## 4. Centroid edge cases

**Atom at the centroid (`d_j ≈ 0`).** Direction `Ω_j` is undefined; `sh_l(Ω_j)` for `l > 0` cannot be evaluated. Handling:
- For `l = 0`: `sh_0` is a position-independent constant, so the contribution is well-defined and should be included.
- For `l > 0`: drop the atom from the sum (equivalently, set the contribution to zero).
- Threshold: `d_j < 1e-6 Å`. Smaller than any meaningful chemical distance; large enough to dodge float-precision corner cases.

This case is rare in practice (it requires an atom to land at the geometric centroid) but it's not impossible — water's oxygen lies almost at the centroid of the H–O–H triangle for some conformers.

**Density-weighted centroid (MUT-e3nn-densitycenter).** Centre `c = Σ_j w_j · positions[j] / Σ_j w_j` with `w_j = ‖features[j]‖₂` (full 127-d L2 norm). Edge cases:
- All-zero coefficients on every atom: fall back to the geometric centroid. Log a warning at registration; this should not happen on real ElektroNN output.
- A single atom dominates (e.g. a heavy atom with much larger coefficients): the centroid sits near that atom, and many other atoms have small `d_j`. The construction is still well-defined; no special handling.

**Hydrogen inclusion.** All atoms (including H) are included, matching the project-wide convention from CLAUDE.md ("Atom-index alignment" section). Filtering H would break the rotation audit if it's done inconsistently between paths.

## 5. Output flattening order — load-bearing for version stability

The embedding is a 1-D vector of length `K × 39`. The flattening order must be **fixed for all time** because changing it would silently invalidate every distance computed before the change without bumping any `version` string.

Canonical order: loop over `(radial_channel k, irrep_l, multiplicity_m_within_l)`, outermost to innermost:

```
out_flat[k * 39 + offset(l, m)]
where offset = {
    (0, 0..13): 0..13,                  # 14 scalars
    (1, 0..13): 14..27,                 # 14 l=1 multiplicities
    (2, 0..4):  28..32,                 # 5  l=2 multiplicities
    (3, 0..3):  33..36,                 # 4  l=3 multiplicities
    (4, 0..1):  37..38,                 # 2  l=4 multiplicities
}
```

If a refactor changes this ordering, bump the `version` string of every Strain B variant.

## 6. Radial basis specification

For `K = 8` Gaussians on `[0, 10] Å` (used by `MUT-e3nn-radial`, `MUT-e3nn-densitycenter`, `MUT-e3nn-cosine`):

- Centres: `np.linspace(0, 10, 8)` → `[0, 10/7, 20/7, ..., 10]`.
- Width: `σ = 10 / 7 ≈ 1.43 Å` (the grid spacing).
- Form: `R_k(d) = exp(-((d - centre_k) / σ)^2 / 2)`. No normalisation factor — the absolute scale of the embedding doesn't matter for Euclidean / cosine distances.
- Atoms with `d_j > 10 Å` get small but non-zero contributions to the upper-shell channels. Do not clip; the Gaussian's tail handles falloff smoothly.

The grid spans 0–10 Å because typical drug-like molecules have heavy-atom radii of 4–7 Å from the centroid; 10 Å covers larger fragments and crystal-packing-relevant separations.

For `K = 1` variants:
- `MUT-e3nn-uniform`: `R(d) = 1`.
- `MUT-e3nn-1overR`: `R(d) = 1 / (d + 0.1)`. The 0.1 Å offset prevents the d=0 singularity for atoms near the centroid and matches USR's regularisation convention.

## 7. Per-variant implementation notes

| Variant | Centre | K | R(d) | Distance |
|---|---|---|---|---|
| MUT-e3nn-uniform | geometric centroid | 1 | `1` | Euclidean |
| MUT-e3nn-1overR | geometric centroid | 1 | `1 / (d + 0.1)` | Euclidean |
| MUT-e3nn-radial | geometric centroid | 8 | Gaussians, `[0, 10] Å` | Euclidean |
| MUT-e3nn-densitycenter | `‖a_j‖₂`-weighted | 8 | Gaussians, `[0, 10] Å` | Euclidean |
| MUT-e3nn-cosine | geometric centroid | 8 | Gaussians, `[0, 10] Å` | Cosine |

Distance dispatch: all variants implement `distances()` (batched) via `scipy.spatial.distance.cdist`, mirroring `MUT-mean`'s pattern. The batched form is the right choice for these — no per-pair-only operation exists.

## 8. Rotation-audit recipe for this strain

Use the same trick from §3, lifted to embedding-level:

```python
R = o3.rand_matrix().numpy()
irreps = o3.Irreps("14x0e + 14x1o + 5x2e + 4x3o + 2x4e")
D = irreps.D_from_matrix(o3.matrix_to_angles(R)).numpy()

features_rot = features @ D.T
positions_rot = positions @ R.T

emb      = strain_b_variant.embed_one(features,     positions)
emb_rot  = strain_b_variant.embed_one(features_rot, positions_rot)

assert np.linalg.norm(emb - emb_rot) < 1e-3
```

This avoids re-running ElektroNN inside the audit, which makes it cheap enough to run on every variant at registration time on the §3.2.1 fixture.

## 9. Dependencies

Add to `pyproject.toml` when this strain is implemented:
- `e3nn` — for `spherical_harmonics`, `Irreps.D_from_matrix`, and as the source-of-truth convention reference.
- `torch` — transitive e3nn dependency; likely already present via the `elektronn` upstream.

All Strain B computations should run on CPU and return `numpy` arrays. e3nn returns `torch` tensors; convert immediately at the boundary. No GPU usage in this strain — the per-molecule work is too small to benefit and would complicate the embedding cache (which is pickle-on-disk numpy).
