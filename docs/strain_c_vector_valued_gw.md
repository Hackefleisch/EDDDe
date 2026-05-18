# Vector-Valued Gromov-Wasserstein for Molecular Electron Densities

## Implementation Guide

This document is the working reference for implementing a Gromov-Wasserstein-style distance between molecules whose atoms carry vector-valued electron-density features (spherical harmonic coefficients predicted by a neural network). It is written to be read top-to-bottom by both a human collaborator and Claude Code.

The approach is staged: we start with the most stripped-down version of the framework that's still meaningful, validate it works at all, then progressively add the physical structure (geometry, rotation handling, basis-function awareness) one piece at a time. Each stage produces a usable distance and an ablation result.

---

## 0. Goal and Scientific Story

**Task:** Compute a distance $\Delta(A, B)$ between two molecules $A$ and $B$, given:
- Atomic positions for each molecule (3D coordinates).
- A vector of spherical-harmonic (SH) coefficients at each atom, predicted by a neural network, representing the local electron density. All molecules use the *same* basis set, so columns of the feature matrix correspond to the same basis functions across the dataset.

**Downstream use:** Find molecules with similar physical/functional properties. The hypothesis the paper is testing is that wavefunction-level information (SH coefficients of the density) carries information beyond purely geometric descriptors, and that a principled vector-valued GW distance captures this information in a way useful for property prediction.

**Implementation philosophy:** Build the simplest version that exercises the framework's machinery, measure it, then add physical structure piece by piece. Each rung up the staircase is a publishable ablation: it tells you which pieces of the mathematical machinery are doing real work for the task.

---

## 1. Mathematical Setup

### 1.1 The objects we're comparing: mvm-spaces

A **metric-vector-measure space (mvm-space)** is a triple $(X, d_X, \mu)$ where:

- $X$ is a finite set of points (the atoms).
- $d_X$ is a distance function on $X$. In the simplest case scalar-valued: $d_X(x_i, x_j) \in \mathbb{R}_{\geq 0}$. In the richer tensor-valued case: $d_X: X \times X \to V \otimes V$.
- $\mu$ is a *vector-valued* measure: at each atom $x_j$ we have a vector $\vec{a}_j \in V \cong \mathbb{R}^n$ (the SH coefficients), and $\mu = \sum_j \vec{a}_j \, \delta_{x_j}$.

**Concrete representation (this is what lives in memory):**
- Positions: $\mathbb{R}^{N \times 3}$ array.
- Distance matrix: $\mathbb{R}^{N \times N}$ for scalar $d_X$.
- Feature matrix: $\mathbb{R}^{n \times N}$ matrix where column $j$ is $\vec{a}_j$ — the SH coefficients at atom $j$.

**Normalization:** We require $\sum_j \rho^*(\vec{a}_j) = 1$ where $\rho^*$ is the dual norm (we use $\ell^2$ norm throughout). This makes $\mu$ a "vector-valued probability measure": collapsing each vector to its norm gives an honest probability distribution over atoms. Without normalization, larger molecules trivially have larger features and dominate.

**Intuition.** Each atom carries not just a mass but a vector of feature values. If we collapse each $\vec{a}_j$ to its scalar norm $\|\vec{a}_j\|$, we recover a classical mm-space — that's the trivial reduction. The interesting cases use the full vector.

### 1.2 Couplings — what a transport plan looks like

Given $\mu$ on $X$ and $\nu$ on $Y$, a **coupling** is a joint object $\pi$ describing how to transport the content of $\mu$ to $\nu$.

In **scalar OT**, $\pi$ is an $N \times M$ matrix where $\pi_{ij}$ = "amount moved from $x_i$ to $y_j$." Marginal constraints: rows sum to $\vec{a}_i$ (where $\vec{a}_i$ is now scalar); columns sum to $\vec{b}_j$.

In **vector-valued OT**, $\pi$ is a 4-index tensor $\pi^{ij}_{kl}$ where:

- $i$ = source atom index in $X$
- $j$ = target atom index in $Y$
- $k$ = source channel (which SH basis function in $X$)
- $l$ = target channel (which SH basis function in $Y$)

Concretely, $\pi \in \mathbb{R}^{N \times M \times n \times n}$. The interpretation:

> $\pi^{ij}_{kl}$ = amount of feature mode $k$ at atom $x_i$ that gets routed to feature mode $l$ at atom $y_j$.

The plan moves mass simultaneously across space *and* between feature channels. This is the new degree of freedom relative to scalar GW.

**Marginal constraints (Definition 2.1 of the paper):**

- Sum out $j$ and contract on $V$ (the source-channel side) to recover $\mu$.
- Sum out $i$ and contract on $W$ (the target-channel side) to recover $\nu$.

In coordinates with $V = W = \mathbb{R}^n$ and $\ell^2$ norm, "contraction" with the $\ell^2$ structure becomes the natural projection onto each channel. Practically, the implementation needs to enforce:

$$\sum_{j, l} \pi^{ij}_{kl} = a_{ki} \quad \forall i, k \qquad \text{and} \qquad \sum_{i, k} \pi^{ij}_{kl} = b_{lj} \quad \forall j, l$$

where $a_{ki}$ is the $k$-th component of feature vector at atom $i$ in molecule $A$, and similarly for $B$. Non-negativity of $\pi$ entries is also required (we are transporting nonnegative quantities — see Section 6 for a note on signed components).

### 1.3 Cost functions

The cost function $c$ assigns a cost to "moving content from a source location/channel to a target location/channel."

**Factored form (Example 3.1):** $c$ separates into a spatial part and a channel-mixing part:
$$c^{kl}(x_i, y_j) = d_{XY}(x_i, y_j) \cdot W_{kl}$$

where $W \in \mathbb{R}^{n \times n}$ is a positive-(semi)definite **channel-mixing matrix** encoding "how compatible is feature mode $k$ with feature mode $l$." This is the form we implement; it is computationally tractable and physically interpretable.

For GW (which we want, because we're comparing across spaces with no canonical $d_{XY}$), we don't have a meaningful $d_{XY}(x_i, y_j)$ between an atom in $A$ and an atom in $B$. Instead, we use the GW substitute below.

### 1.4 The GW-style distortion — why $\pi$ appears twice

In plain OT, you'd compute $\sum_{i,j,k,l} c^{kl}(x_i, y_j) \, \pi^{ij}_{kl}$. Each unit of mass takes one trip; one $\pi$, evaluated at one source-target pair.

GW does something different. It compares *pairwise distances within each molecule*:

$$\mathcal{L}\big((x_0, x_1), (y_0, y_1)\big) = \big| d_X(x_0, x_1) - d_Y(y_0, y_1) \big|^p$$

This says: "if I claim $x_0 \leftrightarrow y_0$ and $x_1 \leftrightarrow y_1$, the within-molecule distance for that pair should match. How wrong is the claim?" The matching across molecules is provided by $\pi$, and we apply it *to both ends of the pair*. The probability of the joint matching $(x_0, x_1) \leftrightarrow (y_0, y_1)$ is $\pi(x_0, y_0) \cdot \pi(x_1, y_1)$.

That's why $\pi$ appears twice in the integral — same plan, evaluated at both endpoints. This makes the optimization **quadratic in $\pi$** rather than linear, and it's the source of GW's computational difficulty (and the framework's NP-hardness in the worst case).

**The full vector-valued GW objective (the formula we implement, Eq. 12 of the paper):**

$$\Delta_p(A, B) = \inf_{\pi \in \Pi(\mu, \nu)} \left( \sum_{i,j,\bar{i},\bar{j}} \sum_{k,l,s,t} \big| c^{kl}(x_i, x_{\bar{i}}) - \tilde{c}^{st}(y_j, y_{\bar{j}}) \big|^p \, \pi^{ij}_{kl} \, \pi^{\bar{i}\bar{j}}_{st} \right)^{1/p}$$

where:

- $c^{kl}(x_i, x_{\bar{i}})$ = $(k, l)$-th component of the *vector-valued* distance between atoms $i$ and $\bar{i}$ in $A$.
- $\tilde{c}^{st}(y_j, y_{\bar{j}})$ = same for molecule $B$ at channels $(s, t)$.
- The optimization is over couplings $\pi$ satisfying the marginal constraints from §1.2.

When $V = W = \mathbb{R}$ (scalar features), this collapses to standard scalar GW.

**Reading the formula:** for every (atom-pair-in-$A$, atom-pair-in-$B$, channel-quadruple) tuple, compute the $L^p$ discrepancy between the within-molecule "distances" (cost function values) and weight by the product of two transport plan entries. Aggregate. Minimize over $\pi$. Take the $p$-th root.

---

## 2. Implementation Staircase

Each rung is a complete, runnable version of the method. Each rung produces a distance $\Delta$ that can be evaluated against the existing experiments. Rungs build on one another: changes are isolated to specific places in the code, and earlier rungs remain runnable as baselines for ablations.

### Rung 1: Topological geometry + scalar feature collapse

**What it does.** Reduces the framework to classical scalar GW with chemistry-derived features. This is the sanity check that the GW machinery does anything sensible at all on molecular data.

**Mathematical specification.**
- $d_X(x_i, x_j)$ = shortest-path distance along bonds (graph distance) in the molecular graph. Integer-valued.
- $\mu(x_j)$ = a *scalar* obtained by collapsing the SH coefficient vector: $\mu_j = \|\vec{a}_j\|_2$. Normalize so $\sum_j \mu_j = 1$.
- $\pi$ is a 2-index tensor $\pi^{ij} \in \mathbb{R}^{N \times M}$.
- Cost: $c(x_i, x_{\bar{i}}, y_j, y_{\bar{j}}) = |d_X(x_i, x_{\bar{i}}) - d_Y(y_j, y_{\bar{j}})|^p$.

**Implementation.** Use any existing scalar GW solver (POT library: `ot.gromov.gromov_wasserstein`). This is essentially a one-call wrapper; no new code required beyond data preparation.

**What this tells us.** Whether the dataset is amenable to GW-style analysis at all. If Rung 1 doesn't beat random on the property-prediction task, something is fundamentally wrong with the data pipeline and we shouldn't proceed.

### Rung 2: Topological geometry + full vector features (abstract channels)

**What it does.** First engagement of the vector-valued machinery. Channels are treated as abstract slots with no built-in physical meaning.

**Mathematical specification.**
- $d_X$ = bond-graph distance, scalar (same as Rung 1).
- $\mu(x_j) = \vec{a}_j \in \mathbb{R}^n$, normalized as $\sum_j \|\vec{a}_j\|_2 = 1$.
- $\pi$ is the full 4-index tensor in $\mathbb{R}^{N \times M \times n \times n}$.
- Channel-mixing: $W = I_n$ (identity). No physics in the channel structure.
- Cost: $c^{kl}(x_i, x_{\bar{i}}) = d_X(x_i, x_{\bar{i}}) \cdot \delta_{kl}$. Equivalently, the cost only "charges" for transporting within the same channel index; cross-channel transport is free in this rung. (Alternative: use $W = I$ to charge cross-channel transport at unit cost. Pick one and document which.)

**Implementation.** Now we leave existing libraries. Need to write:
1. Marginal constraint enforcement for the 4-index tensor.
2. The quadratic objective in $\pi$.
3. An iterative solver (see §3 for Sinkhorn-based approach).

**What this tells us.** Whether the vector-valued extension contributes anything beyond scalar GW. If Rung 2 does not improve on Rung 1, the paper's core contribution is not justified for this task and we need to investigate why before proceeding.

### Rung 3: Euclidean geometry + full vector features

**What it does.** Replaces topological distance with 3D Euclidean distance between atoms. Tests whether spatial geometry beats connectivity.

**Mathematical specification.** Same as Rung 2 except:
- $d_X(x_i, x_j) = \|x_i - x_j\|_2$ (Euclidean distance in 3D).

**Implementation impact.** Trivial — replace the bond-graph distance matrix with a Euclidean distance matrix. All other code identical to Rung 2.

**What this tells us.** Whether 3D conformational information matters for the target task. **Note:** this rung is *not* rotation-invariant in the feature sense — Euclidean distances are rotation-invariant, but if we rotate the molecule the SH coefficients in the features rotate via Wigner D-matrices and become numerically different. This rung will fail the rotation sanity check (§4); that failure is informative and motivates Rung 4.

### Rung 4: Euclidean geometry + rotation-invariant features

**What it does.** Restores rotation invariance by replacing raw SH coefficients with SO(3)-invariant summaries.

**Mathematical specification.** Same as Rung 3 except feature vector at each atom:
$$\vec{a}_j^{\text{inv}} = \big( p_0(j), p_1(j), \ldots, p_{L_{\max}}(j) \big)$$
where $p_\ell(j) = \sum_{m=-\ell}^{\ell} |c_\ell^m(j)|^2$ is the **power spectrum** at angular momentum $\ell$.

This is a strict information loss (we keep "how much density at each angular frequency" but discard "in which direction"), and it's invariant under SO(3) by unitarity of the Wigner matrices.

**Optional upgrade:** Use bispectrum coefficients (triple SH products contracted against Clebsch-Gordan coefficients) for richer invariant features. Implementations available via `e3nn`, `librascal`, or directly. The interface to the framework is identical — only the per-atom feature vector changes.

**What this tells us.** Whether rotation invariance materially affects performance on the target task. If the dataset happens to be aligned (all molecules in a canonical pose), this rung may not improve on Rung 3. If not, expect a substantial jump.

### Rung 5: Euclidean geometry + raw features + overlap-informed channel mixing

**What it does.** Uses the basis-function structure of the SH coefficients via the channel-mixing matrix $W$. This is the rung that justifies the "wavefunction-level" claim of the paper.

**Mathematical specification.** Same as Rung 3 (raw features), except:
- Channel-mixing matrix $W$ is set to the **basis function overlap matrix**:
$$W_{kl} = S_{kl} = \int \phi_k^*(\mathbf{r}) \, \phi_l(\mathbf{r}) \, d\mathbf{r}$$
where $\phi_k$ is the basis function for channel $k$. Since all molecules use the same basis set, $W$ is a single matrix shared across the dataset, computed once.
- Cost: $c^{kl}(x_i, x_{\bar{i}}) = d_X(x_i, x_{\bar{i}}) \cdot W_{kl}$.

**Important constraint:** $W$ should be symmetric and positive semi-definite (overlap matrices are by construction). For the cost to behave properly we may want to add a small ridge $W \to W + \epsilon I$ to ensure strict positive-definiteness.

**Implementation impact.** Compute $W$ once from the basis set definition (most quantum chemistry codes can output the overlap matrix; or compute analytically from Gaussian integrals). Plug into the existing solver.

**What this tells us.** Whether basis-function-aware channel mixing improves over abstract channels. If yes, the wavefunction-level information is genuinely contributing. **Combined with Rung 4 (invariant features)** this is also the rung where rotation handling and basis-function handling can both apply; see §3.5.

### Rung 6 (research-grade, optional): Bond-local frames

**What it does.** Replaces the scalar Euclidean distance with a *vector-valued* distance that encodes the relative orientation of SH coefficients along each bond. This is the most theoretically satisfying use of the paper's tensor-valued machinery.

**Mathematical specification.**
- $D_X(x_i, x_j)$ is now $V \otimes V$-valued, encoding "what does the SH content at $i$ look like relative to the bond direction toward $j$" (and symmetrically for $j$).
- Concretely: rotate the SH coefficients at atom $i$ into a local frame where $\hat{z}$ points along $\hat{r}_{ij} = (x_j - x_i)/\|x_j - x_i\|$, using the Wigner D-matrix that takes the lab frame to this bond frame. This expression is invariant under global rotation by construction: rotating the whole molecule rotates both the bond direction and the SH frames, so their relative expression is unchanged.
- Cost: significantly more complex; see Eq. 11–12 of the paper for the general tensor cost.

**Why it's last.** Designing this requires real care; the 4-index tensor $\pi$ now interacts with a tensor-valued distance, blowing up computational and memory cost. Tackle only if Rungs 1–5 are working and there's a specific deficiency that orientation handling could plausibly fix.

### Where metric learning fits

The channel-mixing matrix $W$ at Rung 5 (or any later rung) has free parameters. A natural metric-learning outer loop:

1. Parameterize $W = W_0 + \theta$, where $W_0$ is the overlap matrix (physics prior) and $\theta$ is a learnable PSD perturbation.
2. Inner loop: for each molecule pair, compute $\Delta(A, B; W)$ via the GW solver.
3. Outer loss: rank correlation between $\{\Delta(A_k, B_k)\}$ and ground-truth property similarities, or contrastive loss pulling chemically similar pairs together.
4. Backpropagate through the inner solver (requires either differentiable Sinkhorn or implicit differentiation at the GW optimum).

This is a substantial engineering project on top of the base framework. Expect to iterate. Worth doing at Rung 5+; not worth doing at earlier rungs because there are too few parameters or too much physics is missing.

---

## 3. Computational and Training Considerations

### 3.1 Solving the inner GW problem (per molecule pair)

The inner objective is quadratic and non-convex. Standard approach:

**Entropic regularization (Sinkhorn-flavored GW).** Add an entropy penalty $-\epsilon H(\pi)$ to make the problem strongly convex in $\pi$ at fixed cost, allowing Sinkhorn-style fixed-point iteration. Outer loop linearizes the quadratic cost around the current $\pi$ and solves a regularized linear OT subproblem. Algorithm sketch:

```
Initialize π (uniform or from a heuristic).
Repeat until convergence:
    1. Compute the linearized cost L(π) given current π:
       L_{ij,kl} = sum over (i', j', s, t) of |c^{kl}(x_i, x_i') - c̃^{st}(y_j, y_j')|^p · π^{i'j'}_{st}
    2. Solve the regularized linear OT problem with cost L and entropy ε:
       new_π = argmin <L, π> - ε H(π)  s.t. marginal constraints
       (Sinkhorn-style iterations)
    3. π ← new_π
```

For the vector-valued case, the linearized cost $L$ is itself a 4-index tensor, and the marginal constraints are the multi-index ones from §1.2. The Sinkhorn step generalizes — it's still alternating projection onto marginal-constraint sets — but the implementation needs to handle the 4-index structure.

**Memory.** The plan $\pi$ has $N \cdot M \cdot n^2$ entries. For $N = M = 50$ atoms and $n = 25$ basis functions per atom, that's $\sim 1.5 \cdot 10^6$ entries per molecule pair — fine. For larger systems memory becomes the binding constraint; plan accordingly.

**Time.** Each linearization step is $O(N^2 M^2 n^4)$ in the worst case (computing the full linearized cost). Sinkhorn iterations are cheaper, $O(NMn^2)$ per iteration. For a dataset with $K$ molecule pairs, total cost scales as $K \cdot \text{(cost per pair)}$. Expect minutes per pair at the rung-5 level for moderate-size molecules; this is a real bottleneck and motivates careful algorithmic work.

**Initialization.** Two reasonable choices:
- **Uniform:** $\pi^{ij}_{kl} = a_{ki} b_{lj} / (\text{total mass})$. Always feasible.
- **Warm start from the scalar GW solution:** solve the scalar GW problem (Rung 1 or 3 cost), then "lift" the resulting 2-index $\pi^{ij}$ to a 4-index plan by setting $\pi^{ij}_{kl} = \pi^{ij} \cdot \tilde{a}_{ki} \tilde{b}_{lj}$ (with appropriate normalization). Often gets to a good optimum faster.

### 3.2 The $\epsilon$ self-loop fix

Two molecules with identical atomic positions but different feature vectors at those positions get $\Delta = 0$ from the geometric cost, because $d_X(x, x) = 0$. This is a real failure mode. Fix from Section 4 of the paper:

- Add a small constant $\epsilon$ to all *diagonal* entries of the within-molecule distance matrix: $d_X(x_i, x_i) := \epsilon$ instead of $0$.
- Choose $\epsilon \leq \min_{i \neq j} d_X(x_i, x_j) / 2$ (or similar) so it doesn't dominate genuine inter-atomic distances.

In graph terms, this is a self-loop of weight $\epsilon$ at each vertex. Implement as a one-line modification of the distance matrix after computation. Document the choice of $\epsilon$.

### 3.3 Choice of $p$

$p$ is a hyperparameter of the distance. Standard choices:
- $p = 2$: Most common in the GW literature, gives a smooth optimization landscape, connects to least-squares intuition.
- $p = 1$: More robust to outlier distance pairs, less smooth. Useful if specific pairs of atoms have unreliable position information.
- $p = \infty$: Worst-case distortion, generally not used in practice.

Default to $p = 2$. Treat it as a hyperparameter to tune at the metric-learning stage.

### 3.4 Signed components in the coupling

The paper formally allows $\pi$ to have signed components when $\mu, \nu$ are full vector-valued measures (not just nonnegative). For our application — features are SH coefficients of a *density*, which is nonnegative as a function — the natural interpretation of mass transport requires $\pi \geq 0$ entrywise. However, individual SH coefficients $c_\ell^m$ can be negative even though the density they represent is nonnegative. This is a real subtlety:

- The norm $\|\vec{a}_j\|$ is always nonnegative regardless of coefficient signs, so the marginal constraints can be enforced.
- But the per-channel mass $a_{kj}$ can be negative if $c_k^m(j) < 0$, which makes the constraint $\sum_{j,l} \pi^{ij}_{kl} = a_{ki}$ require $\pi$ to have signed entries.

**Practical fixes (pick one and document):**
- **Square the SH coefficients per channel** before using them as features: $\tilde{a}_{kj} = (c_k^m(j))^2$. This guarantees nonnegativity. Information loss: phase information.
- **Use the power spectrum** (Rung 4 invariant features), which is nonnegative by construction. This is the cleanest fix and aligns with rotation invariance, but it's only available at Rung 4+.
- **Allow signed $\pi$ and reinterpret** the resulting $\Delta$ as a divergence rather than a transport cost. Theoretically valid but loses the OT interpretation; not recommended without strong reason.

Default recommendation: square the coefficients at Rungs 2-3 (where raw features are used), and switch to power spectrum at Rung 4+.

### 3.5 Combining Rung 4 and Rung 5 (rotation + basis function)

Rung 4 (invariant features) and Rung 5 (overlap-informed $W$) can both apply simultaneously. Two ways to combine:

- **Power spectrum features + identity $W$:** Rung 4 only. Simple, loses cross-$\ell$ information.
- **Power spectrum features + appropriate $W$:** The "channels" at Rung 4 are angular momenta $\ell$, not full $(n, \ell, m)$. The natural $W$ here is diagonal with weights per $\ell$, possibly learned. Cross-$\ell$ mixing is not allowed (would break rotation invariance).
- **Bispectrum features + appropriate $W$:** Richer invariant features with more channel structure.

For ablation purposes, run both Rung 4 and Rung 5 independently, then their combination. Each tells you something different.

### 3.6 Computing $W$ from the basis set

Given the basis set definition (Gaussian or Slater type orbitals, with normalization constants and exponents), the overlap matrix can be computed analytically. Standard tools:
- `pyscf.gto.intor("int1e_ovlp")` — returns the AO overlap matrix.
- Most DFT/quantum chemistry codes can dump the overlap matrix as part of standard output.

The matrix is symmetric, positive semi-definite, and can be quite ill-conditioned if the basis is large. Apply ridge regularization $W \to W + \epsilon I$ if the smallest eigenvalue is below numerical tolerance.

---

## 4. Failure Modes and Sanity Checks

This section lists known failure modes of GW-style methods and the specific failure modes of this vector-valued extension. Each item should be checked against the existing experimental suite; missing checks should be added.

### 4.1 Invariance failures (silent correctness bugs)

**Rotation invariance check** (mandatory at Rung 4+):
- Take a molecule $A$ from the dataset.
- Apply a random rotation $R \in SO(3)$ to its atomic positions and (correctly) to its SH coefficient features via Wigner D-matrices.
- Compute $\Delta(A, R \cdot A)$. Should be $\approx 0$ up to numerical noise (say, $< 10^{-3}$).
- **Failure indicates a bug** in the feature pipeline (most likely: SH features were rotated incorrectly, or the rotation invariance came from collapsing too much information).

**Permutation invariance check** (mandatory at all rungs):
- Take a molecule $A$. Randomly permute its atom indices (and the corresponding columns of the feature matrix).
- Compute $\Delta(A, A_{\text{permuted}})$. Should be $\approx 0$.
- **Failure indicates** the GW solver is finding a local optimum corresponding to the identity permutation rather than the true optimum, or there's a bug in how indices are tracked.

**Identity check:**
- $\Delta(A, A) \approx 0$ for every molecule. (At Rung 6 this can be subtle if bond-local frames are involved; the framework should still give 0 for identical inputs.)

### 4.2 Optimization failures

**Local minima:** GW is non-convex. Different initializations of $\pi$ can give different $\Delta$ values. Mitigation:
- Run with multiple random initializations and take the minimum.
- Use the warm-start strategy (solve scalar GW first, lift to vector GW).
- Increase entropic regularization and anneal: start with high $\epsilon$ (smooth, unique minimum), decrease over iterations.

**Convergence diagnostics:** Track the objective value across iterations. If it's not monotonically decreasing under entropic GW, there's a bug. If it plateaus far from the constraint set (marginal violations), increase iterations or check the Sinkhorn step.

**Sinkhorn divergence:** With small $\epsilon$, Sinkhorn iterations can diverge numerically (overflow in exp). Use log-domain Sinkhorn implementations.

### 4.3 Self-loop / zero-distance failures

**Identical positions, different features:** If two atoms in two molecules have identical positions but different SH coefficients, the spatial cost vanishes and the framework can't tell them apart. Test: construct synthetic pairs with this property and verify $\Delta > 0$ when the $\epsilon$ self-loop fix is applied. Without the fix, $\Delta = 0$ — that's the bug.

**Test for monotonicity in $\epsilon$:** $\Delta$ should be roughly continuous in the choice of self-loop weight $\epsilon$. Discontinuous jumps suggest the optimization is finding qualitatively different solutions for different $\epsilon$ — investigate.

### 4.4 Normalization failures

**Mass conservation:** After computing $\pi$, verify the marginal constraints are satisfied to numerical tolerance (say, $< 10^{-5}$). Violation indicates Sinkhorn didn't converge or the constraint enforcement is buggy.

**Normalization of features:** Each molecule's $\sum_j \|\vec{a}_j\| = 1$ should hold before the solver is called. Add an assertion. Failure here means the data preprocessing is broken.

**Sign issues with raw SH coefficients:** Per §3.4. Test by deliberately introducing a molecule whose SH coefficients sum to a negative value in some channel — current implementation should either reject it or apply the sign-handling fix consistently.

### 4.5 Failure modes specific to property prediction

**Distance-property correlation breakdown:** If $\Delta$ doesn't correlate with property differences on a held-out test set, candidate causes:
- The feature extraction (NN-predicted SH coefficients) doesn't capture properties well — test by replacing features with known-good descriptors (SOAP, Coulomb matrix) and re-running. If those work and ours doesn't, the issue is upstream of the framework.
- Wrong distance type for the task (e.g., using power spectrum when angular information matters for the property).
- Insufficient ablation; rerun all rungs and look for the rung where correlation appears or disappears.

**Outlier sensitivity:** A few molecule pairs with unusually large $\Delta$ may dominate downstream analysis. Check the distribution of pairwise distances; consider using $p = 1$ if outliers are problematic.

**Dataset shift between training and test:** If the NN was trained on one chemical space and the test molecules are out-of-distribution, the SH coefficient predictions are unreliable, and any framework on top of them inherits that unreliability. Verify the NN's prediction quality on the test set before blaming the GW framework.

### 4.6 Computational failures

**Out-of-memory at large $N$:** The 4-index tensor $\pi$ scales poorly. For molecules with $N > 100$ atoms and $n > 50$ basis functions per atom, sparse representations or low-rank factorizations of $\pi$ may be necessary. Test: try the largest-largest pair in your dataset; if it OOMs, set a maximum-molecule-size threshold.

**Wall-clock time per pair:** Track the per-pair runtime. If it grows superlinearly in molecule size beyond expectations, there's likely a quadratic-in-something inefficiency in the inner loop.

### 4.7 Metric learning failure modes (when applicable)

**Overfitting:** With a learnable $W$, the metric can fit noise in the training similarities. Standard fix: held-out validation set, early stopping, regularization of $\theta$ (penalize deviation from $W_0 = $ overlap matrix).

**Gradient instabilities:** Differentiating through Sinkhorn can be numerically unstable, especially at low $\epsilon$. Use implicit differentiation at the optimum, or differentiate through a fixed number of Sinkhorn iterations with moderate $\epsilon$.

**Negative-eigenvalue $W$:** If the metric-learning gradient pushes $W$ outside the PSD cone, the resulting "distance" is ill-defined. Constrain $W$ to be PSD via Cholesky parameterization: $W = LL^T$ where $L$ is the learnable lower-triangular matrix.

---

## 5. Open Questions and Research Directions

Things the paper does not resolve, that we will encounter in implementation:

1. **Is $\Delta$ a true metric?** The paper notes that the symmetry, identity-of-indiscernibles, and triangle inequality properties don't extend automatically to the vector-valued case. Empirically, we should test on triples $(A, B, C)$ whether $\Delta(A, C) \leq \Delta(A, B) + \Delta(B, C)$ holds. If not, $\Delta$ is technically a pre-metric or divergence, which is fine for many ML tasks but matters for theoretical claims.

2. **Connection to random walks (Section 5 of the paper).** Empty in the current draft. May give an alternative computational route via spectral methods. Worth keeping in mind but not pursuing until the main pipeline works.

3. **Tractability for large datasets.** The paper flags this as an open challenge. Sinkhorn-style entropic regularization is the obvious first attack; low-rank factorization and stochastic methods are next. For $K^2$ pairwise distances on a dataset of size $K$, even $O(\text{seconds})$ per pair becomes prohibitive at $K \sim 10^4$. Caching and approximate methods may be required.

4. **Best parameterization of learnable $W$ at Rung 5+.** Diagonal? Block-diagonal in $\ell$? Full PSD? Each has different expressivity and different optimization landscapes.

---

## 6. Quick Reference: Implementation Order

In rough priority:

1. Data pipeline: load molecules with positions, bonds, SH coefficient features. Verify normalization and sign conventions. Compute basis-set overlap matrix once.
2. Rung 1 implementation via POT library. Validate against existing experiments. Sanity-check permutation invariance.
3. Vector-valued GW solver (the core new code). Test on small synthetic mvm-spaces with known answers (e.g., two copies of the same molecule should give 0).
4. Rung 2 (vector features, abstract channels). Compare to Rung 1.
5. Rung 3 (Euclidean geometry). Compare to Rung 2.
6. Rung 4 (invariant features). Run the rotation sanity check; this should pass at Rung 4 and fail at Rung 3.
7. Rung 5 (overlap-informed $W$). Compare to Rung 3 (same geometry, different channel structure) and Rung 4 (same rotation handling, different channel structure).
8. (Optional) Metric learning at Rung 5.
9. (Optional, research) Rung 6 with bond-local frames.

Each step produces an ablation result. The story the paper will tell is: "here's a stripped-down baseline; here's what each piece of physical structure adds; here's the combination that performs best."

---

## Glossary

**mm-space (metric-measure space):** $(X, d, \mu)$ where $\mu$ is a scalar (probability) measure.

**mvm-space (metric-vector-measure space):** Same, but $\mu$ is vector-valued. Each point carries a vector of features.

**Coupling / transport plan:** A joint distribution describing how to rearrange one (vector-valued) measure into another. In the vector-valued case, a 4-index tensor.

**Marginals:** The "summed-out" versions of a coupling. Constraints requiring marginals to recover the original measures define what a valid coupling is.

**Gromov-Wasserstein (GW):** A way to compare metric spaces by matching pairwise distances within each space, rather than comparing points directly across spaces. Quadratic in the transport plan.

**Channel-mixing matrix $W$:** Encodes "how compatible is feature mode $k$ with feature mode $l$." In our setting, taken to be the basis-function overlap matrix at Rung 5.

**Sinkhorn iteration:** Fast iterative algorithm for entropic-regularized OT. Solves the linear OT subproblem inside the GW outer loop.

**SO(3)-invariance:** The property that a quantity doesn't change when the molecule is rotated in 3D. Required for our task because two molecules differing only by orientation have identical physical properties.

**Wigner D-matrix:** The matrix $D^{(\ell)}(R)$ describing how SH coefficients at angular momentum $\ell$ transform under a rotation $R$. Unitary; its determinant is 1.

**Power spectrum, bispectrum:** SO(3)-invariant summaries of SH coefficients. Power spectrum is the squared norm per $\ell$; bispectrum involves triple products.

**SOAP (Smooth Overlap of Atomic Positions):** Standard rotation-invariant atomic environment descriptor in computational chemistry. Available as a drop-in replacement or comparison baseline for our features.
