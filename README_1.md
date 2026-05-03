# Equivariant CNNs on Homogeneous Spaces
### Mathematical Theory and PyTorch Implementation

> **Henry Fallet** — PhD in Mathematics (Algebra) · ENS Paris-Saclay MVA Candidate  
> Based on Cohen, Geiger & Weiler, *NeurIPS 2019* · Weiler & Cesa, *NeurIPS 2019* · Cohen & Welling, *ICLR 2017*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-math.RT-b31b1b)](https://arxiv.org/abs/2205.06185)

---

## Overview

This project bridges **pure representation theory** and **geometric deep learning**. The central question: *what is the most general linear layer that commutes with a group action on structured data?*

The answer — established in Cohen et al. (2019) — is that every equivariant linear map is a **cross-correlation with a bi-equivariant kernel**. This forces the architecture of CNNs from first principles, rather than imposing it as an inductive bias.

The project contains:

- 📄 **[Mathematical exposition (PDF)](math/main.pdf)** — a self-contained proof-level reconstruction of the theory, from principal bundles to kernel constraint theorems
- 🌐 **[Interactive visualization](https://YOUR_USERNAME.github.io/equivariant-cnns/)** — spherical CNN on SO(3), rendered in the browser
- 💻 **[PyTorch implementation](code/)** — four progressive steps, from a rotation-sensitive baseline to a steerable SO(2) network

---

## Mathematical Framework

> Full proofs and LaTeX source are in [`math/`](math/). This section summarizes the key ideas.

### 1 · The Homogeneous Space Setup

Fix a locally compact Lie group $G$ and a closed subgroup $H \leq G$. The **homogeneous space** $G/H$ is the space on which data lives:

$$\mathbb{R}^2 \cong SE(2)/SO(2), \qquad S^2 \cong SO(3)/SO(2), \qquad SO(3) \cong SO(3)/\{e\}.$$

The group $G$ acts on $G/H$ by left multiplication: $g \cdot (kH) = (gk)H$.

### 2 · Principal Bundle Structure

$G$ is an $H$-**principal bundle** over $G/H$ via the quotient map $p : G \to G/H,\ g \mapsto gH$. The right action of $H$ on $G$ (by $g \cdot h = gh$) is free, proper, and fiber-transitive.

A local section $s : U \subseteq G/H \to G$ trivializes the bundle over $U$. The resulting **cocycle**

$$\mathfrak{h} : G/H \times G \to H, \qquad \mathfrak{h}(x, g) = s(g^{-1}x)^{-1}\, g^{-1}\, s(x)$$

measures the frame twist induced by the transformation $g$. It satisfies the cocycle identity $\mathfrak{h}(x, g_1 g_2) = \mathfrak{h}(g_2^{-1}x,\, g_1)\,\mathfrak{h}(x, g_2)$ and plays a central role in the induced representation.

### 3 · Associated Vector Bundle and Feature Fields

Given a representation $\rho : H \to GL(V)$, the **associated vector bundle** is

$$\mathcal{A} = G \times_H V = (G \times V)\,/\!\sim, \qquad (g, v) \sim (gh,\, \rho(h^{-1})v).$$

Its sections — the **feature maps** — are carried in two equivalent encodings:

| Encoding | Domain | Constraint | Use |
|---|---|---|---|
| **Mackey functions** | $f : G \to V$ | $f(gh) = \rho(h^{-1})f(g)$ | Theory, equivariance proofs |
| **Local sections** | $f : U \subseteq G/H \to V$ | (none) | Implementation |

The **lifting isomorphism** $\Lambda : f \mapsto [x \mapsto f(s(x))]$ connects the two. Mackey functions make algebraic computations (group action, convolution) transparent; local sections are what one actually stores in memory.

### 4 · Induced Representation

The group $G$ acts on spaces of sections via the **induced representation** $\pi = \operatorname{Ind}_H^G \rho$. The two realizations are:

$$[\pi_G(g) f](k) = f(g^{-1}k) \qquad \text{(Mackey)}$$

$$[\pi_C(g) f](x) = \rho\!\left(\mathfrak{h}(x, g^{-1})^{-1}\right) f(g^{-1}x) \qquad \text{(local sections)}$$

The Mackey form is a pure left-translation — proving equivariance is immediate. The local-sections form separates the geometric transport on the base from the fiber twist via the cocycle.

### 5 · Equivariant Linear Maps = Cross-Correlations

Let $\mathcal{F}_i$ be spaces of feature fields transforming under $\pi_i = \operatorname{Ind}_{H_i}^G \rho_i$. A $G$-equivariant linear map $\Phi : \mathcal{F}_1 \to \mathcal{F}_2$ must satisfy $\Phi \circ \pi_1(g) = \pi_2(g) \circ \Phi$ for all $g \in G$.

**Theorem 3.1** (Cohen–Geiger–Weiler): Every such $\Phi$ is a cross-correlation

$$[\Phi f](g) = \int_G \kappa(g^{-1}k)\, f(k)\, dk$$

with a kernel $\kappa \in \mathcal{K}_G$ satisfying the **bi-equivariance constraint**:

$$\kappa(h_2\, g\, h_1^{-1}) = \rho_2(h_2)\,\kappa(g)\,\rho_1(h_1), \qquad \forall h_i \in H_i,\ g \in G.$$

Three equivalent kernel spaces reduce the parameter count progressively:

$$\mathcal{K}_G \xrightarrow{\text{eliminate fiber}_1} \mathcal{K}_C \xrightarrow{\text{eliminate }H_2\text{-orbits}} \mathcal{K}_D$$

where $\mathcal{K}_D$ lives on the **double coset space** $H_2 \backslash G / H_1$ with a pointwise intertwining condition. This is the **steerability constraint** on the kernel.

---

## Implementation

The code is organized as four progressive steps on MNIST rotations.

### Step 1 — Baseline CNN (rotation sensitivity)

`code/step1_baseline.py`

A standard LeNet-style CNN trained on MNIST. Evaluated under arbitrary rotations to **document the failure mode**: accuracy drops from ~99% on aligned data to ~45% on rotated inputs. This motivates equivariance by construction.

### Step 2 — Group Equivariant CNN on $\mathbb{Z}_4$

`code/step2_z4_gcnn.py`

A from-scratch implementation of Cohen & Welling (2016). Feature maps are functions $f : \mathbb{Z}_4 \to \mathbb{R}^{C \times H \times W}$ transforming under the regular representation. The group convolution is:

$$[f \star \kappa](g) = \sum_{h \in \mathbb{Z}_4} \kappa(g^{-1}h)\, f(h)$$

Key design choices:
- Lifting layer: $\mathbb{R}^2 \to \mathbb{Z}_4$ (standard convolution replicated over 4 rotations)
- Group layers: convolution over the full group $\mathbb{Z}_4 \ltimes \mathbb{Z}^2$
- Pooling: group-invariant global pooling before the classifier

Result: accuracy stable under 90°-rotation steps. Accuracy gap vs. baseline: **+47 pp** on rotated test set.

### Step 3 — Steerable CNN with Continuous SO(2)

`code/step3_steerable.py`

Implements the kernel constraint for continuous $G = SE(2)$, $H = SO(2)$. Kernels are expanded in a **Fourier basis** over the fiber:

$$\kappa(r, \theta) = \sum_{j} a_j\, \psi_j(r)\, e^{ij\theta}$$

where $\psi_j$ are radial basis functions and the coefficients $a_j$ are the learned parameters. The steerability condition $\kappa(R_\alpha \cdot) = \rho_2(R_\alpha)\,\kappa(\cdot)\,\rho_1(R_\alpha)^{-1}$ is enforced analytically by construction.

### Step 4 — Spherical CNN on SO(3)

`code/spherical_cnn_so3.ipynb` · [Interactive demo](https://YOUR_USERNAME.github.io/equivariant-cnns/)

Extension to $G = SO(3)$, $H = SO(2)$, base space $S^2$. Spherical harmonics $Y_l^m$ replace circular harmonics. The Clebsch–Gordan decomposition replaces pointwise nonlinearities. Demonstrated on point-cloud classification.

---

## Results Summary

| Model | Architecture | MNIST-rot accuracy |
|---|---|---|
| Baseline CNN | Standard LeNet | ~45% |
| $\mathbb{Z}_4$-equivariant CNN | G-conv from scratch | ~92% |
| Steerable SO(2) CNN | Fourier kernel basis | ~98% |

---

## Project Structure

```
equivariant-cnns/
├── README.md
├── math/
│   ├── main.tex          # LaTeX source (self-contained)
│   └── main.pdf          # Compiled PDF
├── code/
│   ├── step1_baseline.py
│   ├── step2_z4_gcnn.py
│   ├── step3_steerable.py
│   └── spherical_cnn_so3.ipynb
└── docs/
    └── index.html        # Interactive SO(3) visualization
```

---

## Mathematical Background

The theory developed here sits at the intersection of:

- **Representation theory**: induced representations $\operatorname{Ind}_H^G \rho$, Mackey's theorem, Frobenius reciprocity
- **Differential geometry**: principal fiber bundles, associated vector bundles, connection forms
- **Harmonic analysis**: Peter–Weyl theorem, Fourier analysis on compact groups, Clebsch–Gordan coefficients
- **Deep learning**: parameter sharing, equivariance as an architectural constraint

For a detailed treatment see the [mathematical exposition](math/main.pdf).

---

## References

```bibtex
@inproceedings{cohen2019general,
  title     = {A General Theory of Equivariant {CNN}s on Homogeneous Spaces},
  author    = {Cohen, Taco S. and Geiger, Mario and Weiler, Maurice},
  booktitle = {NeurIPS},
  year      = {2019}
}

@inproceedings{weiler2019general,
  title     = {General {E(2)}-Equivariant Steerable {CNN}s},
  author    = {Weiler, Maurice and Cesa, Gabriele},
  booktitle = {NeurIPS},
  year      = {2019}
}

@inproceedings{cohen2016steerable,
  title     = {Steerable {CNN}s},
  author    = {Cohen, Taco S. and Welling, Max},
  booktitle = {ICLR},
  year      = {2017}
}
```

---

## Author

**Henry Fallet**  
PhD in Mathematics — *Cherednik Algebras and Generalized KZ Functors* (UPJV, 2021)  
[Publication in *Comptes Rendus Mathématique*](https://comptes-rendus.academie-sciences.fr/mathematique/articles/10.5802/crmath.281/) · [arXiv preprint](https://arxiv.org/abs/2205.06185)  
📧 henryfallet@gmail.com
