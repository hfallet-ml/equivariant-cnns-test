# CNN équivariant SO(3) sur $S^2 = \mathrm{SO}(3)/\mathrm{SO}(2)$

**Référence :** Cohen & Weiler, NeurIPS 2019 — *A General Theory of Equivariant CNNs on Homogeneous Spaces*

**Pipeline :**
1. Point cloud 3D (ModelNet40, 2048 pts/forme)
2. Projection sur $S^2$ via noyau gaussien sphérique
3. CNN équivariant SO(3) avec `e3nn` (irréps de Wigner $D^l$)

**Installation :** `pip install e3nn h5py torch`

## Cellule 1 — Imports


```python
import os
import math
import urllib.request
import zipfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from e3nn import o3
from e3nn.nn import BatchNorm, Gate
import h5py

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Dispositif : {DEVICE}')
print(f'PyTorch    : {torch.__version__}')
import e3nn; print(f'e3nn       : {e3nn.__version__}')
```

    Dispositif : cpu
    PyTorch    : 2.11.0+cpu
    e3nn       : 0.6.0
    

## Cellule 2 — Grille sphérique et projection $\mathbb{R}^3 \to S^2$

**`fibonacci_sphere(N)`** : discrétise l'espace homogène $X = S^2 = \mathrm{SO}(3)/\mathrm{SO}(2)$ en $N$ points quasi-uniformes.

**`project_to_sphere`** : transforme un nuage de points 3D en signal scalaire $f : S^2 \to \mathbb{R}$ via un noyau gaussien sphérique :
$$f(x_i) = \sum_j \exp\!\left(\frac{\langle x_i, p_j \rangle - 1}{\sigma^2}\right)$$
C'est une section du fibré trivial sur $S^2$ — l'entrée `1x0e` du modèle (§2.3 du papier).


```python
def fibonacci_sphere(N: int) -> torch.Tensor:
    """
    N points quasi-uniformes sur S² ⊂ ℝ³ (spirale de Fibonacci).
    Discrétisation de l'espace homogène X = SO(3)/SO(2).
    Returns : (N, 3), vecteurs unitaires.
    """
    golden = (1 + math.sqrt(5)) / 2
    i      = torch.arange(N, dtype=torch.float32)
    alpha  = 2 * math.pi * i / golden          # longitude
    beta   = torch.acos(1 - 2 * (i + 0.5) / N) # colatitude
    return torch.stack([
        torch.sin(beta) * torch.cos(alpha),
        torch.sin(beta) * torch.sin(alpha),
        torch.cos(beta),
    ], dim=-1)


def project_to_sphere(
    points : np.ndarray,
    grid   : torch.Tensor,
    sigma  : float = 0.15,
) -> torch.Tensor:
    """
    Projette un nuage de N_pts points 3D sur la grille sphérique.

    Pipeline :
      1. Centrage + normalisation dans la boule unité
      2. Projection radiale sur S² : p ← p / ‖p‖
      3. Noyau gaussien sphérique : k(x_i, p_j) = exp((⟨x_i,p_j⟩-1)/σ²)
      4. Signal de densité : f(x_i) = Σ_j k(x_i, p_j), normalisé dans [0,1]

    Args :
        points : (N_pts, 3)  nuage de points (float32)
        grid   : (N_grid, 3) grille de Fibonacci
        sigma  : largeur du noyau (plus petit = plus localisé)
    Returns :
        signal : (N_grid, 1) dans [0, 1]
    """
    pts = torch.from_numpy(points).float()
    # 1. Centrage
    pts = pts - pts.mean(dim=0, keepdim=True)
    # 2. Normalisation dans la boule unité
    pts = pts / (pts.norm(dim=-1).max().clamp(min=1e-8))
    # 3. Projection sur S²
    pts = pts / pts.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    # 4. Noyau gaussien sphérique
    cos_d  = (grid @ pts.T).clamp(-1.0, 1.0)          # (N_grid, N_pts)
    kernel = torch.exp((cos_d - 1.0) / (sigma ** 2))  # (N_grid, N_pts)
    signal = kernel.sum(dim=-1, keepdim=True)          # (N_grid, 1)
    return signal / signal.max().clamp(min=1e-8)


def _random_rotation_matrix() -> np.ndarray:
    """Rotation aléatoire uniforme dans SO(3) (algorithme de Shoemake)."""
    M    = np.random.randn(3, 3).astype(np.float32)
    Q, R = np.linalg.qr(M)
    Q   *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

print('Fonctions utilitaires définies.')
```

    Fonctions utilitaires définies.
    

## Cellule 3 — Datasets

Deux datasets disponibles :
- **`ModelNet40Spherical`** : télécharge et charge ModelNet40 réel (≈435 Mo)
- **`SyntheticSphericalDataset`** : nuages aléatoires, aucun téléchargement requis


```python
MODELNET40_URL    = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
MODELNET40_CLASSES = [
    'airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car',
    'chair','cone','cup','curtain','desk','door','dresser','flower_pot',
    'glass_box','guitar','keyboard','lamp','laptop','mantel','monitor',
    'night_stand','person','piano','plant','radio','range_hood','sink',
    'sofa','stairs','stool','table','tent','toilet','tv_stand','vase',
    'wardrobe','xbox',
]


class ModelNet40Spherical(Dataset):
    """
    ModelNet40 projeté sur S².
    Chaque forme = signal f : S² → ℝ sur la grille de Fibonacci.
    """

    def __init__(self, root, split='train', N_grid=256,
                 sigma=0.15, download=True, augment=True):
        assert split in ('train', 'test')
        self.root    = root
        self.split   = split
        self.sigma   = sigma
        self.augment = augment and (split == 'train')
        self.grid    = fibonacci_sphere(N_grid)

        if download:
            self._download()
        self.points, self.labels = self._load_hdf5()
        print(f'ModelNet40 [{split}] : {len(self.labels)} formes, '
              f'grille {N_grid} pts, σ={sigma}')

    def _download(self):
        data_dir = os.path.join(self.root, 'modelnet40_ply_hdf5_2048')
        if os.path.isdir(data_dir):
            print(f'Données déjà présentes dans {data_dir}/')
            return
        os.makedirs(self.root, exist_ok=True)
        zip_path = os.path.join(self.root, 'modelnet40.zip')
        print(f'Téléchargement de ModelNet40 (~435 Mo)...')
        def progress(count, block, total):
            pct = min(count * block / total * 100, 100)
            print(f'\r  {pct:.1f}%', end='', flush=True)
        urllib.request.urlretrieve(MODELNET40_URL, zip_path, reporthook=progress)
        print()
        print('Extraction...')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(self.root)
        os.remove(zip_path)
        print('Téléchargement terminé.')

    def _load_hdf5(self):
        data_dir  = os.path.join(self.root, 'modelnet40_ply_hdf5_2048')
        list_file = os.path.join(data_dir, f'{self.split}_files.txt')
        with open(list_file) as f:
            h5_files = [
                os.path.join(data_dir, os.path.basename(l.strip()))
                for l in f
            ]
        all_pts, all_lbl = [], []
        for path in h5_files:
            with h5py.File(path, 'r') as hf:
                all_pts.append(hf['data'][:].astype(np.float32))
                all_lbl.append(hf['label'][:].astype(np.int64))
        points = np.concatenate(all_pts, axis=0)
        labels = np.concatenate(all_lbl, axis=0).squeeze()
        return points, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pts   = self.points[idx].copy()   # (2048, 3)
        label = int(self.labels[idx])
        if self.augment:
            pts = pts @ _random_rotation_matrix().T
        signal = project_to_sphere(pts, self.grid, self.sigma)
        return signal, torch.tensor(label, dtype=torch.long)


class SyntheticSphericalDataset(Dataset):
    """
    Dataset synthétique : nuages aléatoires projetés sur S².
    Pour tester le pipeline sans télécharger ModelNet40.
    """

    def __init__(self, N_samples=200, N_grid=256,
                 n_classes=40, N_pts=2048, sigma=0.15):
        self.N         = N_samples
        self.n_classes = n_classes
        self.N_pts     = N_pts
        self.sigma     = sigma
        self.grid      = fibonacci_sphere(N_grid)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        label  = idx % self.n_classes
        scales = np.ones(3, dtype=np.float32)
        scales[label % 3] *= 0.1 * (1 + label % 5)  # anisotropie par classe
        pts = (np.random.randn(self.N_pts, 3) * scales).astype(np.float32)
        signal = project_to_sphere(pts, self.grid, self.sigma)
        return signal, torch.tensor(label, dtype=torch.long)

print('Classes Dataset définies.')
```

    Classes Dataset définies.
    

## Cellule 4 — Modèle SphericalCNN

Architecture SO(3)-équivariante avec `e3nn` (Cohen & Weiler §3.1) :

| Bloc | Irréps entrée | Irréps sortie | Objet mathématique |
|------|--------------|---------------|--------------------|
| 1 | `1x0e` | `8x0e+4x1o+2x2e` | $\mathrm{Ind}_H^G \rho_0$ |
| 2 | `8x0e+4x1o+2x2e` | `16x0e+8x1o+4x2e` | représentation induite |
| 3 | `16x0e+8x1o+4x2e` | `32x0e+16x1o+8x2e` | représentation induite |
| Proj. | `32x0e+16x1o+8x2e` | `64x0e` | projection repr. triviale |

**`SO3ConvLayer`** : implémente $\sum_j \mathrm{mask}_{ij} \cdot \mathrm{TP}(f_j, Y^l(x_j))$, la corrélation croisée discrétisée du §3.1.

**`Gate`** : non-linéarité équivariante — les scalaires ($l=0$) modulent les canaux $l>0$ par une sigmoïde.


```python
class SO3ConvLayer(nn.Module):
    """
    Convolution équivariante SO(3) sur S².
    Implémente le noyau bi-équivariant κ ∈ K_G du papier (§3.1) :
        out_i = Σ_j mask[i,j] · TP(f_j, Y^l(x_j))
    où TP = produit tensoriel de Clebsch-Gordan.
    """

    def __init__(self, irreps_in, irreps_out, irreps_sh, points, cutoff=0.3):
        super().__init__()
        # Poids appris : scalaires w_{l1,l2,l3} des coefficients de CG
        self.tp = o3.FullyConnectedTensorProduct(irreps_in, irreps_sh, irreps_out)
        # Harmoniques sphériques Y^l(x_j) — irréps D^l de SO(3)
        sh   = o3.spherical_harmonics(irreps_sh, points, normalize=True)
        # Masque de voisinage : j ∈ N(i) ssi ⟨x_i, x_j⟩ > cutoff
        mask = (points @ points.T > cutoff).float()
        mask /= mask.sum(-1, keepdim=True).clamp(min=1)
        self.register_buffer('sh',   sh)    # (N, dim_sh) — fixe
        self.register_buffer('mask', mask)  # (N, N)      — fixe

    def forward(self, f):
        B, N, _ = f.shape
        sh_e    = self.sh.unsqueeze(0).expand(B, -1, -1)
        # TP(f_j, Y^l(x_j)) pour tout j en parallèle
        contrib = self.tp(
            f.reshape(B * N, -1),
            sh_e.reshape(B * N, -1),
        ).reshape(B, N, -1)
        # Agrégation sur le voisinage
        return torch.einsum('ij, bjd -> bid', self.mask, contrib)


def _build_gate(n_scalars, n_vectors, n_tensors2):
    """
    Non-linéarité Gate équivariante.
    relu sur les scalaires (l=0), sigmoid × canaux l>0.
    """
    irr_s = o3.Irreps(f'{n_scalars}x0e')
    irr_g = o3.Irreps(
        [(n_vectors, o3.Irrep('1o'))] +
        ([(n_tensors2, o3.Irrep('2e'))] if n_tensors2 > 0 else [])
    )
    n_gates = sum(m for m, _ in irr_g)
    return Gate(
        irr_s,                         [torch.relu],
        o3.Irreps(f'{n_gates}x0e'),    [torch.sigmoid],
        irr_g,
    )


class ConvBlock(nn.Module):
    """Bloc Conv → BatchNorm → Gate."""

    def __init__(self, irreps_in, irreps_sh,
                 n_scalars, n_vectors, n_tensors2, points, cutoff):
        super().__init__()
        self.gate = _build_gate(n_scalars, n_vectors, n_tensors2)
        self.conv = SO3ConvLayer(irreps_in, self.gate.irreps_in,
                                 irreps_sh, points, cutoff)
        self.bn         = BatchNorm(self.gate.irreps_in)
        self.irreps_out = self.gate.irreps_out

    def forward(self, x):
        B, N, _ = x.shape
        x = self.conv(x)
        x = self.bn(x.reshape(B * N, -1))
        return self.gate(x).reshape(B, N, -1)


class SphericalCNN(nn.Module):
    """
    CNN équivariant SO(3) sur S² = SO(3)/SO(2).

    Correspondance Cohen & Weiler 2019 :
        Irréps D^l de SO(3)      → 'lx0e', 'lx1o', 'lx2e'
        Représentation induite   → mélange d'irréps par couche
        Noyau bi-équivariant κ   → FullyConnectedTensorProduct
        ∫_{SO(3)} dg (Haar)      → .mean(dim=1) après proj. scalaire
    """

    def __init__(self, n_classes=40, N_points=256, lmax=3, cutoff=0.3):
        super().__init__()
        self.N   = N_points
        pts      = fibonacci_sphere(N_points)
        self.register_buffer('points', pts)
        irreps_sh = o3.Irreps.spherical_harmonics(lmax)

        self.bloc1 = ConvBlock(o3.Irreps('1x0e'),        irreps_sh,  8,  4, 2, pts, cutoff)
        self.bloc2 = ConvBlock(self.bloc1.irreps_out,    irreps_sh, 16,  8, 4, pts, cutoff)
        self.bloc3 = ConvBlock(self.bloc2.irreps_out,    irreps_sh, 32, 16, 8, pts, cutoff)

        C          = 64
        self.proj  = o3.Linear(self.bloc3.irreps_out, o3.Irreps(f'{C}x0e'))
        self.classifier = nn.Sequential(
            nn.LayerNorm(C),
            nn.Linear(C, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.bloc1(x)
        x = self.bloc2(x)
        x = self.bloc3(x)
        # Projection sur les scalaires (représentation triviale)
        x = self.proj(x.reshape(B * self.N, -1)).reshape(B, self.N, -1)
        # Intégrale de Haar discrète : moyenne sur les N points de S²
        x = x.mean(dim=1)        # (B, 64) — vecteur SO(3)-invariant
        return self.classifier(x)

print('Architecture définie.')
```

    Architecture définie.
    

## Cellule 5 — Instanciation

Choisir **un seul** des deux blocs ci-dessous selon si ModelNet40 est disponible.


```python
# ── Hyperparamètres ──────────────────────────────────────────────────────────
N_GRID      = 256   # points sur la grille sphérique
SIGMA       = 0.15  # largeur du noyau gaussien de projection
BATCH_SIZE  = 16
NUM_WORKERS = 0     # mettre 0 si erreurs de fork sous Windows

# ── Option A : ModelNet40 réel (nécessite ~435 Mo et internet) ───────────────
#DATA_ROOT = './data'
#train_ds = ModelNet40Spherical(DATA_ROOT, split='train',
 #                               N_grid=N_GRID, sigma=SIGMA,
  #                              download=True, augment=True)
#val_ds   = ModelNet40Spherical(DATA_ROOT, split='test',
 #                               N_grid=N_GRID, sigma=SIGMA,
  #                              download=False, augment=False)

# ── Option B : données synthétiques (test sans téléchargement) ───────────────
train_ds = SyntheticSphericalDataset(N_samples=320, N_grid=N_GRID, sigma=SIGMA)
val_ds   = SyntheticSphericalDataset(N_samples=80,  N_grid=N_GRID, sigma=SIGMA)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS)

# ── Modèle ───────────────────────────────────────────────────────────────────
model = SphericalCNN(n_classes=40, N_points=N_GRID, lmax=3, cutoff=0.3)
model = model.to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Paramètres apprenables : {n_params:,}')
print(f'  Bloc 1 : 1x0e → {model.bloc1.irreps_out}')
print(f'  Bloc 2 : {model.bloc1.irreps_out} → {model.bloc2.irreps_out}')
print(f'  Bloc 3 : {model.bloc2.irreps_out} → {model.bloc3.irreps_out}')

# Vérification du forward sur un batch
x_test, y_test = next(iter(train_loader))
x_test = x_test.to(DEVICE)
with torch.no_grad():
    logits = model(x_test)
print(f'\nBatch : x={tuple(x_test.shape)}, y={tuple(y_test.shape)}')
print(f'Logits : {tuple(logits.shape)}  ✓')
```

    Paramètres apprenables : 19,074
      Bloc 1 : 1x0e → 8x0e+4x1o+2x2e
      Bloc 2 : 8x0e+4x1o+2x2e → 16x0e+8x1o+4x2e
      Bloc 3 : 16x0e+8x1o+4x2e → 32x0e+16x1o+8x2e
    
    Batch : x=(16, 256, 1), y=(16,)
    Logits : (16, 40)  ✓
    

## Cellule 6 — Entraînement


```python
N_EPOCHS = 50
LR       = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
criterion = nn.CrossEntropyLoss()

history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(1, N_EPOCHS + 1):

    # ── Entraînement ─────────────────────────────────────────────────────────
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for signals, labels in train_loader:
        signals, labels = signals.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(signals), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (model(signals).detach().argmax(1) == labels).sum().item()
        total      += len(labels)

    # ── Validation ───────────────────────────────────────────────────────────
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for signals, labels in val_loader:
            signals, labels = signals.to(DEVICE), labels.to(DEVICE)
            val_correct += (model(signals).argmax(1) == labels).sum().item()
            val_total   += len(labels)

    train_loss = total_loss / total
    train_acc  = correct / total
    val_acc    = val_correct / val_total
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    if epoch % 5 == 0 or epoch == 1:
        print(f'Epoch {epoch:3d}/{N_EPOCHS} | '
              f'Loss {train_loss:.4f} | '
              f'Train {train_acc*100:.1f}% | '
              f'Val {val_acc*100:.1f}%')

print(f'\nMeilleure val acc : {max(history["val_acc"])*100:.1f}%')
```

    Epoch   1/50 | Loss 3.6552 | Train 4.4% | Val 2.5%
    Epoch   5/50 | Loss 2.8692 | Train 11.9% | Val 15.0%
    Epoch  10/50 | Loss 2.5618 | Train 9.1% | Val 12.5%
    Epoch  15/50 | Loss 2.6557 | Train 10.3% | Val 15.0%
    Epoch  20/50 | Loss 2.4472 | Train 10.9% | Val 10.0%
    Epoch  25/50 | Loss 2.4735 | Train 10.6% | Val 15.0%
    Epoch  30/50 | Loss 2.3501 | Train 15.3% | Val 13.8%
    Epoch  35/50 | Loss 2.3069 | Train 15.6% | Val 11.2%
    Epoch  40/50 | Loss 2.4018 | Train 14.1% | Val 18.8%
    Epoch  45/50 | Loss 2.3499 | Train 15.0% | Val 15.0%
    Epoch  50/50 | Loss 2.3468 | Train 12.8% | Val 22.5%
    
    Meilleure val acc : 22.5%
    

## Cellule 7 — Courbes d'apprentissage


```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history['train_loss'], label='Train loss', color='steelblue')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Cross-entropy loss')
axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot([a*100 for a in history['train_acc']], label='Train acc', color='steelblue')
axes[1].plot([a*100 for a in history['val_acc']],   label='Val acc',   color='tomato')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```


```python

```
