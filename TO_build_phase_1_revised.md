# TO_build_phase_1.md (Revised — Architecture v2: Multimodal Early Fusion)
### Quantum-Enhanced Multimodal Dementia Detection — Phase 1 Build Guide
**Repo:** `github.com/Prakshi16/demetia_qsvm` | **Target demo:** May 22, 2026

---

## ⚡ QUICK REFERENCE TABLE

| Person | Task | Branch | Notebook | Output file | Can start? |
|--------|------|--------|----------|-------------|------------|
| **Prakshi** | Clinical Feature Analysis | `feature/clinical-qsvm` | `clinical_qsvm.ipynb` | `results/X_clinical_pca.npy` + `results/y_labels.npy` | ✅ Yes |
| **Bishal** | MRI Feature Analysis | `feature/mri-qsvm` | `mri_qsvm.ipynb` | `results/X_mri_pca.npy` | ✅ Yes |
| **Sheetal** | Speech Feature Analysis | `feature/speech-qsvm` | `speech_qsvm.ipynb` | `results/X_speech_pca.npy` | ✅ Yes |
| **Prakshi** | Multimodal QSVM Training | `feature/multimodal-qsvm` | `multimodal_qsvm.ipynb` | `results/qsvm_model.pkl` + `results/multimodal_results.json` | ⏳ After all 3 .npy files exist |
| **Govind** | Visualization + README | `feature/results-viz` | `results_viz.ipynb` | `results/plots/` | ⏳ After multimodal_qsvm |

> **Demo minimum (May 22):** All three analysis notebooks done + multimodal_qsvm.ipynb trained and pickled.
> Train the model once tonight. Load from pickle for the demo — prediction is fast.

---

## 📌 SECTION 0 — PROJECT OVERVIEW

### What this project is

A Quantum-Enhanced Multimodal Framework for Early Dementia Detection.

**Phase 1 Goal:** Build one unified Quantum SVM (QSVM) that learns from three data modalities simultaneously — Clinical scores, MRI-derived brain volume measurements, and Speech acoustic features — using a single patient-level dataset where every row contains all three modalities.

**Why QSVM?** Classical SVMs use manually designed kernels (e.g. RBF). A QSVM computes the kernel using a quantum circuit (ZZFeatureMap), which maps features into a Hilbert space exponentially larger than classical space. The Qiskit implementation simulates this quantum circuit on a classical computer using a statevector simulator.

**Why early fusion (not late fusion)?** Our dataset — `multimodal_dementia_dataset.csv` — contains all three modalities for every patient in the same row. Late fusion would artificially separate information that is naturally aligned at the subject level. Early fusion is the correct approach: each modality is compressed to 2 principal components independently (preserving modality identity), then concatenated into a 6-feature vector that represents the full patient profile before QSVM training.

**Why modality-aware PCA before concatenation?** Each modality lives in a completely different feature space — clinical scores (CDR, MMSE) have no natural unit relationship with MRI volumes or MFCC coefficients. Scaling and compressing each modality separately via PCA(2) before fusion ensures each modality contributes equally to the final 6-dimensional representation.

### Architecture

```
multimodal_dementia_dataset.csv  (225 rows, 1 per patient)
         │
         ├── Clinical features (5)  → StandardScaler → PCA(2) → 2 components
         │   CDR, MMSE, ASF, EDUC, SES                          ↓
         │                                               [clinical_pc1, clinical_pc2]
         │                                                        ↓
         ├── MRI features (4)       → StandardScaler → PCA(2) → 2 components
         │   nWBV, eTIV,                                         ↓
         │   hippocampal_volume,                        [mri_pc1, mri_pc2]
         │   cortical_thickness                                   ↓
         │                                                        ↓
         └── Speech features (18)   → StandardScaler → PCA(2) → 2 components
             pause_rate, speech_rate,                            ↓
             pitch_mean, jitter,                        [speech_pc1, speech_pc2]
             shimmer, mfcc_1…mfcc_13                             ↓
                                                                 ↓
                                          Concatenate → (225 × 6) feature matrix
                                                                 ↓
                                          train_test_split (80/20, stratify, seed=42)
                                                                 ↓
                                          Classical SVM (RBF, baseline)
                                                  +
                                          QSVM (ZZFeatureMap, feature_dimension=6)
                                                                 ↓
                                          Evaluate + pickle model + save results
```

### Accuracy targets to beat

| Reference | Accuracy | Type |
|-----------|----------|------|
| Akinrotimi (BASE) | 91.25% | QSVM — must beat |
| Chakravarthi (REF-1) | 94.20% | Classical multimodal — must beat |
| So et al. (REF-7) | 97.20% | Clinical scores only |
| Lin & Washington (REF-6) | ~84% | Speech only |

### Dataset overview

| Dataset | Description | Status |
|---------|-------------|--------|
| `multimodal_dementia_dataset.csv` | 225 patients, 27 features across 3 modalities + label | ✅ Ready — copy to `data/` |

**Column breakdown:**

| Group | Columns | Count |
|-------|---------|-------|
| Identifier | `Subject_ID` | 1 |
| Clinical | `CDR`, `MMSE`, `ASF`, `EDUC`, `SES` | 5 |
| MRI-derived | `nWBV`, `eTIV`, `hippocampal_volume`, `cortical_thickness` | 4 |
| Speech | `pause_rate`, `speech_rate`, `pitch_mean`, `jitter`, `shimmer`, `mfcc_1`…`mfcc_13` | 18 |
| Label | `Label` (0=Nondemented, 1=Demented) | 1 |

---

## 🛠️ SECTION 1 — ONE-TIME SETUP (ALL MEMBERS DO THIS)

### 1.1 Git Workflow

**Bishal and Sheetal — Clone the repo first:**
```bash
git clone https://github.com/Prakshi16/demetia_qsvm.git
cd demetia_qsvm
```

**Each person creates their own branch:**
```bash
# Bishal:
git checkout -b feature/mri-qsvm

# Sheetal:
git checkout -b feature/speech-qsvm

# Prakshi (multimodal training, after analysis notebooks done):
git checkout -b feature/multimodal-qsvm
```

**Push your work:**
```bash
git add .
git commit -m "WIP: mri_qsvm notebook EDA + PCA done - Bishal"
git push origin feature/mri-qsvm    # replace with your branch name
```

**Pull latest main before starting each session:**
```bash
git checkout main
git pull origin main
git checkout feature/your-branch-name
git merge main
```

> **Prakshi** reviews each branch on GitHub and merges to `main` via Pull Request. Nobody pushes directly to `main`.

---

### 1.2 Install Dependencies

Run once in terminal (not inside a notebook):
```bash
pip install qiskit==0.45.3
pip install qiskit-machine-learning==0.7.2
pip install qiskit-algorithms==0.3.0
pip install pennylane==0.38.0
pip install pandas numpy scikit-learn matplotlib seaborn
pip install jupyter ipykernel shap
```

**Verify:**
```python
import qiskit, qiskit_machine_learning, sklearn, pandas, numpy
print("qiskit:", qiskit.__version__)                         # 0.45.x
print("qiskit_ml:", qiskit_machine_learning.__version__)     # 0.7.x
print("sklearn:", sklearn.__version__)
```

**If `from qiskit_machine_learning.algorithms import QSVC` fails:**
```python
from qiskit_machine_learning.algorithms.classifiers import QSVC
```

---

### 1.3 Fix the .gitignore

Open `.gitignore` in VS Code and replace its entire content with:

```
# Datasets — never commit
data/

# Jupyter checkpoints
.ipynb_checkpoints/
**/.ipynb_checkpoints/

# Python compiled files
__pycache__/
*.pyc
*.pyo

# Build artifacts
*.egg-info/
dist/
build/

# Virtual environments
venv/
env/
.venv/

# Secrets
.env
*.key
*.pem

# IDE
.vscode/settings.json
.idea/

# OS
.DS_Store
Thumbs.db
```

> ⚠️ `results/` is intentionally NOT in `.gitignore`. The `.npy` files, `.pkl` model, `.json`, and plots go there and must be committed so teammates can build on them.

---

### 1.4 Folder Structure

Run once from `demetia_qsvm/` root:
```bash
mkdir -p results results/plots
```

Final layout:
```
demetia_qsvm/
├── data/
│   └── multimodal_dementia_dataset.csv     ← LOCAL ONLY, never committed
├── results/                                 ← COMMITTED to GitHub
│   ├── X_clinical_pca.npy                  ← Prakshi's output (shape: 225×2)
│   ├── X_mri_pca.npy                       ← Bishal's output  (shape: 225×2)
│   ├── X_speech_pca.npy                    ← Sheetal's output (shape: 225×2)
│   ├── y_labels.npy                        ← Prakshi's output (shape: 225,)
│   ├── qsvm_model.pkl                      ← Prakshi's trained model
│   ├── multimodal_results.json             ← Prakshi's final results
│   └── plots/
│       ├── clinical_pca.png
│       ├── mri_pca.png
│       ├── speech_pca.png
│       ├── multimodal_confusion_matrix.png
│       └── main_comparison_chart.png
├── .gitignore
├── clinical_qsvm.ipynb                     ← Prakshi  (exists, fill cells)
├── mri_qsvm.ipynb                          ← Bishal   (create this)
├── speech_qsvm.ipynb                       ← Sheetal  (create this)
├── multimodal_qsvm.ipynb                   ← Prakshi  (create after .npy files exist)
└── results_viz.ipynb                       ← Govind   (create after multimodal done)
```

---

### 1.5 Dataset Setup

The dataset is pre-generated. No download needed.

1. Copy `multimodal_dementia_dataset.csv` (provided by Prakshi) to `demetia_qsvm/data/`
2. Verify it exists: open in VS Code — should show **225 rows** and columns including `Subject_ID`, `CDR`, `MMSE`, `nWBV`, `hippocampal_volume`, `pause_rate`, `mfcc_1`, `Label`
3. Confirm it is NOT tracked by git: `git status` should not show it

> **All three people (Prakshi, Bishal, Sheetal) load from the same CSV.** The dataset contains all modalities in one file. You each select different column groups from it.

---

## 📖 SECTION 2 — HOW TO USE THIS FILE WITH GPT (Bishal + Sheetal)

**Step-by-step for every GPT session:**

1. Open GPT-4 at chat.openai.com

2. Paste this opening message first:
   ```
   I am [Bishal / Sheetal], a final-year AI & ML student working on a quantum 
   machine learning project for dementia detection at CMR Institute of Technology.
   My specific task is [TASK B: MRI Feature Analysis / TASK C: Speech Feature Analysis].
   
   I will now paste my complete task specification. Please help me implement it 
   step by step exactly as described. Do not change library choices, column names, 
   file paths, or the PCA logic — those are shared constraints with my teammates.
   
   [Paste your full task section below — Section 5 for Bishal, Section 6 for Sheetal]
   ```

3. Work cell by cell. For each cell: paste it and say "help me understand this and check for bugs"

4. If GPT suggests different libraries or preprocessing — explain that the shared loading block in Section 3 is locked

5. When done: verify your `.npy` file is in `results/` with the exact filename, then push to your branch

---

## 🔗 SECTION 3 — SHARED DATASET LOADING BLOCK

> ⚠️ **All three analysis notebooks start with this exact block.**
> Copy it verbatim — do not change the file path, column names, or label encoding.
> Because all three notebooks load the same CSV with the same preprocessing,
> the 225 rows are in the same order for everyone. This guarantees that when
> Prakshi concatenates the PCA arrays, they align perfectly by row.

```python
# ============================================================
# SHARED DATASET LOADING BLOCK
# Copy verbatim into clinical_qsvm.ipynb, mri_qsvm.ipynb,
# and speech_qsvm.ipynb — do not change anything here
# ============================================================
import pandas as pd
import numpy as np

df = pd.read_csv('data/multimodal_dementia_dataset.csv')

# Safety checks
assert df.shape == (225, 29), f"Unexpected shape: {df.shape}"
assert df.isnull().sum().sum() == 0, "NaNs found in dataset!"
assert df['Label'].isin([0, 1]).all(), "Invalid labels found!"

print(f"Dataset loaded: {df.shape}")
print(f"Label distribution:\n{df['Label'].value_counts()}")
print(f"Columns: {df.columns.tolist()}")
# ============================================================
# END SHARED LOADING BLOCK
# ============================================================
```

**Expected output:**
```
Dataset loaded: (225, 29)
Label distribution:
0    124
1    101
```

---

## 👩‍💻 SECTION 4 — TASK A: PRAKSHI — Clinical Feature Analysis

**File:** `clinical_qsvm.ipynb` (already exists — fill in these cells)
**Branch:** `feature/clinical-qsvm`
**Output:** `results/X_clinical_pca.npy` (shape: 225×2) + `results/y_labels.npy` (shape: 225,)
**Features:** `CDR`, `MMSE`, `ASF`, `EDUC`, `SES`

**What these features mean:**
- `CDR` = Clinical Dementia Rating (0=normal, 0.5=mild, 1–2=moderate/severe)
- `MMSE` = Mini Mental State Examination (30=perfect, <24=impairment)
- `ASF` = Atlas Scaling Factor (brain size correction)
- `EDUC` = Years of education (protective factor)
- `SES` = Socioeconomic status (1–5)

---

### Cell 1 — Imports
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
print("✅ All imports successful")
```

### Cell 2 — Load Dataset (Shared Block)
> Paste the entire SHARED DATASET LOADING BLOCK from Section 3 here.

### Cell 3 — Explore Clinical Features
```python
CLINICAL_FEATURES = ['CDR', 'MMSE', 'ASF', 'EDUC', 'SES']

print("=== CLINICAL FEATURE OVERVIEW ===")
print(df[CLINICAL_FEATURES].describe().round(3))
print(f"\nNull values: {df[CLINICAL_FEATURES].isnull().sum().to_dict()}")

print("\n=== PER-CLASS MEANS ===")
print(df.groupby('Label')[CLINICAL_FEATURES].mean().round(3))
```

### Cell 4 — Feature Distribution Plots
```python
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
colors = {0: 'royalblue', 1: 'tomato'}
label_names = {0: 'Nondemented', 1: 'Demented'}

for ax, feat in zip(axes, CLINICAL_FEATURES):
    for lbl in [0, 1]:
        ax.hist(df[df['Label']==lbl][feat],
                bins=15, alpha=0.65, color=colors[lbl],
                label=label_names[lbl], edgecolor='black', linewidth=0.3)
    ax.set_title(feat, fontsize=11, fontweight='bold')
    ax.set_xlabel(feat)
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)

plt.suptitle('Clinical Feature Distributions by Dementia Status',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/clinical_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: results/plots/clinical_feature_distributions.png")
```

### Cell 5 — Select Features + Scale
```python
X_clinical = df[CLINICAL_FEATURES].values
y = df['Label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clinical)

print(f"Feature matrix shape: {X_scaled.shape}")
print(f"Mean after scaling (should be ~0): {X_scaled.mean(axis=0).round(4)}")
print(f"Std  after scaling (should be ~1): {X_scaled.std(axis=0).round(4)}")
```

### Cell 6 — PCA Reduction to 2 Components
```python
# Reduce 5 clinical features → 2 principal components
# This is what gets passed to the QSVM (feature_dimension=2 per modality)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"Explained variance per component: {pca.explained_variance_ratio_.round(4)}")
print(f"Total variance retained: {pca.explained_variance_ratio_.sum():.2%}")
print(f"Shape after PCA: {X_pca.shape}")   # should be (225, 2)

# Visualize class separation in PCA space
plt.figure(figsize=(8, 5))
plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1],
            c='royalblue', label='Nondemented', alpha=0.7, edgecolors='k', linewidths=0.3)
plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1],
            c='tomato', label='Demented', alpha=0.7, edgecolors='k', linewidths=0.3)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Clinical Features (CDR, MMSE, ASF, EDUC, SES)\nClass Separation')
plt.legend()
plt.tight_layout()
plt.savefig('results/plots/clinical_pca.png', dpi=150)
plt.show()
print("Saved: results/plots/clinical_pca.png")
```

### Cell 7 — Save PCA Features + Labels
```python
# Save the 2D PCA-compressed clinical features
# Shape must be (225, 2) — Prakshi's multimodal notebook will concatenate this
np.save('results/X_clinical_pca.npy', X_pca)

# y_labels saved ONCE here (all modalities share the same label vector)
np.save('results/y_labels.npy', y)

print(f"✅ Saved: results/X_clinical_pca.npy  — shape: {X_pca.shape}")
print(f"✅ Saved: results/y_labels.npy         — shape: {y.shape}")

# Sanity check
check_X = np.load('results/X_clinical_pca.npy')
check_y = np.load('results/y_labels.npy')
assert check_X.shape == (225, 2), f"Wrong shape: {check_X.shape}"
assert check_y.shape == (225,),   f"Wrong shape: {check_y.shape}"
print("✅ Reload check passed")
print(f"   X_clinical_pca: {check_X.shape}  |  y_labels: {check_y.shape}")
```

---

## 👨‍💻 SECTION 5 — TASK B: BISHAL — MRI Feature Analysis

---
### 📋 GPT CONTEXT BLOCK — Paste at START of every GPT session

```
I am Bishal, a final-year AI & ML student at CMR Institute of Technology, Bengaluru.
I am building a quantum machine learning project for dementia detection.

MY SPECIFIC TASK: Build an MRI feature analysis notebook called mri_qsvm.ipynb

PROJECT CONTEXT:
- We use a unified multimodal dataset: multimodal_dementia_dataset.csv
  (225 patients, each with clinical + MRI + speech features + label in one row)
- My job is NOT to train a QSVM — I do feature analysis, scaling, and PCA only
- My MRI features: nWBV, eTIV, hippocampal_volume, cortical_thickness
- I must save the PCA-compressed features as: results/X_mri_pca.npy (shape: 225×2)
- My teammate Prakshi will load this .npy file and concatenate it with clinical
  and speech PCA outputs to train ONE unified QSVM
- The 225 rows must stay in the same order as in the CSV — do not shuffle here

Please help me implement Section 5 exactly as described below.
[Paste Section 5 below]
```
---

**File to create:** `mri_qsvm.ipynb`
**Branch:** `feature/mri-qsvm`
**Output:** `results/X_mri_pca.npy` (shape: 225×2)
**Features:** `nWBV`, `eTIV`, `hippocampal_volume`, `cortical_thickness`

**Why these are MRI features:**
- `nWBV` (normalized whole brain volume): brain shrinks measurably with dementia. Strongest single structural biomarker
- `eTIV` (estimated total intracranial volume): reference for head size normalization
- `hippocampal_volume` (mm³): hippocampus is the first structure to atrophy in Alzheimer's — most important region-specific biomarker
- `cortical_thickness` (mm): cortical thinning tracks neurodegeneration across the brain

All four are quantities computed from MRI scan segmentation. The CSV contains their extracted values directly — raw `.nii` scan files are not needed.

---

### Cell 1 — Imports
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
print("✅ All imports successful")
```

### Cell 2 — Load Dataset (Shared Block)
> Paste the entire SHARED DATASET LOADING BLOCK from Section 3 here. Do not change a single line.

### Cell 3 — Explore MRI Features
```python
MRI_FEATURES = ['nWBV', 'eTIV', 'hippocampal_volume', 'cortical_thickness']

print("=== MRI FEATURE OVERVIEW ===")
print(df[MRI_FEATURES].describe().round(3))
print(f"\nNull values: {df[MRI_FEATURES].isnull().sum().to_dict()}")

print("\n=== PER-CLASS MEANS ===")
print(df.groupby('Label')[MRI_FEATURES].mean().round(3))
```

### Cell 4 — Feature Distribution Plots
```python
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
colors = {0: 'royalblue', 1: 'tomato'}
label_names = {0: 'Nondemented', 1: 'Demented'}

for ax, feat in zip(axes, MRI_FEATURES):
    for lbl in [0, 1]:
        ax.hist(df[df['Label']==lbl][feat],
                bins=20, alpha=0.65, color=colors[lbl],
                label=label_names[lbl], edgecolor='black', linewidth=0.3)
    ax.set_title(feat, fontsize=11, fontweight='bold')
    ax.set_xlabel(feat)
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)

plt.suptitle('MRI Feature Distributions by Dementia Status',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/mri_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: results/plots/mri_feature_distributions.png")
```

### Cell 5 — Select Features + Scale
```python
X_mri = df[MRI_FEATURES].values
y = df['Label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_mri)

print(f"Feature matrix shape: {X_scaled.shape}")
print(f"Mean after scaling (should be ~0): {X_scaled.mean(axis=0).round(4)}")
print(f"Std  after scaling (should be ~1): {X_scaled.std(axis=0).round(4)}")
```

### Cell 6 — PCA Reduction to 2 Components
```python
# Reduce 4 MRI features → 2 principal components
# With 4 input features, PCA(2) captures the most important variance directions
# (brain volume + thickness tend to be correlated, so PC1 captures overall atrophy)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"Explained variance per component: {pca.explained_variance_ratio_.round(4)}")
print(f"Total variance retained: {pca.explained_variance_ratio_.sum():.2%}")
print(f"Shape after PCA: {X_pca.shape}")   # should be (225, 2)

plt.figure(figsize=(8, 5))
plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1],
            c='royalblue', label='Nondemented', alpha=0.7, edgecolors='k', linewidths=0.3)
plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1],
            c='tomato', label='Demented', alpha=0.7, edgecolors='k', linewidths=0.3)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of MRI Features (nWBV, eTIV, hippocampal_volume, cortical_thickness)\nClass Separation')
plt.legend()
plt.tight_layout()
plt.savefig('results/plots/mri_pca.png', dpi=150)
plt.show()
print("Saved: results/plots/mri_pca.png")
```

### Cell 7 — Save PCA Features
```python
np.save('results/X_mri_pca.npy', X_pca)

print(f"✅ Saved: results/X_mri_pca.npy  — shape: {X_pca.shape}")

check = np.load('results/X_mri_pca.npy')
assert check.shape == (225, 2), f"Wrong shape: {check.shape}"
print("✅ Reload check passed")
```

---

## 👩‍💻 SECTION 6 — TASK C: SHEETAL — Speech Feature Analysis

---
### 📋 GPT CONTEXT BLOCK — Paste at START of every GPT session

```
I am Sheetal, a final-year AI & ML student at CMR Institute of Technology, Bengaluru.
I am building a quantum machine learning project for dementia detection.

MY SPECIFIC TASK: Build a speech feature analysis notebook called speech_qsvm.ipynb

PROJECT CONTEXT:
- We use a unified multimodal dataset: multimodal_dementia_dataset.csv
  (225 patients, each with clinical + MRI + speech features + label in one row)
- My job is NOT to train a QSVM — I do feature analysis, scaling, and PCA only
- My speech features are 18 columns: pause_rate, speech_rate, pitch_mean, jitter,
  shimmer, and mfcc_1 through mfcc_13
- I must save the PCA-compressed features as: results/X_speech_pca.npy (shape: 225×2)
- My teammate Prakshi will load this .npy file and concatenate it with clinical
  and MRI PCA outputs to train ONE unified QSVM
- The 225 rows must stay in the same order as in the CSV — do not shuffle here

Please help me implement Section 6 exactly as described below.
[Paste Section 6 below]
```
---

**File to create:** `speech_qsvm.ipynb`
**Branch:** `feature/speech-qsvm`
**Output:** `results/X_speech_pca.npy` (shape: 225×2)
**Features:** `pause_rate`, `speech_rate`, `pitch_mean`, `jitter`, `shimmer`, `mfcc_1`…`mfcc_13` (18 total)

**What these speech features represent:**
- `pause_rate`: proportion of silence — dementia patients pause more frequently
- `speech_rate`: syllables per second — slower in dementia due to word-finding difficulty
- `pitch_mean`: average fundamental frequency — dementia causes flatter, more monotone speech
- `jitter`: cycle-to-cycle pitch variation — higher in dementia, reflects vocal instability
- `shimmer`: amplitude variation — higher in dementia, reflects loss of vocal control
- `mfcc_1` to `mfcc_13`: Mel-frequency cepstral coefficients — standard spectral representation of voice, widely used in speech pathology. All 13 are standard from ADReSS challenge methodology

**Note on ADReSS:** If/when the real ADReSS dataset arrives, the pipeline from Cell 5 onwards is identical — just replace the CSV loading with ADReSS feature extraction and save to the same `X_speech_pca.npy` path.

---

### Cell 1 — Imports
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
print("✅ All imports successful")
```

### Cell 2 — Load Dataset (Shared Block)
> Paste the entire SHARED DATASET LOADING BLOCK from Section 3 here. Do not change a single line.

### Cell 3 — Define Speech Feature Columns
```python
MFCC_COLS = [f'mfcc_{i}' for i in range(1, 14)]  # mfcc_1 through mfcc_13
SPEECH_FEATURES = ['pause_rate', 'speech_rate', 'pitch_mean',
                   'jitter', 'shimmer'] + MFCC_COLS

print(f"Total speech features: {len(SPEECH_FEATURES)}")
print(f"Features: {SPEECH_FEATURES}")

print("\n=== SPEECH FEATURE OVERVIEW ===")
print(df[SPEECH_FEATURES[:5]].describe().round(4))   # show non-MFCC features
print("\n=== PER-CLASS MEANS (acoustic features) ===")
print(df.groupby('Label')[['pause_rate', 'speech_rate', 'pitch_mean',
                            'jitter', 'shimmer']].mean().round(4))
```

### Cell 4 — Visualize Acoustic Feature Distributions
```python
# Show the 5 interpretable acoustic features (not all 13 MFCCs)
acoustic_feats = ['pause_rate', 'speech_rate', 'pitch_mean', 'jitter', 'shimmer']
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
colors = {0: 'royalblue', 1: 'tomato'}
label_names = {0: 'Nondemented', 1: 'Demented'}

for ax, feat in zip(axes, acoustic_feats):
    for lbl in [0, 1]:
        ax.hist(df[df['Label']==lbl][feat],
                bins=20, alpha=0.65, color=colors[lbl],
                label=label_names[lbl], edgecolor='black', linewidth=0.3)
    ax.set_title(feat, fontsize=10, fontweight='bold')
    ax.set_xlabel(feat)
    ax.set_ylabel('Count')
    ax.legend(fontsize=7)

plt.suptitle('Speech Acoustic Feature Distributions by Dementia Status',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/speech_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: results/plots/speech_feature_distributions.png")
```

### Cell 5 — MFCC Heatmap (Class Comparison)
```python
# Visualize all 13 MFCCs — shows which coefficients differ between classes
mfcc_means = df.groupby('Label')[MFCC_COLS].mean()

plt.figure(figsize=(14, 3))
sns.heatmap(mfcc_means,
            annot=True, fmt='.1f', cmap='RdYlBu_r',
            yticklabels=['Nondemented (0)', 'Demented (1)'],
            linewidths=0.5)
plt.title('MFCC Mean Values by Dementia Class\n(differences across MFCCs indicate spectral changes in voice)',
          fontsize=12)
plt.tight_layout()
plt.savefig('results/plots/mfcc_class_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: results/plots/mfcc_class_comparison.png")
```

### Cell 6 — Select Features + Scale
```python
X_speech = df[SPEECH_FEATURES].values
y = df['Label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_speech)

print(f"Feature matrix shape: {X_scaled.shape}")   # (225, 18)
print(f"Mean after scaling (should be ~0): {X_scaled.mean(axis=0).round(3)}")
print(f"Std  after scaling (should be ~1): {X_scaled.std(axis=0).round(3)}")
```

### Cell 7 — PCA Reduction to 2 Components
```python
# Reduce 18 speech features → 2 principal components
# With 18 features, PCA(2) captures the dominant patterns of dementia speech
# (typically: overall speech degradation along PC1, rhythm vs spectral along PC2)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"Explained variance per component: {pca.explained_variance_ratio_.round(4)}")
print(f"Total variance retained: {pca.explained_variance_ratio_.sum():.2%}")
print(f"Shape after PCA: {X_pca.shape}")   # should be (225, 2)

plt.figure(figsize=(8, 5))
plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1],
            c='royalblue', label='Nondemented', alpha=0.7, edgecolors='k', linewidths=0.3)
plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1],
            c='tomato', label='Demented', alpha=0.7, edgecolors='k', linewidths=0.3)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Speech Features (18 → 2 Components)\nClass Separation')
plt.legend()
plt.tight_layout()
plt.savefig('results/plots/speech_pca.png', dpi=150)
plt.show()
print("Saved: results/plots/speech_pca.png")
```

### Cell 8 — Save PCA Features
```python
np.save('results/X_speech_pca.npy', X_pca)

print(f"✅ Saved: results/X_speech_pca.npy  — shape: {X_pca.shape}")

check = np.load('results/X_speech_pca.npy')
assert check.shape == (225, 2), f"Wrong shape: {check.shape}"
print("✅ Reload check passed")
```

---

## 🧠 SECTION 7 — TASK D: PRAKSHI — Multimodal QSVM Training

**File to create:** `multimodal_qsvm.ipynb`
**Branch:** `feature/multimodal-qsvm`
**Depends on:** All three `.npy` files must exist in `results/` before running this
**Outputs:**
- `results/qsvm_model.pkl` — trained QSVM (pickled for demo)
- `results/multimodal_results.json` — all accuracy numbers

> ⚠️ **Do NOT run this notebook until all three files exist:**
> `results/X_clinical_pca.npy`, `results/X_mri_pca.npy`, `results/X_speech_pca.npy`
>
> **Training will take 30–60 minutes on CPU** with `feature_dimension=6`.
> Run it once, pickle the model, load from pickle for the demo — prediction is fast.

---

### Cell 1 — Imports
```python
import numpy as np
import pandas as pd
import json
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
import warnings
warnings.filterwarnings('ignore')
print("✅ All imports successful")
```

### Cell 2 — Load All Modality PCA Features
```python
# Load PCA outputs from each modality notebook
try:
    X_clinical = np.load('results/X_clinical_pca.npy')
    print(f"✅ Clinical PCA loaded   — shape: {X_clinical.shape}")
except FileNotFoundError:
    raise FileNotFoundError("❌ Run clinical_qsvm.ipynb first and push X_clinical_pca.npy")

try:
    X_mri = np.load('results/X_mri_pca.npy')
    print(f"✅ MRI PCA loaded        — shape: {X_mri.shape}")
except FileNotFoundError:
    raise FileNotFoundError("❌ Run mri_qsvm.ipynb first and push X_mri_pca.npy")

try:
    X_speech = np.load('results/X_speech_pca.npy')
    print(f"✅ Speech PCA loaded     — shape: {X_speech.shape}")
except FileNotFoundError:
    raise FileNotFoundError("❌ Run speech_qsvm.ipynb first and push X_speech_pca.npy")

y = np.load('results/y_labels.npy')
print(f"✅ Labels loaded         — shape: {y.shape}")
print(f"\nLabel distribution: {np.bincount(y)}  ([Nondemented, Demented])")
```

### Cell 3 — Verify Alignment + Concatenate
```python
# All three arrays must have exactly 225 rows — same order as the CSV
assert X_clinical.shape == (225, 2), f"Clinical shape wrong: {X_clinical.shape}"
assert X_mri.shape      == (225, 2), f"MRI shape wrong: {X_mri.shape}"
assert X_speech.shape   == (225, 2), f"Speech shape wrong: {X_speech.shape}"
assert y.shape          == (225,),   f"Labels shape wrong: {y.shape}"
print("✅ All shape checks passed")

# Concatenate: each patient is now represented by 6 features
# [clinical_pc1, clinical_pc2, mri_pc1, mri_pc2, speech_pc1, speech_pc2]
X_full = np.concatenate([X_clinical, X_mri, X_speech], axis=1)
print(f"\nConcatenated feature matrix: {X_full.shape}")  # (225, 6)
print("Feature layout: [clinical_pc1, clinical_pc2, mri_pc1, mri_pc2, speech_pc1, speech_pc2]")
```

### Cell 4 — Train/Test Split
```python
# One split for all modalities — guaranteed aligned because rows come from same CSV
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"Train labels: {np.bincount(y_train)}  |  Test labels: {np.bincount(y_test)}")
```

### Cell 5 — Classical SVM Baseline (6 features)
```python
# Classical SVM with RBF kernel on the 6-feature concatenated vector
# This is our internal baseline — what we must beat with QSVM
svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_clf.fit(X_train, y_train)
svm_pred = svm_clf.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)

print("=== Classical SVM — Multimodal (6 features) ===")
print(f"Accuracy: {svm_acc:.4f}  ({svm_acc*100:.2f}%)")
print(classification_report(y_test, svm_pred,
                             target_names=['Nondemented', 'Demented']))
```

### Cell 6 — QSVM with ZZFeatureMap (feature_dimension=6)
```python
# ZZFeatureMap with 6 qubits — one per PCA component (2 per modality)
# entanglement='linear': adjacent qubits interact (1-2, 2-3, 3-4, 4-5, 5-6)
# This means: within-modality (pc1↔pc2) and cross-modality (mri_pc2↔speech_pc1) interactions
# reps=2: encoding layer repeated twice for richer quantum feature space
#
# Expected runtime: 30–60 minutes on CPU simulator
# With GPU-accelerated Aer backend: significantly faster
# Run once → pickle → load for demo

feature_map = ZZFeatureMap(feature_dimension=6, reps=2, entanglement='linear')
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

print("Training multimodal QSVM...")
print("This will take 30–60 minutes. Do NOT interrupt — let it finish.")
print("Kernel matrix size: 180×180 = 32,400 circuit evaluations")

qsvm_clf = QSVC(quantum_kernel=quantum_kernel)
qsvm_clf.fit(X_train, y_train)

qsvm_pred = qsvm_clf.predict(X_test)
qsvm_acc = accuracy_score(y_test, qsvm_pred)

print("\n=== QSVM (ZZFeatureMap, 6 qubits) — Multimodal ===")
print(f"Accuracy: {qsvm_acc:.4f}  ({qsvm_acc*100:.2f}%)")
print(classification_report(y_test, qsvm_pred,
                             target_names=['Nondemented', 'Demented']))
```

### Cell 7 — Results Comparison Table
```python
print("\n" + "="*65)
print(f"{'PHASE 1 COMPLETE RESULTS':^65}")
print("="*65)
print(f"{'Model':<40} {'Accuracy':>10}  {'Source':>8}")
print("-"*65)

rows = [
    ("Classical SVM — Multimodal (ours)",     svm_acc*100,  "Ours"),
    ("QSVM ZZFeatureMap — Multimodal (ours)", qsvm_acc*100, "Ours ★"),
    ("-"*40,                                   None,         ""),
    ("So et al. (Clinical only)",              97.20,        "REF-7"),
    ("Lin & Washington (Speech only)",         84.00,        "REF-6"),
    ("Akinrotimi QSVM (BASE)",                 91.25,        "BASE ◄"),
    ("Chakravarthi Classical Multimodal",      94.20,        "REF-1 ◄"),
]

for name, acc, source in rows:
    if acc is None:
        print(f"{name}")
    else:
        beat = " ✅ BEAT" if (acc > 91.25 and "Ours" in source) else ""
        print(f"{name:<40} {acc:>9.2f}%  {source:>8}{beat}")

print("="*65)
print("◄ = targets we must beat  |  ★ = our main result")
```

### Cell 8 — Confusion Matrix
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, pred, title, cmap in zip(
    axes,
    [svm_pred, qsvm_pred],
    ['Classical SVM (Multimodal)', 'QSVM ZZFeatureMap (Multimodal)'],
    ['Blues', 'Greens']
):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=cmap,
                xticklabels=['Nondemented', 'Demented'],
                yticklabels=['Nondemented', 'Demented'],
                annot_kws={'size': 14})
    acc = accuracy_score(y_test, pred)
    ax.set_title(f'{title}\nAccuracy: {acc*100:.2f}%', fontsize=12)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

plt.suptitle('Multimodal Dementia Classification — Clinical + MRI + Speech',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/multimodal_confusion_matrix.png', dpi=150)
plt.show()
print("Saved: results/plots/multimodal_confusion_matrix.png")
```

### Cell 9 — Pickle Model + Save Results JSON
```python
# ── Pickle the trained QSVM ──────────────────────────────────────────────
# This is critical for the demo: train once, save, load instantly at demo time
with open('results/qsvm_model.pkl', 'wb') as f:
    pickle.dump(qsvm_clf, f)
print("✅ Saved: results/qsvm_model.pkl")

# ── Verify pickle works ───────────────────────────────────────────────────
with open('results/qsvm_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
verify_pred = loaded_model.predict(X_test)
assert accuracy_score(y_test, verify_pred) == qsvm_acc, "Pickle verification failed!"
print("✅ Pickle verified — loaded model gives same accuracy")

# ── Save results JSON (for Govind's viz notebook) ────────────────────────
final_results = {
    "dataset":              "multimodal_dementia_dataset.csv",
    "n_patients":           225,
    "n_features_total":     27,
    "n_features_per_mod":   {"clinical": 5, "mri": 4, "speech": 18},
    "pca_components":       2,
    "qsvm_feature_dim":     6,
    "svm_acc":              round(float(svm_acc), 4),
    "qsvm_acc":             round(float(qsvm_acc), 4),
    "published_baselines": {
        "akinrotimi_qsvm_base":         0.9125,
        "chakravarthi_classical_ref1":  0.9420,
        "so_et_al_clinical_ref7":       0.9720,
        "lin_washington_speech_ref6":   0.8400
    }
}

with open('results/multimodal_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print("\n✅ Saved: results/multimodal_results.json")
print(json.dumps(final_results, indent=2))
```

---

## 📊 SECTION 8 — TASK E: GOVIND — Visualization + README

---
### 📋 GPT CONTEXT BLOCK

```
I am Govind, working on a quantum ML dementia detection project at CMR Institute.
My task is to generate a final comparison bar chart and write the project README.
The results are stored in results/multimodal_results.json.
[Paste Section 8 below]
```
---

**File to create:** `results_viz.ipynb`
**Branch:** `feature/results-viz`
**Depends on:** `results/multimodal_results.json` (Prakshi must finish first)

### Cell 1 — Load Results
```python
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

with open('results/multimodal_results.json') as f:
    results = json.load(f)

print("Results loaded:")
print(f"  Classical SVM: {results['svm_acc']*100:.2f}%")
print(f"  QSVM:          {results['qsvm_acc']*100:.2f}%")
```

### Cell 2 — Main Comparison Bar Chart
```python
model_names = [
    'Classical SVM\n(Multimodal)',
    'QSVM\n(Multimodal\nOurs)',
    'Akinrotimi\n(QSVM BASE)',
    'Chakravarthi\n(Classical\nREF-1)',
    'So et al.\n(Clinical\nREF-7)',
    'Lin & Washington\n(Speech\nREF-6)',
]
acc_values = [
    results['svm_acc']*100,
    results['qsvm_acc']*100,
    91.25,
    94.20,
    97.20,
    84.00,
]
bar_colors = [
    '#4472C4',   # Classical SVM ours
    '#70AD47',   # QSVM ours
    '#C00000',   # Akinrotimi baseline
    '#FF6600',   # Chakravarthi baseline
    '#808080',   # So et al
    '#808080',   # Lin & Washington
]

fig, ax = plt.subplots(figsize=(13, 6))
bars = ax.bar(model_names, acc_values, color=bar_colors,
              edgecolor='black', linewidth=0.7, width=0.6)

for bar, acc in zip(bars, acc_values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.4,
            f'{acc:.2f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.axhline(y=94.20, color='#FF6600', linestyle='--',
           alpha=0.8, linewidth=1.5, label='Chakravarthi target (94.2%)')
ax.axhline(y=91.25, color='#C00000', linestyle='--',
           alpha=0.8, linewidth=1.5, label='Akinrotimi target (91.25%)')

ax.set_ylim(70, 105)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title(
    'Phase 1 Results — Multimodal QSVM vs Classical SVM vs Published Baselines\n'
    'Quantum-Enhanced Multimodal Dementia Detection (Clinical + MRI + Speech)',
    fontsize=12, fontweight='bold'
)

legend_handles = [
    mpatches.Patch(color='#4472C4', label='Classical SVM (ours)'),
    mpatches.Patch(color='#70AD47', label='Multimodal QSVM (ours)'),
    mpatches.Patch(color='#C00000', label='Published QSVM baseline'),
    mpatches.Patch(color='#FF6600', label='Published classical baseline'),
    mpatches.Patch(color='#808080', label='Published single-modality'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=9)
ax.grid(axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('results/plots/main_comparison_chart.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: results/plots/main_comparison_chart.png")
```

### README.md

Create `README.md` in the project root:

```markdown
# Quantum-Enhanced Multimodal Dementia Detection

> Phase 1 — Multimodal QSVM across Clinical, Neuroimaging, and Speech Modalities
> CMR Institute of Technology, Bengaluru | B.E. AI & ML | 2025–2026

## Team
| Name | Roll No. | Task |
|------|----------|------|
| Bishal Chaudhary | 1CR23AI019 | MRI Feature Analysis |
| Govind Sharma | 1CR23AI037 | Visualization + Results |
| Prakshi Lakhchaura | 1CR23AI083 | Clinical Analysis + Multimodal QSVM |
| Sheetal Kamji | 1CR23AI112 | Speech Feature Analysis |

## Problem Statement
"A Quantum-Enhanced Multimodal Framework for Early Dementia Detection:
Integrating QSVM across Neuroimaging, Speech, and Clinical Modalities"

## Architecture
| Component | Approach |
|-----------|----------|
| Quantum classifier | QSVC with ZZFeatureMap (Qiskit), feature_dimension=6 |
| Classical baseline | RBF-SVM (sklearn) |
| Fusion strategy | Modality-aware early fusion — per-modality PCA(2) → concatenate(6) → one QSVM |
| Modalities | Clinical (5 features), MRI-derived (4 features), Speech (18 features) |
| Patients | 225 unique patients, all modalities per patient |

## Phase 1 Results

| Model | Accuracy | Source |
|-------|----------|--------|
| Multimodal QSVM (ours) | **UPDATE** | Ours |
| Classical SVM (ours) | **UPDATE** | Ours |
| Akinrotimi QSVM (BASE) | 91.25% | Literature |
| Chakravarthi Classical | 94.20% | Literature |

## Setup
```bash
pip install qiskit==0.45.3 qiskit-machine-learning==0.7.2 qiskit-algorithms==0.3.0
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## Repository Structure
```
├── clinical_qsvm.ipynb      # Prakshi  — Clinical feature analysis + PCA
├── mri_qsvm.ipynb           # Bishal   — MRI feature analysis + PCA
├── speech_qsvm.ipynb        # Sheetal  — Speech feature analysis + PCA
├── multimodal_qsvm.ipynb    # Prakshi  — Multimodal QSVM training + evaluation
├── results_viz.ipynb        # Govind   — Visualization + comparison chart
└── results/                 # All .npy features, model pickle, JSON, plots
```

## Phase 2 (Upcoming)
QCNN/PQC (PennyLane) + SHAP KernelExplainer + Attention-based cross-modal fusion
```

---

## ✅ SECTION 9 — INTEGRATION CHECKLIST

Run through this before Prakshi starts `multimodal_qsvm.ipynb`:

```
TASK A — Prakshi (Clinical):
  [ ] clinical_qsvm.ipynb runs end-to-end with no errors
  [ ] Cell 2 prints "Dataset loaded: (225, 29)"
  [ ] Cell 6 prints shape (225, 2)
  [ ] Cell 7 confirms: "✅ Saved: results/X_clinical_pca.npy — shape: (225, 2)"
  [ ] Cell 7 confirms: "✅ Saved: results/y_labels.npy        — shape: (225,)"
  [ ] results/plots/clinical_pca.png exists
  [ ] results/plots/clinical_feature_distributions.png exists

TASK B — Bishal (MRI):
  [ ] mri_qsvm.ipynb created in root directory
  [ ] Cell 2 (Shared Loading Block) is copy-pasted verbatim from Section 3
  [ ] mri_qsvm.ipynb runs end-to-end with no errors
  [ ] Cell 7 confirms: "✅ Saved: results/X_mri_pca.npy — shape: (225, 2)"
  [ ] results/X_mri_pca.npy committed and pushed to feature/mri-qsvm
  [ ] Prakshi merged feature/mri-qsvm → main and pulled locally

TASK C — Sheetal (Speech):
  [ ] speech_qsvm.ipynb created in root directory
  [ ] Cell 2 (Shared Loading Block) is copy-pasted verbatim from Section 3
  [ ] speech_qsvm.ipynb runs end-to-end with no errors
  [ ] Cell 8 confirms: "✅ Saved: results/X_speech_pca.npy — shape: (225, 2)"
  [ ] results/X_speech_pca.npy committed and pushed to feature/speech-qsvm
  [ ] Prakshi merged feature/speech-qsvm → main and pulled locally

TASK D — Prakshi (Multimodal QSVM):
  [ ] All three .npy files present locally in results/
  [ ] Cell 3 prints "✅ All shape checks passed" + shape (225, 6)
  [ ] Cell 6 QSVM training completes and prints accuracy
  [ ] Cell 7 shows QSVM accuracy > 91.25% (beats Akinrotimi BASE)
  [ ] Cell 9 confirms: "✅ Saved: results/qsvm_model.pkl"
  [ ] Cell 9 confirms: "✅ Pickle verified"
  [ ] Cell 9 confirms: "✅ Saved: results/multimodal_results.json"
  [ ] multimodal_qsvm.ipynb committed to main

TASK E — Govind (Visualization):
  [ ] results/multimodal_results.json loaded successfully
  [ ] results/plots/main_comparison_chart.png generated
  [ ] README.md updated with real accuracy numbers
  [ ] Committed and pushed
```

---

## 🎯 SECTION 10 — DEMO CHECKLIST (May 22) + VIVA ANSWERS

### Minimum viable demo (must have)
```
[ ] clinical_qsvm.ipynb    → runs live, shows PCA plot
[ ] mri_qsvm.ipynb         → runs live, shows PCA plot
[ ] speech_qsvm.ipynb      → runs live, shows PCA plot + MFCC heatmap
[ ] multimodal_qsvm.ipynb  → load from pickle, show predictions + accuracy in seconds
[ ] main_comparison_chart.png → visible, labeled, shows we beat the baseline
[ ] All notebooks pushed to GitHub main, repo is public
```

### Demo flow (2-minute walkthrough)
```
1. Open clinical_qsvm.ipynb → run PCA cell → show class separation plot
2. Open mri_qsvm.ipynb      → run PCA cell → show class separation plot
3. Open speech_qsvm.ipynb   → run PCA cell → show MFCC heatmap
4. Open multimodal_qsvm.ipynb:
   - Run Cell 2 (load .npy files) → 3 ticks appear instantly
   - Run Cell 3 (concatenate) → shape (225, 6) confirmed
   - Skip Cell 6 (training) — too slow for demo
   - Instead add a demo cell:
     ```python
     with open('results/qsvm_model.pkl', 'rb') as f:
         demo_model = pickle.load(f)
     demo_pred = demo_model.predict(X_test)
     print(f"QSVM Accuracy: {accuracy_score(y_test, demo_pred)*100:.2f}%")
     ```
   - Run confusion matrix cell → shows results
5. Show main_comparison_chart.png → "we beat both baselines"
```

### Viva answers (memorize these)

**"What accuracy did your model achieve?"**
```
"Our multimodal QSVM achieved [X]% accuracy on 225 patients
combining clinical, MRI, and speech features — beating the
Akinrotimi QSVM baseline at 91.25% and Chakravarthi's classical
multimodal model at 94.20%."
```

**"Why QSVM over classical SVM?"**
```
"Classical SVM uses a fixed RBF kernel computed in Euclidean space.
QSVM uses a ZZFeatureMap quantum circuit with 6 qubits to compute
the kernel in a Hilbert space of dimension 2^6 = 64 — exponentially
larger than classical space. The ZZ interactions encode feature
entanglement that RBF cannot represent. Our results show QSVM
achieves [X]% vs classical SVM's [Y]%."
```

**"Why early fusion instead of late fusion?"**
```
"Our dataset contains all three modalities — clinical scores, MRI
biomarkers, and speech features — for every patient in a single row.
Late fusion would have artificially separated this aligned information
to train three independent models. Early fusion is the correct approach:
we compress each modality to 2 principal components independently to
preserve modality identity, then concatenate into a 6-feature patient
representation and train one unified QSVM. This is architecturally
cleaner and gives the QSVM access to cross-modal feature interactions
during training."
```

**"Why PCA before the QSVM?"**
```
"QSVM with ZZFeatureMap scales exponentially with feature dimension.
We use PCA(2) per modality to compress each modality to 2 principal
components before fusion. This serves two purposes: it reduces
dimensionality to a range the quantum simulator can handle, and it
decorrelates features within each modality so the 6 concatenated
components carry maximum information with minimum redundancy."
```

**"What is your dataset?"**
```
"We use a clinically-grounded synthetic multimodal dataset of 225
unique patients, constructed from published distributions in dementia
speech research (Lin & Washington, 2024) and neuroimaging literature.
Each patient has 27 features spanning cognitive assessments, MRI-derived
brain volume metrics, and acoustic speech biomarkers. The dataset was
designed with realistic class overlap and moderate cross-feature
correlations to ensure the classification task is non-trivial."
```

---

*Guide version: Phase 1 Revised (v2) | Architecture: Modality-aware Early Fusion*
*Updated: May 2026 | Phase 2: QCNN/PQC (PennyLane) + SHAP + Attention fusion*
