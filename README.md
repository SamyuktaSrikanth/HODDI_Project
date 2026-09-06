# HyperAttDDI: Higher-Order Drug-Drug Interaction Prediction via Adverse-Event-Conditioned Hypergraph Attention

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3+-3C2179.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **HyperAttDDI**, a deep hypergraph neural network architecture designed for combinatorial polypharmacy adverse drug event prediction on the FDA FAERS-based **HODDI** benchmark.

---

## 🔬 Core Innovations

1. **Adverse-Event-Conditioned Attention Pooling (Exp C):**
   * *The Problem in Existing Models (HGNN-SA):* Previous hypergraph models discarded adverse event identities, creating **severe label contradiction** (e.g., $\{D_1, D_2\}$ causing bleeding was labeled $1$, while $\{D_1, D_2\}$ not causing hypotension was labeled $0$, forcing models to learn blurry averages).
   * *HyperAttDDI Solution:* Conditions hyperedge attention queries directly on 768d SapBERT adverse event embeddings. The model learns dynamic attention weights that highlight different drug culprits depending on the specific adverse reaction queried.
2. **Multimodal Drug Attention Fusion (Exp D):**
   * Integrates pre-trained continuous chemical language representations (**ChemBERTa**, 768d) with high-dimensional pharmacological knowledge (**3,793 targets, 1,270 enzymes, 215 transporters, and ATC codes**) through learned attention gating.
3. **Higher-Order Multi-Head Hypergraph Convolutions:**
   * 2-layer multi-head hypergraph message passing that propagates structural and functional information across multi-drug co-prescription networks ($k \ge 3$).

---

## 📐 Model Architecture

```
[Drug SMILES (768d)]  ──► SMILES Encoder (Linear->ReLU->Dropout) ──► h_smiles (256d) ──┐
                                                                                         ├──► [Attention Fusion] ──► X (256d)
[Drug Bio (1024d)]    ──► Bio Encoder (Linear->ReLU->Dropout)    ──► h_bio    (128d) ──┘        │
                                                                                                ▼
[Hypergraph Incidence H] ─────────────────────────────────────────────────────────────► [2-Layer HypergraphConv]
                                                                                                │
                                                                                                ▼ (Node States X_2)
[Side Effect (SapBERT 768d)] ──► SE Encoder ──► h_se (256d) ──┐                                 │
                                                               ├──► [SE-Conditioned Attention Pooling]
[Mean Hyperedge State h_e]   ─────────────────────────────────┘        │ (Query: [h_e || h_se])
                                                                       ▼
                                                              [Combination State h_combo (256d)]
                                                                       │
                                                                       ▼
                                                          [Decoder: [h_combo || h_se]] ──► P(y=1)
```

---

## 🏆 Master Benchmark Results

All models evaluated on the exact same 41-quarter temporal chronological split (29 train, 6 validation, 6 test quarters; 70:15:15 ratio) on the HODDI evaluation subset:

| Stage / Experiment | Architecture Description | Features | Attention Pooling? | Precision | Recall | F1 Score | AUC | PRAUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1 (MLP Baseline)** | 2-layer MLP on flattened vector | SMILES + SapBERT | No | 0.7896 | 0.8510 | 0.8191 | 0.8923 | 0.8671 |
| **GCN (Published Benchmark)** | Pairwise Graph Decomposition | SMILES only | No | 0.7450 | 0.8140 | 0.7780 | 0.8290 | 0.8050 |
| **GAT (Published Benchmark)** | Pairwise Graph Attention | SMILES only | No | 0.7430 | 0.8870 | 0.8090 | 0.8510 | 0.7890 |
| **Stage 2 (Exp A - HGNN-SA)** | 1-layer HGNN baseline (reproduced) | SMILES only | No | 0.8571 | 0.9311 | 0.8926 | 0.9211 | 0.8760 |
| **Stage 3 (Exp B - HyperAttDDI No SE)** | 2-layer Hypergraph + Generic Attention | SMILES only | Generic | 0.8384 | 0.9405 | 0.8865 | 0.9372 | 0.9175 |
| **Stage 4 (Exp C - HyperAttDDI + SE)** | 2-layer Hypergraph + SE-Conditioned Attn | SMILES + SapBERT | **SE-Conditioned** | 0.8724 | **0.9511** | 0.9101 | **0.9518** | **0.9285** |
| **Stage 5 (Exp D - HyperAttDDI + Bio)** | Multimodal Bio Fusion + SE-Conditioned Attn | SMILES + Bio + SapBERT | **SE-Conditioned** | **0.8766** | 0.9469 | **0.9104** | 0.9505 | 0.9252 |
| **Stage 6 (Exp E - Mean Pool Ablation)** | 2-layer Hypergraph + Naive Mean Pooling | SMILES only | Mean Pool | 0.8290 | 0.9380 | 0.8801 | 0.9192 | 0.8965 |

---

## 📊 Ablation Studies

### 1. Adverse-Event Conditioning (Exp C vs. Exp B)
* Without SE conditioning (Exp B), the model cannot distinguish between different adverse reactions, dropping Precision to **0.8384**.
* Injecting SapBERT adverse event conditioning (Exp C) resolves target ambiguity, boosting Precision to **0.8724 (+3.40%)**, AUC to **0.9518 (+1.46%)**, and PRAUC to **0.9285 (+1.10%)**.

### 2. Attention Pooling vs. Mean Pooling (Exp B vs. Exp E)
* Replacing learned attention pooling with naive mean pooling drops AUC from **0.9372 to 0.9192 (-1.80%)** and PRAUC from **0.9175 to 0.8965 (-2.10%)**, confirming that member drugs contribute unequally to adverse reactions.

### 3. Performance Stratified by Interaction Order ($k = 2, 3, 4, 5, 6+$)

| Interaction Order ($k$) | Test Sample Count | MLP (AUC) | GAT (AUC) | HGNN-SA (AUC) | HyperAttDDI (AUC) | HyperAttDDI Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$k = 2$ (Pairs)** | 1,404 | 0.8935 | 0.8520 | 0.9250 | **0.9534** | **+2.84%** |
| **$k = 3$ (Triplets)** | 1,328 | 0.8912 | 0.8490 | 0.9230 | **0.9528** | **+2.98%** |
| **$k = 4$ (Quartets)** | 1,046 | 0.8870 | 0.8350 | 0.9190 | **0.9512** | **+3.22%** |
| **$k = 5$ (Quintets)** | 874 | 0.8790 | 0.8120 | 0.9120 | **0.9495** | **+3.75%** |
| **$k \ge 6$ (Sextets+)** | 1,934 | 0.8650 | 0.7740 | 0.8980 | **0.9468** | **+4.88%** |

> **Key Finding:** While pairwise models (GAT) collapse by **7.8%** as combination size increases, HyperAttDDI maintains robust stability (>0.946 AUC), widening its advantage to **+4.88%** at $k \ge 6$.

---

## 📁 Repository Directory Structure

```
HODDI_Project/
├── README.md                           # Main documentation & reproduction guide
├── requirements.txt                    # Dependencies
├── .gitignore                          # Standard git ignore
│
├── data/                               # Complete dataset to run all experiments (~136 MB)
│   ├── drug_embeddings_768d.pt         # Precomputed ChemBERTa embeddings (29.4 MB)
│   ├── bio_features_1024d.pt           # Targets, enzymes, transporters, ATC (38.3 MB)
│   ├── dictionary/
│   │   ├── Side_effects_unique.csv     # 7,350 MedDRA CUIs with SapBERT embeddings (60.6 MB)
│   │   └── Drugbank_ID_SMILE_all_structure links.csv (5.3 MB)
│   └── evaluation_subset/
│       └── subset_drug2-8_SE5-50/      # 41 quarterly positive/negative FAERS CSVs (7.1 MB)
│
├── experiments/
│   ├── exp_a_baseline_hgnn/            # Exp A: HGNN-SA Baseline reproduction
│   ├── exp_b_hyperattddi_no_se/        # Exp B: HyperAttDDI Architecture (No SE)
│   ├── exp_c_hyperattddi_se/           # Exp C: SE-Conditioned Attention Pooling
│   ├── exp_d_hyperattddi_bio/          # Exp D: Multimodal Biological Feature Fusion
│   ├── exp_e_mean_pool_ablation/       # Exp E: Mean Pooling Ablation
│   └── exp_f_stratified_k/             # Exp F: Stratified Evaluation across k=2..6+
│
├── notebooks/                          # 5 Standalone Kaggle-ready Jupyter Notebooks
│   ├── 01_kaggle_exp_b_no_se.ipynb
│   ├── 02_kaggle_exp_c_with_se.ipynb
│   ├── 03_kaggle_exp_d_multimodal_bio.ipynb
│   ├── 04_kaggle_exp_e_mean_pool_ablation.ipynb
│   └── 05_kaggle_exp_f_stratified_eval.ipynb
│
└── docs/
    ├── HyperAttDDI_Manuscript.md       # Complete research paper manuscript
    └── fairness_audit.md               # Audit verifying fair comparison with HGNN-SA
```

---

## 🚀 Step-by-Step Reproduction Guide

### 1. Installation
```bash
git clone https://github.com/SamyuktaSrikanth/HODDI_Project.git
cd HODDI_Project
pip install -r requirements.txt
```

### 2. Running Locally

#### Experiment A: HGNN-SA Baseline
```bash
cd experiments/exp_a_baseline_hgnn
python run_hgnn_sa.py --epochs 100 --batch_size 64
```

#### Experiment B: HyperAttDDI Architecture (No SE)
```bash
cd experiments/exp_b_hyperattddi_no_se
python run_exp_b.py --epochs 100 --batch_size 64
```

#### Experiment C: HyperAttDDI + SE (Core Model)
```bash
cd experiments/exp_c_hyperattddi_se
python run_exp_c.py --epochs 100 --batch_size 64
```

#### Experiment D: Multimodal Biological Feature Fusion
```bash
cd experiments/exp_d_hyperattddi_bio
python run_exp_d.py --epochs 100 --batch_size 64
```

#### Experiment E: Mean Pooling Ablation
```bash
cd experiments/exp_e_mean_pool_ablation
python run_exp_e.py --epochs 100 --batch_size 64
```

#### Experiment F: Stratified Evaluation across $k$
```bash
cd experiments/exp_f_stratified_k
python stratified_eval_k.py
```

### 3. Running on Kaggle GPU

All notebooks in the `notebooks/` directory are self-contained:
1. Create a new notebook on Kaggle with a **T4 GPU** accelerator.
2. Upload the desired notebook from `notebooks/` (e.g. `02_kaggle_exp_c_with_se.ipynb`).
3. Attach the `data/` folder as input.
4. Run all cells (~4–5 minutes per 100-epoch training).

---

## 🩺 Clinical Case Studies & Interpretability

HyperAttDDI provides faithful, interpretable attention weights across multi-drug regimens:

### Case Study: Cardiovascular Triple Therapy
* **Regimen:** Clopidogrel (P2Y12 inhibitor) + Dabigatran (thrombin inhibitor) + Aspirin (COX-1 inhibitor)
* **Renal Failure Risk:** Predicted probability = **`0.9990`** (Dabigatran attention = **`0.5087`**)
* **Bleeding Risk:** Predicted probability = **`0.8629`** (Dabigatran attention = **`0.5087`**)
* **Non-Events (Hypotension, Cardiac Arrest):** Predicted probability = **`0.0000`**
* **Clinical Rationale:** Dabigatran is >80% renally cleared. Impaired clearance triggers fatal bleeding. The model correctly identifies Dabigatran as the dominant culprit while suppressing false alarms for unrelated conditions.

---

## 📜 Research Paper Manuscript

The complete manuscript detailing theoretical proofs, mathematical derivations, ablation studies, and clinical discussions is available at:
👉 **[`docs/HyperAttDDI_Manuscript.md`](docs/HyperAttDDI_Manuscript.md)**

---

## 📄 License
This project is licensed under the MIT License.
