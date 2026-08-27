# Task Implementation List: HyperAttDDI Project

**Project Name:** Deep Learning-Based Drug–Drug Interaction Prediction (HyperAttDDI)  
**Reference Document:** [SPECIFICATION.md](file:///d:/7thSem%20-%20Copy/HODDI_Project/SPECIFICATION.md) & `MiniProject_Review2_PPT.pdf`

---

## Task Progress Overview

- [ ] **Phase 1: Environment Setup & Data Pipeline (M1)** `(0/4 Completed)`
- [ ] **Phase 2: Core Neural Network Modules (M2 - M4)** `(0/4 Completed)`
- [ ] **Phase 3: Prediction Head, Loss Function & Training Pipeline (M5)** `(0/3 Completed)`
- [ ] **Phase 4: Evaluation Metrics & Cross-Validation Framework** `(0/3 Completed)`
- [ ] **Phase 5: Systematic Ablation Study Suite** `(0/4 Completed)`
- [ ] **Phase 6: Clinical Interpretability & Culprit Drug Profiling** `(0/2 Completed)`
- [ ] **Phase 7: Project Documentation & Verification** `(0/2 Completed)`

---

## Phase 1: Environment Setup & Data Pipeline (M1)

### Task 1.1: Project Directory Structure & Virtual Environment Configuration
- [ ] Create repository folder layout (`data/`, `models/`, `utils/`, `experiments/`).
- [ ] Setup `requirements.txt` / `pyproject.toml` with dependencies (`torch`, `torch-geometric`, `rdkit`, `scikit-learn`, `pandas`, `numpy`).
- [ ] Verify PyTorch CUDA execution & RDKit installation.
- **Dependencies:** None
- **Verification:** Environment test script imports all libraries without error.

### Task 1.2: Raw Data Ingestion & SMILES Processing (Structure Features)
- [ ] Load HODDI drug list and canonical SMILES strings.
- [ ] Implement RDKit extraction script generating 881-dimensional PubChem binary molecular fingerprints.
- [ ] Handle missing SMILES strings and invalid molecular structures gracefully.
- **Dependencies:** Task 1.1
- **Verification:** Matrix $X^{(1)}$ generated with shape $(N_{\text{drugs}}, 881)$ containing binary values.

### Task 1.3: Biological Feature Ingestion (Targets, Enzymes, Pathways)
- [ ] Parse KEGG database profiles for target proteins ($d_2 = 780$).
- [ ] Parse KEGG database profiles for liver enzymes / CYP450 ($d_3 = 129$).
- [ ] Parse KEGG database profiles for biological pathways ($d_4 = 253$).
- [ ] Construct total 2,043-dimensional composite feature vector per drug.
- **Dependencies:** Task 1.1
- **Verification:** Feature matrices $X^{(2)} (N \times 780)$, $X^{(3)} (N \times 129)$, and $X^{(4)} (N \times 253)$ constructed and verified against ground truth IDs.

### Task 1.4: HODDI Prescription Hypergraph & Multi-Label Construct
- [ ] Parse HODDI multi-drug prescription dataset ($\ge 3$ drugs per combination).
- [ ] Build Hypergraph Incidence Matrix $H \in \{0,1\}^{N_{\text{drugs}} \times M_{\text{prescriptions}}}$.
- [ ] Format target multi-label side-effect matrix $Y \in \{0,1\}^{M \times 4569}$.
- **Dependencies:** Tasks 1.2, 1.3
- **Verification:** PyTorch Geometric / custom Hypergraph Data object initialized cleanly with valid node index mappings.

---

## Phase 2: Core Neural Network Modules (M2 - M4)

### Task 2.1: Feature Encoding Sub-MLPs Module (`models/encoders.py`)
- [ ] Implement 4 independent 2-layer MLPs for structure (881), target (780), enzyme (129), and pathway (253) features.
- [ ] Include `Linear` -> `ReLU` -> `Dropout(0.2)` -> `Linear` transformations mapping each feature to 128 dimensions.
- **Dependencies:** Task 1.4
- **Verification:** Unit test checking input shapes $[B, d_k] \to [B, 128]$ for all 4 encoders.

### Task 2.2: Attention-Based Feature Fusion Module (`models/fusion.py`)
- [ ] Build 2-layer Attention Network computing scalar importance weights $\alpha_{i,k}$ across the 4 feature representations for drug $i$.
- [ ] Implement softmax normalization over feature dimensions $\sum_{k=1}^4 \alpha_{i,k} = 1$.
- [ ] Compute fused 128-dimensional embedding vector $\mathbf{z}_i$.
- **Dependencies:** Task 2.1
- **Verification:** Output tensor shape is $[N_{\text{drugs}}, 128]$ and sum of $\alpha$ weights equals 1.0 per drug.

### Task 2.3: Hypergraph Attention Network (HAN) Module (`models/hypergraph.py`)
- [ ] Implement 2-layer Hypergraph Attention layer with hyperedge message passing.
- [ ] Calculate drug-level culprit attention weight $\beta_{i,j}$ for drug $i$ in hyperedge $e_j$.
- [ ] Implement Node-to-Hyperedge aggregation and Hyperedge-to-Node update steps.
- **Dependencies:** Task 2.2
- **Verification:** HAN forward pass updates drug vectors $\mathbf{z}_i \to \mathbf{z}_i'$ without NaN/Inf gradients.

---

## Phase 3: Prediction Head, Loss Function & Training Pipeline (M5)

### Task 3.1: Attention Pooling & Side Effect Predictor (`models/predictor.py`)
- [ ] Implement Attention Pooling to merge prescription drug vectors into 128-dimensional hyperedge vector $\mathbf{u}_j$.
- [ ] Implement 2-layer Classification MLP ($128 \to 512 \to 4569$) with Sigmoid activation.
- **Dependencies:** Task 2.3
- **Verification:** Output probability tensor matches $[M_{\text{batch}}, 4569]$.

### Task 3.2: Custom Focal Loss Implementation (`utils/losses.py`)
- [ ] Implement Focal Loss function $\mathcal{L}_{\text{Focal}}$ with focusing parameter $\gamma = 2.0$ and class balancing factor $\alpha$.
- [ ] Support numerical stability using `torch.logsumexp` / `binary_cross_entropy_with_logits`.
- **Dependencies:** Task 3.1
- **Verification:** Focal Loss gradient test confirms higher weight on hard misclassified samples vs easy negatives.

### Task 3.3: End-to-End Model Assembly & Training Script (`train.py`)
- [ ] Assemble `HyperAttDDI` unified `nn.Module` integrating M2, M3, M4, and M5.
- [ ] Configure Adam optimizer with learning rate scheduler (Cosine Annealing).
- [ ] Implement training step, gradient clipping, logging, and model checkpoint saving.
- **Dependencies:** Tasks 3.1, 3.2
- **Verification:** Successful execution of 1 training epoch with decreasing loss.

---

## Phase 4: Evaluation Metrics & Cross-Validation Framework

### Task 4.1: Evaluation Metrics Engine (`utils/metrics.py`)
- [ ] Implement Multi-label Subset Accuracy (Jaccard Similarity).
- [ ] Implement Macro-F1 score calculation (unweighted average across 4,569 classes).
- [ ] Implement Micro-F1 score calculation.
- [ ] Implement ROC-AUC and PR-AUC evaluators using Scikit-learn.
- **Dependencies:** Task 3.3
- **Verification:** Metrics engine correctly computes all 5 metrics on dummy predictions and ground truth tensors.

### Task 4.2: 5-Fold Stratified Cross-Validation Harness
- [ ] Implement 5-fold prescription dataset splitter maintaining multi-label side-effect proportions.
- [ ] Build cross-validation loop executing full training and evaluation cycles across all 5 folds.
- [ ] Compute mean and standard deviation for Accuracy, Macro-F1, Micro-F1, ROC-AUC, PR-AUC.
- **Dependencies:** Task 4.1
- **Verification:** Complete 5-fold cross-validation run completes and logs summary statistics.

---

## Phase 5: Systematic Ablation Study Suite

### Task 5.1: Ablation Experiment 1 - Fusion Mechanism
- [ ] Implement baseline models replacing Attention Fusion with (a) Equal Weight Concatenation and (b) Mean Pooling.
- [ ] Run 5-fold CV evaluation and compare against full `HyperAttDDI`.
- **Dependencies:** Task 4.2
- **Verification:** Performance metrics recorded in ablation comparison table.

### Task 5.2: Ablation Experiment 2 - Graph Interaction Topology
- [ ] Implement baseline models replacing Hypergraph HAN with (a) Pairwise GCN/GAT graph and (b) Plain MLP without graph structure.
- [ ] Evaluate performance impact on multi-drug combinations ($\ge 3$ drugs).
- **Dependencies:** Task 4.2
- **Verification:** Metric breakdown demonstrates hypergraph superiority for higher-order combinations.

### Task 5.3: Ablation Experiment 3 - Loss Function Comparison
- [ ] Train model variant using standard Binary Cross-Entropy (BCE) loss instead of Focal Loss ($\gamma = 2.0$).
- [ ] Compare Macro-F1 scores on rare vs common side effects.
- **Dependencies:** Task 4.2
- **Verification:** Macro-F1 improvement confirmed for Focal Loss on long-tail rare side effects.

### Task 5.4: Ablation Experiment 4 - Input Feature Modalities
- [ ] Train single-modality baseline models (Structure-only, Target-only, Enzyme-only, Pathway-only).
- [ ] Measure multi-feature fusion gain over single-modality inputs.
- **Dependencies:** Task 4.2
- **Verification:** Complete ablation comparison table logged for all feature combinations.

---

## Phase 6: Clinical Interpretability & Culprit Drug Profiling

### Task 6.1: Culprit Drug Identification & Attention Weight Analysis
- [ ] Extract hypergraph attention weights $\beta_{i,j}$ for sample prescriptions.
- [ ] Rank drugs within 3+ drug prescriptions to identify primary culprit drug driving predicted side effects.
- [ ] Validate culprit scores against known pharmacological case studies.
- **Dependencies:** Task 4.2
- **Verification:** Culprit distribution outputs valid probability weights (e.g. $[0.70, 0.20, 0.10]$).

### Task 6.2: Feature Mechanism Interpretability ($\alpha$ Weight Inspection)
- [ ] Analyze feature attention weights $\alpha_{i,k}$ across different drug classes.
- [ ] Verify that metabolic drugs assign higher $\alpha$ to enzyme features while receptor binders favor target features.
- **Dependencies:** Task 6.1
- **Verification:** Feature weight heatmaps generated showing per-drug mechanism specializations.

---

## Phase 7: Project Documentation & Verification

### Task 7.1: Results Visualization & Summary Artifacts
- [ ] Generate ROC curves, Precision-Recall curves, and Confusion matrices.
- [ ] Draft comprehensive results table comparing `HyperAttDDI` against literature baselines (DeepDDI, DDIMDL, BioChemDDI, SSI-DDI, DSN-DDI).
- **Dependencies:** Tasks 5.1 - 5.4, 6.2
- **Verification:** All plot figures and summary markdown tables saved to `results/`.

### Task 7.2: Final Code Review & Walkthrough Documentation
- [ ] Run static code checks and clean docstrings across all modules.
- [ ] Generate final `walkthrough.md` documenting implementation completion and verification results.
- **Dependencies:** Task 7.1
- **Verification:** End-to-end repository test passes cleanly.
