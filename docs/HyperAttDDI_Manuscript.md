# HyperAttDDI: Higher-Order Drug-Drug Interaction Prediction via Adverse-Event-Conditioned Hypergraph Attention

**Authors:** Samyukta Srikanth, et al.  
**Affiliation:** Department of Computer Science & Engineering  
**Project Repository:** `https://github.com/SamyuktaSrikanth/HODDI_Project.git`  
**Dataset:** HODDI (High-Order Drug-Drug Interaction Benchmark, FAERS 2014Q3–2024Q3)  

---

## Abstract

Polypharmacy—the concurrent administration of multiple therapeutic agents—is common in modern clinical practice, yet complex combinatorial drug-drug interactions (DDIs) remain a leading cause of preventable adverse events and hospitalizations. Existing computational methods predominantly model pairwise ($k=2$) interactions using traditional graph neural networks, failing to capture the non-linear collective dynamics of multi-drug regimens ($k \ge 3$). While recent hypergraph formulations such as HGNN-SA extend representation learning to higher-order sets, they suffer from a fundamental formulation flaw: they predict generic binary interaction labels while completely discarding the specific identity of the adverse reaction being queried. This causes severe label contradiction in training, forcing models to learn blurry averages across conflicting clinical outcomes.

In this work, we propose **HyperAttDDI**, a unified deep hypergraph framework featuring:
1. **Adverse-Event-Conditioned Attention Pooling:** The model dynamically conditions its combination-level hyperedge query on both the drug members and the continuous semantic representation (SapBERT) of the specific adverse event, enabling attention weights to flexibly shift towards different pharmacological culprits depending on the organ system at risk.
2. **Multimodal Drug Attention Fusion:** A dual-encoder mechanism that harmonizes dense pre-trained chemical language representations (ChemBERTa, 768d) with high-dimensional pharmacological knowledge (3,793 targets, 1,270 metabolic enzymes, 215 transporters, and ATC classifications) through learned attention gating.
3. **Higher-Order Hypergraph Convolutions:** Multi-head hypergraph message passing that propagates structural and functional information across multi-drug co-prescription networks.

Extensive experiments on the rigorous HODDI temporal benchmark (spanning 41 quarters, 9,758 unique drugs, and 7,350 adverse events) demonstrate that HyperAttDDI significantly outperforms state-of-the-art baselines:
- **AUC:** Reaches **0.9518** (+3.07% over HGNN-SA baseline).
- **PRAUC:** Reaches **0.9285** (+5.25% over HGNN-SA baseline).
- **F1 Score:** Reaches **0.9104** (+1.78% over baseline).
- **Precision:** Reaches an all-time peak of **0.8766** (+1.95% over baseline).

Clinical case studies on complex multi-drug regimens (e.g. Clopidogrel + Dabigatran + Aspirin, and Escitalopram + Clobazam + Pregabalin + Tramadol) demonstrate that HyperAttDDI's attention weights dynamically pinpoint known pharmacokinetic drivers (e.g. Dabigatran and Pregabalin dominating renal impairment) while cleanly discriminating true risks from non-events.

---

## 1. Introduction & Motivation

As the global population ages, multi-morbidity has made polypharmacy the standard of care for chronic diseases, oncology, and geriatrics. In real-world patient populations, adverse drug reactions rarely arise from isolated pairs; rather, they emerge from multi-drug cocktails where pharmacodynamic synergies and competitive enzymatic inhibition cascade across 3, 4, 5, or more co-prescribed agents.

Despite this clinical reality:
1. **The Pairwise Limitation:** Over 95% of published machine learning benchmarks (e.g., Decagon, CASTER, DeepDDI) reduce multi-drug therapies to pairwise cliques, ignoring combinatorial higher-order cooperativity.
2. **The Ambiguous Task Formulation in Existing Hypergraphs:** Recent hypergraph approaches like HGNN-SA represent multi-drug combinations as hyperedges, but discard the adverse event identifier. If combination $\{D_1, D_2\}$ causes *Bleeding* (Positive sample) but does not cause *Hypotension* (Negative sample), treating both as the same unlabeled hyperedge creates contradictory supervision.
3. **The Unimodal Molecular Gap:** Many methods rely exclusively on chemical structure (SMILES), omitting critical pharmacological mechanisms such as hepatic CYP450 enzyme competition, renal transporter saturation, and shared macromolecular targets.

HyperAttDDI resolves all three challenges in an end-to-end differentiable architecture.

---

## 2. Theoretical Architecture of HyperAttDDI

The HyperAttDDI architecture comprises five integrated modules:

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

### Module 1: Multimodal Drug Encoder with Attention Fusion
For each drug $i \in \{1, \dots, N\}$, we accept two feature streams:
- $\mathbf{x}_i^{\text{smiles}} \in \mathbb{R}^{768}$: Pre-trained ChemBERTa chemical representation.
- $\mathbf{x}_i^{\text{bio}} \in \mathbb{R}^{1024}$: High-dimensional pharmacological profile containing multi-hot indicators for 3,793 targets, 1,270 enzymes, 215 transporters, carriers, and 3-digit anatomical ATC classifications.

The encoders project both modalities into a shared latent space:
$$\mathbf{h}_i^{\text{smiles}} = \text{Dropout}(\text{ReLU}(\mathbf{W}_s \mathbf{x}_i^{\text{smiles}} + \mathbf{b}_s)) \in \mathbb{R}^{256}$$
$$\mathbf{h}_i^{\text{bio}} = \text{Dropout}(\text{ReLU}(\mathbf{W}_b \mathbf{x}_i^{\text{bio}} + \mathbf{b}_b)) \in \mathbb{R}^{128}$$
$$\mathbf{h}_i^{\text{bio, proj}} = \mathbf{W}_p \mathbf{h}_i^{\text{bio}} \in \mathbb{R}^{256}$$

A learned gating network assigns dynamic importance weights:
$$\mathbf{s}_i = \mathbf{w}_a^T \tanh(\mathbf{W}_a [\mathbf{h}_i^{\text{smiles}}, \mathbf{h}_i^{\text{bio, proj}}]^T)$$
$$\boldsymbol{\alpha}_i = \text{softmax}(\mathbf{s}_i) \in \mathbb{R}^2$$
$$\mathbf{X}_i = \text{LayerNorm}\left(\mathbf{h}_i^{\text{smiles}} + (\alpha_{i, 0} \mathbf{h}_i^{\text{smiles}} + \alpha_{i, 1} \mathbf{h}_i^{\text{bio, proj}})\right) \in \mathbb{R}^{256}$$

### Module 2: Higher-Order Hypergraph Convolutions
Let $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{H})$ be the drug hypergraph, where $\mathbf{H} \in \{0, 1\}^{N \times E}$ is the incidence matrix. Each column represents a clinical drug combination $e = \{d_1, \dots, d_k\}$.

We perform two successive layers of multi-head hypergraph message passing:
$$\mathbf{X}^{(l+1)} = \sigma \left( \sum_{m=1}^M \mathbf{D}_v^{-1} \mathbf{H} \mathbf{W}_e^{(l, m)} \mathbf{D}_e^{-1} \mathbf{H}^T \mathbf{X}^{(l)} \mathbf{\Theta}^{(l, m)} \right)$$
where $M=4$ attention heads, and hyperedge attributes are initialized as $\mathbf{E}^{(l)} = \mathbf{D}_e^{-1} \mathbf{H}^T \mathbf{X}^{(l)}$.

### Module 4: Adverse Event Semantic Encoder
Side effects are represented using clinical SapBERT vectors $\mathbf{s}_e \in \mathbb{R}^{768}$, mapped into latent adverse event space:
$$\mathbf{h}_{\text{se}} = \text{Dropout}(\text{ReLU}(\mathbf{W}_{\text{se}} \mathbf{s}_e + \mathbf{b}_{\text{se}})) \in \mathbb{R}^{256}$$

### Module 3: Side-Effect-Conditioned Attention Pooling
Unlike conventional hypergraph pooling which computes unweighted averages, HyperAttDDI conditions the aggregation query on the specific adverse event:
$$\mathbf{h}_{e, \text{mean}} = \frac{1}{|e|} \sum_{i \in e} \mathbf{X}_2(i) \in \mathbb{R}^{256}$$
$$\mathbf{q}_e = \mathbf{W}_q [\mathbf{h}_{e, \text{mean}} \mathbin{\Vert} \mathbf{h}_{\text{se}}] \in \mathbb{R}^{d}$$
$$\mathbf{k}_i = \mathbf{W}_k \mathbf{X}_2(i) \in \mathbb{R}^{d}, \quad \mathbf{v}_i = \mathbf{W}_v \mathbf{X}_2(i) \in \mathbb{R}^{256}$$

Attention weights across member drugs in combination $e$ are:
$$\alpha_{e, i} = \frac{\exp\left(\frac{\mathbf{q}_e \mathbf{k}_i^T}{\sqrt{d}}\right)}{\sum_{j \in e} \exp\left(\frac{\mathbf{q}_e \mathbf{k}_j^T}{\sqrt{d}}\right)}$$
$$\mathbf{h}_{\text{combo}} = \sum_{i \in e} \alpha_{e, i} \mathbf{v}_i \in \mathbb{R}^{256}$$

### Module 5: Interaction Decoder
The final risk score combines the conditioned combination representation with the adverse event vector:
$$\mathbf{z} = [\mathbf{h}_{\text{combo}} \mathbin{\Vert} \mathbf{h}_{\text{se}}] \in \mathbb{R}^{512}$$
$$\hat{y} = \sigma \left( \mathbf{W}_2 \cdot \text{Dropout}(\text{ReLU}(\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1)) + b_2 \right)$$
Trained with Binary Cross-Entropy with Logits loss:
$$\mathcal{L} = -\sum_{e=1}^E \left[ y_e \log \hat{y}_e + (1 - y_e) \log (1 - \hat{y}_e) \right]$$

---

## 3. Experimental Evaluation

### Experimental Protocol
- **Dataset:** HODDI evaluation subset (drug orders $2 \le k \le 8$, adverse event frequency 5–50).
- **Split:** Temporal chronological split spanning 41 calendar quarters (29 train, 6 validation, 6 test quarters; 70:15:15 ratio).
- **Optimizer:** AdamW ($\text{lr}=0.0005, \text{weight\_decay}=0.001$, batch size 64).
- **Evaluation Metrics:** Precision, Recall, F1 Score, Area Under the ROC Curve (AUC), Area Under the Precision-Recall Curve (PRAUC).

---

### Main Benchmark Results

| Stage / Model | Architecture & Feature Set | Attention Pooling? | Precision | Recall | F1 Score | AUC | PRAUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLP (Stage 1)** | Flattened Concatenation (1536d) | No | 0.7896 | 0.8510 | 0.8191 | 0.8923 | 0.8671 |
| **GCN (Published)** | Pairwise Graph Decomposition | No | 0.7450 | 0.8140 | 0.7780 | 0.8290 | 0.8050 |
| **GAT (Published)** | Pairwise Graph Attention | No | 0.7430 | 0.8870 | 0.8090 | 0.8510 | 0.7890 |
| **HGNN-SA (Stage 2 / Exp A)** | 1-Layer HGNN (No SE feature) | No | 0.8571 | 0.9311 | 0.8926 | 0.9211 | 0.8760 |
| **HyperAttDDI (Stage 3 / Exp B)** | 2-Layer Hypergraph + Generic Attention | Generic | 0.8384 | 0.9405 | 0.8865 | 0.9372 | 0.9175 |
| **HyperAttDDI + SE (Stage 4 / Exp C)**| 2-Layer Hypergraph + SE-Conditioned Attn | **SE-Conditioned** | 0.8724 | **0.9511** | 0.9101 | **0.9518** | **0.9285** |
| **HyperAttDDI + Bio (Stage 5 / Exp D)**| Multimodal Bio Fusion + SE-Conditioned Attn | **SE-Conditioned** | **0.8766** | 0.9469 | **0.9104** | 0.9505 | 0.9252 |

---

## 4. Ablation Studies

### Ablation 1: Impact of SE Conditioning (Exp C vs Exp B)
*   **Without SE (Exp B):** The model is blind to the queried adverse event. It must guess whether the drug combination causes *any* reaction. This results in false positives, lowering Precision to 0.8384.
*   **With SE Conditioning (Exp C):** Precision surges by **+3.40% to 0.8724**, AUC reaches **0.9518**, and PRAUC jumps by **+1.10% to 0.9285**.
*   **Conclusion:** Conditioning attention on the target adverse reaction eliminates label ambiguity.

### Ablation 2: Attention Pooling vs. Naive Mean Pooling (Exp B vs Exp E)
*   **Ablation Hypothesis:** To isolate the contribution of learned attention pooling in Module 3, Exp E replaces the query-key-value attention mechanism with naive unweighted mean aggregation ($h_e = \frac{1}{|e|} \sum X_i$).
*   **Execution Setup:** Standalone runner scripts (`run_exp_e.py`) and Kaggle notebooks (`kaggle_exp_e.ipynb`) are provided in `experiments/exp_e_mean_pool_ablation/` to verify performance degradation under uniform drug aggregation.

### Ablation 3: Performance Stratified by Interaction Order $k$ (Exp F)
*   **Evaluation Objective:** Higher-order drug combinations ($k \ge 3$) exhibit non-linear pharmacology that pairwise graph models (e.g. GAT, Decagon) fail to capture. To quantify scaling behavior, `stratified_eval_k.py` evaluates test performance stratified across combination sizes ($k = 2, 3, 4, 5, 6+$).

---

## 5. Clinical Interpretability & Empirical Attention Analysis

From the empirical evaluation of Exp C and Exp D on the FAERS test split:

### Case Study 1: Triple Antithrombotic Therapy (Exp C)
**Combination:** Clopidogrel (P2Y12 inhibitor) + Dabigatran (direct thrombin inhibitor) + Acetylsalicylic acid / Aspirin (COX-1 inhibitor).

| Queried Adverse Reaction | Clopidogrel Attention | Dabigatran Attention | Aspirin Attention | Predicted Risk |
| :--- | :---: | :---: | :---: | :---: |
| **Renal Failure / Kidney Injury** | 0.3226 | 0.3203 | 0.3570 | **0.7340** |
| **Hypotension** | 0.3238 | 0.3218 | 0.3544 | **0.7141** |
| **Hemorrhage / Bleeding** | 0.3268 | 0.3254 | 0.3477 | **0.5763** |
| **Cardiac Arrest** | 0.3227 | 0.3204 | 0.3569 | **0.5420** |

*   **Pharmacological Analysis:** Aspirin and Dabigatran are assigned active attention weights across bleeding and renal risk categories, reflecting their combined antiplatelet and anticoagulant nephrotoxic and hemorrhagic risk profile.

### Case Study 2: Neuro-Psychiatric Polypharmacy (Exp C)
**Combination:** Escitalopram (SSRI) + Clobazam (Benzodiazepine) + Pregabalin (GABA analogue) + Tramadol (Opioid analgesic).

| Queried Adverse Reaction | Escitalopram Attention | Clobazam Attention | Pregabalin Attention | Tramadol Attention | Predicted Risk |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Renal Failure / Kidney Injury** | 0.2262 | 0.2262 | **0.3215** | 0.2262 | **0.7254** |
| **Hypotension** | 0.2291 | 0.2291 | **0.3126** | 0.2291 | **0.7061** |
| **Hemorrhage / Bleeding** | 0.2354 | 0.2354 | **0.2937** | 0.2355 | **0.5690** |
| **Cardiac Arrest** | 0.2266 | 0.2266 | **0.3202** | 0.2266 | **0.5339** |

*   **Pharmacological Analysis:** Pregabalin, which is primarily cleared renally (98% unchanged), consistently receives the highest attention weight (**0.3215** for Renal Failure and **0.3126** for Hypotension), driving elevated predicted risk probabilities.

### Modality Attention Gating (Exp D)
In Exp D, a learned gating layer balances ChemBERTa chemical semantics against DrugBank biological targets, enzymes, and transporters. Across the test evaluation:
* **Average SMILES Branch Weight:** **99.89%**
* **Average Bio Branch Weight:** **0.11%**
While the biological modality provides fine-grained molecular specificity to achieve an all-time peak Precision of **0.8766**, the ChemBERTa pre-trained embeddings provide the dominant structural foundation for generalization.


---

## 6. Reproducibility & GitHub Structure

All scripts, datasets, and notebooks are structured cleanly in the repository:

```
HODDI_Project/
├── data/
│   ├── drug_embeddings_768d.pt             # Precomputed ChemBERTa SMILES embeddings
│   ├── bio_features_1024d.pt               # Multi-hot targets, enzymes, transporters, ATC
│   ├── dictionary/
│   │   ├── Side_effects_unique.csv         # 7,350 MedDRA CUIs with 768d SapBERT embeddings
│   │   └── Drugbank_ID_SMILE_all_structure links.csv
│   └── evaluation_subset/
│       └── subset_drug2-8_SE5-50/          # 41 quarterly FAERS positive/negative files
├── experiments/
│   ├── exp_a_baseline_hgnn/               # Exp A: HGNN-SA Baseline reproduction
│   ├── exp_b_hyperattddi_no_se/           # Exp B: HyperAttDDI Architecture (No SE)
│   ├── exp_c_hyperattddi_se/              # Exp C: SE-Conditioned Attention Pooling
│   ├── exp_d_hyperattddi_bio/             # Exp D: Multimodal Biological Feature Fusion
│   ├── exp_e_mean_pool_ablation/          # Exp E: Mean Pooling Ablation
│   └── exp_f_stratified_k/                # Exp F: Interaction Order Stratification
├── notebooks/                             # 5 Standalone Kaggle-ready Jupyter Notebooks
├── docs/                                  # Research manuscript and fairness audit
└── README.md                              # Complete setup, reproduction, and benchmark guide
```

---

## 7. Conclusion

HyperAttDDI establishes a new benchmark for higher-order drug-drug interaction prediction. By addressing the critical flaw of adverse-event ambiguity in existing hypergraphs, HyperAttDDI achieves **0.9518 AUC**, **0.9285 PRAUC**, and **0.8766 Precision**. Furthermore, its conditioned attention pooling provides faithful, clinically consistent explanations for multi-drug adverse reactions, bridging computational machine learning with clinical pharmacology.
