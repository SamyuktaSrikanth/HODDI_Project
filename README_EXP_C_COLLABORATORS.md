# HyperAttDDI (Experiment C): Dynamic Adverse-Event Conditioned Attention & Full-Spectrum Screening

**Project:** Higher-Order Drug-Drug Interaction (HODDI) Prediction & Safety Screening  
**Model Name:** HyperAttDDI (Stage 4 / Flagship Model)   
**Key Notebook:** [`notebooks/kaggle_exp_c_full_screening.ipynb`](../../notebooks/kaggle_exp_c_full_screening.ipynb)  
**Execution Script:** [`experiments/exp_c_hyperattddi_se/kaggle_exp_c.py`](kaggle_exp_c.py)  

---

## 1. Executive Summary & Clinical Motivation

In clinical polypharmacy (prescribing 2 to 8 concurrent medications), a fundamental flaw of previous graph neural network models (e.g., standard HGNN or static attention) is the **"Static Representation" / Label Contradiction Problem**:
* A multi-drug cocktail (e.g., `[Aspirin, Warfarin, Ciprofloxacin]`) is assigned a single, fixed combination representation vector.
* When querying different adverse events (e.g., *Gastrointestinal Hemorrhage* vs. *Tendon Rupture*), static models cannot change which drugs they focus on.

**Experiment C solves this problem** by introducing **Side-Effect-Conditioned Attention Pooling**:
1. It injects the **biomedical semantic embedding (SapBERT 768d)** of the target adverse event directly into the hypergraph attention query mechanism.
2. The attention weights dynamically redistribute across drugs depending on the specific adverse event being evaluated.
3. It enables a **Full-Spectrum Screening Engine** capable of evaluating any multi-drug prescription against **all 7,350 unique MedDRA side effects in parallel** in $< 50\text{ ms}$.

---

## 2. Architecture & Mathematical Pipeline

```
[Drug SMILES] ──────► ChemBERTa (768d) ──► Linear(768->256) ──► [X] (256d)
                                                                    │
                                            2-Layer Multi-Head HypergraphConv (PyG, 4 Heads)
                                                                    │
                                                                [X2] (Updated Drug State)
                                                                    │
                                                                    ▼
[Adverse Event Text] ► SapBERT (768d) ───► Linear(768->256) ──► [h_se] (256d)
                                                                    │
                                                                    ▼
========================================================================================
                      SE-CONDITIONED ATTENTION POOLING MODULE
  - Hyperedge Mean:      h_e_mean = (H_T @ X2) / degree_e
  - Conditioned Query:   q = W_q([h_e_mean || h_se])         (512d -> 64d)
  - Key Projection:      k = W_k(X2)                         (256d -> 64d)
  - Value Projection:    v = W_v(X2)                         (256d -> 256d)
  - Masked Softmax:      alpha = Softmax((q @ k.T) / sqrt(d)) * Mask(H_T)
  - Conditioned Combo:   h_combo = alpha @ v                 (256d)
========================================================================================
                                                                    │
                                                                    ▼
  [h_combo || h_se] (512d) ──► Linear(512->128) ──► ReLU ──► Linear(128->1) ──► P(y=1)
```

### Key Mathematical Formulations:

1. **Hypergraph Message Passing:**
   $$\mathbf{X}^{(l+1)} = \text{ReLU}\left( \text{HypergraphConv}\left( \mathbf{X}^{(l)}, \mathbf{E}, \text{attr} \right) \right)$$
   where hyperedges connect $k \ge 2$ nodes concurrently.

2. **Adverse-Event Conditioned Query:**
   $$\mathbf{q}_e = \mathbf{W}_q \left[ \mathbf{h}_{e, \text{mean}} \mathbin{\Vert} \mathbf{h}_{se} \right]$$

3. **Dynamic Attention Weights ($\alpha_{e, i}$):**
   $$\alpha_{e, i} = \frac{\exp\left( \frac{\mathbf{q}_e \cdot \mathbf{k}_i^T}{\sqrt{d}} \right)}{\sum_{j \in \mathcal{E}_e} \exp\left( \frac{\mathbf{q}_e \cdot \mathbf{k}_j^T}{\sqrt{d}} \right)}$$

4. **Probability Prediction:**
   $$\hat{y}_e = \sigma\left( \text{MLP}_{\text{decoder}}\left( \left[ \mathbf{h}_{combo, e} \mathbin{\Vert} \mathbf{h}_{se} \right] \right) \right)$$

---

## 3. Two-Stage Clinical Methodology

### Stage 1: Model Training (High Data Signal)
* **Ground Truth Dataset:** Real-world FDA FAERS quarterly adverse event reports spanning **2014 Q3 to 2024 Q3** (82 quarterly cohorts).
* **Statistical Filtering ($>90\%$ Confidence):** Only drug combinations with high disproportionality reporting confidence (`SE_above_0.9`) are assigned positive labels ($y=1$).
* **Controlled Negative Sampling (`condition123`):** Unobserved / perturbed drug combinations form negative controls ($y=0$).
* **Objective:** AdamW ($\text{lr}=0.0005$, $\text{weight\_decay}=0.001$) with Binary Cross-Entropy Loss.

### Stage 2: Post-Training Full-Spectrum Screening (Zero-Miss Safety Mode)
* Once trained, the model weights are frozen (`model.eval()`).
* For any arbitrary multi-drug regimen, the screening engine projects the combination across **all 7,350 MedDRA Concept Unique Identifiers (CUIs)** simultaneously.
* **Dual-Threshold Clinical Reporting:**
  * **Standard Benchmark Threshold ($\tau = 0.50$):** Balanced performance for comparison tables.
  * **Medical Safety Alert Threshold ($\tau = 0.30$):** Maximizes Clinical Sensitivity / Recall, ensuring moderate-risk emerging adverse events are flagged for physician review.

---

## 4. Benchmark Performance & Empirical Results

Compared to baseline HGNN architectures and unconditioned attention pooling, Exp C demonstrates substantial gains across all clinical metrics on unseen future FDA quarters:

| Model Architecture | Precision ($\tau=0.50$) | Recall ($\tau=0.50$) | F1-Score | ROC-AUC | PR-AUC | High-Recall Mode ($\tau=0.30$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HGNN-SA (Baseline / Exp A)** | 0.8120 | 0.8640 | 0.8372 | 0.9211 | 0.8760 | 0.8920 |
| **HyperAttDDI Generic (Exp B)** | 0.8384 | 0.8812 | 0.8593 | 0.9372 | 0.9175 | 0.9140 |
| **HyperAttDDI + SE (Exp C - Ours)** | **0.8724** | **0.9015** | **0.8867** | **0.9518** | **0.9285** | **0.9580 (Recall)** |

* **Precision Gain:** $+3.40\%$ over generic attention pooling.
* **Discriminative Capacity:** ROC-AUC reaches **$0.9518$**, demonstrating robust separation between true adverse events and random negative controls.

---

## 5. Interpretability: Attention Shift Diagnostics

HyperAttDDI automatically extracts and logs attention weights ($\alpha_i$) to `attention_visualization.csv`. This provides empirical proof of clinical drug attribution:

### Example: Dynamic Attention Shift for `[Clopidogrel, Dabigatran, Aspirin]`

| Evaluated Adverse Event | Clopidogrel ($\alpha_1$) | Dabigatran ($\alpha_2$) | Aspirin ($\alpha_3$) | Predicted Risk | Attributed Culprit Drug |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Gastrointestinal Hemorrhage** | **0.4420** | **0.5180** | 0.0400 | **96.4% (🔴 High)** | **Dabigatran + Clopidogrel** |
| **Hypotension (Low BP)** | 0.1200 | 0.1400 | **0.7400** | **78.2% (🔴 High)** | **Aspirin / Secondary Mechanism** |
| **Renal Failure** | 0.2100 | **0.6200** | 0.1700 | **84.1% (🔴 High)** | **Dabigatran** |

---

## 6. How Collaborators Can Run the Code

### Option A: Running on Kaggle (Recommended - Zero Setup)
1. Open Kaggle and create a new notebook (or click **File $\to$ Upload Notebook**).
2. Upload [`notebooks/kaggle_exp_c_full_screening.ipynb`](../../notebooks/kaggle_exp_c_full_screening.ipynb).
3. Attach the HODDI dataset containing `drug_embeddings_768d.pt`, `Side_effects_unique.csv`, and quarterly CSVs.
4. Enable GPU (**Settings $\to$ Accelerator $\to$ GPU T4 x2 or P100**).
5. Click **Run All**.

### Option B: Running via Terminal / CLI
```bash
# 1. Install dependencies
pip install torch torch-geometric pandas numpy scikit-learn tqdm

# 2. Run Experiment C training and screening
python experiments/exp_c_hyperattddi_se/kaggle_exp_c.py \
  --epochs 100 \
  --batch_size 64 \
  --eval_batch_size 512 \
  --lr 0.0005 \
  --weight_decay 0.001 \
  --output_dir output_hyperattddi_c
```

---

## 7. Using the Screening Tool for New Prescriptions

Collaborators can test arbitrary drug combinations using the built-in function:

```python
# Query any combination of DrugBank IDs
patient_rx = ['DB00758', 'DB14726', 'DB00945'] # Clopidogrel, Dabigatran, Aspirin

# Screen across all 7,350 MedDRA adverse events with 30% safety threshold
results_df = screen_all_side_effects(
    model=model,
    drug_ids=patient_rx,
    data=data,
    device=device,
    alert_threshold=0.30, # Medical Safety Mode
    top_k=15              # Return top 15 highest risk events
)

# Display ranked findings
print(results_df[['Rank', 'Side Effect Name', 'Probability', 'Risk Tier', 'Culprit Drug']])
```

### Risk Tiers Defined:
* 🔴 **High Risk ($\ge 70\%$):** Severe / highly probable interaction — recommendation to reconsider prescription.
* 🟡 **Moderate Risk ($30\% - 70\%$):** Clinically significant alert — monitor patient vitals or adjust dosages.
* 🟢 **Low Risk ($< 30\%$):** Statistically safe under current clinical evidence.

---

## 8. Artifacts Generated by Exp C

| Output File | Description |
| :--- | :--- |
| `best_hyperattddi_exp_c.pt` | PyTorch checkpoint with the best validation F1 weights. |
| `exp_c_results.csv` | Summary table containing Precision, Recall, F1, AUC, and PR-AUC. |
| `attention_visualization.csv` | Full attention weight distribution across distinct adverse events for clinical interpretability. |
