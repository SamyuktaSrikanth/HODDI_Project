# Experiment D: HyperAttDDI with Multimodal Biological Attention Fusion

## Description
Extends Module 1 with biological feature encoders (drug targets, metabolic enzymes, transporters, carriers, and ATC classifications from DrugBank) combined via learned attention gating.

## Architecture Details
- **Module 1 (Multimodal Drug Encoder):**
  - SMILES Encoder: ChemBERTa (768) $\to$ `Linear(768 -> 256) -> ReLU -> Dropout`
  - Bio Encoder: Bio profiles (1024) $\to$ `Linear(1024 -> 128) -> ReLU -> Dropout -> Linear(128 -> 256)`
  - Attention Fusion: Learned gating weights $[\alpha_{\text{smiles}}, \alpha_{\text{bio}}]$ per drug $\to$ 256d drug embedding.
- **Module 2:** 2-layer PyG `HypergraphConv(256, 64, heads=4)`.
- **Module 4:** SapBERT `Linear(768 -> 256) -> ReLU -> Dropout`.
- **Module 3:** SE-conditioned attention pooling ($q = W_q([h_{e, \text{mean}} \mathbin{\Vert} h_{\text{se}}])$).
- **Module 5:** Interaction decoder on $[h_{\text{combo}} \mathbin{\Vert} h_{\text{se}}]$.

## Results
- **Precision:** **0.8766** (All-time project high, +1.95% over baseline)
- **F1 Score:** **0.9104** (All-time project high, +1.78% over baseline)
- **AUC:** 0.9505
- **PRAUC:** 0.9252
- **Recall:** 0.9469

## Included Artifacts
- `attention_visualization_exp_d.csv`: Attention and modality gating weights per drug.

## How to Run
```bash
python run_exp_d.py --epochs 100 --batch_size 64
```
Or open `kaggle_exp_d.ipynb` on Kaggle.
