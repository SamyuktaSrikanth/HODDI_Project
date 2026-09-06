# Experiment B: HyperAttDDI Architecture (No Side Effect Conditioning)

## Description
Tests the isolated architectural contribution of HyperAttDDI under the **exact same input conditions as HGNN-SA** (no side-effect features). Proves that our 2-layer multi-head hypergraph and learned attention pooling outperform HGNN-SA purely through architectural superiority.

## Architecture Details
- **Module 1 (Drug Encoder):** `Linear(768 -> 256) -> ReLU -> Dropout(0.1)` on ChemBERTa SMILES features.
- **Module 2 (Hypergraph Encoder):** 2-layer PyG `HypergraphConv` (256 -> 64 * 4 heads = 256 -> 64 * 4 heads = 256).
- **Module 3 (Attention Pooling):** Learned query $q = W_q(h_{e, \text{mean}})$, keys $k = W_k(X_2)$, values $v = W_v(X_2)$. Masked softmax attention pooling over member drugs.
- **Module 5 (Decoder):** `Linear(256 -> 128) -> ReLU -> Dropout(0.1) -> Linear(128 -> 1)` with `BCEWithLogitsLoss`.

## Results
- **AUC:** **0.9372** (+1.61% over HGNN-SA)
- **PRAUC:** **0.9175** (+4.15% over HGNN-SA)
- **F1 Score:** 0.8865
- **Precision:** 0.8384
- **Recall:** **0.9405**

## How to Run
```bash
python run_exp_b.py --epochs 100 --batch_size 64
```
Or open `kaggle_exp_b.ipynb` on Kaggle.
