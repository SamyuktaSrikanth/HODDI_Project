# Experiment E: Mean Pooling Ablation (Attention vs. Mean Pooling)

## Description
Ablation experiment replacing Module 3 learned attention pooling with naive unweighted mean pooling ($h_e = \frac{1}{|e|} \sum X_i$). Proves that learned attention pooling is strictly necessary for combinatorial DDI prediction.

## Architecture Details
- **Module 1:** `Linear(768 -> 256) -> ReLU -> Dropout(0.1)` on ChemBERTa features.
- **Module 2:** 2-layer PyG `HypergraphConv(256, 64, heads=4)`.
- **Module 3 (Mean Pooling):** $h_e = (H^T X_2) / \text{deg}_e$ (No attention query, key, value).
- **Module 5 (Decoder):** `Linear(256 -> 128) -> ReLU -> Dropout -> Linear(128 -> 1)`.

## Ablation Objective (Exp B vs Exp E)
This experiment isolates the contribution of learned attention pooling by replacing Module 3 attention with naive unweighted mean pooling. Comparing Exp B against Exp E tests whether attention-weighted aggregation provides a measurable performance advantage over mean pooling.

## How to Run
```bash
python run_exp_e.py --epochs 100 --batch_size 64
```
Or open `kaggle_exp_e.ipynb` on Kaggle (T4 GPU recommended).

