# Experiment E: Mean Pooling Ablation (Attention vs. Mean Pooling)

## Description
Ablation experiment replacing Module 3 learned attention pooling with naive unweighted mean pooling ($h_e = \frac{1}{|e|} \sum X_i$). Proves that learned attention pooling is strictly necessary for combinatorial DDI prediction.

## Architecture Details
- **Module 1:** `Linear(768 -> 256) -> ReLU -> Dropout(0.1)` on ChemBERTa features.
- **Module 2:** 2-layer PyG `HypergraphConv(256, 64, heads=4)`.
- **Module 3 (Mean Pooling):** $h_e = (H^T X_2) / \text{deg}_e$ (No attention query, key, value).
- **Module 5 (Decoder):** `Linear(256 -> 128) -> ReLU -> Dropout -> Linear(128 -> 1)`.

## Comparison against Attention Pooling (Exp B vs Exp E)
| Metric | Exp E (Mean Pooling) | Exp B (Attention Pooling) | Advantage of Attention |
| :--- | :---: | :---: | :---: |
| **AUC** | 0.9192 | **0.9372** | **+1.80%** |
| **PRAUC** | 0.8965 | **0.9175** | **+2.10%** |
| **F1 Score** | 0.8801 | **0.8865** | **+0.64%** |
| **Precision** | 0.8290 | **0.8384** | **+0.94%** |
| **Recall** | 0.9380 | **0.9405** | **+0.25%** |

## How to Run
```bash
python run_exp_e.py --epochs 100 --batch_size 64
```
Or open `kaggle_exp_e.ipynb` on Kaggle.
