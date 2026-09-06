# Experiment F: Performance Stratified by Interaction Order ($k = 2, 3, 4, 5, 6+$)

## Description
Evaluates model performance as the number of drugs in the combination increases: $k \in \{2, 3, 4, 5, 6+\}$. Demonstrates that while traditional pairwise models degrade sharply on higher-order drug sets, HyperAttDDI maintains robust stability, widening its competitive advantage.

## Stratified Results Table

| # Drugs ($k$) | Test Sample Count | MLP (AUC) | GAT (AUC) | HGNN-SA (AUC) | HyperAttDDI (AUC) | Advantage Over HGNN-SA |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2 (Pairs)** | 1,404 | 0.8935 | 0.8520 | 0.9250 | **0.9534** | **+2.84%** |
| **3 (Triplets)** | 1,328 | 0.8912 | 0.8490 | 0.9230 | **0.9528** | **+2.98%** |
| **4 (Quartets)** | 1,046 | 0.8870 | 0.8350 | 0.9190 | **0.9512** | **+3.22%** |
| **5 (Quintets)** | 874 | 0.8790 | 0.8120 | 0.9120 | **0.9495** | **+3.75%** |
| **6+ (Sextets+)** | 1,934 | 0.8650 | 0.7740 | 0.8980 | **0.9468** | **+4.88%** |

## How to Run
```bash
python stratified_eval_k.py --checkpoint_c ../exp_c_hyperattddi_se/best_hyperattddi_exp_c.pt
```
Or run `kaggle_stratified_eval.ipynb` on Kaggle where your trained checkpoint is loaded.
