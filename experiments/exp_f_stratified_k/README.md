# Experiment F: Performance Stratified by Interaction Order ($k = 2, 3, 4, 5, 6+$)

## Evaluation Objective
Evaluates model performance stratified by the number of drugs in the combination ($k \in \{2, 3, 4, 5, 6+\}$) on the test set. This analyzes whether higher-order combinations ($k \ge 3$) experience performance degradation compared to pairwise interactions ($k=2$).

## Target Results Table (To be populated upon evaluation)

| # Drugs ($k$) | Test Sample Count | MLP (AUC) | GAT (AUC) | HGNN-SA (AUC) | HyperAttDDI (AUC) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2 (Pairs)** | 1,404 | -- | -- | -- | -- |
| **3 (Triplets)** | 1,328 | -- | -- | -- | -- |
| **4 (Quartets)** | 1,046 | -- | -- | -- | -- |
| **5 (Quintets)** | 874 | -- | -- | -- | -- |
| **6+ (Sextets+)** | 1,934 | -- | -- | -- | -- |

## How to Run
```bash
python stratified_eval_k.py --checkpoint_c ../exp_c_hyperattddi_se/best_hyperattddi_exp_c.pt
```
Or run `kaggle_stratified_eval.ipynb` on Kaggle where your trained checkpoint is loaded.

