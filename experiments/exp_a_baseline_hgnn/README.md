# Experiment A: HGNN-SA Baseline Reproduction

## Description
Reproduction of the state-of-the-art Hypergraph Neural Network with Self-Attention (HGNN-SA) baseline from the HODDI paper (Wang et al., 2025).

## Architecture Details
- **Dual Drug Features:** Fixed positive training incidence structural fingerprint (`train_inc_pos`) concatenated with 768d ChemBERTa SMILES embedding.
- **Hypergraph Convolution:** 1-layer PyG `HypergraphConv` (in=256, out=64, heads=3).
- **Hyperedge Attributes:** Linear projection of the transposed incidence matrix.
- **Aggregation:** Direct matrix multiplication ($H^T X$), unweighted.
- **Output:** Linear projection to 2 classes with `CrossEntropyLoss`.

## Results
- **AUC:** 0.9211
- **PRAUC:** 0.8760
- **F1 Score:** 0.8926
- **Precision:** 0.8571
- **Recall:** 0.9311

## How to Run
```bash
python run_hgnn_sa.py --epochs 100 --batch_size 64
```
On Kaggle, run the script with a GPU accelerator.
