# Experiment C: HyperAttDDI with Side-Effect-Conditioned Attention Pooling

## Description
The core scientific contribution of HyperAttDDI. Eliminates the label contradiction problem in HGNN-SA by feeding 768d SapBERT adverse event embeddings as an active conditioning input to the hyperedge attention query. The model learns dynamic attention distributions that change depending on which adverse event is being predicted.

## Architecture Details
- **Module 1:** `Linear(768 -> 256) -> ReLU -> Dropout(0.1)` on ChemBERTa features.
- **Module 2:** 2-layer PyG `HypergraphConv(256, 64, heads=4)`.
- **Module 4 (SE Encoder):** `Linear(768 -> 256) -> ReLU -> Dropout(0.1)` on 768d SapBERT embeddings.
- **Module 3 (SE-Conditioned Attention):**
  $$q = W_q([h_{e, \text{mean}} \mathbin{\Vert} h_{\text{se}}]) \quad (512 \to 64)$$
  $$\alpha = \text{softmax}(q k^T / \sqrt{d}) \odot \text{mask}$$
  $$h_{\text{combo}} = \alpha W_v(X_2) \in \mathbb{R}^{256}$$
- **Module 5 (Decoder):** $[h_{\text{combo}} \mathbin{\Vert} h_{\text{se}}] \in \mathbb{R}^{512} \to \text{Linear}(512 \to 128) \to \text{ReLU} \to \text{Dropout} \to \text{Linear}(128 \to 1)$.

## Results
- **AUC:** **0.9518** (+3.07% over HGNN-SA baseline, +1.46% over Exp B)
- **PRAUC:** **0.9285** (+5.25% over HGNN-SA baseline, +1.10% over Exp B)
- **F1 Score:** **0.9101** (+1.75% over HGNN-SA baseline, +2.36% over Exp B)
- **Precision:** **0.8724** (+1.53% over baseline, +3.40% over Exp B)
- **Recall:** **0.9511** (+2.00% over baseline)

## Included Artifacts
- `attention_visualization.csv`: Attention weight distribution across cardiovascular and neuro-psychiatric drug combinations under diverse adverse events.

## How to Run
```bash
python run_exp_c.py --epochs 100 --batch_size 64
```
Or open `kaggle_exp_c.ipynb` on Kaggle.
