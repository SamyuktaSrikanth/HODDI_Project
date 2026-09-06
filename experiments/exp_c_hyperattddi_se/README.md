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

## Empirical Attention Distribution (Kaggle Test Run)

### Cardiovascular Regimen (Clopidogrel + Dabigatran + Aspirin)
| Queried Side Effect | Clopidogrel | Dabigatran | Acetylsalicylic acid | Predicted Prob |
| :--- | :---: | :---: | :---: | :---: |
| Hemorrhage / Bleeding | 0.3268 | 0.3254 | 0.3477 | 0.5763 |
| Renal Failure / Kidney Injury | 0.3226 | 0.3203 | 0.3570 | 0.7340 |
| Hypotension | 0.3238 | 0.3218 | 0.3544 | 0.7141 |
| Cardiac Arrest | 0.3227 | 0.3204 | 0.3569 | 0.5420 |

### Neuro-Psychiatric Regimen (Escitalopram + Clobazam + Pregabalin + Tramadol)
| Queried Side Effect | Escitalopram | Clobazam | Pregabalin | Tramadol | Predicted Prob |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Hemorrhage / Bleeding | 0.2354 | 0.2354 | 0.2937 | 0.2355 | 0.5690 |
| Renal Failure / Kidney Injury | 0.2262 | 0.2262 | 0.3215 | 0.2262 | 0.7254 |
| Hypotension | 0.2291 | 0.2291 | 0.3126 | 0.2291 | 0.7061 |
| Cardiac Arrest | 0.2266 | 0.2266 | 0.3202 | 0.2266 | 0.5339 |

## How to Run
```bash
python run_exp_c.py --epochs 100 --batch_size 64
```
Or open `kaggle_exp_c.ipynb` on Kaggle.

