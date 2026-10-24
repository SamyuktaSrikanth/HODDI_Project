# Research Paper & Project Guidelines: HyperAttDDI

This document defines strict guidelines for any AI assistant when drafting, editing, or citing research papers, manuscripts, slides, and reports for this project.

---

## 1. Authoritative Single Source of Truth for Experimental Results

* **Primary Verified Files:**
  * [`results/FINAL_CONSOLIDATED_RESULTS.md`](results/FINAL_CONSOLIDATED_RESULTS.md)
  * [`results/results.pdf`](results/results.pdf)
  * [`README.md`](README.md)
* **Master 7-Seed Evaluation Benchmark:**
  Always report and cite the multi-run 7-seed benchmark ($n=7$ independent random seeds: `42, 3407, 54321, 123456, 7, 777, 31415`) on the standardized 41-quarter FDA FAERS temporal split:
  * **Exp A: HGNN-SA (Baseline):** ROC-AUC $0.9413 \pm 0.0021$, PR-AUC $0.8987 \pm 0.0045$, F1 $0.9242 \pm 0.0025$, Precision $0.9001 \pm 0.0024$, Recall $0.9495 \pm 0.0048$
  * **Exp B: HyperAttDDI-Base:** ROC-AUC $0.9689 \pm 0.0037$, PR-AUC $0.9665 \pm 0.0061$, F1 $0.9310 \pm 0.0045$, Precision $0.9239 \pm 0.0108$, Recall $0.9384 \pm 0.0114$
  * **Exp C: HyperAttDDI-SE (Core):** ROC-AUC **$0.9779 \pm 0.0017$**, PR-AUC **$0.9729 \pm 0.0026$**, F1 **$0.9465 \pm 0.0010$**, Precision $0.9377 \pm 0.0043$, Recall **$0.9555 \pm 0.0046$**
  * **Exp D: HyperAttDDI-Bio (Multimodal):** ROC-AUC $0.9770 \pm 0.0014$, PR-AUC $0.9727 \pm 0.0027$, F1 $0.9437 \pm 0.0034$, Precision **$0.9389 \pm 0.0043$**, Recall $0.9486 \pm 0.0060$

---

## 2. STRICTLY IGNORE Legacy / Single-Seed Drafts

When writing paper text, methodology, comparative tables, or statistical analyses:
* **DO NOT USE OR CITE:**
  * `archive/01_kaggle_exp_b_no_se.ipynb` (early single-run prototype)
  * `archive/02_kaggle_exp_c_with_se.ipynb` (early single-run prototype)
  * `archive/07_kaggle_exp_a_hgnn_sa_baseline.ipynb` (early single-run prototype)
  * Any scripts in `archive/`
  * Outdated single-split canonical metrics from early September.
* These files represent preliminary development drafts and are completely superseded by the 7-seed statistical benchmarking suite.

---

## 3. Active Production Experiments to Reference

When describing code reproduction, pipeline architecture, or specific ablations, reference ONLY:
1. **Exp A (7-Seed Baseline Evaluation):** [`experiments/exp_a_baseline_hgnn/kaggle_master_evaluation.ipynb`](experiments/exp_a_baseline_hgnn/kaggle_master_evaluation.ipynb)
2. **Exp B (7-Seed Base Model Evaluation):** [`experiments/exp_b_hyperattddi_base/kaggle_master_evaluation.ipynb`](experiments/exp_b_hyperattddi_base/kaggle_master_evaluation.ipynb)
3. **Exp C (Core Model + 7,350 MedDRA Screening Engine):** [`experiments/exp_c_hyperattddi_se/kaggle_exp_c_full_screening.ipynb`](experiments/exp_c_hyperattddi_se/kaggle_exp_c_full_screening.ipynb)
4. **Exp D (Multimodal Biological Fusion - 7 Seeds):** [`experiments/exp_d_hyperattddi_bio/exp_d.ipynb`](experiments/exp_d_hyperattddi_bio/exp_d.ipynb)
5. **Exp D Modality Gating Weights ($\alpha_{smiles}$ vs. $\alpha_{bio}$):** [`experiments/exp_d_hyperattddi_bio/03_kaggle_exp_d_multimodal_bio.ipynb`](experiments/exp_d_hyperattddi_bio/03_kaggle_exp_d_multimodal_bio.ipynb)
6. **Exp E (Attention Pooling vs. Mean Pooling):** [`experiments/exp_e_mean_pool_ablation/04_kaggle_exp_e_mean_pool_ablation.ipynb`](experiments/exp_e_mean_pool_ablation/04_kaggle_exp_e_mean_pool_ablation.ipynb)
7. **Exp F (Stratified Evaluation $k \in [2, 8]$):** [`experiments/exp_f_stratified_k/05_kaggle_exp_f_stratified_eval.ipynb`](experiments/exp_f_stratified_k/05_kaggle_exp_f_stratified_eval.ipynb)
8. **Exp G (Topological Message-Passing Hops $L=1..4$):** [`experiments/exp_g_message_passing_hops/kaggle_exp_g.ipynb`](experiments/exp_g_message_passing_hops/kaggle_exp_g.ipynb)
9. **Interactive CLI Inference:** [`demo/demo_inference.py`](demo/demo_inference.py)
