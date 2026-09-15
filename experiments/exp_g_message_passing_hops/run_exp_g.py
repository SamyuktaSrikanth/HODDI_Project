import os
import sys
import ast
import math
import time
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as hnn
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class HyperAttDDI_HopAblation(nn.Module):
    """
    HyperAttDDI without Side-Effect Features (Exp B architecture) with parameterized
    Hypergraph Message Passing Depth (L = 1, 2, 3, 4 hops).
    
    Architecture (100% aligned with Exp B):
    - Module 1 (Drug Encoder): ChemBERTa (768) -> Linear(768->256) -> ReLU -> Dropout
    - Module 2 (Hypergraph Encoder): L-layer PyG HypergraphConv(256, 64, heads=4, use_attention=True)
    - Module 3 (Attention Pooling, no SE):
        q = W_q(h_e_mean)
        k_i = W_k(h_i)
        alpha = softmax(q @ k.T / sqrt(d)) (masked by hyperedge membership)
        h_e = alpha @ W_v(h_i)
    - Module 5 (Decoder): Linear(256->128) -> ReLU -> Dropout -> Linear(128->1)
    """
    def __init__(self, in_dim=768, emb_dim=256, conv_dim=64, heads=4, num_layers=2, d=64, p=0.1):
        super().__init__()
        self.d = d
        self.p = p
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        
        # Module 1 (Drug Encoder)
        self.drug_encoder = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(p=p)
        )
        
        # Module 2 (Hypergraph Encoder): L layers
        self.convs = nn.ModuleList()
        if num_layers > 0:
            self.convs.append(hnn.HypergraphConv(emb_dim, conv_dim, heads=heads, use_attention=True, dropout=p))
            for _ in range(1, num_layers):
                self.convs.append(hnn.HypergraphConv(conv_dim * heads, conv_dim, heads=heads, use_attention=True, dropout=p))
                
        # Module 3 (Attention Pooling, no SE)
        self.W_q = nn.Linear(emb_dim, d)
        self.W_k = nn.Linear(emb_dim, d)
        self.W_v = nn.Linear(emb_dim, emb_dim)
        
        # Module 5 (Decoder)
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(128, 1)
        )
        
    def forward(self, drug_features, inc_matrix):
        """
        drug_features: (num_drugs, 768)
        inc_matrix: (num_drugs, batch_hyperedges)
        """
        N, E = inc_matrix.shape
        H_T = inc_matrix.T  # (E, N)
        degree_e = H_T.sum(dim=1, keepdim=True).clamp(min=1)  # (E, 1)
        
        # 1. Drug Encoder
        X = self.drug_encoder(drug_features)  # (N, 256)
        
        # 2. Hypergraph Message Passing across L hops
        if self.num_layers > 0:
            row, col = torch.where(H_T)
            edges = torch.cat([col.view(1, -1), row.view(1, -1)], dim=0).to(drug_features.device)
            X_curr = X
            for conv in self.convs:
                attr = (H_T @ X_curr) / degree_e
                X_next = F.relu(conv(X_curr, edges, hyperedge_attr=attr))
                X_curr = F.dropout(X_next, p=self.p, training=self.training)
            X_final = X_curr
        else:
            X_final = X
            
        # 3. Attention Pooling
        h_e_mean = (H_T @ X_final) / degree_e  # (E, 256)
        q = self.W_q(h_e_mean)                 # (E, d)
        k = self.W_k(X_final)                  # (N, d)
        v = self.W_v(X_final)                  # (N, 256)
        
        scores = torch.matmul(q, k.T) / math.sqrt(self.d)  # (E, N)
        scores = scores.masked_fill(H_T == 0, -1e9)
        alpha = F.softmax(scores, dim=1)  # (E, N)
        h_e = torch.matmul(alpha, v)      # (E, 256)
        
        # 5. Decoder
        logits = self.decoder(h_e).squeeze(-1)  # (E,)
        return logits

def find_dataset_path():
    if os.path.exists('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            if 'drug_embeddings_768d.pt' in files:
                return root
    candidates = ['data', '../data', '../../data', 'dataset', '../dataset', '../../dataset']
    for c in candidates:
        if os.path.exists(c) and os.path.exists(os.path.join(c, 'drug_embeddings_768d.pt')):
            return c
    for c in candidates:
        if os.path.exists(c):
            return c
    return '.'

def load_data(base_path, device):
    if os.path.exists(os.path.join(base_path, 'evaluation_subset')):
        dataset_base_dir = os.path.join(base_path, 'evaluation_subset', 'subset_drug2-8_SE5-50', '')
    elif os.path.exists(os.path.join(base_path, 'subset_drug2-8_SE5-50')):
        dataset_base_dir = os.path.join(base_path, 'subset_drug2-8_SE5-50', '')
    else:
        dataset_base_dir = os.path.join(base_path, 'dataset', 'evaluation_subset', 'subset_drug2-8_SE5-50', '')

    training_sub_ds_names = ['2015Q1', '2015Q2', '2015Q3', '2016Q4', '2017Q1', '2017Q2', '2017Q3', '2017Q4', '2018Q3', '2019Q1', '2019Q2', '2019Q3', '2019Q4', '2020Q1', '2020Q2', '2020Q3', '2020Q4', '2021Q1', '2021Q3', '2021Q4', '2022Q1', '2022Q2', '2022Q3', '2022Q4', '2023Q1', '2023Q2', '2023Q3', '2023Q4', '2024Q2']
    validating_sub_ds_names = ['2014Q3', '2015Q4', '2016Q1', '2016Q3', '2021Q2', '2024Q1']
    testing_sub_ds_names = ['2014Q4', '2016Q2', '2018Q1', '2018Q2', '2018Q4', '2024Q3']
    
    emb_candidates = [
        os.path.join(base_path, "drug_embeddings_768d.pt"),
        os.path.join(base_path, "dataset", "drug_embeddings_768d.pt"),
        "dataset/drug_embeddings_768d.pt"
    ]
    emb_path = None
    for cand in emb_candidates:
        if os.path.exists(cand):
            emb_path = cand
            break
    if emb_path is None:
        raise FileNotFoundError(f"Missing embeddings file: checked {emb_candidates}")
    
    print(f"Loading embeddings from: {emb_path}")
    drug_embeddings_dict = torch.load(emb_path, map_location='cpu', weights_only=True)
    
    def merge_sub_datasets(sub_datasets):
        pos_merged = pd.DataFrame()
        neg_merged = pd.DataFrame()
        for sub_ds in sub_datasets:
            pos_file = os.path.join(dataset_base_dir, f'{sub_ds}_positive_samples_condition123_SE_above_0.9.csv')
            neg_file = os.path.join(dataset_base_dir, f'{sub_ds}_negative_samples_condition123_SE_above_0.9.csv')
            if os.path.exists(pos_file):
                pos_merged = pd.concat([pos_merged, pd.read_csv(pos_file)], axis=0)
            if os.path.exists(neg_file):
                neg_merged = pd.concat([neg_merged, pd.read_csv(neg_file)], axis=0)
        return pos_merged, neg_merged

    print("Loading quarterly CSV files...")
    train_data_pos, train_data_neg = merge_sub_datasets(training_sub_ds_names)
    val_data_pos, val_data_neg = merge_sub_datasets(validating_sub_ds_names)
    test_data_pos, test_data_neg = merge_sub_datasets(testing_sub_ds_names)
    
    all_drugs = set()
    for df in [train_data_pos, train_data_neg, val_data_pos, val_data_neg, test_data_pos, test_data_neg]:
        for drug_list in df['DrugBankID']:
            all_drugs.update([d for d in ast.literal_eval(drug_list) if d.lower() != 'none'])
            
    valid_drugs = sorted(list(d for d in all_drugs if d in drug_embeddings_dict))
    drug_to_index = {drug: idx for idx, drug in enumerate(valid_drugs)}
    num_drugs = len(valid_drugs)
    print(f"Total valid unique drugs: {num_drugs}")

    def build_incidence_matrix(pos_df, neg_df):
        inc_pos = np.zeros((num_drugs, len(pos_df)), dtype=np.float32)
        inc_neg = np.zeros((num_drugs, len(neg_df)), dtype=np.float32)
        
        for col_idx, drug_list in enumerate(pos_df['DrugBankID']):
            for drug_id in ast.literal_eval(drug_list):
                if drug_id in drug_to_index:
                    inc_pos[drug_to_index[drug_id], col_idx] = 1.0
                    
        for col_idx, drug_list in enumerate(neg_df['DrugBankID']):
            for drug_id in ast.literal_eval(drug_list):
                if drug_id in drug_to_index:
                    inc_neg[drug_to_index[drug_id], col_idx] = 1.0
                    
        labels_pos = np.ones(len(pos_df), dtype=np.float32)
        labels_neg = np.zeros(len(neg_df), dtype=np.float32)
        return inc_pos, inc_neg, labels_pos, labels_neg

    print("Building incidence matrices...")
    train_inc_pos, train_inc_neg, train_lab_pos, train_lab_neg = build_incidence_matrix(train_data_pos, train_data_neg)
    val_inc_pos, val_inc_neg, val_lab_pos, val_lab_neg = build_incidence_matrix(val_data_pos, val_data_neg)
    test_inc_pos, test_inc_neg, test_lab_pos, test_lab_neg = build_incidence_matrix(test_data_pos, test_data_neg)
    
    extra_feature = torch.zeros((num_drugs, 768), dtype=torch.float32)
    for drug, idx in drug_to_index.items():
        extra_feature[idx] = drug_embeddings_dict[drug]
        
    train_inc = torch.tensor(np.concatenate([train_inc_pos, train_inc_neg], axis=1), dtype=torch.float32)
    train_lab = torch.tensor(np.concatenate([train_lab_pos, train_lab_neg]), dtype=torch.float32)
    
    val_inc = torch.tensor(np.concatenate([val_inc_pos, val_inc_neg], axis=1), dtype=torch.float32)
    val_lab = np.concatenate([val_lab_pos, val_lab_neg])
    
    test_inc = torch.tensor(np.concatenate([test_inc_pos, test_inc_neg], axis=1), dtype=torch.float32)
    test_lab = np.concatenate([test_lab_pos, test_lab_neg])
    
    return {
        'train_inc': train_inc,
        'train_lab': train_lab,
        'val_inc': val_inc,
        'val_lab': val_lab,
        'test_inc': test_inc,
        'test_lab': test_lab,
        'drug_features': extra_feature.to(device),
        'num_drugs': num_drugs
    }

def evaluate_dataset(model, drug_features, inc_matrix, labels, device, batch_size=512):
    model.eval()
    all_scores = []
    num_edges = inc_matrix.shape[1]
    
    with torch.no_grad():
        for i in range(0, num_edges, batch_size):
            batch_inc = inc_matrix[:, i:i+batch_size].to(device)
            logits = model(drug_features, batch_inc)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(probs)
            
    all_scores = np.concatenate(all_scores, axis=0)
    preds = (all_scores >= 0.5).astype(int)
    
    auc = roc_auc_score(labels, all_scores)
    prauc = average_precision_score(labels, all_scores)
    f1 = f1_score(labels, preds, zero_division=0)
    pr = precision_score(labels, preds, zero_division=0)
    re = recall_score(labels, preds, zero_division=0)
    
    loss_val = F.binary_cross_entropy_with_logits(
        torch.tensor(all_scores, dtype=torch.float32), 
        torch.tensor(labels, dtype=torch.float32)
    ).item()
    
    return auc, prauc, f1, pr, re, loss_val

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    conv_params = sum(p.numel() for conv in model.convs for p in conv.parameters() if p.requires_grad) if len(model.convs) > 0 else 0
    return total, conv_params

def plot_hop_comparison(results_df, output_path='message_passing_hops_comparison.png'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Experiment G: Hypergraph Message Passing Depth & Hop Ablation (Exp B Architecture)', fontsize=15, fontweight='bold')
    
    hops = results_df['hops']
    
    # 1. AUC and PR-AUC vs Hops
    ax = axes[0, 0]
    ax.plot(hops, results_df['test_auc'], marker='o', linewidth=2.5, color='#1f77b4', label='Test ROC-AUC')
    ax.plot(hops, results_df['test_prauc'], marker='s', linewidth=2.5, color='#ff7f0e', label='Test PR-AUC')
    ax.axvline(x=2, color='green', linestyle='--', alpha=0.7, label='Optimal (L=2)')
    ax.set_title('Discriminative Performance vs Message Passing Hops', fontweight='bold')
    ax.set_xlabel('Hypergraph Conv Layers (Hops $L$)')
    ax.set_ylabel('Score')
    ax.set_xticks(hops)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    
    # 2. Parameter Scaling
    ax = axes[0, 1]
    ax.bar(hops - 0.15, results_df['total_params'] / 1e3, width=0.3, label='Total Params (k)', color='#2ca02c', alpha=0.85)
    ax.bar(hops + 0.15, results_df['conv_params'] / 1e3, width=0.3, label='Conv Params (k)', color='#d62728', alpha=0.85)
    ax.set_title('Parameter Scaling per Hop Layer', fontweight='bold')
    ax.set_xlabel('Hypergraph Conv Layers (Hops $L$)')
    ax.set_ylabel('Parameters (Thousands)')
    ax.set_xticks(hops)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    
    # 3. Precision vs Recall Trade-off
    ax = axes[1, 0]
    ax.plot(hops, results_df['test_precision'], marker='^', linewidth=2, color='#9467bd', label='Precision')
    ax.plot(hops, results_df['test_recall'], marker='v', linewidth=2, color='#8c564b', label='Recall')
    ax.plot(hops, results_df['test_f1'], marker='D', linewidth=2, color='#e377c2', label='F1-Score')
    ax.axvline(x=2, color='green', linestyle='--', alpha=0.7)
    ax.set_title('Classification Metrics vs Hops', fontweight='bold')
    ax.set_xlabel('Hypergraph Conv Layers (Hops $L$)')
    ax.set_ylabel('Score')
    ax.set_xticks(hops)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    
    # 4. Pareto Frontier (PR-AUC vs Parameter Cost)
    ax = axes[1, 1]
    scatter = ax.scatter(results_df['total_params'] / 1e3, results_df['test_prauc'], c=hops, cmap='viridis', s=160, edgecolors='black', zorder=5)
    for i, row in results_df.iterrows():
        ax.annotate(f"L={int(row['hops'])}\n({row['test_prauc']:.4f})", 
                    (row['total_params']/1e3 + 15, row['test_prauc'] - 0.002),
                    fontsize=10, fontweight='bold')
    ax.set_title('Pareto Efficiency: PR-AUC vs Model Complexity', fontweight='bold')
    ax.set_xlabel('Total Trainable Parameters (Thousands)')
    ax.set_ylabel('Test PR-AUC')
    ax.grid(True, linestyle=':', alpha=0.6)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Hops ($L$)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Comparison chart saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Experiment G: Hypergraph Message Passing Depth & Hop Ablation (Exp B Architecture)")
    parser.add_argument('--hops', nargs='+', type=int, default=[1, 2, 3, 4], help="List of hop depths to evaluate")
    parser.add_argument('--epochs', type=int, default=100, help="Training epochs per hop model")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (hyperedges)")
    parser.add_argument('--lr', type=float, default=0.0005, help="Learning rate (matches Exp B: 5e-4)")
    parser.add_argument('--weight_decay', type=float, default=0.001, help="Weight decay (matches Exp B: 1e-3)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--val_every', type=int, default=5, help="Validate every N epochs")
    parser.add_argument('--data_dir', type=str, default=None, help="Path to data directory")
    parser.add_argument('--plot_only', action='store_true', help="Only regenerate plot from hop_comparison_metrics.csv")
    args, _ = parser.parse_known_args()
    
    if args.plot_only:
        if os.path.exists('hop_comparison_metrics.csv'):
            df = pd.read_csv('hop_comparison_metrics.csv')
            plot_hop_comparison(df)
            return
        else:
            print("Error: hop_comparison_metrics.csv not found.")
            return

    set_random_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    base_path = args.data_dir if args.data_dir else find_dataset_path()
    data = load_data(base_path, device)
    
    results = []
    
    print("\n" + "="*70)
    print("STARTING EXPERIMENT G: MESSAGE PASSING HOP DEPTH ABLATION (EXP B)")
    print("Controlled Settings (100% Strict Parity with Exp B):")
    print(f"  - Optimizer: AdamW(lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"  - Loss: BCEWithLogitsLoss / FocalLoss")
    print(f"  - Batch Size: {args.batch_size} | Epochs: {args.epochs} | Seed: {args.seed}")
    print("="*70 + "\n")
    
    for h in args.hops:
        print(f"\n>>> [Hop Level L = {h}] Initializing {h}-Layer HyperAttDDI_NoSE Model <<<")
        set_random_seed(args.seed)
        
        model = HyperAttDDI_HopAblation(
            in_dim=768,
            emb_dim=256,
            conv_dim=64,
            heads=4,
            num_layers=h,
            d=64,
            p=0.1
        ).to(device)
        model.apply(init_weights)
        
        total_p, conv_p = count_parameters(model)
        print(f"Model Summary for L={h}: Total Params = {total_p:,} | Conv Params = {conv_p:,}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_auc = 0.0
        best_model_state = None
        epoch_times = []
        
        start_train_t = time.time()
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            
            perm = torch.randperm(data['train_inc'].shape[1])
            train_inc = data['train_inc'][:, perm]
            train_lab = data['train_lab'][perm].to(device)
            
            num_batches = int(np.ceil(train_inc.shape[1] / args.batch_size))
            epoch_loss = 0.0
            
            for b in range(num_batches):
                start_idx = b * args.batch_size
                end_idx = min(start_idx + args.batch_size, train_inc.shape[1])
                
                batch_inc = train_inc[:, start_idx:end_idx].to(device)
                batch_lab = train_lab[start_idx:end_idx]
                
                optimizer.zero_grad()
                logits = model(data['drug_features'], batch_inc)
                loss = criterion(logits, batch_lab)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            epoch_times.append(time.time() - t0)
            
            if epoch % args.val_every == 0 or epoch == args.epochs:
                val_auc, val_prauc, val_f1, _, _, val_loss = evaluate_dataset(
                    model, data['drug_features'], data['val_inc'], data['val_lab'], device
                )
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_state = copy.deepcopy(model.state_dict())
                print(f"Epoch {epoch:03d}/{args.epochs} | Train Loss: {epoch_loss/num_batches:.4f} | Val AUC: {val_auc:.4f} | Val PR-AUC: {val_prauc:.4f}")
                
        total_train_time = time.time() - start_train_t
        avg_epoch_t = np.mean(epoch_times)
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        test_auc, test_prauc, test_f1, test_pr, test_re, test_loss = evaluate_dataset(
            model, data['drug_features'], data['test_inc'], data['test_lab'], device
        )
        
        val_auc, val_prauc, _, _, _, _ = evaluate_dataset(
            model, data['drug_features'], data['val_inc'], data['val_lab'], device
        )
        
        print(f"\n--- Final Results for L={h} Hops ---")
        print(f"Test ROC-AUC:    {test_auc:.4f}")
        print(f"Test PR-AUC:     {test_prauc:.4f}")
        print(f"Test F1-Score:   {test_f1:.4f}")
        print(f"Test Precision:  {test_pr:.4f} | Test Recall: {test_re:.4f}")
        print(f"Avg Epoch Time:  {avg_epoch_t:.2f}s (Total: {total_train_time:.1f}s)")
        
        results.append({
            'hops': h,
            'test_auc': round(test_auc, 4),
            'test_prauc': round(test_prauc, 4),
            'test_f1': round(test_f1, 4),
            'test_precision': round(test_pr, 4),
            'test_recall': round(test_re, 4),
            'test_loss': round(test_loss, 4),
            'val_auc': round(val_auc, 4),
            'val_prauc': round(val_prauc, 4),
            'total_params': total_p,
            'conv_params': conv_p,
            'avg_epoch_time_s': round(avg_epoch_t, 2)
        })

    results_df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("FINAL EXPERIMENT G SUMMARY TABLE:")
    print("="*70)
    print(results_df.to_string(index=False))
    
    results_df.to_csv('hop_comparison_metrics.csv', index=False)
    print("\nSaved metrics to 'hop_comparison_metrics.csv'")
    
    plot_hop_comparison(results_df, 'message_passing_hops_comparison.png')

if __name__ == '__main__':
    main()
