import os
import ast
import math
import time
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    HyperAttDDI with Parameterized Message Passing Depth (Exp G).
    """
    def __init__(self, in_dim=768, emb_dim=256, conv_dim=64, heads=4, num_layers=2, d=64, p=0.1):
        super().__init__()
        self.d = d
        self.p = p
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        
        # Module 1: Drug Encoder
        self.drug_encoder = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(p=p)
        )
        
        # Module 2: Dynamic Hypergraph Message Passing (L layers)
        self.convs = nn.ModuleList()
        if num_layers > 0:
            self.convs.append(hnn.HypergraphConv(emb_dim, conv_dim, heads=heads, use_attention=True, dropout=p))
            for _ in range(1, num_layers):
                self.convs.append(hnn.HypergraphConv(conv_dim * heads, conv_dim, heads=heads, use_attention=True, dropout=p))
                
        # Module 4: Side-Effect Encoder
        self.se_encoder = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(p=p)
        )
        
        # Module 3: Adverse-Event Conditioned Attention Pooling
        self.W_q = nn.Linear(emb_dim * 2, d)
        self.W_k = nn.Linear(emb_dim, d)
        self.W_v = nn.Linear(emb_dim, emb_dim)
        
        # Module 5: Interaction Decoder
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(128, 1)
        )
        
    def forward(self, drug_features, inc_matrix, se_features, return_attention=False):
        N, E = inc_matrix.shape
        H_T = inc_matrix.T
        degree_e = H_T.sum(dim=1, keepdim=True).clamp(min=1)
        
        X = self.drug_encoder(drug_features)
        
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
            
        h_se = self.se_encoder(se_features)
        
        h_e_mean = (H_T @ X_final) / degree_e
        q_input = torch.cat([h_e_mean, h_se], dim=1)
        q = self.W_q(q_input)
        k = self.W_k(X_final)
        v = self.W_v(X_final)
        
        scores = torch.matmul(q, k.T) / math.sqrt(self.d)
        scores = scores.masked_fill(H_T == 0, -1e9)
        alpha = F.softmax(scores, dim=1)
        h_combo = torch.matmul(alpha, v)
        
        z = torch.cat([h_combo, h_se], dim=1)
        logits = self.decoder(z).squeeze(-1)
        
        if return_attention:
            return logits, alpha
        return logits

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    conv_params = sum(p.numel() for p in model.convs.parameters()) if hasattr(model, 'convs') and model.num_layers > 0 else 0
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'conv_params': conv_params,
    }

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
    
    print(f"Loading drug embeddings from: {emb_path}")
    drug_embeddings_dict = torch.load(emb_path, map_location='cpu', weights_only=True)
    
    se_file_candidates = [
        os.path.join(base_path, 'dictionary', 'Side_effects_unique.csv'),
        os.path.join(base_path, 'dataset', 'dictionary', 'Side_effects_unique.csv'),
        'dataset/dictionary/Side_effects_unique.csv'
    ]
    se_file = None
    for cand in se_file_candidates:
        if os.path.exists(cand):
            se_file = cand
            break
    if se_file is None:
        raise FileNotFoundError(f"Missing Side_effects_unique.csv: checked {se_file_candidates}")
        
    print(f"Loading side-effect SapBERT embeddings from: {se_file}")
    se_df = pd.read_csv(se_file)
    feature_cols = [str(i) for i in range(768)]
    se_embeddings_dict = {}
    se_names_dict = {}
    for _, row in se_df.iterrows():
        cui = row['umls_cui_from_meddra']
        se_embeddings_dict[cui] = row[feature_cols].values.astype(np.float32)
        se_names_dict[cui] = str(row['side_effect_name'])
        
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

    def build_incidence_and_se(pos_df, neg_df):
        num_pos = len(pos_df)
        num_neg = len(neg_df)
        
        inc_pos = np.zeros((num_drugs, num_pos), dtype=np.float32)
        se_pos = np.zeros((num_pos, 768), dtype=np.float32)
        pos_cuis = pos_df['SE_above_0.9'].values
        pos_drug_lists = pos_df['DrugBankID'].values
        
        for col_idx in range(num_pos):
            cui = pos_cuis[col_idx]
            if cui in se_embeddings_dict:
                se_pos[col_idx] = se_embeddings_dict[cui]
            for drug_id in ast.literal_eval(pos_drug_lists[col_idx]):
                if drug_id in drug_to_index:
                    inc_pos[drug_to_index[drug_id], col_idx] = 1.0
                    
        inc_neg = np.zeros((num_drugs, num_neg), dtype=np.float32)
        se_neg = np.zeros((num_neg, 768), dtype=np.float32)
        neg_cuis = neg_df['SE_above_0.9'].values
        neg_drug_lists = neg_df['DrugBankID'].values
        
        for col_idx in range(num_neg):
            cui = neg_cuis[col_idx]
            if cui in se_embeddings_dict:
                se_neg[col_idx] = se_embeddings_dict[cui]
            for drug_id in ast.literal_eval(neg_drug_lists[col_idx]):
                if drug_id in drug_to_index:
                    inc_neg[drug_to_index[drug_id], col_idx] = 1.0
                    
        labels_pos = np.ones(num_pos, dtype=np.float32)
        labels_neg = np.zeros(num_neg, dtype=np.float32)
        
        return inc_pos, inc_neg, se_pos, se_neg, labels_pos, labels_neg

    print("Building incidence matrices and SapBERT side-effect representations...")
    train_inc_pos, train_inc_neg, train_se_pos, train_se_neg, train_lab_pos, train_lab_neg = build_incidence_and_se(train_data_pos, train_data_neg)
    val_inc_pos, val_inc_neg, val_se_pos, val_se_neg, val_lab_pos, val_lab_neg = build_incidence_and_se(val_data_pos, val_data_neg)
    test_inc_pos, test_inc_neg, test_se_pos, test_se_neg, test_lab_pos, test_lab_neg = build_incidence_and_se(test_data_pos, test_data_neg)
    
    extra_feature = torch.zeros((num_drugs, 768), dtype=torch.float32)
    for drug, idx in drug_to_index.items():
        extra_feature[idx] = drug_embeddings_dict[drug]
        
    train_inc = torch.tensor(np.concatenate([train_inc_pos, train_inc_neg], axis=1), dtype=torch.float32)
    train_se = torch.tensor(np.concatenate([train_se_pos, train_se_neg], axis=0), dtype=torch.float32)
    train_lab = torch.tensor(np.concatenate([train_lab_pos, train_lab_neg]), dtype=torch.float32)
    
    val_inc = torch.tensor(np.concatenate([val_inc_pos, val_inc_neg], axis=1), dtype=torch.float32)
    val_se = torch.tensor(np.concatenate([val_se_pos, val_se_neg], axis=0), dtype=torch.float32)
    val_lab = np.concatenate([val_lab_pos, val_lab_neg])
    
    test_inc = torch.tensor(np.concatenate([test_inc_pos, test_inc_neg], axis=1), dtype=torch.float32)
    test_se = torch.tensor(np.concatenate([test_se_pos, test_se_neg], axis=0), dtype=torch.float32)
    test_lab = np.concatenate([test_lab_pos, test_lab_neg])
    
    return {
        'train_inc': train_inc,
        'train_se': train_se,
        'train_lab': train_lab,
        'val_inc': val_inc,
        'val_se': val_se,
        'val_lab': val_lab,
        'test_inc': test_inc,
        'test_se': test_se,
        'test_lab': test_lab,
        'drug_features': extra_feature.to(device),
        'num_drugs': num_drugs,
    }

def evaluate_dataset(model, drug_features, inc_matrix, se_matrix, labels, device, batch_size=512):
    model.eval()
    all_scores = []
    num_edges = inc_matrix.shape[1]
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for i in range(0, num_edges, batch_size):
            batch_inc = inc_matrix[:, i:i+batch_size].to(device)
            batch_se = se_matrix[i:i+batch_size].to(device)
            batch_lab = torch.tensor(labels[i:i+batch_size], dtype=torch.float32).to(device)
            
            logits = model(drug_features, batch_inc, batch_se)
            loss = criterion(logits, batch_lab)
            total_loss += loss.item() * len(batch_lab)
            total_batches += len(batch_lab)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(probs)
            
    all_scores = np.concatenate(all_scores, axis=0)
    binary_preds = (all_scores >= 0.5).astype(int)
    
    pr = precision_score(labels, binary_preds, zero_division=0)
    re = recall_score(labels, binary_preds, zero_division=0)
    f1 = f1_score(labels, binary_preds, zero_division=0)
    auc = roc_auc_score(labels, all_scores)
    prauc = average_precision_score(labels, all_scores)
    avg_loss = total_loss / max(1, total_batches)
    
    return {
        'precision': pr,
        'recall': re,
        'f1': f1,
        'auc': auc,
        'prauc': prauc,
        'loss': avg_loss,
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    base_path = find_dataset_path()
    data = load_data(base_path, device)
    
    hops_to_test = [1, 2, 3, 4]
    results = []
    
    for h in hops_to_test:
        print(f"\nTraining {h}-Hop model...")
        set_random_seed(42)
        model = HyperAttDDI_HopAblation(num_layers=h).to(device)
        model.apply(init_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        train_inc = data['train_inc']
        train_se = data['train_se']
        train_lab = data['train_lab']
        num_train = train_inc.shape[1]
        
        best_val_auc = 0.0
        best_state = None
        
        for epoch in range(1, 101):
            model.train()
            perm = torch.randperm(num_train)
            for i in range(0, num_train, 64):
                idx = perm[i:i+64]
                batch_inc = train_inc[:, idx].to(device)
                batch_se = train_se[idx].to(device)
                batch_lab = train_lab[idx].to(device)
                
                optimizer.zero_grad()
                logits = model(data['drug_features'], batch_inc, batch_se)
                loss = criterion(logits, batch_lab)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            if epoch % 5 == 0 or epoch == 100:
                val_res = evaluate_dataset(model, data['drug_features'], data['val_inc'], data['val_se'], data['val_lab'], device)
                if val_res['auc'] > best_val_auc:
                    best_val_auc = val_res['auc']
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    
        model.load_state_dict(best_state)
        model.to(device)
        test_res = evaluate_dataset(model, data['drug_features'], data['test_inc'], data['test_se'], data['test_lab'], device)
        params = count_parameters(model)
        
        results.append({
            'hops': h,
            'test_auc': test_res['auc'],
            'test_prauc': test_res['prauc'],
            'test_f1': test_res['f1'],
            'test_loss': test_res['loss'],
            'total_params': params['total_params'],
        })
        print(f"Hops={h} -> AUC: {test_res['auc']:.4f}, PRAUC: {test_res['prauc']:.4f}, F1: {test_res['f1']:.4f}")
        
    df = pd.DataFrame(results)
    df.to_csv('hop_comparison_metrics.csv', index=False)
    print("All results saved to hop_comparison_metrics.csv")

if __name__ == '__main__':
    main()
