import os
import ast
import math
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

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

class HyperAttDDI_MeanPool(nn.Module):
    """
    Ablation Experiment E: HyperAttDDI with Mean Pooling instead of Attention Pooling.
    
    Ablation Goal:
    - Replace Module 3 learned attention pooling with simple mean aggregation over combination drugs.
    - Proves whether learned attention pooling outperforms naive mean pooling (Exp B vs Exp E).
    
    Architecture:
    - Module 1 (Drug Encoder): ChemBERTa (768) -> Linear(768->256) -> ReLU -> Dropout
    - Module 2 (Hypergraph Encoder): 2-layer PyG HypergraphConv(256, 64, heads=4, use_attention=True)
    - Module 3 (Mean Pooling - No Attention): h_e = (H_T @ X2) / degree_e
    - Module 5 (Decoder): Linear(256->128) -> ReLU -> Dropout -> Linear(128->1)
    """
    def __init__(self, in_dim=768, emb_dim=256, conv_dim=64, heads=4, p=0.1):
        super().__init__()
        self.p = p
        
        # Module 1: Drug Encoder
        self.drug_encoder = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(p=p)
        )
        
        # Module 2: Hypergraph Encoder (2 layers, 64 * 4 = 256)
        self.conv1 = hnn.HypergraphConv(emb_dim, conv_dim, heads=heads, use_attention=True, dropout=p)
        self.conv2 = hnn.HypergraphConv(conv_dim * heads, conv_dim, heads=heads, use_attention=True, dropout=p)
        
        # Module 5: Interaction Decoder (Input is mean-pooled combination embedding)
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
        
        # 1. Drug Encoder
        X = self.drug_encoder(drug_features)  # (N, 256)
        
        # 2. Hypergraph Message Passing
        row, col = torch.where(H_T)
        edges = torch.cat([col.view(1, -1), row.view(1, -1)], dim=0).to(drug_features.device)
        degree_e = H_T.sum(dim=1, keepdim=True).clamp(min=1)
        
        attr1 = (H_T @ X) / degree_e
        X1 = F.relu(self.conv1(X, edges, hyperedge_attr=attr1))
        X1 = F.dropout(X1, p=self.p, training=self.training)
        
        attr2 = (H_T @ X1) / degree_e
        X2 = F.relu(self.conv2(X1, edges, hyperedge_attr=attr2))
        X2 = F.dropout(X2, p=self.p, training=self.training)
        
        # 3. MEAN POOLING (Ablation: equal weighting of member drugs, no attention)
        h_e = (H_T @ X2) / degree_e  # (E, 256)
        
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
    
    print(f"Loading drug embeddings from: {emb_path}")
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

    def build_incidence(pos_df, neg_df):
        num_pos = len(pos_df)
        num_neg = len(neg_df)
        
        inc_pos = np.zeros((num_drugs, num_pos), dtype=np.float32)
        pos_drug_lists = pos_df['DrugBankID'].values
        for col_idx in range(num_pos):
            for drug_id in ast.literal_eval(pos_drug_lists[col_idx]):
                if drug_id in drug_to_index:
                    inc_pos[drug_to_index[drug_id], col_idx] = 1.0
                    
        inc_neg = np.zeros((num_drugs, num_neg), dtype=np.float32)
        neg_drug_lists = neg_df['DrugBankID'].values
        for col_idx in range(num_neg):
            for drug_id in ast.literal_eval(neg_drug_lists[col_idx]):
                if drug_id in drug_to_index:
                    inc_neg[drug_to_index[drug_id], col_idx] = 1.0
                    
        labels_pos = np.ones(num_pos, dtype=np.float32)
        labels_neg = np.zeros(num_neg, dtype=np.float32)
        return inc_pos, inc_neg, labels_pos, labels_neg

    print("Building incidence matrices...")
    train_inc_pos, train_inc_neg, train_lab_pos, train_lab_neg = build_incidence(train_data_pos, train_data_neg)
    val_inc_pos, val_inc_neg, val_lab_pos, val_lab_neg = build_incidence(val_data_pos, val_data_neg)
    test_inc_pos, test_inc_neg, test_lab_pos, test_lab_neg = build_incidence(test_data_pos, test_data_neg)
    
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
        'num_drugs': num_drugs,
        'drug_to_index': drug_to_index,
        'valid_drugs': valid_drugs,
        'test_pos_df': test_data_pos,
        'test_neg_df': test_data_neg
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
    binary_preds = (all_scores >= 0.5).astype(int)
    
    pr = precision_score(labels, binary_preds, zero_division=0)
    re = recall_score(labels, binary_preds, zero_division=0)
    f1 = f1_score(labels, binary_preds, zero_division=0)
    auc = roc_auc_score(labels, all_scores)
    prauc = average_precision_score(labels, all_scores)
    
    return {
        'precision': pr,
        'recall': re,
        'f1': f1,
        'auc': auc,
        'prauc': prauc,
        'scores': all_scores
    }

def main():
    parser = argparse.ArgumentParser(description="HyperAttDDI Ablation: Mean Pooling (Exp E)")
    parser.add_argument('--lr', type=float, default=0.0005, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.001, help="Weight decay")
    parser.add_argument('--epochs', type=int, default=100, help="Training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--eval_batch_size', type=int, default=512, help="Eval batch size")
    parser.add_argument('--emb_dim', type=int, default=256, help="Embedding dimension")
    parser.add_argument('--conv_dim', type=int, default=64, help="Hypergraph conv dimension per head")
    parser.add_argument('--heads', type=int, default=4, help="Attention heads")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout probability")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--output_dir', type=str, default='output_hyperattddi_e', help="Output directory")
    args, _ = parser.parse_known_args()
    
    set_random_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    base_path = find_dataset_path()
    print(f"Dataset root found: {base_path}")
    
    data = load_data(base_path, device)
    drug_features = data['drug_features']
    train_inc = data['train_inc']
    y_train = data['train_lab']
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    model = HyperAttDDI_MeanPool(
        in_dim=768,
        emb_dim=args.emb_dim,
        conv_dim=args.conv_dim,
        heads=args.heads,
        p=args.dropout
    ).to(device)
    
    model.apply(init_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"\nModel initialized: {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
    print("Starting training (HyperAttDDI Ablation: Mean Pooling - Exp E)...\n")
    
    max_valid_f1 = 0.0
    best_epoch = -1
    best_metrics = None
    best_weights = None
    
    num_samples = train_inc.shape[1]
    num_batches = num_samples // args.batch_size
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        indices = torch.randperm(num_samples)
        shuffled_train_inc = train_inc[:, indices]
        shuffled_y_train = y_train[indices]
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1:03d}/{args.epochs:03d}", leave=False)
        for b in pbar:
            optimizer.zero_grad()
            batch_inc = shuffled_train_inc[:, b * args.batch_size:(b + 1) * args.batch_size].to(device)
            batch_y = shuffled_y_train[b * args.batch_size:(b + 1) * args.batch_size].to(device)
            
            logits = model(drug_features, batch_inc)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / num_batches
        
        # Validation
        val_res = evaluate_dataset(
            model, drug_features, data['val_inc'], data['val_lab'], device, batch_size=args.eval_batch_size
        )
        
        val_f1 = val_res['f1']
        val_auc = val_res['auc']
        
        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f} | Val Precision: {val_res['precision']:.4f} | Val Recall: {val_res['recall']:.4f}")
        
        if val_f1 > max_valid_f1:
            max_valid_f1 = val_f1
            best_epoch = epoch + 1
            best_metrics = val_res
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_weights, os.path.join(args.output_dir, 'best_hyperattddi_exp_e.pt'))
            print(f"  --> Saved new best model checkpoint (Val F1: {val_f1:.4f})")
            
    print(f"\n=======================================================")
    print(f"Training Complete. Best Validation F1: {max_valid_f1:.4f} (Epoch {best_epoch})")
    print(f"Loading best checkpoint for Final Test Evaluation...")
    print(f"=======================================================")
    
    model.load_state_dict(best_weights)
    model.to(device)
    
    test_res = evaluate_dataset(
        model, drug_features, data['test_inc'], data['test_lab'], device, batch_size=args.eval_batch_size
    )
    
    print("\n--- Final Test Results (HyperAttDDI Mean Pooling - Exp E) ---")
    print(f"Precision: {test_res['precision']:.4f}")
    print(f"Recall:    {test_res['recall']:.4f}")
    print(f"F1 Score:  {test_res['f1']:.4f}")
    print(f"AUC:       {test_res['auc']:.4f}")
    print(f"PRAUC:     {test_res['prauc']:.4f}")
    print("-------------------------------------------------------------\n")
    
    res_df = pd.DataFrame([{
        'Model': 'HyperAttDDI (Exp E - Mean Pooling Ablation)',
        'Precision': test_res['precision'],
        'Recall': test_res['recall'],
        'F1': test_res['f1'],
        'AUC': test_res['auc'],
        'PRAUC': test_res['prauc'],
        'Best_Epoch': best_epoch,
        'Best_Val_F1': max_valid_f1
    }])
    res_df.to_csv(os.path.join(args.output_dir, 'exp_e_results.csv'), index=False)
    print(f"Results saved to: {os.path.join(args.output_dir, 'exp_e_results.csv')}")

if __name__ == '__main__':
    main()
