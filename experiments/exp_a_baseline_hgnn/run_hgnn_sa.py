import os
import glob
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch_geometric.nn as hnn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from tqdm import tqdm
import random

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HGNN_SA(nn.Module):
    def __init__(self, input_num, input_feature_num, emb_dim=128, conv_dim=64, head=3, p=0.1):
        super(HGNN_SA, self).__init__()
        self.emb_dim = emb_dim
        self.conv_dim = conv_dim
        self.head = head
        self.p = p
        
        self.linear_encoder = nn.Linear(input_feature_num, emb_dim)
        self.pre_linear = nn.Linear(768, emb_dim) 
        self.in_channel = 2 * emb_dim
        
        self.relu = nn.ReLU()
        self.hypergraph_conv = hnn.HypergraphConv(self.in_channel, conv_dim, heads=head, use_attention=True, dropout=p)
        self.hyper_attr_liner = nn.Linear(input_num, self.in_channel)
        self.hyperedge_linear = nn.Linear(conv_dim * head, 2)
        
    def forward(self, input_features, incidence_matrix, extra_feature):
        incidence_matrix_T = incidence_matrix.T
        
        input_nodes_features = self.relu(self.linear_encoder(input_features))
        
        row, col = torch.where(incidence_matrix_T)
        edges = torch.cat((col.view(1, -1), row.view(1, -1)), dim=0).to(incidence_matrix.device)
        
        hyperedge_attr = self.hyper_attr_liner(incidence_matrix_T)
        
        extra_feat = self.relu(self.pre_linear(extra_feature))
        input_nodes_features = torch.cat((extra_feat, input_nodes_features), dim=1)
        
        input_nodes_features = self.hypergraph_conv(input_nodes_features, edges, hyperedge_attr=hyperedge_attr)
        hyperedge_feature = torch.mm(incidence_matrix_T, input_nodes_features)
        return self.hyperedge_linear(hyperedge_feature)

def find_dataset_path():
    # Kaggle mounts uploaded datasets under /kaggle/input/
    if os.path.exists('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            if 'drug_embeddings_768d.pt' in files:
                return root
    return 'dataset'

def load_data(base_path):
    dataset_base_dir = os.path.join(base_path, 'evaluation_subset', 'subset_drug2-8_SE5-50', '')
    training_sub_ds_names = ['2015Q1', '2015Q2', '2015Q3', '2016Q4', '2017Q1', '2017Q2', '2017Q3', '2017Q4', '2018Q3', '2019Q1', '2019Q2', '2019Q3', '2019Q4', '2020Q1', '2020Q2', '2020Q3', '2020Q4', '2021Q1', '2021Q3', '2021Q4', '2022Q1', '2022Q2', '2022Q3', '2022Q4', '2023Q1', '2023Q2', '2023Q3', '2023Q4', '2024Q2']
    validating_sub_ds_names = ['2014Q3', '2015Q4', '2016Q1', '2016Q3', '2021Q2', '2024Q1']
    testing_sub_ds_names = ['2014Q4', '2016Q2', '2018Q1', '2018Q2', '2018Q4', '2024Q3']
    
    emb_path = os.path.join(base_path, "drug_embeddings_768d.pt")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Missing embeddings file: {emb_path}. Make sure the dataset is loaded correctly.")
    drug_embeddings_dict = torch.load(emb_path, weights_only=True)
    
    def merge_sub_datasets(sub_datasets):
        pos_merged_data = pd.DataFrame()
        neg_merged_data = pd.DataFrame()
        for sub_ds in sub_datasets:
            pos_file_path = os.path.join(dataset_base_dir, f'{sub_ds}_positive_samples_condition123_SE_above_0.9.csv')
            neg_file_path = os.path.join(dataset_base_dir, f'{sub_ds}_negative_samples_condition123_SE_above_0.9.csv')
            if os.path.exists(pos_file_path):
                pos_merged_data = pd.concat([pos_merged_data, pd.read_csv(pos_file_path)], axis=0)
            if os.path.exists(neg_file_path):
                neg_merged_data = pd.concat([neg_merged_data, pd.read_csv(neg_file_path)], axis=0)
        return pos_merged_data, neg_merged_data

    print("Loading csv files...")
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
                    inc_pos[drug_to_index[drug_id], col_idx] = 1
                    
        for col_idx, drug_list in enumerate(neg_df['DrugBankID']):
            for drug_id in ast.literal_eval(drug_list):
                if drug_id in drug_to_index:
                    inc_neg[drug_to_index[drug_id], col_idx] = 1
                    
        labels_pos = np.ones(len(pos_df), dtype=np.int64)
        labels_neg = np.zeros(len(neg_df), dtype=np.int64)
        return inc_pos, inc_neg, labels_pos, labels_neg

    print("Building incidence matrices...")
    train_inc_pos, train_inc_neg, train_lab_pos, train_lab_neg = build_incidence_matrix(train_data_pos, train_data_neg)
    val_inc_pos, val_inc_neg, val_lab_pos, val_lab_neg = build_incidence_matrix(val_data_pos, val_data_neg)
    test_inc_pos, test_inc_neg, test_lab_pos, test_lab_neg = build_incidence_matrix(test_data_pos, test_data_neg)
    
    extra_feature = torch.zeros((num_drugs, 768), dtype=torch.float32)
    for drug, idx in drug_to_index.items():
        extra_feature[idx] = drug_embeddings_dict[drug]
        
    return {
        'train_inc_pos': torch.tensor(train_inc_pos, dtype=torch.float32).to(device),
        'train_inc_neg': torch.tensor(train_inc_neg, dtype=torch.float32).to(device),
        'val_inc_pos': torch.tensor(val_inc_pos, dtype=torch.float32).to(device),
        'val_inc_neg': torch.tensor(val_inc_neg, dtype=torch.float32).to(device),
        'test_inc_pos': torch.tensor(test_inc_pos, dtype=torch.float32).to(device),
        'test_inc_neg': torch.tensor(test_inc_neg, dtype=torch.float32).to(device),
        'train_lab_pos': train_lab_pos,
        'train_lab_neg': train_lab_neg,
        'val_lab_pos': val_lab_pos,
        'val_lab_neg': val_lab_neg,
        'test_lab_pos': test_lab_pos,
        'test_lab_neg': test_lab_neg,
        'extra_feature': extra_feature.to(device),
        'num_drugs': num_drugs
    }

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def main():
    print(f"Using device: {device}")
    set_random_seed(42)
    
    base_path = find_dataset_path()
    print(f"Found dataset path: {base_path}")
    data = load_data(base_path)
    
    train_inc = torch.cat([data['train_inc_pos'], data['train_inc_neg']], dim=1)
    y_train = torch.tensor(np.concatenate([data['train_lab_pos'], data['train_lab_neg']]), dtype=torch.long).to(device)
    val_inc = torch.cat([data['val_inc_pos'], data['val_inc_neg']], dim=1)
    y_valid = torch.tensor(np.concatenate([data['val_lab_pos'], data['val_lab_neg']]), dtype=torch.long).to(device)
    test_inc = torch.cat([data['test_inc_pos'], data['test_inc_neg']], dim=1)
    y_test = torch.tensor(np.concatenate([data['test_lab_pos'], data['test_lab_neg']]), dtype=torch.long).to(device)
    
    indices = torch.randperm(train_inc.shape[1])
    train_inc = train_inc[:, indices]
    y_train = y_train[indices]
    
    model = HGNN_SA(
        input_num=data['num_drugs'],
        input_feature_num=data['train_inc_pos'].shape[1],
        emb_dim=128,
        conv_dim=64,
        head=3,
        p=0.1
    ).to(device)
    
    model.apply(init_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 100
    batch_size = 64
    max_valid_f1 = 0
    best_model_wts = None
    
    print("Starting training...")
    
    # Using tqdm for a progress bar!
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = train_inc.shape[1] // batch_size
        
        for e in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            optimizer.zero_grad()
            batch_inc = train_inc[:, e * batch_size:(e + 1) * batch_size]
            batch_y = y_train[e * batch_size:(e + 1) * batch_size]
            
            y_pred = model(data['train_inc_pos'], batch_inc, data['extra_feature'])
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        model.eval()
        with torch.no_grad():
            y_pred_val = model(data['train_inc_pos'], val_inc, data['extra_feature'])
            y_pred_val_softmax = F.softmax(y_pred_val, dim=1)
            scores = y_pred_val_softmax[:, 1].cpu().numpy()
            
        b_score = [int(s >= 0.5) for s in scores]
        y_valid_np = y_valid.cpu().numpy()
        f1 = f1_score(y_valid_np, b_score, zero_division=0)
        
        # Print progress at the end of each epoch
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/num_batches:.4f} | Val F1: {f1:.4f}")
        
        if f1 > max_valid_f1:
            max_valid_f1 = f1
            best_model_wts = {k: v.cpu() for k, v in model.state_dict().items()}
            
    print(f"Best Validation F1: {max_valid_f1:.4f}")
    
    model.load_state_dict(best_model_wts)
    model.to(device)
    model.eval()
    with torch.no_grad():
        y_pred_test = model(data['train_inc_pos'], test_inc, data['extra_feature'])
        y_pred_test_softmax = F.softmax(y_pred_test, dim=1)
        test_scores = y_pred_test_softmax[:, 1].cpu().numpy()
        
    y_test_np = y_test.cpu().numpy()
    b_score_test = [int(s >= 0.5) for s in test_scores]
    
    pr = precision_score(y_test_np, b_score_test, zero_division=0)
    re = recall_score(y_test_np, b_score_test, zero_division=0)
    f1 = f1_score(y_test_np, b_score_test, zero_division=0)
    auc_score = roc_auc_score(y_test_np, test_scores)
    aupr = average_precision_score(y_test_np, test_scores)
    
    print("\n--- Final Test Results (HGNN-SA) ---")
    print(f"Precision: {pr:.4f}")
    print(f"Recall:    {re:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc_score:.4f}")
    print(f"AUPR:      {aupr:.4f}")

if __name__ == '__main__':
    main()
