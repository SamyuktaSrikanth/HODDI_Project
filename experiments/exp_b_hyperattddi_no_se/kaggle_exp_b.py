import os, ast, math, copy, random, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as hnn
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, average_precision_score

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class HyperAttDDI_NoSE(nn.Module):
    """
    HyperAttDDI without Side-Effect Features (Exp B).
    Now takes both SMILES features and Structural Incidence features (like Exp A).
    """
    def __init__(self, struct_dim, in_dim=768, emb_dim=256, conv_dim=64, heads=4, d=64, p=0.1):
        super().__init__()
        self.d = d
        self.p = p
        
        # 1. Drug Encoder (Text + Structure)
        self.struct_encoder = nn.Linear(struct_dim, 128)
        self.drug_encoder = nn.Linear(in_dim, 128)
        self.input_dropout = nn.Dropout(p=p)
        
        # 2. Hypergraph Encoder: L=2 layers (conv_dim * heads = 256)
        self.conv1 = hnn.HypergraphConv(emb_dim, conv_dim, heads=heads, use_attention=True, dropout=p)
        self.conv2 = hnn.HypergraphConv(conv_dim * heads, conv_dim, heads=heads, use_attention=True, dropout=p)
        
        # 3. Attention Pooling
        self.W_q = nn.Linear(emb_dim, d)
        self.W_k = nn.Linear(emb_dim, d)
        self.W_v = nn.Linear(emb_dim, emb_dim)
        
        # 5. Decoder
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(128, 1)
        )
        
    def forward(self, drug_features, struct_features, inc_matrix):
        N, E = inc_matrix.shape
        H_T = inc_matrix.T  
        
        s_emb = F.relu(self.struct_encoder(struct_features))
        d_emb = F.relu(self.drug_encoder(drug_features))
        X = torch.cat([s_emb, d_emb], dim=1)  # (N, 256)
        X = self.input_dropout(X)
        
        row, col = torch.where(H_T)
        edges = torch.cat([col.view(1, -1), row.view(1, -1)], dim=0).to(drug_features.device)
        
        degree_e = H_T.sum(dim=1, keepdim=True).clamp(min=1)
        
        attr1 = (H_T @ X) / degree_e
        X1 = F.relu(self.conv1(X, edges, hyperedge_attr=attr1))
        X1 = F.dropout(X1, p=self.p, training=self.training)
        
        attr2 = (H_T @ X1) / degree_e
        X2 = F.relu(self.conv2(X1, edges, hyperedge_attr=attr2))
        X2 = F.dropout(X2, p=self.p, training=self.training)
        
        h_e_mean = (H_T @ X2) / degree_e
        q = self.W_q(h_e_mean)
        k = self.W_k(X2)
        v = self.W_v(X2)
        
        scores = torch.matmul(q, k.T) / math.sqrt(self.d)
        scores = scores.masked_fill(H_T == 0, -1e9)
        alpha = F.softmax(scores, dim=1)
        h_e = torch.matmul(alpha, v)
        
        logits = self.decoder(h_e).squeeze(-1)
        return logits

def find_file(filename, search_roots=['/kaggle/input', '/kaggle/working', '.', '..', '../..', 'dataset']):
    for root_dir in search_roots:
        if os.path.exists(root_dir):
            for root, dirs, files in os.walk(root_dir):
                if filename in files:
                    return os.path.join(root, filename)
    return None

def find_dir(dirname, search_roots=['/kaggle/input', '.', '..', '../..', 'dataset']):
    for root_dir in search_roots:
        if os.path.exists(root_dir):
            for root, dirs, files in os.walk(root_dir):
                if dirname in dirs:
                    return os.path.join(root, dirname)
    return None

def load_data(device):
    subset_dir = find_dir('subset_drug2-8_SE5-50')
    dict_path = find_file('Drugbank_ID_SMILE_all_structure links.csv')
    smiles_ds = pd.read_csv(dict_path)
    # CRITICAL FIX: DO NOT drop NaNs. This keeps all 10,250 drugs.
    drugbank_to_smiles = smiles_ds.set_index('DrugBank ID')['SMILES'].to_dict()

    merged_dir = os.path.join(subset_dir, 'merged_subset')
    all_ds = pd.concat([pd.read_csv(os.path.join(merged_dir, 'positive_samples_2014Q3_2024Q3_step6.csv')), 
                        pd.read_csv(os.path.join(merged_dir, 'negative_samples_2014Q3_2024Q3_step6.csv'))], axis=0)

    all_drugs = set()
    for drug_ids in all_ds['DrugBankID']:
        all_drugs.update([d for d in ast.literal_eval(drug_ids) if d.lower() != 'none'])

    drug_to_index_raw = {drug: idx for idx, drug in enumerate(all_drugs)}
    raw_drug_id_list = [None] * len(all_drugs)
    for did in drug_to_index_raw:
        raw_drug_id_list[drug_to_index_raw[did]] = did

    drug_id_list = [did for did in raw_drug_id_list if did in drugbank_to_smiles]
    drug_to_index = {did: idx for idx, did in enumerate(drug_id_list)}
    num_drugs = len(drug_id_list)

    # 4. Generate ChemBERTa Embeddings on the fly to ensure 10,250 drugs with EXACT author padding
    cache_path = '/kaggle/working/drug_embeddings_768d_author_padded_10250.pt'
    if os.path.exists(cache_path):
        print(f"Loading cached author-padded embeddings from: {cache_path}")
        extra_feature = torch.load(cache_path, map_location=device)
    else:
        print("Computing author-identical ChemBERTa embeddings on GPU (takes ~45s)...")
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        m_name = 'seyonec/PubChem10M_SMILES_BPE_450k'
        local_dir = find_dir('PubChem10M_SMILES_BPE_450k')
        if local_dir:
            m_name = local_dir
        tokenizer = AutoTokenizer.from_pretrained(m_name)
        chem_model = AutoModelForMaskedLM.from_pretrained(m_name).to(device)
        chem_model.eval()

        smiles_list = [str(drugbank_to_smiles[did]) for did in drug_id_list]
        chemberta_feat = []
        batch_size = 64
        with torch.no_grad():
            for i in tqdm(range(0, len(smiles_list), batch_size), desc="ChemBERTa Embeddings"):
                batch_smiles = smiles_list[i:i+batch_size]
                tokens = tokenizer(batch_smiles, return_tensors="pt", max_length=256, padding='max_length', truncation=True).to(device)
                outputs = chem_model(**tokens, output_hidden_states=True)
                chemberta_feat.append(outputs.hidden_states[-1].mean(dim=1))

        extra_feature = torch.cat(chemberta_feat, dim=0)
        torch.save(extra_feature, cache_path)

    def merge_quarters(sub_datasets):
        pos_merged, neg_merged = [], []
        for sub_ds in sub_datasets:
            pos_f = os.path.join(subset_dir, f'{sub_ds}_positive_samples_condition123_SE_above_0.9.csv')
            neg_f = os.path.join(subset_dir, f'{sub_ds}_negative_samples_condition123_SE_above_0.9.csv')
            if os.path.exists(pos_f): pos_merged.append(pd.read_csv(pos_f))
            if os.path.exists(neg_f): neg_merged.append(pd.read_csv(neg_f))
        return pd.concat(pos_merged, axis=0), pd.concat(neg_merged, axis=0)

    def build_incidence_pair(pos_df, neg_df):
        num_pos, num_neg = len(pos_df), len(neg_df)
        inc_pos = np.zeros((num_drugs, num_pos), dtype=np.float32)
        inc_neg = np.zeros((num_drugs, num_neg), dtype=np.float32)
        for col_idx, drug_list in enumerate(pos_df['DrugBankID']):
            for did in ast.literal_eval(drug_list):
                if did in drug_to_index: inc_pos[drug_to_index[did], col_idx] = 1.0
        for col_idx, drug_list in enumerate(neg_df['DrugBankID']):
            for did in ast.literal_eval(drug_list):
                if did in drug_to_index: inc_neg[drug_to_index[did], col_idx] = 1.0
        return inc_pos, inc_neg, np.ones(num_pos, dtype=np.int64), np.zeros(num_neg, dtype=np.int64)

    train_pos, train_neg = merge_quarters(['2015Q1', '2015Q2', '2015Q3', '2016Q4', '2017Q1', '2017Q2', '2017Q3', '2017Q4', '2018Q3', '2019Q1', '2019Q2', '2019Q3', '2019Q4', '2020Q1', '2020Q2', '2020Q3', '2020Q4', '2021Q1', '2021Q3', '2021Q4', '2022Q1', '2022Q2', '2022Q3', '2022Q4', '2023Q1', '2023Q2', '2023Q3', '2023Q4', '2024Q2'])
    val_pos, val_neg = merge_quarters(['2014Q3', '2015Q4', '2016Q1', '2016Q3', '2021Q2', '2024Q1'])
    test_pos, test_neg = merge_quarters(['2014Q4', '2016Q2', '2018Q1', '2018Q2', '2018Q4', '2024Q3'])

    train_inc_pos, train_inc_neg, train_lpos, train_lneg = build_incidence_pair(train_pos, train_neg)
    val_inc_pos, val_inc_neg, val_lpos, val_lneg = build_incidence_pair(val_pos, val_neg)
    test_inc_pos, test_inc_neg, test_lpos, test_lneg = build_incidence_pair(test_pos, test_neg)

    return {
        'train_inc_pos': torch.tensor(train_inc_pos, dtype=torch.float32).to(device),
        'train_inc_all': torch.tensor(np.concatenate([train_inc_pos, train_inc_neg], axis=1), dtype=torch.float32),
        'y_train_all': torch.tensor(np.concatenate([train_lpos, train_lneg]), dtype=torch.float32).to(device),
        'val_inc_all': torch.tensor(np.concatenate([val_inc_pos, val_inc_neg], axis=1), dtype=torch.float32).to(device),
        'y_val_all': np.concatenate([val_lpos, val_lneg]),
        'test_inc_all': torch.tensor(np.concatenate([test_inc_pos, test_inc_neg], axis=1), dtype=torch.float32).to(device),
        'y_test_all': np.concatenate([test_lpos, test_lneg]),
        'drug_features': extra_feature,
        'num_drugs': num_drugs,
    }

def evaluate(model, drug_features, struct_features, inc_matrix, labels, device, threshold=0.5, batch_size=512):
    model.eval()
    all_scores = []
    with torch.no_grad():
        for i in range(0, inc_matrix.shape[1], batch_size):
            batch_inc = inc_matrix[:, i:i+batch_size].to(device)
            logits = model(drug_features, struct_features, batch_inc)
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
    all_scores = np.concatenate(all_scores, axis=0)
    preds = (all_scores >= threshold).astype(int)
    return f1_score(labels, preds, zero_division=0), roc_auc_score(labels, all_scores), average_precision_score(labels, all_scores), precision_score(labels, preds, zero_division=0), recall_score(labels, preds, zero_division=0), all_scores

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_data(device)
    
    seeds = [42, 3407, 54321, 123456, 7, 777, 31415]
    metrics = {"f1": [], "auc": [], "prauc": [], "pr": [], "re": []}
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        set_random_seed(seed)
        
        model = HyperAttDDI_NoSE(
            struct_dim=data['train_inc_pos'].shape[1],
            in_dim=768, emb_dim=256, conv_dim=64, heads=4, d=64, p=0.1
        ).to(device)
        
        if seed == seeds[0]:
            print(f"Total Trainable Parameters (Exp B): {count_parameters(model):,}")
            
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.001)
        
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1, gamma=2.0, reduction='mean'):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.reduction = reduction
            def forward(self, inputs, targets):
                bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
                pt = torch.exp(-bce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
                if self.reduction == 'mean': return focal_loss.mean()
                return focal_loss.sum()
                
        criterion = FocalLoss(gamma=2.0)
        
        best_f1, best_state, best_thresh = 0, None, 0.5
        
        for epoch in tqdm(range(100), desc="Training"):
            model.train()
            generator = torch.Generator().manual_seed(seed)
            perm = torch.randperm(data['train_inc_all'].shape[1], generator=generator)
            train_inc = data['train_inc_all'][:, perm]
            y_train = data['y_train_all'][perm]
            
            for i in range(0, train_inc.shape[1], 64):
                batch_inc = train_inc[:, i:i+64].to(device)
                batch_y = y_train[i:i+64]
                
                optimizer.zero_grad()
                logits = model(data['drug_features'], data['train_inc_pos'], batch_inc)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
            _, _, _, _, _, val_scores = evaluate(model, data['drug_features'], data['train_inc_pos'], data['val_inc_all'], data['y_val_all'], device)
            
            best_epoch_thresh = 0.5
            best_epoch_f1 = 0.0
            for thresh in np.arange(0.1, 0.9, 0.02):
                b_score = (val_scores >= thresh).astype(int)
                f1 = f1_score(data['y_val_all'], b_score, zero_division=0)
                if f1 > best_epoch_f1:
                    best_epoch_f1 = f1
                    best_epoch_thresh = thresh
            
            if best_epoch_f1 > best_f1:
                best_f1 = best_epoch_f1
                best_thresh = best_epoch_thresh
                best_state = copy.deepcopy(model.state_dict())
                
        model.load_state_dict(best_state)
        f1, auc, prauc, pr, re, _ = evaluate(model, data['drug_features'], data['train_inc_pos'], data['test_inc_all'], data['y_test_all'], device, threshold=best_thresh)
        
        metrics["f1"].append(f1)
        metrics["auc"].append(auc)
        metrics["prauc"].append(prauc)
        metrics["pr"].append(pr)
        metrics["re"].append(re)
        print(f"Seed {seed} Test -> AUC: {auc:.4f} | PR-AUC: {prauc:.4f} | F1: {f1:.4f} | PR: {pr:.4f} | RE: {re:.4f}")

    print("\nFINAL EXP B RESULTS (7 Seeds):")
    for k, v in metrics.items():
        print(f"{k.upper()}: {np.mean(v):.4f} ± {np.std(v):.4f}")
