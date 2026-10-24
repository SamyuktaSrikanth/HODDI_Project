import os, ast, math, copy, random, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as hnn
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_score, recall_score

# ==========================================
# 1. UTILS & DATA LOADING
# ==========================================
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

def get_or_compute_drug_embeddings(drug_id_list, drugbank_to_smiles, base_path, device):
    emb_path = find_file('drug_embeddings_768d.pt')
    cached_emb = {}
    if emb_path and os.path.exists(emb_path):
        print(f"Loading pre-cached drug embeddings from: {emb_path}")
        cached_emb = torch.load(emb_path, map_location='cpu', weights_only=True)
            
    missing_drugs = [d for d in drug_id_list if d not in cached_emb and d in drugbank_to_smiles]
    
    if len(missing_drugs) > 0:
        print(f"Computing embeddings for {len(missing_drugs)} missing drugs via ChemBERTa...")
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        model_name = "seyonec/PubChem10M_SMILES_BPE_450k"
        
        local_dir = find_dir('PubChem10M_SMILES_BPE_450k')
        if local_dir and os.path.exists(local_dir):
            model_name = local_dir
            
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        chem_model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        chem_model.eval()
        
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(missing_drugs), batch_size):
                batch_ids = missing_drugs[i:i+batch_size]
                batch_smiles = [str(drugbank_to_smiles[did]) for did in batch_ids]
                tokens = tokenizer(batch_smiles, return_tensors="pt", max_length=256, padding='max_length', truncation=True)
                tokens = {k: v.to(device) for k, v in tokens.items()}
                outputs = chem_model(**tokens, output_hidden_states=True)
                vectors = outputs.hidden_states[-1].mean(dim=1).cpu()
                for did, v in zip(batch_ids, vectors):
                    cached_emb[did] = v
        print("Missing drug embeddings successfully generated!")

    num_drugs = len(drug_id_list)
    extra_feature = torch.zeros((num_drugs, 768), dtype=torch.float32)
    for idx, did in enumerate(drug_id_list):
        if did in cached_emb:
            extra_feature[idx] = cached_emb[did]
        else:
            print(f"Warning: No SMILES/embedding found for drug {did}, using zero vector.")
            
    return extra_feature.to(device)

def load_data(device):
    print("Loading Data...")
    subset_dir = find_dir('subset_drug2-8_SE5-50')
    dict_path = find_file('Drugbank_ID_SMILE_all_structure links.csv')
    smiles_ds = pd.read_csv(dict_path)
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

    def build_incidence_pair(pos_df, neg_df):
        num_pos = len(pos_df)
        inc_pos = np.zeros((num_drugs, num_pos), dtype=np.float32)
        for col_idx, drug_list in enumerate(pos_df['DrugBankID']):
            for did in ast.literal_eval(drug_list):
                if did in drug_to_index:
                    inc_pos[drug_to_index[did], col_idx] = 1.0

        num_neg = len(neg_df)
        inc_neg = np.zeros((num_drugs, num_neg), dtype=np.float32)
        for col_idx, drug_list in enumerate(neg_df['DrugBankID']):
            for did in ast.literal_eval(drug_list):
                if did in drug_to_index:
                    inc_neg[drug_to_index[did], col_idx] = 1.0

        lpos = np.ones(num_pos, dtype=np.int64)
        lneg = np.zeros(num_neg, dtype=np.int64)
        return inc_pos, inc_neg, lpos, lneg

    training_sub_ds_names = [
        '2015Q1', '2015Q2', '2015Q3', '2016Q4', '2017Q1', '2017Q2', '2017Q3', '2017Q4',
        '2018Q3', '2019Q1', '2019Q2', '2019Q3', '2019Q4', '2020Q1', '2020Q2', '2020Q3',
        '2020Q4', '2021Q1', '2021Q3', '2021Q4', '2022Q1', '2022Q2', '2022Q3', '2022Q4',
        '2023Q1', '2023Q2', '2023Q3', '2023Q4', '2024Q2'
    ]
    validating_sub_ds_names = ['2014Q3', '2015Q4', '2016Q1', '2016Q3', '2021Q2', '2024Q1']
    testing_sub_ds_names = ['2014Q4', '2016Q2', '2018Q1', '2018Q2', '2018Q4', '2024Q3']

    def merge_quarters(sub_datasets):
        pos_merged, neg_merged = [], []
        for sub_ds in sub_datasets:
            pos_f = os.path.join(subset_dir, f'{sub_ds}_positive_samples_condition123_SE_above_0.9.csv')
            neg_f = os.path.join(subset_dir, f'{sub_ds}_negative_samples_condition123_SE_above_0.9.csv')
            if os.path.exists(pos_f): pos_merged.append(pd.read_csv(pos_f))
            if os.path.exists(neg_f): neg_merged.append(pd.read_csv(neg_f))
        return pd.concat(pos_merged, axis=0), pd.concat(neg_merged, axis=0)

    train_pos, train_neg = merge_quarters(training_sub_ds_names)
    train_ipos, train_ineg, train_lpos, train_lneg = build_incidence_pair(train_pos, train_neg)

    val_pos, val_neg = merge_quarters(validating_sub_ds_names)
    val_ipos, val_ineg, val_lpos, val_lneg = build_incidence_pair(val_pos, val_neg)

    test_pos, test_neg = merge_quarters(testing_sub_ds_names)
    test_ipos, test_ineg, test_lpos, test_lneg = build_incidence_pair(test_pos, test_neg)

    extra_feature = get_or_compute_drug_embeddings(drug_id_list, drugbank_to_smiles, subset_dir, device)

    return {
        'train_inc_pos': torch.tensor(train_ipos, dtype=torch.float32).to(device),
        'train_inc_all': torch.tensor(np.hstack([train_ipos, train_ineg]), dtype=torch.float32),
        'y_train_all': np.concatenate([train_lpos, train_lneg]),
        
        'val_inc_all': torch.tensor(np.hstack([val_ipos, val_ineg]), dtype=torch.float32).to(device),
        'y_val_all': np.concatenate([val_lpos, val_lneg]),
        
        'test_inc_all': torch.tensor(np.hstack([test_ipos, test_ineg]), dtype=torch.float32).to(device),
        'y_test_all': np.concatenate([test_lpos, test_lneg]),
        
        'drug_features': extra_feature,
    }

# ==========================================
# 2. MODEL DEFINITIONS
# ==========================================

class HGNN_SA(nn.Module):
    def __init__(self, input_feature_num, input_num, emb_dim=128, conv_dim=64, heads=3, p=0.1, in_channel=256):
        super().__init__()
        self.p = p
        self.pre_linear = nn.Linear(768, emb_dim)
        self.linear_encoder = nn.Linear(input_feature_num, emb_dim)
        self.hyper_attr_liner = nn.Linear(input_num, 2 * emb_dim)
        self.hypergraph_conv = hnn.HypergraphConv(2 * emb_dim, conv_dim, heads=heads, use_attention=True, dropout=p)
        self.hyperedge_linear = nn.Linear(conv_dim * heads, 2)
        
    def forward(self, input_features, incidence_matrix, extra_feature):
        incidence_matrix_T = incidence_matrix.T
        input_nodes_features = F.relu(self.linear_encoder(input_features))
        row, col = torch.where(incidence_matrix_T)
        edges = torch.cat((col.view(1, -1), row.view(1, -1)), dim=0)
        
        hyperedge_attr = self.hyper_attr_liner(incidence_matrix_T)
        extra_feat = F.relu(self.pre_linear(extra_feature))
        input_nodes_features = torch.cat((extra_feat, input_nodes_features), dim=1)
        
        input_nodes_features = self.hypergraph_conv(input_nodes_features, edges, hyperedge_attr=hyperedge_attr)
        hyperedge_feature = torch.mm(incidence_matrix_T, input_nodes_features)
        return self.hyperedge_linear(hyperedge_feature)
        
    def predict(self, input_features, incidence_matrix, extra_feature):
        logits = self.forward(input_features, incidence_matrix, extra_feature)
        return F.softmax(logits, dim=1)

class HyperAttDDI_NoSE(nn.Module):
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

# ==========================================
# 3. TRAINING & EVALUATION ROUTINES
# ==========================================

def train_exp_a(seed, data, device):
    set_random_seed(seed)
    model = HGNN_SA(
        input_feature_num=data['train_inc_pos'].shape[1], 
        input_num=data['train_inc_pos'].shape[0], 
        emb_dim=128, conv_dim=64, heads=3, p=0.1
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_f1, best_state, best_thresh = 0, None, 0.5
    for epoch in tqdm(range(100), desc=f"Exp A (Seed {seed})"):
        model.train()
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(data['train_inc_all'].shape[1], generator=generator)
        train_inc = data['train_inc_all'][:, perm]
        y_train = data['y_train_all'][perm]
        
        for i in range(0, train_inc.shape[1], 64):
            batch_inc = train_inc[:, i:i+64].to(device)
            batch_y = torch.tensor(y_train[i:i+64], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            y_pred = model(data['train_inc_pos'], batch_inc, data['drug_features'])
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        with torch.no_grad():
            val_probs = model.predict(data['train_inc_pos'], data['val_inc_all'], data['drug_features'])
            val_scores = val_probs[:, 1].cpu().numpy()
            
        best_epoch_thresh = 0.5
        best_epoch_f1 = 0.0
        for thresh in np.arange(0.1, 0.9, 0.02):
            b_score = (val_scores >= thresh).astype(int)
            f1 = f1_score(data['y_val_all'], b_score, zero_division=0)
            if f1 > best_epoch_f1:
                best_epoch_f1, best_epoch_thresh = f1, thresh
                
        if best_epoch_f1 > best_f1:
            best_f1, best_thresh, best_state = best_epoch_f1, best_epoch_thresh, copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_probs = model.predict(data['train_inc_pos'], data['test_inc_all'], data['drug_features'])
        test_scores = test_probs[:, 1].cpu().numpy()
    
    b_preds = (test_scores >= best_thresh).astype(int)
    return {
        'f1': f1_score(data['y_test_all'], b_preds, zero_division=0),
        'auc': roc_auc_score(data['y_test_all'], test_scores),
        'prauc': average_precision_score(data['y_test_all'], test_scores),
        'pr': precision_score(data['y_test_all'], b_preds, zero_division=0),
        're': recall_score(data['y_test_all'], b_preds, zero_division=0)
    }

def train_exp_b(seed, data, device):
    set_random_seed(seed)
    model = HyperAttDDI_NoSE(struct_dim=data['train_inc_pos'].shape[1], in_dim=768, emb_dim=256, conv_dim=64, heads=4, d=64, p=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.001)
    criterion = FocalLoss(gamma=2.0)
    
    best_f1, best_state, best_thresh = 0, None, 0.5
    for epoch in tqdm(range(100), desc=f"Exp B (Seed {seed})"):
        model.train()
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(data['train_inc_all'].shape[1], generator=generator)
        train_inc = data['train_inc_all'][:, perm]
        y_train = data['y_train_all'][perm]
        
        for i in range(0, train_inc.shape[1], 64):
            batch_inc = train_inc[:, i:i+64].to(device)
            batch_y = torch.tensor(y_train[i:i+64], dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            logits = model(data['drug_features'], data['train_inc_pos'], batch_inc)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        all_scores = []
        with torch.no_grad():
            for i in range(0, data['val_inc_all'].shape[1], 512):
                batch_inc = data['val_inc_all'][:, i:i+512].to(device)
                logits = model(data['drug_features'], data['train_inc_pos'], batch_inc)
                all_scores.append(torch.sigmoid(logits).cpu().numpy())
        val_scores = np.concatenate(all_scores, axis=0)
        
        best_epoch_thresh = 0.5
        best_epoch_f1 = 0.0
        for thresh in np.arange(0.1, 0.9, 0.02):
            b_score = (val_scores >= thresh).astype(int)
            f1 = f1_score(data['y_val_all'], b_score, zero_division=0)
            if f1 > best_epoch_f1:
                best_epoch_f1, best_epoch_thresh = f1, thresh
                
        if best_epoch_f1 > best_f1:
            best_f1, best_thresh, best_state = best_epoch_f1, best_epoch_thresh, copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_state)
    model.eval()
    all_scores = []
    with torch.no_grad():
        for i in range(0, data['test_inc_all'].shape[1], 512):
            batch_inc = data['test_inc_all'][:, i:i+512].to(device)
            logits = model(data['drug_features'], data['train_inc_pos'], batch_inc)
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
    test_scores = np.concatenate(all_scores, axis=0)
    
    b_preds = (test_scores >= best_thresh).astype(int)
    return {
        'f1': f1_score(data['y_test_all'], b_preds, zero_division=0),
        'auc': roc_auc_score(data['y_test_all'], test_scores),
        'prauc': average_precision_score(data['y_test_all'], test_scores),
        'pr': precision_score(data['y_test_all'], b_preds, zero_division=0),
        're': recall_score(data['y_test_all'], b_preds, zero_division=0)
    }

# ==========================================
# 4. MASTER EXECUTION & STATISTICS
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print("  HYPERATT DDI MASTER EVALUATION SCRIPT (EXP A vs EXP B) ")
    print("="*60)
    
    data = load_data(device)
    seeds = [42, 3407, 54321, 123456, 7, 777, 31415]
    
    # Store results across seeds
    results_A = {m: [] for m in ['f1', 'auc', 'prauc', 'pr', 're']}
    results_B = {m: [] for m in ['f1', 'auc', 'prauc', 'pr', 're']}
    
    for seed in seeds:
        print(f"\n[--- PROCESSING SEED {seed} ---]")
        
        # Train Exp A
        res_A = train_exp_a(seed, data, device)
        for k in res_A: results_A[k].append(res_A[k])
        print(f"Exp A -> AUC: {res_A['auc']:.4f} | PR-AUC: {res_A['prauc']:.4f}")
        
        # Train Exp B
        res_B = train_exp_b(seed, data, device)
        for k in res_B: results_B[k].append(res_B[k])
        print(f"Exp B -> AUC: {res_B['auc']:.4f} | PR-AUC: {res_B['prauc']:.4f}")
        
    print("\n======================================================")
    print("   FINAL STATISTICAL SIGNIFICANCE REPORT (7 SEEDS)    ")
    print("======================================================")
    
    metric_names = {'auc': 'ROC-AUC', 'prauc': 'PR-AUC', 'f1': 'F1-Score', 'pr': 'Precision', 're': 'Recall'}
    
    for key, display_name in metric_names.items():
        arr_A = np.array(results_A[key])
        arr_B = np.array(results_B[key])
        
        t_stat, p_val = stats.ttest_rel(arr_B, arr_A)
        mean_diff = np.mean(arr_B - arr_A)
        
        sig_marker = "✅ SIGNIFICANT" if p_val < 0.05 else "❌ NOT SIGNIFICANT"
        
        print(f"\n[{display_name}]")
        print(f"  Exp A Mean : {np.mean(arr_A):.4f} ± {np.std(arr_A):.4f}")
        print(f"  Exp B Mean : {np.mean(arr_B):.4f} ± {np.std(arr_B):.4f}")
        print(f"  Jump       : +{mean_diff*100:.2f}%")
        print(f"  P-Value    : {p_val:.5f} ({sig_marker})")
