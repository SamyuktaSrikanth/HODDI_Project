import os, sys, subprocess, ast, math, copy, random, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_geometric.nn as hnn
    print("Using PyTorch Geometric (hnn.HypergraphConv).")
except ImportError:
    print("torch-geometric not installed (offline mode). Using native pure-PyTorch HypergraphConv.")
    class HypergraphConv(nn.Module):
        """Pure PyTorch implementation of HypergraphConv with attention pooling."""
        def __init__(self, in_channels, out_channels, heads=1, use_attention=True, dropout=0.0, negative_slope=0.2, bias=True, **kwargs):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.heads = heads
            self.use_attention = use_attention
            self.dropout = dropout
            self.negative_slope = negative_slope
            
            self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
            if use_attention:
                self.att = nn.Parameter(torch.empty(1, heads, 2 * out_channels))
                nn.init.xavier_uniform_(self.att)
            if bias:
                self.bias = nn.Parameter(torch.zeros(heads * out_channels))
            else:
                self.register_parameter('bias', None)
            nn.init.xavier_uniform_(self.lin.weight)
            
        def forward(self, x, hyperedge_index, hyperedge_weight=None, hyperedge_attr=None, num_edges=None):
            num_nodes = x.size(0)
            if num_edges is None:
                num_edges = int(hyperedge_index[1].max()) + 1 if hyperedge_index.numel() > 0 else 0
            node_idx = hyperedge_index[0]
            edge_idx = hyperedge_index[1]
            
            x_proj = self.lin(x).view(-1, self.heads, self.out_channels)
            
            if self.use_attention and hyperedge_attr is not None:
                edge_proj = self.lin(hyperedge_attr).view(-1, self.heads, self.out_channels)
                x_i = x_proj[node_idx]
                x_j = edge_proj[edge_idx]
                
                alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
                alpha = F.leaky_relu(alpha, self.negative_slope)
                
                alpha_exp = torch.exp(alpha - alpha.max(dim=0, keepdim=True).values)
                sum_exp = torch.zeros(num_edges, self.heads, device=x.device).index_add_(0, edge_idx, alpha_exp)
                alpha = alpha_exp / (sum_exp[edge_idx] + 1e-16)
                alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            else:
                alpha = None

            deg_v = torch.zeros(num_nodes, device=x.device).index_add_(0, node_idx, torch.ones_like(node_idx, dtype=torch.float32))
            inv_deg_v = 1.0 / deg_v.clamp(min=1.0)
            
            deg_e = torch.zeros(num_edges, device=x.device).index_add_(0, edge_idx, torch.ones_like(edge_idx, dtype=torch.float32))
            inv_deg_e = 1.0 / deg_e.clamp(min=1.0)
            
            msg_node = x_proj[node_idx] * inv_deg_e[edge_idx].view(-1, 1, 1)
            if alpha is not None:
                msg_node = msg_node * alpha.unsqueeze(-1)
            
            h_e = torch.zeros(num_edges, self.heads, self.out_channels, device=x.device)
            h_e.index_add_(0, edge_idx, msg_node)
            
            msg_edge = h_e[edge_idx] * inv_deg_v[node_idx].view(-1, 1, 1)
            if alpha is not None:
                msg_edge = msg_edge * alpha.unsqueeze(-1)
                
            out = torch.zeros(num_nodes, self.heads, self.out_channels, device=x.device)
            out.index_add_(0, node_idx, msg_edge)
            
            out = out.view(-1, self.heads * self.out_channels)
            if self.bias is not None:
                out = out + self.bias
            return out

    class HNNModule:
        pass
    hnn = HNNModule()
    hnn.HypergraphConv = HypergraphConv

from tqdm import tqdm
from scipy import stats
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    precision_score, recall_score
)

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
    if not subset_dir:
        raise FileNotFoundError("Could not find directory 'subset_drug2-8_SE5-50'")

    dict_path = find_file('Drugbank_ID_SMILE_all_structure links.csv')
    if not dict_path:
        raise FileNotFoundError("Could not find 'Drugbank_ID_SMILE_all_structure links.csv'")
    smiles_ds = pd.read_csv(dict_path)
    drugbank_to_smiles = smiles_ds.set_index('DrugBank ID')['SMILES'].to_dict()

    merged_dir = os.path.join(subset_dir, 'merged_subset')
    all_ds = pd.concat([
        pd.read_csv(os.path.join(merged_dir, 'positive_samples_2014Q3_2024Q3_step6.csv')), 
        pd.read_csv(os.path.join(merged_dir, 'negative_samples_2014Q3_2024Q3_step6.csv'))
    ], axis=0, ignore_index=True)
    
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
    print(f"Total valid unique drugs: {num_drugs}")

    # Load Side Effect SapBERT 768d embeddings
    se_file = find_file('Side_effects_unique.csv')
    if not se_file:
        raise FileNotFoundError("Could not find 'Side_effects_unique.csv'")
    print(f"Loading Side-Effect SapBERT embeddings from: {se_file}")
    se_df = pd.read_csv(se_file)
    feature_cols = [str(i) for i in range(768)]
    se_embeddings_dict = {}
    for _, row in se_df.iterrows():
        cui = row['umls_cui_from_meddra']
        se_embeddings_dict[cui] = row[feature_cols].values.astype(np.float32)

    def build_incidence_pair(pos_df, neg_df):
        num_pos = len(pos_df)
        inc_pos = np.zeros((num_drugs, num_pos), dtype=np.float32)
        se_pos = np.zeros((num_pos, 768), dtype=np.float32)
        pos_cuis = pos_df['SE_above_0.9'].values
        pos_drug_lists = pos_df['DrugBankID'].values

        for col_idx in range(num_pos):
            cui = pos_cuis[col_idx]
            if cui in se_embeddings_dict:
                se_pos[col_idx] = se_embeddings_dict[cui]
            for did in ast.literal_eval(pos_drug_lists[col_idx]):
                if did in drug_to_index:
                    inc_pos[drug_to_index[did], col_idx] = 1.0

        num_neg = len(neg_df)
        inc_neg = np.zeros((num_drugs, num_neg), dtype=np.float32)
        se_neg = np.zeros((num_neg, 768), dtype=np.float32)
        neg_cuis = neg_df['SE_above_0.9'].values
        neg_drug_lists = neg_df['DrugBankID'].values

        for col_idx in range(num_neg):
            cui = neg_cuis[col_idx]
            if cui in se_embeddings_dict:
                se_neg[col_idx] = se_embeddings_dict[cui]
            for did in ast.literal_eval(neg_drug_lists[col_idx]):
                if did in drug_to_index:
                    inc_neg[drug_to_index[did], col_idx] = 1.0

        lpos = np.ones(num_pos, dtype=np.int64)
        lneg = np.zeros(num_neg, dtype=np.int64)
        return inc_pos, inc_neg, se_pos, se_neg, lpos, lneg

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
        return pd.concat(pos_merged, axis=0, ignore_index=True), pd.concat(neg_merged, axis=0, ignore_index=True)

    print("Building quarterly incidence matrices and side-effect vectors...")
    train_pos, train_neg = merge_quarters(training_sub_ds_names)
    train_ipos, train_ineg, train_se_pos, train_se_neg, train_lpos, train_lneg = build_incidence_pair(train_pos, train_neg)

    val_pos, val_neg = merge_quarters(validating_sub_ds_names)
    val_ipos, val_ineg, val_se_pos, val_se_neg, val_lpos, val_lneg = build_incidence_pair(val_pos, val_neg)

    test_pos, test_neg = merge_quarters(testing_sub_ds_names)
    test_ipos, test_ineg, test_se_pos, test_se_neg, test_lpos, test_lneg = build_incidence_pair(test_pos, test_neg)

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
        'train_se_all': torch.tensor(np.concatenate([train_se_pos, train_se_neg], axis=0), dtype=torch.float32),
        'val_se_all': torch.tensor(np.concatenate([val_se_pos, val_se_neg], axis=0), dtype=torch.float32).to(device),
        'test_se_all': torch.tensor(np.concatenate([test_se_pos, test_se_neg], axis=0), dtype=torch.float32).to(device),
    }

# ==========================================
# 2. MODEL DEFINITIONS
# ==========================================

class HyperAttDDI_NoSE(nn.Module):
    """Exp B: Baseline without side-effect features. Pure drug structure + hypergraph architecture."""
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

class HyperAttDDI_SE(nn.Module):
    """Exp C-fixed: SE-Conditioned Attention WITH structural features restored (fair ablation vs Exp B)."""
    def __init__(self, struct_dim, in_dim=768, emb_dim=256, conv_dim=64, heads=4, d=64, p=0.1):
        super().__init__()
        self.d = d
        self.p = p
        
        # Module 1: Drug Encoder — dual-branch (same as Exp B)
        self.struct_encoder = nn.Linear(struct_dim, 128)
        self.drug_encoder = nn.Linear(in_dim, 128)
        self.input_dropout = nn.Dropout(p=p)
        
        # Module 2: Hypergraph Encoder (2 layers, 64 * 4 = 256)
        self.conv1 = hnn.HypergraphConv(emb_dim, conv_dim, heads=heads, use_attention=True, dropout=p)
        self.conv2 = hnn.HypergraphConv(conv_dim * heads, conv_dim, heads=heads, use_attention=True, dropout=p)
        
        # Module 4: Side-Effect Encoder (SapBERT projection 768 -> 256)
        self.se_encoder = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(p=p)
        )
        
        # Module 3: SE-Conditioned Attention Pooling
        # KEY DIFFERENCE: Query is conditioned on BOTH the combination representation and adverse event [h_e_mean || h_se] (512 -> d)
        self.W_q = nn.Linear(emb_dim * 2, d)
        self.W_k = nn.Linear(emb_dim, d)
        self.W_v = nn.Linear(emb_dim, emb_dim)
        
        # Module 5: Interaction Decoder takes [h_combo || h_se] (512 -> 1)
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(128, 1)
        )
        
    def forward(self, drug_features, struct_features, inc_matrix, se_features):
        N, E = inc_matrix.shape
        H_T = inc_matrix.T  # (E, N)
        
        # 1. Drug Encoder — dual-branch (restored from Exp B)
        s_emb = F.relu(self.struct_encoder(struct_features))
        d_emb = F.relu(self.drug_encoder(drug_features))
        X = torch.cat([s_emb, d_emb], dim=1)  # (N, 128+128=256)
        X = self.input_dropout(X)
        
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
        
        # 4. Side Effect Encoder
        h_se = self.se_encoder(se_features)  # (E, 256)
        
        # 3. SE-Conditioned Attention Pooling
        h_e_mean = (H_T @ X2) / degree_e  # (E, 256)
        q_input = torch.cat([h_e_mean, h_se], dim=1)  # (E, 512)
        q = self.W_q(q_input)            # (E, d)
        k = self.W_k(X2)                  # (N, d)
        v = self.W_v(X2)                  # (N, 256)
        
        scores = torch.matmul(q, k.T) / math.sqrt(self.d)  # (E, N)
        scores = scores.masked_fill(H_T == 0, -1e9)
        alpha = F.softmax(scores, dim=1)  # (E, N)
        h_combo = torch.matmul(alpha, v)  # (E, 256)
        
        # 5. Decoder
        z = torch.cat([h_combo, h_se], dim=1)  # (E, 512)
        logits = self.decoder(z).squeeze(-1)    # (E,)
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

def train_exp_b(seed, data, device):
    set_random_seed(seed)
    model = HyperAttDDI_NoSE(
        struct_dim=data['train_inc_pos'].shape[1], 
        in_dim=768, emb_dim=256, conv_dim=64, heads=4, d=64, p=0.1
    ).to(device)
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

def train_exp_c(seed, data, device):
    set_random_seed(seed)
    model = HyperAttDDI_SE(
        struct_dim=data['train_inc_pos'].shape[1],
        in_dim=768, emb_dim=256, conv_dim=64, heads=4, d=64, p=0.1
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.001)
    criterion = FocalLoss(gamma=2.0)
    
    num_samples = data['train_inc_all'].shape[1]
    best_f1, best_state, best_thresh = 0, None, 0.5
    for epoch in tqdm(range(100), desc=f"Exp C (Seed {seed})"):
        model.train()
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(num_samples, generator=generator)
        train_inc = data['train_inc_all'][:, perm]
        train_se = data['train_se_all'][perm]
        y_train = data['y_train_all'][perm]
        
        for i in range(0, num_samples, 64):
            batch_inc = train_inc[:, i:i+64].to(device)
            batch_se = train_se[i:i+64].to(device)
            batch_y = torch.tensor(y_train[i:i+64], dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            logits = model(data['drug_features'], data['train_inc_pos'], batch_inc, batch_se)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        all_scores = []
        with torch.no_grad():
            for i in range(0, data['val_inc_all'].shape[1], 512):
                batch_inc = data['val_inc_all'][:, i:i+512].to(device)
                batch_se = data['val_se_all'][i:i+512].to(device)
                logits = model(data['drug_features'], data['train_inc_pos'], batch_inc, batch_se)
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
            batch_se = data['test_se_all'][i:i+512].to(device)
            logits = model(data['drug_features'], data['train_inc_pos'], batch_inc, batch_se)
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
    print("=" * 60)
    print("  HYPERATT DDI MASTER EVALUATION SCRIPT (EXP B vs EXP C) ")
    print("=" * 60)
    
    data = load_data(device)
    seeds = [42, 3407, 54321, 123456, 7, 777, 31415]
    
    # Store results across seeds
    results_B = {m: [] for m in ['f1', 'auc', 'prauc', 'pr', 're']}
    results_C = {m: [] for m in ['f1', 'auc', 'prauc', 'pr', 're']}
    
    for seed in seeds:
        print(f"\n[--- PROCESSING SEED {seed} ---]")
        
        # Train Exp B
        res_B = train_exp_b(seed, data, device)
        for k in res_B: results_B[k].append(res_B[k])
        print(f"Exp B -> AUC: {res_B['auc']:.4f} | PR-AUC: {res_B['prauc']:.4f} | F1: {res_B['f1']:.4f}")
        
        # Train Exp C
        res_C = train_exp_c(seed, data, device)
        for k in res_C: results_C[k].append(res_C[k])
        print(f"Exp C -> AUC: {res_C['auc']:.4f} | PR-AUC: {res_C['prauc']:.4f} | F1: {res_C['f1']:.4f}")
        
    print("\n======================================================")
    print("   FINAL STATISTICAL SIGNIFICANCE REPORT (7 SEEDS)    ")
    print("======================================================")
    
    metric_names = {'auc': 'ROC-AUC', 'prauc': 'PR-AUC', 'f1': 'F1-Score', 'pr': 'Precision', 're': 'Recall'}
    
    for key, display_name in metric_names.items():
        arr_B = np.array(results_B[key])
        arr_C = np.array(results_C[key])
        
        t_stat, p_val = stats.ttest_rel(arr_C, arr_B)
        mean_diff = np.mean(arr_C - arr_B)
        
        sig_marker = "✅ SIGNIFICANT" if p_val < 0.05 else "❌ NOT SIGNIFICANT"
        
        print(f"\n[{display_name}]")
        print(f"  Exp B Mean : {np.mean(arr_B):.4f} ± {np.std(arr_B):.4f}")
        print(f"  Exp C Mean : {np.mean(arr_C):.4f} ± {np.std(arr_C):.4f}")
        print(f"  Jump       : {mean_diff*100:+.2f}%")
        print(f"  P-Value    : {p_val:.5f} ({sig_marker})")
