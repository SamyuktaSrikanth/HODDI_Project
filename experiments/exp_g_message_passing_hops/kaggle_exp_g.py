import os, ast, math, copy, random, time, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    - Drug Encoder: struct_encoder (Linear struct_dim -> 128) + drug_encoder (Linear 768 -> 128) => (N, 256)
    - Hypergraph Encoder: L layers of PyG HypergraphConv(256, 64, heads=4, use_attention=True)
    - Attention Pooling: Learned attention over member drugs without SE conditioning:
        q = W_q(h_e_mean), k = W_k(X_L), v = W_v(X_L)
        alpha = softmax(q @ k.T / sqrt(d)) (masked by hyperedge membership)
        h_e = alpha @ v
    - Decoder: Linear(256 -> 128) -> ReLU -> Dropout -> Linear(128 -> 1)
    """
    def __init__(self, struct_dim, in_dim=768, emb_dim=256, conv_dim=64, heads=4, num_layers=2, d=64, p=0.1):
        super().__init__()
        self.d = d
        self.p = p
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        
        # 1. Drug Encoder (Text + Structure) - 100% matched to Exp B
        self.struct_encoder = nn.Linear(struct_dim, 128)
        self.drug_encoder = nn.Linear(in_dim, 128)
        self.input_dropout = nn.Dropout(p=p)
        
        # 2. Hypergraph Message Passing Layers (L layers)
        self.convs = nn.ModuleList()
        if num_layers > 0:
            self.convs.append(hnn.HypergraphConv(emb_dim, conv_dim, heads=heads, use_attention=True, dropout=p))
            for _ in range(1, num_layers):
                self.convs.append(hnn.HypergraphConv(conv_dim * heads, conv_dim, heads=heads, use_attention=True, dropout=p))
                
        # 3. Attention Pooling (No SE) - 100% matched to Exp B
        self.W_q = nn.Linear(emb_dim, d)
        self.W_k = nn.Linear(emb_dim, d)
        self.W_v = nn.Linear(emb_dim, emb_dim)
        
        # 4. Interaction Decoder - 100% matched to Exp B
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
        
        degree_e = H_T.sum(dim=1, keepdim=True).clamp(min=1)
        
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
            
        h_e_mean = (H_T @ X_final) / degree_e
        q = self.W_q(h_e_mean)
        k = self.W_k(X_final)
        v = self.W_v(X_final)
        
        scores = torch.matmul(q, k.T) / math.sqrt(self.d)
        scores = scores.masked_fill(H_T == 0, -1e9)
        alpha = F.softmax(scores, dim=1)
        h_e = torch.matmul(alpha, v)
        
        logits = self.decoder(h_e).squeeze(-1)
        return logits

def find_file(filename, search_roots=['/kaggle/input', '/kaggle/working', '.', '..', '../..', 'dataset']):
    norm_fn = filename.lower().replace(' ', '_').replace('-', '_')
    for root_dir in search_roots:
        if os.path.exists(root_dir):
            for root, dirs, files in os.walk(root_dir):
                for f in files:
                    if f == filename or f.lower() == filename.lower():
                        return os.path.join(root, f)
                    f_norm = f.lower().replace(' ', '_').replace('-', '_')
                    if f_norm == norm_fn:
                        return os.path.join(root, f)
                    if 'drugbank' in f.lower() and 'smile' in f.lower() and f.endswith('.csv'):
                        return os.path.join(root, f)
    return None

def find_dir(dirname, search_roots=['/kaggle/input', '.', '..', '../..', 'dataset']):
    norm_dn = dirname.lower().replace(' ', '_').replace('-', '_')
    for root_dir in search_roots:
        if os.path.exists(root_dir):
            for root, dirs, files in os.walk(root_dir):
                for d in dirs:
                    if d == dirname or d.lower() == dirname.lower():
                        return os.path.join(root, d)
                    d_norm = d.lower().replace(' ', '_').replace('-', '_')
                    if d_norm == norm_dn:
                        return os.path.join(root, d)
                    if 'subset_drug' in d.lower() or 'drug2-8' in d.lower():
                        return os.path.join(root, d)
    return None

def list_kaggle_inputs():
    all_found = []
    if os.path.exists('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            for f in files:
                all_found.append(os.path.join(root, f))
    return all_found

def load_data(device):
    subset_dir = find_dir('subset_drug2-8_SE5-50')
    if subset_dir is None:
        inputs = list_kaggle_inputs()
        raise FileNotFoundError(
            f"Could not find dataset directory 'subset_drug2-8_SE5-50'.\n"
            f"Please ensure you added the HODDI dataset to your Kaggle Notebook under '+ Add Input'.\n"
            f"Files currently in /kaggle/input:\n" + "\n".join(inputs[:20])
        )
        
    dict_path = find_file('Drugbank_ID_SMILE_all_structure links.csv')
    if dict_path is None:
        inputs = list_kaggle_inputs()
        raise FileNotFoundError(
            f"Could not find SMILES links dictionary 'Drugbank_ID_SMILE_all_structure links.csv'.\n"
            f"Please ensure the SMILES dataset is added to your Kaggle Notebook inputs.\n"
            f"Files currently in /kaggle/input:\n" + "\n".join(inputs[:20])
        )
        
    print(f"Found SMILES dictionary: {dict_path}")
    print(f"Found subset directory: {subset_dir}")
    smiles_ds = pd.read_csv(dict_path)
    drugbank_to_smiles = smiles_ds.set_index('DrugBank ID')['SMILES'].to_dict()

    def merge_quarters(sub_datasets):
        pos_merged, neg_merged = [], []
        for sub_ds in sub_datasets:
            pos_name = f'{sub_ds}_positive_samples_condition123_SE_above_0.9.csv'
            neg_name = f'{sub_ds}_negative_samples_condition123_SE_above_0.9.csv'
            
            # Check subset_dir or global find_file
            pos_p = os.path.join(subset_dir, pos_name)
            if not os.path.exists(pos_p): pos_p = find_file(pos_name)
            neg_p = os.path.join(subset_dir, neg_name)
            if not os.path.exists(neg_p): neg_p = find_file(neg_name)
            
            if pos_p and os.path.exists(pos_p): pos_merged.append(pd.read_csv(pos_p))
            if neg_p and os.path.exists(neg_p): neg_merged.append(pd.read_csv(neg_p))
            
        pos_df = pd.concat(pos_merged, axis=0) if len(pos_merged) > 0 else pd.DataFrame()
        neg_df = pd.concat(neg_merged, axis=0) if len(neg_merged) > 0 else pd.DataFrame()
        return pos_df, neg_df

    train_quarters = ['2015Q1', '2015Q2', '2015Q3', '2016Q4', '2017Q1', '2017Q2', '2017Q3', '2017Q4', '2018Q3', '2019Q1', '2019Q2', '2019Q3', '2019Q4', '2020Q1', '2020Q2', '2020Q3', '2020Q4', '2021Q1', '2021Q3', '2021Q4', '2022Q1', '2022Q2', '2022Q3', '2022Q4', '2023Q1', '2023Q2', '2023Q3', '2023Q4', '2024Q2']
    val_quarters = ['2014Q3', '2015Q4', '2016Q1', '2016Q3', '2021Q2', '2024Q1']
    test_quarters = ['2014Q4', '2016Q2', '2018Q1', '2018Q2', '2018Q4', '2024Q3']

    print("Loading quarterly CSV files...")
    train_pos, train_neg = merge_quarters(train_quarters)
    val_pos, val_neg = merge_quarters(val_quarters)
    test_pos, test_neg = merge_quarters(test_quarters)

    # Build drug universe from all available quarter splits
    all_sample_dfs = [df for df in [train_pos, train_neg, val_pos, val_neg, test_pos, test_neg] if not df.empty]
    all_ds = pd.concat(all_sample_dfs, axis=0)

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
    print(f"Total drugs after SMILES filtering: {num_drugs}")

    emb_path = find_file('drug_embeddings_768d.pt')
    cache_path = '/kaggle/working/drug_embeddings_768d_author_padded_10250.pt'
    
    if os.path.exists(cache_path):
        print(f"Loading cached author-padded embeddings from: {cache_path}")
        extra_feature = torch.load(cache_path, map_location=device)
    elif emb_path and os.path.exists(emb_path):
        print(f"Loading pre-computed drug embeddings from: {emb_path}")
        raw_emb = torch.load(emb_path, map_location='cpu')
        if isinstance(raw_emb, dict):
            extra_feature = torch.zeros((num_drugs, 768), dtype=torch.float32)
            for idx, did in enumerate(drug_id_list):
                if did in raw_emb:
                    extra_feature[idx] = raw_emb[did] if torch.is_tensor(raw_emb[did]) else torch.tensor(raw_emb[did])
            extra_feature = extra_feature.to(device)
        elif torch.is_tensor(raw_emb) and raw_emb.shape[0] == num_drugs:
            extra_feature = raw_emb.to(device)
        else:
            extra_feature = None
    else:
        extra_feature = None

    if extra_feature is None:
        print("Computing author-identical ChemBERTa embeddings on GPU (takes ~45s)...")
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        m_name = 'seyonec/PubChem10M_SMILES_BPE_450k'
        local_dir = find_dir('PubChem10M_SMILES_BPE_450k')
        if local_dir:
            m_name = local_dir
            
        try:
            tokenizer = AutoTokenizer.from_pretrained(m_name, use_fast=True)
        except Exception:
            try:
                tokenizer = AutoTokenizer.from_pretrained(m_name, use_fast=False)
            except Exception:
                from transformers import RobertaTokenizerFast
                tokenizer = RobertaTokenizerFast.from_pretrained(m_name)
                
        chem_model = AutoModelForMaskedLM.from_pretrained(m_name).to(device)
        chem_model.eval()

        smiles_list = [str(drugbank_to_smiles.get(did, '')) for did in drug_id_list]
        chemberta_feat = []
        batch_size = 64
        with torch.no_grad():
            for i in tqdm(range(0, len(smiles_list), batch_size), desc="ChemBERTa Embeddings"):
                batch_smiles = smiles_list[i:i+batch_size]
                tokens = tokenizer(batch_smiles, return_tensors="pt", max_length=256, padding='max_length', truncation=True).to(device)
                outputs = chem_model(**tokens, output_hidden_states=True)
                chemberta_feat.append(outputs.hidden_states[-1].mean(dim=1))

        extra_feature = torch.cat(chemberta_feat, dim=0)
        try:
            torch.save(extra_feature, cache_path)
        except Exception:
            pass

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
    return (
        f1_score(labels, preds, zero_division=0),
        roc_auc_score(labels, all_scores),
        average_precision_score(labels, all_scores),
        precision_score(labels, preds, zero_division=0),
        recall_score(labels, preds, zero_division=0),
        all_scores
    )

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
    parser = argparse.ArgumentParser(description="Exp G: Message Passing Hops on Exp B Architecture")
    parser.add_argument('--hops', nargs='+', type=int, default=[1, 2, 3, 4], help="Hop depths to evaluate")
    parser.add_argument('--epochs', type=int, default=100, help="Epochs per hop model")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.0005, help="Learning rate (matches Exp B)")
    parser.add_argument('--weight_decay', type=float, default=0.001, help="Weight decay (matches Exp B)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    args, _ = parser.parse_known_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data = load_data(device)
    struct_dim = data['train_inc_pos'].shape[1]
    
    results = []
    
    print("\n" + "="*70)
    print("STARTING EXPERIMENT G: MESSAGE PASSING HOP DEPTH ABLATION (EXP B)")
    print("Controlled Settings (100% Strict Parity with Exp B):")
    print(f"  - Optimizer: AdamW(lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"  - Loss: FocalLoss(gamma=2.0)")
    print(f"  - Batch Size: {args.batch_size} | Epochs: {args.epochs} | Seed: {args.seed}")
    print(f"  - Features: ChemBERTa (768d) + Positive Incidence Structure ({struct_dim}d)")
    print("="*70 + "\n")
    
    for h in args.hops:
        print(f"\n>>> [Hop Level L = {h}] Initializing {h}-Layer HyperAttDDI_NoSE Model <<<")
        set_random_seed(args.seed)
        
        model = HyperAttDDI_HopAblation(
            struct_dim=struct_dim,
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
        criterion = FocalLoss(gamma=2.0)
        
        best_f1, best_state, best_thresh = 0, None, 0.5
        epoch_times = []
        
        start_train_t = time.time()
        for epoch in tqdm(range(args.epochs), desc=f"Training L={h}"):
            t0 = time.time()
            model.train()
            
            generator = torch.Generator().manual_seed(args.seed + epoch)
            perm = torch.randperm(data['train_inc_all'].shape[1], generator=generator)
            train_inc = data['train_inc_all'][:, perm]
            y_train = data['y_train_all'][perm]
            
            for i in range(0, train_inc.shape[1], args.batch_size):
                batch_inc = train_inc[:, i:i+args.batch_size].to(device)
                batch_y = y_train[i:i+args.batch_size]
                
                optimizer.zero_grad()
                logits = model(data['drug_features'], data['train_inc_pos'], batch_inc)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
            epoch_times.append(time.time() - t0)
            
            # Validation threshold search
            _, _, _, _, _, val_scores = evaluate(
                model, data['drug_features'], data['train_inc_pos'],
                data['val_inc_all'], data['y_val_all'], device
            )
            
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
                
        total_train_time = time.time() - start_train_t
        avg_epoch_t = np.mean(epoch_times)
        
        # Test evaluation with best checkpoint
        if best_state is not None:
            model.load_state_dict(best_state)
            
        test_f1, test_auc, test_prauc, test_pr, test_re, _ = evaluate(
            model, data['drug_features'], data['train_inc_pos'],
            data['test_inc_all'], data['y_test_all'], device, threshold=best_thresh
        )
        
        val_f1, val_auc, val_prauc, _, _, _ = evaluate(
            model, data['drug_features'], data['train_inc_pos'],
            data['val_inc_all'], data['y_val_all'], device, threshold=best_thresh
        )
        
        print(f"\n--- Results for L={h} Hops ---")
        print(f"Test ROC-AUC:    {test_auc:.4f}")
        print(f"Test PR-AUC:     {test_prauc:.4f}")
        print(f"Test F1-Score:   {test_f1:.4f} (Optimal Thresh: {best_thresh:.2f})")
        print(f"Test Precision:  {test_pr:.4f}")
        print(f"Test Recall:     {test_re:.4f}")
        print(f"Val ROC-AUC:     {val_auc:.4f} | Val PR-AUC: {val_prauc:.4f}")
        print(f"Avg Epoch Time:  {avg_epoch_t:.2f}s (Total: {total_train_time:.1f}s)")
        
        results.append({
            'hops': h,
            'test_auc': round(test_auc, 4),
            'test_prauc': round(test_prauc, 4),
            'test_f1': round(test_f1, 4),
            'test_precision': round(test_pr, 4),
            'test_recall': round(test_re, 4),
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
    print("\nMetrics saved to 'hop_comparison_metrics.csv'")
    
    plot_hop_comparison(results_df, 'message_passing_hops_comparison.png')

if __name__ == '__main__':
    main()
