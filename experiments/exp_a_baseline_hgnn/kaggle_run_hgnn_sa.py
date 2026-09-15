import os
import ast
import copy
import random
import argparse
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

# ---------------------------------------------------------
# 1. Reproducibility & Initialization
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 2. HGNN-SA Model (CLOSEgaps with enable_hygnn=False, L=1)
# ---------------------------------------------------------
class HGNN_SA(nn.Module):
    """
    HGNN-SA Baseline Architecture faithfully matching HODDI demo4/CLOSEgaps.py.
    - Incidence pattern encoder: Linear(input_feature_num, emb_dim)
    - Chemical feature encoder: Linear(768, emb_dim)
    - Combined node features: concat -> 2 * emb_dim (256)
    - Hyperedge attribute generator: Linear(num_drugs, 2 * emb_dim)
    - 1-layer HypergraphConv with multi-head attention (heads=3)
    - Sum aggregation across hyperedges
    - Classification head: Linear(conv_dim * heads, 2) -> CrossEntropyLoss
    """
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
        self.hypergraph_conv = hnn.HypergraphConv(
            self.in_channel, conv_dim, heads=head, use_attention=True, dropout=p
        )
        self.hyper_attr_liner = nn.Linear(input_num, self.in_channel)
        self.hyperedge_linear = nn.Linear(conv_dim * head, 2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_features, incidence_matrix, extra_feature):
        incidence_matrix_T = incidence_matrix.T  # (batch_edges, num_drugs)
        
        # 1. Structural incidence encoding
        input_nodes_features = self.relu(self.linear_encoder(input_features))
        
        # 2. Extract edge connectivity for PyG HypergraphConv
        row, col = torch.where(incidence_matrix_T)
        edges = torch.cat((col.view(1, -1), row.view(1, -1)), dim=0).to(incidence_matrix.device)
        
        # 3. Learned hyperedge attributes from incidence pattern
        hyperedge_attr = self.hyper_attr_liner(incidence_matrix_T)
        
        # 4. Chemical feature encoding and concatenation
        extra_feat = self.relu(self.pre_linear(extra_feature))
        input_nodes_features = torch.cat((extra_feat, input_nodes_features), dim=1)
        
        # 5. Hypergraph convolution message passing
        input_nodes_features = self.hypergraph_conv(
            input_nodes_features, edges, hyperedge_attr=hyperedge_attr
        )
        
        # 6. Readout: sum pooling over hyperedges
        hyperedge_feature = torch.mm(incidence_matrix_T, input_nodes_features)
        
        # 7. Classification logits (2-class)
        return self.hyperedge_linear(hyperedge_feature)

    def predict(self, input_features, incidence_matrix, extra_feature):
        return self.softmax(self.forward(input_features, incidence_matrix, extra_feature))

# ---------------------------------------------------------
# 3. Path Detection & Data Loading
# ---------------------------------------------------------
def find_file(filename, search_roots=['/kaggle/input', '.', '..', '../..', 'dataset']):
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

def find_dataset_path():
    p = find_file('drug_embeddings_768d.pt')
    if p:
        return os.path.dirname(p)
    d = find_dir('subset_drug2-8_SE5-50')
    if d:
        return os.path.dirname(os.path.dirname(d))
    return 'dataset'

def load_smiles_dictionary(base_path=None):
    p = find_file('Drugbank_ID_SMILE_all_structure links.csv')
    if p and os.path.exists(p):
        print(f"Loading SMILES mapping from: {p}")
        smiles_ds = pd.read_csv(p)
        # CRITICAL FIX: Do NOT drop NaNs. HODDI original code keeps them.
        # This restores the 615 missing drugs and brings total to 10,250 matching the paper.
        return smiles_ds.set_index('DrugBank ID')['SMILES'].to_dict()
    raise FileNotFoundError("Could not find 'Drugbank_ID_SMILE_all_structure links.csv'")

def get_or_compute_drug_embeddings(drug_id_list, drugbank_to_smiles, base_path, device):
    """
    Loads pre-cached ChemBERTa embeddings (768d). If any drugs are missing from the cache,
    computes embeddings on-the-fly using ChemBERTa so that all drugs are strictly covered.
    """
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
        
        # Check for local weights folder first
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

def load_data(base_path, device):
    subset_dir = find_dir('subset_drug2-8_SE5-50')
    if subset_dir is None:
        raise FileNotFoundError("Could not locate 'subset_drug2-8_SE5-50' directory across search paths.")
    print(f"Using evaluation subset directory: {subset_dir}")

    # 1. Load SMILES Dictionary
    drugbank_to_smiles = load_smiles_dictionary(base_path)

    # 2. Load merged datasets to get full drug universe (matching demo4_HGNN/main.py lines 420-429)
    merged_dir = os.path.join(subset_dir, 'merged_subset')
    pos_step6 = os.path.join(merged_dir, 'positive_samples_2014Q3_2024Q3_step6.csv')
    neg_step6 = os.path.join(merged_dir, 'negative_samples_2014Q3_2024Q3_step6.csv')
    
    if not (os.path.exists(pos_step6) and os.path.exists(neg_step6)):
        raise FileNotFoundError(f"Missing merged step6 files at {merged_dir}")
        
    print(f"Reading merged dataset step6 files from {merged_dir}...")
    all_ds_pos = pd.read_csv(pos_step6)
    all_ds_neg = pd.read_csv(neg_step6)
    all_ds = pd.concat([all_ds_pos, all_ds_neg], axis=0)

    all_drugs = set()
    for drug_ids in all_ds['DrugBankID']:
        all_drugs.update([d for d in ast.literal_eval(drug_ids) if d.lower() != 'none'])

    drug_to_index_raw = {drug: idx for idx, drug in enumerate(all_drugs)}
    raw_drug_id_list = [None] * len(all_drugs)
    for did in drug_to_index_raw:
        raw_drug_id_list[drug_to_index_raw[did]] = did

    # Filter drugs by SMILES availability (matches demo4/main.py lines 478-480)
    drug_id_list = [did for did in raw_drug_id_list if did in drugbank_to_smiles]
    drug_to_index = {did: idx for idx, did in enumerate(drug_id_list)}
    num_drugs = len(drug_id_list)
    print(f"Total drugs after SMILES intersection (should be 10,250): {num_drugs}")

    # 3. Quarter splits
    training_sub_ds_names = [
        '2015Q1', '2015Q2', '2015Q3', '2016Q4', '2017Q1', '2017Q2', '2017Q3', '2017Q4',
        '2018Q3', '2019Q1', '2019Q2', '2019Q3', '2019Q4', '2020Q1', '2020Q2', '2020Q3',
        '2020Q4', '2021Q1', '2021Q3', '2021Q4', '2022Q1', '2022Q2', '2022Q3', '2022Q4',
        '2023Q1', '2023Q2', '2023Q3', '2023Q4', '2024Q2'
    ]
    validating_sub_ds_names = ['2014Q3', '2015Q4', '2016Q1', '2016Q3', '2021Q2', '2024Q1']
    testing_sub_ds_names = ['2014Q4', '2016Q2', '2018Q1', '2018Q2', '2018Q4', '2024Q3']

    def merge_quarters(sub_datasets):
        pos_merged = []
        neg_merged = []
        for sub_ds in sub_datasets:
            pos_f = os.path.join(subset_dir, f'{sub_ds}_positive_samples_condition123_SE_above_0.9.csv')
            neg_f = os.path.join(subset_dir, f'{sub_ds}_negative_samples_condition123_SE_above_0.9.csv')
            if os.path.exists(pos_f):
                pos_merged.append(pd.read_csv(pos_f))
            if os.path.exists(neg_f):
                neg_merged.append(pd.read_csv(neg_f))
        pos_df = pd.concat(pos_merged, axis=0) if len(pos_merged) > 0 else pd.DataFrame()
        neg_df = pd.concat(neg_merged, axis=0) if len(neg_merged) > 0 else pd.DataFrame()
        return pos_df, neg_df

    def build_incidence_pair(pos_df, neg_df):
        num_pos = len(pos_df)
        num_neg = len(neg_df)
        inc_pos = np.zeros((num_drugs, num_pos), dtype=np.float32)
        inc_neg = np.zeros((num_drugs, num_neg), dtype=np.float32)
        
        for col_idx, drug_list in enumerate(pos_df['DrugBankID']):
            for did in ast.literal_eval(drug_list):
                if did in drug_to_index:
                    inc_pos[drug_to_index[did], col_idx] = 1.0
                    
        for col_idx, drug_list in enumerate(neg_df['DrugBankID']):
            for did in ast.literal_eval(drug_list):
                if did in drug_to_index:
                    inc_neg[drug_to_index[did], col_idx] = 1.0
                    
        labels_pos = np.ones(num_pos, dtype=np.int64)
        labels_neg = np.zeros(num_neg, dtype=np.int64)
        return inc_pos, inc_neg, labels_pos, labels_neg

    print("Merging quarterly CSV files for train, val, and test splits...")
    train_pos_df, train_neg_df = merge_quarters(training_sub_ds_names)
    val_pos_df, val_neg_df = merge_quarters(validating_sub_ds_names)
    test_pos_df, test_neg_df = merge_quarters(testing_sub_ds_names)

    print("Building incidence matrices...")
    train_inc_pos, train_inc_neg, train_lab_pos, train_lab_neg = build_incidence_pair(train_pos_df, train_neg_df)
    val_inc_pos, val_inc_neg, val_lab_pos, val_lab_neg = build_incidence_pair(val_pos_df, val_neg_df)
    test_inc_pos, test_inc_neg, test_lab_pos, test_lab_neg = build_incidence_pair(test_pos_df, test_neg_df)

    # Convert to tensors
    train_inc_pos_t = torch.tensor(train_inc_pos, dtype=torch.float32).to(device)
    train_inc_all = torch.tensor(np.concatenate([train_inc_pos, train_inc_neg], axis=1), dtype=torch.float32)
    y_train_all = np.concatenate([train_lab_pos, train_lab_neg])

    val_inc_all = torch.tensor(np.concatenate([val_inc_pos, val_inc_neg], axis=1), dtype=torch.float32).to(device)
    y_val_all = np.concatenate([val_lab_pos, val_lab_neg])

    test_inc_all = torch.tensor(np.concatenate([test_inc_pos, test_inc_neg], axis=1), dtype=torch.float32).to(device)
    y_test_all = np.concatenate([test_lab_pos, test_lab_neg])

    # 4. Drug embeddings tensor
    extra_feature = get_or_compute_drug_embeddings(drug_id_list, drugbank_to_smiles, base_path, device)

    return {
        'train_inc_pos': train_inc_pos_t,
        'train_inc_all': train_inc_all,
        'y_train_all': y_train_all,
        'val_inc_all': val_inc_all,
        'y_val_all': y_val_all,
        'test_inc_all': test_inc_all,
        'y_test_all': y_test_all,
        'extra_feature': extra_feature,
        'num_drugs': num_drugs,
    }

# ---------------------------------------------------------
# 4. Training & Multi-Seed Evaluation Protocol
# ---------------------------------------------------------
def train_and_eval_seed(seed, data, args, device):
    set_random_seed(seed)
    print(f"\n==================================================")
    print(f"  Running HGNN-SA Training with Seed: {seed}")
    print(f"==================================================")

    train_inc_all = data['train_inc_all']
    y_train_all = data['y_train_all']
    num_train_samples = train_inc_all.shape[1]

    # Deterministic shuffle per seed (matching main.py line 527)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_train_samples, generator=g)
    train_inc_shuffled = train_inc_all[:, perm]
    y_train_shuffled = torch.tensor(y_train_all[perm.numpy()], dtype=torch.long)

    model = HGNN_SA(
        input_num=data['num_drugs'],
        input_feature_num=data['train_inc_pos'].shape[1],
        emb_dim=args.emb_dim,
        conv_dim=args.conv_dim,
        head=args.head,
        p=args.p
    ).to(device)
    
    if seed == args.seeds[0]:
        print(f"Total Trainable Parameters (Exp A): {count_parameters(model):,}")
    
    model.apply(init_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    best_weights = None
    batch_size = args.batch_size
    num_batches = num_train_samples // batch_size

    for epoch in range(1, args.epoch + 1):
        model.train()
        epoch_loss = 0.0
        
        for b in range(num_batches):
            optimizer.zero_grad()
            batch_inc = train_inc_shuffled[:, b * batch_size:(b + 1) * batch_size].to(device)
            batch_y = y_train_shuffled[b * batch_size:(b + 1) * batch_size].to(device)
            
            y_pred = model(data['train_inc_pos'], batch_inc, data['extra_feature'])
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation at each epoch with Threshold Tuning
        model.eval()
        with torch.no_grad():
            val_probs = model.predict(data['train_inc_pos'], data['val_inc_all'], data['extra_feature'])
            val_scores = val_probs[:, 1].cpu().numpy()
            
        best_epoch_thresh = 0.5
        best_epoch_f1 = 0.0
        for thresh in np.arange(0.1, 0.9, 0.02):
            b_score = (val_scores >= thresh).astype(int)
            f1 = f1_score(data['y_val_all'], b_score, zero_division=0)
            if f1 > best_epoch_f1:
                best_epoch_f1 = f1
                best_epoch_thresh = thresh
        
        if best_epoch_f1 > best_val_f1:
            best_val_f1 = best_epoch_f1
            best_thresh = best_epoch_thresh
            best_weights = copy.deepcopy(model.state_dict())
            
        if epoch % 10 == 0 or epoch == args.epoch:
            val_auc = roc_auc_score(data['y_val_all'], val_scores)
            print(f"Epoch {epoch:03d}/{args.epoch} | Loss: {epoch_loss/num_batches:.4f} | Val F1: {best_epoch_f1:.4f} @ {best_epoch_thresh:.2f} (Best: {best_val_f1:.4f} @ {best_thresh:.2f}) | Val AUC: {val_auc:.4f}")

    # Load best checkpoint for test set evaluation
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        test_probs = model.predict(data['train_inc_pos'], data['test_inc_all'], data['extra_feature'])
        test_scores = test_probs[:, 1].cpu().numpy()
        
    y_test = data['y_test_all']
    b_preds = (test_scores >= best_thresh).astype(int)
    
    test_pr = precision_score(y_test, b_preds, zero_division=0)
    test_re = recall_score(y_test, b_preds, zero_division=0)
    test_f1 = f1_score(y_test, b_preds, zero_division=0)
    test_auc = roc_auc_score(y_test, test_scores)
    test_prauc = average_precision_score(y_test, test_scores)
    
    print(f"\n--- Seed {seed} Test Results ---")
    print(f"  Precision: {test_pr:.4f}")
    print(f"  Recall:    {test_re:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    print(f"  ROC-AUC:   {test_auc:.4f}")
    print(f"  PR-AUC:    {test_prauc:.4f}")

    return {
        'seed': seed,
        'precision': test_pr,
        'recall': test_re,
        'f1': test_f1,
        'auc': test_auc,
        'prauc': test_prauc,
    }

# ---------------------------------------------------------
# 5. Main Driver
# ---------------------------------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description="HODDI HGNN-SA Multi-Seed Baseline Reproduction")
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 3407, 54321, 123456, 7, 777, 31415], help='Random seeds to test')
    parser.add_argument('--epoch', type=int, default=100, help='Epochs per seed')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--emb_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--conv_dim', type=int, default=64, help='Convolution dimension')
    parser.add_argument('--head', type=int, default=3, help='Attention heads')
    parser.add_argument('--p', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--output_csv', type=str, default='exp_a_hgnn_sa_results.csv', help='Results file')
    args, unknown = parser.parse_known_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting HGNN-SA 7-Seed Reproduction Benchmark...")
    print(f"Target Device: {device}")
    print(f"Seeds: {args.seeds}")

    base_path = find_dataset_path()
    print(f"Base dataset path detected: {base_path}")
    data = load_data(base_path, device)

    results = []
    for seed in args.seeds:
        seed_res = train_and_eval_seed(seed, data, args, device)
        results.append(seed_res)

    res_df = pd.DataFrame(results)
    res_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved raw per-seed results to: {args.output_csv}")

    # Print Summary Table
    print("\n" + "=" * 65)
    print("      HGNN-SA MULTI-SEED REPRODUCTION SUMMARY vs PAPER")
    print("=" * 65)
    print(f"{'Metric':<15}{'Our Reproduction (Mean ± Std)':<32}{'HODDI Paper (Table 7)':<20}")
    print("-" * 65)
    
    paper_benchmarks = {
        'precision': '0.906 ± 0.002',
        'f1': '0.933 ± 0.001',
        'auc': '0.957 ± 0.003',
        'prauc': '0.939 ± 0.008',
        'recall': 'N/A'
    }

    for metric in ['precision', 'recall', 'f1', 'auc', 'prauc']:
        m_mean = res_df[metric].mean()
        m_std = res_df[metric].std()
        paper_val = paper_benchmarks.get(metric, 'N/A')
        print(f"{metric.upper():<15}{f'{m_mean:.4f} ± {m_std:.4f}':<32}{paper_val:<20}")
    print("=" * 65)

if __name__ == '__main__':
    main()
