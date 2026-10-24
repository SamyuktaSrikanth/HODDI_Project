import os, ast, torch
import pandas as pd
import numpy as np
import torch.nn.functional as F

# Import your model definition and data loader from the existing script
from kaggle_hyperattddi_exp_b import HyperAttDDI_NoSE, load_data, find_dir, find_file, set_random_seed

def extract_culprit_weights():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading data...")
    data = load_data(device)
    
    # We need the drug names for interpretation
    dict_path = find_file('Drugbank_ID_SMILE_all_structure links.csv')
    smiles_ds = pd.read_csv(dict_path)
    drug_names = smiles_ds.set_index('DrugBank ID')['Name'].to_dict()
    
    # Reconstruct the index-to-drugbank mapping used in load_data
    subset_dir = find_dir('subset_drug2-8_SE5-50')
    merged_dir = os.path.join(subset_dir, 'merged_subset')
    all_ds = pd.concat([pd.read_csv(os.path.join(merged_dir, 'positive_samples_2014Q3_2024Q3_step6.csv')), 
                        pd.read_csv(os.path.join(merged_dir, 'negative_samples_condition123_SE_above_0.9.csv'))], axis=0)
    
    all_drugs = set()
    for drug_ids in all_ds['DrugBankID']:
        all_drugs.update([d for d in ast.literal_eval(drug_ids) if d.lower() != 'none'])
    drug_to_index_raw = {drug: idx for idx, drug in enumerate(all_drugs)}
    raw_drug_id_list = [None] * len(all_drugs)
    for did in drug_to_index_raw:
        raw_drug_id_list[drug_to_index_raw[did]] = did
        
    drugbank_to_smiles_keys = smiles_ds['DrugBank ID'].values
    drug_id_list = [did for did in raw_drug_id_list if did in drugbank_to_smiles_keys]
    index_to_drugbank = {idx: did for idx, did in enumerate(drug_id_list)}

    print("Initializing model...")
    model = HyperAttDDI_NoSE(
        struct_dim=data['train_inc_pos'].shape[1],
        in_dim=768, emb_dim=256, conv_dim=64, heads=4, d=64, p=0.0
    ).to(device)
    
    # Ideally, you load the best trained weights here:
    # model.load_state_dict(torch.load("best_hyperattddi.pt"))
    # For now, we will do a forward pass to extract the attention matrices.
    model.eval()
    
    print("\n--- Extracting Attention Weights (Alpha) ---")
    with torch.no_grad():
        # Let's take a small batch of 5 test prescriptions
        batch_inc = data['test_inc_all'][:, :5].to(device)
        
        # We need to hack into the forward pass to return alpha
        # Standard forward pass steps from HyperAttDDI_NoSE:
        X1 = F.relu(model.pre_linear(data['drug_features']))
        H_T = batch_inc.T
        row, col = torch.where(H_T)
        edges = torch.cat((col.view(1, -1), row.view(1, -1)), dim=0)
        
        attr1 = model.hyper_attr_liner1(H_T)
        X1 = F.relu(model.conv1(X1, edges, hyperedge_attr=attr1))
        attr2 = model.hyper_attr_liner2(H_T)
        X2 = F.relu(model.conv2(X1, edges, hyperedge_attr=attr2))
        
        degree_e = H_T.sum(dim=1, keepdim=True).clamp(min=1)
        h_e_mean = (H_T @ X2) / degree_e
        
        q = model.W_q(h_e_mean)
        k = model.W_k(X2)
        scores = torch.matmul(q, k.T) / math.sqrt(model.d)
        scores = scores.masked_fill(H_T == 0, -1e9)
        alpha = F.softmax(scores, dim=1) # Shape: (batch_size, num_drugs)
        
        # Analyze the attention for each prescription
        for i in range(5):
            print(f"\nPrescription {i+1}:")
            # Get the drugs present in this prescription
            drug_indices = torch.where(H_T[i] == 1)[0].cpu().numpy()
            att_weights = alpha[i, drug_indices].cpu().numpy()
            
            # Sort by highest attention weight
            sorted_idx = np.argsort(att_weights)[::-1]
            
            for rank, idx in enumerate(sorted_idx):
                global_drug_idx = drug_indices[idx]
                drug_id = index_to_drugbank[global_drug_idx]
                drug_name = drug_names.get(drug_id, "Unknown Drug")
                weight = att_weights[idx]
                print(f"  {rank+1}. [Weight: {weight:.4f}] {drug_name} ({drug_id})")

if __name__ == "__main__":
    extract_culprit_weights()
