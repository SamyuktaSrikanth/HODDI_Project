import os, ast, math, copy, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as hnn
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from kaggle_hyperattddi_exp_b import load_data, set_random_seed

class HyperAttDDI_LayerAblation(nn.Module):
    def __init__(self, struct_dim, in_dim=768, emb_dim=256, conv_dim=64, heads=4, d=64, p=0.1, num_layers=2):
        super().__init__()
        self.p = p
        self.L = num_layers
        self.d = d
        
        self.pre_linear = nn.Linear(in_dim, emb_dim)
        
        # Parameterized Hypergraph Layers
        self.convs = nn.ModuleList()
        self.hyper_attr_liners = nn.ModuleList()
        
        # First layer
        self.convs.append(hnn.HypergraphConv(emb_dim, conv_dim, heads=heads, use_attention=True, dropout=p))
        self.hyper_attr_liners.append(nn.Linear(struct_dim, emb_dim))
        
        # Subsequent layers
        for _ in range(1, num_layers):
            self.convs.append(hnn.HypergraphConv(conv_dim * heads, conv_dim, heads=heads, use_attention=True, dropout=p))
            self.hyper_attr_liners.append(nn.Linear(struct_dim, conv_dim * heads))
            
        self.W_q = nn.Linear(conv_dim * heads, d)
        self.W_k = nn.Linear(conv_dim * heads, d)
        self.W_v = nn.Linear(conv_dim * heads, conv_dim * heads)
        
        self.decoder = nn.Sequential(
            nn.Linear(conv_dim * heads, 128),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(128, 1)
        )

    def forward(self, drug_features, struct_features, batch_inc):
        X = F.relu(self.pre_linear(drug_features))
        H_T = batch_inc.T
        
        row, col = torch.where(H_T)
        edges = torch.cat((col.view(1, -1), row.view(1, -1)), dim=0)
        
        # Message Passing through L layers
        for i in range(self.L):
            attr = self.hyper_attr_liners[i](H_T)
            X = self.convs[i](X, edges, hyperedge_attr=attr)
            if i < self.L - 1:
                X = F.relu(X)
            X = F.dropout(X, p=self.p, training=self.training)
            
        # Attention Pooling
        degree_e = H_T.sum(dim=1, keepdim=True).clamp(min=1)
        h_e_mean = (H_T @ X) / degree_e
        
        q = self.W_q(h_e_mean)
        k = self.W_k(X)
        v = self.W_v(X)
        
        scores = torch.matmul(q, k.T) / math.sqrt(self.d)
        scores = scores.masked_fill(H_T == 0, -1e9)
        alpha = F.softmax(scores, dim=1)
        h_e = torch.matmul(alpha, v)
        
        return self.decoder(h_e).squeeze(-1)

def evaluate(model, data, device):
    model.eval()
    all_scores = []
    with torch.no_grad():
        for i in range(0, data['val_inc_all'].shape[1], 512):
            batch_inc = data['val_inc_all'][:, i:i+512].to(device)
            logits = model(data['drug_features'], data['train_inc_pos'], batch_inc)
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
    all_scores = np.concatenate(all_scores, axis=0)
    
    # Fast threshold tuning
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (all_scores >= thresh).astype(int)
        f1 = f1_score(data['y_val_all'], preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    auc = roc_auc_score(data['y_val_all'], all_scores)
    prauc = average_precision_score(data['y_val_all'], all_scores)
    return best_f1, auc, prauc

if __name__ == "__main__":
    print("THIS SCRIPT IS DESIGNED TO BE RUN ON KAGGLE")
    # You can import this model directly in your Kaggle notebook to run L=1,2,3,4
