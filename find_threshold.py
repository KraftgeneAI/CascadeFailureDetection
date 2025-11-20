import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sys
import os

# Import your model and dataset
from multimodal_cascade_model import UnifiedCascadePredictionModel
from cascade_dataset import CascadeDataset, collate_cascade_batch

def find_best_thresholds(checkpoint_path, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from: {checkpoint_path}")
    
    # 1. Load Model
    model = UnifiedCascadePredictionModel(
        embedding_dim=128, hidden_dim=128, num_gnn_layers=3, heads=4
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 2. Load Validation Data
    val_dataset = CascadeDataset(f"{data_dir}/val", mode='full_sequence')
    val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_cascade_batch)
    
    print("Running inference on validation set...")
    all_node_probs = []
    all_node_labels = []
    all_cascade_probs = []
    all_cascade_labels = []
    
    # 3. Collect Probabilities (Inference Phase)
    with torch.no_grad():
        for batch in tqdm(val_loader):
            batch_device = {}
            for k, v in batch.items():
                if k == 'graph_properties': continue 
                if isinstance(v, torch.Tensor):
                    batch_device[k] = v.to(device)
            
            if 'node_failure_labels' not in batch_device: continue
            
            outputs = model(batch_device)
            
            # Node Level
            probs = outputs['failure_probability'].squeeze(-1).cpu() # [B, N]
            labels = batch_device['node_failure_labels'].cpu()       # [B, N]
            all_node_probs.append(probs.view(-1))
            all_node_labels.append(labels.view(-1))
            
            # Cascade Level (Max probability in the graph)
            c_probs = probs.max(dim=1)[0]
            c_labels = (labels.max(dim=1)[0] > 0.5).float()
            all_cascade_probs.append(c_probs)
            all_cascade_labels.append(c_labels)

    # Concatenate everything
    node_probs = torch.cat(all_node_probs)
    node_labels = torch.cat(all_node_labels)
    casc_probs = torch.cat(all_cascade_probs)
    casc_labels = torch.cat(all_cascade_labels)
    
    # 4. Grid Search for Best Thresholds
    print("\nCalculating optimal thresholds...")
    
    def get_f1(preds, targets):
        tp = (preds * targets).sum()
        fp = (preds * (1-targets)).sum()
        fn = ((1-preds) * targets).sum()
        return 2*tp / (2*tp + fp + fn + 1e-7)

    best_node_f1 = 0
    best_node_thresh = 0
    best_casc_f1 = 0
    best_casc_thresh = 0
    
    # Test thresholds from 0.05 to 0.95
    thresholds = np.arange(0.05, 0.96, 0.01)
    
    for t in thresholds:
        # Node F1
        n_preds = (node_probs > t).float()
        nf1 = get_f1(n_preds, node_labels).item()
        if nf1 > best_node_f1:
            best_node_f1 = nf1
            best_node_thresh = t
            
        # Cascade F1
        c_preds = (casc_probs > t).float()
        cf1 = get_f1(c_preds, casc_labels).item()
        if cf1 > best_casc_f1:
            best_casc_f1 = cf1
            best_casc_thresh = t
            
    print("="*50)
    print("OPTIMAL MODEL CONFIGURATION")
    print("="*50)
    print(f"Best Node F1:    {best_node_f1:.4f}  at Threshold: {best_node_thresh:.2f}")
    print(f"Best Cascade F1: {best_casc_f1:.4f}  at Threshold: {best_casc_thresh:.2f}")
    print("="*50)

if __name__ == "__main__":
    # Usage: python find_thresholds.py
    find_best_thresholds("checkpoints/best_model.pth", "data")