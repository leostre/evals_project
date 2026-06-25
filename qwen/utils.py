import torch 
import os 
import numpy as np 
from itertools import product 
import gc 
from tqdm import tqdm
import transformer_lens.utils as utils

from consts import *

SAVE_DIR = './current'

# ------------------ 3. LINEAR PROBE ------------------
class LinearProbe(torch.nn.Module):
    def __init__(self, d_model, n_categories, sigmoid=False):
        super().__init__()
        self.linear = torch.nn.Linear(d_model, n_categories)
        self.sigmoid = sigmoid
    
    def forward(self, x):
        x = self.linear(x)
        if self.sigmoid:
            x = torch.nn.functional.sigmoid(x)
        return x


# ====================== TRAINING (BINARY CLASSIFICATION WITH PER-TARGET METRICS) ======================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from itertools import product 


def train_clf_probes(save_dir, act_types=None, agg_funcs=None, layers=None):
    layers = layers or LAYERS 
    act_types = act_types or ACTIVATION_TYPES 
    agg_funcs = agg_funcs or AGG_FUNCS_ACTIVATIONS
    results = []

    for act_type, layer_idx, agg_type in product(act_types, layers, agg_funcs):
        try:

            print(f"\n=== Training classification probe: {act_type} @ layer {layer_idx} ({layer_idx}) ===")
            
            generator = torch.Generator().manual_seed(42)
            full_dataset = ActivationDataset(act_type, layer_idx, 'all', agg_type=agg_type)
            
            # Ensure labels are binary (0 or 1)
            # If labels are multi-dimensional, we'll handle each target separately
            labels = full_dataset.labels
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)  # Make it (n_samples, 1)
            
            n_targets = labels.size(-1)
            
            # Check if binary (all values are 0 or 1)
            unique_values = torch.unique(labels)
            if not torch.all((unique_values == 0) | (unique_values == 1)):
                print(f"Warning: Labels contain values other than 0 and 1. Values: {unique_values}")
                # Convert to binary using threshold 0.5 if needed
                labels = (labels > 0.5).float()
            
            # Update dataset with potentially converted labels
            full_dataset.labels = labels
            
            train_ds, val_ds = torch.utils.data.random_split(
                full_dataset, lengths=[0.8, 0.2],
                generator=generator
            )
            
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=64)
            
            # Binary classification probe with multiple outputs (each target is binary)
            n_targets = full_dataset.labels.size(-1)
            probe = LinearProbe(full_dataset.data[0].size(-1), n_targets).to(DEVICE).to(torch.float32)
            
            optimizer = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=1e-1)
            criterion = nn.BCEWithLogitsLoss()  # For binary classification
            
            best_val_auc = -float('inf')
            best_state = None
            best_metrics = {}
            
            for epoch in range(EPOCHS):
                # Training
                probe.train()
                epoch_train_loss = 0.0
                for Xb, yb in train_loader:
                    Xb, yb = Xb.to(DEVICE).to(torch.float32), yb.to(DEVICE).to(torch.float32)
                    optimizer.zero_grad()
                    logits = probe(Xb)  # Shape: (batch_size, n_targets)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item()
                
                avg_train_loss = epoch_train_loss / len(train_loader)
                
                # Validation
                probe.eval()
                val_logits, val_true = [], []
                with torch.no_grad():
                    for Xb, yb in val_loader:
                        Xb = Xb.to(torch.float32).to(DEVICE)
                        logits = probe(Xb)
                        val_logits.append(logits.cpu().numpy())
                        val_true.append(yb.numpy())
                
                val_logits = np.concatenate(val_logits, axis=0)
                val_true = np.concatenate(val_true, axis=0)
                val_preds = (val_logits > 0).astype(np.float32)  # Threshold at 0 for logits
                
                # Overall metrics (averaged across targets)
                overall_accuracy = accuracy_score(val_true.flatten(), val_preds.flatten())
                overall_precision = precision_score(val_true.flatten(), val_preds.flatten(), average='binary', zero_division=0)
                overall_recall = recall_score(val_true.flatten(), val_preds.flatten(), average='binary', zero_division=0)
                overall_f1 = f1_score(val_true.flatten(), val_preds.flatten(), average='binary', zero_division=0)
                
                # ROC AUC (handle cases with only one class)
                try:
                    overall_auc = roc_auc_score(val_true.flatten(), val_logits.flatten())
                except ValueError:
                    overall_auc = 0.5  # Random performance if only one class present
                
                # Per-target metrics
                per_target_accuracy = []
                per_target_precision = []
                per_target_recall = []
                per_target_f1 = []
                per_target_auc = []
                per_target_confusion = []
                
                for target_idx in range(n_targets):
                    target_true = val_true[:, target_idx]
                    target_pred = val_preds[:, target_idx]
                    target_logit = val_logits[:, target_idx]
                    
                    # Standard metrics
                    acc = accuracy_score(target_true, target_pred)
                    prec = precision_score(target_true, target_pred, zero_division=0)
                    rec = recall_score(target_true, target_pred, zero_division=0)
                    f1 = f1_score(target_true, target_pred, zero_division=0)
                    
                    # ROC AUC
                    try:
                        auc = roc_auc_score(target_true, target_logit)
                    except ValueError:
                        auc = 0.5
                    
                    # Confusion matrix
                    tn, fp, fn, tp = confusion_matrix(target_true, target_pred, labels=[0, 1]).ravel()
                    
                    per_target_accuracy.append(acc)
                    per_target_precision.append(prec)
                    per_target_recall.append(rec)
                    per_target_f1.append(f1)
                    per_target_auc.append(auc)
                    per_target_confusion.append({'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp})
                
                # Print metrics (optional, commented out for cleaner output)
                # print(f"\nEpoch {epoch+1:2d}")
                # print(f"  Overall: Acc={overall_accuracy:.4f}, F1={overall_f1:.4f}, AUC={overall_auc:.4f}")
                # n_print = min(5, n_targets)
                # print(f"  Per-target AUC (first {n_print}): " + 
                #       ", ".join([f"T{i}: {per_target_auc[i]:.3f}" for i in range(n_print)]))
                
                # Save best model based on overall AUC
                if overall_auc > best_val_auc:
                    best_val_auc = overall_auc
                    best_metrics = {
                        'overall': {
                            'accuracy': overall_accuracy,
                            'precision': overall_precision,
                            'recall': overall_recall,
                            'f1': overall_f1,
                            'auc': overall_auc
                        },
                        'per_target': {
                            'accuracy': per_target_accuracy,
                            'precision': per_target_precision,
                            'recall': per_target_recall,
                            'f1': per_target_f1,
                            'auc': per_target_auc,
                            'confusion': per_target_confusion
                        }
                    }
                    best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

            # Load best model and save
            probe.load_state_dict(best_state)
            torch.save(best_state, f"{save_dir}/classification_probe_l{layer_idx}_{act_type}_{agg_type}.pt")
            
            # Store results with per-target info as JSON strings
            results.append({
                "layer_idx": layer_idx,
                "act_type": act_type,
                'agg_type': agg_type,
                "final_tr_loss": avg_train_loss,
                # Overall metrics
                "overall_accuracy": best_metrics['overall']['accuracy'],
                "overall_precision": best_metrics['overall']['precision'],
                "overall_recall": best_metrics['overall']['recall'],
                "overall_f1": best_metrics['overall']['f1'],
                "overall_auc": best_metrics['overall']['auc'],
                # Per-target metrics (stored as lists)
                "per_target_accuracy": best_metrics['per_target']['accuracy'],
                "per_target_precision": best_metrics['per_target']['precision'],
                "per_target_recall": best_metrics['per_target']['recall'],
                "per_target_f1": best_metrics['per_target']['f1'],
                "per_target_auc": best_metrics['per_target']['auc'],
                "per_target_confusion": best_metrics['per_target']['confusion'],
                "n_targets": n_targets
            })
        except Exception as e:
            import traceback 
            print(f"Error with {act_type}, {layer_idx}, {agg_type}:")
            print(traceback.format_exc())
            continue

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save wide format with per-target lists
    results_df.to_csv(os.path.join(save_dir, 'lin_probes_classification_per_target.csv'), index=False)

    # Create tidy format (one row per target)
    tidy_results = []
    for _, row in results_df.iterrows():
        for target_idx in range(row['n_targets']):
            tidy_results.append({
                'layer_idx': row['layer_idx'],
                'act_type': row['act_type'],
                'agg_type': row['agg_type'],
                'target_idx': target_idx,
                'accuracy': row['per_target_accuracy'][target_idx],
                'precision': row['per_target_precision'][target_idx],
                'recall': row['per_target_recall'][target_idx],
                'f1': row['per_target_f1'][target_idx],
                'auc': row['per_target_auc'][target_idx],
                'confusion_tn': row['per_target_confusion'][target_idx]['tn'],
                'confusion_fp': row['per_target_confusion'][target_idx]['fp'],
                'confusion_fn': row['per_target_confusion'][target_idx]['fn'],
                'confusion_tp': row['per_target_confusion'][target_idx]['tp'],
                'final_tr_loss': row['final_tr_loss']
            })

    tidy_results_df = pd.DataFrame(tidy_results)
    tidy_results_df.to_csv(os.path.join(save_dir, 'lin_probes_classification_per_target_tidy.csv'), index=False)

    print("Done. Results saved in two formats:")
    return tidy_results


# ====================== SAVE ACTIVATIONS PER BATCH ======================
def save_activations(model, save_dir, texts, labels, split_name="all",):
    """Save activations batch-by-batch to avoid OOM"""
    os.makedirs(save_dir, exist_ok=True)
    # Filter out invalid texts
    valid_indices = [i for i, text in enumerate(texts) if text and isinstance(text, str) and len(text.strip()) > 0]
    valid_texts = [texts[i] for i in valid_indices]
    valid_labels = [labels[i] for i in valid_indices]
    
    print(f"Filtered out {len(texts) - len(valid_texts)} invalid samples")
    
    if len(valid_texts) == 0:
        print("No valid texts found!")
        return
    
    # Pre-compute all layer indices for each percentage
    layer_indices = LAYERS
    
    # Create all directories upfront
    for act_type in ACTIVATION_TYPES:
        for layer_idx in layer_indices:
            act_dir = f"{save_dir}/{act_type}_layer{layer_idx}_{split_name}"
            os.makedirs(act_dir, exist_ok=True)
    
    # Process batches
    for i in tqdm(range(0, len(valid_texts), BATCH_SIZE), desc=f"Processing batches {split_name}"):
        batch_texts = valid_texts[i:i+BATCH_SIZE]
        batch_labels = valid_labels[i:i+BATCH_SIZE]
        
        # Skip empty batches
        if not batch_texts:
            continue
        
        try:
            # Method 1: Try to_tokens first
            tokens = model.to_tokens(batch_texts, truncate=True)
            
            # Check if tokens are valid
            if tokens is None or tokens.numel() == 0:
                print(f"Warning: Empty tokens for batch {i//BATCH_SIZE}")
                continue
            
            # Single forward pass for ALL activation types and layers
            with torch.no_grad():
                # Create filter that captures all needed activations
                def names_filter(name):
                    for act_type in ACTIVATION_TYPES:
                        for layer_idx in layer_indices:
                            if name == utils.get_act_name(act_type, layer_idx):
                                return True
                    return False
                
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=names_filter,
                    return_type=None
                )
                
                # Extract and save all activations from this single forward pass
                for act_type in ACTIVATION_TYPES:
                    for layer_idx in layer_indices:
                        act_key = utils.get_act_name(act_type, layer_idx)
                        act = cache[act_key]  # (batch, seq, d)
                        
                        act_dir = f"{save_dir}/{act_type}_layer{layer_idx}_{split_name}"
                        
                        # Apply all aggregation functions
                        for agg_func_name, act_func in AGG_FUNCS_ACTIVATIONS.items():
                            pooled = act_func(act)
                            
                            # Save batch
                            torch.save({
                                "activations": pooled,
                                "labels": batch_labels
                            }, f"{act_dir}/batch_{i//BATCH_SIZE:06d}_{agg_func_name}.pt")
                
                # Clean up
                del cache, tokens
                gc.collect()
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing batch {i//BATCH_SIZE}: {e}")
            print(f"Batch texts sample: {batch_texts[:2]}")
            continue
    
class ActivationDataset(torch.utils.data.Dataset):
    def __init__(self, act_type, layer_idx, split="all", gathering_func=None, agg_type='last'):
        self.data, self.labels = gathering_func(act_type, layer_idx, split, agg_type)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        return data, self.labels[idx]
    
def get_probe_name(layer_idx, act_type, agg_type):
    return f'classification_probe_l{layer_idx}_{act_type}_{agg_type}.pt'

def load_probe(probe_dir, layer_idx, act_type, agg_type):
    path = os.path.join(probe_dir, get_probe_name(layer_idx, act_type, agg_type))
    state = torch.load(path)
    probe = LinearProbe(state['linear.weight'].size(-1), 1)
    probe.load_state_dict(state)
    return probe 

def load_all_probes(probe_dir, layers, act_types, agg_types):
    probes = {}
    for layer_idx, act_type, agg_type in product(layers, act_types, agg_types):
        probes[get_probe_name(layer_idx, act_type, agg_type)] = load_probe(probe_dir, layer_idx, act_type, agg_type)
    return probes
    
# ====================== DATA PREPARATION ======================
def load_probe_data_cached_rgr(act_type, layer_idx, agg_type, sigmoid, act_dir, probe_dir):
    """Cached data loading"""
    cache_key = f"{act_type}_{layer_idx}_{agg_type}"
    
    if not hasattr(load_probe_data_cached_rgr, 'cache'):
        load_probe_data_cached_rgr.cache = {}
    
    if cache_key in load_probe_data_cached_rgr.cache:
        return load_probe_data_cached_rgr.cache[cache_key]
    
    probe_path = f'{probe_dir}/regression_probe_l{layer_idx}_{act_type}_{agg_type}.pt'
    full_dataset = ActivationDataset(act_type, layer_idx, 'all', gathering_func=partial(), agg_type=agg_type)
    
    probe_state = torch.load(probe_path)
    probe = LinearProbe(full_dataset.data.size(-1), full_dataset.data.size(-2), sigmoid=sigmoid)
    probe.load_state_dict(probe_state)
    probe.eval()
    
    with torch.no_grad():
        outputs = probe(full_dataset.data.to(torch.float32)).cpu().numpy()
        
        # Option 2: Per-target centering (across questions for each target)
        # Shape: (n_questions, n_targets)
        mean = np.mean(outputs, axis=0, keepdims=True)  # (1, n_targets)
        std = np.std(outputs, axis=0, keepdims=True)    # (1, n_targets)
        
        # Avoid division by zero
        std = np.where(std < 1e-8, 1.0, std)
        
        outputs = (outputs - mean) / std

    
    
    result = {
        'outputs': outputs,  # Shape: (n_questions, n_targets)
        'probe_weight': probe.linear.weight.detach().numpy(),
        'labels': full_dataset.labels,
        'activations': full_dataset.data,
        'n_samples': len(full_dataset.labels)
    }
    
    load_probe_data_cached_rgr.cache[cache_key] = result
    return result

def gather_full_tensor_sfq(act_type, layer_idx, split='all', agg_type='last', act_dir=None):
    act_dir = act_dir or SAVE_DIR
    files = sorted([
            f for f in os.listdir(f"{act_dir}/{act_type}_layer{layer_idx}_{split}")
            if f.endswith(".pt") and agg_type in f
     ])
    activations = []
    scores = []
    for file in files:
        f = torch.load(f"{act_dir}/{act_type}_layer{layer_idx}_{split}/{file}")
        scores.extend(f['labels'])    
        activations.append(f['activations'])
        
    activations = torch.cat(
        activations, dim=0
    )
    scores = torch.tensor(np.array(scores), dtype=torch.float32).view(-1, 1)
    return activations, scores


GATHERING_FUNCS = {
    'sfq': gather_full_tensor_sfq, 
}