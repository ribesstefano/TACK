import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import json
import time
import numpy as np
import pandas as pd
import toml
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from data_loader import PROTACLoader, collate_fn
from model import PROTAC_STAN

USE_WANDB = True


def setup_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_batch_to_device(batch, device):
    batch = batch.to(device)
    for attr in ['x', 'edge_index', 'edge_attr', 'batch']:
        if hasattr(batch, attr) and getattr(batch, attr).device != device:
            setattr(batch, attr, getattr(batch, attr).to(device))
    return batch


def evaluate(model, loader, device, criterion=None):
    model.eval()
    all_probs, all_preds, all_targets = [], [], []
    running_loss = 0.0
    
    with torch.no_grad():
        for data in loader:
            protac = move_batch_to_device(data['protac'], device)
            e3 = data['e3_ligase'].to(device)
            poi = data['poi'].to(device)
            label = data['label'].to(device)
            
            outputs = model(protac, e3, poi)
            if criterion:
                running_loss += criterion(outputs, label).item()
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(label.cpu().numpy())
    
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'f1': f1_score(all_targets, all_preds),
        'loss': running_loss / len(loader) if criterion else 0.0
    }
    try:
        metrics['auc'] = roc_auc_score(all_targets, all_probs)
    except ValueError:
        metrics['auc'] = 0.0
    
    return metrics, np.array(all_targets), np.array(all_probs), np.array(all_preds)


def train_fold(model, train_loader, test_loader, device, cfg, fold_id, checkpoints_dir):
    optimizer = optim.Adam(model.parameters(), lr=cfg['train']['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    patience, best_loss, counter = 30, float('inf'), 0
    best_auc, best_metrics, best_weights = 0.0, {}, None
    
    for epoch in range(cfg['train']['num_epochs']):
        model.train()
        running_loss = 0.0
        
        for data in train_loader:
            protac = move_batch_to_device(data['protac'], device)
            e3 = data['e3_ligase'].to(device)
            poi = data['poi'].to(device)
            label = data['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(protac, e3, poi)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        val_metrics, _, _, _ = evaluate(model, test_loader, device, criterion)
        
        if USE_WANDB:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics['loss'],
                "val_auc": val_metrics['auc'],
                "val_acc": val_metrics['accuracy'],
                "val_f1": val_metrics['f1']
            })
        
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_weights = copy.deepcopy(model.state_dict())
            best_metrics = {
                'fold': fold_id,
                'epoch': epoch + 1,
                'roc_auc': val_metrics['auc'],
                'accuracy': val_metrics['accuracy'],
                'f1_score': val_metrics['f1'],
                'train_loss': train_loss,
                'val_loss': val_metrics['loss']
            }
            if USE_WANDB:
                import wandb
                wandb.summary['best_val_auc'] = best_auc
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{checkpoints_dir}/fold_{fold_id}_epoch_{epoch+1}.pt")
        
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"  Fold {fold_id}: Early stopping at epoch {epoch+1}")
                break
    
    return best_metrics, best_weights


def save_predictions(targets, probs, preds, meta_df, output_path, fold_id, task):
    out_df = pd.DataFrame({
        'target': targets,
        'pred': preds,
        'confidence': probs,
        'fold': fold_id,
        'group': meta_df['SMILES_Scaffold_Cluster'].values[:len(targets)] if 'SMILES_Scaffold_Cluster' in meta_df.columns else 'scaffold',
        'method': 'protac-stan',
        'task': task
    })
    out_df.to_csv(output_path, index=False)
    print(f"  -> Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train PROTAC-STAN with cross-validation")
    parser.add_argument('--task', type=str, choices=['bin', 'dc50', 'dmax'], required=True)
    parser.add_argument('--data_dir', type=str, default='data/custom')
    parser.add_argument('--config', type=str, default='config.toml')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    args = parser.parse_args()
    
    global USE_WANDB
    USE_WANDB = not args.no_wandb
    
    cfg = toml.load(args.config)
    cfg['experiment'] = {'task': args.task}
    
    dataset_name = f"custom_{args.task}"
    heldout_name = f"held_out_{args.task}"
    splits_file = f"cv_splits_{args.task}.json"
    csv_file = os.path.join(args.data_dir, f"custom_{args.task}.csv")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = time.strftime('%Y%m%d_%H%M')
    
    output_dir = args.output_dir or f"results_{args.task}_{timestamp}"
    checkpoints_dir = f"{output_dir}/checkpoints"
    preds_dir = f"{output_dir}/predictions"
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)
    
    experiment_group = f"CV_{args.task.upper()}_{timestamp}"
    wandb_project = f"protac-stan-{args.task}"
    
    print(f"Task: {args.task.upper()}")
    print(f"Results: {output_dir}")
    print(f"Device: {device}")
    
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    full_df = pd.read_csv(csv_file)
    results = []
    
    heldout_loader, heldout_df = None, None
    try:
        _, heldout_loader = PROTACLoader(root=args.data_dir, name=heldout_name, batch_size=cfg['train']['batch_size'], train_ratio=0.0)
        heldout_df = pd.read_csv(os.path.join(args.data_dir, f'{heldout_name}.csv'))
        print(f"Loaded held-out set: {len(heldout_df)} rows")
    except Exception as e:
        print(f"Held-out skipped: {e}")
    
    for split in splits:
        fold_id = split['global_fold_id']
        ckpt_path = f"{checkpoints_dir}/fold_{fold_id}_best.pt"
        
        if os.path.exists(ckpt_path):
            print(f"Found checkpoint for Fold {fold_id}. Skipping...")
            continue
        
        print(f"\nTraining Fold {fold_id}...")
        
        if USE_WANDB:
            import wandb
            wandb.init(
                project=wandb_project,
                group=experiment_group,
                name=f"fold_{fold_id}",
                tags=[args.task, "cv"],
                config=cfg,
                reinit=True
            )
        
        setup_seed(cfg['model']['seed'] + fold_id)
        
        train_loader, test_loader = PROTACLoader(
            root=args.data_dir,
            name=dataset_name,
            batch_size=cfg['train']['batch_size'],
            collate_fn=collate_fn,
            train_indices=split['train_idx'],
            test_indices=split['test_idx']
        )
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=cfg['train']['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True
        )
        
        model = PROTAC_STAN(cfg['model']).to(device)
        metrics, best_weights = train_fold(model, train_loader, test_loader, device, cfg, fold_id, checkpoints_dir)
        
        print(f"  -> Best Val AUC: {metrics.get('roc_auc', 0):.4f}")
        torch.save(best_weights, ckpt_path)
        
        model.load_state_dict(best_weights)
        _, targets, probs, preds = evaluate(model, test_loader, device)
        meta = full_df.iloc[split['test_idx']].reset_index(drop=True)
        save_predictions(targets, probs, preds, meta, f"{preds_dir}/preds_fold_{fold_id}.csv", fold_id, args.task)
        
        if heldout_loader:
            _, ht, hp, hpred = evaluate(model, heldout_loader, device)
            try:
                hau = roc_auc_score(ht, hp)
                if USE_WANDB:
                    import wandb
                    wandb.summary['heldout_auc'] = hau
                metrics['heldout_auc'] = hau
                save_predictions(ht, hp, hpred, heldout_df, f"{preds_dir}/heldout_fold_{fold_id}.csv", fold_id, args.task)
            except ValueError:
                pass
        
        results.append(metrics)
        if USE_WANDB:
            import wandb
            wandb.finish()
        pd.DataFrame(results).to_csv(f"{output_dir}/summary.csv", index=False)


if __name__ == "__main__":
    main()
