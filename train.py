import pandas
import Utils
import dgl
from model import HGCNDTA
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import json

os.makedirs('ckpt', exist_ok=True)
os.makedirs('resultfinal', exist_ok=True)

cuda_name = 'cuda:0'
LR = 0.0001
NUM_EPOCHS = 120
PATIENCE = 10
device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64

loss_fn = nn.MSELoss(reduction='mean')

FIXED_SEEDS = [100, 5000, 800, 777, 12345, 99, 512, 468, 26863, 78427]

file_path = './DataSet03/processed/'
datasets = ['train.bin', 'val.bin', 'test2016.bin', 'test2013.bin']


def load_data_loaders():
    data_loaders = {}
    for ds in datasets:
        dataset = Utils.LoadedHeteroDataSet(file_path + ds)
        loader = dgl.dataloading.GraphDataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            collate_fn=Utils.custom_collate_fn,
            shuffle=(ds == 'train.bin')
        )
        data_loaders[ds] = loader
    return data_loaders


def format_metrics(metrics):
    return (f"Loss={metrics['loss']:.4f}, "
            f"RMSE={metrics['RMSE']:.4f}, "
            f"MAE={metrics['MAE']:.4f}, "
            f"CORR={metrics['CORR']:.4f}, "
            f"CI={metrics['c_index']:.4f}")


def train_model(model, device, loader, optimizer, epoch, loss_f):
    model.train()
    outputs = []
    targets = []
    total_loss = 0.0
    batch_count = 0

    for batch_idx, (graphs, lables) in enumerate(loader, 1):
        g = dgl.batch(graphs).to(device)
        lables = torch.tensor(lables, dtype=torch.float32).to(device).unsqueeze(1)
        feat = {
            'ligand_atom': g.nodes['ligand_atom'].data['feat'],
            'residue': g.nodes['residue'].data['feat']
        }
        optimizer.zero_grad()
        out = model(g, feat)
        loss_value = loss_f(out, lables)
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss_value.item()
        outputs.append(out.cpu().detach().numpy().reshape(-1))
        targets.append(lables.cpu().detach().numpy().reshape(-1))
        batch_count += 1

        if batch_idx % 20 == 0:
            print(f"  Epoch [{epoch}/{NUM_EPOCHS}] Batch [{batch_idx}/{len(loader)}] "
                  f"Avg Loss: {total_loss / batch_count:.4f}")

    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)

    return {
        'loss': total_loss / batch_count,
        'c_index': Utils.c_index(targets, outputs),
        'RMSE': Utils.RMSE(targets, outputs),
        'MAE': Utils.MAE(targets, outputs),
        'SD': Utils.SD(targets, outputs),
        'CORR': Utils.CORR(targets, outputs),
    }


def test_model(model, device, loader, loss_f):
    model.eval()
    outputs = []
    targets = []
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for graphs, lables in loader:
            g = dgl.batch(graphs).to(device)
            lables = torch.tensor(lables, dtype=torch.float32).to(device).unsqueeze(1)
            feat = {
                'ligand_atom': g.nodes['ligand_atom'].data['feat'],
                'residue': g.nodes['residue'].data['feat']
            }
            out = model(g, feat)
            loss_value = loss_f(out, lables)
            total_loss += loss_value.item()
            batch_count += 1
            outputs.append(out.cpu().numpy().reshape(-1))
            targets.append(lables.cpu().numpy().reshape(-1))

    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)

    return {
        'loss': total_loss / batch_count,
        'c_index': Utils.c_index(targets, outputs),
        'RMSE': Utils.RMSE(targets, outputs),
        'MAE': Utils.MAE(targets, outputs),
        'SD': Utils.SD(targets, outputs),
        'CORR': Utils.CORR(targets, outputs),
    }


def train_single_seed(seed, seed_index):
    print(f"\n{'=' * 80}")
    print(f"Seed {seed_index + 1}/{len(FIXED_SEEDS)}: {seed}")
    print(f"{'=' * 80}")

    dgl.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    data_loaders = load_data_loaders()

    model = HGCNDTA().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=10,
        verbose=True, threshold=0.0001, min_lr=1e-6
    )

    train_losses = []
    val_losses = []
    lr_history = []

    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0
    best_model_state = None

    model_path = f'./ckpt/best_model_seed_{seed}.pt'

    for epoch in range(1, NUM_EPOCHS + 1):
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} | LR: {current_lr:.6f} ---")

        train_out = train_model(model, device, data_loaders['train.bin'],
                                optimizer, epoch, loss_fn)
        train_losses.append(train_out['loss'])
        print(f"[TRAIN] {format_metrics(train_out)}")

        val_out = test_model(model, device, data_loaders['val.bin'], loss_fn)
        val_losses.append(val_out['loss'])
        print(f"[VAL]   {format_metrics(val_out)}")

        test2016_out = test_model(model, device, data_loaders['test2016.bin'], loss_fn)
        test2013_out = test_model(model, device, data_loaders['test2013.bin'], loss_fn)
        print(f"[TEST2016] {format_metrics(test2016_out)}")
        print(f"[TEST2013] {format_metrics(test2013_out)}")

        scheduler.step(val_out['loss'])
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"  LR: {current_lr:.6f} -> {new_lr:.6f}")

        if val_out['loss'] < best_val_loss:
            best_val_loss = val_out['loss']
            best_epoch = epoch
            patience_counter = 0
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
            }
            print(f"  Best model updated at Epoch {epoch} (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at Epoch {epoch}, best Epoch: {best_epoch}")
            break

    torch.save(best_model_state, model_path)
    print(f"\nBest model saved: {model_path}")

    model.load_state_dict({k: v.to(device) for k, v in best_model_state['model_state_dict'].items()})

    final_val = test_model(model, device, data_loaders['val.bin'], loss_fn)
    final_2016 = test_model(model, device, data_loaders['test2016.bin'], loss_fn)
    final_2013 = test_model(model, device, data_loaders['test2013.bin'], loss_fn)

    print(f'\nSeed {seed} Final Results (Best Model @ Epoch {best_epoch})')
    print(f'Val:      {format_metrics(final_val)}')
    print(f'Test2016: {format_metrics(final_2016)}')
    print(f'Test2013: {format_metrics(final_2013)}')

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.axvline(x=best_epoch - 1, color='r', linestyle='--', alpha=0.7,
                label=f'Best Epoch {best_epoch}')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Train / Val Loss (Seed {seed})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'./resultfinal/loss_curve_seed_{seed}.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(lr_history, label='LR', linewidth=2, color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title(f'LR Schedule (Seed {seed})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'./resultfinal/lr_curve_seed_{seed}.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'seed': seed,
        'best_epoch': best_epoch,
        'val_loss': best_val_loss,
        'val_corr': final_val['CORR'],
        'test2016_ci': final_2016['c_index'],
        'test2016_corr': final_2016['CORR'],
        'test2013_ci': final_2013['c_index'],
        'test2013_corr': final_2013['CORR'],
        'val_results': final_val,
        'test2016_results': final_2016,
        'test2013_results': final_2013,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }


all_results = []

for idx, seed in enumerate(FIXED_SEEDS):
    result = train_single_seed(seed, idx)
    all_results.append(result)

    progress = {
        'completed_count': len(all_results),
        'total_count': len(FIXED_SEEDS),
        'completed_seeds': [r['seed'] for r in all_results],
    }
    with open('./resultfinal/training_progress.json', 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)

val_corrs = [r['val_corr'] for r in all_results]
test2016_corrs = [r['test2016_corr'] for r in all_results]
test2016_cis = [r['test2016_ci'] for r in all_results]
test2013_corrs = [r['test2013_corr'] for r in all_results]
test2013_cis = [r['test2013_ci'] for r in all_results]


def print_stats(values, name):
    print(f"{name}:")
    print(f"  Mean +- Std : {np.mean(values):.4f} +- {np.std(values):.4f}")
    print(f"  Min / Max   : {np.min(values):.4f} / {np.max(values):.4f}")
    print()


print_stats(val_corrs, "Validation CORR")
print_stats(test2016_corrs, "Test2016 CORR")
print_stats(test2016_cis, "Test2016 CI")
print_stats(test2013_corrs, "Test2013 CORR")
print_stats(test2013_cis, "Test2013 CI")

print(f"{'Seed':<10} {'Val CORR':<12} {'Test2016 CORR':<15} {'Test2016 CI':<12} "
      f"{'Test2013 CORR':<15} {'Test2013 CI':<12} {'Best Epoch':<12}")
print("-" * 100)
for r in all_results:
    print(f"{r['seed']:<10} {r['val_corr']:<12.4f} {r['test2016_corr']:<15.4f} "
          f"{r['test2016_ci']:<12.4f} {r['test2013_corr']:<15.4f} "
          f"{r['test2013_ci']:<12.4f} {r['best_epoch']:<12}")

summary = {
    'training_config': {
        'lr': LR,
        'epochs': NUM_EPOCHS,
        'patience': PATIENCE,
        'batch_size': BATCH_SIZE,
        'loss': 'MSELoss(reduction=mean)',
        'optimizer': 'AdamW(weight_decay=1e-4)',
        'grad_clip': 5.0,
        'scheduler': 'ReduceLROnPlateau(mode=min, factor=0.7, patience=10, min_lr=1e-6)',
        'save_strategy': 'Best validation loss',
    },
    'seeds': FIXED_SEEDS,
    'statistics': {
        'val_corr': {'mean': float(np.mean(val_corrs)), 'std': float(np.std(val_corrs)),
                     'min': float(np.min(val_corrs)), 'max': float(np.max(val_corrs))},
        'test2016_corr': {'mean': float(np.mean(test2016_corrs)), 'std': float(np.std(test2016_corrs)),
                          'min': float(np.min(test2016_corrs)), 'max': float(np.max(test2016_corrs))},
        'test2016_ci': {'mean': float(np.mean(test2016_cis)), 'std': float(np.std(test2016_cis)),
                        'min': float(np.min(test2016_cis)), 'max': float(np.max(test2016_cis))},
        'test2013_corr': {'mean': float(np.mean(test2013_corrs)), 'std': float(np.std(test2013_corrs)),
                          'min': float(np.min(test2013_corrs)), 'max': float(np.max(test2013_corrs))},
        'test2013_ci': {'mean': float(np.mean(test2013_cis)), 'std': float(np.std(test2013_cis)),
                        'min': float(np.min(test2013_cis)), 'max': float(np.max(test2013_cis))},
    },
    'detailed_results': [
        {
            'seed': r['seed'],
            'best_epoch': r['best_epoch'],
            'val_loss': float(r['val_loss']),
            'val_corr': float(r['val_corr']),
            'test2016_corr': float(r['test2016_corr']),
            'test2016_ci': float(r['test2016_ci']),
            'test2013_corr': float(r['test2013_corr']),
            'test2013_ci': float(r['test2013_ci']),
        }
        for r in all_results
    ],
}

with open('./resultfinal/multi_seed_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\nResults saved:")
print(f"  Model weights: ./ckpt/best_model_seed_{{seed}}.pt")
print(f"  Summary JSON:  ./resultfinal/multi_seed_summary.json")
print(f"  Loss curves:   ./resultfinal/loss_curve_seed_{{seed}}.png")
print(f"  LR curves:     ./resultfinal/lr_curve_seed_{{seed}}.png")