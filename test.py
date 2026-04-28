import pandas as pd
import Utils
import dgl
from model import HGCNDTA
import torch
import numpy as np
from numba import njit
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
import os

cuda_name = 'cuda:0'
device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64

FIXED_SEEDS = [100, 5000, 800, 777, 12345, 99, 512, 468, 26863, 78427]
MODEL_DIR = 'ckpt'

file_path = './DataSet03/processed/'


def load_data_loaders():
    data_loaders = {}
    for ds in ['test2016.bin', 'test2013.bin']:
        dataset = Utils.LoadedHeteroDataSet(file_path + ds)
        loader = dgl.dataloading.GraphDataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            collate_fn=Utils.custom_collate_fn,
            shuffle=False
        )
        data_loaders[ds] = loader
    return data_loaders


data_loaders = load_data_loaders()


@njit
def c_index(y_true, y_pred):
    summ = 0
    pair = 0
    for i in range(1, len(y_true)):
        for j in range(0, i):
            pair += 1
            if y_true[i] > y_true[j]:
                summ += 1 * (y_pred[i] > y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            elif y_true[i] < y_true[j]:
                summ += 1 * (y_pred[i] < y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            else:
                pair -= 1
    if pair != 0:
        return summ / pair
    else:
        return 0


def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def MAE(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def CORR(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


def SD(y_true, y_pred):
    from sklearn.linear_model import LinearRegression
    y_pred = y_pred.reshape((-1, 1))
    lr = LinearRegression().fit(y_pred, y_true)
    y_ = lr.predict(y_pred)
    return np.sqrt(np.square(y_true - y_).sum() / (len(y_pred) - 1))


def auc(y_true, y_pred):
    threshold = np.mean(y_true)
    y_true_bi = (y_true > threshold).astype(int)
    y_pred_bi = (y_pred > threshold).astype(int)
    return roc_auc_score(y_true_bi, y_pred_bi)


def calculate_all_metrics(y_true, y_pred):
    return {
        'CI': float(c_index(y_true, y_pred)),
        'RMSE': float(RMSE(y_true, y_pred)),
        'MAE': float(MAE(y_true, y_pred)),
        'CORR': float(CORR(y_true, y_pred)),
        'SD': float(SD(y_true, y_pred)),
        'AUC': float(auc(y_true, y_pred)),
    }


def predict_on_loader(model, device, loader):
    model.eval()
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for graphs, labels in loader:
            g = dgl.batch(graphs).to(device)
            labels = torch.tensor(labels, dtype=torch.float32).to(device).unsqueeze(1)
            feat = {
                'ligand_atom': g.nodes['ligand_atom'].data['feat'],
                'residue': g.nodes['residue'].data['feat']
            }
            out = model(g, feat)
            all_outputs.append(out.cpu().numpy().reshape(-1))
            all_targets.append(labels.cpu().numpy().reshape(-1))
    targets = np.concatenate(all_targets).reshape(-1)
    outputs = np.concatenate(all_outputs).reshape(-1)
    return targets, outputs


all_results = []

for seed in FIXED_SEEDS:
    model_path = os.path.join(MODEL_DIR, f'best_model_seed_{seed}.pt')
    if not os.path.exists(model_path):
        continue
    model = HGCNDTA().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    result = {'seed': seed}
    for dataset_name in ['test2016.bin', 'test2013.bin']:
        y_true, y_pred = predict_on_loader(model, device, data_loaders[dataset_name])
        result[dataset_name] = calculate_all_metrics(y_true, y_pred)
    all_results.append(result)

print(f"{'Seed':<10} {'CI':<10} {'RMSE':<10} {'MAE':<10} {'CORR':<10} {'SD':<10} {'AUC':<10}")
for result in all_results:
    m = result['test2016.bin']
    print(f"{result['seed']:<10} {m['CI']:<10.4f} {m['RMSE']:<10.4f} {m['MAE']:<10.4f} {m['CORR']:<10.4f} {m['SD']:<10.4f} {m['AUC']:<10.4f}")

print()
print(f"{'Seed':<10} {'CI':<10} {'RMSE':<10} {'MAE':<10} {'CORR':<10} {'SD':<10} {'AUC':<10}")
for result in all_results:
    m = result['test2013.bin']
    print(f"{result['seed']:<10} {m['CI']:<10.4f} {m['RMSE']:<10.4f} {m['MAE']:<10.4f} {m['CORR']:<10.4f} {m['SD']:<10.4f} {m['AUC']:<10.4f}")

print()
for dataset_name, label in [('test2016.bin', 'Test2016'), ('test2013.bin', 'Test2013')]:
    print(f"{label} Mean +- Std:")
    for metric_name in ['CI', 'RMSE', 'MAE', 'CORR', 'SD', 'AUC']:
        values = [r[dataset_name][metric_name] for r in all_results]
        print(f"  {metric_name:<6}: {np.mean(values):.4f} +- {np.std(values):.4f}  (min: {np.min(values):.4f}, max: {np.max(values):.4f})")
    print()