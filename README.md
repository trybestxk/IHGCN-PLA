# IHGCN-PLA: Interpretable Heterogeneous Graph Convolutional Network for Protein-Ligand Binding Affinity Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2.0](https://img.shields.io/badge/PyTorch-2.2.0-orange.svg)](https://pytorch.org/)
[![DGL 1.1.2](https://img.shields.io/badge/DGL-1.1.2-green.svg)](https://www.dgl.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

IHGCN-PLA is a novel deep learning framework for structure-based protein-ligand binding affinity prediction. The model features **early-stage multimodal fusion** at the convolution layer level, enabling residue nodes to simultaneously aggregate intra-protein structural signals and ligand binding information within each convolutional layer.

![model](C:\Users\kyx\Desktop\DTA_Heterogeneous_Graph_NENU_2025 (3)\model.png)

Unlike existing methods that fuse protein and ligand representations only at the final prediction stage, IHGCN-PLA introduces **operator-level heterogeneity** through specialized graph convolution operators:

- **GINConv** for ligand covalent bond topology
- **GATConv** (8-head attention) for residue-residue spatial interactions
- **GraphSAGE-LSTM** for protein-ligand cross-modal interface modeling

---

## Features

- Early-stage multimodal fusion within each convolutional layer
- Interpretable predictions via HeteroGNNExplainer (77.4% contribution from interaction edges)
- Lightweight pocket-centric architecture without pretrained language model embeddings
- State-of-the-art performance on CASF-2016 and CASF-2013 benchmarks
- NSCLC drug repositioning with 58.3% clinical validation rate
- Reproducible 10-seed experiments with mean ± std reporting

---

## Performance

| Dataset | CI | R | AUC | RMSE | SD |
|---------|-----|---|-----|------|----|
| **CASF-2016** | 0.809 ± 0.004 | 0.813 ± 0.006 | 0.796 ± 0.019 | 1.314 ± 0.051 | 1.266 ± 0.019 |
| **CASF-2013** | 0.805 ± 0.004 | 0.810 ± 0.007 | 0.780 ± 0.025 | 1.371 ± 0.042 | 1.319 ± 0.021 |

Results reported as mean ± standard deviation across 10 independent random seeds.

---

## Installation

```bash
git clone https://github.com/trybestxk/IHGCN-PLA.git
cd IHGCN-PLA

conda env create -f environment.yml
conda activate ihgcn-pla
```

**Requirements:** Python 3.8+, PyTorch 2.2.0, DGL 1.1.2, CUDA 11.8+

---

## Data Preparation

### PDBbind v2020

Download from the [official website](http://www.pdbbind.org.cn/) and place under `data/pdbbind2020/`.

### NSCLC Dataset

Download from [Hugging Face](https://huggingface.co/trybestxk/IHGCN-PLA-data):

```bash
huggingface-cli download trybestxk/IHGCN-PLA-data --repo-type dataset --local-dir data/nsclc/
```

### Preprocessing

```bash
python preprocess.py \
    --data_path data/pdbbind2020/ \
    --output_path data/processed/ \
    --pocket_cutoff 8.0
```

---

## Usage

### Training

```bash
# Multi-seed training (recommended)
python train_fixed_seeds.py

# Single seed
python train.py --seed 100 --lr 0.0001 --batch_size 64 --max_epochs 120
```

Model weights are saved to `./ckpt/best_model_seed_{seed}.pt`

### Evaluation

```bash
# Evaluate all 10 models
python evaluate_all_models.py

# Evaluate single model
python evaluate.py --checkpoint ckpt/best_model_seed_100.pt
```

### Prediction

```bash
python predict.py \
    --checkpoint ckpt/best_model_seed_100.pt \
    --protein_file examples/protein.pdb \
    --ligand_file examples/ligand.sdf \
    --output predictions.csv
```

---

## Pre-trained Weights

Pre-trained weights for all 10 seeds are available on [Hugging Face](https://huggingface.co/trybestxk/IHGCN-PLA-models):

```bash
huggingface-cli download trybestxk/IHGCN-PLA-models --local-dir ckpt/
```

---

## Repository Structure

```
IHGCN-PLA/
├── ckpt/                        # Model weights (.pt files)
├── data/                        # Datasets
│   ├── pdbbind2020/
│   ├── nsclc/
│   └── processed/
├── resultfinal/                 # Training logs and figures
├── model.py                     # IHGCN-PLA architecture
├── train_fixed_seeds.py         # Multi-seed training
├── evaluate_all_models.py       # Multi-model evaluation
├── predict.py                   # Inference
├── preprocess.py                # Data preprocessing
├── Utils.py                     # Utility functions
├── environment.yml              # Conda environment
├── requirements.txt             # Pip requirements
└── README.md
```

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.0001 |
| Batch size | 64 |
| Max epochs | 120 |
| Early stopping patience | 10 |
| Optimizer | AdamW (weight_decay=1e-4) |
| LR scheduler | ReduceLROnPlateau (factor=0.7, patience=10, min_lr=1e-6) |
| Gradient clipping | max_norm=5.0 |
| Loss function | MSELoss |
| Random seeds | [100, 5000, 800, 777, 12345, 99, 512, 468, 26863, 78427] |

---

## Model Architecture

### Heterogeneous Graph Construction

| Component | Description |
|-----------|-------------|
| Nodes | Ligand atoms + Pocket residues (108-dim features) |
| Residue-residue edges | C𝛼-C𝛼 distance ≤ 6.0 Å |
| Protein-ligand edges | C𝛼-atom distance ≤ 8.0 Å |

### Convolution Operators

| Edge Type | Operator | Rationale |
|-----------|----------|-----------|
| Ligand bonds | GINConv | Preserves molecular graph isomorphism |
| Residue-residue | GATConv (8 heads) | Models variable interaction importance |
| Protein-ligand | GraphSAGE-LSTM | Captures sequential binding dynamics |

### Feature Dimensions

Layer 1: 108 → 108, Layer 2: 108 → 216, Layer 3: 216 → 432

Final: Global max pooling → Concatenation (864-dim) → FC (864→512→256→1)

---

## Citation

```bibtex
@article{wang2025ihgcnpla,
  title={IHGCN-PLA: An Interpretable Heterogeneous Graph Convolutional Network for Protein-Ligand Binding Affinity Prediction with Multimodal Interaction Fusion},
  author={Wang, Guishen and Kong, Yuxiang and Fu, Yuyouqiang and Li, Gaoyang and Cao, Chen},
  journal={Journal of Biomedical Informatics},
  year={2025},
  publisher={Elsevier}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [PDBbind](http://www.pdbbind.org.cn/) for binding affinity data
- [ChEMBL](https://www.ebi.ac.uk/chembl/) for bioactivity data
- [Deep Graph Library (DGL)](https://www.dgl.ai/) for graph neural network framework
