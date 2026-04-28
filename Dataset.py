# @Version : 3.7
# @Author : 孔渝翔
# @Time  ：2024/10/29 15:58
# @FName  :Dataset.py
import torch
from dgl.data import DGLDataset
from Bio.PDB import PDBParser
from rdkit import Chem
import dgl
import numpy as np
import pandas as pd
import os

from tqdm import tqdm

class MyHeteroDataSet(DGLDataset):
    def __init__(self, save_dir, ligands, residues, i_edges, labels, affinity_dict,
                 raw_dir=None, force_reload=False, verbose=True):
        self.ligands = ligands
        self.residues = residues
        self.i_edges = i_edges
        self.graphs = []
        self.labels = torch.tensor(labels, dtype=torch.float32)  # 统一为 labels
        self.affinity_dict = affinity_dict
        super().__init__(name="Mydataset", save_dir=save_dir)

    def download(self):
        pass

    def process(self):
        for mol, residue, interaction, label in tqdm(zip(self.ligands, self.residues, self.i_edges, self.labels),
                                                     total=len(self.ligands),desc="处理图像特征中",mininterval=5.0):
            if len(self.ligands) == len(self.labels) == len(self.residues) == len(self.i_edges):
                mol_feats = np.asarray(mol[1])
                residue_feats = np.asarray(residue[1])
                c_edges = torch.tensor(mol[2]).transpose(1, 0)
                r_edges = torch.tensor(residue[2]).transpose(1, 0)
                interaction_edges = torch.tensor(interaction).transpose(1, 0)

                graph_data = {
                    ('ligand_atom', 'bond', 'ligand_atom'): (c_edges[0], c_edges[1]),
                    ('residue', 'r_r_interation', 'residue'): (r_edges[0], r_edges[1]),
                    ('ligand_atom', 'l_r_interaction', 'residue'): (interaction_edges[1], interaction_edges[0]),
                                  }

                num_nodes = {
                    'ligand_atom': mol[0],
                    'residue': residue[0]
                }

                g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes)
                g.nodes['ligand_atom'].data['feat'] = torch.tensor(mol[1], dtype=torch.float32)
                g.nodes['residue'].data['feat'] = torch.tensor(residue[1], dtype=torch.float32)
                self.graphs.append(g)
            else:
                raise Exception("数据不匹配，发生了错误！")

    def save(self):
        dgl.save_graphs(self.save_dir, self.graphs, labels={'labels': self.labels})

    def load(self):
        self.graphs, labels = dgl.load_graphs(self.save_dir)
        self.labels = labels['labels']

    def has_cache(self):
        return os.path.exists(self.save_dir)

    def __getitem__(self, item):
        return self.graphs[item], self.labels[item]  # 确保返回图和标签

    def __len__(self):
        return len(self.graphs)
