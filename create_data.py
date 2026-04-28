# @Version : 3.7
# @Author : 孔渝翔
# @Time  ：2024/10/29 20:58
# @FName  :scratch06.py

import numpy as np
import pandas as pd
from rdkit import Chem
import networkx as nt
import os
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
import logging
from Dataset import MyHeteroDataSet
from tqdm import tqdm
from dgl.dataloading import GraphDataLoader




# 独热编码
def one_of_k_encoding(x, allowset):
    if x not in allowset:
        x = allowset[-1]
    return list(map(lambda s: x == s, allowset))


def one_of_k_encoding_uk(x, allowset):
    # if x not in allowset:
    #     # raise Exception(f"{x} not in allowset ")
    return list(map(lambda s: x == s, allowset))


# 获取分子的特征
def get_molfeatures(atom):
    return np.array(
        # 获取当前的原子符号
        one_of_k_encoding(atom.GetSymbol(),
                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                           'Pt', 'Hg', 'Pb', 'Unknown'])
        +
        # 获取当前原子的键数
        one_of_k_encoding_uk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        +
        # 获取当前原子的氢原子数量
        one_of_k_encoding_uk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        +
        # 获取隐式价的个数
        one_of_k_encoding_uk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        +
        # 获取手性标签
        one_of_k_encoding_uk(str(atom.GetChiralTag()),
                             ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER',
                              'misc']) +
        # 获取形式电荷
        one_of_k_encoding_uk(atom.GetFormalCharge(), [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc']) +
        # 获取自由基电子数
        one_of_k_encoding_uk(atom.GetNumRadicalElectrons(), [0, 1, 2, 3, 4, 'misc']) +
        # 获取当前原子的杂糅化状态
        one_of_k_encoding_uk(str(atom.GetHybridization()), ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc']) +
        # 判断当前原子是否存在环
        [atom.IsInRing()] +
        # 判断是否为芬芳形
        [atom.GetIsAromatic()]

    )


# 创建药分子的原子图
def smiles_tograph(mol):
    features = []  # 创建整个图的特征列表
    c_size = mol.GetNumAtoms()  # 获取当前分子中的原子数量
    for atom in mol.GetAtoms():
        feature = get_molfeatures(atom)
        features.append(feature / sum(feature))
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    # 由于分子的遍历顺序和键的遍历顺序相同，不用过多考虑顺序问题
    g = nt.Graph(edges).to_directed()  # 创建有向图 单双键的考虑

    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index  # 返回节点个数，特征，邻接矩阵

#设置原子的id顺序
def set_number(pok):
    i = 1
    for atom in pok.get_atoms():
        atom.set_serial_number(i)
        i += 1
    return pok

#获取残基的特征
def get_residues_feature(pok, pdb_file):  # 用来返回一个残基的特征
    # 处理残基中每个原子的特征，并且平均后返回
    strcut = remove_pocket_Hs(pok) #去除残基的氢原子
    struct = set_number(strcut)  #对残基中的原子进行顺序编号
    mol = Chem.MolFromPDBFile(pdb_file) #将pocket转换为分子，来和残基中的原子对应
    if mol is None:  #判断分子是不是错误类型
        mol = Chem.MolFromPDBFile(pdb_file, sanitize=False)
    features = []
    for residue in pok.get_residues():
        atoms_list = []
        atoms_index = [atom.serial_number for atom in residue.get_atoms()]  # 获取当前残基中的原子的序号
        for i, atom in enumerate(mol.GetAtoms(), start=1):
            if i < atoms_index[0]:
                continue
            else:
                if i <= atoms_index[-1]:
                    feature = get_molfeatures(atom)
                    atoms_list.append(feature / sum(feature))
                else:
                    break
        features.append(np.mean(atoms_list, axis=0))
    return features

#获取残基中CA原子的空间坐标
def get_atom_position(residue, atom_name='CA'):  # 获取残基中CA原子的位置
    # if residue.get_resname() != 'GLY':
    #     return residue[atom_name]
    # else:
    try:
        atom = residue['CA']  # 针对数据错误数据集中没有CA原子
    except(Exception):
        atom = None

    return atom

#计算两个原子之间的距离
def calculate_distance(atom1, atom2):  # 计算参加中碳原子的距离
    return atom1 - atom2

#在残基之间建边
def build_residue_edges(residues, atom_name='CA', cutoff_distance=6, pock_file=None):  # 在残基之间构建边
    edges = []
    residues_list = [residue for residue in residues]
    for i, res1 in enumerate(residues_list):  # 和特征的下标正好对应
        pos1 = get_atom_position(res1, atom_name)
        if pos1 is None:
            continue
        for j, res2 in enumerate(residues_list[i + 1:], start=i + 1):
            pos2 = get_atom_position(res2, atom_name)
            if pos2 is None:
                continue
            distance = calculate_distance(pos1, pos2)
            if distance < cutoff_distance:
                edges.append([i, j])
    return edges

#去除残基中的氢原子
def remove_pocket_Hs(pok):
    for model in pok:
        for chain in model:
            for residue in chain:
                h_list = [atom for atom in residue if atom.element == 'H']
                for atom in h_list:
                    residue.detach_child(atom.get_id())
    return pok

#建立残基图
def Get_ResidueGraph(pok, pdb_file):  # 建立残基图
    r_features = []  # 存储每个残基的特征
    r_size = len(list(pok.get_residues()))
    r_features = get_residues_feature(pok, pdb_file)
    Pdb = PDBParser(QUIET=True)
    pb = Pdb.get_structure("new_pb", pdb_file)
    edges = build_residue_edges(pb.get_residues(), pock_file=pdb_file)
    g = nt.Graph(edges).to_directed()
    r_edges = []
    for e1, e2 in g.edges:
        r_edges.append([e1, e2])
    return r_size, r_features, r_edges


# 在残基与药分子之间建立边
def build_interaction(mol, pock, atom="CA", distance_threshold=8.0):
    residues = pock.get_residues()
    conformer = mol.GetConformer()  # 获取一个分子构象
    interaction_edges = []
    for i, residue in enumerate(residues):
        pos = get_atom_position(residue, atom)  # 返回CA原子对象
        if pos == None:
            continue
        r_coord = pos.get_coord()  # 获取当前残基的CA原子的坐标
        for m_atom in mol.GetAtoms():  # atoms 也是从0开始索引
            atom_coord = conformer.GetAtomPosition(m_atom.GetIdx())  # 获取当前原子的坐标
            distance = np.linalg.norm(r_coord - atom_coord)
            if distance < distance_threshold:
                interaction_edges.append([i, m_atom.GetIdx()])

    return interaction_edges  # 残基到药分子

error_intersaction=[]
def crate_HGraph(file_list):#创建数据集
    affinity_data = pd.read_csv('./DataSet/' + '/affinity_data.csv')
    pid = list(affinity_data["pdbid"])
    pkd = list(affinity_data["-logKd/Ki"])
    mol_list=[]
    residue_list=[]
    affinity_dict = {key: value for key, value in zip(pid, pkd)}
    for file_path in file_list:
        print("当前处理的数据集为："+file_path)
        mol_files = os.listdir(file_path + '/mol2/')
        print("药分子文件个数：",len(mol_files))
        pocket_files = os.listdir(file_path + '/pocket/')
        print("蛋白质文件文件个数：", len(mol_files))
        process_data_file = file_path.split("/")[2]+".bin"
        PDBreader = PDBParser(QUIET=True)
        ligands=[]
        residues=[]
        i_edges=[]
        lables=[]
        for mol, pok in tqdm(zip(mol_files, pocket_files),
                             total=len(mol_files),desc="提取特征中",mininterval=5.0):
            mol2 = Chem.MolFromMol2File(file_path + 'mol2/' + mol)
            if mol2 is None:  # 判断分子是不是错误类型
                mol2 = Chem.MolFromMol2File(file_path + 'mol2/' + mol,sanitize=False)
            pk = PDBreader.get_structure(pok.split("_")[0], file_path + '/pocket/' + pok)
            c_size, c_features, c_edges = smiles_tograph(mol2)
            r_size, r_features, r_edges = Get_ResidueGraph(pk,file_path + 'pocket/' + pok)
            interaction_edges = build_interaction(mol2, pk)
            if interaction_edges==[]:
                interaction_edges = build_interaction(mol2, pk,distance_threshold=10.0)
                if interaction_edges==[]:
                    error_intersaction.append(mol)
                    continue
            ligands.append([c_size,c_features,c_edges])
            residues.append([r_size,r_features,r_edges])
            i_edges.append(interaction_edges)
            lables.append(affinity_dict[pk.id])
        MyDataSet=MyHeteroDataSet(save_dir=os.path.join('./DataSet03/processed/',process_data_file),
                                  ligands=ligands,residues=residues,i_edges=i_edges,labels=lables,affinity_dict=affinity_dict)
        print(error_intersaction)

files=['E:/PDBBind/train/']
crate_HGraph(files)


