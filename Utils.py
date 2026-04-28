# @Version : 3.7
# @Author : 孔渝翔
# @Time  ：2024/10/25 9:14
# @FName  :Utils.py
import os

import torch
import dgl
from torch.nn import Module
import numpy as np
import sklearn.metrics as m
from scipy.stats import pearsonr

from numba import njit
import matplotlib.pyplot as plt  # 导入 matplotlib 绘图库
import numpy as np  # 导入 numpy 用于数值计算
from matplotlib.colors import LinearSegmentedColormap  # 导入颜色映射工具

class LoadedHeteroDataSet:
    def __init__(self, file_path):
        self.graphs, labels = dgl.load_graphs(file_path)
        self.labels = labels['labels'] if 'labels' in labels else None

    def __getitem__(self, index):
        return self.graphs[index], self.labels[index]

    def __len__(self):
        return len(self.graphs)

def custom_collate_fn(batch):
    graphs, labels = zip(*batch)
    return graphs, labels


three_to_one_map = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

list_residue=[i for i in three_to_one_map.keys()]
list_residue.append("Unknow")

def activation_relu(dict,apply_relu):
     for key,value in dict.items():
         dict[key]=apply_relu(value)
     return dict


def feat_cat(out,key):
    feat=list(out[key])
    print(feat)
    t=tuple()
    # for tensor in enumerate(feat):
    #     view=tensor.view(1,-1)
    #     feat.append(view)
    # feat=torch.cat(feat,dim=1)
    return feat

# def get_Protein_seq(PDB_file):
#     PDBreader=PDBParser(QUIET=True) #创建PDB对象文件
#     id=PDB_file.split("_")[0] #对PDB中的header的id进行索引
#     ps=PDBreader.get_structure(id,PDB_file) #创建PDB文件对象
#     seq="" #存储蛋白质序列
#      #标准残基序列的字典
#     for model in ps: #遍历PDBStructure
#         for chains in model: #遍历出蛋白质链
#             for residues in chains: #遍历出链中的残基
#                 if residues.get_resname() in three_to_one_map: #判断当前的残基是否在当前标准的字典中
#                     temp=three_to_one_map[residues.get_resname()]
#                     #three to one 编码
#                     seq+=temp #累加当前的序列
#
#
#     return seq #返回蛋白质特征序列
#
#
# def get_Smiles(Mol2_file):
#     smiles=Chem.MolFromMol2File(Mol2_file,sanitize=True,removeHs=True,
#                                 cleanupSubstructures=True)
#     smiles=Chem.MolToSmiles(smiles)
#     return  smiles

def draw_p(categories,savepath):
    # 定义红蓝配色方案（从蓝到红渐变，中间为深灰色）
    colors = [
        "#0B6EED",  # 蓝色 (value 0)
        "#0B6EED",  # 新增紫色过渡
        "#BB00A1",  # 原第三色
        "#E6007F",  # 原第二色
        "#FF0054"  # 玫红色 (value 1)
    ]
    # 创建线性分段颜色映射
    cmap = LinearSegmentedColormap.from_list("red_blue_deep", colors)

    # 数据准备：将每个类别的特征掩码数据拼接成数组

    # 设置 matplotlib 的默认样式
    plt.style.use('default')
    # 更新全局参数，配置字体、字号、画布尺寸等
    plt.rcParams.update({
        'font.family': 'Arial',  # 使用 Arial 字体
        'axes.labelsize': 9,  # 坐标轴标签字号
        'xtick.labelsize': 8,  # x 轴刻度标签字号
        'ytick.labelsize': 8,  # y 轴刻度标签字号
        'figure.dpi': 300,  # 图像分辨率
        'figure.figsize': (4, 4),  # 画布尺寸（宽，高）
        'axes.spines.top': False,  # 隐藏顶部轴线
        'axes.spines.right': False,  # 隐藏右侧轴线
        'axes.linewidth': 0.5  # 轴线宽度
    })

    # 创建画布和网格布局
    fig = plt.figure()  # 创建一个新的画布
    # 定义网格布局，5 行 2 列，左侧占 96%，右侧占 4%，子图间无垂直间距
    gs = fig.add_gridspec(5, 2, width_ratios=[0.96, 0.04], hspace=0)
    # 创建左侧 5 个子图
    axes = [fig.add_subplot(gs[i, 0]) for i in range(5)]
    # 创建右侧共享的颜色条区域
    cax = fig.add_subplot(gs[:, 1])

    # 定义全局绘图参数
    plot_params = {
        's': 6,  # 散点大小
        'alpha': 0.9,  # 透明度
        'edgecolors': '#FFFFFF',  # 散点边缘颜色（白色）
        'linewidths': 0.1  # 散点边缘线宽
    }

    # 遍历每个类别，绘制子图
    for ax, (label, data) in zip(axes, categories):
        # 生成 y 轴扰动，使散点分布更清晰
        y = np.random.normal(0, 0.025, len(data))  # 均值为 0，标准差为 0.025 的正态分布

        # 绘制散点图
        scatter = ax.scatter(
            data, y,  # x 和 y 数据
            c=data,  # 颜色基于数据值
            cmap=cmap,  # 使用定义的颜色映射
            vmin=0, vmax=1,  # 颜色映射范围
            **plot_params  # 应用全局绘图参数
        )

        # 设置 y 轴标签
        ax.set_ylabel(label,
                      rotation=0,  # 标签不旋转
                      fontsize=9,  # 标签字号
                      labelpad=4,  # 标签与轴线的间距
                      va='center',  # 垂直居中
                      ha='right',  # 水平右对齐
                      y=0.45)  # 标签在 y 轴上的位置

        # 优化刻度系统
        ax.set_yticks([])  # 隐藏 y 轴刻度
        ax.set_ylim(-0.15, 0.15)  # 设置 y 轴范围
        ax.tick_params(axis='x', length=0)  # 隐藏 x 轴刻度线

    # 设置全局 x 轴属性
    for ax in axes:
        ax.set_xticks(np.linspace(0, 1, 11))  # x 轴刻度从 0 到 1，间隔 0.1
        ax.axvline(0.5, color='#2F4F4F', ls='--', lw=0.8, zorder=0)  # 添加中线
        ax.spines['bottom'].set_position(('outward', 0))  # 底部轴线外移 2pt

    # 只在最下面的子图显示 x 轴刻度
    for ax in axes[:-1]:
        ax.tick_params(axis='x', length=0, labelbottom=False)  # 隐藏 x 轴刻度和标签
    axes[-1].tick_params(axis='x', length=3, pad=2)  # 显示 x 轴刻度

    # 配置颜色条
    cbar = plt.colorbar(scatter, cax=cax, ticks=np.linspace(0, 1, 6))  # 设置颜色条刻度
    cbar.set_label('Mask Value',  # 颜色条标签
                   rotation=270,  # 标签旋转 270 度
                   labelpad=14,  # 标签与颜色条的间距
                   fontsize=9)  # 标签字号
    cbar.outline.set_visible(False)  # 隐藏颜色条边框
    cbar.ax.tick_params(width=0.5, length=3)  # 设置颜色条刻度线的宽度和长度

    # 消除子图间隔
    for ax in axes[:-1]:  # 遍历前 4 个子图
        ax.spines['bottom'].set_visible(False)  # 隐藏底部轴线
    axes[-1].spines['bottom'].set_visible(True)  # 显示最后一个子图的底部轴线

    # 增加左侧公共 y 轴刻度
    fig.text(0.1, 0.5, '',  # 公共 y 轴标签
             rotation=90,  # 旋转 90 度
             va='center',  # 垂直居中
             ha='center',  # 水平居中
             fontsize=9)  # 标签字号

    # 优化全局布局
    plt.xlim(0, 1)  # 设置 x 轴范围
    # 调整子图间距，确保布局紧凑
    plt.subplots_adjust(left=0.15, right=0.75, top=0.75, bottom=0.25)

    plt.savefig(savepath,
                transparent=True,  # 背景透明
                bbox_inches='tight',  # 紧凑布局
                pad_inches=0.01)  # 内边距

    plt.show()  # 显示图像


class HeteroMaxPooling(Module):
    def __init__(self,):
        super(HeteroMaxPooling, self).__init__()

    def forward(self, g, input_feats):
        pooled_feats = {}
        for ntype in g.ntypes:
                feats = input_feats[ntype]
                pooled_feats[ntype] = torch.max(feats, dim=0).values
        return pooled_feats


list_set=['test2013','test2016','val']
def writer_result(dicts,file_path,seed=None):
    with open(file_path,'w') as f:
        f.write(f"seed:{seed}\n")
        for dict,str in zip(dicts,list_set):
            f.write(f"{str}\n")
            for k,v in dict.items():
                f.write(f"{k}:{v}\n")
            f.write("\n")
            f.write("\n")

def writer(path,residue,ligand,bond,rr,rl,dict):
    with open(path,'w') as f:
        f.write(f"residue:{residue}\n")
        f.write(f"ligand:{ligand}\n")
        f.write(f"bond :{bond}\n")
        f.write(f"rr:{rr}\n")
        f.write(f"rl:{rl}\n")
        for k,v in dict.items():
            f.write(f"{k}:{v}\n")


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
    return np.sqrt(m.mean_squared_error(y_true, y_pred))


def MAE(y_true, y_pred):
    return m.mean_absolute_error(y_true, y_pred)


def CORR(y_true, y_pred):
   return  pearsonr(y_true,y_pred)[0]


def SD(y_true, y_pred):
    from sklearn.linear_model import LinearRegression
    y_pred = y_pred.reshape((-1,1))
    lr = LinearRegression().fit(y_pred,y_true)
    y_ = lr.predict(y_pred)
    return np.sqrt(np.square(y_true - y_).sum() / (len(y_pred) - 1))


def auc(y_true,y_pred):
    threshold=np.mean(y_true)
    y_true_bi=(y_true>threshold).astype(int)
    y_pred_bi=(y_pred>threshold).astype(int)
    auc=roc_auc_score(y_true_bi,y_pred_bi)
    return auc




