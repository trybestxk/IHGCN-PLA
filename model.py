# @Version : 3.7
# @Author : 孔渝翔
# @Time  ：2024/11/4 21:07
# @FName  :model.py
import dgl
import dgl.nn.pytorch as dglnn
import torch.nn
from torch.nn import Module
import torch.nn as t
from Utils import activation_relu as Relu_
import  pandas as pd

def my_agg_func(tensors, dsttype):
    global i
    if dsttype == 'residue':
        tensors[1]=torch.sum(tensors[1],dim=1)
        RUle=torch.nn.ReLU()
        tensors[0]=RUle(tensors[0])
    stacked = torch.stack(tensors, dim=0)
    return torch.sum(stacked,dim=0)

class HGCNDTA(Module):
    def __init__(self, in_feat_ligand=108,
                 in_feat_residue=108, ):
        super().__init__()
        self.xr01 = torch.nn.Sequential(
            torch.nn.Linear(in_features=108,out_features=108),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=108,out_features=108),
        )
        self.xr02 = torch.nn.Sequential(
            torch.nn.Linear(in_features=108, out_features=108*2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=108*2, out_features=108*2),
        )

        self.xr03 = torch.nn.Sequential(
            torch.nn.Linear(in_features=108*2, out_features=108*4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=108*4, out_features=108*4),
        )

        self.fb1=torch.nn.BatchNorm1d(108)
        self.fb2=torch.nn.BatchNorm1d(108*2)
        self.fb3=torch.nn.BatchNorm1d(108*4)
        self.rb1 = torch.nn.BatchNorm1d(108)
        self.rb2 = torch.nn.BatchNorm1d(108*2)
        self.rb3 = torch.nn.BatchNorm1d(108*4)

        self.hconv01 = dglnn.HeteroGraphConv({
            'bond': dglnn.GINConv(apply_func=self.xr01, aggregator_type='sum'),
            'r_r_interation': dglnn.GATConv(in_feats=108, out_feats=108 , num_heads=8),
            'l_r_interaction': dglnn.SAGEConv(in_feats=108, out_feats=108,
                                              aggregator_type='lstm')
            # 1280 2160
        }, aggregate=my_agg_func)

        self.hconv02 = dglnn.HeteroGraphConv({
            'bond': dglnn.GINConv(apply_func=self.xr02,aggregator_type='sum'),
            'r_r_interation': dglnn.GATConv(in_feats=108, out_feats=108*2,num_heads=8),
            'l_r_interaction': dglnn.SAGEConv(in_feats=108, out_feats=108*2,
                                              aggregator_type='lstm')
            # 1280 2160
        }, aggregate=my_agg_func)

        self.hconv03 = dglnn.HeteroGraphConv({
            'bond': dglnn.GINConv(apply_func=self.xr03,aggregator_type='sum'), # 108*2*4*2
            'r_r_interation':  dglnn.GATConv(in_feats=108*2, out_feats=108*4,num_heads=8),
            'l_r_interaction': dglnn.SAGEConv(in_feats=108*2, out_feats=108*4,
                                              aggregator_type='lstm')
        }, aggregate=my_agg_func)

        self.fc_l01 = torch.nn.Linear(in_features=in_feat_ligand * 4, out_features=1024)
        self.fc_l02 = torch.nn.Linear(in_features=1024, out_features=128)

        self.fc_r01 = torch.nn.Linear(in_features=432, out_features=1024)
        self.fc_r02 = torch.nn.Linear(in_features=1024, out_features=128)

        # 全连接
        self.fc01 = torch.nn.Linear(in_features=108*4*2, out_features=512)
        self.fc02 = torch.nn.Linear(in_features=512, out_features=256)
        self.fc03 = torch.nn.Linear(in_features=256, out_features=1)
        self.Relu = t.ReLU()
        self.Drop = t.Dropout(p=0.1)

    def forward(self, graph, feat,eweight=None):
        g=graph
        h=feat
        x = self.hconv01(g, h)
        x = Relu_(x, self.Relu)
        x['ligand_atom']=self.fb1(x['ligand_atom'])
        x['residue']=self.rb1(x['residue'])


        x = self.hconv02(g, x)
        x = Relu_(x, self.Relu)
        x['ligand_atom'] = self.fb2(x['ligand_atom'])
        x['residue'] = self.rb2(x['residue'])

        x = self.hconv03(g, x)
        x = Relu_(x, self.Relu)
        x['ligand_atom'] = self.fb3(x['ligand_atom'])
        x['residue'] = self.rb3(x['residue'])

        g.nodes['ligand_atom'].data['feat'] = x['ligand_atom']
        g.nodes['residue'].data['feat'] = x['residue']
        ligand_feat = dgl.readout_nodes(g, 'feat', op='max', ntype='ligand_atom')
        residue_feat = dgl.readout_nodes(g, 'feat', op='max', ntype='residue')


        conbine = torch.cat((ligand_feat, residue_feat), dim=1)
        x3 = self.fc01(conbine)
        x3 = self.Relu(x3)
        x3 = self.Drop(x3)
        x3 = self.fc02(x3)
        x3 = self.Relu(x3)
        x3 = self.Drop(x3)

        x3 = self.fc03(x3)

        return x3
