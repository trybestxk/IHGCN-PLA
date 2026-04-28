import torch
from torch import nn
from dgl import khop_in_subgraph, NID
from tqdm import tqdm
from math import sqrt

class HeteroGNNExplainer(nn.Module):
    def __init__(self,
                 model,
                 num_hops,
                 lr=0.01,
                 num_epochs=100,
                 *,
                 alpha1=0.005,
                 alpha2=1.0,
                 beta1=1.0,
                 beta2=0.1,
                 log=True):
        super(HeteroGNNExplainer, self).__init__()
        self.model = model
        self.num_hops = num_hops
        self.lr = lr
        self.num_epochs = num_epochs
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.log = log

    def _init_masks(self, graph, feat):
        device = graph.device
        feat_masks = {}
        std = 0.1
        for node_type, feature in feat.items():
            _, feat_size = feature.size()
            feat_masks[node_type] = nn.Parameter(torch.randn(1, feat_size, device=device) * std)

        edge_masks = {}
        for canonical_etype in graph.canonical_etypes:
            src_num_nodes = graph.num_nodes(canonical_etype[0])
            dst_num_nodes = graph.num_nodes(canonical_etype[-1])
            num_nodes_sum = src_num_nodes + dst_num_nodes
            num_edges = graph.num_edges(canonical_etype)
            std = nn.init.calculate_gain('relu')
            if num_nodes_sum > 0:
                std *= sqrt(2.0 / num_nodes_sum)
            edge_masks[canonical_etype] = nn.Parameter(torch.randn(num_edges, device=device) * std)

        return feat_masks, edge_masks

    def _loss_regularize(self, loss, feat_masks, edge_masks):
        eps = 1e-15

        for edge_mask in edge_masks.values():
            edge_mask = edge_mask.sigmoid()
            loss = loss + self.alpha1 * torch.sum(edge_mask)
            ent = - edge_mask * torch.log(edge_mask + eps) - (1 - edge_mask) * torch.log(1 - edge_mask + eps)
            loss = loss + self.alpha2 * ent.mean()

        for feat_mask in feat_masks.values():
            feat_mask = feat_mask.sigmoid()
            loss = loss + self.beta1 * torch.mean(feat_mask)
            ent = - feat_mask * torch.log(feat_mask + eps) - (1 - feat_mask) * torch.log(1 - feat_mask + eps)
            loss = loss + self.beta2 * ent.mean()

        return loss

    def explain_node(self, ntype, node_id, graph, feat, target, **kwargs):
        self.model = self.model.to(graph.device)
        self.model.eval()

        # 获取邻居节点子图（k-hop邻居）
        sg, inverse_indices = khop_in_subgraph(graph, {ntype: node_id}, self.num_hops)
        inverse_indices = inverse_indices[ntype]
        sg_nodes = sg.ndata[NID]

        # 获取子图中的特征
        sg_feat = {node_type: feat[node_type][sg_nodes[node_type].long()] for node_type in sg_nodes.keys()}

        # 获取初始预测值（回归任务）
        with torch.no_grad():
            model_output = self.model(graph=sg, feat=sg_feat, **kwargs)

            # 如果模型输出是一个张量，则直接使用
            logits = model_output.squeeze(dim=-1)  # 对于回归任务，去掉多余的维度
            pred_value = logits  # 可以保存预测值用于进一步分析或日志

        # 初始化特征掩码和边掩码
        feat_mask, edge_mask = self._init_masks(sg, sg_feat)

        # 准备优化器
        params = [*feat_mask.values(), *edge_mask.values()]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        # 使用进度条（如果需要日志）
        if self.log:
            pbar = tqdm(total=self.num_epochs)
            pbar.set_description(f'解释节点 {node_id}，类型 {ntype}')

        # 训练过程
        for _ in range(self.num_epochs):
            optimizer.zero_grad()

            # 计算特征掩码
            h = {}
            for node_type, sg_node_feat in sg_feat.items():
                h[node_type] = sg_node_feat * feat_mask[node_type].sigmoid()

            # 计算边掩码
            eweight = {}
            for canonical_etype, canonical_etype_mask in edge_mask.items():
                eweight[canonical_etype] = canonical_etype_mask.sigmoid()

            # 计算模型输出
            model_output = self.model(graph=sg, feat=h, eweight=eweight, **kwargs)

            # 如果模型输出是一个张量，则直接使用
            logits = model_output.squeeze(dim=-1)

            # 假设这是回归任务，计算均方误差损失
            loss = nn.MSELoss()(logits, pred_value)
            loss = self._loss_regularize(loss, feat_mask, edge_mask)
            loss.backward()
            optimizer.step()

            # 更新进度条
            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()
        print('train mask',logits,pred_value)
        # 将掩码转换为 sigmoid 值（介于 0 和 1 之间）
        for node_type in feat_mask:
            feat_mask[node_type] = feat_mask[node_type].detach().sigmoid().squeeze()

        for canonical_etype in edge_mask:
            edge_mask[canonical_etype] = edge_mask[canonical_etype].detach().sigmoid()

        return inverse_indices, sg, feat_mask, edge_mask

    def explain_graph(self, graph, feat, target, **kwargs):
        self.model = self.model.to(graph.device)
        self.model.eval()

        # Get the initial prediction for regression task
        with torch.no_grad():
            # Assuming the model returns a tuple, we extract the first element (logits)
            logits=self.model(graph=graph, feat=feat, **kwargs)
            pred_value = logits

        feat_mask, edge_mask = self._init_masks(graph, feat)

        params = [*feat_mask.values(), *edge_mask.values()]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        if self.log:
            pbar = tqdm(total=self.num_epochs)
            pbar.set_description('Explain graph')
        for _ in range(self.num_epochs):
            optimizer.zero_grad()
            h = {}
            for node_type, node_feat in feat.items():
                h[node_type] = node_feat * feat_mask[node_type].sigmoid()

            eweight = {}
            for canonical_etype, canonical_etype_mask in edge_mask.items():
                eweight[canonical_etype] = canonical_etype_mask.sigmoid()

            logits = self.model(graph=graph, feat=h, eweight=eweight, **kwargs)

            loss = nn.MSELoss()(logits, pred_value)
            loss = self._loss_regularize(loss, feat_mask, edge_mask)
            loss.backward()
            optimizer.step()

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()

        for node_type in feat_mask:
            feat_mask[node_type] = feat_mask[node_type].detach().sigmoid().squeeze()

        for canonical_etype in edge_mask:
            edge_mask[canonical_etype] = edge_mask[canonical_etype].detach().sigmoid()

        return feat_mask, edge_mask,pred_value,logits

