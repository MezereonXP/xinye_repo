import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.nn import Linear
from torch_geometric.nn import GCN2Conv, AGNNConv, FiLMConv, PNAConv
from torch_geometric.nn import SuperGATConv, APPNP
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Embedding, ModuleList
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import BatchNorm


#TODO Add deg
class PinSAGE(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , deg
                 , num_layers
                 , dropout
                 , batchnorm=True):
        super(PinSAGE, self).__init__()
        self.node_emb = Embedding(in_channels, hidden_channels)
        self.edge_emb = Embedding(1, 50)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
            conv = PNAConv(in_channels=75, out_channels=75,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=50, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(75))

        self.mlp = Sequential(Linear(hidden_channels, 50), ReLU(), Linear(50, 25), ReLU(),
                              Linear(25, out_channels))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x.squeeze())
        edge_attr = self.edge_emb(edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_add_pool(x, batch)
        return self.mlp(x)