import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.nn import Linear
from torch_geometric.nn import GCN2Conv, AGNNConv, FiLMConv
from torch_geometric.nn import SuperGATConv, APPNP
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.nn import GINConv, global_add_pool



# GCN2ConvEnsemble
class GCN2ConvEnsemble(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0., batchnorm=True):
        super().__init__()
        n = 3
        self.n = n
        self.nets = torch.nn.ModuleList()
        for i in range(n):
            self.nets.append(
                Net(in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, alpha=0.1, theta=0.5, shared_weights=True, dropout=dropout))

        #self.fc = torch.nn.Linear(n, 1)

    def forward(self, x, adj):
        out = [self.nets[0](x, adj).unsqueeze(0)]
        for i in range(1, self.n):
            out.append(self.nets[i](x, adj).unsqueeze(0))
        out = torch.cat(out)
        #out = self.fc(out.reshape(-1, self.n)).reshape(-1, 2)
        out = torch.mean(out, dim=0)
        return out
    
    def reset_parameters(self):
        for net in self.nets:
            net.reset_parameters()



class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, alpha, theta, shared_weights=True, dropout=0.0):
        super(Net, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GCN2Conv(hidden_channels, alpha, theta, layer + 1, shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, x, adjs):
        for i, (adj_t, _, size) in enumerate(adjs):
            # x_target = x[:size[1]]
            x = F.dropout(x, self.dropout, training=self.training)
            x = x_0 = self.lins[0](x).relu()

            for conv in self.convs:
                x = F.dropout(x, self.dropout, training=self.training)
                x = conv(x, x_0, adj_t)
                x = x.relu()

            x = F.dropout(x, self.dropout, training=self.training)
            x = self.lins[1](x)

        return x.log_softmax(dim=-1)
    
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        