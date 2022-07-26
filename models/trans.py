from typing import Union

from torch import Tensor
from torch_sparse import SparseTensor
import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch.nn import Linear

class TransNet(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , heads=2
                 , batchnorm=True):
        super(TransNet, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            if i < num_layers:
                concat = True
            else:
                concat = False
                hidden_channels = out_channels
            conv = TransformerConv(in_channels, hidden_channels, heads,
                                   concat=concat, beta=True, dropout=dropout)
            self.convs.append(conv)
            in_channels = hidden_channels

            if i < num_layers:
                self.norms.append(torch.nn.LayerNorm(hidden_channels))


    def reset_parameters(self):
        for i in range(len(self.convs)):
            self.convs[i].reset_parameters()
        for i in range(len(self.norms)):
            self.norms[i].reset_parameters()
        

    def forward(self, x, edge_index: Union[Tensor, SparseTensor]):
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x, edge_index)).relu()
        return self.convs[-1](x, edge_index)