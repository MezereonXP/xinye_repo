from typing import Union

from torch import Tensor
from torch_sparse import SparseTensor
import torch
import torch.nn.functional as F
from torch_geometric.nn import APPNP
from torch.nn import Linear

class APPNPNet(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , batchnorm=True
                 , K=10
                 , alpha=0.1):
        super(APPNPNet, self).__init__()

        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        
        self.appnp = APPNP(K, alpha, dropout=dropout)
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index: Union[Tensor, SparseTensor]):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.appnp(x, edge_index)
        return F.log_softmax(x, dim=1)