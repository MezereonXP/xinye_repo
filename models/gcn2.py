
import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.nn import Linear
from torch_geometric.nn import GCN2Conv, AGNNConv, FiLMConv
from torch_geometric.nn import SuperGATConv, APPNP
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from tqdm import tqdm


def convert(adj_t):
    row, col, edge_attr = adj_t.t().coo()
    edge_index = torch.stack([row, col], dim=0)
    return edge_index, edge_attr

class GCN2Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, alpha=0.1, theta=0.5, shared_weights=True, dropout=0.0, batchnorm=False):
        super(GCN2Net, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GCN2Conv(hidden_channels, alpha, theta, layer + 1, shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, x, adj_t):
        # print(x.shape)
        # x = x.float()
        x = F.dropout(x, p=0.5, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        # print(x.dtype, x_0.dtype)
        # x = F.dropout(x, self.dropout, training=self.training)
        for i, (adj, _, size) in enumerate(adj_t):
            edge_index, edge_attr = convert(adj)
            x = F.dropout(x, p=0.5, training=self.training)
            # x = x_0 = self.lins[0](x).relu()
            x = self.convs[i](x, x_0, edge_index)
            x = F.relu(x)
            # print(x, x_0)
            
        x = self.lins[1](x)
        return x.log_softmax(dim=-1)

    
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
            
    
    def inference_all(self, data):
        x, adj_t = data.x, data.adj_t
        x = x_0 = self.lins[0](x).relu()
        for i, conv in enumerate(self.convs):
            x = conv(x, x_0, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.convs[-1](x, adj_t)
        x = self.lins[1](x)
        return x.log_softmax(dim=-1)
    
    
    def inference(self, x_all, layer_loader, device):
        pbar = tqdm(total=x_all.size(0) * self.num_layers, ncols=80)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        xs_ori = []
        for i in range(self.num_layers):
            xs = []
            for idx, (batch_size, n_id, adj) in enumerate(layer_loader):
                # edge_index, _, size = adj.to(device)
                adj_t, _, size = adj.to(device)
                edge_index, edge_attr = convert(adj_t)
                x = x_all[n_id].to(device)
                # x_target = x[:size[1]]
                if i == 0:
                    x = x_0 = self.lins[0](x).relu()
                    xs_ori.append(x_0.detach().cpu())
                
                x = self.convs[i](x, xs_ori[idx].to(device), edge_index.float())
                x = F.relu(x)
                if i == (self.num_layers - 1):
                    x = self.lins[1](x)
                xs.append(x.detach().cpu())
                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)
        # print(x_all.shape)
        pbar.close()
        del xs, xs_ori
        torch.cuda.empty_cache()
        return x_all.log_softmax(dim=-1).to(device)