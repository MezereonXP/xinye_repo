import torch
from torch_geometric.nn import GENConv, BatchNorm
from tqdm import tqdm
import numpy as np


def convert(adj_t):
    row, col, edge_attr = adj_t.t().coo()
    edge_index = torch.stack([row, col], dim=0)
    return edge_index, edge_attr


class PNANetNS(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads, dropout, batchnorm=True):
        super(PNANetNS, self).__init__()

        # aggregators = ['mean', 'min', 'max', 'std']
        # scalers = ['identity', 'amplification', 'attenuation']
        
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        # deg = torch.tensor(np.load("/home/kxp/workspace/2022_finvcup_baseline/notebook/degree.npz"))
        for i in range(1, num_layers + 1):
            if i == num_layers:
                hidden_channels = out_channels
            conv = conv = GENConv(in_channels, hidden_channels, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            
            self.convs.append(conv)
            in_channels = hidden_channels

            if i < num_layers:
                self.norms.append(torch.nn.LayerNorm(hidden_channels, elementwise_affine=True))

    def reset_parameters(self):
        for i in range(len(self.convs)):
            self.convs[i].reset_parameters()
        for i in range(len(self.norms)):
            self.norms[i].reset_parameters()

    def forward(self, x, adjs):
        # for conv, norm in zip(self.convs, self.norms):
        #     x = norm(conv(x, edge_index)).relu()
        # return self.convs[-1](x, edge_index).log_softmax(dim=-1)
        for i, (adj, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            
            # print(x.shape)
            # print(x_target.shape)
            # x = self.convs[i]((x, x_target), edge_index)
            edge_index, edge_attr = convert(adj)
            edge_attr = edge_attr.unsqueeze(1).float()
            # print(edge_attr.shape)
            # print(edge_index)
            if i + 1 < len(self.convs):
                x = self.convs[i]((x, x_target), edge_index)
                # x = self.norms[i](x).relu()
                x = torch.nn.functional.gelu(self.norms[i](x))
                # x = torch.nn.functional.gelu(x)
            else:
                x = self.convs[i]((x, x_target), edge_index)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, layer_loader, device, data=None):
        pbar = tqdm(total=x_all.size(0) * self.num_layers, ncols=80)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in layer_loader:
                adj_t, _, size = adj.to(device)
                edge_index, edge_attr = convert(adj_t)
                edge_attr = edge_attr.unsqueeze(1).float()
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                if i + 1 < len(self.convs):
                    x = self.convs[i]((x, x_target), edge_index)
                    # x = self.norms[i](x).relu()
                    x = torch.nn.functional.gelu(self.norms[i](x))
                    # x = torch.nn.functional.gelu(x)
                else:
                    x = self.convs[i]((x, x_target), edge_index)
                del adj
                torch.cuda.empty_cache()
                xs.append(x)
                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all.log_softmax(dim=-1)
