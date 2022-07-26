import torch
from torch_geometric.nn import TransformerConv
from tqdm import tqdm
import torch.nn.functional as F


def convert(adj_t):
    row, col, edge_attr = adj_t.t().coo()
    edge_index = torch.stack([row, col], dim=0)
    return edge_index, edge_attr


class TransNetNSV2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, heads=2, batchnorm=True):
        super(TransNetNSV2, self).__init__()

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            if i < num_layers:
                concat = True
            else:
                concat = False
                hidden_channels = out_channels
            conv = TransformerConv(in_channels, hidden_channels, heads,
                                   concat=concat, beta=True, dropout=dropout, edge_dim=23)
            self.convs.append(conv)
            in_channels = hidden_channels*heads

            if i < num_layers:
                self.norms.append(torch.nn.LayerNorm(hidden_channels*heads))

    def reset_parameters(self):
        for i in range(len(self.convs)):
            self.convs[i].reset_parameters()
        for i in range(len(self.norms)):
            self.norms[i].reset_parameters()

    def forward(self, x, adjs):
        # for conv, norm in zip(self.convs, self.norms):
        #     x = norm(conv(x, edge_index)).relu()
        # return self.convs[-1](x, edge_index).log_softmax(dim=-1)
        for i, (adj, eid, size) in enumerate(adjs):
            x_target = x[:size[1]]
            edge_index, edge_attr = convert(adj)
            edge_attr_vector = F.one_hot(eid[:, 0].long(), num_classes=23).float()
            # print(edge_attr_vector)
            if i + 1 < len(self.convs):
                x = self.convs[i]((x, x_target), edge_index, edge_attr_vector)
                x = self.norms[i](x).relu()
            else:
                x = self.convs[i]((x, x_target), edge_index, edge_attr_vector)
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
                adj_t, eid, size = adj.to(device)
                edge_index, edge_attr = convert(adj_t)
                eid_v = data.edge_attr[eid][:, 0]
                # edge_attr = edge_attr.unsqueeze(1).float()
                edge_attr_vector = F.one_hot(eid_v.long(), num_classes=23).float()
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                if i + 1 < len(self.convs):
                    x = self.convs[i]((x, x_target), edge_index, edge_attr_vector)
                    x = self.norms[i](x).relu()
                else:
                    x = self.convs[i]((x, x_target), edge_index, edge_attr_vector)
                xs.append(x)
                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all.log_softmax(dim=-1)
