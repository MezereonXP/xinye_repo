# dataset name: XYGraphP1

from utils import XYGraphP1
from utils.utils import prepare_folder
from models import MLP, MLPLinear, GCN, SAGE, GAT, GATv2, APPNPNet, GCN2ConvEnsemble, GCN2Net, SageEnsemble, PinSAGE, TransNet

import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import pandas as pd
from torch_geometric.utils import degree
import numpy as np

from sklearn.metrics import confusion_matrix


class FocalLoss(nn.Module):

    def __init__(self, gamma=20.0, alpha=0.99, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

mlp_parameters = {'lr':0.01
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.0
              , 'batchnorm': False
              , 'l2':5e-7
             }

gcn_parameters = {'lr':0.01
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.0
              , 'batchnorm': False
              , 'l2':5e-7
             }

sage_parameters = {'lr':0.001
              , 'num_layers':2
              , 'hidden_channels':64
              , 'dropout':0.1
              , 'batchnorm': False
              , 'l2':5e-7
             }
sage_ensemble_parameters = {'lr':0.01
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.
              , 'batchnorm': False
              , 'l2':5e-7
             }


appnp_parameters = {'lr':1e-3
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.1
              , 'batchnorm': False
              , 'l2':5e-7
              , 'K': 10
              , 'alpha': 0.1
             }
gcn2ensemble_parameters = {'lr':1e-3
              , 'num_layers':3
              , 'hidden_channels':128
              , 'dropout':0
              , 'batchnorm': False
              , 'l2':5e-7
             }
gcn2_parameters = {'lr':1e-2
              , 'num_layers':3
              , 'hidden_channels':128
              , 'dropout':0
              , 'batchnorm': False
              , 'l2':5e-7
             }

trans_parameters = {'lr':1e-2
              , 'num_layers':2
              , 'hidden_channels':64
              , 'dropout':0
              , 'batchnorm': False
              , 'l2':5e-7
              , 'heads':2
             }

spline_parameters = {'lr':1e-2
              , 'num_layers':3
              , 'hidden_channels':128
              , 'dropout':0
              , 'batchnorm': False
              , 'l2':5e-7
             }


def shuffle_minibatch(inputs, targets, device):
    mixup_alpha = 0.1

    batch_size = inputs.shape[0]
    rp1 = torch.randperm(batch_size)
    inputs1 = inputs[rp1]
    targets1 = targets[rp1]

    rp2 = torch.randperm(batch_size)
    inputs2 = inputs[rp2]
    targets2 = targets[rp2]

    ma = np.random.beta(mixup_alpha, mixup_alpha, [batch_size])
    ma_img = ma[:, None]

    inputs1 = inputs1 * torch.from_numpy(ma_img).to(device).float()
    inputs2 = inputs2 * torch.from_numpy(1 - ma_img).to(device).float()

    targets1 = targets1.float() * torch.from_numpy(ma).to(device).float()
    targets2 = targets2.float() * torch.from_numpy(1 - ma).to(device).float()

    inputs_shuffle = (inputs1 + inputs2).to(device)
    targets_shuffle = (targets1 + targets2).to(device)

    return inputs_shuffle, targets_shuffle.round().long()


def train(model, data, train_idx, optimizer, no_conv=False, device=None):
    # data.y is labels of shape (N, ) 
    model.train()
    
    tmp_x, tmp_y = data.x, data.y
    # tmp_x[train_idx], tmp_y[train_idx] = shuffle_minibatch(tmp_x[train_idx], tmp_y[train_idx], device)
    # tmp_x, tmp_y = shuffle_minibatch(data.x[train_idx], data.y)
    optimizer.zero_grad()
    if no_conv:
        out = model(data.x[train_idx])
    else:
        # out = model(data.x, data.adj_t)[train_idx]
        out = model(tmp_x, data.adj_t)[train_idx]
    # print(data.adj_t)
    # out = model(data.x, data.adj_t, data.edge_attr)
        
    # loss = F.nll_loss(out, data.y[train_idx])
    loss = F.nll_loss(out, tmp_y[train_idx], weight=torch.tensor([0.01, 0.99]).cuda())
    # loss = FocalLoss()(out, tmp_y[train_idx])
    # loss = F.nll_loss(out, tmp_y[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, no_conv=False):
    # data.y is labels of shape (N, )
    model.eval()
    
    if no_conv:
        out = model(data.x)
    else:
        out = model(data.x, data.adj_t)
    # out = model(data.x, data.adj_t, data.edge_attr)
        
    y_pred = out.exp()  # (N,num_classes)
    losses = dict()
    for key in ['train', 'valid', 'test']:
        node_id = split_idx[key]
        losses[key] = F.nll_loss(out[node_id], data.y[node_id]).item()
        if key == 'valid':
            # print(out[node_id].argmax(1))
            print(confusion_matrix(data.y[node_id].cpu().numpy(), out[node_id].argmax(1).cpu().numpy()))
            
    return losses, y_pred
        
            
def main():
    parser = argparse.ArgumentParser(description='gnn_models')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='XYGraphP1')
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--epochs', type=int, default=200)
    
    args = parser.parse_args()
    print(args)
    
    no_conv = False
    if args.model in ['mlp']: no_conv = True        
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = XYGraphP1(root='./', name='xydata', transform=T.Compose([T.ToSparseTensor()]))
    
    nlabels = dataset.num_classes
    if args.dataset in ['XYGraphP1']: nlabels = 2
        
    data = dataset[0]
    # data.x = torch.tensor(np.load('./fixed_features.npy'))
    data.x = torch.cat([torch.tensor(np.load('./fixed_features.npy')), data.x[:, -1].unsqueeze(1)], dim=-1)
    
    # data=T.AddSelfLoops()(data)
    data.adj_t = data.adj_t.to_symmetric()
        
    if args.dataset in ['XYGraphP1']:
        x = data.x
        x = (x-x.mean(0))/x.std(0)
        data.x = x
        # data.y[data.y > 1] = 0
    if data.y.dim()==2:
        data.y = data.y.squeeze(1)        
    
    split_idx = {'train':data.train_mask, 'valid':data.valid_mask, 'test':data.test_mask}
        
    data = data.to(device)
    train_idx = split_idx['train'].to(device)
        
    model_dir = prepare_folder(args.dataset, args.model)
    print('model_dir:', model_dir)
        
    if args.model == 'mlp':
        para_dict = mlp_parameters
        model_para = mlp_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = MLP(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
    if args.model == 'gcn':   
        para_dict = gcn_parameters
        model_para = gcn_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')        
        model = GCN(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
    if args.model == 'sage':        
        para_dict = sage_parameters
        model_para = sage_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')        
        model = SAGE(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
    if args.model == 'sage-ensemble':        
        para_dict = sage_ensemble_parameters
        model_para = sage_ensemble_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')        
        model = SageEnsemble(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
    if args.model == 'appnp':
        para_dict = appnp_parameters
        model_para = appnp_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')        
        model = APPNPNet(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
    if args.model == 'gcn2ensemble':
        para_dict = gcn2ensemble_parameters
        model_para = gcn2ensemble_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')        
        model = GCN2ConvEnsemble(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
    
    if args.model == 'gcn2':
        para_dict = gcn2_parameters
        model_para = gcn2_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')        
        model = GCN2Net(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
        
    if args.model == 'pinsage':
        
        # Compute the maximum in-degree in the training data.
        max_degree = -1
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

        # Compute the in-degree histogram tensor
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
        
        para_dict = spline_parameters
        model_para = spline_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')        
        model = PinSAGE(in_channels = data.x.size(-1), out_channels = nlabels, deg=deg, **model_para).to(device)
    
    if args.model == 'trans':
        para_dict = trans_parameters
        model_para = trans_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')        
        model = TransNet(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
        
    

    print(f'Model {args.model} initialized')

    print(sum(p.numel() for p in model.parameters()))

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
    # optimizer = torch.optim.SGD(model.parameters(), lr=para_dict['lr'])
    min_valid_loss = 1e8

    for epoch in range(1, args.epochs+1):
        loss = train(model, data, train_idx, optimizer, no_conv, device)
        losses, out = test(model, data, split_idx, no_conv)
        train_loss, valid_loss, test_loss = losses['train'], losses['valid'], losses['test']

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), model_dir+'model.pt')

        if epoch % args.log_steps == 0:
            print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_loss:.3f}%, '
                      f'Valid: {100 * valid_loss:.3f}% '
                      f'Test: {100 * test_loss:.3f}%')


if __name__ == "__main__":
    main()
