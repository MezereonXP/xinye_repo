# dataset name: XYGraphP1

from numpy import ediff1d
from models.sage_neighsampler import SAGE_NeighSamplerEnsemble
from models.trans_neighsampler_v3 import TransNetNSV3
from utils import XYGraphP1
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from torch_geometric.loader import NeighborSampler
# from torch_geometric.loader import ImbalancedSampler, NeighborLoader
from models import SAGE_NeighSampler, GAT_NeighSampler, GATv2_NeighSampler, GCN2ConvEnsemble, GCN2Net, TransNetNS, TransNetNSV2, PNANetNS
from tqdm import tqdm

import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import pandas as pd

from sklearn.metrics import confusion_matrix
import numpy as np


class FocalLoss(nn.Module):

    def __init__(self, gamma=10.0, alpha=0.99, size_average=False):
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

        # logpt = F.log_softmax(input, dim=1)
        logpt = input
        # 获取target对应的0/1分类置信度
        logpt = logpt.gather(1, target)
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
        

eval_metric = 'auc'

sage_neighsampler_parameters = {'lr':0.003
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.
              , 'batchnorm': False
              , 'l2':5e-7
             }

gat_neighsampler_parameters = {'lr':0.003
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.0
              , 'batchnorm': False
              , 'l2':5e-7
             , 'layer_heads':[4,1]
             }

gatv2_neighsampler_parameters = {'lr':0.003
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.1
              , 'batchnorm': False
              , 'l2':5e-6
             , 'layer_heads':[4,4]
             }

gcn2ensemble_parameters = {'lr':1e-3
              , 'num_layers':3
              , 'hidden_channels':128
              , 'dropout':0
              , 'batchnorm': False
              , 'l2':5e-7
             }
gcn2_parameters = {'lr':1e-2
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0
              , 'batchnorm': False
              , 'l2':5e-7
             }

trans_parameters = {'lr':1e-3
              , 'num_layers':3
              , 'hidden_channels':64
              , 'dropout':0.3
              , 'batchnorm': False
              , 'l2':5e-4
              , 'heads':7
             }

transv2_parameters = {'lr':1e-3
              , 'num_layers':2
              , 'hidden_channels':64
              , 'dropout':0.3
              , 'batchnorm': False
              , 'l2':5e-4
              , 'heads':30
             }

transv3_parameters = {'lr':5e-4
              , 'num_layers':3
              , 'hidden_channels':32
              , 'dropout':0.3
              , 'batchnorm': False
              , 'l2':5e-4
              , 'heads':7
             }

pna_parameters = {'lr':1e-3
              , 'num_layers':3
              , 'hidden_channels':128
              , 'dropout':0.1
              , 'batchnorm': False
              , 'l2':5e-4
              , 'heads':4
             }

def train_nolabel(epoch, train_loader, model, data, train_idx, optimizer, device, no_conv=False):
    model.train()

    pbar = tqdm(total=train_idx.size(0), ncols=80)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [(edge_index.to(device), data.edge_attr[e_id], size) for (edge_index, e_id, size) in adjs]
        
        optimizer.zero_grad()
        # noise = torch.normal(0, 1, size=data.x[n_id].shape).to(device)
        out = model(data.x[n_id], adjs)
        
        tmp_y = out.argmax(1)
        
        alpha = 0.001
        # loss = F.nll_loss(out[tmp_y <= 1], tmp_y[tmp_y <= 1])
        loss = alpha*F.nll_loss(out[tmp_y <= 1], tmp_y[tmp_y <= 1], weight=torch.tensor([0.01, 0.99]).to(device))
        # loss = torch.log(1. + F.nll_loss(out, data.y[n_id[:batch_size]]))
        # loss = F.nll_loss(out, data.y[n_id[:batch_size]], weight=torch.tensor([1., 20.]).cuda())
        # loss = FocalLoss(gamma=10.0, alpha=0.01)(out[tmp_y <= 1], tmp_y[tmp_y <= 1])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        # total_loss += float(F.nll_loss(out, data.y[n_id[:batch_size]]))
        pbar.update(batch_size)

    pbar.close()
    loss = total_loss / len(train_loader)

    return loss


def train(epoch, train_loader, model, data, train_idx, optimizer, device, no_conv=False):
    model.train()

    pbar = tqdm(total=train_idx.size(0), ncols=80)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [(edge_index.to(device), data.edge_attr[e_id], size) for (edge_index, e_id, size) in adjs]
        
        optimizer.zero_grad()
        # noise = torch.normal(0, 1, size=data.x[n_id].shape).to(device)
        out = model(data.x[n_id], adjs)
        tmp_y = data.y[n_id[:batch_size]]
        
        # loss = F.nll_loss(out[tmp_y <= 1], tmp_y[tmp_y <= 1])
        loss = F.nll_loss(out[tmp_y <= 1], tmp_y[tmp_y <= 1], weight=torch.tensor([0.01, 0.99]).to(device))
        # loss = torch.log(1. + F.nll_loss(out, data.y[n_id[:batch_size]]))
        # loss = F.nll_loss(out, data.y[n_id[:batch_size]], weight=torch.tensor([1., 20.]).cuda())
        # loss = FocalLoss(gamma=10.0, alpha=0.01)(out[tmp_y <= 1], tmp_y[tmp_y <= 1])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        # total_loss += float(F.nll_loss(out, data.y[n_id[:batch_size]]))
        pbar.update(batch_size)

    pbar.close()
    loss = total_loss / len(train_loader)

    return loss


@torch.no_grad()
def test(layer_loader, model, data, split_idx, device, no_conv=False):
    # data.y is labels of shape (N, ) 
    model.eval()
    
    out = model.inference(data.x, layer_loader, device, data)
#     out = model.inference_all(data)
    y_pred = out.exp()  # (N,num_classes)   
    
    losses = dict()
    for key in ['train', 'valid', 'test']:
        node_id = split_idx[key]
        node_id = node_id.to(device)
        losses[key] = F.nll_loss(out[node_id], data.y[node_id], weight=torch.tensor([0.1, 0.9]).to(device)).item()
        # losses[key] = FocalLoss()(out[node_id], data.y[node_id]).item()
        
        if key == 'valid':
            # print(out[node_id].argmax(1))
            print(confusion_matrix(data.y[node_id].cpu().numpy(), out[node_id].argmax(1).cpu().numpy()))
            
    return losses, y_pred


@torch.no_grad()
def test_v2(train_loader, val_loader, test_loader, model, data, split_idx, device, no_conv=False):
    # data.y is labels of shape (N, ) 
    model.eval()
    
    out = torch.zeros((data.y.shape[0], 2)).to(device)
    
    for batch_size, n_id, adjs in train_loader:
        adjs = [(edge_index.to(device), data.edge_attr[e_id], size) for (edge_index, e_id, size) in adjs]    
        out[n_id[:batch_size]] = model(data.x[n_id], adjs)
    
    for batch_size, n_id, adjs in val_loader:
        adjs = [(edge_index.to(device), data.edge_attr[e_id], size) for (edge_index, e_id, size) in adjs]    
        out[n_id[:batch_size]] = model(data.x[n_id], adjs)
    
    for batch_size, n_id, adjs in test_loader:
        adjs = [(edge_index.to(device), data.edge_attr[e_id], size) for (edge_index, e_id, size) in adjs]    
        out[n_id[:batch_size]] = model(data.x[n_id], adjs)
    
    losses = dict()
    for key in ['train', 'valid', 'test']:
        node_id = split_idx[key]
        node_id = node_id.to(device)
        tmp_y = data.y[node_id]
        # losses[key] = F.nll_loss(out[node_id], data.y[node_id], weight=torch.tensor([0.01, 0.99]).to(device)).item()
        # loss = FocalLoss(gamma=2.0, alpha=0.01)(out[tmp_y <= 1], tmp_y[tmp_y <= 1])
        if key in ['train', 'valid']:
            losses[key] = F.nll_loss(out[node_id][tmp_y<=1], data.y[node_id][tmp_y<=1], weight=torch.tensor([0.1, 0.9]).to(device))
            # losses[key] = FocalLoss(gamma=2.0, alpha=0.01)(out[node_id][tmp_y<=1], data.y[node_id][tmp_y<=1])
        else:
            losses[key] = 0.
        if key == 'valid':
            # print(out[node_id].argmax(1))
            print(confusion_matrix(data.y[node_id].cpu().numpy(), out[node_id].argmax(1).cpu().numpy()))
    
    y_pred = out.exp()  # (N,num_classes)  
    return losses, y_pred
        
            
def main():
    parser = argparse.ArgumentParser(description='minibatch_gnn_models')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='XYGraphP1')
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--epochs', type=int, default=100)
    
    args = parser.parse_args()
    print(args)
    
    no_conv = False
    if args.model in ['mlp']: no_conv = True        
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    # T.ToUndirected(), 
    dataset = XYGraphP1(root='./', name='xydata', transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]))
    
    nlabels = dataset.num_classes
    if args.dataset =='XYGraphP1': nlabels = 2
        
    data = dataset[0]
    
    
    
    # data.x = data.x[:,:2]
    # data.x = data.x.index_select(1, torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16]))
    # for i in range(data.x.shape[-1]):
    #     tmp = data.x[:, i]
    #     tmp[tmp == -1] = 0
    #     data.x[:, i] = tmp
    # data.x = data.x[,]
    # data.adj_t = data.adj_t.to_symmetric()
    # data.x = torch.cat([torch.tensor(np.load('./fixed_features.npy')), data.x[:, -1].unsqueeze(1)], dim=-1)
    data.x = torch.cat([data.x, torch.tensor(np.load('./notebook/type2.npy')).unsqueeze(1)], dim=-1)
    data.x = torch.cat([data.x, torch.tensor(np.load('./notebook/type3.npy')).unsqueeze(1)], dim=-1)
    data.x = torch.cat([data.x, torch.tensor(np.load('./notebook/in_degree.npy')).unsqueeze(1)], dim=-1)
    data.x = torch.cat([data.x, torch.tensor(np.load('./notebook/out_degree.npy')).unsqueeze(1)], dim=-1)
    data.x = torch.cat([data.x, torch.tensor(np.load('./notebook/all_degree.npy')).unsqueeze(1)], dim=-1)

    data.x = torch.cat([data.x, torch.tensor(np.load('./notebook/max_contact_degree_all.npy')).unsqueeze(1)], dim=-1)
    data.x = torch.cat([data.x, torch.tensor(np.load('./notebook/max_contact_degree_in.npy')).unsqueeze(1)], dim=-1)
    data.x = torch.cat([data.x, torch.tensor(np.load('./notebook/max_contact_degree_out.npy')).unsqueeze(1)], dim=-1)
    data.x = torch.cat([data.x, torch.tensor(np.load('./notebook/min_contact_degree_all.npy')).unsqueeze(1)], dim=-1)
    data.x = torch.cat([data.x, torch.tensor(np.load('./notebook/min_contact_degree_in.npy')).unsqueeze(1)], dim=-1)
    data.x = torch.cat([data.x, torch.tensor(np.load('./notebook/min_contact_degree_out.npy')).unsqueeze(1)], dim=-1)
    # data.x = torch.cat([data.x, (data.y == 2).float(), (data.y == 3).float()], dim=-1)
    
    
    
    
    if args.dataset in ['XYGraphP1']:
        x = data.x
        x = (x-x.mean(0))/(x.std(0) + 1e-40)
        data.x = x
    
    # data.x = torch.cat([data.x, torch.tensor(np.load('./notebook/act_behavior.npy')).float()], dim=-1)
    
    use_edge_vec = True
    use_edge_ts = True
    use_back_nodes = True
    if use_edge_vec:
        # edge_vec = np.load("./edge_vec.npy")
        edge_vec = np.load("./edge_attr_directed_v2.npy")
        edge_vec = torch.tensor(edge_vec, dtype=torch.float)
        data.x = torch.cat([data.x, edge_vec], dim=-1)
        print("Node features after add edge attr:", data.x.size())
    if use_edge_ts:

        edge_vec = np.load("./edge_ts_v2.npy")
        edge_vec = torch.tensor(edge_vec, dtype=torch.float)
        data.x = torch.cat([data.x, edge_vec], dim=-1)

        edge_vec = np.load("./edge_ts_v3.npy")
        edge_vec = torch.tensor(edge_vec, dtype=torch.float)
        data.x = torch.cat([data.x, edge_vec], dim=-1)
        print("Node features after add edge ts:", data.x.size())

    if use_back_nodes:
        edge_vec = np.load("./back_nodes_vec.npy")
        edge_vec = torch.tensor(edge_vec, dtype=torch.float)
        data.x = torch.cat([data.x, edge_vec], dim=-1)
        print("Node features after add edge ts:", data.x.size())
    

    
        
    # data.x = data.x[:,1:]
    
    # data.x = torch.cat([data.x, torch.tensor(np.load('./notebook/node2vec.npy'))], dim=-1)
        
    if data.y.dim()==2:
        data.y = data.y.squeeze(1)        
    
    split_idx = {'train':data.train_mask, 'valid':data.valid_mask, 'test':data.test_mask}
        
    data = data.to(device)
    
    train_idx = split_idx['train']
    
    # # # # All nodes
    # tmp_set = set(list(data.train_mask.cpu().numpy())  + list(data.valid_mask.cpu().numpy()) + list(data.test_mask.cpu().numpy()))
    # bg_idx = []
    # for i in range(len(data.y)):
    #     if i not in tmp_set:
    #         bg_idx.append(i)
            
    # bg_idx = torch.tensor(bg_idx)
    # train_idx = torch.cat([train_idx, torch.tensor(bg_idx)]).to(device)
        
        
    model_dir = prepare_folder(args.dataset, args.model)
    print('model_dir:', model_dir)

    # , 15, 12, 10
    train_loader = NeighborSampler(data.adj_t, node_idx=train_idx, sizes=[10, 5, 5], 
                                   batch_size=1024, shuffle=True, num_workers=12)
    
    # train_loader_nolabel = NeighborSampler(data.adj_t, node_idx=bg_idx, sizes=[20, 15], 
    #                                batch_size=2048, shuffle=True, num_workers=12)
    
    # sampler = ImbalancedSampler(data, input_nodes=train_idx)
    # train_loader = NeighborLoader(data, input_nodes=train_idx, num_neighbors=[10, 7, 4], sampler=sampler, batch_size=2048, shuffle=True, num_workers=12)
    # test_idx = torch.cat([data.train_mask, data.valid_mask, data.test_mask])
    
    # layer_loader = NeighborSampler(data.adj_t, node_idx=None, sizes=[-1], batch_size=4096, shuffle=False, num_workers=12)
    
    # # All nodes
    valid_idx = split_idx['valid']
    # tmp_set = set(list(data.train_mask.cpu().numpy()) + list(data.test_mask.cpu().numpy()))
    # bg_idx = []
    # for i in range(len(data.y)):
    #     if i not in tmp_set:
    #         bg_idx.append(i)
    # valid_idx = torch.cat([valid_idx, torch.tensor(bg_idx)]).to(device)
    
    test_idx = split_idx['test']
    # tmp_set = set(list(data.train_mask.cpu().numpy()) + list(data.valid_mask.cpu().numpy()))
    # bg_idx = []
    # for i in range(len(data.y)):
    #     if i not in tmp_set:
    #         bg_idx.append(i)
    # test_idx = torch.cat([test_idx, torch.tensor(bg_idx)]).to(device)
    
    
    train_loader_v2 = NeighborSampler(data.adj_t, node_idx=train_idx, sizes=[-1, -1, -1], batch_size=4096, shuffle=False, num_workers=12)
    valid_loader = NeighborSampler(data.adj_t, node_idx=valid_idx, sizes=[-1, -1, -1], batch_size=4096, shuffle=False, num_workers=12)
    test_loader = NeighborSampler(data.adj_t, node_idx=test_idx, sizes=[-1, -1, -1], batch_size=4096, shuffle=False, num_workers=12)
    
    # train_loader_v2 = NeighborSampler(data.adj_t, node_idx=train_idx, sizes=[-1, -1, -1, -1, -1], batch_size=4096, shuffle=False, num_workers=12)
    # valid_loader = NeighborSampler(data.adj_t, node_idx=valid_idx, sizes=[-1, -1, -1, -1, -1], batch_size=4096, shuffle=False, num_workers=12)
    # test_loader = NeighborSampler(data.adj_t, node_idx=test_idx, sizes=[-1, -1, -1, -1, -1], batch_size=4096, shuffle=False, num_workers=12)
    
    
    if args.model == 'sage_neighsampler':
        para_dict = sage_neighsampler_parameters
        model_para = sage_neighsampler_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = SAGE_NeighSampler(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
    if args.model == 'sage_neighsampler_ens':
        para_dict = sage_neighsampler_parameters
        model_para = sage_neighsampler_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = SAGE_NeighSamplerEnsemble(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
        
    if args.model == 'gat_neighsampler':   
        para_dict = gat_neighsampler_parameters
        model_para = gat_neighsampler_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')        
        model = GAT_NeighSampler(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
    if args.model == 'gatv2_neighsampler':        
        para_dict = gatv2_neighsampler_parameters
        model_para = gatv2_neighsampler_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')        
        model = GATv2_NeighSampler(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
    
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
    
    if args.model == 'trans':
        para_dict = trans_parameters
        model_para = trans_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')        
        model = TransNetNS(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
    
    if args.model == 'transv2':
        para_dict = transv2_parameters
        model_para = transv2_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')        
        model = TransNetNSV2(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
    if args.model == 'transv3':
        para_dict = transv3_parameters
        model_para = transv3_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')        
        model = TransNetNSV3(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
    
    if args.model == 'pna':
        para_dict = pna_parameters
        model_para = pna_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')        
        model = PNANetNS(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
        
    

    print(f'Model {args.model} initialized')
    
    # raw_data = np.load('xydata/raw/phase1_gdata.npz')
    # ts = np.concatenate([raw_data['edge_timestamp'], raw_data['edge_timestamp']])
    # ts = torch.tensor(ts).to(device)

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
    # optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=para_dict['lr'])
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=para_dict['lr'])
    min_valid_loss = 1e8
    max_valid_auc = 1e-8

    for epoch in range(1, args.epochs+1):
        loss = train(epoch, train_loader, model, data, train_idx, optimizer, device, no_conv)
        
        # loss_2 = train_nolabel(epoch, train_loader_nolabel, model, data, bg_idx, optimizer, device, no_conv)
        
        # losses, out = test(layer_loader, model, data, split_idx, device, no_conv)
        losses, out = test_v2(train_loader_v2, valid_loader, test_loader, model, data, split_idx, device, no_conv)
        train_loss, valid_loss, test_loss = losses['train'], losses['valid'], losses['test']
        
        evaluator = Evaluator('auc')
        preds_train, preds_valid = out[data.train_mask], out[data.valid_mask]
        y_train, y_valid = data.y[data.train_mask], data.y[data.valid_mask]
        train_auc = evaluator.eval(y_train, preds_train)['auc']
        valid_auc = evaluator.eval(y_valid, preds_valid)['auc']
        print('train_auc:',train_auc)
        print('valid_auc:',valid_auc)

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), model_dir+'model.pt')
            
        if valid_auc > max_valid_auc:
            max_valid_auc = valid_auc
            torch.save(model.state_dict(), model_dir+'model.pt')

        if epoch % args.log_steps == 0:
            print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                    #   f'Loss_NoLabel: {loss_2:.4f}, '
                      f'Train: {100 * train_loss:.3f}%, '
                      f'Valid: {100 * valid_loss:.3f}% '
                      f'Test: {100 * test_loss:.3f}%')


if __name__ == "__main__":
    main()
