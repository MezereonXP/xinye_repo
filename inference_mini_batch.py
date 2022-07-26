# dataset name: XYGraphP1

from tkinter import N
from models.trans_neighsampler_v2 import TransNetNSV2
from models.trans_neighsampler_v3 import TransNetNSV3
from utils import XYGraphP1
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from torch_geometric.data import NeighborSampler
from models import SAGE_NeighSampler, GAT_NeighSampler, GATv2_NeighSampler, SAGE_NeighSamplerEnsemble, TransNetNS
from tqdm import tqdm

import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import pandas as pd
import numpy as np


sage_neighsampler_parameters = {'lr':0.003
              , 'num_layers':3
              , 'hidden_channels':200
              , 'dropout':0.1
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
              , 'num_layers':3
              , 'hidden_channels':128
              , 'dropout':0.1
              , 'batchnorm': False
              , 'l2':5e-6
             , 'layer_heads':[4,1]
             }
gcn2_parameters = {'lr':1e-2
              , 'num_layers':3
              , 'hidden_channels':128
              , 'dropout':0
              , 'batchnorm': False
              , 'l2':5e-7
             }

trans_parameters = {'lr':1e-3
              , 'num_layers':2
              , 'hidden_channels':32
              , 'dropout':0.3
              , 'batchnorm': False
              , 'l2':5e-4
              , 'heads':30
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
              , 'num_layers':2
              , 'hidden_channels':64
              , 'dropout':0.3
              , 'batchnorm': False
              , 'l2':5e-4
              , 'heads':15
             }

@torch.no_grad()
def test(layer_loader, model, data, device, no_conv=False, ts=None):
    # data.y is labels of shape (N, ) 
    model.eval()
    
    out = model.inference(data.x, layer_loader, device, data)
#     out = model.inference_all(data)
    y_pred = out.exp()  # (N,num_classes)   
                
    return y_pred


@torch.no_grad()
def test_v2(train_loader, val_loader, test_loader, model, data, device, no_conv=False):
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
    
    y_pred = out.exp()  # (N,num_classes)  
    return y_pred
        
            
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

    dataset = XYGraphP1(root='./', name='xydata', transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]))
    
    nlabels = dataset.num_classes
    if args.dataset =='XYGraphP1': nlabels = 2
        
    data = dataset[0]
    
    
    # data.x = torch.tensor(np.load('./fixed_features.npy'))
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
    # data.x = data.x[:,1:]
    # # data.adj_t = data.adj_t.to_symmetric()
    # for i in range(data.x.shape[-1]):
    #     tmp = data.x[:, i]
    #     # tmp[tmp == -1] = torch.median(tmp[tmp!=-1])
    #     tmp[tmp == -1] = 0
    #     data.x[:, i] = tmp
    
    if args.dataset in ['XYGraphP1']:
        x = data.x
        x = (x-x.mean(0))/(1e-30 + x.std(0))
        data.x = x
    
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
        
    
        # data.y[data.y > 1] = 0
    if data.y.dim()==2:
        data.y = data.y.squeeze(1)            
        
    data = data.to(device)
    
    
    test_idx = data.test_mask
    # tmp_set = set(list(data.train_mask) + list(data.valid_mask))
    # bg_idx = []
    # for i in range(len(data.y)):
    #     if i not in tmp_set:
    #         bg_idx.append(i)
    # test_idx = torch.cat([test_idx, torch.tensor(bg_idx).to(device)]).to(device)
        
    # layer_loader = NeighborSampler(data.adj_t, node_idx=None, sizes=[-1], batch_size=4096, shuffle=False, num_workers=12)        
    train_loader_v2 = NeighborSampler(data.adj_t, node_idx=data.train_mask, sizes=[-1, -1], batch_size=4096, shuffle=False, num_workers=12)
    valid_loader = NeighborSampler(data.adj_t, node_idx=data.valid_mask, sizes=[-1, -1], batch_size=4096, shuffle=False, num_workers=12)
    test_loader = NeighborSampler(data.adj_t, node_idx=test_idx, sizes=[-1, -1], batch_size=4096, shuffle=False, num_workers=12)
    
    
    
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
        
    


    print(f'Model {args.model} initialized')


    model_file = './model_files/{}/{}/model.pt'.format(args.dataset, args.model)
    print('model_file:', model_file)
    model.load_state_dict(torch.load(model_file, map_location=device))
    
    # raw_data = np.load('xydata/raw/phase1_gdata.npz')
    # ts = np.concatenate([raw_data['edge_timestamp'], raw_data['edge_timestamp']])
    # ts = torch.tensor(ts).to(device)

    # out = test(layer_loader, model, data, device, no_conv)
    out = test_v2(train_loader_v2, valid_loader, test_loader, model, data, device, no_conv)

    evaluator = Evaluator('auc')
    preds_train, preds_valid = out[data.train_mask], out[data.valid_mask]
    y_train, y_valid = data.y[data.train_mask], data.y[data.valid_mask]
    train_auc = evaluator.eval(y_train, preds_train)['auc']
    valid_auc = evaluator.eval(y_valid, preds_valid)['auc']
    print('train_auc:',train_auc)
    print('valid_auc:',valid_auc)
    
    preds = out[data.test_mask].cpu().numpy()
    np.save('./submit/preds.npy', preds)


if __name__ == "__main__":
    main()
