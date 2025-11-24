import torch.nn as nn
from model_TAQ import Model
from utils_TAQ import *
import torch
from sklearn.metrics import roc_auc_score
import random
import dgl
from sklearn.metrics import average_precision_score
import argparse
from tqdm import tqdm
import time
import numpy as np
import scipy.sparse as sp
import math
import time
import numpy as np
from torch_geometric.data import Data
from tae import TAEAugmenter
import torch_geometric as pyg
# from bitarray import bitarray

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default='reddit')
parser.add_argument('--lr', type=float,default='0.001')
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--num_epoch', type=int,default='300')
parser.add_argument('--use_pseudo_labels', action='store_true')
parser.add_argument('--use_tae', action='store_true', help='Enable TAE augmentation')

args = parser.parse_args()


dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_dict = load_mat(
    dataset=args.dataset,
    train_rate=0.3,
    val_rate=0.1,
    normal_rate=0.5
)

adj = data_dict['adj']
features = data_dict['feat']
ano_labels = data_dict['ano_labels']
idx_train = data_dict['idx_train']
idx_val = data_dict['idx_val']
idx_test = data_dict['idx_test']

labeled_normal_idx = data_dict['labeled_normal_idx']
unlabeled_nodes = data_dict['unlabeled_nodes']

if args.dataset in ['Amazon', 'tf_finace', 'reddit', 'elliptic']:
    features, _ = preprocess_features(features)
else:
    features = features.todense()

dgl_graph = adj_to_dgl_graph(adj)

nb_nodes = features.shape[0]
ft_size = features.shape[1]


all_idx = list(range(nb_nodes))
raw_adj = adj
adj = normalize_adj(adj)
raw_adj = (raw_adj + sp.eye(raw_adj.shape[0])).todense()
adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
features = torch.FloatTensor(features)
adj = torch.FloatTensor(adj)
adj = torch.FloatTensor(adj[np.newaxis])
raw_adj = torch.FloatTensor(raw_adj[np.newaxis])


model = Model(ft_size, args.embedding_dim, 'prelu')
model.load_state_dict(torch.load('test_reddit.pth', map_location='cpu'))


model.eval()
with torch.no_grad():
    emb_val, logits_val = model(features, adj)
    logits_val = np.squeeze(logits_val[:, idx_val, :].cpu().detach().numpy())
    if np.isnan(logits_val).any():
        logits_val = np.nan_to_num(logits_val, nan=0.0)

    val_auroc = roc_auc_score(ano_labels[idx_val], logits_val)
    val_auprc = average_precision_score(ano_labels[idx_val], logits_val, pos_label=1)

with torch.no_grad():
    emb_test, logits_test = model(features, adj)
    logits_test = np.squeeze(logits_test[:, idx_test, :].cpu().detach().numpy())
    if np.isnan(logits_test).any():
        logits_test = np.nan_to_num(logits_test, nan=0.0)

    test_auroc = roc_auc_score(ano_labels[idx_test], logits_test)
    test_auprc = average_precision_score(ano_labels[idx_test], logits_test, pos_label=1)


print(f"Test AUROC: {test_auroc:.4f}, Test AUPRC: {test_auprc:.4f}")