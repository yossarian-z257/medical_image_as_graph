import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import torchvision.io as io
import numpy as np
from IPython.display import Image as dImage
import os
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from functools import lru_cache
from torchvision import transforms
import torchvision 
import torch
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv
from torch_geometric.loader import NeighborLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph, to_networkx, from_networkx
from torch_geometric.loader import DataLoader
import os
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from sklearn.metrics import roc_auc_score
import numpy as np
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool
from sklearn.metrics import confusion_matrix
import torch_geometric.transforms as T
import random
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import Sequential  as Seq, GCNConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool, global_add_pool
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GINConv
# from sklearn.metrics import roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
current_file_path = os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=100000)
def get_feture_extractor_model(model_name):
    
    # auto_transform = weights.transforms()

    if model_name == 'resnet18':

        model = models.resnet18(pretrained=True).to(device)
        
    elif model_name == 'efficientnet_b0':
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights = weights).to(device)
        
    else:
        weights = torchvision.models.DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights = weights).to(device)#torch.load(f"{current_file_path}/saved_models/model_densenet_head.pt").to(device)
        
    layes_names = get_graph_node_names(model)
    # model.eval()
    feature_extractor = create_feature_extractor(
        model, return_nodes=['flatten'])

    return model, feature_extractor

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(hidden_channels, 512)
        self.conv2 = GCNConv(512, 256)
        self.conv3 = GCNConv(256,128)
        self.conv4 = GCNConv(128, 64)
        self.lin1 = Linear(64, 32)
        #self.lin2 = Linear(128,64)
        self.lin = Linear(32, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = x.relu()
        x = self.lin(x)
        
        return x

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        
        self.conv1 = GATConv(hidden_channels, self.hid, heads=self.in_head, dropout=0.3)
        self.conv2 = GATConv(self.hid*self.in_head, 32, concat=False,
                             heads=self.out_head, dropout=0.3)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)
        self.lin1 = Linear(32, 2)

    def forward(self,x, edge_index,batch):
        
        # Dropout before the GAT layer is used to avoid overfitting
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.lin1(x)

        return x



class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self,input_dim ,dim_h = 64):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(input_dim, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        # self.lin1 = Linear(dim_h*3, dim_h*3)
        # self.lin2 = Linear(dim_h*3, 2)
        self.lin1 = Linear(dim_h*3, 64)
        self.lin2 = Linear(64, 2)

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        return  F.log_softmax(h, dim=1)


# not used anywhere, testing for mid layers of GNN
# model_gcn_jmpk = Sequential('x, edge_index, batch', [
#     (GCNConv(1024, 512), 'x, edge_index -> x1'),
#     ReLU(inplace=True),
#     (GCNConv(512, 256), 'x1, edge_index -> x2'),
#     ReLU(inplace=True),
#     (GCNConv(256, 128), 'x2, edge_index -> x3'),
#     ReLU(inplace=True),
#     (GCNConv(128, 64), 'x3, edge_index -> x4'),
#     ReLU(inplace=True),
#     (GCNConv(64, 64), 'x4, edge_index -> x5'),
#     ReLU(inplace=True),
#     (lambda x4, x5: [x4, x5], 'x4, x5 -> xs'),
#     (JumpingKnowledge("max", 64, num_layers=4), 'xs -> x'),
#     # (JumpingKnowledge("lstm", 64, num_layers=2), 'xs -> x'),
#     (global_mean_pool, 'x, batch -> x'),
#     (Dropout(p=0.8), 'x -> x'),
#     # Linear(2 * 64, 2),
#     (Linear(64, 32), 'x -> x'),
#     ReLU(inplace=True),
#     (Linear(32, 2), 'x -> x'),
# ])