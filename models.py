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

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    