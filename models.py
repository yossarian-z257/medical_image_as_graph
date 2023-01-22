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



def get_feture_extractor_model(model):
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    auto_transform = weights.transforms()

    if model == 'resnet18':
        model = models.resnet18(pretrained=True).to(device)
    elif model == 'efficientnet_b0':
        model = models.efficientnet_b0(weights = weights).to(device)
    else:
        model = torch.load(f"{current_file_path}/saved_models/model_densenet_head.pt").to(device)

    layes_names = get_graph_node_names(model)
    return model