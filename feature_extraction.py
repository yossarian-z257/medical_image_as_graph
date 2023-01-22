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
from models import get_feture_extractor_model



current_file_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fet_from_img(img,i = 0, model = 'densenet121'):


    model = get_feture_extractor_model(model)
    feature_extractor = create_feature_extractor(
        model, return_nodes=['flatten'])

    transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    #transforms.CenterCrop(224),
    #  transforms.RandomRotation(20),
    #  transforms.RandomHorizontalFlip(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_normalized = transform_norm(img).float()
    img_normalized = img_normalized.unsqueeze_(0)
    img_normalized = img_normalized.to(device)

    with torch.no_grad():
        model.eval()
        output =model(img_normalized) # not using it for now, to use this to assign edge features layer
        out = feature_extractor(img_normalized)
        return out['flatten'],i

