from torch import nn
from torch import optim
import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict
from PIL import Image
import numpy as np
import numpy
import json

def save_checkpoint(checkpoint_name, model, optimizer, args, classifier):
    checkpoint = {'classifier': classifier,
                  'pretrained_model': args.arch,
                  'hidden_units': args.hidden_units,
                  'model': model,
                  'learnrate': args.learnrate,
                  'epochs': args.epochs,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, checkpoint_name)
    
def load_checkpoint(checkpoint_name):
    checkpoint = torch.load(checkpoint_name)
    pretrained_model = checkpoint['pretrained_model']
    
    #create a new model
    model = getattr(torchvision.models, pretrained_model)(pretrained = True)
    
    #set model attributes
    model.classifier = checkpoint["classifier"]
    model.epochs = checkpoint["epochs"]
    model.optimizer = checkpoint["optimizer"]
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    
    return model

def load_names(pathname):
    with open(pathname) as file:
        category_names = json.load(file)
    
    return category_names