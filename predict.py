import argparse
import torch
from torchvision import datasets, transforms

from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict
from PIL import Image
import numpy as np
import numpy

from helper_functions import load_names, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", action = "store", default = "checkpoint.pth")
    parser.add_argument("--top_k", dest = "top_k", default = "5")
    parser.add_argument("--pathname", dest = "pathname", default = "flowers/test/19/image_06197.jpg")
    parser.add_argument("--category_names", dest = "category_names", default = "cat_to_name.json")
    parser.add_argument("--gpu", action = "store", default = "gpu")
                        
    return parser.parse_args()

def process_image(image_name):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    image = Image.open(image_name)
    
    if image.size[0] > image.size[1]:
        size = [image.size[0], 256] #size needs to have length and width
    else:
        size = [256, image.size[1]] 
        
    image.thumbnail(size)
    
    #crop out the center
    left = (256 - 224)/2
    right = (256 + 224)/2
    top = (256 - 224)/2
    bottom = (256 + 224)/2
    
    image = image.crop((left, top, right, bottom))
    
    #convert color channel
    image = np.array(image)
    image = image/255
    
    #normalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = ((image - mean)/std)
        
    #transpose
    image = np.transpose(image, (2, 0, 1))
    
    return image

def predict(image_path, model, topk=5, gpu = "gpu"):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if gpu == "gpu":
        model.cuda()
    else:
        model.cpu()
           
    image = process_image(image_path)
    image = torch.from_numpy(np.array([image])) # make an np.array
    image = image.float() #make sure values are floats
    
    if gpu == "gpu":
        with torch.no_grad():
            outputs = model.forward(image.cuda()) #get outputs
    else:
        with torch.no_grad():
            outputs = model.forward(image) #get outputs
      
    probability = F.softmax(outputs.data, dim = 1)
    top_probability = np.array(probability.topk(topk)[0][0])
    
    #display names
    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    top_classes = [np.int(index_to_class[i]) for i in np.array(probability.topk(topk)[1][0])]
    
    return top_probability, top_classes



def main():
    args = parse_args()
    model = load_checkpoint(args.checkpoint)
    names = load_names(args.category_names)
    gpu = args.gpu
    category_names = load_names(args.category_names)

    image_name = args.pathname
    probability, classes = predict(image_name, model, int(args.top_k), gpu)
    
    label = [category_names[str(index)] for index in classes]
    
    print("The selected file is:", image_name)
    print("The top probability's name is:", label[0], "with a probability of:", probability[0])

main()