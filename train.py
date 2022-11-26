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

from helper_functions import save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest = "data_dir", action = "store")
    parser.add_argument("--arch", dest = "arch", default = "vgg16", choices = ["vgg16", "densenet121"])
    parser.add_argument("--learning_rate", dest = "learnrate", default = "0.003")
    parser.add_argument("--hidden_units", dest = "hidden_units", default = "2016")
    parser.add_argument("--epochs", dest = "epochs", default = "3")
    parser.add_argument("--gpu", action = "store", default = "gpu")
    parser.add_argument("--save_dir", dest = "save_dir", action = "store", default = "checkpoint.pth")
                        
    return parser.parse_args()

def train(model, epochs, criterion, optimizer, trainloader, validloader, gpu):
    steps = 0
    print_every = 10

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            steps += 1
            
            if gpu == "gpu":
                model.cuda()
                inputs, labels = inputs.cuda(), labels.cuda()
            else:
                model.cpu()

            # IMPORTANT: Set optimizer to zero!
            optimizer.zero_grad()

            # Forwards
            outputs = (model.forward(inputs)).cuda()
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()

            # Update running_loss
            running_loss += train_loss.item()

            # Print info every print_every
            if steps % print_every == 0:

                # Now test the classifier on the validation dataset
                model.eval()
                valid_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for inputs_v, labels_v in validloader:
                        
                        if gpu == "gpu":
                            model.cuda()
                            inputs_v, labels_v = inputs.cuda(), labels.cuda()
                        else:
                            pass

                        # IMPORTANT: Set optimizer to zero!
                        optimizer.zero_grad()

                        outputs_v = model.forward(inputs_v)
                        v_loss = criterion(outputs_v, labels_v) #'v' to indicate that this is for the validation set

                        valid_loss += v_loss.item()

                        # Accuracy:
                        probablity = torch.exp(outputs)
                        top_p, top_class = probablity.topk(1, dim = 1)
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

                print(f"Epoch number: {epoch+1}/{epochs}.. "
                      f"Training loss: {running_loss/print_every:.4f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.4f}.. "
                      f"Test accuracy: {accuracy/len(validloader):.4f}")
                running_loss = 0

def main():
    args = parse_args()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    training_data = datasets.ImageFolder(train_dir, transform = training_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform = validation_transforms)
    testing_data = datasets.ImageFolder(test_dir, transform = testing_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(training_data, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(validation_data, batch_size = 64)
    testloader = torch.utils.data.DataLoader(testing_data, batch_size = 64)
    
    model = getattr(models, args.arch)(pretrained = True)
  
    for param in model.parameters():
        param.requires_grad = False
    
    if args.arch == "vgg16":
        classifier = nn.Sequential(OrderedDict([
                          ("fc1", nn.Linear(25088, 5012)),
                          ("relu", nn.ReLU()),
                          ("droupout", nn.Dropout(0.2)),
                          ("fc2", nn.Linear(5012, 2076)),
                          ("relu", nn.ReLU()),
                          ("fc3", nn.Linear(2076, 102)), #Go down to 102 bc there are 102 types of flowers in dataset
                          ("output", nn.LogSoftmax(dim = 1))
                          ]))
    elif args.arch == "densenet121":
        classifier = nn.Sequential(OrderedDict([
                          ("fc1", nn.Linear(1064, 512)),
                          ("relu", nn.ReLU()),
                          ("droupout", nn.Dropout(0.2)),
                          ("fc2", nn.Linear(512, 276)),
                          ("relu", nn.ReLU()),
                          ("fc3", nn.Linear(276, 102)), #Go down to 102 bc there are 102 types of flowers in dataset
                          ("output", nn.LogSoftmax(dim = 1))
                          ]))
      
    model.classifier = classifier
    criterion = nn.NLLLoss()
    learnrate = float(args.learnrate)
    optimizer = optim.Adam(model.classifier.parameters(), lr = learnrate)
    epochs = int(args.epochs)
    gpu = args.gpu
    train(model, epochs, criterion, optimizer, trainloader, validloader, gpu)
    
    model.class_to_idx = training_data.class_to_idx
    checkpoint_name = args.save_dir
    
    save_checkpoint(checkpoint_name, model, optimizer, args, classifier)
       
main()