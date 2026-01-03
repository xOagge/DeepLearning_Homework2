# -*- coding: utf-8 -*-


#https://github.com/MedMNIST/MedMNIST


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms

from medmnist import BloodMNIST, INFO

import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

import json, os 


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device    )

#addition to making the seed fixed so the experiment is replicable
import random

def resetSeed():
    seed = 52 #used 42 in all report evolutions. 52 gives an usual result for model 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data Loading

data_flag = 'bloodmnist'
print(data_flag)
info = INFO[data_flag]
print(len(info['label']))
n_classes = len(info['label'])

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

import time

# --------- Before Training ----------
total_start = time.time()

#Training Function

def train_epoch(loader, model, criterion, optimizer):

    model.train() # Set model in training mode
    total_loss = 0.0 # intiialize loss sum

    # For each abtch: forward pass, loss calculation, backpropagation, model update
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        labels = labels.squeeze().long()  # <-- ensure correct shape and type
        #reset optimizer gradient
        optimizer.zero_grad()
        #apply foward pass on the model, outputs is the prediction. shape: (n_examples x n_classes)
        outputs = model(imgs)
        #calculate loss of outputs respective to labels. loss is 0-dim tensor
        loss = criterion(outputs, labels)
        #applying backward() to tensor loss, torch backdates the used tensors to calculate loss
        # , resulting in the dL/dW for each layer
        loss.backward()
        #updates model parameters according to W -> W - n * dL/dW. optimizer has access to the model parameters in its definition
        optimizer.step()

        total_loss += loss.item() #item is so that instead of 0-dim tensor we get float

    #return average loss of the batches over this epoch
    return total_loss / len(loader)

#Evaluation Function

def evaluate(loader, model):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.squeeze().long()

            outputs = model(imgs)
            preds += outputs.argmax(dim=1).cpu().tolist()
            targets += labels.tolist()

    return accuracy_score(targets, preds)


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)

    x = list(range(1, epochs + 1))

    plt.plot(x, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


#autoamtion of main code, its essentially the same for both exercises
#only changing class that defines the model.
def run_exercise(model_class, n_classes, softmax_use, file_prefix):
    #reset seed
    resetSeed()
    #hyperparameters definition
    batch_size = 64
    epochs = 200
    lr = 0.001

    train_dataset = BloodMNIST(split='train', transform=transform, download=True, size=28)
    val_dataset   = BloodMNIST(split='val',   transform=transform, download=True, size=28)
    test_dataset  = BloodMNIST(split='test',  transform=transform, download=True, size=28)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # initialize the model using the passed class
    model = model_class(n_classes=n_classes, softmax_use=softmax_use).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # training loop
    train_losses = []
    val_accs = []
    test_accs = []

    start_time = time.time() #start chronometer

    for epoch in range(epochs):
        epoch_start = time.time()

        train_loss = train_epoch(train_loader, model, criterion, optimizer)
        val_acc = evaluate(val_loader, model)
        test_acc = evaluate(test_loader, model)

        train_losses.append(train_loss)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        print(f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | "
            f"Time: {epoch_time:.2f} sec")

    # Save the model
    torch.save(model.state_dict(), f"{file_prefix}.pth")
    print("Model saved as bloodmnist_cnn.pth")

    end_time = time.time() #end chronometer

    #print elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    #and save on file
    time_file = "time_measures.json"
    if os.path.exists(time_file) and os.path.getsize(time_file) > 0:
        with open(time_file, "r") as f: times = json.load(f)
    else: times = {}
    times[file_prefix] = elapsed_time
    with open(time_file, "w") as f: json.dump(times, f, indent=2)

    # Plot results
    plot(epochs, train_losses, ylabel='Loss', name=f'{file_prefix}-training-loss')
    plot(epochs, val_accs, ylabel='Accuracy', name=f'{file_prefix}-validation-accuracy')
    plot(epochs, test_accs, ylabel='Accuracy', name=f'{file_prefix}-test-accuracy')
    #and save on a file
    metrics_file = "model_train_metrics.json"
    if os.path.exists(metrics_file) and os.path.getsize(metrics_file) > 0:
        with open(metrics_file, "r") as f: metrics = json.load(f)
    else: metrics = {}

    metrics[file_prefix] = { "train_loss": train_losses, "val_accuracy": val_accs,
        "test_accuracy": test_accs}

    with open(metrics_file, "w") as f: json.dump(metrics, f, indent=2)


#Deleting content in the json files if exists
open("time_measures.json", "w").close()
open("model_train_metrics.json", "w").close()
print("Deleted Contents of json files")

#--------------------------------------  Q1.1)  --------------------------------------

#Defining the CNN model with the nn.Module class, as suggested. This class will define
# the CNN fixed as the indicated in the homework
class BloodMNIST_CNN(nn.Module):
    def __init__(self, n_classes, softmax_use=False):
        super(BloodMNIST_CNN, self).__init__()
        self.softmax_use = softmax_use
        #L_out = ((L_in + 2P - K) / S)  + 1, for S=1, P=1, K=3, W and H remains the same

        # layer 1: 
        #     -Input image is RGB, so 3 channels
        #     -Output: 32 channels feature map
        #     -Kernel 3x3, size=3
        #     -kernel slide with stride and padding 1
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # layer 2: 
        #     -Input: 32 channels feature map
        #     -Output: 64 channels feature map
        #     -Kernel 3x3, size=3
        #     -kernel slide with stride and padding 1
        # 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # layer 3: 
        #     -Input: 64 channels feature map
        #     -Output: 128 channels feature map
        #     -Kernel 3x3, size=3
        #     -kernel slide with stride and padding 1
        # 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Linear Layer 1:
        #     -Input: stride and padding 1 preserves the dimension of the height and width, so
        # Input feature map is 28x28x128, meaning number of activations in feature map is 28x28x128,
        #     -Output: 256, specified in homework
        # 
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        # Linear Layer 2:
        #     -Input: 256, number of outputs of previous layer
        #     -Output: n_classes, to have a final classification of the input image of the model
        # 
        self.fc2 = nn.Linear(256, n_classes) # 
    

    def forward(self, x):
        # forward pass through convolution layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # ------- Flatten feature map -------
        # x shape: [batch_size, channels, height, width]
        # Flatten to get shape: [batch_size, channels*height*width]
        #            just like: (n_examples, features)
        x = x.reshape(x.size(0), -1)
        
        # forward pass through linear layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # use softmax if its specified to use softmax
        if self.softmax_use: x = F.softmax(x, dim=1)
            
        # return prediction of the model for the batch
        return x
    
#run_exercise(BloodMNIST_CNN, n_classes, False, "Q1_1_NoSoftmax")
run_exercise(BloodMNIST_CNN, n_classes, True, "Q1_1_YesSoftmax")



#--------------------------------------  Q1.2)  --------------------------------------

#define a new model with the required specifications
class BloodMNIST_CNN_MaxPool2d(nn.Module):
    def __init__(self, n_classes, softmax_use):
        super(BloodMNIST_CNN_MaxPool2d, self).__init__()

        self.softmax_use = softmax_use

        # MaxPooling layer applyed after each Convolution Layer
        # Now the feature map input of a conv layer is half in height and width
        # and channels is the same
        self.pool = nn.MaxPool2d(kernel_size=2)

        # layer 1: 
        #     -Input image is RGB, so 3 channels
        #     -Output: 32 channels feature map, specified in homework
        #     -Kernel 3x3, size=3
        #     -kernel slide with stride and padding 1
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # layer 2: 
        #     -Input: 32 channels feature map (/2 due to maxpooling)
        #     -Output: 64 channels feature map
        #     -Kernel 3x3, size=3
        #     -kernel slide with stride and padding 1
        # 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # layer 3: 
        #     -Input: 64 channels feature map
        #     -Output: 128 channels feature map
        #     -Kernel 3x3, size=3
        #     -kernel slide with stride and padding 1
        # 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Linear Layer 1:
        #     -Input: stride and padding 1 preserves the dimension of the height and width, and 
        # maxpooling applyed 3 times turns 28x28 -> 14x14 -> 7x7 -> 3x3, so the input feature
        # map is 3x3x128, meaning number of activations in feature map is 28x28x128,
        #     -Output: 256, specified in homework
        # 
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        # Linear Layer 2:
        #     -Input: 256, number of outputs of previous layer
        #     -Output: n_classes, to have a final classification of the input image of the model
        # 
        self.fc2 = nn.Linear(256, n_classes)
    

    def forward(self, x):
        # forward pass through convolution layers, appying maxpooling after relu activation
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # ------- Flatten feature map -------
        # x shape: [batch_size, channels, height, width]
        # Flatten to get shape: [batch_size, channels*height*width]
        #            just like: (n_examples, features)
        x = x.reshape(x.size(0), -1)
        
        # forward pass through linear layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # use softmax if its specified to use softmax
        if self.softmax_use: x = F.softmax(x, dim=1)
            
        # return prediction of the model for the batch
        return x
    
#run_exercise(BloodMNIST_CNN_MaxPool2d, n_classes, False, "Q1_2_NoSoftmax")
#run_exercise(BloodMNIST_CNN_MaxPool2d, n_classes, True, "Q1_2_YesSoftmax")