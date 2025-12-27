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


device = "cuda" if torch.cuda.is_available() else "cpu"


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

#Defining the CNN model with the nn.Module class, as suggested. This class will define
# the CNN fixed as the indicated in the homework
class BloodMNIST_CNN(nn.Module):
    def __init__(self, n_classes):
        super(BloodMNIST_CNN, self).__init__()
        """layer 1: 
            -Input image is RGB, so 3 channels
            -Output: 32 channels feature map
            -Kernel 3x3, size=3
            -kernel slide with stride and padding 1
        """
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        """layer 2: 
            -Input: 32 channels feature map
            -Output: 64 channels feature map
            -Kernel 3x3, size=3
            -kernel slide with stride and padding 1
        """
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        """layer 3: 
            -Input: 64 channels feature map
            -Output: 128 channels feature map
            -Kernel 3x3, size=3
            -kernel slide with stride and padding 1
        """
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        """Linear Layer 1:
            -Input: stride and padding 1 preserves the dimension of the height and width, so
        Input feature map is 28x28x128, meaning number of activations in feature map is 28x28x128,
            -Output: 256, specified in homework
        """
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        """Linear Layer 2:
            -Input: 256, number of outputs of previous layer
            -Output: n_classes, to have a final classification of the input image of the model
        """
        self.fc2 = nn.Linear(256, n_classes) # 
    

    def forward(self, x, use_softmax=False):
        # forward pass through convolution layers
        x = nn.ReLU(self.conv1(x))
        x = nn.ReLU(self.conv2(x))
        x = nn.ReLU(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Linear block
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        if use_softmax:
            x = F.softmax(x, dim=1)
            
        return x

#Training Function

def train_epoch(loader, model, criterion, optimizer):

    model.train() # Set model in training mode
    total_loss = 0.0 # intiialize loss sum

    # For each abtch: forward pass, loss calculation, backpropagation, model update
    for imgs, labels in loader:
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
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


#hyperparameters
batch_size = 64
epochs = 30
lr = 0.001

train_dataset = BloodMNIST(split='train', transform=transform, download=True, size=28)
val_dataset   = BloodMNIST(split='val',   transform=transform, download=True, size=28)
test_dataset  = BloodMNIST(split='test',  transform=transform, download=True, size=28)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# initialize the model
# get an optimizer
# get a loss criterion

### YOUR CODE HERE ###

# training loop
### you can use the code below or implement your own loop ###
train_losses = []
val_accs = []
test_accs = []
for epoch in range(epochs):

    epoch_start = time.time()

    train_loss = train_epoch(train_loader, model, criterion, optimizer)
    val_acc = evaluate(val_loader, model)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

    train_losses.append(train_loss)
    val_accs.append(val_acc)

    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | "
          f"Time: {epoch_time:.2f} sec")

#Test Accuracy
test_acc = evaluate(test_loader, model)
print("Test Accuracy:", test_acc)
test_accs.append(test_acc)


#Save the model
#torch.save(model.state_dict(), "bloodmnist_cnn.pth")
#print("Model saved as bloodmnist_cnn.pth")


# --------- After Training ----------
total_end = time.time()
total_time = total_end - total_start

print(f"\nTotal training time: {total_time/60:.2f} minutes "
      f"({total_time:.2f} seconds)")

#print('Final Test acc: %.4f' % (evaluate(model, test_X, test_y)))

#config = "{}-{}-{}-{}-{}".format(opt.learning_rate, opt.optimizer, opt.no_maxpool, opt.no_softmax,)
config = "{}".format(str(0.1))

plot(epochs, train_losses, ylabel='Loss', name='CNN-training-loss-{}'.format(config))
plot(epochs, val_accs, ylabel='Accuracy', name='CNN-validation-accuracy-{}'.format(config))
plot(epochs, test_accs, ylabel='Accuracy', name='CNN-test-accuracy-{}'.format(config))