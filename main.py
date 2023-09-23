from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import os
import argparse
from torch.optim.lr_scheduler import OneCycleLR
from torch_lr_finder import LRFinder
from models.resnet import *
from utils import *

# Call the functions from utils.py and models/resnet.py

def train_test_split(dataloader_arg=dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True)):
    train_transforms, test_transforms = transform()
    dataloader_args = dataloader_arg
    train_dataset,test_dataset,train_loader, test_loader, classes = train_testloader(dataloader_args,train_transforms,test_transforms)
    return train_dataset,test_dataset,train_loader, test_loader, classes


def select_ResNet(device=get_device()):
    net = ResNet18()
    net = net.to(device)
    model_summary(net, (3,32,32))
    return net

# --------------------------------------------------------------------------------------------------------------------------------------------------------------

# Choice of Optimizer Function

def optimizer(net, lr=0.01, momentum=0.9, weight_decay=5e-4, optim_input='SGD'):
    optim_dict = {
        'SGD': optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay),
        'Adam': optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay),
        'RMSprop': optim.RMSprop(net.parameters(), lr=lr)
    }

    optimizer = optim_dict[optim_input]
    return optimizer


# --------------------------------------------------------------------------------------------------------------------------------------------------------------

# LR Finder


def find_max_lr(net, optimizer_, train_loader,step_mode='exp',num_iter=100, device=get_device()):
    criterion = nn.CrossEntropyLoss()
    lr_finder = LRFinder(net, optimizer_, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=10, num_iter=num_iter,step_mode=step_mode)
    _, max_lr = lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset() # to reset the model and optimizer to their initial state
    return max_lr


# --------------------------------------------------------------------------------------------------------------------------------------------------------------

# Train and Test

train_losses = []
train_acc = []
lr_lst = []




def get_lr(optimizer):
    """"
    for tracking how your learning rate is changing throughout training
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_incorrect_predictions(pred, target):
    pred = pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    indices = (pred.eq(target.view_as(pred)) == False).nonzero()[:, 0].tolist()
    return indices, pred[indices], target.view_as(pred)[indices]

def train(model, device, train_loader, optimizer, epoch,criterion,scheduler=None):
    step_scheduler = lambda: scheduler.step() if scheduler else None
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # print("data shape", data.shape)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data).to(device)

        # print("y_pred shape", y_pred.shape)

        # Calculate loss
        loss = criterion(y_pred, target)
        train_loss += loss.item() * len(data)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm
    
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        step_scheduler()
        
    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/processed)
    lr_ = lambda: lr_lst.append(max(step_scheduler().get_last_lr())) if scheduler else None
    lr_()

test_losses = []
test_acc = []
misclassified_imgs = []
misclassified_labels = []
misclass_pred = []

def test(model, device, test_loader,criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Find 10 misclassified images for the BN model, and show them as a 5x2 image matrix
            misclass_mask = (pred.eq(target.view_as(pred)) == False).nonzero()[:, 0]
            misclassified_imgs.extend(data[misclass_mask])
            misclassified_labels.extend(target.view_as(pred)[misclass_mask])
            misclass_pred.extend(pred[misclass_mask])

                  



    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))

    return test_acc, test_losses, misclassified_imgs, misclassified_labels, misclass_pred


# Train-Test Loop
def model_execute(model,device,train_loader,test_loader,max_LR,optim,max_lr_epoch=5,criterion=nn.CrossEntropyLoss(),optimizer="Adam",EPOCH=20,lr_scheduler="One Cycle Policy",lr=0.03):
    scheduler = lambda: OneCycleLR(optim, max_lr=max_LR, epochs=max_lr_epoch, steps_per_epoch=len(train_loader)) if scheduler == "One Cycle Policy" else None
    scheduler()
    for epoch in range(EPOCH):
        print("EPOCH:", epoch)
        train(model, device, train_loader, optim, epoch,criterion,scheduler=scheduler())
        test_acc, test_losses, misclassified_imgs, misclassified_labels, misclass_pred = test(model, device, test_loader,criterion)
    return test_acc, test_losses, misclassified_imgs, misclassified_labels, misclass_pred