import albumentations as A
import PIL
from PIL import Image, ImageOps, ImageEnhance
from PIL import __version__ as PILLOW_VERSION
PIL.PILLOW_VERSION = PIL.__version__
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import math
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
class PyTorchDataset(datasets.CIFAR10):

    def __init__(self, root="~/data", train=True, download=True, transform=None):

        super().__init__(root=root, train=train, download=download, transform=transform)

        self.transform = transform

    def __getitem__(self, index):

        image, label = self.data[index], self.targets[index]

        if self.transform is not None:

            transformed = self.transform(image=image)

            image = transformed["image"]

        return image, label


def train_testloader(dataloader_args,train,test):

    train_dataset = PyTorchDataset(root='./data/cifar_10', train=True, transform=train, download=True)
    test_dataset = PyTorchDataset(root='./data/cifar_10', train=False, transform=test, download=True)

    # train dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)

    # classes in CIFAR 10 dataset
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_dataset,test_dataset,train_loader, test_loader, classes



def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


def transform(means=[0.4914, 0.4822, 0.4465],stds=[0.2470, 0.2435, 0.2616]):
    train_transforms = A.Compose(
    [
        A.Normalize(mean=means, std=stds, always_apply=True),
        A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
        A.RandomCrop(height=32, width=32, always_apply=True),
        A.HorizontalFlip(),
        A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=means),
        ToTensorV2(),
    ]
    )

    test_transforms = A.Compose(
        [
            A.Normalize(mean=means, std=stds, always_apply=True),
            ToTensorV2(),
        ]
    )

    return train_transforms, test_transforms


# Find 10 misclassified images, and show them as a 5x2 image matrix in 3 separately annotated images. 

def get_misclassified_data(model, device, test_loader):
    
    # Prepare the model for evaluation
    model.eval()

    # List for storing misclassified images
    misclassified_data = []

    # Reset the Gradients
    with torch.no_grad():
        # Extract images, labels in a batch
        for data, target in test_loader:
            # Move the data to device
            data,target = data.to(device),target.to(device)

            # Extract single batch of images, labels
            for img,label in zip(data,target):
                
                # Add a batch dimension
                img = img.unsqueeze(0)
                # Get prediction
                output = model(img)
                # Convert output probabilities to predicted class through one-hot encoding
                pred = output.argmax(dim=1, keepdim=True)

                # Compare prediction and true label
                if pred.item() != label.item():
                    misclassified_data.append((img, label, pred))

    return misclassified_data


def display_gradcam_output(misclass_data,
                           classes,
                           model,
                           inv_normalize: transforms.Normalize,
                           target_layers: list['model_layer'],
                           targets=None,
                           no_samples:int = 10,
                           transparence:float = 0.60):
    """
        Function to visualize GradCam output on the data
    :param data: List[Tuple(image, label)]
    :param classes: Name of classes in the dataset
    :param inv_normalize: Mean and Standard deviation values of the dataset
    :param model: Model architecture
    :param target_layers: Layers on which GradCam should be executed
    :param targets: Classes to be focused on for GradCam
    :param number_of_samples: Number of images to print
    :param transparency: Weight of Normal image when mixed with activations
    """

    # Plot configuration
    fig = plt.figure(figsize=(10, 10))
    x_count = 5
    if no_samples is None:
        y_count = 1
    else:
        y_count = math.floor(no_samples / x_count)

    # Create an object for GradCAM
    cam = GradCAM(model=model, target_layers=target_layers,use_cuda=True)

    # Iterate over number of specified images
    for i in range(no_samples):
        plt.subplot(y_count,x_count,i+1)
        input_tensor = misclass_data[i][0]

        # Get the activations of the layer for the images
        grayscale_cam = cam(input_tensor=input_tensor,targets=targets)
        grayscale_cam = grayscale_cam[0,:]



        # Get the original image
        img = input_tensor.squeeze(0).to('cpu')
        img = inv_normalize(img)
        rgb_img = np.transpose(img.cpu().numpy(), [1, 2, 0])

        # mix the normal image with the activations
        visualization = show_cam_on_image(rgb_img, grayscale_cam,use_rgb=True,image_weight=transparence)

        # Plot the GradCAM output along with the original image
        plt.imshow(visualization)
        plt.title("Pred: {} Act: {}".format(classes[misclass_data[i][2].item()],classes[misclass_data[i][1].item()]))
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()



def plot_acc(train_acc,test_acc,train_losses,test_losses):
    t = [t_items.item() for t_items in train_losses]
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(t)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    plt.savefig('acc-loss.png')