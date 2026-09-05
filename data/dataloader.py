"""
dataloader.py

Dataset loading utilities for SNN training.

Functions:
    mnistDataLoader: Load the MNIST dataset.
    fashionMnistDataLoader: Load the Fashion-MNIST dataset.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

def mnistDataLoader(batchSize, dataPath):
    """
    Load MNIST dataset with standard normalization.

    Args:
        batchSize (int): Batch size for DataLoader.
        dataPath (str or Path): Directory to store/load MNIST data.
        
    Returns:
        trainLoader, testLoader: PyTorch DataLoaders for training and testing.
    """

    transform = transforms.Compose([transforms.Resize((28,28)), transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize((0,), (1,))])

    trainSet = torchvision.datasets.MNIST(root=dataPath, train=True, download=True, transform=transform)
    testSet = torchvision.datasets.MNIST(root=dataPath, train=False, download=True, transform=transform)

    testSet = Subset(testSet, range(3000))

    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batchSize, shuffle=True, drop_last=True)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=batchSize, shuffle=True, drop_last=False)

    return trainLoader, testLoader

def fashionMnistDataLoader(batchSize, dataPath):
    """
    Load Fashion-MNIST dataset with standard normalization.

    Args:
        batchSize (int): Batch size for DataLoader.
        dataPath (str or Path): Directory to store/load Fashion-MNIST data.
        
    Returns:
        trainLoader, testLoader: PyTorch DataLoaders for training and testing.
    """

    transform = transforms.Compose([transforms.Resize((28,28)), transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize((0,), (1,))])

    trainSet = torchvision.datasets.FashionMNIST(root=dataPath, train=True, download=True, transform=transform)
    testSet = torchvision.datasets.FashionMNIST(root=dataPath, train=False, download=True, transform=transform)

    testSet = Subset(testSet, range(3000))

    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batchSize, shuffle=True, drop_last=True)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=batchSize, shuffle=True, drop_last=False)

    return trainLoader, testLoader
