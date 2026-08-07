"""
config.py

Model and dataset configuration parameters for SNN experiments.

Classes:
    CommonConfig:
        Shared training and dataset settings.

    MNIST:
        Network configuration for the MNIST dataset.

    FashionMNIST:
        Network configuration for the Fashion-MNIST dataset.
"""

class CommonConfig:
    BATCHSIZE = 128
    DATAPATH = "data"
    MINSUP_1 = 0.25
    MINSUP_2 = 0.35
    BETA = 0.95
    TIMESTEPS = 20
    EPOCH = 1

class MNIST(CommonConfig):
    INPUT = 28*28
    HIDDEN1 = 100
    HIDDEN2 = 60
    OUTPUT = 10

class FashionMNIST(CommonConfig):
    INPUT = 28*28
    HIDDEN1 = 100
    HIDDEN2 = 60
    OUTPUT = 10
