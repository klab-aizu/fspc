import torch
from pathlib import Path
from data import dataloader

import config 
from models.mlp_snn_v1 import MLP_SNN_V1
from models.mlp_snn_v2 import MLP_SNN_V2
from train_test_model.train_test import trainNetwork, testNetwork
from evaluation.evaluation import runExperiment, visualizeResults
from compression import compression

def main():

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    def runPipeline(trainLoader, testLoader, epoch, model, modelName, minerFunc, datasetName, networkType, fspList=None, pmtList=None, device="cpu"):
        """
        Train or load a model, run experiments, and visualize results.

        Args:
            trainLoader: DataLoader for training data.
            testLoader: DataLoader for test data.
            epoch: Number of training epochs.
            model: Neural network model to train/evaluate.
            modelName: Model checkpoint name.
            minerFunc: Function for mining spike patterns.
            datasetName: String name of the dataset.
            networkType: String name of the network type.
            fspList: Frequent spike pattern configurations.
            pmtList: Pattern matching threshold configurations.
            device: Execution device (CPU/GPU).

        Returns:
            None:
                Saves experiment results and generates visualizations.
        """

        # Add the ".pt" extension
        modelName += ".pt"

        # Resolve model file path
        modelPath = Path("deployed_models") / modelName

        # Train model if non-existent, otherwise load saved state
        if not modelPath.exists():
            trainNetwork(model, trainLoader, epoch, device)
            testNetwork(model, testLoader, device, modelName)

        print(f"Loading model {modelName}...")
        model.load_state_dict(torch.load(modelPath, map_location=device))

        outputFile = f"{modelName}.csv"

        # Execute experiment and save metrics
        runExperiment(model, testLoader, device, fspList, pmtList, True, minerFunc, outputFile)

        # Visualize Results
        visualizeResults(outputFile, datasetName, networkType)

    mnistTrainLoader, mnistTestLoader = dataloader.mnistDataLoader(config.MNIST.BATCHSIZE, config.MNIST.DATAPATH)

    fspList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pmtList = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    modelName = "snn_mlp_mnist"

    snnMlp = MLP_SNN_V1(config.MNIST.INPUT, config.MNIST.HIDDEN1, config.MNIST.OUTPUT, 
                        config.MNIST.TIMESTEPS, config.MNIST.BETA)

    runPipeline(mnistTrainLoader, mnistTestLoader, config.MNIST.EPOCH, snnMlp, modelName, 
                compression.pamiFpgrowth, "MNIST", "784-100-10 MLP SNN", fspList, pmtList, device)


if __name__ == "__main__":
    main()

