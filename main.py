"""
main.py

Central execution script for running SNN compression experiments via terminal.
Usage:
    python main.py <model>_<version>_<dataset>

Examples:
    python main.py mlp_v1_mnist
    python main.py mlp_v2_mnist
    python main.py mlp_v1_fmnist
    python main.py mlp_v2_fashionmnist
"""

import sys
import argparse
from pathlib import Path
import torch

import config
from data import dataloader
from models.mlp_snn_v1 import MLP_SNN_V1
from models.mlp_snn_v2 import MLP_SNN_V2
from train_test_model.train_test import trainNetwork, testNetwork
from evaluation.evaluation import runExperiment, visualizeResults
from compression import compression


def parse_experiment_string(exp_str: str):
    """
    Parse experiment string in format <model>_<version>_<dataset>.
    Returns parsed identifiers and validated configuration objects.
    """
    parts = exp_str.strip().lower().split("_")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid target format: '{exp_str}'.\n"
            f"Expected syntax: <model>_<version>_<dataset> (e.g., mlp_v1_mnist, mlp_v2_fmnist)"
        )

    model_type, version, dataset_name = parts

    # 1. Validate Model Type
    if model_type != "mlp":
        raise ValueError(f"Unsupported model type '{model_type}'. Supported: 'mlp'")

    # 2. Validate Version
    if version not in ("v1", "v2"):
        raise ValueError(f"Unsupported model version '{version}'. Supported: 'v1', 'v2'")

    # 3. Validate Dataset & Select Config + DataLoader
    if dataset_name in ("mnist",):
        cfg = config.MNIST
        dataset_display = "MNIST"
        trainLoader, testLoader = dataloader.mnistDataLoader(cfg.BATCHSIZE, cfg.DATAPATH)
    elif dataset_name in ("fmnist", "fashionmnist"):
        cfg = config.FashionMNIST
        dataset_display = "Fashion-MNIST"
        trainLoader, testLoader = dataloader.fashionMnistDataLoader(cfg.BATCHSIZE, cfg.DATAPATH)
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: 'mnist', 'fmnist', 'fashionmnist'")

    # 4. Instantiate Selected Model
    if version == "v1":
        model = MLP_SNN_V1(
            numInputs=cfg.INPUT,
            numHidden1=cfg.HIDDEN1,
            numOutputs=cfg.OUTPUT,
            numTimesteps=cfg.TIMESTEPS,
            beta=cfg.BETA,
            minSup=config.CommonConfig.MINSUP_1
        )
        networkType = f"{cfg.INPUT}-{cfg.HIDDEN1}-{cfg.OUTPUT} MLP SNN V1"
    else:  # v2
        model = MLP_SNN_V2(
            numInputs=cfg.INPUT,
            numHidden1=cfg.HIDDEN1,
            numHidden2=cfg.HIDDEN2,
            numOutputs=cfg.OUTPUT,
            numTimesteps=cfg.TIMESTEPS,
            beta=cfg.BETA,
            minSup=config.CommonConfig.MINSUP_1
        )
        networkType = f"{cfg.INPUT}-{cfg.HIDDEN1}-{cfg.HIDDEN2}-{cfg.OUTPUT} MLP SNN V2"

    return {
        "model": model,
        "trainLoader": trainLoader,
        "testLoader": testLoader,
        "epoch": cfg.EPOCH,
        "datasetName": dataset_display,
        "networkType": networkType,
        "modelName": f"{model_type}_{version}_{dataset_name}"
    }


def runPipeline(trainLoader, testLoader, epoch, model, modelName, minerFunc, datasetName, networkType, fspList, pmtList, device):
    """
    Train or load a model, run compression experiments, and generate visualizations.
    """
    model = model.to(device)

    # Ensure directories exist
    deployed_dir = Path("deployed_models")
    deployed_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    modelPath = deployed_dir / f"{modelName}.pt"

    # Train model if checkpoint does not exist, otherwise load saved weights
    if not modelPath.exists():
        print(f"\nModel checkpoint {modelPath} not found. Starting training for {epoch} epoch(s)...")
        trainNetwork(model, trainLoader, epoch, device)
        testNetwork(model, testLoader, device, modelName)
    else:
        print(f"\nFound existing checkpoint at {modelPath}.")

    print(f"Loading weights into {modelName}...")
    model.load_state_dict(torch.load(modelPath, map_location=device))

    outputFile = f"{modelName}.csv"

    # Execute compression experiments
    print(f"\nStarting compression experiments for {modelName}...")
    runExperiment(
        model=model, 
        testLoader=testLoader, 
        device=device, 
        fspList=fspList, 
        pmtList=pmtList, 
        compressionMode=True, 
        minerFunc=minerFunc, 
        filename=outputFile
    )

    # Generate analytical graphs and text report
    print(f"\nGenerating visualization plots and summary reports...")
    visualizeResults(outputFile, datasetName, networkType)


def main():
    parser = argparse.ArgumentParser(
        description="Run SNN compression experiments using syntax: <model>_<version>_<dataset>"
    )
    parser.add_argument(
        "target",
        type=str,
        help="Experiment target in format <model>_<version>_<dataset> (e.g., mlp_v1_mnist, mlp_v2_fmnist)"
    )
    parser.add_argument(
        "--miner",
        type=str,
        default="pami_fpgrowth",
        choices=["pami_fpgrowth", "mlextend_fpmax", "pami_max_fpgrowth"],
        help="Pattern mining algorithm to use for FSPC (default: pami_fpgrowth)"
    )

    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Map miner CLI string to compression functions
    miner_map = {
        "pami_fpgrowth": compression.pamiFpgrowth,
        "mlextend_fpmax": compression.mlextendFpmax,
        "pami_max_fpgrowth": compression.pamiMaxFpgrowth,
    }
    selectedMiner = miner_map[args.miner]

    # Parse and build pipeline setup
    exp = parse_experiment_string(args.target)

    # Hyperparameter grids
    fspList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pmtList = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Run complete workflow
    runPipeline(
        trainLoader=exp["trainLoader"],
        testLoader=exp["testLoader"],
        epoch=exp["epoch"],
        model=exp["model"],
        modelName=exp["modelName"],
        minerFunc=selectedMiner,
        datasetName=exp["datasetName"],
        networkType=exp["networkType"],
        fspList=fspList,
        pmtList=pmtList,
        device=device
    )


if __name__ == "__main__":
    main()
