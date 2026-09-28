"""
main.py

Central execution script for running SNN compression experiments via terminal.
Usage:
    python main.py <model>_<version>_<mining-func>_<dataset>

Examples:
    python main.py mlp_v1_fpmax_mnist
    python main.py mlp_v2_fpgrowth_mnist
    python main.py mlp_v1_maxfpgrowth_fmnist
    python main.py mlp_v2_fpmax_fashionmnist
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
    Parse experiment string in format: <model>_<version>_<mining-func>_<dataset>
    """
    parts = exp_str.strip().lower().split("_")
    if len(parts) < 4:
        raise ValueError(
            f"Invalid target format: '{exp_str}'.\n"
            f"Expected syntax: <model>_<version>_<mining-func>_<dataset>\n"
            f"Examples: mlp_v1_fpmax_mnist, mlp_v2_fpgrowth_fmnist"
        )

    model_type = parts[0]
    version = parts[1]
    dataset_name = parts[-1]
    # Join intermediate tokens to support miner names like pami_fpgrowth or fpmax
    miner_str = "_".join(parts[2:-1])

    # 1. Validate Model Type
    if model_type != "mlp":
        raise ValueError(f"Unsupported model type '{model_type}'. Supported: 'mlp'")

    # 2. Validate Version
    if version not in ("v1", "v2"):
        raise ValueError(f"Unsupported model version '{version}'. Supported: 'v1', 'v2'")

    # 3. Validate Mining Function
    miner_lookup = {
        "fpmax": compression.mlextendFpmax,
        "mlextendfpmax": compression.mlextendFpmax,
        "mlextend_fpmax": compression.mlextendFpmax,
        "fpgrowth": compression.pamiFpgrowth,
        "pamifpgrowth": compression.pamiFpgrowth,
        "pami_fpgrowth": compression.pamiFpgrowth,
        "maxfpgrowth": compression.pamiMaxFpgrowth,
        "pamimaxfpgrowth": compression.pamiMaxFpgrowth,
        "pami_maxfpgrowth": compression.pamiMaxFpgrowth,
    }

    if miner_str not in miner_lookup:
        raise ValueError(
            f"Unsupported mining function '{miner_str}'.\n"
            f"Supported miners: 'fpmax', 'fpgrowth', 'maxfpgrowth'"
        )
    minerFunc = miner_lookup[miner_str]

    # 4. Validate Dataset & Select Config + DataLoader
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

    # 5. Instantiate Selected Model
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

    # Model checkpoint is independent of the mining algorithm (avoids redundant training)
    checkpoint_name = f"{model_type}_{version}_{dataset_name}"
    # Full experiment folder name
    experiment_tag = f"{model_type}_{version}_{miner_str}_{dataset_name}"

    return {
        "model": model,
        "trainLoader": trainLoader,
        "testLoader": testLoader,
        "epoch": cfg.EPOCH,
        "datasetName": dataset_display,
        "networkType": networkType,
        "checkpointName": checkpoint_name,
        "experimentTag": experiment_tag,
        "minerFunc": minerFunc
    }


def runPipeline(trainLoader, testLoader, epoch, model, checkpointName, experimentTag, minerFunc, datasetName, networkType, fspList, pmtList, device):
    """
    Train or load a model, run experiments, and save all data inside metrics/<experimentTag>/.
    """
    model = model.to(device)

    # Directories
    deployed_dir = Path("deployed_models")
    deployed_dir.mkdir(parents=True, exist_ok=True)

    # Target folder: metrics/<model>_<version>_<mining-func>_<dataset>/
    experiment_metrics_dir = Path("metrics") / experimentTag
    experiment_metrics_dir.mkdir(parents=True, exist_ok=True)

    modelPath = deployed_dir / f"{checkpointName}.pt"

    # Train model if checkpoint does not exist
    if not modelPath.exists():
        print(f"\nModel checkpoint {modelPath} not found. Starting training for {epoch} epoch(s)...")
        trainNetwork(model, trainLoader, epoch, device)
        testNetwork(model, testLoader, device, checkpointName)
    else:
        print(f"\nFound existing checkpoint at {modelPath}.")

    print(f"Loading weights into {checkpointName}...")
    model.load_state_dict(torch.load(modelPath, map_location=device))

    # Output file relative path inside metrics/
    relative_csv_path = f"{experimentTag}/{experimentTag}.csv"

    print(f"\n[Starting Evaluation]")
    print(f"Destination Folder: {experiment_metrics_dir}/")
    print(f"Mining Algorithm:   {minerFunc.__name__}")

    # Execute compression experiments (saves CSV into metrics/<experimentTag>/)
    runExperiment(
        model=model, 
        testLoader=testLoader, 
        device=device, 
        fspList=fspList, 
        pmtList=pmtList, 
        compressionMode=True, 
        minerFunc=minerFunc, 
        filename=relative_csv_path
    )

    # Generate analytical graphs and text report inside metrics/<experimentTag>/
    print(f"\nGenerating plots and summary report in {experiment_metrics_dir}/...")
    visualizeResults(relative_csv_path, datasetName, networkType)


def main():
    parser = argparse.ArgumentParser(
        description="Run SNN compression experiments using syntax: <model>_<version>_<mining-func>_<dataset>"
    )
    parser.add_argument(
        "target",
        type=str,
        help="Experiment target in format <model>_<version>_<mining-func>_<dataset> (e.g., mlp_v1_fpmax_mnist)"
    )

    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parse CLI string
    exp = parse_experiment_string(args.target)

    # Hyperparameter grids
    fspList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pmtList = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Execute workflow
    runPipeline(
        trainLoader=exp["trainLoader"],
        testLoader=exp["testLoader"],
        epoch=exp["epoch"],
        model=exp["model"],
        checkpointName=exp["checkpointName"],
        experimentTag=exp["experimentTag"],
        minerFunc=exp["minerFunc"],
        datasetName=exp["datasetName"],
        networkType=exp["networkType"],
        fspList=fspList,
        pmtList=pmtList,
        device=device
    )


if __name__ == "__main__":
    main()
