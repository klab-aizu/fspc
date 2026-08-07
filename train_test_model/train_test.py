"""
train_test.py

Training, testing, and deployment utilities for snnTorch SNN models.

Functions:
    trainNetwork: Train an SNN model.
    testNetwork: Evaluate model accuracy.
    deployModel: Save trained model weights.
"""

import torch
from torch import nn
from torch.amp import autocast, GradScaler
import snntorch.functional as SF
from pathlib import Path

def trainNetwork(net, trainLoader, numEpoch, device):
    """
    Train a fully-connected snnTorch SNN.

    Args:
        net: snnTorch model.
        trainLoader: Training DataLoader.
        numEpoch: Number of training epochs.
        device: Torch device.

    Returns:
        None. The network is trained in-place.
    """

    net = net.to(device)
    numTimesteps = net.num_timesteps

    # Optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)

    # Loss function
    # Applied to membrane potentials at each timestep
    lossFn = nn.CrossEntropyLoss()

    # Mixed precision
    scaler = GradScaler()

    totalBatches = len(trainLoader)

    net.train()

    for epoch in range(numEpoch):

        runningLoss = 0.0
        runningCorrect = 0
        runningTotal = 0

        for iteration, (data, target) in enumerate(trainLoader):

            # Move data
            data = data.to(device, non_blocking=True).flatten(1)

            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            with autocast(device_type="cuda"):

                spkRec, memRec, _ = net(data)

                # memRec shape:
                # [numTimesteps, batch, classes]

                # Apply CE loss over all timesteps
                loss = lossFn(memRec.reshape(-1, memRec.size(-1)), target.repeat(numTimesteps))

            # Backpropagation
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            # Accuracy calculation
            with torch.no_grad():

                # Spike accumulation over time
                pred = spkRec.sum(0).argmax(1)
                correct = (pred == target).sum().item()
                batchAcc = (correct / target.size(0)) * 100

                runningLoss += loss.item()
                runningCorrect += correct
                runningTotal += target.size(0)

                runningAvgLoss = (runningLoss / (iteration + 1))
                runningAcc = (runningCorrect / runningTotal) * 100


            # Reset neurons if using persistent states
            if hasattr(net, "reset_states"):
                net.reset_states()

            # Logging
            currentBatch = iteration + 1

            if currentBatch % 100 == 0 or currentBatch == totalBatches:

                print(
                    f"Epoch [{epoch+1}/{numEpoch}] | "
                    f"Batch [{currentBatch}/{totalBatches}] | "
                    f"Loss: {loss.item():.4f} | "
                    f"Batch Acc: {batchAcc:.2f}% | "
                    f"Avg Loss: {runningAvgLoss:.4f} | "
                    f"Avg Acc: {runningAcc:.2f}% "
                    f"({runningCorrect}/{runningTotal})"
                )

        # Epoch summary
        epochLoss = runningLoss / totalBatches
        epochAcc = (runningCorrect / runningTotal) * 100

        print("\n" + "=" * 60)
        print(
            f"Epoch {epoch+1} Complete | "
            f"Loss: {epochLoss:.4f} | "
            f"Accuracy: {epochAcc:.2f}%"
        )
        print("=" * 60 + "\n")


def testNetwork(net, testLoader, device, modelName):
    """
    Evaluate a trained snnTorch SNN and optionally deploy the model.

    Args:
        net: Trained neural network instance.
        testLoader: PyTorch DataLoader for test data.
        device: Torch device for computation (CUDA/CPU).
        modelName: Name used for logging and model deployment.

    Returns:
        float: Final test accuracy percentage.
    """

    net.eval()

    correct = 0
    total = 0

    total_batches = len(testLoader)

    with torch.no_grad():

        for iteration, (data, target) in enumerate(testLoader):

            data = data.to(device, non_blocking=True).flatten(1)
            target = target.to(device, non_blocking=True)

            spkRec, memRec, _ = net(data)

            pred = spkRec.sum(0).argmax(1)

            correct += (pred == target).sum().item()
            total += target.size(0)

            running_acc = 100 * correct / total

            if (iteration + 1) % 100 == 0 or (iteration + 1) == total_batches:

                print(
                    f"Testing [{modelName}] | "
                    f"Batch [{iteration+1}/{total_batches}] | "
                    f"Running Acc: {running_acc:.2f}%"
                )

    final_acc = 100 * correct / total

    print(f"Final Accuracy: {final_acc:.2f}%")

    deployModel(modelName, net)


def deployModel(modelName, model):
    """
    Save a PyTorch model's state dictionary to a designated deployment folder.

    Args:
        modelName (str): Name of the model file. Automatically appends '.pt' if missing.
        model (torch.nn.Module): The PyTorch model instance to save.

    Returns:
        None. The model's state dictionary is saved to the 'deployedModels' directory.
    """

    deployPath = Path("deployed_models")

    # Ensure directory exists
    deployPath.mkdir(parents=True, exist_ok=True)
    
    # Ensure correct file extension
    if not modelName.endswith(".pt"):
        modelName += ".pt"
    
    # Full file path
    fullPath = deployPath / modelName
    
    print(f"Deploying model {modelName} to: {fullPath}")
    
    # Save model
    torch.save(model.state_dict(), fullPath)