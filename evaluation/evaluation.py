"""
evaluation.py

Utilities for evaluating model performance and analyzing spike compression.

Functions:
runInference: Run inference and collect accuracy, compression, runtime, and memory metrics.

runExperiment: Evaluate multiple compression parameter configurations and save results.

visualizeResults: Generate evaluation plots and experiment summary reports.
"""

from statistics import mean
import time
import pandas as pd
import numpy as np
import torch
import psutil
import platform
import matplotlib.pyplot as plt
from pathlib import Path

def runInference(model, testLoader, device, numFSP=None, pmt=None, compressionMode=False, minerFunc=None):
    """
    Run model inference on a test dataset with optional spike compression.

    Args:
        model: Neural network model used for inference.
        testLoader: DataLoader providing test samples and labels.
        device: Hardware device used for model execution (CPU/GPU).
        numFSP: Number of frequent spike patterns used for compression.
        pmt: Pattern matching threshold for spike pattern compression.
        compressionMode: Enables or disables spike compression.
        minerFunc: Function used to mine frequent spike patterns.

    Returns:
        tuple:
            Accuracy percentage, mean compression ratio, mean runtime per
            batch, mean memory usage per batch, and compression method name.
    """
        
    # Initialize evaluation metrics counters and storage lists
    total = 0
    correct = 0

    batchCompressions = []
    batchRuntime = []
    batchMemory = []

    # Prepare model for evaluation on specified device
    model.to(device)
    model.eval()

    totalBatches = len(testLoader)

    # Print run configuration details
    if compressionMode:
        print(f"\nTest - Compression Enabled with {minerFunc.__name__} (Total Batches: {totalBatches})")
    else:
        print(f"\nBaseline - Compression Disabled (Total Batches: {totalBatches})")

    # Disable gradient tracking for inference
    with torch.no_grad():

        for batchIdx, (data, targets) in enumerate(testLoader, start=1):

            print(f"Processing batch: {batchIdx}/{totalBatches}")

            # Move inputs to target device and flatten batch
            data = data.to(device).view(data.size(0), -1)
            targets = targets.to(device)

            # Perform forward pass
            spkOut, _, metrics = model(data, compressionMode, minerFunc, numFSP, pmt)

            # Record batch performance metrics if reported
            if metrics:
                batchCompressions.append(metrics["reduction_len"])
                batchRuntime.append(metrics["runtime"])
                batchMemory.append(metrics["rss_mem"])

            # Determine predicted classes from spike output
            _, predicted = spkOut.sum(dim=0).max(1)

            # Track accuracy statistics
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    # Calculate final dataset accuracy and average batch metrics
    accuracy = (correct / total) * 100

    meanCompression = mean(batchCompressions) if batchCompressions else 0
    meanRuntime = mean(batchRuntime) if batchRuntime else 0
    meanMemory = mean(batchMemory) if batchMemory else 0

    name = minerFunc.__name__ if minerFunc else "Baseline"

    return accuracy, meanCompression, meanRuntime, meanMemory, name

def runExperiment(model, testLoader, device, fspList, pmtList, compressionMode, minerFunc, filename):
    """
    Evaluate compression performance across multiple FSP and PMT configurations.

    Args:
        model: Neural network model evaluated during experiments.
        testLoader: DataLoader providing test samples and labels.
        device: Hardware device used for model execution (CPU/GPU).
        fspList: List of frequent spike pattern counts to evaluate.
        pmtList: List of pattern matching thresholds to evaluate.
        compressionMode: Enables or disables spike compression.
        minerFunc: Function used for mining frequent spike patterns.
        filename: Output CSV filename for storing experiment metrics.

    Returns:
        None:
            Saves experiment results, including compression metrics and
            baseline performance, to a CSV file.
    """

    results = []

    # Grid search across all combinations of FSP and PMT hyperparameter values
    for fsp in fspList:

        for pmt in pmtList:
            # Run inference for current parameter pair
            accuracy, meanCompression, meanRuntime, meanMemory, name = runInference(model, testLoader, device, fsp, pmt, compressionMode, minerFunc)

            # Store metrics for current parameter configuration
            results.append({
                f"FSP Count": fsp,
                f"PMT Value": pmt,
                f"{name} Accuracy": accuracy,
                f"{name} Mean Compression": meanCompression,
                f"{name} Mean Runtime per Batch (s)": meanRuntime,
                f"{name} Mean Memory per Batch (bytes)": meanMemory 
            })

    # Measure baseline run without compression
    timeStart = time.time()

    accBase, _, _, _, _ = runInference(model, testLoader, device)

    timeEnd = time.time()

    baseElapsedTime = timeEnd - timeStart

    print(f"Baseline Accuracy: {accBase:.3f}%")
    print(f"Baseline Runtime: {baseElapsedTime:.3f} secs")

    # Combine experiment results and append baseline reference values
    dfResults = pd.DataFrame(results)
    dfResults["Baseline_Accuracy"] = accBase
    dfResults["Baseline_Runtime"] = baseElapsedTime

    # Ensure output directory exists
    metrics_folder = Path("metrics")
    metrics_folder.mkdir(exist_ok=True)

    # Save metrics to CSV file
    save_path = metrics_folder / filename
    dfResults.to_csv(save_path, index=False)

    print(f"Done! CSV saved as {save_path}")

def visualizeResults(filename, datasetName, networkType):
    """
    Generate visual analysis plots and summary reports from experiment metrics.

    Args:
        filename: CSV file containing saved experiment results.
        datasetName: Name of the dataset used in the experiment.
        networkType: Type or architecture name of the evaluated neural network.

    Returns:
        None:
            Generates and saves accuracy-compression plots, runtime-memory
            analysis plots, and a text summary report containing experiment
            results and system specifications.
    """

    # Load experiment metrics
    metrics_folder = Path("metrics")
    file_path = metrics_folder / filename
    
    if not file_path.exists():
        print(f"Error: {file_path} does not exist.")
        return

    df = pd.DataFrame(pd.read_csv(file_path))

    # Dynamically extract full column names and model/miner name prefix ({name})
    acc_col = [c for c in df.columns if "Accuracy" in c and "Baseline" not in c][0]
    comp_col = [c for c in df.columns if "Mean Compression" in c][0]
    runtime_col = [c for c in df.columns if "Mean Runtime" in c][0]
    memory_col = [c for c in df.columns if "Mean Memory" in c][0]

    # Extract {name} prefix directly from column header
    model_name = acc_col.replace(" Accuracy", "").strip()

    baseline_acc = df["Baseline_Accuracy"].iloc[0]

    # Calculate tradeoff metric (normalized harmonic mean of accuracy and compression)
    norm_acc = (df[acc_col] - df[acc_col].min()) / (df[acc_col].max() - df[acc_col].min() + 1e-8)
    norm_comp = (df[comp_col] - df[comp_col].min()) / (df[comp_col].max() - df[comp_col].min() + 1e-8)
    df["Tradeoff_Score"] = 2 * (norm_acc * norm_comp) / (norm_acc + norm_comp + 1e-8)

    # 1. Best PMT overall and per FSP analysis
    best_row = df.loc[df["Tradeoff_Score"].idxmax()]
    print(f"Best Overall Combo -> FSP: {best_row['FSP Count']}, PMT: {best_row['PMT Value']}")

    # 2. Graph Setup (2 Combination Plots: FSP variable & PMT variable)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # --- Plot A: Variable FSP (Keeping PMT constant at lowest value) ---
    base_pmt = df["PMT Value"].min()
    df_fsp_var = df[df["PMT Value"] == base_pmt].sort_values("FSP Count")

    ax1 = axes[0]
    ax2 = ax1.twinx()

    b1 = ax1.bar(df_fsp_var["FSP Count"].astype(str), df_fsp_var[comp_col], color="skyblue", alpha=0.6, label=comp_col)
    l1 = ax2.plot(df_fsp_var["FSP Count"].astype(str), df_fsp_var[acc_col], color="crimson", marker="o", linewidth=2, label=acc_col)
    l2 = ax2.axhline(y=baseline_acc, color="black", linestyle="--", linewidth=1.5, label=f"Baseline Accuracy ({baseline_acc:.2f}%)")

    ax2.text(0.01, baseline_acc, f" Baseline: {baseline_acc:.2f}%", transform=ax2.get_yaxis_transform(), va='bottom', ha='left', color='black', fontsize=9, fontweight='bold')

    # Dynamic Vertical Separation (Plot A)
    max_comp_a = df_fsp_var[comp_col].max()
    if max_comp_a > 0:
        ax1.set_ylim(0, max_comp_a * 2.2)  # Bars occupy bottom ~45%

    min_acc_a = min(df_fsp_var[acc_col].min(), baseline_acc)
    max_acc_a = max(df_fsp_var[acc_col].max(), baseline_acc)
    range_acc_a = max_acc_a - min_acc_a if max_acc_a != min_acc_a else 1.0
    ax2.set_ylim(min_acc_a - range_acc_a * 1.1, max_acc_a + range_acc_a * 0.2)  # Line occupies top ~50%

    ax1.set_xlabel("FSP Count")
    ax1.set_ylabel("Compression (%)", color="skyblue")
    ax2.set_ylabel("Accuracy (%)", color="crimson")
    ax1.set_title(f"{datasetName}: FSP Variation (Constant PMT = {base_pmt})")

    # Combined Legend placed above/outside Plot A
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=True)

    # --- Plot B: Variable PMT (Keeping FSP constant at 1) ---
    df_pmt_var = df[df["FSP Count"] == 1].sort_values("PMT Value")

    ax3 = axes[1]
    ax4 = ax3.twinx()

    b2 = ax3.bar(df_pmt_var["PMT Value"].astype(str), df_pmt_var[comp_col], color="lightgreen", alpha=0.6, label=comp_col)
    l3 = ax4.plot(df_pmt_var["PMT Value"].astype(str), df_pmt_var[acc_col], color="darkorange", marker="s", linewidth=2, label=acc_col)
    l4 = ax4.axhline(y=baseline_acc, color="black", linestyle="--", linewidth=1.5, label=f"Baseline Accuracy ({baseline_acc:.2f}%)")

    ax4.text(0.01, baseline_acc, f" Baseline: {baseline_acc:.2f}%", transform=ax4.get_yaxis_transform(), va='bottom', ha='left', color='black', fontsize=9, fontweight='bold')

    # Dynamic Vertical Separation (Plot B)
    max_comp_b = df_pmt_var[comp_col].max()
    if max_comp_b > 0:
        ax3.set_ylim(0, max_comp_b * 2.5)  

    min_acc_b = min(df_pmt_var[acc_col].min(), baseline_acc)
    max_acc_b = max(df_pmt_var[acc_col].max(), baseline_acc)
    range_acc_b = max_acc_b - min_acc_b if max_acc_b != min_acc_b else 1.0
    ax4.set_ylim(min_acc_b - range_acc_b * 1.5, max_acc_b + range_acc_b * 0.5)  # Line occupies top ~50%

    ax3.set_xlabel("PMT Value")
    ax3.set_ylabel("Compression (%)", color="lightgreen")
    ax4.set_ylabel("Accuracy (%)", color="darkorange")
    ax3.set_title(f"{datasetName}: PMT Variation (Constant FSP = 1)")

    # Combined Legend placed above/outside Plot B
    handles3, labels3 = ax3.get_legend_handles_labels()
    handles4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(handles3 + handles4, labels3 + labels4, loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=True)

    plt.tight_layout()
    plots_save_path = metrics_folder / f"{datasetName}_accuracy_compression_analysis.png"
    plt.savefig(plots_save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Supplementary Graph: Runtime and Memory Analysis
    fig, ax_rt = plt.subplots(figsize=(11, 6))
    ax_mem = ax_rt.twinx()

    labels = [f"FSP:{r['FSP Count']}|PMT:{r['PMT Value']}" for _, r in df.iterrows()]
    x_indices = np.arange(len(labels))

    l_rt = ax_rt.plot(x_indices, df[runtime_col], color="purple", marker="^", label=runtime_col)
    l_mem = ax_mem.plot(x_indices, df[memory_col] / (1024 ** 2), color="teal", marker="d", linestyle=":", label=f"{model_name} Mean Memory per Batch (MB)")

    ax_rt.set_xticks(x_indices[::max(1, len(labels)//10)])
    ax_rt.set_xticklabels(labels[::max(1, len(labels)//10)], rotation=45)
    ax_rt.set_xlabel("FSP | PMT Combination")
    ax_rt.set_ylabel(f"{runtime_col} (s)", color="purple")
    ax_mem.set_ylabel(f"{model_name} Mean Memory per Batch (MB)", color="teal")
    ax_rt.set_title(f"{datasetName}: Runtime & Memory Analysis", pad=35)

    # Combined Legend placed to the right side outside the plot
    h_rt, lab_rt = ax_rt.get_legend_handles_labels()
    h_mem, lab_mem = ax_mem.get_legend_handles_labels()
    ax_rt.legend(h_rt + h_mem, lab_rt + lab_mem, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=True)

    plt.tight_layout()
    supp_save_path = metrics_folder / f"{datasetName}_runtime_memory_analysis.png"
    plt.savefig(supp_save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 4. System Specs Extraction and Text File Export
    cpu_info = platform.processor() or platform.machine()
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected"

    report_path = metrics_folder / f"{datasetName}_summary_report.txt"
    with open(report_path, "w") as f:
        f.write("=====================================================\n")
        f.write(f"          EXPERIMENT ANALYSIS REPORT: {datasetName}\n")
        f.write("=====================================================\n\n")
        
        f.write("--- PC SYSTEM HARDWARE SPECIFICATIONS ---\n")
        f.write(f"OS: {platform.system()} {platform.release()}\n")
        f.write(f"CPU: {cpu_info}\n")
        f.write(f"RAM: {ram_gb} GB\n")
        f.write(f"GPU: {gpu_info}\n\n")

        f.write("--- BEST ACCURACY-COMPRESSION TRADEOFF COMBINATION ---\n")
        f.write(f"Neural Network Type: {networkType}\n")
        f.write(f"FSP Count: {best_row['FSP Count']}\n")
        f.write(f"PMT Value: {best_row['PMT Value']}\n")
        f.write(f"{acc_col}: {best_row[acc_col]:.3f}%\n")
        f.write(f"Baseline Accuracy: {baseline_acc:.3f}%\n")
        f.write(f"{comp_col}: {best_row[comp_col]:.4f}%\n")
        f.write(f"{runtime_col}: {best_row[runtime_col]:.4f} sec\n")
        f.write(f"{memory_col}: {best_row[memory_col]:.2f} bytes ({best_row[memory_col] / (1024**2):.2f} MB)\n\n")

        f.write("--- OVERALL EXPERIMENT METRICS SUMMARY ---\n")
        f.write(f"Overall {runtime_col}: {df[runtime_col].mean():.4f} sec\n")
        f.write(f"Overall {memory_col}: {df[memory_col].mean():.2f} bytes ({df[memory_col].mean() / (1024**2):.2f} MB)\n")

    print(f"Visualization complete!")
    print(f"Saved plots to: {plots_save_path} and {supp_save_path}")
    print(f"Saved text report to: {report_path}")


    
            

