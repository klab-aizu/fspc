"""
evaluation.py

Adaptive evaluation script for FSPC models.
Automatically detects and evaluates:
  - Single-layer compression models (e.g., MLP_SNN_V1)
  - Multi-layer compression models  (e.g., MLP_SNN_V2)

Functions:
    runInference: Dynamic inference supporting 1-layer or multi-layer compression.
    runExperiment: Grid-search runner with adaptive DataFrame column generation.
    visualizeResults: Plotting and reporting compatible with both model versions.
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
    Run model inference on a test dataset, dynamically adapting to either
    single-layer (V1) or dual-layer (V2) compression outputs.
    """
    total = 0
    correct = 0

    # Layer 1 metric lists (or single layer for V1)
    batchCompressions1 = []
    batchRuntime1 = []
    batchMemory1 = []

    # Layer 2 metric lists (for V2)
    batchCompressions2 = []
    batchRuntime2 = []
    batchMemory2 = []

    is_multi_layer = False

    model.to(device)
    model.eval()

    totalBatches = len(testLoader)

    if compressionMode:
        miner_name = minerFunc.__name__ if minerFunc else "None"
        print(f"\nTest - Compression Enabled with {miner_name} (Total Batches: {totalBatches})")
    else:
        print(f"\nBaseline - Compression Disabled (Total Batches: {totalBatches})")

    with torch.no_grad():
        for batchIdx, (data, targets) in enumerate(testLoader, start=1):
            if batchIdx % 50 == 0 or batchIdx == totalBatches:
                print(f"Processing batch: {batchIdx}/{totalBatches}")

            data = data.to(device, non_blocking=True).view(data.size(0), -1)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            spkOut, _, metrics = model(
                data, 
                compressionMode=compressionMode, 
                patternMiningFunc=minerFunc, 
                numFsp=numFSP, 
                pmt=pmt
            )

            # DYNAMIC DETECTION: Check if metrics is single-layer dict (V1) or tuple (V2)
            if metrics is not None:
                if isinstance(metrics, tuple):
                    is_multi_layer = True
                    m1, m2 = metrics
                else:
                    m1, m2 = metrics, None

                # Record Layer 1 (or single layer)
                if m1:
                    batchCompressions1.append(m1.get("reduction_len", 0.0))
                    batchRuntime1.append(m1.get("runtime", 0.0))
                    batchMemory1.append(m1.get("rss_mem", 0))

                # Record Layer 2 (if present)
                if m2:
                    batchCompressions2.append(m2.get("reduction_len", 0.0))
                    batchRuntime2.append(m2.get("runtime", 0.0))
                    batchMemory2.append(m2.get("rss_mem", 0))

            # Predicted class over time dimension
            pred = spkOut.sum(dim=0).argmax(dim=1)
            total += targets.size(0)
            correct += (pred == targets).sum().item()

            # State reset between test batches (SpikingJelly/snnTorch safe)
            if hasattr(model, "reset_states"):
                model.reset_states()

    accuracy = (correct / total) * 100.0

    meanComp1 = mean(batchCompressions1) if batchCompressions1 else 0.0
    meanRuntime1 = mean(batchRuntime1) if batchRuntime1 else 0.0
    meanMemory1 = mean(batchMemory1) if batchMemory1 else 0.0

    name = minerFunc.__name__ if minerFunc else "Baseline"

    # Assemble return metrics dictionary
    metricsDict = {
        "is_multi_layer": is_multi_layer,
        "comp1": meanComp1,
        "runtime1": meanRuntime1,
        "memory1": meanMemory1,
    }

    if is_multi_layer:
        meanComp2 = mean(batchCompressions2) if batchCompressions2 else 0.0
        meanRuntime2 = mean(batchRuntime2) if batchRuntime2 else 0.0
        meanMemory2 = mean(batchMemory2) if batchMemory2 else 0.0

        metricsDict.update({
            "comp2": meanComp2,
            "runtime2": meanRuntime2,
            "memory2": meanMemory2,
            # Combined / Total metrics across both layers
            "compTotal": (meanComp1 + meanComp2) / 2.0 if (meanComp1 and meanComp2) else (meanComp1 or meanComp2),
            "runtimeTotal": meanRuntime1 + meanRuntime2,
            "memoryTotal": max(meanMemory1, meanMemory2)
        })
    else:
        metricsDict.update({
            "compTotal": meanComp1,
            "runtimeTotal": meanRuntime1,
            "memoryTotal": meanMemory1
        })

    return accuracy, metricsDict, name


def runExperiment(model, testLoader, device, fspList, pmtList, compressionMode, minerFunc, filename):
    """
    Evaluate compression performance across multiple FSP and PMT configurations.
    Adapts CSV column structure dynamically based on single- vs. multi-layer models.
    """
    results = []

    # Detect layer dimensions dynamically for clean column labels
    h1 = getattr(model, "num_hidden1", "H1")
    h2 = getattr(model, "num_hidden2", None)
    out_dim = getattr(model, "num_outputs", "Out")

    for fsp in fspList:
        for pmt in pmtList:
            accuracy, metricsDict, name = runInference(
                model=model, 
                testLoader=testLoader, 
                device=device, 
                numFSP=fsp, 
                pmt=pmt, 
                compressionMode=compressionMode, 
                minerFunc=minerFunc
            )

            # Core parameters
            row = {
                "FSP Count": fsp,
                "PMT Value": pmt,
                f"{name} Accuracy": accuracy,
            }

            # If multi-layer (V2), add per-layer breakdown
            if metricsDict["is_multi_layer"]:
                label_l1 = f"({h1}->{h2})" if h2 else "(Layer 1)"
                label_l2 = f"({h2}->{out_dim})" if h2 else "(Layer 2)"

                row.update({
                    f"{name} Mean Compression {label_l1}": metricsDict["comp1"],
                    f"{name} Mean Runtime per Batch {label_l1} (s)": metricsDict["runtime1"],
                    f"{name} Mean Memory per Batch {label_l1} (bytes)": metricsDict["memory1"],
                    f"{name} Mean Compression {label_l2}": metricsDict["comp2"],
                    f"{name} Mean Runtime per Batch {label_l2} (s)": metricsDict["runtime2"],
                    f"{name} Mean Memory per Batch {label_l2} (bytes)": metricsDict["memory2"],
                })

            # Universal columns (ensures visualizeResults works for both V1 and V2)
            row.update({
                f"{name} Mean Compression": metricsDict["compTotal"],
                f"{name} Mean Runtime per Batch (s)": metricsDict["runtimeTotal"],
                f"{name} Mean Memory per Batch (bytes)": metricsDict["memoryTotal"],
            })

            results.append(row)

    # Measure baseline run without compression
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    timeStart = time.time()

    accBase, _, _ = runInference(model=model, testLoader=testLoader, device=device, compressionMode=False)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    timeEnd = time.time()

    baseElapsedTime = timeEnd - timeStart

    print(f"\nBaseline Accuracy: {accBase:.3f}%")
    print(f"Baseline Runtime: {baseElapsedTime:.3f} secs")

    dfResults = pd.DataFrame(results)
    dfResults["Baseline_Accuracy"] = accBase
    dfResults["Baseline_Runtime"] = baseElapsedTime

    metrics_folder = Path("metrics")
    metrics_folder.mkdir(exist_ok=True)

    save_path = metrics_folder / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    dfResults.to_csv(save_path, index=False)

    print(f"Done! CSV saved as {save_path}")


def visualizeResults(filename, datasetName, networkType):
    """
    Generate visual analysis plots and summary reports from experiment metrics.
    Fully compatible with both single-layer and multi-layer CSV outputs.
    """
    metrics_folder = Path("metrics")
    file_path = metrics_folder / filename

    if not file_path.exists():
        print(f"Error: {file_path} does not exist.")
        return

    output_dir = file_path.parent
    exp_prefix = file_path.stem

    df = pd.DataFrame(pd.read_csv(file_path))

    # Dynamically extract full column names and model/miner name prefix ({name})
    acc_col = [c for c in df.columns if "Accuracy" in c and "Baseline" not in c][0]
    comp_col = [c for c in df.columns if c.endswith("Mean Compression")][0]
    runtime_col = [c for c in df.columns if c.endswith("Mean Runtime per Batch (s)")][0]
    memory_col = [c for c in df.columns if c.endswith("Mean Memory per Batch (bytes)")][0]

    model_name = acc_col.replace(" Accuracy", "").strip()
    baseline_acc = df["Baseline_Accuracy"].iloc[0]

    # Calculate tradeoff metric
    norm_acc = (df[acc_col] - df[acc_col].min()) / (df[acc_col].max() - df[acc_col].min() + 1e-8)
    norm_comp = (df[comp_col] - df[comp_col].min()) / (df[comp_col].max() - df[comp_col].min() + 1e-8)
    df["Tradeoff_Score"] = 2 * (norm_acc * norm_comp) / (norm_acc + norm_comp + 1e-8)

    best_row = df.loc[df["Tradeoff_Score"].idxmax()]
    print(f"Best Overall Combo -> FSP: {best_row['FSP Count']}, PMT: {best_row['PMT Value']}")

    # Check if this experiment contains multi-layer breakdown columns
    layer_comp_cols = [c for c in df.columns if "Mean Compression (" in c]

    # --- Graph 1: Two-Panel Analysis (FSP Variation & PMT Variation) ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Plot A: Variable FSP (Constant PMT)
    base_pmt = df["PMT Value"].min()
    df_fsp_var = df[df["PMT Value"] == base_pmt].sort_values("FSP Count")

    ax1 = axes[0]
    ax2 = ax1.twinx()

    ax1.bar(df_fsp_var["FSP Count"].astype(str), df_fsp_var[comp_col], color="skyblue", alpha=0.6, label=comp_col)
    ax2.plot(df_fsp_var["FSP Count"].astype(str), df_fsp_var[acc_col], color="crimson", marker="o", linewidth=2, label=acc_col)
    ax2.axhline(y=baseline_acc, color="black", linestyle="--", linewidth=1.5, label=f"Baseline Accuracy ({baseline_acc:.2f}%)")

    ax2.text(0.01, baseline_acc, f" Baseline: {baseline_acc:.2f}%", transform=ax2.get_yaxis_transform(), va='bottom', ha='left', color='black', fontsize=9, fontweight='bold')

    max_comp_a = df_fsp_var[comp_col].max()
    if max_comp_a > 0:
        ax1.set_ylim(0, max_comp_a * 2.2)

    min_acc_a = min(df_fsp_var[acc_col].min(), baseline_acc)
    max_acc_a = max(df_fsp_var[acc_col].max(), baseline_acc)
    range_acc_a = max_acc_a - min_acc_a if max_acc_a != min_acc_a else 1.0
    ax2.set_ylim(min_acc_a - range_acc_a * 1.1, max_acc_a + range_acc_a * 0.2)

    ax1.set_xlabel("FSP Count")
    ax1.set_ylabel("Compression (%)", color="skyblue")
    ax2.set_ylabel("Accuracy (%)", color="crimson")
    ax1.set_title(f"{datasetName}: FSP Variation (Constant PMT = {base_pmt})")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=True)

    # Plot B: Variable PMT (Constant FSP = 1)
    df_pmt_var = df[df["FSP Count"] == 1].sort_values("PMT Value")

    ax3 = axes[1]
    ax4 = ax3.twinx()

    ax3.bar(df_pmt_var["PMT Value"].astype(str), df_pmt_var[comp_col], color="lightgreen", alpha=0.6, label=comp_col)
    ax4.plot(df_pmt_var["PMT Value"].astype(str), df_pmt_var[acc_col], color="darkorange", marker="s", linewidth=2, label=acc_col)
    ax4.axhline(y=baseline_acc, color="black", linestyle="--", linewidth=1.5, label=f"Baseline Accuracy ({baseline_acc:.2f}%)")

    ax4.text(0.01, baseline_acc, f" Baseline: {baseline_acc:.2f}%", transform=ax4.get_yaxis_transform(), va='bottom', ha='left', color='black', fontsize=9, fontweight='bold')

    max_comp_b = df_pmt_var[comp_col].max()
    if max_comp_b > 0:
        ax3.set_ylim(0, max_comp_b * 2.5)

    min_acc_b = min(df_pmt_var[acc_col].min(), baseline_acc)
    max_acc_b = max(df_pmt_var[acc_col].max(), baseline_acc)
    range_acc_b = max_acc_b - min_acc_b if max_acc_b != min_acc_b else 1.0
    ax4.set_ylim(min_acc_b - range_acc_b * 1.5, max_acc_b + range_acc_b * 0.5)

    ax3.set_xlabel("PMT Value")
    ax3.set_ylabel("Compression (%)", color="lightgreen")
    ax4.set_ylabel("Accuracy (%)", color="darkorange")
    ax3.set_title(f"{datasetName}: PMT Variation (Constant FSP = 1)")

    h3, l3 = ax3.get_legend_handles_labels()
    h4, l4 = ax4.get_legend_handles_labels()
    ax3.legend(h3 + h4, l3 + l4, loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=True)

    plt.tight_layout()
    plots_save_path = output_dir / f"{exp_prefix}_accuracy_compression_analysis.png"
    plt.savefig(plots_save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # --- Graph 2: Runtime and Memory Analysis ---
    fig, ax_rt = plt.subplots(figsize=(11, 6))
    ax_mem = ax_rt.twinx()

    labels = [f"FSP:{r['FSP Count']}|PMT:{r['PMT Value']}" for _, r in df.iterrows()]
    x_indices = np.arange(len(labels))

    ax_rt.plot(x_indices, df[runtime_col], color="purple", marker="^", label=runtime_col)
    ax_mem.plot(x_indices, df[memory_col] / (1024 ** 2), color="teal", marker="d", linestyle=":", label=f"{model_name} Mean Memory per Batch (MB)")

    ax_rt.set_xticks(x_indices[::max(1, len(labels)//10)])
    ax_rt.set_xticklabels(labels[::max(1, len(labels)//10)], rotation=45)
    ax_rt.set_xlabel("FSP | PMT Combination")
    ax_rt.set_ylabel(f"{runtime_col} (s)", color="purple")
    ax_mem.set_ylabel(f"{model_name} Mean Memory per Batch (MB)", color="teal")
    ax_rt.set_title(f"{datasetName}: Runtime & Memory Analysis", pad=35)

    h_rt, lab_rt = ax_rt.get_legend_handles_labels()
    h_mem, lab_mem = ax_mem.get_legend_handles_labels()
    ax_rt.legend(h_rt + h_mem, lab_rt + lab_mem, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=True)

    plt.tight_layout()
    supp_save_path = output_dir / f"{exp_prefix}_runtime_memory_analysis.png"
    plt.savefig(supp_save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # --- Text Report Export ---
    cpu_info = platform.processor() or platform.machine()
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected"

    report_path = output_dir / f"{exp_prefix}_summary_report.txt"
    with open(report_path, "w") as f:
        f.write("=====================================================\n")
        f.write(f" EXPERIMENT ANALYSIS REPORT: {datasetName}\n")
        f.write("=====================================================\n\n")
        f.write("--- PC SYSTEM HARDWARE SPECIFICATIONS ---\n")
        f.write(f"OS: {platform.system()} {platform.release()}\n")
        f.write(f"CPU: {cpu_info}\n")
        f.write(f"RAM: {ram_gb} GB\n")
        f.write(f"GPU: {gpu_info}\n\n")
        f.write("--- BEST ACCURACY-COMPRESSION TRADEOFF COMBINATION ---\n")
        f.write(f"Neural Network Type: {networkType}\n")
        f.write(f"Architecture Type: {'Multi-Layer Compression' if layer_comp_cols else 'Single-Layer Compression'}\n")
        f.write(f"FSP Count: {best_row['FSP Count']}\n")
        f.write(f"PMT Value: {best_row['PMT Value']}\n")
        f.write(f"{acc_col}: {best_row[acc_col]:.3f}%\n")
        f.write(f"Baseline Accuracy: {baseline_acc:.3f}%\n")
        f.write(f"{comp_col} (Total): {best_row[comp_col]:.4f}%\n")

        # If multi-layer, print per-layer breakdown
        for col in layer_comp_cols:
            f.write(f"  - {col}: {best_row[col]:.4f}%\n")

        f.write(f"{runtime_col}: {best_row[runtime_col]:.4f} sec\n")
        f.write(f"{memory_col}: {best_row[memory_col]:.2f} bytes ({best_row[memory_col] / (1024**2):.2f} MB)\n\n")
        f.write("--- OVERALL EXPERIMENT METRICS SUMMARY ---\n")
        f.write(f"Overall {runtime_col}: {df[runtime_col].mean():.4f} sec\n")
        f.write(f"Overall {memory_col}: {df[memory_col].mean():.2f} bytes ({df[memory_col].mean() / (1024**2):.2f} MB)\n")

    print(f"Visualization complete!")
    print(f"Saved plots to: {plots_save_path} and {supp_save_path}")
    print(f"Saved text report to: {report_path}")
