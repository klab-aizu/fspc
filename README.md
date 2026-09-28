# FSPC: Frequent Spike Pattern Compression

Official implementation of the paper **"[FSPC: A Lossy Spike Compression Through Correlated-AER Merging in Spiking Neural Networks](https://ieeexplore.ieee.org/document/11310938)"** (IEEE MCSoC 2025).

FSPC is a lossy spike compression framework designed to compress correlated spatial spike events in hidden layers into compact symbolic identifiers. Models are implemented with [snnTorch](https://snntorch.readthedocs.io/), with frequent pattern mining powered by [PAMI](https://github.com/UdayLab/PAMI) and [mlxtend](https://github.com/rasbt/mlxtend).

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/klab-aizu/fspc.git
cd fspc
pip install -r requirements.txt
```

---

## Repository Structure

```text
fspc/
├── main.py                # Central CLI execution script
├── config.py              # Hyperparameters and network configuration
├── data/
│   └── dataloader.py      # DataLoader definitions (MNIST, Fashion-MNIST)
├── compression/
│   └── compression.py     # AER conversion, mining drivers, and compression routines
├── evaluation/
│   └── evaluation.py      # Adaptive inference, experiment runner, and visualization
├── models/
│   ├── mlp_snn_v1.py      # 3-layer MLP SNN (single-layer compression: 100 -> 10)
│   └── mlp_snn_v2.py      # 4-layer MLP SNN (dual-layer compression: 100 -> 60 and 60 -> 10)
├── train_test_model/
│   └── train_test.py      # Training loop and deployment utilities
├── deployed_models/       # Saved PyTorch checkpoint weights (.pt)
└── metrics/               # Per-experiment output metrics, plots, and reports
```

> **Note:** `deployed_models/` and `metrics/` are created automatically on the first run.

---

## Running Experiments via Terminal

`main.py` is invoked via a unified command-line syntax:

```bash
python main.py <model>_<version>_<mining-func>_<dataset>
```

### Argument Syntax

| Token | Options | Description |
| :--- | :--- | :--- |
| `<model>` | `mlp` | Network topology family. |
| `<version>` | `v1`, `v2` | `v1`: 3-layer MLP ($100 \to 10$ compression)<br>`v2`: 4-layer MLP ($100 \to 60$ and $60 \to 10$ compression) |
| `<mining-func>` | `fpmax`, `fpgrowth`, `maxfpgrowth` | Pattern mining algorithm: `fpmax` (`mlxtend`), `fpgrowth` (`PAMI`), or `maxfpgrowth` (`PAMI`). |
| `<dataset>` | `mnist`, `fmnist`, `fashionmnist` | Target dataset. |

---

### Examples

```bash
# 1. 3-layer MLP with FPMax on MNIST
python main.py mlp_v1_fpmax_mnist

# 2. 4-layer MLP (dual-layer compression) with PAMI FP-Growth on MNIST
python main.py mlp_v2_fpgrowth_mnist

# 3. 3-layer MLP with PAMI MaxFP-Growth on Fashion-MNIST
python main.py mlp_v1_maxfpgrowth_fmnist

# 4. 4-layer MLP with FPMax on Fashion-MNIST
python main.py mlp_v2_fpmax_fashionmnist
```

---

## Workflow Lifecycle

1. **Automatic Checkpoint Verification**:
   The script checks `deployed_models/` for an existing checkpoint:
   ```text
   deployed_models/<model>_<version>_<dataset>.pt
   ```
   * If the checkpoint is **not found**, the network is trained automatically and saved.
   * If the checkpoint **exists**, training is skipped and weights are loaded immediately.

   *(Checkpoints are independent of the mining algorithm. Training runs only once per model-dataset pair).*

2. **Compression Grid Search**:
   Inference is evaluated across predefined parameter sweeps:
   * **Frequent Spike Pattern Count (`numFsp`)**: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
   * **Pattern Matching Threshold (`pmt`)**: `[0.0, 0.1, ..., 1.0]`

3. **Per-Experiment Output Organization**:
   All evaluation data is automatically saved inside a dedicated subdirectory under `metrics/`:
   ```text
   metrics/<model>_<version>_<mining-func>_<dataset>/
   ├── <target>.csv                                  # Full tabular metrics
   ├── <target>_accuracy_compression_analysis.png    # FSP & PMT trade-off plots
   ├── <target>_runtime_memory_analysis.png          # Latency & memory consumption plots
   └── <target>_summary_report.txt                   # Hardware specs & optimal operating point
   ```

---

## Model Architectures

* **`MLP_SNN_V1` (Single-Layer Compression)**:
  `Input (784) -> FC1 -> LIF1 (100) -> [FSPC Compression] -> FC2 -> LIF2 (10)`
  
  Compresses spike events transmitted from the 100-neuron hidden layer to the output layer.

* **`MLP_SNN_V2` (Dual-Layer Compression)**:
  `Input (784) -> FC1 -> LIF1 (100) -> [Compression 1] -> FC2 -> LIF2 (60) -> [Compression 2] -> FC3 -> LIF3 (10)`
  
  Applies compression independently at both hidden-layer boundaries, tracking per-layer metrics (100 -> 60 and 60 -> 10) alongside overall bandwidth reduction.

---

## Adding Custom Datasets and Models

1. **Custom Datasets**: Define a new DataLoader function in `data/dataloader.py` returning `(trainLoader, testLoader)`. Register the dataset key and dimensions in `config.py`.
2. **Custom Models**: Implement your model class under `models/`. Ensure `forward()` accepts `(x, compressionMode, patternMiningFunc, numFsp, pmt)` and returns `(spkOut, memOut, batchMetrics)`.
3. **Command-Line Registration**: Add your model or dataset identifier to `parse_experiment_string()` in `main.py`.

---

## Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@inproceedings{Ganesh2025FSPC,
  author    = {Ganesh, Satvik and Yuga, Hanyu and Wang, Zhishang and Dang, Khanh N.},
  title     = {FSPC: A Lossy Spike Compression Through Correlated-AER Merging in Spiking Neural Networks},
  booktitle = {2025 IEEE 18th International Symposium on Embedded Multicore/Many-core Systems-on-Chip (MCSoC)},
  year      = {2025},
  pages     = {386--393},
  doi       = {10.1109/MCSoC67473.2025.00068}
}
```

---

## License

This project is open-source software licensed under the [GPLv3 License](https://github.com/klab-aizu/fspc/blob/main/LICENSE).
