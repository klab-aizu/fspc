
# FSPC: Frequent Spike Pattern Compression

Official implementation of the paper **"[FSPC: A Lossy Spike Compression Through Correlated-AER Merging in Spiking Neural Networks](https://ieeexplore.ieee.org/document/11310938)"** (IEEE MCSoC 2025).

FSPC is a spike compression framework designed to merge frequently occurring spike patterns into compact symbols using pattern mining algorithms. Models are built using [snnTorch](https://snntorch.readthedocs.io/), with pattern mining powered by [PAMI](https://github.com/UdayLab/PAMI) and [mlxtend](https://github.com/rasbt/mlxtend).

---

## Installation

Download the repository and install the required dependencies:

```bash
cd fspc
pip install -r requirements.txt

```

---

## Repository Structure

```text
fspc/
├── main.py                # Central execution script for running experiments
├── config.py              # Configuration parameters for SNN models and datasets
├── data/
│   └── dataloader.py      # Dataset loading utilities
├── compression/
│   └── compression.py     # Utilities for spike data conversion and pattern compression
├── evaluation/
│   └── evaluation.py      # Model performance, runtime, and memory evaluation utilities
├── models/
│   ├── mlp_snn_v1.py      # 3-layer MLP SNN with hidden-layer compression & reconstruction
│   └── mlp_snn_v2.py      # 4-layer MLP SNN with hidden-layer compression & reconstruction
├── train_test_model/
│   └── train_test.py      # Training and deployment workflows for snnTorch models
├── deployed_models/       # Directory where trained .pt model checkpoints are stored
└── metrics/               # Directory where output CSV and analysis files are saved

```

> **Note:** The `deployed_models/` and `metrics/` directories will be created automatically if they do not exist.

---

## Quickstart Tutorial

To run a default experiment with model training and spike compression evaluation:

1. Execute `main.py`:

```bash
cd fspc
python main.py

```

2. The script will automatically train the SNN model (if a trained checkpoint is not already present in `deployed_models/`).
3. Once trained, batch-level spike pattern compression and inference metrics will be evaluated.
4. Exported summary metrics, CSV files, and visual plots will be saved to the `metrics/` directory.

---

## Adding New Datasets and Models

To extend the framework with custom architectures or datasets:

1. **New Datasets:** Add your custom loading logic inside `data/dataloader.py`.
2. **New Models:** Create a new model definition file under `models/`.
3. **Configurations:** Update `config.py` with the appropriate hyperparameters (e.g., input/output dimensions, timesteps, beta values).

---

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{Ganesh2025FSPC,
  author    = {Satvik Ganesh and Hanyu Yuga and Zhishang Wang and Khanh N. Dang},
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
