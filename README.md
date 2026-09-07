# DiffLUT-Net

DiffLUT-Net is a differentiable lookup-table network for classification and
hardware-oriented inference. Each DiffLUT-Net neuron is a six-input LUT with
64 trainable truth-table entries. The default model jointly learns the LUT
contents and the input connections, and can be discretized for evaluation or
exported as synthesizable Verilog.

The repository contains:

- the DiffLUT-Net layer and its differentiable LUT operation;
- the DiffLUT optimizer;
- one training entry point;
- seven unified dataset interfaces;
- automatic dataset download and preprocessing;
- JSON accuracy histories; and
- combinational and registered Verilog export.

## Installation

### Environment

The experiments use the following environment:

| Component | Version |
|---|---|
| Operating system | Rocky Linux 9.8 |
| Python | 3.9.25 |
| PyTorch | 2.8.0+cu128 |
| torchvision | 0.23.0 |
| CUDA runtime bundled with PyTorch | 12.8 |
| NVIDIA driver | 610.43.02 |
| GPU | NVIDIA H200 NVL |

### Install

From the repository root, install the pinned dependencies and the project:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r experiments/requirements.txt
python -m pip install -e .
```

The dependency versions are specified in
[`experiments/requirements.txt`](experiments/requirements.txt).

## Datasets

All datasets are accessed through the same loader. Missing data is downloaded
and preprocessed automatically on first use.

| CLI dataset name | Dataset | Augmentation |
|---|---|:---:|
| `cifar-10-8-thresholds` | CIFAR-10 | No |
| `cifar-10-8-thresholds-aug` | CIFAR-10 | Yes |
| `fashion_mnist-8thresholds` | Fashion-MNIST | No |
| `mnist-8thresholds` | MNIST | No |
| `mnist-aug-8thresholds` | MNIST | Yes |
| `jsc-openml` | Jet Substructure Classification, OpenML | — |
| `jsc-cernbox` | Jet Substructure Classification, CERNBox | — |

## Training

The single training entry point is `experiments/main.py`. The dataset, layer
widths, and number of layers are the configuration-specific arguments.

For example, the two-layer JSC CERNBox model is trained with:

```bash
python experiments/main.py \
  --dataset jsc-cernbox \
  --num_neurons 4000 2000 \
  --num_layers 2 \
  --tau 74.3 \
  --num-iterations 200000
```

`--num_neurons` accepts one width per layer. Use the dataset, layer widths,
`tau`, and training steps reported in the paper to run the other configurations.

Use `python experiments/main.py --help` to display every available option.

### Main parameters

| Argument | Default | Description |
|---|---:|---|
| `--dataset` | required | One of the seven dataset identifiers above. |
| `--num_neurons` | required | Space-separated output widths of the LUT layers. |
| `--num_layers` | required | Number of LUT layers; normally equal to the number of supplied widths. |
| `--tau` | 10 | Divisor used by the final class-wise `GroupSum` aggregation. |
| `--batch-size` | 100 | Number of samples in each training batch. |
| `--num-iterations` | 200,000 | Number of optimizer steps. |
| `--eval-freq` | 1,000 | Number of steps between accuracy evaluations. |
| `--learning-rate` | 1e-7 | Initial optimizer learning rate. |
| `--lr-end` | 1e-9 | Final learning rate of the cosine schedule. |
| `--warmup-steps` | 5,000 | Linear learning-rate warmup length. |
| `--optimizer` | `difflut` | Optimizer choice: `difflut` or `adam`. |
| `--connections` | `learnable` | Input connection strategy. |
| `--train_samples` | 0 | Monte Carlo samples used by the DiffLUT optimizer; zero uses the deterministic mean. |
| `--grad-factor` | 1.0 | Gradient scaling factor applied by LUT layers. |
| `--penalty` | 1.0 | Fixed regularization weight of the DiffLUT optimizer. |
| `--lamda_init` | 10 | Initial magnitude of the binary-distribution natural parameters. |
| `--seed` | 0 | Random seed used by PyTorch, NumPy, and Python. |
| `--export_verilog` | disabled | Export the trained discrete model as Verilog. |

The learning-rate schedule is enabled by default. The standard configuration
uses batch size 100, evaluation every 1,000 steps, learning rate
`1e-7 -> 1e-9`, deterministic training weights (`--train_samples 0`),
learnable connections, gradient factor 1.0, penalty 1.0, and
`lamda_init=10`.

## Outputs

Unless `DIFFLUT_RESULTS_DIR` is set, each run is written under:

```text
results/<dataset>_DiffLUT-Net_<timestamp>/
└── metrics/
    ├── args.txt
    └── history.json
```

`args.txt` records the complete configuration. `history.json` contains only
the evaluation step and train, validation, and test accuracies. Loss is shown
in the terminal but is not stored in the JSON file.

Set a different output root when needed:

```bash
DIFFLUT_RESULTS_DIR=/path/to/results python experiments/main.py ...
```

## Verilog export

Add `--export_verilog` to export the trained discrete network:

```bash
python experiments/main.py \
  --dataset mnist-8thresholds \
  --num_neurons 2000 1000 \
  --num_layers 2 \
  --tau 7.0 \
  --export_verilog
```

The run directory will additionally contain:

```text
Verilogwithoutreg/logic_net.v
Verilogwithreg/logic_net.v
```

The first file is combinational. The second inserts registers between network
stages.

## Project structure

```text
difflogic/                 DiffLUT-Net layers and logic operations
optimizers/                DiffLUT optimizer
experiments/main.py        Training entry point
experiments/datasets/      Unified loaders and JSC downloaders
experiments/utils/train.py Training and evaluation helpers
experiments/utils/         Verilog export utilities
```

## License

DiffLUT-Net is released under the MIT License. See [LICENSE](LICENSE).
