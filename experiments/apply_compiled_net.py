import os
import sys
from types import SimpleNamespace

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from difflogic import CompiledLogicNet
from experiments.datasets import load_dataset

torch.set_num_threads(1)

dataset = 'mnist-8thresholds'
batch_size = 1_000

_, _, test_loader = load_dataset(SimpleNamespace(
    dataset=dataset,
    batch_size=batch_size,
    valid_set_size=0.0,
    seed=0,
))

for num_bits in [
    # 8,
    # 16,
    # 32,
    64
]:
    save_lib_path = 'lib/{:08d}_{}.so'.format(0, num_bits)
    compiled_model = CompiledLogicNet.load(save_lib_path, 10, num_bits)

    correct, total = 0, 0
    for (data, labels) in test_loader:
        data = torch.nn.Flatten()(data).bool().numpy()

        output = compiled_model.forward(data)

        correct += (output.argmax(-1) == labels).float().sum()
        total += output.shape[0]

    acc3 = correct / total
    print('COMPILED MODEL', num_bits, acc3)
