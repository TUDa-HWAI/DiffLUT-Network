import math

import numpy as np
import torch

from difflogic import PackBitsTensor


def train(model, x, y, loss_fn, optimizer):
    """Perform a single training step"""
    x = model(x)
    loss = loss_fn(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def difflut_train(model, x, y, loss_fn, optimizer, mapping_optimizer=None):
    """Perform one DiffLUT optimizer training step."""
    if mapping_optimizer is not None:
        mapping_optimizer.zero_grad()

    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        if mapping_optimizer is not None:
            # Retain the graph for the gate-parameter gradient calculation.
            loss.backward(retain_graph=True)
        return loss

    loss = optimizer.step(closure)
    if mapping_optimizer is not None:
        mapping_optimizer.step()
    return loss.item()


def difflut_eval(model, loader, optimizer, soft=False):
    """Evaluate DiffLUT-Net with deterministic hard or relaxed weights."""
    original_training = model.training
    parameters = optimizer.param_groups[0]['params']
    original_parameters = torch.nn.utils.parameters_to_vector(parameters).detach().clone()
    model.train(mode=soft)

    total_correct = 0
    total_samples = 0
    try:
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to('cuda').round(), y.to('cuda')
                weights = [optimizer.state['mu']] if soft else None
                outputs = optimizer.get_mc_predictions(model, x, raw_noises=weights)
                predictions = torch.stack(outputs, dim=2).mean(dim=2).argmax(-1)
                total_correct += (predictions == y).sum().item()
                total_samples += y.shape[0]
    finally:
        torch.nn.utils.vector_to_parameters(original_parameters, parameters)
        model.train(original_training)

    return total_correct / total_samples if total_samples else 0.0


def eval(model, loader, mode):
    """Evaluate the model"""
    orig_mode = model.training
    with torch.no_grad():
        model.train(mode=mode)
        res = np.mean(
            [
                (model(x.to('cuda').round()).argmax(-1) == y.to('cuda')).to(torch.float32).mean().item()
                for x, y in loader
            ]
        )
        model.train(mode=orig_mode)
    return res.item()

def packbits_eval(model, loader):
    """Evaluate using PackBitsTensor"""
    orig_mode = model.training
    with torch.no_grad():
        model.eval()
        res = np.mean(
            [
                (model(PackBitsTensor(x.to('cuda').reshape(x.shape[0], -1).round().bool())).argmax(-1) == y.to(
                    'cuda')).to(torch.float32).mean().item()
                for x, y in loader
            ]
        )
        model.train(mode=orig_mode)
    return res.item()

class TempScheduler:
    """Temperature scheduler for simulated annealing"""
    def __init__(self, t_start: float, t_end: float, total_steps: int,
                 scheme: str = "cosine", warmup_ratio: float = 0.05, hold_ratio: float = 0.10):
        self.t_start = float(t_start)
        self.t_end = float(t_end)
        self.total = max(1, int(total_steps))
        self.scheme = scheme
        self.warm = int(warmup_ratio * self.total)
        self.hold = int(hold_ratio * self.total)

    def __call__(self, step: int) -> float:
        s = min(max(step, 0), self.total - 1)

        # 1) Linear warmup: slowly decrease from 1.5*t_start to t_start
        if s < self.warm:
            a = s / max(1, self.warm)
            return 1.5 * self.t_start - a * 0.5 * self.t_start

        # 2) Hold phase
        if s < self.warm + self.hold:
            return self.t_start

        # 3) Main annealing phase
        prog = (s - self.warm - self.hold) / max(1, self.total - self.warm - self.hold)

        if self.scheme == "cosine":
            return self.t_end + 0.5 * (self.t_start - self.t_end) * (1 + math.cos(math.pi * prog))
        if self.scheme == "exp":
            return self.t_start * ((self.t_end / self.t_start) ** prog)
        if self.scheme == "linear":
            return self.t_start + (self.t_end - self.t_start) * prog
        if self.scheme == "inv_sqrt":
            return self.t_end + (self.t_start - self.t_end) / max(1e-6, math.sqrt(1 + 9 * prog))

        return self.t_start + (self.t_end - self.t_start) * prog
