"""DiffLUT-Net layer implementation."""

import torch

from .difflut_net_functional import difflut_net_op
from .functional import GradFactor, get_unique_connections_lut6
from .packbitstensor import PackBitsTensor


class _LearnableMappingFunction(torch.autograd.Function):
    """Select discrete inputs while supplying a soft surrogate gradient."""

    @staticmethod
    def forward(ctx, x, weights, tau):
        mapping = weights.argmax(dim=0)
        ctx.save_for_backward(x, weights, tau)
        return x[:, mapping]

    @staticmethod
    def backward(ctx, output_grad):
        x, weights, tau = ctx.saved_tensors
        weights_grad = (2 * x - 1).T @ output_grad
        input_grad = output_grad @ torch.softmax(weights / tau, dim=0).T
        return input_grad, weights_grad, None


class _LearnableMapping(torch.nn.Module):
    def __init__(self, input_size, output_size, device, tau=0.001):
        super().__init__()
        self.weights = torch.nn.Parameter(
            torch.rand(input_size, output_size, dtype=torch.float32, device=device)
        )
        self.tau = float(tau)

    def forward(self, x):
        tau = self.weights.new_tensor(self.tau)
        return _LearnableMappingFunction.apply(x, self.weights, tau)


class DiffLUTNetLayer(torch.nn.Module):
    """A differentiable six-input LUT with 64 trainable truth-table entries."""

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            device: str = 'cuda',
            grad_factor: float = 1.,
            connections: str = 'learnable',
            implementation: str = None,
    ):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros(out_dim, 64, device=device))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor
        self.connections = connections

        # DiffLUT-Net currently uses its PyTorch implementation for every device.
        self.implementation = 'python'
        if connections not in ('random', 'lut-based', 'unique', 'learnable'):
            raise ValueError(connections)

        if connections == 'learnable':
            self.mapping = _LearnableMapping(in_dim, out_dim * 6, device=device)
        else:
            self.indices = self._get_connections(connections, device)

        self.num_neurons = out_dim
        self.num_weights = out_dim * 64

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            raise NotImplementedError('PackBitsTensor is not supported by DiffLUT-Net')
        if self.grad_factor != 1.:
            x = GradFactor.apply(x, self.grad_factor)
        return self._forward_python(x)

    def _forward_python(self, x):
        assert x.shape[-1] == self.in_dim, (x.shape[-1], self.in_dim)

        if self.connections == 'learnable':
            mapped = self.mapping(x)
            inputs = tuple(
                mapped[..., i * self.out_dim:(i + 1) * self.out_dim]
                for i in range(6)
            )
        else:
            if any(index.dtype != torch.int64 for index in self.indices):
                self.indices = tuple(index.long() for index in self.indices)
            inputs = tuple(x[..., index] for index in self.indices)

        weights = torch.clamp(self.weights, 0.0, 1.0)
        if not self.training:
            weights = (weights > 0.5).float()
        return difflut_net_op(*inputs, weights)

    def _get_connections(self, connections, device):
        if connections != 'learnable' and self.out_dim * 6 < self.in_dim:
            raise ValueError(
                f'{self.out_dim} neurons cannot cover {self.in_dim} inputs with six-input LUTs'
            )
        if connections == 'random':
            indices = torch.randperm(6 * self.out_dim) % self.in_dim
            indices = torch.randperm(self.in_dim)[indices].reshape(6, self.out_dim)
            return tuple(indices[i].long().to(device) for i in range(6))
        if connections in ('unique', 'lut-based'):
            return get_unique_connections_lut6(self.in_dim, self.out_dim, device)
        raise ValueError(connections)

    def extra_repr(self):
        mode = 'train' if self.training else 'eval'
        return f'{self.in_dim}, {self.out_dim}, {mode}'
