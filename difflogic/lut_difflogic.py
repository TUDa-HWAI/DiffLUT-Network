import torch
try:
    import difflogic_cuda
except ImportError:
    difflogic_cuda = None
import numpy as np
from .functional import bin_op_s, get_unique_connections_lut6, GradFactor
from .packbitstensor import PackBitsTensor


########################################################################################################################


class LutlogicLayer(torch.nn.Module):
    """
    Lut-based logic 3-2-1
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            device: str = 'cuda',
            grad_factor: float = 1.,
            implementation: str = None,
            connections: str = 'lut-based',
    ):
        """
        :param in_dim:      input dimensionality of the layer
        :param out_dim:     output dimensionality of the layer
        :param device:      device (options: 'cuda' / 'cpu')
        :param grad_factor: for deep models (>6 layers), the grad_factor should be increased (e.g., 2) to avoid vanishing gradients
        :param implementation: implementation to use (options: 'cuda' / 'python'). cuda is around 100x faster than python
        :param connections: method for initializing the connectivity of the logic gate net
        """
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(out_dim, 6, 16, device=device))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor
        self.temp = 1.0

        """
        The CUDA implementation is the fast implementation. As the name implies, the cuda implementation is only 
        available for device='cuda'. The `python` implementation exists for 2 reasons:
        1. To provide an easy-to-understand implementation of differentiable logic gate networks 
        2. To provide a CPU implementation of differentiable logic gate networks 
        """
        self.implementation = implementation
        if self.implementation is None and device == 'cuda':
            self.implementation = 'cuda'
        elif self.implementation is None and device == 'cpu':
            self.implementation = 'python'
        assert self.implementation in ['cuda', 'python'], self.implementation

        self.connections = connections
        assert self.connections == 'lut-based', self.connections
        self.indices = self.get_connections(self.connections, device)

        if self.implementation == 'cuda':
            """
            CUDA version not implemented yet
            """
            given_x_indices_of_y = [[] for _ in range(in_dim)]
            indices_np = [idx.cpu().numpy() for idx in self.indices]  # 6 arrays
            # -- Build reverse dependency table --
            for y in range(out_dim):
                # Iterate over 6 input ports: each input signal can be used by multiple output gates
                for idx_np in indices_np:
                    given_x_indices_of_y[idx_np[y]].append(y)
            self.given_x_indices_of_y_start = torch.tensor(
                np.array([0] + [len(g) for g in given_x_indices_of_y]).cumsum(), device=device, dtype=torch.int64)
            self.given_x_indices_of_y = torch.tensor(
                [item for sublist in given_x_indices_of_y for item in sublist], dtype=torch.int64, device=device)

        self.num_neurons = out_dim * 6
        self.num_weights = out_dim * 6

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            assert not self.training, 'PackBitsTensor is not supported for the differentiable training mode.'
            assert self.device == 'cuda', 'PackBitsTensor is only supported for CUDA, not for {}. ' \
                                          'If you want fast inference on CPU, please use CompiledDiffLogicModel.' \
                                          ''.format(self.device)

        else:
            if self.grad_factor != 1.:
                eff_factor = self.grad_factor / max(self.temp, 1e-6)
                x = GradFactor.apply(x, eff_factor)

        if self.implementation == "cuda":
            return self.forward_cuda(x)

        elif self.implementation == 'python':
            return self.forward_python(x)

        else:
            raise ValueError(f"Unknown implementation type: {self.implementation}")

    def forward_python(self, x):
        assert x.shape[-1] == self.in_dim, (x[0].shape[-1], self.in_dim)

        # Ensure all index tensors are of type int64
        if any(idx.dtype != torch.int64 for idx in self.indices):
            self.indices = tuple(idx.long() for idx in self.indices)

        a1, a2, a3, a4, a5, a6 = self.indices
        x1, x2, x3, x4, x5, x6 = (
            x[..., a1], x[..., a2], x[..., a3],
            x[..., a4], x[..., a5], x[..., a6]
        )
        if self.training:
            # Apply softmax over the 16 logic functions for each gate
            w = torch.nn.functional.softmax(self.weights, dim=-1)  # [out_dim, 5, 16]
        else:
            # Use one-hot fixed gates during inference
            w = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).float()  # [out_dim, 5, 16]

        g1 = bin_op_s(x1, x2, w[:, 0])  # Layer 1 gate 1
        g2 = bin_op_s(x3, x4, w[:, 1])  # Layer 1 gate 2
        g3 = bin_op_s(x5, x6, w[:, 2])  # Layer 1 gate 3
        g4 = bin_op_s(g1, g2, w[:, 3])  # Layer 2 gate 4
        g5 = bin_op_s(g2, g3, w[:, 4])
        out = bin_op_s(g4, g5, w[:, 5])  # Layer 3 gate 5
        return out

    def forward_cuda(self, x):
        """
        CUDA forward pass for 3-2-1 structured LUT layer:
        6 input ports in total, internally composed of 6 logic gates g1-g6.
        Call CUDA kernel (LutLogicLayerCudaFunction).
        """
        assert x.device.type == "cuda", x.device
        assert x.ndim == 2, f"Expected [batch, in_dim], got {x.shape}"

        # Convert to [in_dim, batch] to match kernel memory layout
        x = x.transpose(0, 1).contiguous()
        assert x.shape[0] == self.in_dim, (x.shape, self.in_dim)

        # Get indices for the six input terminals (6 inputs per output gate)
        a1, a2, a3, a4, a5, a6 = self.indices

        # Weights: 6 sets of gate parameters (corresponding to g1...g6)
        if self.training:
            w = torch.nn.functional.softmax(self.weights , dim=-1).to(x.dtype)
        else:
            w = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(x.dtype)

        # CUDA Function call (unrolls the 3-2-1 internal logic on the C++/CUDA side)
        out = LutLogicLayerCudaFunction.apply(
            x, a1, a2, a3, a4, a5, a6, w,
            self.given_x_indices_of_y_start, self.given_x_indices_of_y
        )

        # Convert back to [batch, out_dim]
        return out.transpose(0, 1)

    def forward_cuda_eval(self, x_packbits: PackBitsTensor):
        """
        (Optional) CUDA forward for bit-packed inference.
        Currently unused; reserved for future optimization.
        """
        raise NotImplementedError("PackBitsTensor (bit-packed) inference not implemented.")


    def extra_repr(self):
        return '{}, {}, {}'.format(self.in_dim, self.out_dim, 'train' if self.training else 'eval')

    def get_connections(self, connections, device='cuda'):
        assert self.out_dim * 6 >= self.in_dim, 'The number of neurons ({}) must not be smaller than half of the ' \
                                                'number of inputs ({}) because otherwise not all inputs could be ' \
                                                'used or considered.'.format(self.out_dim, self.in_dim)
        if connections == 'lut-based':
            a1, a2, a3, a4, a5, a6 = get_unique_connections_lut6(self.in_dim, self.out_dim, device)
            return a1, a2, a3, a4, a5, a6

        else:
            raise ValueError(connections)

    def set_temperature(self, t: float):
        self.temp = float(t)

class LutLogicLayerCudaFunction(torch.autograd.Function):
    """
    Autograd bridge for CUDA-based 3-2-1 LUT logic layer.
    Calls difflogic_cuda.lut_forward / lut_backward_x / lut_backward_w.
    """

    @staticmethod
    def forward(ctx, x, a1, a2, a3, a4, a5, a6, w,
                given_x_indices_of_y_start, given_x_indices_of_y):
        """
        x: [in_dim, batch]
        a1..a6: [out_dim]  Input indices for each gate
        w: [out_dim, 6, 16]  Softmax/one-hot weights for each gate
        given_x_indices_of_y_*: CSR index table (for backward pass)
        """
        # Save tensors needed for backward pass
        ctx.save_for_backward(x, a1, a2, a3, a4, a5, a6, w,
                              given_x_indices_of_y_start, given_x_indices_of_y)

        # Call CUDA kernel to execute forward pass
        out = difflogic_cuda.lut_forward(
            x, a1, a2, a3, a4, a5, a6, w,
            given_x_indices_of_y_start, given_x_indices_of_y
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        grad_out: [out_dim, batch]
        Calculate gradients for input x and weights w
        """
        (x, a1, a2, a3, a4, a5, a6, w,
         given_x_indices_of_y_start, given_x_indices_of_y) = ctx.saved_tensors

        grad_out = grad_out.contiguous()
        grad_x = grad_w = None

        # -- Calculate input gradients --
        if ctx.needs_input_grad[0]:
            grad_x = difflogic_cuda.lut_backward_x(
                x, a1, a2, a3, a4, a5, a6, w, grad_out,
                given_x_indices_of_y_start, given_x_indices_of_y
            )

        # -- Calculate weight gradients (training only) --
        if ctx.needs_input_grad[7]:
            grad_w = difflogic_cuda.lut_backward_w(
                x, a1, a2, a3, a4, a5, a6, grad_out
            )

        # Returned gradients order must match forward arguments
        return (
            grad_x,   # ∂L/∂x
            None, None, None, None, None, None,  # a1..a6 (non-trainable)
            grad_w,   # ∂L/∂w
            None, None  # given_x_indices_of_y_start / given_x_indices_of_y
        )
