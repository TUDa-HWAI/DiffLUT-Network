import torch
try:
    import difflogic_cuda
except ImportError:
    difflogic_cuda = None
import numpy as np
from .functional import bin_op_s, get_unique_connections_lut6, GradFactor
from .packbitstensor import PackBitsTensor
from .lut_difflogic import LutlogicLayer


########################################################################################################################


class ResidualBlock(LutlogicLayer):
    """
    LutlogicLayer with residual connection: y = f_logic(x) + α * P(x)
    """
    def __init__(self, in_dim, out_dim, alpha=1.0, **kwargs):
        super().__init__(in_dim, out_dim, **kwargs)
        self.alpha = alpha  # Residual ratio

        if in_dim != out_dim:
            self.shortcut = torch.nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.shortcut = torch.nn.Identity()

    def forward(self, x):
        logic_out = super().forward(x)
        shortcut = self.shortcut(x)
        out = (logic_out + self.alpha * shortcut) / (1.0 + self.alpha)
        # Limit output range to prevent overflow
        out = torch.clamp(out, 0.0, 1.0)
        return out
