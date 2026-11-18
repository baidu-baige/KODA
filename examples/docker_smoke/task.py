"""Small standalone GELU task for validating the sandbox toolchain."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate="tanh")


def get_inputs():
    return [torch.randn(2, 512, 1024, dtype=torch.float32)]


def get_init_inputs():
    return []
