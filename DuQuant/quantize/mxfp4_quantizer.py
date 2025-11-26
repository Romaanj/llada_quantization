import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MXFP4Quantizer(nn.Module):
    """
    Helper that simulates MXFP4 (E2M1) microscaling quantization.

    MXFP4 uses:
      * 1 sign bit + 2 exponent bits + 1 mantissa bit (E2M1) -> 8 positive levels incl. zero
      * Shared block scale encoded as E8M0 (power-of-two) for every 32 values (default)
    """

    FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

    def __init__(self, block_size: int = 32, eps: float = 1e-8):
        super().__init__()
        assert block_size > 0, "mxfp4 block_size must be positive"
        self.block_size = block_size
        self.eps = eps
        self.max_fp_value = 6.0  # Largest representable magnitude in E2M1 (before scaling)
        self.register_buffer("_levels", self.FP4_LEVELS.clone(), persistent=False)
        self.last_block_scales: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        this is fake quantizer. for MXFP4!
        """
        if x.numel() == 0:
            return x

        orig_dtype = x.dtype
        device = x.device
        levels = self._levels.to(device)

        x_fp32 = x.to(torch.float32)
        num_vectors = math.prod(x_fp32.shape[:-1])
        last_dim = x_fp32.shape[-1]
        reshaped = x_fp32.reshape(num_vectors, last_dim)

        pad = (self.block_size - (last_dim % self.block_size)) % self.block_size
        if pad > 0:
            reshaped = F.pad(reshaped, (0, pad))
            last_dim += pad
        num_blocks = last_dim // self.block_size

        blocks = reshaped.view(num_vectors, num_blocks, self.block_size)
        absmax = blocks.abs().amax(dim=-1, keepdim=True)

        scale = absmax / self.max_fp_value
        scale = torch.clamp(scale, min=self.eps)

        # Quantize scale to nearest power-of-two to mimic E8M0 shared exponent.
        log2 = torch.log2(scale)
        quant_log2 = torch.round(log2)
        scale = torch.pow(2.0, quant_log2)

        # Prevent inf/NaN for zero blocks
        zero_mask = (absmax <= self.eps)
        scale = torch.where(zero_mask, torch.ones_like(scale), scale)

        normalized = blocks / scale
        sign = normalized.sign()
        magnitude = normalized.abs().unsqueeze(-1)  # (..., block, size, 1)
        diffs = torch.abs(magnitude - levels.view(1, 1, 1, -1))
        indices = torch.argmin(diffs, dim=-1)
        quantized_mag = levels[indices]

        quantized = sign * quantized_mag
        quantized = torch.where(zero_mask, torch.zeros_like(quantized), quantized)
        dequantized = quantized * scale

        self.last_block_scales = scale.detach().clone()

        dequantized = dequantized.view(num_vectors, num_blocks * self.block_size)
        if pad > 0:
            dequantized = dequantized[:, :-pad]
        return dequantized.reshape(x_fp32.shape).to(orig_dtype)


