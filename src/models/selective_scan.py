"""
Selective Scan Interface with Native PyTorch Vectorized Fallback.
Provides seamless execution on Windows without requiring C++/CUDA kernel compilation,
while automatically hooking into mamba_ssm.ops.selective_scan_interface if available.
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F

# Attempt importing official compiled CUDA selective_scan_fn
_CUDA_SELECTIVE_SCAN_AVAILABLE = False
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _cuda_selective_scan_fn
    _CUDA_SELECTIVE_SCAN_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _cuda_selective_scan_fn = None
    _CUDA_SELECTIVE_SCAN_AVAILABLE = False


def _selective_scan_ref(
    u: torch.Tensor,             # [B, D, L]
    delta: torch.Tensor,         # [B, D, L]
    A: torch.Tensor,             # [D, N]
    B: torch.Tensor,             # [B, N, L]
    C: torch.Tensor,             # [B, N, L]
    D: torch.Tensor | None = None,      # [D]
    z: torch.Tensor | None = None,      # [B, D, L]
    delta_bias: torch.Tensor | None = None,  # [D]
    delta_softplus: bool = False,
    return_last_state: bool = False,
):
    """
    Pure PyTorch implementation of selective scan for state-space models.
    Supports arbitrary batch size, sequence length, and backward pass on any device (CPU/CUDA).
    """
    dtype = u.dtype
    u = u.float()
    delta = delta.float()
    
    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(-1).float()
    if delta_softplus:
        delta = F.softplus(delta)
        
    b, d, l = u.shape
    n = A.shape[1]
    
    # Discretize continuous state space parameters (Zero-Order Hold)
    # delta: [B, D, L], A: [D, N] -> deltaA: [B, D, L, N]
    delta_expanded = delta.unsqueeze(-1)  # [B, D, L, 1]
    A_expanded = A.unsqueeze(0).unsqueeze(2)  # [1, D, 1, N]
    deltaA = torch.exp(delta_expanded * A_expanded)  # [B, D, L, N]
    
    # deltaB: [B, D, L, N]
    # B: [B, N, L] -> permute to [B, 1, L, N]
    B_expanded = B.permute(0, 2, 1).unsqueeze(1)  # [B, 1, L, N]
    deltaB = delta_expanded * B_expanded  # [B, D, L, N]
    
    # State recurrence
    x = torch.zeros(b, d, n, device=u.device, dtype=torch.float32)
    ys = []
    
    # C: [B, N, L] -> [B, L, N]
    C_perm = C.permute(0, 2, 1)  # [B, L, N]
    
    for t in range(l):
        x = deltaA[:, :, t, :] * x + deltaB[:, :, t, :] * u[:, :, t].unsqueeze(-1)
        # y_t = C_t * x_t: sum over state dimension N
        # C_perm[:, t, :]: [B, N] -> [B, 1, N]
        c_t = C_perm[:, t, :].unsqueeze(1)  # [B, 1, N]
        y_t = (x * c_t).sum(dim=-1)          # [B, D]
        ys.append(y_t)
        
    y = torch.stack(ys, dim=-1)  # [B, D, L]
    
    if D is not None:
        y = y + u * D.unsqueeze(0).unsqueeze(-1).float()
        
    if z is not None:
        y = y * F.silu(z.float())
        
    out = y.to(dtype)
    if return_last_state:
        return out, x.to(dtype)
    return out


def selective_scan_fn(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
):
    """
    Unified Selective Scan Function.
    Delegates to official CUDA kernel if available and on GPU, otherwise uses PyTorch reference.
    """
    if _CUDA_SELECTIVE_SCAN_AVAILABLE and u.is_cuda and _cuda_selective_scan_fn is not None:
        try:
            return _cuda_selective_scan_fn(
                u, delta, A, B, C, D=D, z=z,
                delta_bias=delta_bias,
                delta_softplus=delta_softplus,
                return_last_state=return_last_state
            )
        except Exception:
            # Fall back to reference if CUDA kernel encounters any dimension/alignment mismatch
            pass
            
    return _selective_scan_ref(
        u, delta, A, B, C, D=D, z=z,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
        return_last_state=return_last_state
    )
