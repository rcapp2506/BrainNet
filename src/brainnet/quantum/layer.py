"""Layer quanvoluzionale come modulo PyTorch con autograd parameter-shift.

Pesi quantistici condivisi tra tutti i canali (depthwise / weight-sharing,
la proprieta' convoluzionale). Tutti i canali in una sola chiamata Estimator.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .engine import QuantumEngine, BackendManager


class _QuanvFunction(torch.autograd.Function):
    """Forward/backward del layer quanvoluzionale via parameter-shift batched."""

    @staticmethod
    def forward(ctx, input_patches, weights, engine):
        out = engine.forward(input_patches.detach().cpu().numpy(),
                             weights.detach().cpu().numpy())
        ctx.save_for_backward(input_patches, weights)
        ctx.engine = engine
        return torch.tensor(out, dtype=torch.float32, device=input_patches.device)

    @staticmethod
    def backward(ctx, grad_output):
        input_patches, weights = ctx.saved_tensors
        engine = ctx.engine
        device = grad_output.device

        skip_input = not input_patches.requires_grad   # dimezza i PUB se input staccato
        _, gw_jac, gx_jac = engine.forward_and_gradient(
            input_patches.detach().cpu().numpy(),
            weights.detach().cpu().numpy(),
            skip_input_grad=skip_input)

        go = grad_output.detach().cpu().numpy()
        grad_weights = np.einsum("jnq,nq->j", gw_jac, go)

        if skip_input or gx_jac is None:
            grad_inputs = None
        else:
            gi = np.einsum("inq,nq->ni", gx_jac, go)
            grad_inputs = torch.tensor(gi, dtype=torch.float32, device=device)

        return (grad_inputs,
                torch.tensor(grad_weights, dtype=torch.float32, device=device),
                None)


class QuantumConvLayer(nn.Module):
    """Convoluzione quanvoluzionale: patch k x k -> <Z> scalare per canale."""

    def __init__(self, num_qubits, kernel_size, stride, backend_manager: BackendManager,
                 ansatz="rzrx", measure_qubit=0, n_parallel_chunks=1,
                 detach_input_grad=True):
        super().__init__()
        self.n = num_qubits
        self.kernel_size = kernel_size
        self.stride = stride
        self.detach_input_grad = detach_input_grad
        self.engine = QuantumEngine(num_qubits, backend_manager, ansatz,
                                    measure_qubit, n_parallel_chunks)

        init = (backend_manager.rng.random(self.engine.num_weights) * 2 - 1) * 0.3
        self.quantum_weights = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        P = H_out * W_out

        patches_all = []
        for c in range(C):
            xc = x[:, c:c + 1, :, :]
            patches = F.unfold(xc, kernel_size=self.kernel_size, stride=self.stride)
            p_min = patches.min(dim=2, keepdim=True).values
            p_max = patches.max(dim=2, keepdim=True).values
            rng = (p_max - p_min).clamp(min=1e-8)
            patches = (patches - p_min) / rng * np.pi          # angle encoding [0, pi]
            patches_all.append(patches.permute(0, 2, 1).contiguous())

        flat = torch.cat(patches_all, dim=1).reshape(-1, self.n)
        if self.detach_input_grad:
            flat = flat.detach()
        q = _QuanvFunction.apply(flat, self.quantum_weights, self.engine)
        return q.reshape(B, C, H_out, W_out)
