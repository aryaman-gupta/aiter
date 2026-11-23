# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Strided (e.g. transposed) scales: CK and CK-tile read the scales through raw
pointers, so gemm_a8w8_blockscale passes them contiguous, and the CK ops reject
them rather than misread them."""

import pytest
import torch

from aiter import dtypes
from aiter.ops import gemm_op_a8w8 as gemm_mod

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="a8w8 blockscale tests require a CUDA/HIP device",
)

BLOCK = 128
M, N, K = 33, 2560, 1280


def operands():
    torch.manual_seed(0)
    x = (torch.randn(M, K, device="cuda") * 2).to(dtypes.fp8)
    w = (torch.randn(N, K, device="cuda") * 2).to(dtypes.fp8)
    # Scales stored [K/128, M] and [K/128, N/128], passed as transposed views.
    x_scale = (torch.rand(K // BLOCK, M, device="cuda") + 0.5).t()
    w_scale = (torch.rand(K // BLOCK, N // BLOCK, device="cuda") + 0.5).t()
    assert not x_scale.is_contiguous() and not w_scale.is_contiguous()
    return x, w, x_scale, w_scale


@pytest.mark.parametrize("libtype", ["ck", "cktile"])
def test_strided_scales_match_contiguous(monkeypatch, libtype):
    config = {"libtype": libtype, "splitK": 0, "kernelName": ""}
    monkeypatch.setattr(gemm_mod, "get_CKGEMM_config", lambda *a, **k: config)
    x, w, x_scale, w_scale = operands()
    strided = gemm_mod.gemm_a8w8_blockscale(x, w, x_scale, w_scale)
    contiguous = gemm_mod.gemm_a8w8_blockscale(
        x, w, x_scale.contiguous(), w_scale.contiguous()
    )
    torch.testing.assert_close(strided, contiguous, rtol=0, atol=0)


@pytest.mark.parametrize(
    "op", [gemm_mod.gemm_a8w8_blockscale_ck, gemm_mod.gemm_a8w8_blockscale_cktile]
)
def test_ck_ops_reject_strided_scales(op):
    x, w, x_scale, w_scale = operands()
    y = torch.empty(M, N, dtype=dtypes.bf16, device="cuda")
    with pytest.raises(RuntimeError, match="must be contiguous"):
        op(x, w, x_scale, w_scale, y)
