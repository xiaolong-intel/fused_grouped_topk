from typing import Optional, Tuple

import torch
from torch import Tensor


@torch.library.register_fake("sycl_fused_grouped_topk::fused_grouped_topk")
def _fake_fused_grouped_topk(
    hidden_states: Tensor,
    gating_output: Tensor,
    n_topk: int,
    renormalize: bool,
    n_expert_group: int,
    n_topk_group: int,
    scoring_func: str,
    routed_scaling_factor: float,
    bias: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    num_tokens = gating_output.shape[0]
    values = torch.empty(
        (num_tokens, n_topk),
        device=gating_output.device,
        dtype=torch.float32,
    )
    indices = torch.empty(
        (num_tokens, n_topk),
        device=gating_output.device,
        dtype=torch.int32,
    )
    return values, indices


def fused_grouped_topk(
    hidden_states: Tensor,
    gating_output: Tensor,
    n_topk: int,
    renormalize: bool,
    n_expert_group: int,
    n_topk_group: int,
    scoring_func: str = "sigmoid",
    routed_scaling_factor: float = 1.0,
    bias: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    return torch.ops.sycl_fused_grouped_topk.fused_grouped_topk.default(
        hidden_states,
        gating_output,
        n_topk,
        renormalize,
        n_expert_group,
        n_topk_group,
        scoring_func,
        routed_scaling_factor,
        bias,
    )


@torch.library.register_fake("sycl_fused_grouped_topk::grouped_topk")
def _fake_grouped_topk(
    scores: Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: Tensor,
    scoring_func: int,
) -> Tuple[Tensor, Tensor]:
    num_tokens = scores.shape[0]
    values = torch.empty(
        (num_tokens, topk),
        device=scores.device,
        dtype=torch.float32,
    )
    indices = torch.empty(
        (num_tokens, topk),
        device=scores.device,
        dtype=torch.int32,
    )
    return values, indices


def grouped_topk(
    scores: Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: Tensor,
    scoring_func: int = 1,
) -> Tuple[Tensor, Tensor]:
    return torch.ops.sycl_fused_grouped_topk.grouped_topk.default(
        scores,
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )

