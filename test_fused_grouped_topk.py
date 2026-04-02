import argparse
from pathlib import Path
import sys
import unittest
import time
import torch

import sycl_fused_grouped_topk as ext


TEST_TOKENS = 4096
TEST_HIDDEN = 712
TEST_EXPERTS = 256
ENABLE_PROFILE = False
ENABLE_BENCHMARK = False
PROFILE_WARMUP = 10
PROFILE_ACTIVE = 30
PROFILE_REPEAT = 1
PROFILE_SORT = None
PROFILE_ROW_LIMIT = 20
PROFILE_TARGET = "both"
PROFILE_TRACE_DIR = "./tensorboard_traces"
BENCHMARK_WARMUP = 50
BENCHMARK_ITERS = 200
BENCHMARK_TARGET = "grouped"


def _parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Run minimal fused_grouped_topk XPU test",
        add_help=False,
    )
    parser.add_argument("-t", "--tokens", type=int, default=TEST_TOKENS, dest="tokens")
    parser.add_argument("-h", "--hidden", type=int, default=TEST_HIDDEN, dest="hidden")
    parser.add_argument("-e", "--experts", type=int, default=TEST_EXPERTS, dest="experts")
    parser.add_argument("--profile", action="store_true", dest="profile")
    parser.add_argument("--benchmark", action="store_true", dest="benchmark")
    parser.add_argument("--profile-warmup", type=int, default=PROFILE_WARMUP, dest="profile_warmup")
    parser.add_argument("--profile-active", type=int, default=PROFILE_ACTIVE, dest="profile_active")
    parser.add_argument("--profile-repeat", type=int, default=PROFILE_REPEAT, dest="profile_repeat")
    parser.add_argument("--profile-sort", type=str, default=PROFILE_SORT, dest="profile_sort")
    parser.add_argument("--profile-row-limit", type=int, default=PROFILE_ROW_LIMIT, dest="profile_row_limit")
    parser.add_argument(
        "--profile-target",
        type=str,
        default=PROFILE_TARGET,
        choices=("both", "fused", "grouped"),
        dest="profile_target",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default=PROFILE_TRACE_DIR,
        dest="tensorboard_dir",
        help="Directory for TensorBoard profiler traces",
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=BENCHMARK_WARMUP,
        dest="benchmark_warmup",
        help="Number of warmup iterations before benchmark timing",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=BENCHMARK_ITERS,
        dest="benchmark_iters",
        help="Number of timed benchmark iterations",
    )
    parser.add_argument(
        "--benchmark-target",
        type=str,
        default=BENCHMARK_TARGET,
        choices=("both", "fused", "grouped"),
        dest="benchmark_target",
        help="Kernel path to benchmark in loop mode",
    )
    parser.add_argument("--help", action="help", help="show this help message and exit")
    return parser.parse_known_args()


def _group_topk_params(num_experts: int):
    #n_expert_group = 8 if num_experts >= 8 and num_experts % 8 == 0 else 1
    n_expert_group = 8
    n_topk_group = 2
    max_topk = n_topk_group * (num_experts // n_expert_group)
    n_topk = 8
    return n_topk, n_expert_group, n_topk_group


def _run_op_once(num_tokens: int, hidden_dim: int, num_experts: int, hidden_states=None, gating_output=None, bias=None):
    device = "xpu"
    n_topk, n_expert_group, n_topk_group = _group_topk_params(num_experts)
    routed_scaling_factor = 1.0
    
    if hidden_states is None:
        hidden_states = torch.randn(
            num_tokens, hidden_dim, device=device, dtype=torch.float16
        )
    if gating_output is None:
        gating_output = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float16
        )
    values, indices = ext.fused_grouped_topk(
        hidden_states,
        gating_output,
        n_topk,
        True,
        n_expert_group,
        n_topk_group,
        "sigmoid",
        routed_scaling_factor,
        bias,
    )
    return values, indices, n_topk


def _run_grouped_topk_once(num_tokens: int, num_experts: int, hidden_states=None, gating_output=None, bias=None):
    # grouped_topk kernel now matches fused_grouped_topk signature
    device = "xpu:7"
    hidden_dim = 712
    n_topk, n_group, topk_group = _group_topk_params(num_experts)
    
    if hidden_states is None:
        hidden_states = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.float32)
    if gating_output is None:
        gating_output = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    values, indices = ext.grouped_topk(
        hidden_states,
        gating_output,
        n_topk,
        True,
        n_group,
        topk_group,
        "sigmoid",
        1.0,
        bias,
    )
    return values, indices, n_topk



def _profile_kernel(label: str, fn, warmup_steps: int, active_steps: int, repeat: int,
                    activities, sort_by: str, tensorboard_dir: str | None):
    use_tensorboard = tensorboard_dir is not None
    on_trace_ready = None
    trace_root = None
    if use_tensorboard:
        trace_root = Path(tensorboard_dir) / label
        trace_root.mkdir(parents=True, exist_ok=True)
        on_trace_ready = torch.profiler.tensorboard_trace_handler(str(trace_root),use_gzip=False)

    schedule = torch.profiler.schedule(
        wait=0,
        warmup=warmup_steps,
        active=active_steps,
        repeat=repeat,
    )

    total_steps = (warmup_steps + active_steps) * repeat
    if total_steps <= 0:
        raise ValueError("Profiler schedule must have at least one step")

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=on_trace_ready,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        experimental_config=torch.profiler.ExperimentalConfig(
            enable_profiling=True,
            verbose=True,
        ) if hasattr(torch.profiler, 'ExperimentalConfig') else None,
    ) as prof:
        for _ in range(total_steps):
            fn()
            prof.step()

    torch.xpu.synchronize()
    print(f"\n=== Profile run ({label}) ===")
    print(prof.key_averages().table(sort_by=sort_by, row_limit=PROFILE_ROW_LIMIT))
    if trace_root is not None:
        print(f"TensorBoard trace written to: {trace_root}")


def _run_profile(num_tokens: int, hidden_dim: int, num_experts: int):
    if not torch.xpu.is_available():
        raise RuntimeError("XPU is not available for profiling")
    
    device = "xpu"
    print("Allocating profiling tensors...")

    test_hidden = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.float16)
    test_gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float16)
    test_bias = torch.zeros(num_experts, device=device, dtype=torch.float16)

    print("Profiling Fused Kernel...")
    # Increase warmup and active iterations for better stability
    for _ in range(50):
        _run_grouped_topk_once(num_tokens, num_experts, test_hidden, test_gating, test_bias)
    torch.xpu.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if hasattr(torch.profiler.ProfilerActivity, "XPU"):
        activities.append(torch.profiler.ProfilerActivity.XPU)

    default_sort = (
        "self_xpu_time_total"
        if hasattr(torch.profiler.ProfilerActivity, "XPU")
        else "self_cpu_time_total"
    )
    sort_by = PROFILE_SORT or default_sort
    if PROFILE_TARGET in ("both", "fused"):
        _profile_kernel(
            "fused",
            lambda: _run_op_once(
                num_tokens, hidden_dim, num_experts, test_hidden, test_gating, test_bias),
            PROFILE_WARMUP,
            PROFILE_ACTIVE,
            PROFILE_REPEAT,
            activities,
            sort_by,
            PROFILE_TRACE_DIR,
        )

    torch.xpu.synchronize()
    print("Profiling Grouped Kernel...")
    if PROFILE_TARGET in ("both", "grouped"):
        _profile_kernel(
            "grouped",
            lambda: _run_grouped_topk_once(
                num_tokens, num_experts, test_hidden, test_gating, test_bias),
            PROFILE_WARMUP,
            PROFILE_ACTIVE,
            PROFILE_REPEAT,
            activities,
            sort_by,
            PROFILE_TRACE_DIR,
        )

def _pytorch_grouped_topk_reference(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"
    # Move to CPU to avoid XPU OOM on intermediate tensors
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_token = scores.size(0)
    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )
    else:
        group_scores = (
            scores.view(num_token, num_expert_group, -1).max(dim=-1).values
        )  # [n, n_group]
    # For batch invariance, use sorted=True to ensure deterministic expert selection
    use_sorted = True
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=use_sorted)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.size(-1) // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=use_sorted)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(
            tmp_scores, k=topk, dim=-1, sorted=use_sorted
        )

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

def _run_benchmark(num_tokens: int, hidden_dim: int, num_experts: int):
    if not torch.xpu.is_available():
        raise RuntimeError("XPU is not available for benchmark")

    device = "xpu"
    print("Allocating benchmark tensors...")
    test_hidden = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)
    test_gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.bfloat16)
    test_bias = torch.zeros(num_experts, device=device, dtype=torch.bfloat16)

    benchmark_targets = []
    if BENCHMARK_TARGET in ("both", "fused"):
        benchmark_targets.append((
            "fused",
            lambda: _run_op_once(
                num_tokens, hidden_dim, num_experts,
                test_hidden, test_gating, test_bias),
        ))
    if BENCHMARK_TARGET in ("both", "grouped"):
        benchmark_targets.append((
            "grouped",
            lambda: _run_grouped_topk_once(
                num_tokens, num_experts, test_hidden, test_gating, test_bias),
        ))

    for label, fn in benchmark_targets:
        print(f"\n=== Benchmark run ({label}) ===")
        print("testing dtype=", test_hidden.dtype)
        start_event = torch.xpu.Event(enable_timing=True)
        end_event = torch.xpu.Event(enable_timing=True)
        for _ in range(BENCHMARK_WARMUP):
            fn()
        torch.xpu.synchronize()

        #start_time = time.perf_counter()
        start_event.record()
        for _ in range(BENCHMARK_ITERS):
            fn()
        end_event.record()
        torch.xpu.synchronize()
        elapsed_s = start_event.elapsed_time(end_event) / 1000.0
        total_ms = elapsed_s * 1000.0
        avg_ms = total_ms / BENCHMARK_ITERS
        print(f"warmup_iters={BENCHMARK_WARMUP} benchmark_iters={BENCHMARK_ITERS}")
        print(f"total_ms={total_ms:.3f} avg_ms={avg_ms:.6f}")

class TestFusedGroupedTopkMinimal(unittest.TestCase):
    @unittest.skipIf(not torch.xpu.is_available(), "requires Intel XPU")
    def test_correct_validation(self):
        print("start test_correct_validation")
        num_experts = 256
        num_tokens = TEST_TOKENS
        hidden_dim = 712
        device = "xpu:7"
        # n_topk, n_expert_group, n_topk_group = _group_topk_params(num_experts)
        n_topk = 8
        n_expert_group = 8
        n_topk_group = 4
        routed_scaling_factor = 1.0

        hidden_states = torch.randn(
            num_tokens, hidden_dim, device=device, dtype=torch.bfloat16
        )
        # gating_output here likely acts as 'scores' for grouped_topk reference logic
        # but fused_grouped_topk might compute scores internally if it includes the gating matmul.
        # Assuming for this test script that 'gating_output' is the pre-computed score matrix
        # tailored for the 'fused' kernel if it behaves that way.
        
        gating_output = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float16
        )
        bias = torch.zeros(num_experts, device=device, dtype=torch.bfloat16)


        # Test 1: Run Fused Kernel
        values_fused, indices_fused = ext.fused_grouped_topk(
            hidden_states,
            gating_output,
            n_topk,
            True,
            n_expert_group,
            n_topk_group,
            "softmax",
            routed_scaling_factor,
            None,
        )

        # Test 2: Run Grouped TopK (The one we ported)
        # We use the same 'gating_output' as 'scores'
        values_grouped, indices_grouped = ext.grouped_topk(
            hidden_states,
            gating_output,
            n_topk,
            True,
            n_expert_group,
            n_topk_group,
            "softmax",
            routed_scaling_factor,
            None,
        )

        values_cpu, indices_cpu = _pytorch_grouped_topk_reference(
            hidden_states,
            gating_output,
            n_topk,
            True,
            n_expert_group,
            n_topk_group,
            "softmax",
            routed_scaling_factor,
            None,
        )
        self.assertEqual(values_grouped.shape, (num_tokens, n_topk))
        self.assertEqual(indices_grouped.shape, (num_tokens, n_topk))
        self.assertEqual(values_cpu.shape, (num_tokens, n_topk))
        self.assertEqual(indices_cpu.shape, (num_tokens, n_topk))
        # Note: Exact comparison might fail if floating point order differs or
        # if fused_grouped_topk does the MatMul internally vs grouped_topk taking scores.
        # But we can check bounds.
        self.assertTrue(torch.all(indices_grouped >= 0).item())
        self.assertTrue(torch.all(indices_grouped < num_experts).item())
        
        # Check indices match rate
        indices_fused_cpu = indices_fused.detach().cpu()
        values_fused_cpu = values_fused.detach().cpu()
        indices_grouped_cpu = indices_grouped.detach().cpu()
        values_grouped_cpu = values_grouped.detach().cpu()
        indices_cpu_ref = indices_cpu.detach().cpu()
        values_cpu_ref = values_cpu.detach().cpu()

        mask_fused_grouped = indices_fused_cpu == indices_grouped_cpu
        mask_grouped_cpu = indices_grouped_cpu == indices_cpu_ref
        mask_fused_cpu = indices_fused_cpu == indices_cpu_ref
        combined_mask = mask_fused_grouped & mask_grouped_cpu & mask_fused_cpu

        if not combined_mask.all().item():
            mismatch_positions = (~combined_mask).nonzero(as_tuple=False).cpu()
            print("First mismatched entries:")
            for token_idx, topk_idx in mismatch_positions[:20].tolist():
                print(
                    "token=", token_idx,
                    "topk=", topk_idx,
                    "fused_idx=", int(indices_fused_cpu[token_idx, topk_idx]),
                    "grouped_idx=", int(indices_grouped_cpu[token_idx, topk_idx]),
                    "cpu_idx=", int(indices_cpu_ref[token_idx, topk_idx]),
                    "fused_val=", float(values_fused_cpu[token_idx, topk_idx]),
                    "grouped_val=", float(values_grouped_cpu[token_idx, topk_idx]),
                    "cpu_val=", float(values_cpu_ref[token_idx, topk_idx]),
                )
            mismatch_tokens = torch.unique(mismatch_positions[:, 0]).tolist()
            print(f"Mismatch token rows (up to 20): {mismatch_tokens[:20]}")
        
        match_rate_fused_grouped = mask_fused_grouped.float().mean().item()
        match_rate_grouped_cpu = mask_grouped_cpu.float().mean().item()
        match_rate_fused_cpu = mask_fused_cpu.float().mean().item()
        print(f"Indices match rate fused vs grouped: {match_rate_fused_grouped}")
        print(f"Indices match rate grouped vs cpu: {match_rate_grouped_cpu}")
        print(f"Indices match rate fused vs cpu: {match_rate_fused_cpu}")
        self.assertTrue(match_rate_grouped_cpu == 1, "Grouped kernel should match CPU reference 100%")

    
       

if __name__ == "__main__":
    args, remaining = _parse_cli_args()
    TEST_TOKENS = args.tokens
    TEST_HIDDEN = args.hidden
    TEST_EXPERTS = args.experts
    ENABLE_PROFILE = args.profile
    ENABLE_BENCHMARK = args.benchmark
    PROFILE_WARMUP = args.profile_warmup
    PROFILE_ACTIVE = args.profile_active
    PROFILE_REPEAT = args.profile_repeat
    PROFILE_SORT = args.profile_sort
    PROFILE_ROW_LIMIT = args.profile_row_limit
    PROFILE_TARGET = args.profile_target
    PROFILE_TRACE_DIR = args.tensorboard_dir
    BENCHMARK_WARMUP = args.benchmark_warmup
    BENCHMARK_ITERS = args.benchmark_iters
    BENCHMARK_TARGET = args.benchmark_target

    if TEST_TOKENS <= 0 or TEST_HIDDEN <= 0 or TEST_EXPERTS <= 0:
        raise SystemExit("-t/-h/-e must be positive integers")
    if PROFILE_WARMUP < 0 or PROFILE_ACTIVE <= 0 or PROFILE_REPEAT <= 0:
        raise SystemExit("profile warmup>=0, active>0, repeat>0")
    if PROFILE_ROW_LIMIT <= 0:
        raise SystemExit("--profile-row-limit must be > 0")
    if BENCHMARK_WARMUP < 0 or BENCHMARK_ITERS <= 0:
        raise SystemExit("benchmark warmup>=0, benchmark iters>0")
    if PROFILE_TRACE_DIR:
        Path(PROFILE_TRACE_DIR).mkdir(parents=True, exist_ok=True)

    if ENABLE_BENCHMARK:
        _run_benchmark(TEST_TOKENS, TEST_HIDDEN, TEST_EXPERTS)

    if ENABLE_PROFILE:
        _run_profile(TEST_TOKENS, TEST_HIDDEN, TEST_EXPERTS)

    if (ENABLE_BENCHMARK or ENABLE_PROFILE) and not remaining:
        raise SystemExit(0)

    sys.argv = [sys.argv[0], *remaining]
    unittest.main()
