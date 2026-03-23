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
    n_expert_group = 8 if num_experts >= 8 and num_experts % 8 == 0 else 1
    n_topk_group = 4 if n_expert_group >= 2 else 1
    max_topk = n_topk_group * (num_experts // n_expert_group)
    #n_topk = min(4, max_topk)
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
    if bias is None:
        bias = torch.zeros(num_experts, device=device, dtype=torch.float16)

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


def _run_grouped_topk_once(num_tokens: int, num_experts: int, scores=None, bias=None):
    # grouped_topk kernel expects float32 inputs currently
    device = "xpu"
    n_topk, n_group, topk_group = _group_topk_params(num_experts)
    
    if scores is None:
        scores = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    if bias is None:
        bias = torch.zeros(num_experts, device=device, dtype=torch.float32)

    values, indices = ext.grouped_topk(
        scores,
        n_group,
        topk_group,
        n_topk,
        True,
        1.0,
        bias,
        1,
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
    # Pre-allocate Fused Kernel inputs (FP16)
    fused_hidden = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.float16)
    fused_gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float16)
    fused_bias = torch.zeros(num_experts, device=device, dtype=torch.float16)

    # Pre-allocate Grouped Kernel inputs (FP32)
    grouped_scores = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    grouped_bias = torch.zeros(num_experts, device=device, dtype=torch.float32)

    print("Profiling Fused Kernel...")
    # Increase warmup and active iterations for better stability
    for _ in range(50):
        _run_op_once(num_tokens, hidden_dim, num_experts, fused_hidden, fused_gating, fused_bias)
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
                num_tokens, hidden_dim, num_experts, fused_hidden, fused_gating, fused_bias),
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
                num_tokens, num_experts, grouped_scores, grouped_bias),
            PROFILE_WARMUP,
            PROFILE_ACTIVE,
            PROFILE_REPEAT,
            activities,
            sort_by,
            PROFILE_TRACE_DIR,
        )


def _run_benchmark(num_tokens: int, hidden_dim: int, num_experts: int):
    if not torch.xpu.is_available():
        raise RuntimeError("XPU is not available for benchmark")

    device = "xpu"
    print("Allocating benchmark tensors...")
    fused_hidden = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.float16)
    fused_gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float16)
    fused_bias = torch.zeros(num_experts, device=device, dtype=torch.float16)
    grouped_scores = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    grouped_bias = torch.zeros(num_experts, device=device, dtype=torch.float32)

    benchmark_targets = []
    if BENCHMARK_TARGET in ("both", "fused"):
        benchmark_targets.append((
            "fused",
            lambda: _run_op_once(
                num_tokens, hidden_dim, num_experts,
                fused_hidden, fused_gating, fused_bias),
        ))
    if BENCHMARK_TARGET in ("both", "grouped"):
        benchmark_targets.append((
            "grouped",
            lambda: _run_grouped_topk_once(
                num_tokens, num_experts, grouped_scores, grouped_bias),
        ))

    for label, fn in benchmark_targets:
        print(f"\n=== Benchmark run ({label}) ===")
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
        num_experts = TEST_EXPERTS
        num_tokens = TEST_TOKENS
        hidden_dim = 712
        device = "xpu"
        n_topk, n_expert_group, n_topk_group = _group_topk_params(num_experts)
        routed_scaling_factor = 1.0

        hidden_states = torch.randn(
            num_tokens, hidden_dim, device=device, dtype=torch.float16
        )
        # gating_output here likely acts as 'scores' for grouped_topk reference logic
        # but fused_grouped_topk might compute scores internally if it includes the gating matmul.
        # Assuming for this test script that 'gating_output' is the pre-computed score matrix
        # tailored for the 'fused' kernel if it behaves that way.
        
        gating_output = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float16
        )
        bias = torch.zeros(num_experts, device=device, dtype=torch.float16)
        
        # Test 1: Run Fused Kernel
        values_fused, indices_fused = ext.fused_grouped_topk(
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
        
        # Test 2: Run Grouped TopK (The one we ported)
        # We use the same 'gating_output' as 'scores'
        values_grouped, indices_grouped = ext.grouped_topk(
            gating_output.float(),
            n_expert_group,
            n_topk_group,
            n_topk,
            True,
            routed_scaling_factor,
            bias.float(),
            1, # SCORING_SIGMOID
        )
        self.assertEqual(values_grouped.shape, (num_tokens, n_topk))
        self.assertEqual(indices_grouped.shape, (num_tokens, n_topk))
        # Note: Exact comparison might fail if floating point order differs or
        # if fused_grouped_topk does the MatMul internally vs grouped_topk taking scores.
        # But we can check bounds.
        self.assertTrue(torch.all(indices_grouped >= 0).item())
        self.assertTrue(torch.all(indices_grouped < num_experts).item())
        
        # Check indices match rate
        match_rate = (indices_fused == indices_grouped).float().mean().item()
        print(f"Indices match rate: {match_rate}")
        self.assertTrue(match_rate > 0.95, "Indices should match > 95%")
       

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
