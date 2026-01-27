from pathlib import Path
import modal

CURRENT_DIR = Path(__file__).parent
REMOTE_DIR = Path("/my_extension")

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu130")
    .uv_pip_install("ninja", "triton")
    .add_local_dir(CURRENT_DIR, remote_path=REMOTE_DIR)
)
app = modal.App("sm100-matmul", image=image)


def get_module():
    import torch
    import torch.utils.cpp_extension

    print(f"{torch.__version__=}")
    print(f"{torch.version.cuda=}")

    torch.utils.cpp_extension.load(
        "module",
        sources=[str(p) for p in REMOTE_DIR.glob("matmul*")],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["-lcuda"],
        verbose=True,
        is_python_module=False,
    )

    return torch.ops.my_matmul


@app.function(gpu="B200")
def test_matmul(version: str = "v0", shape: str = "1024,1024,1024", with_timing: bool = False):
    import torch

    my_matmul = get_module()

    M, N, K = map(int, shape.split(","))
    print(f"{M=}, {N=}, {K=}")
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)

    f = getattr(my_matmul, f"matmul_{version}")

    # Warmup
    for _ in range(3):
        _ = f(A, B)
    torch.cuda.synchronize()

    # Correctness check
    C_custom = f(A, B)
    C_ref = A @ B

    max_diff = (C_custom - C_ref).abs().max().item()
    print(f"Max difference: {max_diff}")

    if max_diff < 1e-3:
        print("PASSED!")
    else:
        print("FAILED!")

    # Timing with CUDA events
    if with_timing:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Time custom kernel
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(100):
            _ = f(A, B)
        end_event.record()
        torch.cuda.synchronize()

        custom_time_ms = start_event.elapsed_time(end_event) / 100
        custom_tflops = 2 * M * N * K / custom_time_ms / 1e9

        # Time cuBLAS
        start_event.record()
        for _ in range(100):
            _ = torch.matmul(A, B)
        end_event.record()
        torch.cuda.synchronize()

        cublas_time_ms = start_event.elapsed_time(end_event) / 100
        cublas_tflops = 2 * M * N * K / cublas_time_ms / 1e9

        print(f"\n{'='*50}")
        print(f"TIMING for {version} ({shape})")
        print(f"{'='*50}")
        print(f"cuBLAS:  {cublas_time_ms:.4f} ms  ({cublas_tflops:.2f} TFLOPS)")
        print(f"{version}:     {custom_time_ms:.4f} ms  ({custom_tflops:.2f} TFLOPS)")
        print(f"Ratio:   {custom_time_ms / cublas_time_ms:.2f}x slower than cuBLAS")
        print(f"{'='*50}\n")

    return max_diff


@app.function(gpu="B200")
def benchmark(shape: str = "4096,4096,4096"):
    import time
    import torch
    from triton.testing import do_bench

    my_matmul = get_module()

    M, N, K = map(int, shape.split(","))
    print(f"{M=}, {N=}, {K=}")
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)

    def bench_and_print(f, name):
        torch.cuda.synchronize()
        time.sleep(0.5)

        latency_ms = do_bench(lambda: f(A, B), warmup=10, rep=100, return_mode="median")
        tflops = 2 * M * N * K / latency_ms / 1e9
        print(f"{name}:\t{latency_ms:.4f} ms\t{tflops:.2f} TFLOPS")

    output_ref = torch.matmul(A, B)
    bench_and_print(torch.matmul, "CuBLAS")

    for version in ["v0", "v1"]:
        f = getattr(my_matmul, f"matmul_{version}")
        out = f(A, B)
        torch.cuda.synchronize()
        try:
            torch.testing.assert_close(out, output_ref, rtol=1e-3, atol=1e-3)
            print(f"{version}: Correctness check PASSED")
        except AssertionError as e:
            print(f"{version}: Correctness check FAILED")
            print(e)
            continue
        bench_and_print(f, version)


@app.function(gpu="B200")
def profile(version: str = "v0", shape: str = "1024,1024,1024"):
    import torch
    from collections import defaultdict

    TAGS = [
        "SETUP",
        "ISSUE_TMA",
        "ISSUE_MMA",
        "WAIT_TMA",
        "WAIT_MMA",
        "WAIT_MAINLOOP",
        "WAIT_EPILOGUE",
        "EPILOGUE",
    ]

    my_matmul = get_module()

    M, N, K = map(int, shape.split(","))
    print(f"{M=}, {N=}, {K=}")
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)

    f = getattr(my_matmul, f"profile_matmul_{version}", None)
    if f is None:
        print(f"No profiling function for {version}")
        return {"events": [], "stats": {}}

    # Calculate actual number of blocks based on grid size (16x16 block tiles)
    BLOCK_SIZE = 16
    num_blocks_x = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks_y = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    NUM_BLOCKS = num_blocks_x * num_blocks_y
    NUM_ENTRIES = 1000

    print(f"Grid size: {num_blocks_x} x {num_blocks_y} = {NUM_BLOCKS} blocks")

    profiler = torch.zeros(NUM_BLOCKS, 1 + NUM_ENTRIES * 4, dtype=torch.int64, device="cuda")

    for _ in range(5):
        f(A, B, profiler, NUM_ENTRIES)

    torch.cuda.synchronize()
    profiler.zero_()
    f(A, B, profiler, NUM_ENTRIES)
    torch.cuda.synchronize()

    profile_data = profiler.tolist()
    events = []

    # Aggregate stats by tag
    tag_durations = defaultdict(list)

    for bid, data in enumerate(profile_data):
        for i in range(min(data[0], NUM_ENTRIES)):
            sm_id, tag, start, duration = data[1 + i * 4 : 1 + (i + 1) * 4]
            events.append(dict(name=TAGS[tag], ph="X", ts=start, dur=duration, pid=sm_id, tid=sm_id + bid))
            tag_durations[TAGS[tag]].append(duration)

    if events:
        offset = min([evt["ts"] for evt in events])
        for evt in events:
            evt["ts"] -= offset

    # Print summary stats
    print("\n" + "=" * 50)
    print(f"PROFILE SUMMARY for {version} ({shape})")
    print("=" * 50)

    stats = {}
    total_duration = 0
    for tag in TAGS:
        if tag in tag_durations:
            durations = tag_durations[tag]
            avg = sum(durations) / len(durations)
            total = sum(durations)
            total_duration += total
            stats[tag] = {"count": len(durations), "avg_cycles": avg, "total_cycles": total}
            print(f"{tag:20s}: count={len(durations):6d}, avg={avg:12.1f} cycles, total={total:15.0f} cycles")

    print("-" * 50)
    print(f"{'TOTAL':20s}: {total_duration:15.0f} cycles")

    # Estimate time (B200 ~2GHz clock)
    gpu_clock_ghz = 2.0
    total_time_ms = total_duration / (gpu_clock_ghz * 1e9) * 1000
    print(f"Estimated wall time: {total_time_ms:.3f} ms (assuming {gpu_clock_ghz} GHz)")
    print("=" * 50 + "\n")

    return {"events": events, "stats": stats}


@app.local_entrypoint()
def main(action: str = "test", version: str = "v0", shape: str = "1024,1024,1024", timing: bool = True):
    if action == "test":
        result = test_matmul.remote(version, shape, timing)
        print(f"Result: {result}")

    elif action == "benchmark":
        benchmark.remote(shape)

    elif action == "profile":
        import gzip
        import json

        result = profile.remote(version, shape)
        events = result.get("events", [])
        if events:
            trace = dict(traceEvents=events)
            gzip.open("trace.json.gz", "w").write(json.dumps(trace).encode("utf-8"))
            print("Trace saved to trace.json.gz (view in chrome://tracing or Perfetto)")
        else:
            print("No profile data collected")

    elif action == "all":
        import gzip
        import json

        # Test + timing
        print("=" * 60)
        print("STEP 1: Testing correctness and timing")
        print("=" * 60)
        test_matmul.remote(version, shape, timing)

        # Profile breakdown
        print("\n" + "=" * 60)
        print("STEP 2: Detailed profiling")
        print("=" * 60)
        result = profile.remote(version, shape)
        events = result.get("events", [])
        if events:
            trace = dict(traceEvents=events)
            gzip.open("trace.json.gz", "w").write(json.dumps(trace).encode("utf-8"))
            print("Trace saved to trace.json.gz")
