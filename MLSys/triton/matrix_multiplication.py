"""Tiled Matrix Multiplication (GEMM) implemented with Triton.

This module implements a high-performance tiled matrix multiplication kernel
(C = A @ B) that leverages Triton's auto-tuning infrastructure to select the
best tile sizes and pipeline stages for the target GPU.

Key features:
    - 2-D tiling with configurable BLOCK_M / BLOCK_N / BLOCK_K.
    - L2 cache–friendly iteration order via `GROUP_SIZE_M` swizzling.
    - Triton `autotune` over multiple tile / stage / warp configurations.
    - Full autograd support (forward + backward) via `TritonMatmul.apply`.
    - Benchmark utility comparing against `torch.matmul` (cuBLAS).

Usage:
    python matrix_multiplication.py   # run correctness test + benchmark
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Autotune configurations
# ---------------------------------------------------------------------------

def _get_autotune_configs():
    """Return a list of `triton.Config` objects for autotuning.

    Each config specifies tile dimensions (BLOCK_M, BLOCK_N, BLOCK_K),
    the number of pipeline stages, and the number of warps.
    """
    return [
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8},
            num_stages=3, num_warps=8,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
            num_stages=5, num_warps=2,
        ),
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
            num_stages=5, num_warps=2,
        ),
    ]


# ---------------------------------------------------------------------------
# Forward kernel:  C = A @ B
# ---------------------------------------------------------------------------

@triton.autotune(configs=_get_autotune_configs(), key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(
    # 指针
    a_ptr, b_ptr, c_ptr,
    # 矩阵维度
    M, N, K,
    # 矩阵 strides (元素为单位)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # 编译期常量 (由 autotune 决定)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr = None,
):
    """Compute a BLOCK_M x BLOCK_N tile of C = A @ B.

    Each program instance is responsible for one output tile.  The K dimension
    is iterated in chunks of BLOCK_K, accumulating partial dot-products into
    a register-resident accumulator.

    L2 cache reuse is improved by *swizzling* the program-id → tile mapping:
    programs that are close in pid share either the same A-row-tile or the same
    B-col-tile, so the corresponding data is more likely to stay in L2.

    Args:
        a_ptr: Pointer to A of shape (M, K).
        b_ptr: Pointer to B of shape (K, N).
        c_ptr: Pointer to C of shape (M, N).
        M, N, K: Matrix dimensions.
        stride_am, stride_ak: Row / col strides of A.
        stride_bk, stride_bn: Row / col strides of B.
        stride_cm, stride_cn: Row / col strides of C.
        BLOCK_M, BLOCK_N, BLOCK_K: Tile dimensions (constexpr).
        GROUP_SIZE_M: Number of row-tiles grouped together for L2 swizzling.
        ACTIVATION: Optional fused activation (e.g. ``tl.math.fast_expf`` for
            element-wise exp after the matmul).  ``None`` means identity.
    """
    # ---- 1. 计算当前 program 所负责的输出 tile 坐标 ----
    pid = tl.program_id(axis=0)

    # 输出矩阵在 M 和 N 方向各有多少个 tile
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # L2 cache 友好的 swizzle: 将相邻 pid 映射到同一"组"内的 tile
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- 2. 计算 A / B 子块的起始指针偏移 ----
    # A 子块: 行范围 [pid_m*BM, pid_m*BM + BM), 列从 0 开始，每次步进 BK
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # 指向 A 的第一个 BLOCK_M x BLOCK_K 块
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # 指向 B 的第一个 BLOCK_K x BLOCK_N 块
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # ---- 3. 沿 K 维分块累加 ----
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # 边界 mask: 避免越界读取
        a_mask = offs_k[None, :] < (K - k * BLOCK_K)
        b_mask = offs_k[:, None] < (K - k * BLOCK_K)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # 矩阵乘累加 (在 SRAM 中完成)
        accumulator += tl.dot(a, b)

        # 移动到下一个 K 块
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 可选的逐元素激活函数
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)

    c = accumulator.to(tl.float16)

    # ---- 4. 将结果写回全局显存 ----
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def leaky_relu(x):
    """Leaky ReLU activation: max(0.01x, x)."""
    return tl.where(x >= 0, x, 0.01 * x)


# ---------------------------------------------------------------------------
# Python wrapper helpers
# ---------------------------------------------------------------------------

def triton_matmul(a: torch.Tensor, b: torch.Tensor, activation: str = "") -> torch.Tensor:
    """Launch the Triton GEMM kernel to compute C = A @ B.

    Args:
        a: Left operand of shape (M, K), fp16, CUDA.
        b: Right operand of shape (K, N), fp16, CUDA.
        activation: Optional fused activation name. Supported values:
            ``""`` (identity) or ``"leaky_relu"``.

    Returns:
        Result tensor C of shape (M, N), fp16.
    """
    assert a.shape[1] == b.shape[0], (
        f"Inner dimensions must match: A is {a.shape}, B is {b.shape}"
    )
    assert a.is_cuda and b.is_cuda, "Both inputs must be CUDA tensors"

    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # 1-D grid: 每个 program 算一个 BLOCK_M x BLOCK_N 的输出 tile
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation if activation else None,
    )
    return c


# ---------------------------------------------------------------------------
# PyTorch Autograd Function
# ---------------------------------------------------------------------------

class TritonMatmul(torch.autograd.Function):
    """Autograd wrapper for the Triton tiled GEMM.

    Forward:  C = A @ B
    Backward: grad_A = grad_C @ B^T
              grad_B = A^T @ grad_C
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(a, b)
        return triton_matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        a, b = ctx.saved_tensors

        # grad_A = grad_C @ B^T
        grad_a = triton_matmul(grad_output, b.t().contiguous())
        # grad_B = A^T @ grad_C
        grad_b = triton_matmul(a.t().contiguous(), grad_output)

        return grad_a, grad_b


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Convenience API: compute A @ B using the Triton GEMM kernel.

    Supports autograd. Inputs are cast to fp16 internally if needed.

    Args:
        a: Left operand of shape (M, K).
        b: Right operand of shape (K, N).

    Returns:
        Result tensor of shape (M, N).
    """
    if a.dtype != torch.float16:
        a = a.half()
    if b.dtype != torch.float16:
        b = b.half()
    a = a.contiguous()
    b = b.contiguous()
    return TritonMatmul.apply(a, b)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],
        x_vals=[128 * i for i in range(2, 33)],
        line_arg='provider',
        line_vals=['triton', 'cublas'],
        line_names=['Triton', 'cuBLAS'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='TFLOPS',
        plot_name='matmul-performance',
        args={},
    )
)
def benchmark(M, N, K, provider):
    """Benchmark Triton GEMM vs torch.matmul (cuBLAS).

    Reports throughput in TFLOPS (2 * M * N * K floating-point ops).
    """
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a, b), quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_matmul(a, b), quantiles=quantiles
        )

    # 计算 TFLOPS: 矩阵乘的浮点运算量 = 2 * M * N * K
    tflops = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return tflops(ms), tflops(max_ms), tflops(min_ms)


# ---------------------------------------------------------------------------
# Correctness test
# ---------------------------------------------------------------------------

def test_correctness():
    """Verify Triton GEMM matches torch.matmul (forward + backward)."""
    torch.manual_seed(0)
    shapes = [
        (16, 16, 16),
        (32, 64, 128),
        (128, 256, 64),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (37, 51, 73),       # 非对齐尺寸
        (128, 513, 255),    # 非对齐尺寸
    ]

    for m, n, k in shapes:
        # ---- Forward ----
        a = torch.randn((m, k), device='cuda', dtype=torch.float16)
        b = torch.randn((k, n), device='cuda', dtype=torch.float16)

        c_ref = torch.matmul(a, b)
        c_tri = triton_matmul(a, b)

        assert torch.allclose(c_ref, c_tri, atol=1e-2, rtol=1e-2), (
            f"Forward mismatch for shape ({m}, {n}, {k})\n"
            f"  max diff = {(c_ref - c_tri).abs().max().item():.6f}"
        )

        print(f"  [PASS] shape=({m:>4d}, {n:>4d}, {k:>4d})  "
              f"max_diff={( c_ref - c_tri).abs().max().item():.6f}")

    # ---- Leaky ReLU fusion test ----
    print("\n  Testing fused leaky_relu activation...")
    a = torch.randn((128, 64), device='cuda', dtype=torch.float16)
    b = torch.randn((64, 256), device='cuda', dtype=torch.float16)
    c_tri = triton_matmul(a, b, activation="leaky_relu")
    c_ref = torch.matmul(a, b)
    c_ref = torch.where(c_ref >= 0, c_ref, 0.01 * c_ref)
    assert torch.allclose(c_ref, c_tri, atol=1e-2, rtol=1e-2), (
        f"Leaky ReLU fusion mismatch, "
        f"max diff = {(c_ref - c_tri).abs().max().item():.6f}"
    )
    print("  [PASS] Fused leaky_relu activation")

    # ---- Backward (autograd) test ----
    print("\n  Testing backward pass (autograd)...")
    a = torch.randn((64, 32), device='cuda', dtype=torch.float16, requires_grad=True)
    b = torch.randn((32, 48), device='cuda', dtype=torch.float16, requires_grad=True)
    a_ref = a.clone().detach().requires_grad_(True)
    b_ref = b.clone().detach().requires_grad_(True)

    c_ref = torch.matmul(a_ref, b_ref)
    c_tri = TritonMatmul.apply(a, b)

    grad = torch.randn_like(c_ref)
    c_ref.backward(grad)
    c_tri.backward(grad)

    assert torch.allclose(a_ref.grad, a.grad, atol=1e-1, rtol=1e-1), (
        f"Backward grad_A mismatch, "
        f"max diff = {(a_ref.grad - a.grad).abs().max().item():.6f}"
    )
    assert torch.allclose(b_ref.grad, b.grad, atol=1e-1, rtol=1e-1), (
        f"Backward grad_B mismatch, "
        f"max diff = {(b_ref.grad - b.grad).abs().max().item():.6f}"
    )
    print("  [PASS] Backward pass (autograd)")

    print("\nAll correctness tests passed!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Tiled Matrix Multiplication (GEMM) — Correctness Test")
    print("=" * 60)
    test_correctness()

    print()
    print("=" * 60)
    print("Tiled Matrix Multiplication (GEMM) — Benchmark (Triton vs cuBLAS)")
    print("=" * 60)
    benchmark.run(show_plots=False, print_data=True)
