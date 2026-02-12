"""Fused Softmax operator implemented with Triton.

This module implements a numerically-stable fused Softmax kernel that
performs the entire softmax computation (max-subtract, exp, sum, normalize)
in a single GPU kernel launch, avoiding redundant global memory round-trips
compared to the naive PyTorch decomposition.

Key features:
    - Online numerically-stable softmax (subtract row-max before exp).
    - Supports arbitrary row lengths via BLOCK_SIZE compile-time constant.
    - Full autograd support (forward + backward) via `FusedSoftmax.apply`.
    - Benchmark utility comparing against `torch.softmax`.

Usage:
    python softmax.py              # run correctness test + benchmark
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute softmax for one row per program instance.

    Each program handles exactly one row of the input matrix.  The row is
    loaded into SRAM, and the three reduction passes (max, exp-sum, normalize)
    are fused into a single kernel so global memory is read once and written
    once.

    Args:
        output_ptr: Pointer to the output tensor (same shape as input).
        input_ptr: Pointer to the input tensor of shape (M, N).
        input_row_stride: Stride (in elements) between consecutive rows of
            the input tensor.
        output_row_stride: Stride (in elements) between consecutive rows of
            the output tensor.
        n_cols: Number of columns N in each row.
        BLOCK_SIZE: Compile-time block size, must be >= n_cols and a power
            of two.
    """
    # 每个 program 处理一行
    row_idx = tl.program_id(0)

    # 计算当前行的起始指针
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # 列偏移向量 [0, 1, 2, ..., BLOCK_SIZE - 1]
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 从全局显存加载一整行到 SRAM，超出 n_cols 的位置用 -inf 填充
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    # ---------- 数值稳定的 softmax ----------
    # 1. 求行最大值 (用于数值稳定)
    row_max = tl.max(row, axis=0)

    # 2. 减去最大值后求 exp
    numerator = tl.exp(row - row_max)

    # 3. 求 exp 之和
    denominator = tl.sum(numerator, axis=0)

    # 4. 归一化
    softmax_output = numerator / denominator

    # 写回全局显存
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


# ---------------------------------------------------------------------------
# Backward kernel
# ---------------------------------------------------------------------------

@triton.jit
def softmax_backward_kernel(
    grad_input_ptr,
    grad_output_ptr,
    output_ptr,
    grad_input_row_stride,
    grad_output_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute the backward pass of softmax for one row per program.

    Given the forward output ``y = softmax(x)`` and upstream gradient
    ``dy``, the local Jacobian gives:
        dx_i = y_i * (dy_i - sum_j(y_j * dy_j))

    Args:
        grad_input_ptr: Pointer to the gradient w.r.t. input (output).
        grad_output_ptr: Pointer to the upstream gradient dy.
        output_ptr: Pointer to the forward softmax output y.
        grad_input_row_stride: Row stride for grad_input.
        grad_output_row_stride: Row stride for grad_output.
        output_row_stride: Row stride for forward output.
        n_cols: Number of columns.
        BLOCK_SIZE: Compile-time block size (power of two, >= n_cols).
    """
    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # 加载前向输出 y 和上游梯度 dy
    out_row_ptr = output_ptr + row_idx * output_row_stride
    y = tl.load(out_row_ptr + col_offsets, mask=mask, other=0.0)

    grad_out_row_ptr = grad_output_ptr + row_idx * grad_output_row_stride
    dy = tl.load(grad_out_row_ptr + col_offsets, mask=mask, other=0.0)

    # dx_i = y_i * (dy_i - sum_j(y_j * dy_j))
    sum_ydy = tl.sum(y * dy, axis=0)
    dx = y * (dy - sum_ydy)

    # 写回
    grad_in_row_ptr = grad_input_ptr + row_idx * grad_input_row_stride
    tl.store(grad_in_row_ptr + col_offsets, dx, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper helpers
# ---------------------------------------------------------------------------

def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 that is >= n."""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def fused_softmax_forward(x: torch.Tensor) -> torch.Tensor:
    """Launch the Triton softmax forward kernel.

    Args:
        x: Input tensor of shape (M, N), must be contiguous and on CUDA.

    Returns:
        Softmax output tensor of the same shape and dtype.
    """
    assert x.is_cuda and x.ndim == 2, "Input must be a 2-D CUDA tensor"
    n_rows, n_cols = x.shape
    # BLOCK_SIZE 必须 >= n_cols 且为 2 的幂
    BLOCK_SIZE = _next_power_of_2(n_cols)

    output = torch.empty_like(x)

    # 每行一个 program instance
    softmax_kernel[(n_rows,)](
        output,
        x,
        x.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def fused_softmax_backward(
    grad_output: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Launch the Triton softmax backward kernel.

    Args:
        grad_output: Upstream gradient tensor of shape (M, N).
        output: Forward softmax output tensor of shape (M, N).

    Returns:
        Gradient w.r.t. input tensor of the same shape.
    """
    n_rows, n_cols = output.shape
    BLOCK_SIZE = _next_power_of_2(n_cols)

    grad_input = torch.empty_like(output)

    softmax_backward_kernel[(n_rows,)](
        grad_input,
        grad_output,
        output,
        grad_input.stride(0),
        grad_output.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return grad_input


# ---------------------------------------------------------------------------
# PyTorch Autograd Function
# ---------------------------------------------------------------------------

class FusedSoftmax(torch.autograd.Function):
    """Autograd wrapper for the Triton fused softmax."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        output = fused_softmax_forward(x)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (output,) = ctx.saved_tensors
        grad_input = fused_softmax_backward(grad_output, output)
        return grad_input


def fused_softmax(x: torch.Tensor) -> torch.Tensor:
    """Convenience API: compute fused softmax along the last dimension.

    Supports autograd. Input is reshaped to 2-D internally if needed.

    Args:
        x: Input tensor of arbitrary shape (softmax is applied along dim=-1).

    Returns:
        Softmax output of the same shape.
    """
    original_shape = x.shape
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim > 2:
        x = x.view(-1, x.shape[-1])
    x = x.contiguous()
    out = FusedSoftmax.apply(x)
    return out.view(original_shape)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='fused-softmax-throughput',
        args={'M': 4096},
    )
)
def benchmark(M, N, provider):
    """Benchmark Triton fused softmax vs torch.softmax.

    Reports throughput in GB/s (read + write = 2 * M * N * element_size).
    """
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.softmax(x, dim=-1), quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fused_softmax_forward(x), quantiles=quantiles
        )

    # 吞吐量: 读一次 + 写一次
    gbps = lambda ms: 2 * M * N * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# ---------------------------------------------------------------------------
# Correctness test
# ---------------------------------------------------------------------------

def test_correctness():
    """Verify Triton fused softmax matches torch.softmax (forward + backward)."""
    torch.manual_seed(0)
    shapes = [(1, 37), (2, 128), (64, 513), (128, 1024), (256, 4096)]
    for m, n in shapes:
        # ---- Forward ----
        x = torch.randn(m, n, device='cuda', dtype=torch.float32)
        y_ref = torch.softmax(x, dim=-1)
        y_tri = fused_softmax_forward(x)
        assert torch.allclose(y_ref, y_tri, atol=1e-5, rtol=1e-5), (
            f"Forward mismatch for shape ({m}, {n})"
        )

        # ---- Backward ----
        x_ref = x.clone().requires_grad_(True)
        x_tri = x.clone().requires_grad_(True)

        out_ref = torch.softmax(x_ref, dim=-1)
        out_tri = FusedSoftmax.apply(x_tri)

        grad = torch.randn_like(out_ref)
        out_ref.backward(grad)
        out_tri.backward(grad)

        assert torch.allclose(
            x_ref.grad, x_tri.grad, atol=1e-5, rtol=1e-5
        ), f"Backward mismatch for shape ({m}, {n})"

        print(f"  [PASS] shape=({m:>4d}, {n:>4d})")

    print("All correctness tests passed!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Fused Softmax — Correctness Test")
    print("=" * 60)
    test_correctness()

    print()
    print("=" * 60)
    print("Fused Softmax — Benchmark (Triton vs PyTorch)")
    print("=" * 60)
    benchmark.run(show_plots=False, print_data=True)
