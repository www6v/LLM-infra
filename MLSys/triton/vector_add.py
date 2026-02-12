"""Triton Vector Addition Kernel.

This module implements a simple vector addition kernel using Triton,
demonstrating the basic usage of Triton for GPU programming.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for element-wise vector addition.

    Each program instance processes a contiguous block of elements.

    Args:
        x_ptr: Pointer to the first input vector.
        y_ptr: Pointer to the second input vector.
        output_ptr: Pointer to the output vector.
        n_elements: Total number of elements in the vectors.
        BLOCK_SIZE: Number of elements each program instance processes.
    """
    # Determine the starting index for this program instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to guard memory operations against out-of-bounds accesses
    mask = offsets < n_elements

    # Load input vectors from DRAM
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Perform element-wise addition
    output = x + y

    # Store the result back to DRAM
    tl.store(output_ptr + offsets, output, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute element-wise vector addition using Triton.

    Args:
        x: First input tensor (must be on CUDA device).
        y: Second input tensor (must be on CUDA device, same shape as x).

    Returns:
        Output tensor containing x + y.

    Raises:
        AssertionError: If inputs are not 1-D, not on CUDA, or have mismatched shapes.
    """
    assert x.is_cuda and y.is_cuda, "Input tensors must be on CUDA device"
    assert x.shape == y.shape, "Input tensors must have the same shape"

    output = torch.empty_like(x)
    n_elements = output.numel()

    # Configure the grid: each program instance handles BLOCK_SIZE elements
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch the Triton kernel
    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output


def benchmark(size: int, dtype=torch.float32, num_warmup: int = 10, num_iters: int = 100):
    """Benchmark Triton vector_add against PyTorch native addition.

    Args:
        size: Number of elements in the vectors.
        dtype: Data type for the tensors.
        num_warmup: Number of warmup iterations.
        num_iters: Number of timed iterations.
    """
    x = torch.randn(size, device='cuda', dtype=dtype)
    y = torch.randn(size, device='cuda', dtype=dtype)

    # Warmup
    for _ in range(num_warmup):
        _ = vector_add(x, y)
        _ = x + y
    torch.cuda.synchronize()

    # Benchmark Triton
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        _ = vector_add(x, y)
    end.record()
    torch.cuda.synchronize()
    triton_time = start.elapsed_time(end) / num_iters

    # Benchmark PyTorch
    start.record()
    for _ in range(num_iters):
        _ = x + y
    end.record()
    torch.cuda.synchronize()
    torch_time = start.elapsed_time(end) / num_iters

    print(f"Vector size: {size:>10d}")
    print(f"  Triton:  {triton_time:.4f} ms")
    print(f"  PyTorch: {torch_time:.4f} ms")
    print(f"  Speedup: {torch_time / triton_time:.2f}x")
    print()


def main():
    """Run correctness test and performance benchmark."""
    # --- Correctness Test ---
    print("=" * 50)
    print("Correctness Test")
    print("=" * 50)

    torch.manual_seed(0)
    size = 98432  # Use a non-power-of-2 size to test masking logic
    x = torch.randn(size, device='cuda')
    y = torch.randn(size, device='cuda')

    triton_output = vector_add(x, y)
    torch_output = x + y

    max_diff = torch.max(torch.abs(triton_output - torch_output)).item()
    print(f"Size: {size}")
    print(f"Max difference: {max_diff}")
    print(f"Correctness: {'PASSED' if torch.allclose(triton_output, torch_output) else 'FAILED'}")
    print()

    # --- Performance Benchmark ---
    print("=" * 50)
    print("Performance Benchmark")
    print("=" * 50)

    for size in [1024, 65536, 1048576, 16777216, 67108864]:
        benchmark(size)


if __name__ == "__main__":
    main()
