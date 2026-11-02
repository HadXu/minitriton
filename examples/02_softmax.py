"""融合 softmax：一个程序实例处理一整行，max / exp / sum / 除法全在寄存器里完成。

运行：uv run python examples/02_softmax.py
"""

import torch

import minitriton as mt
import minitriton.language as tl


@mt.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)  # 归约成标量 -> warp shuffle + shared memory
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_row_stride + cols, num / den, mask=mask)


def softmax(x):
    n_rows, n_cols = x.shape
    block = 1 << (n_cols - 1).bit_length()  # 向上取 2 的幂
    num_warps = 4 if block <= 2048 else 8
    out = torch.empty_like(x)
    kernel = softmax_kernel[(n_rows,)](out, x, x.stride(0), out.stride(0), n_cols,
                                       BLOCK_SIZE=block, num_warps=num_warps)
    return out, kernel


def bench(fn, iters=50):
    for _ in range(5):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


if __name__ == "__main__":
    x = torch.randn(4096, 1000, device="cuda")
    out, kernel = softmax(x)
    torch.testing.assert_close(out, torch.softmax(x, dim=1))
    print("结果正确 ✓")
    print("\n生成的 CUDA 中 tl.max 那一段：")
    lines = kernel.asm["cuda"].splitlines()
    start = next(i for i, l in enumerate(lines) if "tt.reduce" in l)
    print("\n".join(lines[start:start + 25]))

    gbps = lambda ms: 2 * x.numel() * x.element_size() / ms / 1e6
    t_mt, t_torch = bench(lambda: softmax(x)), bench(lambda: torch.softmax(x, dim=1))
    print(f"minitriton: {t_mt:.3f} ms ({gbps(t_mt):.0f} GB/s)   torch: {t_torch:.3f} ms ({gbps(t_torch):.0f} GB/s)")
