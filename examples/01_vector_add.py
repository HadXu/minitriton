"""第一个例子：向量加法，并打印编译流水线的每一个阶段。

运行：uv run python examples/01_vector_add.py
"""

import torch

import minitriton as mt
import minitriton.language as tl


@mt.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)  # 第几个程序实例 (= CUDA 的 blockIdx.x)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 这个实例负责的一整块下标
    mask = offsets < n_elements  # 处理最后一块越界的部分
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (mt.cdiv(n, meta["BLOCK_SIZE"]),)
    kernel = add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024, num_warps=4)
    return out, kernel


if __name__ == "__main__":
    x = torch.randn(98432, device="cuda")
    y = torch.randn(98432, device="cuda")
    out, kernel = add(x, y)
    torch.testing.assert_close(out, x + y)
    print("结果正确 ✓\n")

    for stage in ["python", "ttir", "ttir_opt", "cuda", "ptxas"]:
        print(f"{'=' * 30} {stage} {'=' * 30}")
        print(kernel.asm[stage])
    print(f"{'=' * 30} ptx (前 60 行) {'=' * 30}")
    print("\n".join(kernel.asm["ptx"].splitlines()[:60]))
