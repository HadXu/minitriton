"""分块矩阵乘法：二维 program 网格 + for 循环 + tl.dot。

运行：uv run python examples/03_matmul.py
"""

import torch

import minitriton as mt
import minitriton.language as tl


@mt.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    # [BLOCK_M, 1] + [1, BLOCK_K] -> [BLOCK_M, BLOCK_K] 的指针矩阵
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):  # acc / a_ptrs / b_ptrs 自动成为循环携带变量
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(a, b, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = (mt.cdiv(M, BLOCK_M), mt.cdiv(N, BLOCK_N))
    kernel = matmul_kernel[grid](a, b, c, M, N, K,
                                 a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                                 BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4)
    return c, kernel


if __name__ == "__main__":
    a = torch.randn(1000, 700, device="cuda", dtype=torch.float16)
    b = torch.randn(700, 900, device="cuda", dtype=torch.float16)
    c, kernel = matmul(a, b)
    torch.testing.assert_close(c.float(), a.float() @ b.float(), atol=1e-2, rtol=1e-2)
    print("结果正确 ✓\n")
    print("优化后的 TTIR（注意 LICM 把循环里不变的 mask 计算挪到了循环外）：")
    print(kernel.asm["ttir_opt"])
    print("\nptxas 报告：")
    print(kernel.asm["ptxas"])

    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for name, fn in [("minitriton", lambda: matmul(a, b)), ("torch (cuBLAS)", lambda: a @ b)]:
        fn()
        start.record()
        for _ in range(10):
            fn()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / 10
        print(f"{name:>15}: {ms:.3f} ms, {2 * 4096**3 / ms / 1e9:.1f} TFLOPS")
