import pytest
import torch

import minitriton as mt
import minitriton.language as tl


@pytest.mark.parametrize("n", [1, 100, 1024, 5000])
def test_add_scalar_and_mul(n):
    @mt.jit
    def kernel(x_ptr, out_ptr, n, alpha, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x * alpha + 1, mask=mask)

    x = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    kernel[(mt.cdiv(n, 256),)](x, out, n, 2.5, BLOCK=256)
    torch.testing.assert_close(out, x * 2.5 + 1)


def test_fp16_and_math():
    @mt.jit
    def kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=1.0)
        y = tl.exp(x) + tl.sqrt(tl.abs(x)) - tl.maximum(x, 0.0)
        tl.store(out_ptr + offs, y, mask=mask)

    x = torch.randn(300, device="cuda", dtype=torch.float16)
    out = torch.empty_like(x)
    kernel[(1,)](x, out, 300, BLOCK=512)
    xf = x.float()
    ref = (torch.exp(xf) + torch.sqrt(xf.abs()) - xf.clamp(min=0)).half()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_int_ops_and_where():
    @mt.jit
    def kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)
        y = tl.where(x % 2 == 0, x // 2, -x) + (x << 1) - tl.minimum(x, 3)
        tl.store(out_ptr + offs, y)

    x = torch.arange(0, 128, device="cuda", dtype=torch.int32)
    out = torch.empty_like(x)
    kernel[(1,)](x, out, BLOCK=128)
    ref = torch.where(x % 2 == 0, x // 2, -x) + (x << 1) - x.clamp(max=3)
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("axis", [0, 1, None])
@pytest.mark.parametrize("kind", ["sum", "max", "min"])
def test_reduce_2d(axis, kind):
    @mt.jit
    def kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, AXIS: tl.constexpr, KIND: tl.constexpr):
        x = tl.load(x_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :])
        if KIND == "sum":
            r = tl.sum(x, axis=AXIS)
        elif KIND == "max":
            r = tl.max(x, axis=AXIS)
        else:
            r = tl.min(x, axis=AXIS)
        if AXIS == 0:
            tl.store(out_ptr + tl.arange(0, N), r)
        elif AXIS == 1:
            tl.store(out_ptr + tl.arange(0, M), r)
        else:
            tl.store(out_ptr, r)

    M, N = 16, 64
    x = torch.randn(M, N, device="cuda")
    ref = {"sum": x.sum, "max": x.amax, "min": x.amin}[kind](dim=axis) if axis is not None else \
        {"sum": x.sum, "max": x.max, "min": x.min}[kind]()
    out = torch.empty(ref.numel() or 1, device="cuda")
    kernel[(1,)](x, out, M=M, N=N, AXIS=axis, KIND=kind)
    torch.testing.assert_close(out.view_as(ref), ref, atol=1e-4, rtol=1e-4)


def test_broadcast_outer_product():
    @mt.jit
    def kernel(a_ptr, b_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
        a = tl.load(a_ptr + tl.arange(0, M))
        b = tl.load(b_ptr + tl.arange(0, N))
        out = a[:, None] * b[None, :]
        tl.store(out_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], out)

    a, b = torch.randn(32, device="cuda"), torch.randn(48, device="cuda")
    out = torch.empty(32, 48, device="cuda")
    kernel[(1,)](a, b, out, M=32, N=48, num_warps=2)
    torch.testing.assert_close(out, a[:, None] * b[None, :])


def test_loop_with_carried_values_and_swap():
    @mt.jit
    def kernel(out_ptr, n_iter, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        a = offs * 0 + 0
        b = offs * 0 + 1
        total = 0
        for i in range(n_iter):  # fibonacci：a, b = b, a + b
            a, b = b, a + b
            total += i
        tl.store(out_ptr + offs, a + total)

    out = torch.empty(64, device="cuda", dtype=torch.int32)
    kernel[(1,)](out, 10, BLOCK=64)
    assert (out == 55 + 45).all()


def test_dynamic_if_and_static_range():
    @mt.jit
    def kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)
        if pid % 2 == 0:
            y = x * 2
        else:
            y = x + 100
        for _ in tl.static_range(3):
            y = y + 1
        tl.store(out_ptr + offs, y)

    x = torch.randn(4, 128, device="cuda")
    out = torch.empty_like(x)
    kernel[(4,)](x, out, BLOCK=128)
    ref = torch.where(torch.arange(4, device="cuda")[:, None] % 2 == 0, x * 2, x + 100) + 3
    torch.testing.assert_close(out, ref)


@mt.jit
def leaky_relu(x, slope):
    return tl.where(x > 0, x, x * slope)


def test_inline_jit_function_and_scalar_store():
    @mt.jit
    def kernel(x_ptr, out_ptr, sum_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        y = leaky_relu(tl.load(x_ptr + offs), 0.1)
        tl.store(out_ptr + offs, y)
        tl.store(sum_ptr, tl.sum(y))

    x = torch.randn(256, device="cuda")
    out, s = torch.empty_like(x), torch.empty(1, device="cuda")
    kernel[(1,)](x, out, s, BLOCK=256)
    ref = torch.nn.functional.leaky_relu(x, 0.1)
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(s[0], ref.sum(), atol=1e-4, rtol=1e-4)


def test_softmax_and_matmul_examples():
    import importlib.util
    import pathlib

    def load(name):
        path = pathlib.Path(__file__).parent.parent / "examples" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    x = torch.randn(37, 781, device="cuda")
    torch.testing.assert_close(load("02_softmax").softmax(x)[0], torch.softmax(x, 1))
    a = torch.randn(130, 70, device="cuda", dtype=torch.float16)
    b = torch.randn(70, 99, device="cuda", dtype=torch.float16)
    c, _ = load("03_matmul").matmul(a, b)
    torch.testing.assert_close(c.float(), a.float() @ b.float(), atol=1e-2, rtol=1e-2)


def test_compile_error_points_to_source_line():
    @mt.jit
    def kernel(x_ptr, BLOCK: tl.constexpr):
        x = tl.load(x_ptr + tl.arange(0, BLOCK))
        tl.store(x_ptr + tl.arange(0, BLOCK), x[1:])

    with pytest.raises(Exception, match=r"x\[1:\]"):
        kernel[(1,)](torch.empty(16, device="cuda"), BLOCK=16)
