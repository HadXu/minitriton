"""jit.py —— @jit 装饰器与 kernel[grid](...) 启动（对应 triton/runtime/jit.py）。

每次启动 kernel 时：
  1. 根据实参确定"特化签名"：张量 -> 指针类型，int -> i32/i64，constexpr -> 具体的值
  2. 以签名为 key 查缓存；没有就走 compiler.compile 编译一份
  3. 计算 grid，调用 CUDA driver 启动
"""

import ast
import inspect
import textwrap

from . import compiler
from . import language as tl

TORCH_DTYPES = None


def _torch_dtypes():
    global TORCH_DTYPES
    if TORCH_DTYPES is None:
        import torch

        TORCH_DTYPES = {torch.float32: tl.float32, torch.float16: tl.float16, torch.int32: tl.int32,
                        torch.int64: tl.int64, torch.bool: tl.int1}
    return TORCH_DTYPES


def _is_constexpr(annotation):
    return annotation is tl.constexpr or (isinstance(annotation, str) and "constexpr" in annotation)


def dtype_of(value):
    """运行期参数的类型。"""
    if hasattr(value, "data_ptr"):
        if not value.is_cuda:
            raise ValueError("张量必须在 GPU 上")
        return tl.pointer_type(_torch_dtypes()[value.dtype])
    if isinstance(value, bool):
        return tl.int1
    if isinstance(value, int):
        return tl.int32 if -(2**31) <= value < 2**31 else tl.int64
    if isinstance(value, float):
        return tl.float32
    raise TypeError(f"不支持的 kernel 参数类型: {type(value)}")


class JITFunction:
    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        lines, self.starting_line = inspect.getsourcelines(fn)
        self.src = textwrap.dedent("".join(lines))
        self.signature = inspect.signature(fn)
        self.arg_names = list(self.signature.parameters)
        self.constexpr_names = {n for n, p in self.signature.parameters.items() if _is_constexpr(p.annotation)}
        self.cache = {}

    def parse(self):
        fdef = ast.parse(self.src).body[0]
        assert isinstance(fdef, ast.FunctionDef)
        return fdef

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f"{self.__name__} 需要用 {self.__name__}[grid](...) 启动，或在另一个 @jit 函数里调用")

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.run(*args, grid=grid, **kwargs)

    def specialize(self, args, kwargs, num_warps):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        constexprs = {n: arguments[n] for n in self.arg_names if n in self.constexpr_names}
        signature = {n: dtype_of(arguments[n]) for n in self.arg_names if n not in self.constexpr_names}
        key = (tuple((n, str(t)) for n, t in signature.items()), tuple(constexprs.items()), num_warps)
        return arguments, signature, constexprs, key

    def warmup(self, *args, num_warps=4, **kwargs):
        """只编译不运行，返回 CompiledKernel（可以查看 .asm）。"""
        _, signature, constexprs, key = self.specialize(args, kwargs, num_warps)
        if key not in self.cache:
            self.cache[key] = compiler.compile(self, signature, constexprs, num_warps)
        return self.cache[key]

    def run(self, *args, grid, num_warps=4, **kwargs):
        arguments, signature, constexprs, key = self.specialize(args, kwargs, num_warps)
        kernel = self.cache.get(key)
        if kernel is None:
            kernel = self.cache[key] = compiler.compile(self, signature, constexprs, num_warps)
        if callable(grid):
            grid = grid(arguments)
        grid = tuple(grid) + (1,) * (3 - len(grid))
        kernel.launch(grid, [arguments[n] for n in signature])
        return kernel


def jit(fn):
    return JITFunction(fn)
