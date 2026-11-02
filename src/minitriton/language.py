"""minitriton.language —— 写 kernel 时使用的 DSL（对应 triton.language）。

重要：这里的大部分函数在普通 Python 中 *不能* 执行！
@jit 并不会运行你的函数，而是读取它的源码 (AST)，由 frontend.py 识别出
对这些函数的调用，再交给 semantic.py 翻译成 IR 指令。
所以这里的函数只是"符号"(stub)，真正的语义实现在 semantic.py。
"""

import functools


# --------------------------------------------------------------------------
# 数据类型
# --------------------------------------------------------------------------
class dtype:
    def __init__(self, name, kind, bits):
        self.name = name  # IR 中的名字：i1 / i32 / i64 / f16 / f32
        self.kind = kind  # 'bool' | 'int' | 'float' | 'ptr'
        self.bits = bits

    def is_bool(self):
        return self.kind == "bool"

    def is_int(self):
        return self.kind == "int"

    def is_floating(self):
        return self.kind == "float"

    def is_ptr(self):
        return self.kind == "ptr"

    def __eq__(self, other):
        return isinstance(other, dtype) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"tl.{self.name}"


class pointer_type(dtype):
    """指向 element_ty 的全局内存指针，例如 !tt.ptr<f32>（对应 C 的 float*）。"""

    def __init__(self, element_ty):
        super().__init__(f"!tt.ptr<{element_ty.name}>", "ptr", 64)
        self.element_ty = element_ty


int1 = dtype("i1", "bool", 1)
int32 = dtype("i32", "int", 32)
int64 = dtype("i64", "int", 64)
float16 = dtype("f16", "float", 16)
float32 = dtype("f32", "float", 32)


class constexpr:
    """类型注解标记：`BLOCK: tl.constexpr` 表示该参数是编译期常量。

    编译期常量会直接"烧"进生成的代码里（例如数组大小），
    不同的取值会触发重新编译（JIT 特化）。
    """


# --------------------------------------------------------------------------
# 内建函数（stub）
# --------------------------------------------------------------------------
def _builtin(fn):
    @functools.wraps(fn)
    def stub(*args, **kwargs):
        raise RuntimeError(f"tl.{fn.__name__} 只能在 @minitriton.jit 修饰的 kernel 中使用")

    stub.__mt_builtin__ = fn.__name__
    return stub


@_builtin
def program_id(axis): ...
@_builtin
def num_programs(axis): ...
@_builtin
def arange(start, end): ...
@_builtin
def load(pointer, mask=None, other=None): ...
@_builtin
def store(pointer, value, mask=None): ...
@_builtin
def zeros(shape, dtype): ...
@_builtin
def full(shape, value, dtype): ...
@_builtin
def where(condition, x, y): ...
@_builtin
def maximum(x, y): ...
@_builtin
def minimum(x, y): ...
@_builtin
def exp(x): ...
@_builtin
def log(x): ...
@_builtin
def sqrt(x): ...
@_builtin
def rsqrt(x): ...
@_builtin
def sin(x): ...
@_builtin
def cos(x): ...
@_builtin
def floor(x): ...
@_builtin
def abs(x): ...
@_builtin
def sum(input, axis=None): ...
@_builtin
def max(input, axis=None): ...
@_builtin
def min(input, axis=None): ...
@_builtin
def dot(a, b, acc=None): ...
@_builtin
def static_range(*args):
    """编译期展开的循环：for i in tl.static_range(4) 会把循环体复制 4 份。"""


def cdiv(a, b):
    """向上取整除法。在 host 端和 kernel 内都能用。"""
    return (a + b - 1) // b


# 参数全是编译期常量时，前端直接在 Python 里执行它；否则翻译成 IR
cdiv.__mt_builtin__ = "cdiv"
cdiv.__mt_constexpr_ok__ = True
