"""semantic.py —— 语义层（对应 triton/language/semantic.py）。

前端遇到 `x + y`、`tl.load(...)` 这类表达式时调用这里的函数，负责：
  1. 类型提升：i32 + f32 -> f32，f16 + f32 -> f32 ...
  2. 广播：标量 -> 张量 (tt.splat)，[M,1] -> [M,N] (tt.expand_dims + tt.broadcast)
  3. 生成对应的 IR 指令

经过这一层之后，IR 里的每条逐元素指令的操作数都具有完全相同的类型和形状，
后端就不需要再关心广播和类型提升了。
"""

from . import language as tl
from .ir import TensorType


class tensor:
    """前端里的"张量"：包装一个 IR Value（handle）。Python 的 int/float/bool 则是编译期常量。"""

    def __init__(self, handle):
        self.handle = handle

    @property
    def type(self):
        return self.handle.type

    @property
    def dtype(self):
        return self.type.dtype

    @property
    def shape(self):
        return self.type.shape

    def __repr__(self):
        return f"tensor({self.type})"


def _op(builder, name, operands, dtype, shape=None, attrs=None):
    """创建一条单结果指令。默认结果形状与第一个操作数相同。"""
    shape = operands[0].shape if shape is None else shape
    op = builder.create(name, [o.handle for o in operands], [TensorType(dtype, shape)], attrs)
    return tensor(op.result)


def _check_const(x, what):
    if isinstance(x, tensor):
        raise TypeError(f"{what} 必须是编译期常量 (tl.constexpr)，但得到的是 {x}")
    return x


# --------------------------------------------------------------------------
# 常量与类型转换
# --------------------------------------------------------------------------
def _py_dtype(x):
    if isinstance(x, bool):
        return tl.int1
    if isinstance(x, int):
        return tl.int32 if -(2**31) <= x < 2**31 else tl.int64
    if isinstance(x, float):
        return tl.float32
    raise TypeError(f"无法把 {x!r} 转换成张量")


def constant(builder, value, dtype):
    value = bool(value) if dtype.is_bool() else int(value) if dtype.is_int() else float(value)
    op = builder.create("arith.constant", [], [TensorType(dtype)], {"value": value})
    return tensor(op.result)


def to_tensor(builder, x, dtype=None):
    if isinstance(x, tensor):
        return x if dtype is None else cast(builder, x, dtype)
    return constant(builder, x, dtype or _py_dtype(x))


def cast(builder, x, dtype):
    x = to_tensor(builder, x)
    src = x.dtype
    if src == dtype:
        return x
    if src.is_ptr() or dtype.is_ptr():
        raise TypeError(f"不支持指针类型转换: {src} -> {dtype}")
    if dtype.is_bool():
        return binary(builder, "!=", x, 0)
    if src.is_bool():
        name = "arith.uitofp" if dtype.is_floating() else "arith.extui"
    elif src.is_floating() and dtype.is_floating():
        name = "arith.extf" if dtype.bits > src.bits else "arith.truncf"
    elif src.is_floating():
        name = "arith.fptosi"
    elif dtype.is_floating():
        name = "arith.sitofp"
    else:
        name = "arith.extsi" if dtype.bits > src.bits else "arith.trunci"
    return _op(builder, name, [x], dtype)


def _result_dtype(x, y):
    """二元运算的类型提升规则。Python 常量是"弱类型"，会跟随张量的类型。"""
    if isinstance(x, tensor) and isinstance(y, tensor):
        dx, dy = x.dtype, y.dtype
        if dx.is_floating() or dy.is_floating():
            return max((d for d in (dx, dy) if d.is_floating()), key=lambda d: d.bits)
        return max(dx, dy, key=lambda d: d.bits)
    t, p = (x, y) if isinstance(x, tensor) else (y, x)
    dt = t.dtype
    if isinstance(p, float) and not dt.is_floating():
        return tl.float32
    if dt.is_bool() and not isinstance(p, bool):
        return tl.int32
    if dt == tl.int32 and _py_dtype(p) == tl.int64:
        return tl.int64
    return dt


# --------------------------------------------------------------------------
# 广播
# --------------------------------------------------------------------------
def broadcast_shapes(*shapes):
    rank = max(len(s) for s in shapes)
    padded = [(1,) * (rank - len(s)) + tuple(s) for s in shapes]
    out = []
    for dims in zip(*padded):
        non1 = {d for d in dims if d != 1}
        if len(non1) > 1:
            raise ValueError(f"形状无法广播: {shapes}")
        out.append(non1.pop() if non1 else 1)
    return tuple(out)


def splat(builder, x, shape):
    return _op(builder, "tt.splat", [x], x.dtype, shape)


def expand_dims(builder, x, axis):
    x = to_tensor(builder, x)
    if x.type.is_scalar:
        return splat(builder, x, (1,))
    if axis < 0:
        axis += len(x.shape) + 1
    shape = list(x.shape)
    shape.insert(axis, 1)
    return _op(builder, "tt.expand_dims", [x], x.dtype, tuple(shape), {"axis": axis})


def broadcast_to(builder, x, shape):
    x = to_tensor(builder, x)
    shape = tuple(shape)
    if x.shape == shape:
        return x
    if x.type.is_scalar:
        return splat(builder, x, shape)
    while len(x.shape) < len(shape):
        x = expand_dims(builder, x, 0)
    for s, d in zip(x.shape, shape):
        if s != d and s != 1:
            raise ValueError(f"无法把形状 {x.shape} 广播到 {shape}")
    if x.shape == shape:
        return x
    return _op(builder, "tt.broadcast", [x], x.dtype, shape)


def broadcast_pair(builder, x, y):
    if x.shape == y.shape:
        return x, y
    shape = broadcast_shapes(x.shape, y.shape)
    return broadcast_to(builder, x, shape), broadcast_to(builder, y, shape)


# --------------------------------------------------------------------------
# 算术
# --------------------------------------------------------------------------
FLOAT_OPS = {"+": "arith.addf", "-": "arith.subf", "*": "arith.mulf", "/": "arith.divf", "%": "arith.remf"}
INT_OPS = {
    "+": "arith.addi", "-": "arith.subi", "*": "arith.muli", "//": "arith.divsi", "%": "arith.remsi",
    "&": "arith.andi", "|": "arith.ori", "^": "arith.xori", "<<": "arith.shli", ">>": "arith.shrsi",
}
CMPF = {"<": "olt", "<=": "ole", ">": "ogt", ">=": "oge", "==": "oeq", "!=": "une"}
CMPI = {"<": "slt", "<=": "sle", ">": "sgt", ">=": "sge", "==": "eq", "!=": "ne"}
BITWISE = {"&", "|", "^"}


def binary(builder, op, x, y):
    if any(isinstance(v, tensor) and v.dtype.is_ptr() for v in (x, y)):
        return _ptr_arith(builder, op, x, y)
    dt = _result_dtype(x, y)
    if dt.is_bool() and op not in BITWISE and op not in CMPI:
        dt = tl.int32
    if op == "/" and not dt.is_floating():
        dt = tl.float32  # Python 语义：整数 / 整数 = 浮点
    if dt.is_floating() and op in INT_OPS and op not in FLOAT_OPS and op != "//":
        raise TypeError(f"浮点数不支持运算符 {op}")
    x, y = broadcast_pair(builder, to_tensor(builder, x, dt), to_tensor(builder, y, dt))
    if op in CMPI:
        if dt.is_floating():
            return _op(builder, "arith.cmpf", [x, y], tl.int1, attrs={"predicate": CMPF[op]})
        return _op(builder, "arith.cmpi", [x, y], tl.int1, attrs={"predicate": CMPI[op]})
    if op == "//" and dt.is_floating():
        return math_unary(builder, "floor", binary(builder, "/", x, y))
    name = FLOAT_OPS[op] if dt.is_floating() else INT_OPS[op]
    return _op(builder, name, [x, y], dt)


def _ptr_arith(builder, op, x, y):
    """指针 + 整数偏移 -> tt.addptr（偏移以元素为单位，和 C 指针运算一样）。"""
    if op not in ("+", "-"):
        raise TypeError(f"指针不支持运算符 {op}")
    ptr_first = isinstance(x, tensor) and x.dtype.is_ptr()
    ptr, off = (x, y) if ptr_first else (y, x)
    if op == "-" and not ptr_first:
        raise TypeError("不支持 整数 - 指针")
    off = to_tensor(builder, off)
    if off.dtype.is_bool():
        off = cast(builder, off, tl.int32)
    if not off.dtype.is_int():
        raise TypeError(f"指针偏移必须是整数，得到 {off.dtype}")
    if op == "-":
        off = unary(builder, "-", off)
    ptr, off = broadcast_pair(builder, ptr, off)
    return _op(builder, "tt.addptr", [ptr, off], ptr.dtype)


def unary(builder, op, x):
    x = to_tensor(builder, x)
    if op == "+":
        return x
    if op == "-":
        if x.dtype.is_floating():
            return _op(builder, "arith.negf", [x], x.dtype)
        return binary(builder, "-", 0, x)
    if op in ("~", "not"):
        return binary(builder, "^", x, True if x.dtype.is_bool() else -1)
    raise TypeError(f"不支持的一元运算 {op}")


# --------------------------------------------------------------------------
# tl.xxx 内建函数
# --------------------------------------------------------------------------
def program_id(builder, axis):
    _check_const(axis, "axis")
    return tensor(builder.create("tt.get_program_id", [], [TensorType(tl.int32)], {"axis": axis}).result)


def num_programs(builder, axis):
    _check_const(axis, "axis")
    return tensor(builder.create("tt.get_num_programs", [], [TensorType(tl.int32)], {"axis": axis}).result)


def arange(builder, start, end):
    _check_const(start, "arange 的 start")
    _check_const(end, "arange 的 end")
    ty = TensorType(tl.int32, (end - start,))
    return tensor(builder.create("tt.make_range", [], [ty], {"start": start, "end": end}).result)


def _shape(shape):
    _check_const(shape, "shape")
    return (shape,) if isinstance(shape, int) else tuple(shape)


def full(builder, shape, value, dtype):
    return broadcast_to(builder, constant(builder, value, dtype), _shape(shape))


def zeros(builder, shape, dtype):
    return full(builder, shape, 0, dtype)


def load(builder, pointer, mask=None, other=None):
    ptr = to_tensor(builder, pointer)
    if not ptr.dtype.is_ptr():
        raise TypeError(f"tl.load 需要指针，得到 {ptr.dtype}")
    elt = ptr.dtype.element_ty
    if mask is None:
        return _op(builder, "tt.load", [ptr], elt)
    mask = cast(builder, mask, tl.int1)
    ptr, mask = broadcast_pair(builder, ptr, mask)
    # 被 mask 掉的位置取 other（默认 0），这样结果是确定的
    other = broadcast_to(builder, to_tensor(builder, 0 if other is None else other, elt), ptr.shape)
    return _op(builder, "tt.load", [ptr, mask, other], elt)


def store(builder, pointer, value, mask=None):
    ptr = to_tensor(builder, pointer)
    if not ptr.dtype.is_ptr():
        raise TypeError(f"tl.store 需要指针，得到 {ptr.dtype}")
    value = to_tensor(builder, value, ptr.dtype.element_ty)
    operands = [ptr, value]
    if mask is not None:
        operands.append(cast(builder, mask, tl.int1))
    shape = broadcast_shapes(*(o.shape for o in operands))
    operands = [broadcast_to(builder, o, shape) for o in operands]
    builder.create("tt.store", [o.handle for o in operands], [])


def _promote_pair(builder, x, y):
    if not isinstance(x, tensor) and not isinstance(y, tensor):
        x = to_tensor(builder, x)
    dt = _result_dtype(x, y)
    return to_tensor(builder, x, dt), to_tensor(builder, y, dt), dt


def where(builder, condition, x, y):
    cond = cast(builder, condition, tl.int1)
    x, y, dt = _promote_pair(builder, x, y)
    shape = broadcast_shapes(cond.shape, x.shape, y.shape)
    ops = [broadcast_to(builder, v, shape) for v in (cond, x, y)]
    return _op(builder, "arith.select", ops, dt, shape)


def minmax(builder, kind, x, y):
    x, y, dt = _promote_pair(builder, x, y)
    cdt = tl.float32 if dt == tl.float16 else tl.int32 if dt.is_bool() else dt
    x, y = broadcast_pair(builder, cast(builder, x, cdt), cast(builder, y, cdt))
    if cdt.is_floating():
        name = "arith.maximumf" if kind == "max" else "arith.minimumf"
    else:
        name = "arith.maxsi" if kind == "max" else "arith.minsi"
    return cast(builder, _op(builder, name, [x, y], cdt), dt)


def math_unary(builder, name, x):
    """数学函数统一在 f32 上计算（f16 先升精度再降回来）。"""
    x = to_tensor(builder, x)
    if name == "abs" and x.dtype.is_int():
        return _op(builder, "math.absi", [x], x.dtype)
    orig = x.dtype if x.dtype.is_floating() else tl.float32
    x = cast(builder, x, tl.float32)
    r = _op(builder, "math.absf" if name == "abs" else f"math.{name}", [x], tl.float32)
    return cast(builder, r, orig)


def reduce(builder, input, axis, kind):
    x = to_tensor(builder, input)
    if x.type.is_scalar:
        raise TypeError("不能对标量做归约")
    orig = x.dtype
    cdt = tl.float32 if orig == tl.float16 else tl.int32 if orig.is_bool() else orig
    x = cast(builder, x, cdt)
    if axis is None:
        shape = ()  # 对所有元素归约，得到标量
    else:
        _check_const(axis, "axis")
        axis = axis + len(x.shape) if axis < 0 else axis
        shape = x.shape[:axis] + x.shape[axis + 1:]
    r = _op(builder, "tt.reduce", [x], cdt, shape, {"axis": axis, "kind": kind})
    return cast(builder, r, orig) if orig == tl.float16 else r


def dot(builder, a, b, acc=None):
    if len(a.shape) != 2 or len(b.shape) != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"tl.dot 形状不匹配: {a.shape} @ {b.shape}")
    if not (a.dtype.is_floating() and b.dtype.is_floating()):
        raise TypeError("tl.dot 目前只支持浮点输入")
    shape = (a.shape[0], b.shape[1])
    operands = [a, b]
    if acc is not None:
        operands.append(broadcast_to(builder, cast(builder, acc, tl.float32), shape))
    return _op(builder, "tt.dot", operands, tl.float32, shape)


def cdiv(builder, x, y):
    return binary(builder, "//", binary(builder, "-", binary(builder, "+", x, y), 1), y)


def _reducer(kind):
    return lambda builder, input, axis=None: reduce(builder, input, axis, kind)


def _math(name):
    return lambda builder, x: math_unary(builder, name, x)


# 前端用 tl 函数的名字在这里查找实现
BUILTINS = {
    "program_id": program_id, "num_programs": num_programs, "arange": arange,
    "load": load, "store": store, "zeros": zeros, "full": full, "where": where,
    "dot": dot, "cdiv": cdiv,
    "maximum": lambda builder, x, y: minmax(builder, "max", x, y),
    "minimum": lambda builder, x, y: minmax(builder, "min", x, y),
    "sum": _reducer("sum"), "max": _reducer("max"), "min": _reducer("min"),
    **{n: _math(n) for n in ("exp", "log", "sqrt", "rsqrt", "sin", "cos", "floor", "abs")},
}
