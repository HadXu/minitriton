"""ir.py —— 一个极简的、类 MLIR 的 SSA 中间表示（对应 Triton 的 TTIR）。

核心概念：
  Value      SSA 值，只被定义一次。类型是 TensorType（标量就是 shape=() 的张量）。
  Operation  一条指令：名字 + 操作数 + 结果 + 属性 + 可选的嵌套区域(region)。
  Block      指令列表，可以有参数（例如 for 循环的归纳变量和循环携带变量）。
  Function   kernel 本身：参数 + 函数体 Block。

注意 Triton IR 的关键特点：它是"块级"(block-level) 的 ——
一个值可以是 tensor<1024xf32>，代表一整个程序实例(program)处理的数据块，
而不是单个线程的标量。把块分配给线程是后端 (codegen.py) 的工作。
"""

from contextlib import contextmanager


class TensorType:
    def __init__(self, dtype, shape=()):
        self.dtype = dtype
        self.shape = tuple(shape)

    @property
    def is_scalar(self):
        return self.shape == ()

    @property
    def numel(self):
        n = 1
        for s in self.shape:
            n *= s
        return n

    def __eq__(self, other):
        return isinstance(other, TensorType) and (self.dtype, self.shape) == (other.dtype, other.shape)

    def __hash__(self):
        return hash((self.dtype, self.shape))

    def __str__(self):
        if self.is_scalar:
            return str(self.dtype)
        return "tensor<" + "x".join(map(str, self.shape)) + f"x{self.dtype}>"


class Value:
    def __init__(self, type, owner=None):
        self.type = type
        self.owner = owner  # 定义它的 Operation；Block 参数为 None


class Block:
    def __init__(self, args=None):
        self.args = list(args or [])
        self.ops = []


class Operation:
    def __init__(self, name, operands, result_types, attrs=None, regions=None):
        self.name = name
        self.operands = list(operands)
        self.results = [Value(t, self) for t in result_types]
        self.attrs = dict(attrs or {})
        self.regions = list(regions or [])

    @property
    def result(self):
        assert len(self.results) == 1
        return self.results[0]


class Function:
    def __init__(self, name, arg_types, arg_names):
        self.name = name
        self.args = [Value(t) for t in arg_types]
        self.arg_names = list(arg_names)
        self.body = Block()


class Builder:
    """在当前插入点 (self.block) 末尾追加指令。"""

    def __init__(self):
        self.block = None

    def create(self, name, operands, result_types, attrs=None, regions=None):
        op = Operation(name, operands, result_types, attrs, regions)
        self.block.ops.append(op)
        return op

    @contextmanager
    def insertion_point(self, block):
        saved, self.block = self.block, block
        try:
            yield
        finally:
            self.block = saved


def walk(block):
    """按顺序递归遍历所有指令（包括嵌套区域里的）。"""
    for op in block.ops:
        yield op
        for region in op.regions:
            yield from walk(region)


def number_values(func):
    """给每个 Value 一个名字：函数参数用参数名，其他值用 0,1,2...（打印和 codegen 共用）。"""
    names = dict(zip(func.args, func.arg_names))
    counter = 0

    def visit(block):
        nonlocal counter
        for v in block.args:
            names[v] = str(counter)
            counter += 1
        for op in block.ops:
            for r in op.results:
                names[r] = str(counter)
                counter += 1
            for region in op.regions:
                visit(region)

    visit(func.body)
    return names


# --------------------------------------------------------------------------
# 打印成 MLIR 风格的文本
# --------------------------------------------------------------------------
def format_op(op, names):
    """返回一条指令的单行文本（对带区域的指令只返回头部）。"""
    n = lambda v: "%" + names[v]
    res = ", ".join(n(r) for r in op.results)
    lhs = f"{res} = " if res else ""
    if op.name == "scf.for":
        lb, ub, step, *inits = op.operands
        body = op.regions[0]
        iv, *carried = body.args
        s = f"{lhs}scf.for {n(iv)} = {n(lb)} to {n(ub)} step {n(step)}"
        if carried:
            s += " iter_args(" + ", ".join(f"{n(a)} = {n(i)}" for a, i in zip(carried, inits)) + ")"
            s += " -> (" + ", ".join(str(r.type) for r in op.results) + ")"
        return s + " {"
    if op.name == "scf.if":
        s = f"{lhs}scf.if {n(op.operands[0])}"
        if op.results:
            s += " -> (" + ", ".join(str(r.type) for r in op.results) + ")"
        return s + " {"
    operands = ", ".join(n(o) for o in op.operands)
    attrs = ""
    if op.attrs:
        attrs = " {" + ", ".join(f"{k} = {v!r}" for k, v in op.attrs.items()) + "}"
    types = op.results or op.operands
    ty = " : " + ", ".join(str(v.type) for v in types) if types else ""
    return f"{lhs}{op.name}{' ' if operands else ''}{operands}{attrs}{ty}"


def print_function(func):
    names = number_values(func)
    lines = []

    def visit(block, indent):
        pad = "  " * indent
        for op in block.ops:
            lines.append(pad + format_op(op, names))
            if op.name == "scf.for":
                visit(op.regions[0], indent + 1)
                lines.append(pad + "}")
            elif op.name == "scf.if":
                visit(op.regions[0], indent + 1)
                lines.append(pad + "} else {")
                visit(op.regions[1], indent + 1)
                lines.append(pad + "}")

    params = ", ".join(f"%{n}: {a.type}" for n, a in zip(func.arg_names, func.args))
    lines.append(f"tt.func @{func.name}({params}) {{")
    visit(func.body, 1)
    lines.append("}")
    return "\n".join(lines)
