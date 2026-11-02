"""codegen.py —— 把块级 IR 降低 (lower) 成 CUDA C（对应 Triton 的 TTGIR -> LLVM IR 这一段）。

这是整个编译器里最关键的一步：IR 描述的是"整个程序实例处理一个数据块"，
而 GPU 上真正执行的是 num_warps * 32 个线程。我们需要决定：
  **块里的每个元素由哪个线程、存在哪个寄存器里？** —— 这就是 Triton 里的 "layout"。

本项目使用一种最简单的 layout（Triton 的 BlockedLayout 的简化版）：
  把张量按行优先展平成一维，线性下标为 idx 的元素由线程 (idx % T) 持有，
  存在该线程的寄存器数组的第 (idx / T) 个位置，T = 线程总数。

      tensor<8xf32>, T = 4 个线程:
        元素:   0  1  2  3  4  5  6  7
        线程:   0  1  2  3  0  1  2  3      -> 线程 t 的寄存器 v[j] 存元素 t + j*T

这样相邻线程访问相邻地址，全局内存访问是合并 (coalesced) 的。

有了 layout，大部分指令的翻译就很直接了：
  - 逐元素运算 (add/mul/exp/...)：每个线程对自己的寄存器数组做循环，完全不需要通信
  - expand_dims：线性顺序不变，什么都不用做
  - broadcast / reduce / dot：元素需要在线程之间交换 -> 经过 shared memory
    (Triton 里叫 convert_layout，也是用 shared memory 实现的)
"""

from contextlib import contextmanager

from . import language as tl
from .ir import number_values, format_op

CTYPES = {"i1": "bool", "i32": "int", "i64": "long long", "f16": "__half", "f32": "float"}

C_RESERVED = {
    "tid", "idx", "j", "k", "m", "n", "w", "off", "acc", "base", "smem_a", "smem_b", "mt_smem",
    "int", "float", "bool", "char", "long", "short", "double", "void", "const", "for", "if", "else",
    "while", "do", "return", "switch", "case", "default", "break", "continue", "struct", "union",
    "auto", "register", "static", "extern", "signed", "unsigned", "sizeof", "goto", "volatile",
    "inline", "typedef", "enum", "new", "delete", "class", "template", "this", "true", "false",
}

PRELUDE = """\
#include <cuda_fp16.h>
#include <math.h>
"""


def ctype(dt):
    if dt.is_ptr():
        return ctype(dt.element_ty) + "*"
    return CTYPES[dt.name]


def sizeof(dt):
    return 8 if dt.is_ptr() else max(dt.bits // 8, 1)


def literal(value, dt):
    if dt.is_bool():
        return "true" if value else "false"
    if dt.is_int():
        return f"{value}LL" if dt == tl.int64 else str(value)
    if value != value:
        s = "NAN"
    elif value in (float("inf"), float("-inf")):
        s = "INFINITY" if value > 0 else "-INFINITY"
    else:
        s = repr(float(value)) + "f"
    return f"__float2half({s})" if dt == tl.float16 else s


def to_float(x, dt):
    return f"__half2float({x})" if dt == tl.float16 else f"(float)({x})"


def cast_expr(x, src, dst):
    if src == tl.float16:
        x, src = f"__half2float({x})", tl.float32
    if dst == tl.float16:
        return f"__float2half((float)({x}))"
    return f"({ctype(dst)})({x})"


def strides(shape):
    s, out = 1, []
    for d in reversed(shape):
        out.append(s)
        s *= d
    return out[::-1]


def coord_expr(dim, shape):
    """线性下标 idx 在第 dim 维上的坐标的 C 表达式。"""
    st = strides(shape)[dim]
    e = "idx" if st == 1 else f"(idx / {st})"
    return e if dim == 0 else f"({e} % {shape[dim]})"


def _bin(sym):
    return lambda a, op: f"{a[0]} {sym} {a[1]}"


CMP_SYMBOLS = {"olt": "<", "ole": "<=", "ogt": ">", "oge": ">=", "oeq": "==", "une": "!=",
               "slt": "<", "sle": "<=", "sgt": ">", "sge": ">=", "eq": "==", "ne": "!="}

# 逐元素指令 -> C 表达式。a 是操作数在 C 里的写法（标量是 v3，张量是 v3[j]）
ELEMENTWISE = {
    "arith.addf": _bin("+"), "arith.addi": _bin("+"), "arith.subf": _bin("-"), "arith.subi": _bin("-"),
    "arith.mulf": _bin("*"), "arith.muli": _bin("*"), "arith.divf": _bin("/"), "arith.divsi": _bin("/"),
    "arith.remsi": _bin("%"), "arith.andi": _bin("&"), "arith.ori": _bin("|"), "arith.xori": _bin("^"),
    "arith.shli": _bin("<<"), "arith.shrsi": _bin(">>"), "tt.addptr": _bin("+"),
    "arith.remf": lambda a, op: f"fmodf({a[0]}, {a[1]})",
    "arith.cmpf": lambda a, op: f"{a[0]} {CMP_SYMBOLS[op.attrs['predicate']]} {a[1]}",
    "arith.cmpi": lambda a, op: f"{a[0]} {CMP_SYMBOLS[op.attrs['predicate']]} {a[1]}",
    "arith.select": lambda a, op: f"{a[0]} ? {a[1]} : {a[2]}",
    "arith.negf": lambda a, op: f"-{a[0]}",
    "arith.maximumf": lambda a, op: f"fmaxf({a[0]}, {a[1]})",
    "arith.minimumf": lambda a, op: f"fminf({a[0]}, {a[1]})",
    "arith.maxsi": lambda a, op: f"max({a[0]}, {a[1]})",
    "arith.minsi": lambda a, op: f"min({a[0]}, {a[1]})",
    "math.exp": lambda a, op: f"expf({a[0]})", "math.log": lambda a, op: f"logf({a[0]})",
    "math.sqrt": lambda a, op: f"sqrtf({a[0]})", "math.rsqrt": lambda a, op: f"rsqrtf({a[0]})",
    "math.sin": lambda a, op: f"sinf({a[0]})", "math.cos": lambda a, op: f"cosf({a[0]})",
    "math.floor": lambda a, op: f"floorf({a[0]})", "math.absf": lambda a, op: f"fabsf({a[0]})",
    "math.absi": lambda a, op: f"({a[0]} < 0 ? -{a[0]} : {a[0]})",
    "tt.splat": lambda a, op: a[0],
    "tt.make_range": lambda a, op: f"{op.attrs['start']} + idx",
}
CAST_OPS = {"arith.uitofp", "arith.extui", "arith.extf", "arith.truncf", "arith.fptosi",
            "arith.sitofp", "arith.extsi", "arith.trunci"}
for _name in CAST_OPS:
    ELEMENTWISE[_name] = lambda a, op: cast_expr(a[0], op.operands[0].type.dtype, op.result.type.dtype)


class CudaCodegen:
    def __init__(self, func, num_warps):
        self.func = func
        self.num_warps = num_warps
        self.T = num_warps * 32  # 一个 program (CUDA block) 的线程数
        self.names = number_values(func)
        self.alias = {}  # 不产生代码的指令（expand_dims、for 的结果）直接复用别的变量
        self.lines = []
        self.depth = 0
        self.smem_bytes = 0
        self.tmp_counter = 0
        self.arg_cnames = {}
        for a, n in zip(func.args, func.arg_names):
            self.arg_cnames[a] = n + "_" if n in C_RESERVED or (n[0] == "v" and n[1:].isdigit()) else n

    # ---------------------------------------------------------------- 工具
    def emit(self, line):
        self.lines.append("  " * self.depth + line if line else "")

    @contextmanager
    def indent(self, head, tail="}"):
        self.emit(head)
        self.depth += 1
        yield
        self.depth -= 1
        self.emit(tail)

    def n(self, v):
        while v in self.alias:
            v = self.alias[v]
        if v in self.arg_cnames:
            return self.arg_cnames[v]
        return "v" + self.names[v]

    def elems(self, ty):
        """每个线程持有的元素个数（寄存器数组的长度）。"""
        return (ty.numel + self.T - 1) // self.T

    def operand(self, v):
        return self.n(v) if v.type.is_scalar else self.n(v) + "[j]"

    @contextmanager
    def foreach(self, ty, need_idx=False):
        """遍历当前线程持有的所有元素：j 是寄存器下标，idx 是该元素在块中的线性下标。"""
        self.emit("#pragma unroll")
        with self.indent(f"for (int j = 0; j < {self.elems(ty)}; ++j) {{"):
            guard = ty.numel % self.T != 0
            if need_idx or guard:
                self.emit(f"const int idx = tid + j * {self.T};")
            if guard:
                self.emit(f"if (idx >= {ty.numel}) continue;")
            yield

    def declare(self, v):
        """声明一个值：标量是一个 C 变量，张量是每个线程的一个寄存器数组。"""
        ct = ctype(v.type.dtype)
        if v.type.is_scalar:
            self.emit(f"{ct} {self.n(v)};")
        else:
            self.emit(f"{ct} {self.n(v)}[{self.elems(v.type)}];")

    def copy(self, dst, src, ty):
        if ty.is_scalar:
            self.emit(f"{dst} = {src};")
        else:
            with self.foreach(ty):
                self.emit(f"{dst}[j] = {src}[j];")

    def to_smem(self, v, offset, name):
        """把张量 v 按线性下标写进 shared memory，之后任何线程都能读到任何元素。"""
        ct = ctype(v.type.dtype)
        self.emit(f"{ct}* {name} = reinterpret_cast<{ct}*>(mt_smem + {offset});")
        with self.foreach(v.type, need_idx=True):
            self.emit(f"{name}[idx] = {self.n(v)}[j];")
        end = offset + v.type.numel * sizeof(v.type.dtype)
        self.smem_bytes = max(self.smem_bytes, end)
        return (end + 15) // 16 * 16

    # ---------------------------------------------------------------- 入口
    def generate(self):
        params = ", ".join(f"{ctype(a.type.dtype)} {self.arg_cnames[a]}" for a in self.func.args)
        out = [PRELUDE]
        self.depth = 1
        self.emit("extern __shared__ __align__(16) unsigned char mt_smem[];")
        self.emit("const int tid = threadIdx.x;")
        self.gen_block(self.func.body)
        out.append(f'extern "C" __global__ void __launch_bounds__({self.T}) {self.func.name}({params}) {{')
        out.extend(self.lines)
        out.append("}")
        return "\n".join(out) + "\n"

    def gen_block(self, block, yield_targets=None):
        for op in block.ops:
            if op.name == "scf.yield":
                self.gen_yield(op, yield_targets)
            else:
                self.emit("")
                self.emit("// " + format_op(op, self.names))
                self.gen_op(op)

    def gen_op(self, op):
        name = op.name
        if name in ELEMENTWISE:
            return self.gen_elementwise(op, ELEMENTWISE[name])
        handler = getattr(self, "gen_" + name.replace(".", "_"), None)
        if handler is None:
            raise NotImplementedError(f"codegen 不支持 {name}")
        handler(op)

    # ---------------------------------------------------------------- 逐元素
    def gen_elementwise(self, op, expr):
        r = op.result
        if r.type.is_scalar:
            args = [self.n(o) for o in op.operands]
            self.emit(f"{ctype(r.type.dtype)} {self.n(r)} = {expr(args, op)};")
            return
        self.declare(r)
        with self.foreach(r.type, need_idx=op.name == "tt.make_range"):
            args = [self.operand(o) for o in op.operands]
            self.emit(f"{self.n(r)}[j] = {expr(args, op)};")

    def gen_arith_constant(self, op):
        dt = op.result.type.dtype
        self.emit(f"const {ctype(dt)} {self.n(op.result)} = {literal(op.attrs['value'], dt)};")

    def gen_tt_get_program_id(self, op):
        self.emit(f"const int {self.n(op.result)} = blockIdx.{'xyz'[op.attrs['axis']]};")

    def gen_tt_get_num_programs(self, op):
        self.emit(f"const int {self.n(op.result)} = gridDim.{'xyz'[op.attrs['axis']]};")

    def gen_tt_expand_dims(self, op):
        self.emit("// (无需代码：expand_dims 不改变元素的线性顺序，也就不改变 layout)")
        self.alias[op.result] = op.operands[0]

    def gen_tt_return(self, op):
        pass

    # ---------------------------------------------------------------- 内存
    def gen_tt_load(self, op):
        r = op.result
        if r.type.is_scalar:
            p = [self.n(o) for o in op.operands]
            val = f"*{p[0]}" if len(p) == 1 else f"{p[1]} ? *{p[0]} : {p[2]}"
            self.emit(f"{ctype(r.type.dtype)} {self.n(r)} = {val};")
            return
        self.declare(r)
        with self.foreach(r.type):
            p = [self.operand(o) for o in op.operands]
            val = f"*{p[0]}" if len(p) == 1 else f"{p[1]} ? *{p[0]} : {p[2]}"
            self.emit(f"{self.n(r)}[j] = {val};")

    def gen_tt_store(self, op):
        if op.operands[0].type.is_scalar:
            p = [self.n(o) for o in op.operands]
            cond = "tid == 0" + (f" && {p[2]}" if len(p) == 3 else "")
            self.emit(f"if ({cond}) *{p[0]} = {p[1]};  // 标量 store 只需要一个线程执行")
            return
        with self.foreach(op.operands[0].type):
            p = [self.operand(o) for o in op.operands]
            stmt = f"*{p[0]} = {p[1]};"
            self.emit(f"if ({p[2]}) {stmt}" if len(p) == 3 else stmt)

    # ---------------------------------------------------------------- 需要线程间通信的指令
    def gen_tt_broadcast(self, op):
        src, r = op.operands[0], op.result
        self.declare(r)
        with self.indent("{"):
            self.emit("__syncthreads();  // 等之前对 shared memory 的读完成")
            self.to_smem(src, 0, "smem_a")
            self.emit("__syncthreads();")
            # 目标元素的坐标 -> 源元素的线性下标（被广播的维度坐标恒为 0）
            src_st = strides(src.type.shape)
            terms = []
            for d, size in enumerate(src.type.shape):
                if size != 1:
                    c = coord_expr(d, r.type.shape)
                    terms.append(c if src_st[d] == 1 else f"{c} * {src_st[d]}")
            src_idx = " + ".join(terms) or "0"
            with self.foreach(r.type, need_idx=True):
                self.emit(f"{self.n(r)}[j] = smem_a[{src_idx}];")

    def combine(self, kind, dt, a, b):
        if kind == "sum":
            return f"{a} + {b}"
        if dt.is_floating():
            return f"{'fmaxf' if kind == 'max' else 'fminf'}({a}, {b})"
        return f"{kind}({a}, {b})"

    def identity(self, kind, dt):
        if kind == "sum":
            return literal(0, dt)
        if dt.is_floating():
            return "-INFINITY" if kind == "max" else "INFINITY"
        lo, hi = (-(2**31), 2**31 - 1) if dt == tl.int32 else (-(2**63), 2**63 - 1)
        v = lo if kind == "max" else hi
        return f"({v + 1}{'LL' if dt == tl.int64 else ''} - 1)" if v < 0 else literal(v, dt)

    def gen_tt_reduce(self, op):
        src, r = op.operands[0], op.result
        kind, dt = op.attrs["kind"], r.type.dtype
        ct = ctype(dt)
        comb = lambda a, b: self.combine(kind, dt, a, b)
        self.declare(r)
        if r.type.is_scalar:
            # 归约成标量：经典的三级归约
            with self.indent("{"):
                self.emit("// 1) 每个线程先归约自己寄存器里的元素")
                self.emit(f"{ct} acc = {self.identity(kind, dt)};")
                with self.foreach(src.type):
                    self.emit(f"acc = {comb('acc', self.n(src) + '[j]')};")
                self.emit("// 2) warp 内：用 shuffle 指令在 32 个线程的寄存器之间直接交换数据")
                self.emit(f"for (int off = 16; off > 0; off >>= 1) acc = "
                          f"{comb('acc', '__shfl_xor_sync(0xffffffffu, acc, off)')};")
                if self.num_warps > 1:
                    self.emit("// 3) warp 之间：每个 warp 的结果经 shared memory 汇总")
                    self.emit("__syncthreads();")
                    self.emit(f"{ct}* smem_a = reinterpret_cast<{ct}*>(mt_smem);")
                    self.smem_bytes = max(self.smem_bytes, self.num_warps * sizeof(dt))
                    self.emit("if (tid % 32 == 0) smem_a[tid / 32] = acc;")
                    self.emit("__syncthreads();")
                    self.emit("acc = smem_a[0];")
                    self.emit(f"for (int w = 1; w < {self.num_warps}; ++w) acc = {comb('acc', 'smem_a[w]')};")
                self.emit(f"{self.n(r)} = acc;  // 所有线程都拿到同一个结果")
            return
        # 沿某个轴归约：写进 shared memory，每个线程负责自己持有的输出元素
        axis, shape = op.attrs["axis"], src.type.shape
        src_st = strides(shape)
        out_dims = [d for d in range(len(shape)) if d != axis]
        terms = []
        for k, d in enumerate(out_dims):
            c = coord_expr(k, r.type.shape)
            terms.append(c if src_st[d] == 1 else f"{c} * {src_st[d]}")
        base = " + ".join(terms) or "0"
        with self.indent("{"):
            self.emit("__syncthreads();")
            self.to_smem(src, 0, "smem_a")
            self.emit("__syncthreads();")
            with self.foreach(r.type, need_idx=True):
                self.emit(f"const int base = {base};")
                self.emit(f"{ct} acc = smem_a[base];")
                step = "k" if src_st[axis] == 1 else f"k * {src_st[axis]}"
                self.emit(f"for (int k = 1; k < {shape[axis]}; ++k) acc = {comb('acc', f'smem_a[base + {step}]')};")
                self.emit(f"{self.n(r)}[j] = acc;")

    def gen_tt_dot(self, op):
        a, b = op.operands[:2]
        acc = op.operands[2] if len(op.operands) == 3 else None
        r = op.result
        (M, K), N = a.type.shape, b.type.shape[1]
        self.declare(r)
        with self.indent("{"):
            self.emit("// 朴素实现：A、B 放进 shared memory，每个线程算自己持有的 C 元素")
            self.emit("// (真正的 Triton 在这里会用 Tensor Core 的 mma / wgmma / tcgen05 指令)")
            self.emit("__syncthreads();")
            off = self.to_smem(a, 0, "smem_a")
            self.to_smem(b, off, "smem_b")
            self.emit("__syncthreads();")
            with self.foreach(r.type, need_idx=True):
                self.emit(f"const int m = idx / {N}, n = idx % {N};")
                self.emit(f"float acc = {self.operand(acc) if acc is not None else '0.0f'};")
                ea = to_float(f"smem_a[m * {K} + k]", a.type.dtype)
                eb = to_float(f"smem_b[k * {N} + n]", b.type.dtype)
                self.emit(f"for (int k = 0; k < {K}; ++k) acc += {ea} * {eb};")
                self.emit(f"{self.n(r)}[j] = acc;")

    # ---------------------------------------------------------------- 控制流
    def gen_scf_for(self, op):
        lb, ub, step, *inits = op.operands
        body = op.regions[0]
        iv, *carried = body.args
        # 循环携带变量：在循环外声明并赋初值，循环体末尾 (yield) 更新它们
        for arg, init in zip(carried, inits):
            self.declare(arg)
            self.copy(self.n(arg), self.n(init), arg.type)
        for res, arg in zip(op.results, carried):
            self.alias[res] = arg  # 循环结束后，结果就是这些变量的最终值
        ivn, ct = self.n(iv), ctype(iv.type.dtype)
        cmp = op.attrs["cmp"]
        with self.indent(f"for ({ct} {ivn} = {self.n(lb)}; {ivn} {cmp} {self.n(ub)}; {ivn} += {self.n(step)}) {{"):
            self.gen_block(body, carried)

    def gen_scf_if(self, op):
        for r in op.results:
            self.declare(r)
        with self.indent(f"if ({self.n(op.operands[0])}) {{", "} else {"):
            self.gen_block(op.regions[0], op.results)
        self.depth += 1
        self.gen_block(op.regions[1], op.results)
        self.depth -= 1
        self.emit("}")

    def gen_yield(self, op, targets):
        if not targets:
            return
        self.emit("// " + format_op(op, self.names))
        pairs = [(self.n(t), self.n(s), t.type) for t, s in zip(targets, op.operands) if self.n(t) != self.n(s)]
        dst_names = {d for d, _, _ in pairs}
        if any(s in dst_names for _, s, _ in pairs):
            # 例如 a, b = b, a：先全部拷到临时变量，避免互相覆盖
            tmps = []
            for d, s, ty in pairs:
                tmp = f"tmp{self.tmp_counter}"
                self.tmp_counter += 1
                ct = ctype(ty.dtype)
                self.emit(f"{ct} {tmp}" + ("" if ty.is_scalar else f"[{self.elems(ty)}]") + ";")
                self.copy(tmp, s, ty)
                tmps.append((d, tmp, ty))
            pairs = tmps
        for d, s, ty in pairs:
            self.copy(d, s, ty)


def generate(func, num_warps):
    cg = CudaCodegen(func, num_warps)
    src = cg.generate()
    return src, cg.smem_bytes
