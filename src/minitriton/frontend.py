"""frontend.py —— 把 Python 函数的 AST 翻译成 IR（对应 triton/compiler/code_generator.py）。

这是"Python 变成底层代码"的第一步。做法和 Triton 完全一样：
  1. inspect.getsource 拿到 kernel 源码，ast.parse 得到语法树
  2. 用 ast.NodeVisitor 遍历语法树，一边"解释"一边生成 IR：
       - Python 的 int/float/bool 视为编译期常量，直接在 Python 里计算（常量折叠）
       - 涉及 tensor 的运算交给 semantic.py 生成 IR 指令
       - for/if 变成带区域的 scf.for / scf.if 指令
  3. 本身不会执行 kernel 里的任何代码。
"""

import ast
import builtins
import operator

from . import ir
from . import language as tl
from . import semantic
from .ir import Builder, TensorType
from .semantic import tensor


class CompilationError(Exception):
    def __init__(self, jit_fn, node, msg):
        lineno = getattr(node, "lineno", None)
        where = ""
        if lineno is not None:
            line = jit_fn.src.splitlines()[lineno - 1]
            where = f"\n  {jit_fn.fn.__code__.co_filename}:{jit_fn.starting_line + lineno - 1}\n    {line.strip()}"
        super().__init__(f"编译 {jit_fn.__name__} 出错: {msg}{where}")


BIN_OPS = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%",
    ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^", ast.LShift: "<<", ast.RShift: ">>",
}
CMP_OPS = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}
PY_OPS = {
    "+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv,
    "//": operator.floordiv, "%": operator.mod, "&": operator.and_, "|": operator.or_,
    "^": operator.xor, "<<": operator.lshift, ">>": operator.rshift, "<": operator.lt,
    "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq, "!=": operator.ne,
}


def is_const(x):
    return not isinstance(x, tensor)


class TensorMethod:
    """x.to(tl.float16) 这种张量方法调用。"""

    def __init__(self, obj, name):
        self.obj, self.name = obj, name


class CodeGenerator(ast.NodeVisitor):
    def __init__(self, jit_fn, builder, scope):
        self.jit_fn = jit_fn
        self.gscope = jit_fn.fn.__globals__  # kernel 所在模块的全局变量（例如 tl）
        self.b = builder
        self.scope = scope  # 局部变量名 -> tensor 或 Python 常量
        self.ret = None
        self.returned = False
        self.region_depth = 0

    # ---------------------------------------------------------------- 基础设施
    def visit(self, node):
        try:
            return super().visit(node)
        except CompilationError:
            raise
        except Exception as e:
            raise CompilationError(self.jit_fn, node, f"{type(e).__name__}: {e}") from e

    def generic_visit(self, node):
        raise NotImplementedError(f"不支持的语法: {type(node).__name__}")

    def visit_body(self, stmts):
        for stmt in stmts:
            if self.returned:
                break
            self.visit(stmt)

    # ---------------------------------------------------------------- 语句
    def visit_Expr(self, node):
        self.visit(node.value)

    def visit_Pass(self, node):
        pass

    def visit_Assign(self, node):
        if len(node.targets) != 1:
            raise NotImplementedError("不支持 a = b = c")
        self.assign(node.targets[0], self.visit(node.value))

    def visit_AnnAssign(self, node):
        self.assign(node.target, self.visit(node.value))

    def assign(self, target, value):
        if isinstance(target, ast.Name):
            self.scope[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            values = tuple(value)
            if len(values) != len(target.elts):
                raise ValueError("解包数量不匹配")
            for t, v in zip(target.elts, values):
                self.assign(t, v)
        else:
            raise NotImplementedError(f"不支持给 {type(target).__name__} 赋值")

    def visit_AugAssign(self, node):
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError("+= 的左边只能是变量名")
        lhs = self.visit(node.target)
        self.scope[node.target.id] = self.binary(BIN_OPS[type(node.op)], lhs, self.visit(node.value))

    def visit_Return(self, node):
        if self.region_depth > 0:
            raise NotImplementedError("不支持在 for/if 内部 return")
        self.ret = self.visit(node.value) if node.value is not None else None
        self.returned = True

    # ---------------------------------------------------------------- 控制流
    def dry_run(self, stmts, overrides=None):
        """在一个丢弃的 Block 里先"试跑"一遍语句，看看哪些变量会被修改、修改后是什么类型。

        这和 Triton 的做法一致：生成 scf.for 之前必须先知道循环携带变量 (iter_args) 有哪些。
        """
        saved_scope, saved_ret = self.scope, (self.ret, self.returned)
        self.scope = {**saved_scope, **(overrides or {})}
        self.region_depth += 1
        try:
            with self.b.insertion_point(ir.Block()):
                self.visit_body(stmts)
            return self.scope
        finally:
            self.scope = saved_scope
            self.ret, self.returned = saved_ret
            self.region_depth -= 1

    def changed_vars(self, before, after, ignore=()):
        names = []
        for n, v in before.items():
            if n in ignore or n not in after or after[n] is v:
                continue
            if is_const(v) and is_const(after[n]):
                if v == after[n]:
                    continue
                raise TypeError(f"编译期常量 {n} 在循环/分支中被修改，这在编译期无法确定")
            names.append(n)
        return names

    def var_type(self, v):
        return v.type if isinstance(v, tensor) else TensorType(semantic._py_dtype(v))

    def convert(self, v, ty):
        """把 v 转换成类型 ty（用于循环的初值 / yield 值）。"""
        v = semantic.cast(self.b, semantic.to_tensor(self.b, v), ty.dtype)
        return semantic.broadcast_to(self.b, v, ty.shape)

    def visit_For(self, node):
        if node.orelse:
            raise NotImplementedError("不支持 for ... else")
        it = node.iter
        fn = self.visit(it.func) if isinstance(it, ast.Call) else None
        if fn is not builtins.range and fn is not tl.static_range:
            raise NotImplementedError("for 循环只支持 range(...) 和 tl.static_range(...)")
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError("循环变量必须是单个名字")
        var = node.target.id
        args = [self.visit(a) for a in it.args]

        if fn is tl.static_range:  # 编译期展开：直接在 Python 里循环，循环体被复制多份
            for i in range(*[semantic._check_const(a, "static_range 的参数") for a in args]):
                self.scope[var] = i
                self.visit_body(node.body)
            return

        lb, ub, step = (0, args[0], 1) if len(args) == 1 else (*args, 1)[:3]
        # 循环方向在编译期决定：步长是常量负数时用 >
        cmp = ">" if is_const(step) and step < 0 else "<"
        int_ty = tl.int64 if any(isinstance(a, tensor) and a.dtype == tl.int64 for a in (lb, ub, step)) else tl.int32
        lb, ub, step = (semantic.to_tensor(self.b, a, int_ty) for a in (lb, ub, step))

        # 第一步：试跑循环体，找出循环携带变量
        dummy_iv = tensor(ir.Value(TensorType(int_ty)))
        after = self.dry_run(node.body, {var: dummy_iv})
        names = self.changed_vars(self.scope, after, ignore={var})
        types = [self.var_type(after[n]) for n in names]
        inits = [self.convert(self.scope[n], t) for n, t in zip(names, types)]

        # 第二步：真正生成 scf.for
        body = ir.Block([ir.Value(TensorType(int_ty))] + [ir.Value(t) for t in types])
        op = self.b.create("scf.for", [lb.handle, ub.handle, step.handle] + [i.handle for i in inits],
                           types, {"cmp": cmp}, [body])
        saved = self.scope
        self.scope = dict(saved)
        self.scope[var] = tensor(body.args[0])
        for n, a in zip(names, body.args[1:]):
            self.scope[n] = tensor(a)
        self.region_depth += 1
        with self.b.insertion_point(body):
            self.visit_body(node.body)
            yields = [self.convert(self.scope[n], t) for n, t in zip(names, types)]
            self.b.create("scf.yield", [y.handle for y in yields], [])
        self.region_depth -= 1
        self.scope = saved
        for n, r in zip(names, op.results):
            self.scope[n] = tensor(r)

    def visit_If(self, node):
        cond = self.visit(node.test)
        if is_const(cond):  # 编译期 if：只翻译被选中的分支，另一个分支直接消失
            self.visit_body(node.body if cond else node.orelse)
            return
        if not cond.type.is_scalar:
            raise TypeError("if 的条件必须是标量；逐元素的选择请使用 tl.where")
        cond = semantic.cast(self.b, cond, tl.int1)

        then_scope = self.dry_run(node.body)
        else_scope = self.dry_run(node.orelse)
        names = self.changed_vars(self.scope, then_scope) + self.changed_vars(self.scope, else_scope)
        names += [n for n in then_scope if n not in self.scope and n in else_scope]
        names = list(dict.fromkeys(names))
        types = []
        for n in names:
            changed_in_then = n not in self.scope or then_scope[n] is not self.scope[n]
            types.append(self.var_type(then_scope[n] if changed_in_then else else_scope[n]))

        then_blk, else_blk = ir.Block(), ir.Block()
        op = self.b.create("scf.if", [cond.handle], types, regions=[then_blk, else_blk])
        saved = self.scope
        self.region_depth += 1
        for blk, stmts in ((then_blk, node.body), (else_blk, node.orelse)):
            self.scope = dict(saved)
            with self.b.insertion_point(blk):
                self.visit_body(stmts)
                yields = [self.convert(self.scope[n], t) for n, t in zip(names, types)]
                self.b.create("scf.yield", [y.handle for y in yields], [])
        self.region_depth -= 1
        self.scope = saved
        for n, r in zip(names, op.results):
            self.scope[n] = tensor(r)

    # ---------------------------------------------------------------- 表达式
    def visit_Constant(self, node):
        return node.value

    def visit_Name(self, node):
        if node.id in self.scope:
            return self.scope[node.id]
        if node.id in self.gscope:
            return self.gscope[node.id]
        if hasattr(builtins, node.id):
            return getattr(builtins, node.id)
        raise NameError(f"未定义的名字 {node.id}")

    def visit_Tuple(self, node):
        return tuple(self.visit(e) for e in node.elts)

    visit_List = visit_Tuple

    def visit_Attribute(self, node):
        obj = self.visit(node.value)
        if isinstance(obj, tensor):
            if node.attr in ("dtype", "shape"):
                return getattr(obj, node.attr)
            if node.attr == "to":
                return TensorMethod(obj, "to")
            raise AttributeError(f"tensor 没有属性 {node.attr}")
        return getattr(obj, node.attr)

    def binary(self, op, lhs, rhs):
        if is_const(lhs) and is_const(rhs):
            return PY_OPS[op](lhs, rhs)  # 常量折叠
        return semantic.binary(self.b, op, lhs, rhs)

    def visit_BinOp(self, node):
        return self.binary(BIN_OPS[type(node.op)], self.visit(node.left), self.visit(node.right))

    def visit_Compare(self, node):
        if len(node.ops) != 1:
            raise NotImplementedError("不支持连续比较 a < b < c")
        return self.binary(CMP_OPS[type(node.ops[0])], self.visit(node.left), self.visit(node.comparators[0]))

    def visit_UnaryOp(self, node):
        x = self.visit(node.operand)
        op = {ast.USub: "-", ast.UAdd: "+", ast.Invert: "~", ast.Not: "not"}[type(node.op)]
        if is_const(x):
            return {"-": operator.neg, "+": operator.pos, "~": operator.invert, "not": operator.not_}[op](x)
        return semantic.unary(self.b, op, x)

    def visit_BoolOp(self, node):
        values = [self.visit(v) for v in node.values]
        if all(is_const(v) for v in values):
            return all(values) if isinstance(node.op, ast.And) else any(values)
        op = "&" if isinstance(node.op, ast.And) else "|"
        result = semantic.cast(self.b, values[0], tl.int1)
        for v in values[1:]:
            result = semantic.binary(self.b, op, result, semantic.cast(self.b, v, tl.int1))
        return result

    def visit_IfExp(self, node):
        cond = self.visit(node.test)
        if is_const(cond):
            return self.visit(node.body if cond else node.orelse)
        return semantic.where(self.b, cond, self.visit(node.body), self.visit(node.orelse))

    def visit_Subscript(self, node):
        obj = self.visit(node.value)
        if is_const(obj):
            return obj[self.visit(node.slice)]
        # 只支持 x[:, None] / x[None, :] 这种增加维度的写法
        items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        dim = 0
        for item in items:
            if isinstance(item, ast.Constant) and item.value is None:
                obj = semantic.expand_dims(self.b, obj, dim)
            elif isinstance(item, ast.Slice) and item.lower is None and item.upper is None and item.step is None:
                pass
            else:
                raise NotImplementedError("张量下标只支持 ':' 和 None")
            dim += 1
        return obj

    def visit_Call(self, node):
        fn = self.visit(node.func)
        args = [self.visit(a) for a in node.args]
        kwargs = {k.arg: self.visit(k.value) for k in node.keywords}
        all_const = all(is_const(a) for a in [*args, *kwargs.values()])

        from .jit import JITFunction  # 避免循环 import

        if isinstance(fn, TensorMethod):  # x.to(dtype)
            return semantic.cast(self.b, fn.obj, *args, **kwargs)
        if isinstance(fn, JITFunction):  # 调用另一个 @jit 函数：内联展开
            return self.call_jit(fn, args, kwargs)
        name = getattr(fn, "__mt_builtin__", None)
        if name is not None and not (all_const and getattr(fn, "__mt_constexpr_ok__", False)):
            if name not in semantic.BUILTINS:
                raise NotImplementedError(f"tl.{name} 不能在这里使用")
            return semantic.BUILTINS[name](self.b, *args, **kwargs)
        if fn in (builtins.range,):
            raise NotImplementedError("range 只能用在 for 循环里")
        if all_const:  # 普通 Python 函数 + 常量参数：编译期直接执行，例如 min(a, b)、float('inf')
            return fn(*args, **kwargs)
        raise TypeError(f"不能把张量传给普通 Python 函数 {fn!r}")

    def call_jit(self, jit_fn, args, kwargs):
        bound = jit_fn.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        gen = CodeGenerator(jit_fn, self.b, dict(bound.arguments))
        gen.visit_body(jit_fn.parse().body)
        return gen.ret


def ast_to_ttir(jit_fn, signature, constexprs):
    """signature: 运行期参数名 -> dtype；constexprs: 编译期参数名 -> 值。"""
    params = [n for n in jit_fn.arg_names if n not in constexprs]
    func = ir.Function(jit_fn.__name__, [TensorType(signature[n]) for n in params], params)
    builder = Builder()
    builder.block = func.body
    scope = {n: tensor(a) for n, a in zip(params, func.args)}
    scope.update(constexprs)
    CodeGenerator(jit_fn, builder, scope).visit_body(jit_fn.parse().body)
    builder.create("tt.return", [], [])
    return func
