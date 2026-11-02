"""passes.py —— IR 上的优化 pass（对应 Triton/MLIR 里的 canonicalize、cse、licm 等）。

编译器的一个核心思想：前端只管"正确地"生成代码，哪怕很啰嗦；
然后由一系列独立的、小的 pass 把它变好。这里实现三个最经典的：
  - CSE  (公共子表达式消除)：两条一模一样的纯指令只保留一条
  - LICM (循环不变量外提)：  循环体里不依赖循环的计算挪到循环外
  - DCE  (死代码消除)：      结果没人用的纯指令删掉
"""

from .ir import walk

SIDE_EFFECT_OPS = {"tt.store", "tt.return", "scf.yield", "scf.for", "scf.if"}


def has_side_effects(op):
    return op.name in SIDE_EFFECT_OPS


def is_pure(op):
    # load 没有副作用（可以删），但读内存的结果可能被 store 改变，所以不算"纯"（不能合并/外提）
    return not has_side_effects(op) and op.name != "tt.load"


def cse(func):
    mapping = {}  # 被消除的 Value -> 替代它的 Value

    def visit(block, table):
        table = dict(table)  # 外层的表达式在内层区域可见，反之不行
        kept = []
        for op in block.ops:
            op.operands = [mapping.get(o, o) for o in op.operands]
            for region in op.regions:
                visit(region, table)
            if is_pure(op) and not op.regions:
                key = (op.name, tuple(map(id, op.operands)), tuple(sorted(op.attrs.items())),
                       tuple(r.type for r in op.results))
                if key in table:
                    mapping.update(zip(op.results, table[key]))
                    continue
                table[key] = op.results
            kept.append(op)
        block.ops = kept

    visit(func.body, {})


def licm(func):
    def visit(block):
        new_ops = []
        for op in block.ops:
            for region in op.regions:
                visit(region)  # 先处理内层循环，内层外提出来的指令可能还能继续外提
            if op.name == "scf.for":
                body = op.regions[0]
                defined = set(body.args)
                for inner in walk(body):
                    defined.update(inner.results)
                kept = []
                for inner in body.ops:
                    if is_pure(inner) and not inner.regions and not any(o in defined for o in inner.operands):
                        new_ops.append(inner)  # 放到 for 前面
                        defined.difference_update(inner.results)
                    else:
                        kept.append(inner)
                body.ops = kept
            new_ops.append(op)
        block.ops = new_ops

    visit(func.body)


def dce(func):
    while True:
        used = set()
        for op in walk(func.body):
            used.update(op.operands)
        removed = False

        def visit(block):
            nonlocal removed
            kept = []
            for op in block.ops:
                for region in op.regions:
                    visit(region)
                if not has_side_effects(op) and not any(r in used for r in op.results):
                    removed = True
                    continue
                kept.append(op)
            block.ops = kept

        visit(func.body)
        if not removed:
            return


def optimize(func):
    cse(func)
    licm(func)
    cse(func)
    dce(func)
