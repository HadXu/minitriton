# minitriton

一个用于学习的迷你 [Triton](https://github.com/triton-lang/triton)。大约 1900 行 Python（含大量中文注释），
完整走一遍 **"Python 函数 → GPU 机器码"** 的编译流水线，结构和命名都尽量对齐真正的 Triton，
读懂之后再去读 Triton 源码会轻松很多。

```python
import minitriton as mt
import minitriton.language as tl

@mt.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

kernel = add_kernel[(mt.cdiv(n, 1024),)](x, y, out, n, BLOCK_SIZE=1024)
print(kernel.asm["ttir"])   # 也可以看 "ttir_opt" / "cuda" / "ptx" / "ptxas"
```

## 快速开始

```bash
uv sync
uv run python examples/01_vector_add.py   # 打印每个编译阶段的产物 —— 建议从这里开始
uv run python examples/02_softmax.py      # 归约：warp shuffle + shared memory
uv run python examples/03_matmul.py       # 二维网格、for 循环、tl.dot、LICM
uv run pytest tests                       # 测试

# 把所有中间产物写到目录里，方便用编辑器对照着看
MINITRITON_DUMP_DIR=./dump uv run python examples/03_matmul.py
```

需要：NVIDIA GPU、CUDA Toolkit（`nvcc`、`ptxas`；找不到时设置 `CUDA_HOME`）、PyTorch（只用来分配显存和做对比）。

## 编译流水线

```
 Python 源码 ──inspect.getsource──▶ AST
     │  frontend.py    遍历 AST，"符号执行"，生成块级 IR
     ▼
 TTIR (Triton IR)       %5 = arith.addi %4, %3 : tensor<1024xi32>
     │  passes.py      CSE / LICM / DCE
     ▼
 TTIR (优化后)
     │  codegen.py     ★ 决定 layout：块里每个元素由哪个线程、哪个寄存器持有
     ▼
 CUDA C                 int v5[8]; for (j...) v5[j] = v4[j] + v3[j];
     │  nvcc -ptx      (真正的 Triton：TTGIR → LLVM IR → LLVM NVPTX 后端)
     ▼
 PTX
     │  ptxas          (和 Triton 一模一样)
     ▼
 cubin ──cuModuleLoadData / cuLaunchKernel──▶ 在 GPU 上运行
```

| 文件 | 作用 | 对应的 Triton 源码 |
|---|---|---|
| [language.py](src/minitriton/language.py) | `tl.*` 的 API 和 dtype；函数本身只是"符号" | `python/triton/language/core.py` |
| [jit.py](src/minitriton/jit.py) | `@jit`、`kernel[grid](...)`、参数特化和缓存 | `python/triton/runtime/jit.py` |
| [frontend.py](src/minitriton/frontend.py) | AST → IR | `python/triton/compiler/code_generator.py` |
| [semantic.py](src/minitriton/semantic.py) | 类型提升、广播、每个 `tl.xxx` 生成什么指令 | `python/triton/language/semantic.py` |
| [ir.py](src/minitriton/ir.py) | SSA IR 数据结构和打印 | MLIR + `lib/Dialect/Triton` |
| [passes.py](src/minitriton/passes.py) | 优化 pass | MLIR 的 canonicalize/cse/licm |
| [codegen.py](src/minitriton/codegen.py) | 块级 IR → 每线程 CUDA C | `lib/Conversion/TritonGPUToLLVM` + `TritonToTritonGPU` |
| [compiler.py](src/minitriton/compiler.py) | 串起所有阶段、调用 nvcc/ptxas、缓存 | `python/triton/compiler/compiler.py` + `backends/nvidia/compiler.py` |
| [driver.py](src/minitriton/driver.py) | ctypes 调用 libcuda 加载/启动 | `backends/nvidia/driver.py` |

建议的阅读顺序：`jit.py` → `frontend.py` → `semantic.py` → `ir.py` → `passes.py` → `codegen.py` → `compiler.py`。

## 核心概念

### 1. kernel 从来没有被"执行"过

`@jit` 拿到的是函数的**源码**。`frontend.py` 用 `ast.NodeVisitor` 遍历语法树，每遇到一个表达式就决定：

- 两边都是 Python 常量（`int/float/bool`，包括 `tl.constexpr` 参数）→ 直接在 Python 里算出结果。
  所以 `BLOCK_SIZE * 2`、`float("inf")`、`if KIND == "sum":` 在编译期就消失了；
- 有一边是 `tensor`（IR 里的一个值）→ 交给 `semantic.py` 生成 IR 指令。

这就是为什么 `BLOCK_SIZE` 必须是 `tl.constexpr`：`tl.arange(0, BLOCK_SIZE)` 的形状要在编译期确定。
每组不同的 constexpr 值（以及参数的 dtype）都会编译出一份单独的 kernel（`JITFunction.cache`）。

### 2. 块级 IR

```
%3 = tt.make_range {start = 0, end = 1024} : tensor<1024xi32>
%4 = tt.splat %2 : tensor<1024xi32>              // 标量 -> 张量
%5 = arith.addi %4, %3 : tensor<1024xi32>
```

IR 里的值是整块数据（`tensor<1024xi32>`），而不是单个线程的标量。这是 Triton 和 CUDA 编程模型最大的区别：
**你描述"一个程序实例处理一块数据"，编译器负责把这块数据分给线程。**

`semantic.py` 会把所有隐式的东西显式化：类型提升 (`i32 + f32` → 插入 `arith.sitofp`)、
广播 (`[64,1] + [1,32]` → `tt.broadcast`)、指针运算 (`ptr + offs` → `tt.addptr`)。
所以到了后端，每条逐元素指令的操作数形状和类型都完全一致。

### 3. 控制流：`scf.for` 和循环携带变量

```python
acc = tl.zeros((BM, BN), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BK)):
    acc += tl.dot(a, b)
```

SSA 里一个值只能定义一次，那 `acc` 怎么在循环里被修改？答案是把它变成循环的参数：

```
%54, %55, %56 = scf.for %57 = %40 to %39 step %37 iter_args(%58 = %21, %59 = %32, %60 = %34) {
    ...
    scf.yield %75, %76, %74
}
```

生成 `scf.for` 之前必须知道哪些变量被修改了。`frontend.py` 的做法（和 Triton 一致）是先在一个
丢弃的 Block 里**试跑**一遍循环体（`dry_run`），比较前后的变量表，然后再正式生成。`if` 同理。

### 4. ★ Layout：从"块"到"线程"（codegen.py）

这是整个项目最值得看的部分。本项目使用最简单的 layout：

```
tensor<8xf32>，4 个线程：
  元素:  0  1  2  3  4  5  6  7
  线程:  0  1  2  3  0  1  2  3     线程 t 的寄存器数组 v[j] 存第 t + j*T 个元素
```

有了它，每类指令的翻译方式就清楚了：

| 指令 | 是否需要线程间通信 | 生成的代码 |
|---|---|---|
| 逐元素运算、load/store | 否 | 每个线程循环处理自己的 `v[j]`（`#pragma unroll` 后全部变成寄存器） |
| `expand_dims` | 否 | 什么都不生成 —— 线性顺序没变，layout 就没变 |
| `broadcast` | 是 | 写 shared memory → `__syncthreads()` → 按新坐标读回来 |
| 归约成标量 | 是 | 线程内归约 → warp 内 `__shfl_xor_sync` → warp 间经 shared memory |
| 沿轴归约、`dot` | 是 | 经 shared memory |

相邻线程访问相邻地址，全局内存访问自然是合并 (coalesced) 的。

真正的 Triton 在这里复杂得多：它有一个专门的 TritonGPU 方言，给每个张量附上 layout
（`#blocked`、`#mma`、`#shared`、`#slice`...），通过 pass 尽量选择不需要通信的 layout，
只在必要时插入 `convert_layout`（用 shared memory 实现，正是我们 `broadcast` 做的事）。

### 5. 优化 pass

打开 `examples/03_matmul.py` 的输出对比 `ttir` 和 `ttir_opt`：

- **CSE**：两次 `tl.load(..., mask=mask)` 生成的两个相同的 `0.0` 常量合并成一个；
- **LICM**：循环里 `offs_m[:, None] < M` 不依赖循环变量，被挪到循环外 —— 否则每次迭代都要做一次 shared memory 广播；
- **DCE**：删掉没人使用的指令。

## 和真正的 Triton 差在哪（也是很好的练习题）

`03_matmul.py` 在 B200 上大约 18 TFLOPS，而 cuBLAS 超过 1300 TFLOPS。差距来自：

1. **Tensor Core**：`tl.dot` 应该降低成 `mma.sync` / `wgmma` (Hopper) / `tcgen05.mma` (Blackwell)，而不是标量 FMA。
2. **更好的 layout**：每个线程持有 4 个连续元素（`sizePerThread=[4]`）就能用 128-bit 向量化访存（`ld.global.v4`）。
   练习：修改 `codegen.py` 的 layout，让线程 t 持有元素 `[4t, 4t+4)`。
3. **软件流水线**：Triton 会把下一轮的 load 提前到这一轮计算时发出（`num_stages`），用 `cp.async` / TMA 异步拷贝到 shared memory。
4. **广播不需要 shared memory**：`offs_m[:, None]` 这种由 `arange` 算出来的值，可以在每个线程里直接按坐标重新计算（rematerialization）。
   练习：在 `codegen.py` 里给 `make_range`/`splat`/`expand_dims` 链的 broadcast 走重算路径。
5. **参数特化**：Triton 会检查整数参数是否等于 1、指针是否 16 字节对齐，据此生成更好的代码。
6. **直接生成 LLVM IR / PTX**：本项目借助 nvcc 把 CUDA C 编译成 PTX，便于阅读；Triton 自己生成 LLVM IR。
   练习：为逐元素 kernel 直接生成 PTX 文本（不经过 nvcc），用 `ptxas` 编译。

其他已知的简化：`//` 和 `%` 对负整数是 C 语义（向零取整）；不支持 `while`、`break`、在 `for/if` 里 `return`；
`tl.dot` 的输入只支持浮点；只支持 1 个 CUDA 设备。

## 支持的 API

`tl.program_id` `tl.num_programs` `tl.arange` `tl.load` `tl.store` `tl.zeros` `tl.full` `tl.where`
`tl.maximum` `tl.minimum` `tl.exp` `tl.log` `tl.sqrt` `tl.rsqrt` `tl.sin` `tl.cos` `tl.floor` `tl.abs`
`tl.sum` `tl.max` `tl.min` `tl.dot` `tl.cdiv` `tl.static_range` `x.to(dtype)` `x[:, None]`，
dtype：`tl.int1` `tl.int32` `tl.int64` `tl.float16` `tl.float32`。

语法：算术/比较/位运算、`for ... in range(...)`、`if`（编译期常量条件或标量条件）、条件表达式、
调用其他 `@jit` 函数（会被内联）、用常量参数调用普通 Python 函数（编译期执行）。
