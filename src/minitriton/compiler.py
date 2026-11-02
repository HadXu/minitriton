"""compiler.py —— 把各个阶段串成完整的编译流水线（对应 triton/compiler/compiler.py）。

    Python 源码
      │  frontend.ast_to_ttir        (AST -> 块级 IR)
      ▼
    TTIR
      │  passes.optimize             (CSE / LICM / DCE)
      ▼
    TTIR (优化后)
      │  codegen.generate            (决定 layout，块级 IR -> 每线程的 CUDA C)
      ▼
    CUDA C
      │  nvcc -ptx                   (Triton 这里是 LLVM IR -> NVPTX 后端)
      ▼
    PTX  (NVIDIA 的虚拟汇编)
      │  ptxas                       (和 Triton 完全一样)
      ▼
    cubin (GPU 机器码 SASS)
      │  cuModuleLoadData
      ▼
    可以启动的 kernel

每个阶段的产物都保存在 CompiledKernel.asm 里，方便学习时逐层查看。
"""

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import codegen, frontend, ir, passes
from .driver import driver

CACHE_DIR = Path(os.environ.get("MINITRITON_CACHE_DIR", Path.home() / ".cache" / "minitriton"))


def _cuda_bin(tool):
    path = shutil.which(tool)
    if path:
        return path
    home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    path = os.path.join(home, "bin", tool)
    if not os.path.exists(path):
        raise RuntimeError(f"找不到 {tool}，请设置 CUDA_HOME")
    return path


def build_cubin(cuda_src, arch):
    """CUDA C -> PTX -> cubin。结果按源码哈希缓存在磁盘上，避免每次运行都调用 nvcc。"""
    key = hashlib.sha256(f"{arch}\n{cuda_src}".encode()).hexdigest()[:32]
    cache = CACHE_DIR / key
    files = [cache / "kernel.ptx", cache / "kernel.cubin", cache / "ptxas.txt"]
    if not all(f.exists() for f in files):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "kernel.cu").write_text(cuda_src)
            for cmd in (
                [_cuda_bin("nvcc"), "-ptx", f"-arch=compute_{arch}", "-O3", "-o", "kernel.ptx", "kernel.cu"],
                [_cuda_bin("ptxas"), f"-arch=sm_{arch}", "-O3", "-v", "-o", "kernel.cubin", "kernel.ptx"],
            ):
                p = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
                if p.returncode != 0:
                    raise RuntimeError(f"{' '.join(cmd)} 失败:\n{p.stderr}\n--- CUDA 源码 ---\n{cuda_src}")
            (tmp / "ptxas.txt").write_text(p.stderr)
            cache.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(tmp / f.name, f)
    return files[0].read_text(), files[1].read_bytes(), files[2].read_text()


CTYPES_ARG = {"i1": ctypes.c_bool, "i32": ctypes.c_int32, "i64": ctypes.c_int64, "f32": ctypes.c_float}


class CompiledKernel:
    def __init__(self, name, asm, signature, num_warps, smem_bytes, function):
        self.name = name
        self.asm = asm  # 各阶段的产物：python / ttir / ttir_opt / cuda / ptx / ptxas
        self.signature = signature  # 运行期参数名 -> dtype
        self.num_warps = num_warps
        self.smem_bytes = smem_bytes
        self.function = function

    def launch(self, grid, args):
        import torch

        c_args = []
        for (name, dt), v in zip(self.signature.items(), args):
            if dt.is_ptr():
                c_args.append(ctypes.c_void_p(v.data_ptr()))
            else:
                c_args.append(CTYPES_ARG[dt.name](v))
        stream = torch.cuda.current_stream().cuda_stream
        driver().launch(self.function, grid, self.num_warps * 32, self.smem_bytes, stream, c_args)


def compile(jit_fn, signature, constexprs, num_warps):
    import torch

    asm = {"python": jit_fn.src}
    func = frontend.ast_to_ttir(jit_fn, signature, constexprs)
    asm["ttir"] = ir.print_function(func)
    passes.optimize(func)
    asm["ttir_opt"] = ir.print_function(func)
    cuda_src, smem_bytes = codegen.generate(func, num_warps)
    asm["cuda"] = cuda_src
    major, minor = torch.cuda.get_device_capability()
    asm["ptx"], cubin, asm["ptxas"] = build_cubin(cuda_src, f"{major}{minor}")

    dump_dir = os.environ.get("MINITRITON_DUMP_DIR")
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        ext = {"python": "py", "ttir": "ttir", "ttir_opt": "opt.ttir", "cuda": "cu", "ptx": "ptx", "ptxas": "ptxas.txt"}
        for stage, text in asm.items():
            Path(dump_dir, f"{func.name}.{ext[stage]}").write_text(text)

    function = driver().load(cubin, func.name, smem_bytes)
    return CompiledKernel(func.name, asm, signature, num_warps, smem_bytes, function)
