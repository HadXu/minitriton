"""driver.py —— 用 ctypes 直接调用 CUDA Driver API (libcuda.so) 加载和启动 kernel。

Triton 的 backends/nvidia/driver.py 做的是同样的事（它用一小段 C 扩展）：
  cuModuleLoadData    把 cubin 加载到 GPU
  cuModuleGetFunction 按名字拿到 kernel 函数句柄
  cuLaunchKernel      指定 grid/block/shared memory/stream 启动
"""

import ctypes

CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8


class CudaDriver:
    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.lib.cuLaunchKernel.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
        ]

    def check(self, err, what):
        if err != 0:
            msg = ctypes.c_char_p()
            self.lib.cuGetErrorString(err, ctypes.byref(msg))
            raise RuntimeError(f"{what} 失败: {msg.value.decode() if msg.value else err}")

    def ensure_context(self):
        """复用 PyTorch 已经创建的 primary context，这样我们的 kernel 能直接访问 torch 张量。"""
        import torch

        dev = torch.cuda.current_device()
        torch.cuda.init()
        ctx = ctypes.c_void_p()
        self.check(self.lib.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
        if not ctx.value:
            self.check(self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
            self.check(self.lib.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

    def load(self, cubin, name, smem_bytes):
        self.ensure_context()
        mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
        self.check(self.lib.cuModuleLoadData(ctypes.byref(mod), ctypes.c_char_p(cubin)), "cuModuleLoadData")
        self.check(self.lib.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode()), "cuModuleGetFunction")
        if smem_bytes > 48 * 1024:  # 超过 48KB 的动态 shared memory 需要显式开启
            self.check(self.lib.cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes),
                       "cuFuncSetAttribute")
        return fn

    def launch(self, fn, grid, num_threads, smem_bytes, stream, args):
        # kernelParams 是 void*[]，每一项指向一个参数的值
        params = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
        self.check(self.lib.cuLaunchKernel(fn, *grid, num_threads, 1, 1, smem_bytes, stream, params, None),
                   "cuLaunchKernel")


_driver = None


def driver():
    global _driver
    if _driver is None:
        _driver = CudaDriver()
    return _driver
