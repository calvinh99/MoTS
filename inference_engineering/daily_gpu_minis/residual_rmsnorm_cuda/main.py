from pathlib import Path
import modal

cuda_source = Path(__file__).with_name("residual_rmsnorm.cu")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .pip_install("torch")
    .add_local_file(
        str(cuda_source),
        remote_path="/root/residual_rmsnorm.cu",
    )
)

app = modal.App("daily-gpu-mini-residual-rmsnorm")


@app.function(image=image, gpu="H100", timeout=60)
def run():
    import ctypes
    import subprocess

    import torch
    import torch.nn.functional as F

    so = "/tmp/residual_rmsnorm.so"
    subprocess.run(
        [
            "nvcc", "-O3", "-std=c++17", "-arch=sm_90",
            "-lineinfo", "--shared", "-Xcompiler", "-fPIC",
            "/root/residual_rmsnorm.cu", "-o", so,
        ],
        check=True,
    )
    lib = ctypes.CDLL(so)
    lib.residual_rmsnorm_cuda.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
    ]
    lib.residual_rmsnorm_cuda.restype = None

    M, D, eps, reps = 4096, 768, 1e-5, 100
    versions = (1, 2, 3, 4, 5)
    torch.manual_seed(42)
    X = torch.randn(M, D, device="cuda", dtype=torch.float16)
    R = torch.randn(M, D, device="cuda", dtype=torch.float16)
    W = torch.rand(D, device="cuda", dtype=torch.float16) + 0.5
    U = torch.empty_like(X)
    Y = torch.empty_like(X)

    def launch(version):
        lib.residual_rmsnorm_cuda(
            ctypes.c_void_p(X.data_ptr()),
            ctypes.c_void_p(R.data_ptr()),
            ctypes.c_void_p(W.data_ptr()),
            ctypes.c_void_p(U.data_ptr()),
            ctypes.c_void_p(Y.data_ptr()),
            M,
            D,
            ctypes.c_float(eps),
            version,
        )

    U_ref = X + R
    Y_ref = F.rms_norm(U_ref, (D,), weight=W, eps=eps)

    for version in versions:
        U.zero_()
        Y.zero_()
        launch(version)
        torch.cuda.synchronize()
        print(
            f"k{version}  U max abs err  {(U - U_ref).abs().max().item():.4f}  "
            f"Y max abs err  {(Y - Y_ref).abs().max().item():.4f}"
        )

    def bench(fn):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(reps):
            fn()
        stop.record()
        torch.cuda.synchronize()
        return start.elapsed_time(stop) / reps

    def torch_fn():
        F.rms_norm(X + R, (D,), weight=W, eps=eps)

    # Useful FLOPs, not issued instructions:
    #   MD add (U=X+R) + 2MD (square+sum) + M (*1/D, +eps, sqrt)
    #   + MD (div by rms) + MD (mul by W)  =  5MD + 3M
    flops = M * (5 * D + 3)
    ms_torch = bench(torch_fn)
    print(f"torch   {ms_torch:7.3f} ms  {flops / (ms_torch * 1e6):7.1f} GFLOP/s")
    for version in versions:
        ms = bench(lambda v=version: launch(v))
        print(
            f"k{version}     {ms:7.3f} ms  {flops / (ms * 1e6):7.1f} GFLOP/s  "
            f"({100.0 * ms_torch / ms:.1f}% of torch)"
        )


@app.local_entrypoint()
def main():
    run.remote()
