import subprocess
from pathlib import Path

import modal

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    ).entrypoint([])  # eliminates NVDA license spam from nvidia_entrypoint.sh
)

# image will be the default docker container for funcs that run on this app
app = modal.App("sgemm", image=image)

@app.function(gpu="A10", timeout=60)
def go(src: str):  # the function that runs on the A10, src is the kernel source code
    Path("/tmp/sgemm.cu").write_text(src)  # dump the kernel
    subprocess.run(
        [
            "nvcc",  # NVDA cuda compiler
            "-O3",  # optimize compiled binary
            "-arch=sm_86",  # A10 architecture
            "--ptxas-options=-v",  # verbose, print registers & SMEM usage
            "-lcublas",  # link cuBLAS so our code can load the cuBLAS sgemm kernel
            "/tmp/sgemm.cu",  # what is getting compiled
            "-o",
            "/tmp/sgemm"  # the compiled binary
        ],
        check=True  # if nvcc fails raise error
    )
    subprocess.run(["/tmp/sgemm"], check=True)

@app.local_entrypoint()  # need this for modal cli
def main():
    go.remote((Path(__file__).parent / "sgemm.cu").read_text())  # find sgemm.cu in the same dir