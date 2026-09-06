#include <cstdio>
#include <cstdlib>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// M: seq_len, usually batch_size * sequence_length.
// D: hidden dimension size
//
// X, R ∈ R^(M×D): sublayer output and residual stream.
// W ∈ R^D: learned RMSNorm weight shared by every token.
//
// For each token row m:
//
//   U[m,j] = X[m,j] + R[m,j]                         // residual addition
//   rms[m] = sqrt((1/D) * Σ_j U[m,j]^2 + epsilon)   // row-wise RMS
//   Y[m,j] = (U[m,j] / rms[m]) * W[j]               // normalized output
//
// The fused kernel outputs U for the continuing residual stream and Y as the
// pre-normalized input to the next attention or MLP sublayer.

void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(1);
    }
}

__global__ void residual_rmsnorm_1(
    const __half* X,
    const __half* R,
    const __half* W,
    __half* U,
    __half* Y,
    int M,
    int D,
    float eps
) {
    uint threadPos = blockIdx.x * blockDim.x + threadIdx.x;

    uint tRow = threadPos / D;
    uint tCol = threadPos % D;

    if (tRow < M && tCol < D) {
        U[tRow * D + tCol] = X[tRow * D + tCol] + R[tRow * D + tCol];
    }

    __syncthreads();  // make sure the entire row of U is done (we have this outside of the if so that OOB threads also hit syncthreads)

    if (tRow < M && tCol < D) {
        float sumSq = 0.f;
        for (int j = 0; j < D; ++j) {
            float u = __half2float(U[tRow * D + j]);
            sumSq += u * u;
        }
        float rms = sqrtf(1.f/D * sumSq + eps);
        float u = __half2float(U[tRow * D + tCol]);
        Y[tRow * D + tCol] = __float2half(u / rms * __half2float(W[tCol]));
    }
}

__global__ void residual_rmsnorm_2(
    const __half* X,
    const __half* R,
    const __half* W,
    __half* U,
    __half* Y,
    int M,
    int D,
    float eps
) {
    uint threadPos = blockIdx.x * blockDim.x + threadIdx.x;
    uint tid = threadIdx.x;

    uint tRow = threadPos / D;
    uint tCol = threadPos % D;

    if (tRow < M && tCol < D) {
        U[tRow * D + tCol] = X[tRow * D + tCol] + R[tRow * D + tCol];
    }

    __syncthreads();  // make sure the entire row of U is done (we have this outside of the if so that OOB threads also hit syncthreads)

    extern __shared__ float sqs[];  // dynamic smem using extern, need to allocate at launch

    if (tRow < M && tCol < D) {
        // since block has D threads, we can just use tid to know which col of U each thread needs to square
        float u = __half2float(U[tRow * D + tid]);
        sqs[tid] = u * u;
    }

    __syncthreads();

    if (tRow < M && tCol < D && tid == 0) {
        float sumSq = 0.f;
        for (int i = 0; i < D; ++i) {
            sumSq += sqs[i];
        }
        sqs[0] = sumSq;
    }

    __syncthreads();

    float sumSq = sqs[0];
    if (tRow < M && tCol < D) {
        float rms = sqrtf(1.f/D * sumSq + eps);
        float u = __half2float(U[tRow * D + tCol]);
        Y[tRow * D + tCol] = __float2half(u / rms * __half2float(W[tCol]));
    }
}

extern "C" void residual_rmsnorm_cuda(
    const __half* X,
    const __half* R,
    const __half* W,
    __half* U,
    __half* Y,
    int M,
    int D,
    float eps
) {
    dim3 grid(M);
    dim3 block(D);  // only works now cuz D < 1024 thread per block limit (important due to RMSNorm dependency on whole row for RMS)
    // residual_rmsnorm_1<<<grid, block>>>(X, R, W, U, Y, M, D, eps);
    residual_rmsnorm_2<<<grid, block, D * sizeof(float)>>>(X, R, W, U, Y, M, D, eps);
}
