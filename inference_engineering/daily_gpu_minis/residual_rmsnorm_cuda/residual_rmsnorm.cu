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
    // lowest hanging fruit from first kernel is to not calculate the sum of squares across D
    // in each thread, we compute square in each thread, write to smem, then one thread does the sums
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

__global__ void residual_rmsnorm_3(
    const __half* X,
    const __half* R,
    const __half* W,
    __half* U,
    __half* Y,
    int M,
    int D,
    float eps
) {
    // tree reduction of the summing, better SM utilization
    // oh also read u one less time lol
    uint threadPos = blockIdx.x * blockDim.x + threadIdx.x;
    uint tid = threadIdx.x;

    uint tRow = threadPos / D;
    uint tCol = threadPos % D;

    float u = 0.f;
    extern __shared__ float sqs[];  // dynamic smem using extern, need to allocate at launch

    if (tRow < M && tCol < D) {
        u = __half2float(X[tRow * D + tCol] + R[tRow * D + tCol]);  // torch adds in half
        U[tRow * D + tCol] = __float2half(u);
        sqs[tid] = u * u;
    }

    __syncthreads();

    uint stride = 1;
    while (stride < D) {
        stride <<= 1;  // multiply by power of 2 until it is largest after D
    }
    for (stride >>= 1; stride > 0; stride >>= 1) {  // starts at 512, 256, ... , 1, 0
        if (tid < stride && tid + stride < D) {  // 0-255 + 512-767
            sqs[tid] += sqs[tid + stride];
        }
        __syncthreads();
    }

    float sumSq = sqs[0];
    if (tRow < M && tCol < D) {
        float rstd = rsqrt(1.f/D * sumSq + eps);  // reciprocal std, std cuz we are basically doing sqrt(var + eps)
        Y[tRow * D + tCol] = __float2half(u * rstd * __half2float(W[tCol]));
    }
}

__global__ void residual_rmsnorm_4(
    const __half* X,
    const __half* R,
    const __half* W,
    __half* U,
    __half* Y,
    int M,
    int D,
    float eps
) {
    // warp shuffling
    uint threadPos = blockIdx.x * blockDim.x + threadIdx.x;
    uint tid = threadIdx.x;

    uint tRow = threadPos / D;
    uint tCol = threadPos % D;

    float u = 0.f;
    extern __shared__ float sqs[];  // dynamic smem using extern, need to allocate at launch

    if (tRow < M && tCol < D) {
        u = __half2float(X[tRow * D + tCol] + R[tRow * D + tCol]);  // torch adds in half
        U[tRow * D + tCol] = __float2half(u);
        sqs[tid] = u * u;
    }
    __syncthreads();

    uint stride = 1;
    while (stride < D) {
        stride <<= 1;  // multiply by power of 2 until it is largest after D
    }
    for (stride >>= 1; stride >= 32; stride >>= 1) {  // starts at 512, 256, ... , 1, 0
        if (tid < stride && tid + stride < D) {  // 0-255 + 512-767
            sqs[tid] += sqs[tid + stride];
        }
        __syncthreads();
    }

    // warp shuffling, one warp's threads can access each others' registers
    float val = 0.f;
    if (tid < 32) {  // warp 0 only
        val = sqs[tid];  // put their values into register `val` instead of smem
        for (int offset = 16; offset > 0; offset >>=1) {
            // Every single lane in the warp (due to mask 0xffffffff) takes +offset lane (down)
            // its register `val` and loads it into other. So lane 0 will take lane 16's val
            // and load it into other (new register). lane 31 will try to load lane 47 and fail (off the warp) so by default its `other` becomes its own `val`.
            float other = __shfl_down_sync(0xffffffff, val, offset);
            if (tid + offset < 32) {
                val += other;
            }
        }
    }
    if (tid == 0) {
        sqs[0] = val;
    }
    __syncthreads();

    float sumSq = sqs[0];
    if (tRow < M && tCol < D) {
        float rstd = rsqrt(1.f/D * sumSq + eps);  // reciprocal std, std cuz we are basically doing sqrt(var + eps)
        Y[tRow * D + tCol] = __float2half(u * rstd * __half2float(W[tCol]));
    }
}

__global__ void residual_rmsnorm_5(
    const __half* X,
    const __half* R,
    const __half* W,
    __half* U,
    __half* Y,
    int M,
    int D,
    float eps
) {
    // warp shuffling
    uint threadPos = blockIdx.x * blockDim.x + threadIdx.x;
    uint tid = threadIdx.x;
    uint lane = tid % 32;
    uint wid = tid / 32;

    uint tRow = threadPos / D;
    uint tCol = threadPos % D;

    float u = 0.f;
    float wsq = 0.f;  // this warp's square
    if (tRow < M && tCol < D) {
        u = __half2float(X[tRow * D + tCol] + R[tRow * D + tCol]);  // torch adds in half
        U[tRow * D + tCol] = __float2half(u);
        wsq = u * u;
    }

    // warp shuffling for all 24 warps, one warp's threads can access each others' registers
    for (int offset = 16; offset > 0; offset >>=1) {
        float other = __shfl_down_sync(0xffffffff, wsq, offset);
        if (lane + offset < 32) {
            wsq += other;
        }
    }

    // the first thread in each warp has the warp square sum
    __shared__ float wsqs[32];
    if (lane == 0) {
        wsqs[wid] = wsq;
    }
    __syncthreads();

    // one more warp shuffle to sum these 24 values
    if (wid == 0) {
        float val = (lane < 24) ? wsqs[lane] : 0.f;  // pad values 24..31 to 0s
        for (int offset = 16; offset > 0; offset >>=1) {
            float other = __shfl_down_sync(0xffffffff, val, offset);
            if (lane + offset < 32) {
                val += other;
            }
        }
        if (lane == 0) {
            wsqs[0] = val;
        }
    }
    __syncthreads();

    float sumSq = wsqs[0];
    if (tRow < M && tCol < D) {
        float rstd = rsqrt(1.f/D * sumSq + eps);  // reciprocal std, std cuz we are basically doing sqrt(var + eps)
        Y[tRow * D + tCol] = __float2half(u * rstd * __half2float(W[tCol]));
    }
}

__global__ void residual_rmsnorm_6(
    const __half* X,
    const __half* R,
    const __half* W,
    __half* U,
    __half* Y,
    int M,
    int D,
    float eps
) {
    // warp shuffling
    uint threadPos = blockIdx.x * blockDim.x + threadIdx.x;
    uint tid = threadIdx.x;
    uint lane = tid % 32;
    uint wid = tid / 32;

    uint tRow = threadPos / D;
    uint tCol = threadPos % D;

    // sum of 3 squares
    float sq = 0.f;
    while (uint i = tCol; i < D; i += 256) {
        if (tRow < M && i < D) {
            u = __half2float(X[tRow * D + i] + R[tRow * D + i]);  // torch adds in half
            U[tRow * D + i] = __float2half(u);
            sq += u * u;
        }
    }

    // warp shuffling for 8 warps, one warp's threads can access each others' registers
    for (int offset = 16; offset > 0; offset >>=1) {
        float other = __shfl_down_sync(0xffffffff, sq, offset);
        if (lane + offset < 32) {
            sq += other;
        }
    }

    // the first thread in each warp has the warp square sum
    __shared__ float wsqs[32];
    if (lane == 0) {
        wsqs[wid] = wsq;
    }
    __syncthreads();

    // one more warp shuffle to sum these 24 values
    if (wid == 0) {
        float val = (lane < 24) ? wsqs[lane] : 0.f;  // pad values 24..31 to 0s
        for (int offset = 16; offset > 0; offset >>=1) {
            float other = __shfl_down_sync(0xffffffff, val, offset);
            if (lane + offset < 32) {
                val += other;
            }
        }
        if (lane == 0) {
            wsqs[0] = val;
        }
    }
    __syncthreads();

    float sumSq = wsqs[0];
    if (tRow < M && tCol < D) {
        float rstd = rsqrt(1.f/D * sumSq + eps);  // reciprocal std, std cuz we are basically doing sqrt(var + eps)
        Y[tRow * D + tCol] = __float2half(u * rstd * __half2float(W[tCol]));
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
    float eps,
    int version
) {
    dim3 grid(M);
    dim3 block(D);  // only works now cuz D < 1024 thread per block limit (important due to RMSNorm dependency on whole row for RMS)
    size_t smem = D * sizeof(float);
    switch (version) {
        case 1:
            residual_rmsnorm_1<<<grid, block>>>(X, R, W, U, Y, M, D, eps);
            break;
        case 2:
            residual_rmsnorm_2<<<grid, block, smem>>>(X, R, W, U, Y, M, D, eps);
            break;
        case 3:
            residual_rmsnorm_3<<<grid, block, smem>>>(X, R, W, U, Y, M, D, eps);
            break;
        case 4:
            residual_rmsnorm_4<<<grid, block, smem>>>(X, R, W, U, Y, M, D, eps);
            break;
        case 5:
            residual_rmsnorm_5<<<grid, block>>>(X, R, W, U, Y, M, D, eps);
            break;
        case 6:
            dim3 grid(M);
            dim3 block(256);  // now we fit 8 blocks per SM, more occupancy than the 1536
            residual_rmsnorm_6<<<grid, block>>>(X, R, W, U, Y, M, D, eps);
            break;
        default:
            std::fprintf(stderr, "unknown kernel version %d\n", version);
            std::exit(1);
    }
}
