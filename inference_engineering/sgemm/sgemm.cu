#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
constexpr int N = 4096;
constexpr int REPS = 10;
constexpr int TILESIZE = 32;

void cudaCheck(cudaError_t e) {
    // this function allows us to avoid silent errors
    if (e) {
        printf("%s\n", cudaGetErrorString(e));
        exit(1);
    }
}

__global__ void sgemm_naive(
    int M, int N, int K, float alpha,
    const float *A, const float *B,
    float beta, float *C
) {
    // which row of C and which col of C this thread is responsible for
    // A is MxK and B is KxN and C is MxN
    uint threadRow = blockIdx.x * blockDim.x + threadIdx.x;
    uint threadCol = blockIdx.y * blockDim.y + threadIdx.y;

    float tmp = 0.f;
    if (threadRow < M && threadCol < N) {
        for (int k = 0; k < K; k++) {
            tmp += A[(threadRow * K) + k] * B[(k * N) + threadCol];
        }
    }
    C[(threadRow * N) + threadCol] = alpha * tmp + beta * C[(threadRow * N) + threadCol];
}

void launch_naive(float *A, float *B, float *C) {
    dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(N, 32), 1); // 16,394 blocks 128x128 blocks
    dim3 block(32, 32, 1);
    sgemm_naive<<<grid, block>>>(N, N, N, 1.f, A, B, 0.f, C);
}

__global__ void sgemm_gmem_coalesce(
    int M, int N, int K, float alpha,
    const float *A, const float *B,
    float beta, float *C
) {
    // which row of C and which col of C this thread is responsible for
    // A is MxK and B is KxN and C is MxN
    uint threadRow = blockIdx.x * blockDim.x + threadIdx.y;
    uint threadCol = blockIdx.y * blockDim.y + threadIdx.x;

    float tmp = 0.f;
    if (threadRow < M && threadCol < N) {
        for (int k = 0; k < K; k++) {
            tmp += A[(threadRow * K) + k] * B[(k * N) + threadCol];
        }
    }
    C[(threadRow * N) + threadCol] = alpha * tmp + beta * C[(threadRow * N) + threadCol];
}

void launch_gmem_coalesce(float *A, float *B, float *C) {
    dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(N, 32), 1); // 16,394 blocks 128x128 blocks
    dim3 block(32, 32, 1);
    sgemm_gmem_coalesce<<<grid, block>>>(N, N, N, 1.f, A, B, 0.f, C);
}

__global__ void sgemm_smem(
    int M, int N, int K, float alpha,
    const float *A, const float *B,
    float beta, float *C
) {
    uint localRow = threadIdx.y;
    uint localCol = threadIdx.x;
    uint tileRow = blockIdx.y * blockDim.y;
    uint tileCol = blockIdx.x * blockDim.x;
    uint threadRow = tileRow + localRow;
    uint threadCol = tileCol + localCol;

    __shared__ float As[TILESIZE][TILESIZE];
    __shared__ float Bs[TILESIZE][TILESIZE];

    if (threadRow < M && threadCol < N) {
        float tmp = 0.f;

        for (int k = 0; k < K; k+=TILESIZE) {
            As[localRow][localCol] = A[threadRow * K + k + localCol];
            Bs[localRow][localCol] = B[(k + localRow) * N + threadCol];

            __syncthreads();

            for (int i = 0; i < TILESIZE; i++) {
                tmp += As[localRow][i] * Bs[i][localCol];
            }

            __syncthreads();
        }

        C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
    }
}

void launch_smem(float *A, float *B, float *C) {
    dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(N, 32), 1); // 16,394 blocks 128x128 blocks
    dim3 block(32, 32, 1);
    sgemm_smem<<<grid, block>>>(N, N, N, 1.f, A, B, 0.f, C);
}

__global__ void sgemm_smem_non_coalesce(
    int M, int N, int K, float alpha,
    const float *A, const float *B,
    float beta, float *C
) {
    uint localRow = threadIdx.x;
    uint localCol = threadIdx.y;
    uint tileRow = blockIdx.y * blockDim.y;
    uint tileCol = blockIdx.x * blockDim.x;
    uint threadRow = tileRow + localRow;
    uint threadCol = tileCol + localCol;

    __shared__ float As[TILESIZE][TILESIZE];
    __shared__ float Bs[TILESIZE][TILESIZE];

    if (threadRow < M && threadCol < N) {
        float tmp = 0.f;

        for (int k = 0; k < K; k+=TILESIZE) {
            As[localCol][localRow] = A[threadRow * K + k + localCol];
            Bs[localCol][localRow] = B[(k + localRow) * N + threadCol];

            __syncthreads();

            for (int i = 0; i < TILESIZE; i++) {
                tmp += As[i][localRow] * Bs[localCol][i];
            }

            __syncthreads();
        }

        C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
    }
}

void launch_smem_non_coalesce(float *A, float *B, float *C) {
    dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(N, 32), 1); // 16,394 blocks 128x128 blocks
    dim3 block(32, 32, 1);
    sgemm_smem_non_coalesce<<<grid, block>>>(N, N, N, 1.f, A, B, 0.f, C);
}

void launch_cublas(float *A, float *B, float *C) {
    // cublas is column-major whereas we store A, B, C row-major, so we need to swap B & A position NxK and KxM so C is NxM (but stored column major result is same as C MxN stored row major)
    // we force cublas to use fp32 instead of tf32 (tensor cores is cheating)
    static cublasHandle_t h = nullptr;  // stores some metadata for cublas
    if (!h)
        cublasCreate(&h);
    float alpha = 1.f, beta = 0.f;
    cublasGemmEx(
        h, CUBLAS_OP_N, CUBLAS_OP_N,  // session, and telling cublas to not apply transpose to A/B
        N, N, N,  // M, N, K
        &alpha, B, CUDA_R_32F, N, A, CUDA_R_32F, N, &beta, C, CUDA_R_32F, N,  // ptr, dtype, leading dim
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT  // fp32 not tf32, let cublas pick kernel
    );
}

float bench(void (*fn)(float *, float *, float *), float *A, float *B, float *C) {
    fn(A, B, C);  // warmup
    cudaCheck(cudaDeviceSynchronize());
    cudaEvent_t start, stop;  // timestamp on GPU clock not CPU
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < REPS; i++)
        fn(A, B, C);
    cudaEventRecord(stop);
    cudaCheck(cudaEventSynchronize(stop));  // tell CPU to wait for GPU to finish
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms / REPS;
}

int main() {
    // print H100 properties
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("%s  CC %d.%d  SMs=%d  N=%d  FP32\n",
           p.name, p.major, p.minor, p.multiProcessorCount, N);
    printf("  warp=%d  %d thr/SM  %d thr/block  %zuKB smem/SM  %zuKB smem/block\n",
           p.warpSize, p.maxThreadsPerMultiProcessor, p.maxThreadsPerBlock,
           p.sharedMemPerMultiprocessor / 1024, p.sharedMemPerBlock / 1024);

    const size_t bytes = (size_t)N * N * sizeof(float);
    float *A, *B, *C, *C_ref;
    float *hA = (float *)malloc(bytes);
    float *hB = (float *)malloc(bytes);
    float *hC = (float *)malloc(bytes);
    float *hCref = (float *)malloc(bytes);
    cudaCheck(cudaMalloc(&A, bytes));
    cudaCheck(cudaMalloc(&B, bytes));
    cudaCheck(cudaMalloc(&C, bytes));
    cudaCheck(cudaMalloc(&C_ref, bytes));

    srand(42);
    for (int i = 0; i < N * N; i++) {
        hA[i] = (float)(rand() % 5) + 0.01f * (rand() % 5);
        hB[i] = (float)(rand() % 5) + 0.01f * (rand() % 5);
    }
    cudaCheck(cudaMemcpy(A, hA, bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(B, hB, bytes, cudaMemcpyHostToDevice));

    launch_cublas(A, B, C_ref);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(hCref, C_ref, bytes, cudaMemcpyDeviceToHost));

    float ms_cublas = bench(launch_cublas, A, B, C);
    double flops = 2.0 * N * N * N;
    printf("\n%-16s %-3s %9s %7s %9s %12s\n",
           "kernel", "ok", "max_diff", "ms", "GFLOP/s", "vs cublas");
    printf("%-16s %-3s %9s %7.3f %9.0f %12s\n",
           "cublas", "-", "-", ms_cublas, flops / (ms_cublas * 1e6), "-");

    struct Kernel {
        void (*fn)(float *, float *, float *);
        const char *name;
    };
    Kernel kernels[] = {
        {launch_naive, "naive"},
        {launch_gmem_coalesce, "gmem_coalesce"},
        {launch_smem, "smem"},
        {launch_smem_non_coalesce, "smem_non_coalesce"}
    };

    int rc = 0;
    // CHANGE THIS
    for (Kernel k : kernels) {
        k.fn(A, B, C);
        cudaCheck(cudaDeviceSynchronize());  // wait for GPU work to finish
        cudaCheck(cudaMemcpy(hC, C, bytes, cudaMemcpyDeviceToHost));  // GPU -> CPU

        int mismatches = 0;
        float max_diff = 0.f;
        for (int i = 0; i < N * N; i++) {
            float d = fabsf(hC[i] - hCref[i]);
            if (d > max_diff)
                max_diff = d;
            if (d > 1e-2f)
                mismatches++;
        }
        float ms = bench(k.fn, A, B, C);
        printf("%-20s %-3s %9.4f %7.3f %9.0f %11.1f%%\n",
               k.name, mismatches ? "no" : "yes", max_diff, ms,
               flops / (ms * 1e6), 100.0 * ms_cublas / ms);
        if (mismatches)
            rc = 1;
    }

    free(hA);
    free(hB);
    free(hC);
    free(hCref);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(C_ref);
    return rc;
}
