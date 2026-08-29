#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))  // not a function, compiler replaces instance of CEIL_DIV with the defined expression
constexpr int N = 4096;  // compiletime vs runtime var
constexpr int REPS = 10;

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
  // A is MxK, B is KxN, C is MxN
  // C = alpha * (A @ B) + beta * C
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;  // simple, we map the actual location in 4096x4096 C to the actual block & thread indices, no reinterpretation needed
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < M && y < N) {  // when matrix shapes aren't perfectly dividable by 32 or whatever blockDim then we may end up with out of bounds
    float tmp = 0.0f;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

void launch_naive(float *A, float *B, float *C) {
  dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(N, 32));  // 128 x 128 = 16,384 blocks, each block 1024 threads
  dim3 block(32, 32);
  sgemm_naive<<<grid, block>>>(N, N, N, 1.f, A, B, 0.f, C);
}

__global__ void sgemm_gmem_coalesce(
  int M, int N, int K, float alpha,
  const float *A, const float *B,
  float beta, float *C
) {
  // Warps are formed along threadIdx.x, so we need to load consecutive memory from A & B as x/y increases
  // in naive kernel, when x increases (think multiple threads in same warp) each thread loads a 
  // different row of A & the same col of B (same y value) and write to different rows of C.
  // We cannot coalesce the read from A and write to C as they are across diff rows (non-consecutive).
  // view siboehm blog for nice visual.
  // Basically we want to share the same row of A, but have each thread use diff cols of B. Then
  // each warp of threads are stepping through one row at a time (ik it's confusing so look at the siboehm blog visual, each thread is stepping through col but as a warp of threads it's a row vs originally each thread is stepping through a row but the entire warp is a col at each step).
  // A is MxK, B is KxN, C is MxN
  // C = alpha * (A @ B) + beta * C
  const uint x = blockIdx.x * blockDim.x + threadIdx.y;
  const uint y = blockIdx.y * blockDim.y + threadIdx.x;  // it's the same 32x32 block, but we make threads in the same warp (threadIdx.x increments) access the same A row (threadIdx.y) and different B cols and write to C as a row (if you think of all threads together)
  if (x < M && y < N) {  // when matrix shapes aren't perfectly dividable by 32 or whatever blockDim then we may end up with out of bounds
    float tmp = 0.0f;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

void launch_gmem_coalesce(float *A, float *B, float *C) {
  dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(N, 32));  // 128 x 128 = 16,384 blocks, each block 1024 threads
  dim3 block(32, 32);
  sgemm_gmem_coalesce<<<grid, block>>>(N, N, N, 1.f, A, B, 0.f, C);
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

int compare(
  void (*fn1)(float *, float *, float *),
  const char *fn1_name,
  void (*fn2)(float *, float *, float *),
  const char *fn2_name,
  float *A, float *B, float *C, float *C_ref,
  float *hC, float *hCref, size_t bytes
) {
  fn1(A, B, C_ref);
  cudaCheck(cudaDeviceSynchronize());  // wait for GPU work to finish
  fn2(A, B, C);
  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaMemcpy(hC, C, bytes, cudaMemcpyDeviceToHost));  // GPU -> CPU
  cudaCheck(cudaMemcpy(hCref, C_ref, bytes, cudaMemcpyDeviceToHost));

  int mismatches = 0;
  float max_diff = 0.f;
  for (int i = 0; i < N * N; i++) {
    float d = fabsf(hC[i] - hCref[i]);
    if (d > max_diff)
      max_diff = d;
    if (d > 1e-2f)
      mismatches++;
  }
  printf("%s vs %s: mismatches=%d  max_diff=%.4f\n", fn1_name, fn2_name,
         mismatches, max_diff);

  float ms_fn1 = bench(fn1, A, B, C);
  float ms_fn2 = bench(fn2, A, B, C);
  double flops = 2.0 * N * N * N;
  printf("%-20s %7.3f ms  %7.0f GFLOP/s\n", fn1_name, ms_fn1,
         flops / (ms_fn1 * 1e6));
  printf("%-20s %7.3f ms  %7.0f GFLOP/s  (%.1f%% of %s)\n", fn2_name, ms_fn2,
         flops / (ms_fn2 * 1e6), 100.0 * ms_fn1 / ms_fn2, fn1_name);
  return mismatches ? 1 : 0;
}

int main() {
  // print A10 properties
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("%s  CC %d.%d  SMs=%d\n", p.name, p.major, p.minor,
         p.multiProcessorCount);
  printf("N=%d  C = A @ B  FP32\n", N);

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

  int rc = 0;
  rc |= compare(launch_cublas, "cublas", launch_naive, "naive",
                A, B, C, C_ref, hC, hCref, bytes);
  rc |= compare(launch_cublas, "cublas", launch_gmem_coalesce, "gmem_coalesce",
                A, B, C, C_ref, hC, hCref, bytes);

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