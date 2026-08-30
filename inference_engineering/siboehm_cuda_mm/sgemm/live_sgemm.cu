#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

constexpr int N = 4096;  // compiletime vs runtime var
constexpr int REPS = 10;

void cudaCheck(cudaError_t e) {
  // this function allows us to avoid silent errors
  if (e) {
    printf("%s\n", cudaGetErrorString(e));
    exit(1);
  }
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
  // print A10 properties
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("%s  CC %d.%d  SMs=%d\n", p.name, p.major, p.minor,
         p.multiProcessorCount);
  printf("N=%d  C = A @ B  FP32\n", N);

  const size_t bytes = (size_t)N * N * sizeof(float);
  float *A, *B, *C;
  float *hA = (float *)malloc(bytes);
  float *hB = (float *)malloc(bytes);
  cudaCheck(cudaMalloc(&A, bytes));
  cudaCheck(cudaMalloc(&B, bytes));
  cudaCheck(cudaMalloc(&C, bytes));

  srand(42);
  for (int i = 0; i < N * N; i++) {
    hA[i] = (float)(rand() % 5) + 0.01f * (rand() % 5);
    hB[i] = (float)(rand() % 5) + 0.01f * (rand() % 5);
  }
  cudaCheck(cudaMemcpy(A, hA, bytes, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(B, hB, bytes, cudaMemcpyHostToDevice));
	
  float ms = bench(launch_cublas, A, B, C);
  double flops = 2.0 * N * N * N;
  printf("cublas %7.3f ms  %7.0f GFLOP/s\n", ms, flops / (ms * 1e6));

  free(hA);
  free(hB);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  return 0;
}