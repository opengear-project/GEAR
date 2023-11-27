#include <torch/torch.h>
#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel to perform matrix multiplication
__global__ void matrixMulKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

int main() {
    // Dimensions of matrices
    int M = 3, N = 4, K = 2;

    // Create input tensors using PyTorch
    torch::Tensor inputA = torch::rand({M, N});
    torch::Tensor inputB = torch::rand({N, K});

    // Allocate memory on the GPU
    float* d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * K * sizeof(float));
    cudaMalloc((void**)&d_C, M * K * sizeof(float));

    // Copy input tensors from CPU to GPU
    cudaMemcpy(d_A, inputA.data_ptr<float>(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, inputB.data_ptr<float>(), N * K * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((K + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

    // Copy the result back from GPU to CPU
    torch::Tensor output = torch::from_blob(d_C, {M, K});

    // Print the output tensor
    std::cout << "Output Tensor:" << std::endl;
    std::cout << output << std::endl;

    // Clean up GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}