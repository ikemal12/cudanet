#include <cuda_runtime.h>
#include <iostream>

__global__ void conv2d_kernel(const float* input, const float *kernel, float* output,
                             int inputWidth, int inputHeight, int kernelWidth, int kernelHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int outputWidth = inputWidth - kernelWidth + 1;
    int outputHeight = inputHeight - kernelHeight + 1;

    if (x < outputWidth && y < outputHeight) {
        float sum = 0.0f;

        for (int ky = 0; ky < kernelHeight; ++ky) {
            for (int kx = 0; kx < kernelWidth; ++kx) {
                int ix = x + kx;
                int iy = y + ky;
                sum += input[iy * inputWidth + ix] * kernel[ky * kernelWidth + kx];
            }
        }
        output[y * outputWidth + x] = sum;
    }
}


void conv2d(const float* input, const float* kernel, float* output,
            int inputWidth, int inputHeight, int kernelWidth, int kernelHeight) {
    
    int outputWidth = inputWidth - kernelWidth + 1;
    int outputHeight = inputHeight - kernelHeight + 1;

    float *d_input, *d_kernel, *d_output;

    cudaMalloc(&d_input, sizeof(float) * inputWidth * inputHeight);
    cudaMalloc(&d_kernel, sizeof(float) * kernelWidth * kernelHeight);
    cudaMalloc(&d_output, sizeof(float) * outputWidth * outputHeight);

    cudaMemcpy(d_input, input, sizeof(float) * inputWidth * inputHeight, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, sizeof(float) * kernelWidth * kernelHeight, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x,
                  (outputHeight + blockSize.y - 1) / blockSize.y);

    conv2d_kernel<<<gridSize, blockSize>>>(d_input, d_kernel, d_output,
                                           inputWidth, inputHeight,
                                           kernelWidth, kernelHeight);

    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, sizeof(float) * outputWidth * outputHeight, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}