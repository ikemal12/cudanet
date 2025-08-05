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