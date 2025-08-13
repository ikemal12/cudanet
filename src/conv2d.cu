#include <cuda_runtime.h>
#include <iostream>

__global__ void conv2d_kernel(const float* input, const float *kernel, float* output,
                             int inputWidth, int inputHeight, int kernelWidth, int kernelHeight) {
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int blockDimX = blockDim.x;
    int blockDimY = blockDim.y;
    
    int outputX = bx * blockDimX + tx;
    int outputY = by * blockDimY + ty;
    
    int outputWidth = inputWidth - kernelWidth + 1;
    int outputHeight = inputHeight - kernelHeight + 1;
    
    extern __shared__ float sharedInput[];
    __shared__ float sharedKernel[49]; 
    
    int sharedWidth = blockDimX + kernelWidth - 1;
    int sharedHeight = blockDimY + kernelHeight - 1;
    
    int kernelSize = kernelWidth * kernelHeight;
    int flatThreadId = ty * blockDimX + tx;
    if (flatThreadId < kernelSize) {
        sharedKernel[flatThreadId] = kernel[flatThreadId];
    }
    
    int inputStartX = bx * blockDimX;
    int inputStartY = by * blockDimY;
    
    int elementsToLoad = sharedWidth * sharedHeight;
    int threadsPerBlock = blockDimX * blockDimY;
    int loadsPerThread = (elementsToLoad + threadsPerBlock - 1) / threadsPerBlock;
    
    for (int i = 0; i < loadsPerThread; i++) {
        int linearIdx = flatThreadId + i * threadsPerBlock;
        if (linearIdx < elementsToLoad) {
            int sharedY = linearIdx / sharedWidth;
            int sharedX = linearIdx % sharedWidth;
            
            int globalX = inputStartX + sharedX;
            int globalY = inputStartY + sharedY;
            
            if (globalX < inputWidth && globalY < inputHeight) {
                sharedInput[linearIdx] = input[globalY * inputWidth + globalX];
            } else {
                sharedInput[linearIdx] = 0.0f; 
            }
        }
    }
    
    __syncthreads();
    
    if (outputX < outputWidth && outputY < outputHeight) {
        float sum = 0.0f;
        
        for (int ky = 0; ky < kernelHeight; ++ky) {
            for (int kx = 0; kx < kernelWidth; ++kx) {
                int sharedX = tx + kx;
                int sharedY = ty + ky;
                int sharedIdx = sharedY * sharedWidth + sharedX;
                int kernelIdx = ky * kernelWidth + kx;
                
                sum += sharedInput[sharedIdx] * sharedKernel[kernelIdx];
            }
        }
        output[outputY * outputWidth + outputX] = sum;
    }
}


__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // ReLU: max(0, x)
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}


__global__ void maxpool2d_kernel(const float* input, float* output,
                                int inputWidth, int inputHeight,
                                int poolSize, int stride) {
    
   
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int blockDimX = blockDim.x;
    int blockDimY = blockDim.y;
    
   
    int outputX = bx * blockDimX + tx;
    int outputY = by * blockDimY + ty;
    
    
    int outputWidth = (inputWidth - poolSize) / stride + 1;
    int outputHeight = (inputHeight - poolSize) / stride + 1;
    
    if (outputX < outputWidth && outputY < outputHeight) {
        int inputStartX = outputX * stride;
        int inputStartY = outputY * stride;

        float maxVal = -INFINITY;
        
        for (int py = 0; py < poolSize; ++py) {
            for (int px = 0; px < poolSize; ++px) {
                int inputX = inputStartX + px;
                int inputY = inputStartY + py;
                
            
                if (inputX < inputWidth && inputY < inputHeight) {
                    float val = input[inputY * inputWidth + inputX];
                    maxVal = fmaxf(maxVal, val);
                }
            }
        }
        output[outputY * outputWidth + outputX] = maxVal;
    }
}



void conv2d(const float* input, int inputHeight, int inputWidth,
            const float* kernel, int kernelHeight, int kernelWidth,
            float* output) {
    
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

    int sharedWidth = blockSize.x + kernelWidth - 1;
    int sharedHeight = blockSize.y + kernelHeight - 1;
    size_t sharedMemSize = sharedWidth * sharedHeight * sizeof(float);

    conv2d_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_input, d_kernel, d_output,
        inputWidth, inputHeight,
        kernelWidth, kernelHeight);

    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, sizeof(float) * outputWidth * outputHeight, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}


void relu(const float* input, int height, int width, float* output) {
    int size = height * width;
    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float) * size);
    cudaMalloc(&d_output, sizeof(float) * size);
    cudaMemcpy(d_input, input, sizeof(float) * size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    relu_kernel<<<gridSize, blockSize>>>(d_input, d_output, size);
    
   
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}


void maxpool2d(const float* input, int inputHeight, int inputWidth, 
               int poolSize, int stride, float* output) {
    
  
    int outputWidth = (inputWidth - poolSize) / stride + 1;
    int outputHeight = (inputHeight - poolSize) / stride + 1;
    
   
    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float) * inputWidth * inputHeight);
    cudaMalloc(&d_output, sizeof(float) * outputWidth * outputHeight);
    
   
    cudaMemcpy(d_input, input, sizeof(float) * inputWidth * inputHeight, cudaMemcpyHostToDevice);
    
   
    dim3 blockSize(16, 16);
    dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x,
                  (outputHeight + blockSize.y - 1) / blockSize.y);
    
   
    maxpool2d_kernel<<<gridSize, blockSize>>>(
        d_input, d_output,
        inputWidth, inputHeight,
        poolSize, stride);
    
    
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, sizeof(float) * outputWidth * outputHeight, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}


void conv2d_batch(const float* input, int batchSize, int inputHeight, int inputWidth,
                  const float* kernel, int kernelHeight, int kernelWidth,
                  float* output) {
    
    int outputWidth = inputWidth - kernelWidth + 1;
    int outputHeight = inputHeight - kernelHeight + 1;
    int inputSize = inputWidth * inputHeight;
    int outputSize = outputWidth * outputHeight;

    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, sizeof(float) * batchSize * inputSize);
    cudaMalloc(&d_kernel, sizeof(float) * kernelWidth * kernelHeight);
    cudaMalloc(&d_output, sizeof(float) * batchSize * outputSize);

    cudaMemcpy(d_input, input, sizeof(float) * batchSize * inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, sizeof(float) * kernelWidth * kernelHeight, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x,
                  (outputHeight + blockSize.y - 1) / blockSize.y,
                  batchSize); 

    int sharedWidth = blockSize.x + kernelWidth - 1;
    int sharedHeight = blockSize.y + kernelHeight - 1;
    size_t sharedMemSize = sharedWidth * sharedHeight * sizeof(float);

    for (int b = 0; b < batchSize; ++b) {
        conv2d_kernel<<<gridSize, blockSize, sharedMemSize>>>(
            d_input + b * inputSize, d_kernel, d_output + b * outputSize,
            inputWidth, inputHeight, kernelWidth, kernelHeight);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, sizeof(float) * batchSize * outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

void relu_batch(const float* input, int batchSize, int height, int width, float* output) {
    int totalSize = batchSize * height * width;
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float) * totalSize);
    cudaMalloc(&d_output, sizeof(float) * totalSize);
    cudaMemcpy(d_input, input, sizeof(float) * totalSize, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (totalSize + blockSize - 1) / blockSize;
    relu_kernel<<<gridSize, blockSize>>>(d_input, d_output, totalSize);
    
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, sizeof(float) * totalSize, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}

void maxpool2d_batch(const float* input, int batchSize, int inputHeight, int inputWidth,
                     int poolSize, int stride, float* output) {
    
    int outputWidth = (inputWidth - poolSize) / stride + 1;
    int outputHeight = (inputHeight - poolSize) / stride + 1;
    int inputSize = inputWidth * inputHeight;
    int outputSize = outputWidth * outputHeight;
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float) * batchSize * inputSize);
    cudaMalloc(&d_output, sizeof(float) * batchSize * outputSize);
    
    cudaMemcpy(d_input, input, sizeof(float) * batchSize * inputSize, cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x,
                  (outputHeight + blockSize.y - 1) / blockSize.y);
    
    for (int b = 0; b < batchSize; ++b) {
        maxpool2d_kernel<<<gridSize, blockSize>>>(
            d_input + b * inputSize, d_output + b * outputSize,
            inputWidth, inputHeight, poolSize, stride);
    }
    
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, sizeof(float) * batchSize * outputSize, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}