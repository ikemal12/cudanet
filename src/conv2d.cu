#include <cuda_runtime.h>
#include <iostream>

__global__ void conv2d_kernel(const float* input, const float *kernel, float* output,
                             int inputChannels, int inputWidth, int inputHeight, 
                             int outputChannels, int kernelWidth, int kernelHeight) {
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int outputChannel = blockIdx.z;
    
    int outputX = bx * blockDim.x + tx;
    int outputY = by * blockDim.y + ty;
    
    int outputWidth = inputWidth - kernelWidth + 1;
    int outputHeight = inputHeight - kernelHeight + 1;
    
    if (outputX < outputWidth && outputY < outputHeight && outputChannel < outputChannels) {
        float sum = 0.0f;
        
        for (int ic = 0; ic < inputChannels; ++ic) {
            for (int ky = 0; ky < kernelHeight; ++ky) {
                for (int kx = 0; kx < kernelWidth; ++kx) {
                    int inputX = outputX + kx;
                    int inputY = outputY + ky;
                    
                    int inputIdx = ic * inputHeight * inputWidth + inputY * inputWidth + inputX;
                    int kernelIdx = outputChannel * inputChannels * kernelHeight * kernelWidth + 
                                   ic * kernelHeight * kernelWidth + ky * kernelWidth + kx;
                    
                    sum += input[inputIdx] * kernel[kernelIdx];
                }
            }
        }
        
        int outputIdx = outputChannel * outputHeight * outputWidth + outputY * outputWidth + outputX;
        output[outputIdx] = sum;
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
                                int poolSize, int stride, int channels = 1) {
    
   
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z; 
    int blockDimX = blockDim.x;
    int blockDimY = blockDim.y;
    
   
    int outputX = bx * blockDimX + tx;
    int outputY = by * blockDimY + ty;
    
    
    int outputWidth = (inputWidth - poolSize) / stride + 1;
    int outputHeight = (inputHeight - poolSize) / stride + 1;
    
    if (outputX < outputWidth && outputY < outputHeight && bz < channels) {
        int inputChannelOffset = bz * inputHeight * inputWidth;
        int outputChannelOffset = bz * outputHeight * outputWidth;
        
        int inputStartX = outputX * stride;
        int inputStartY = outputY * stride;

        float maxVal = -INFINITY;
        
        for (int py = 0; py < poolSize; ++py) {
            for (int px = 0; px < poolSize; ++px) {
                int inputX = inputStartX + px;
                int inputY = inputStartY + py;
                
            
                if (inputX < inputWidth && inputY < inputHeight) {
                    float val = input[inputChannelOffset + inputY * inputWidth + inputX];
                    maxVal = fmaxf(maxVal, val);
                }
            }
        }
        output[outputChannelOffset + outputY * outputWidth + outputX] = maxVal;
    }
}



void conv2d(const float* input, int inputHeight, int inputWidth,
            const float* kernel, int kernelHeight, int kernelWidth,
            float* output) {
 
    conv2d_multichannel(input, 1, inputHeight, inputWidth, kernel, 1, kernelHeight, kernelWidth, output);
}

void conv2d_multichannel(const float* input, int inputChannels, int inputHeight, int inputWidth,
                        const float* kernel, int outputChannels, int kernelHeight, int kernelWidth,
                        float* output) {
    
    int outputWidth = inputWidth - kernelWidth + 1;
    int outputHeight = inputHeight - kernelHeight + 1;
    
    int inputSize = inputChannels * inputHeight * inputWidth;
    int kernelSize = outputChannels * inputChannels * kernelHeight * kernelWidth;
    int outputSize = outputChannels * outputHeight * outputWidth;

    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, sizeof(float) * inputSize);
    cudaMalloc(&d_kernel, sizeof(float) * kernelSize);
    cudaMalloc(&d_output, sizeof(float) * outputSize);

    cudaMemcpy(d_input, input, sizeof(float) * inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, sizeof(float) * kernelSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x,
                  (outputHeight + blockSize.y - 1) / blockSize.y,
                  outputChannels);

    conv2d_kernel<<<gridSize, blockSize>>>(
        d_input, d_kernel, d_output,
        inputChannels, inputWidth, inputHeight,
        outputChannels, kernelWidth, kernelHeight);

    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, sizeof(float) * outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}


void relu(const float* input, int height, int width, float* output) {
    relu_multichannel(input, 1, height, width, output);
}

void relu_multichannel(const float* input, int channels, int height, int width, float* output) {
    int totalSize = channels * height * width;
    
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


void maxpool2d(const float* input, int inputHeight, int inputWidth, 
               int poolSize, int stride, float* output) {
                
    maxpool2d_multichannel(input, 1, inputHeight, inputWidth, poolSize, stride, output);
}

void maxpool2d_multichannel(const float* input, int channels, int inputHeight, int inputWidth, 
                           int poolSize, int stride, float* output) {
  
    int outputWidth = (inputWidth - poolSize) / stride + 1;
    int outputHeight = (inputHeight - poolSize) / stride + 1;
    int inputSize = channels * inputHeight * inputWidth;
    int outputSize = channels * outputHeight * outputWidth;
    
   
    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float) * inputSize);
    cudaMalloc(&d_output, sizeof(float) * outputSize);
    
   
    cudaMemcpy(d_input, input, sizeof(float) * inputSize, cudaMemcpyHostToDevice);
    
   
    dim3 blockSize(16, 16);
    dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x,
                  (outputHeight + blockSize.y - 1) / blockSize.y,
                  channels);
    
   
    maxpool2d_kernel<<<gridSize, blockSize>>>(
        d_input, d_output,
        inputWidth, inputHeight,
        poolSize, stride, channels);
    
    
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, sizeof(float) * outputSize, cudaMemcpyDeviceToHost);
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
            1, inputWidth, inputHeight, 1, kernelWidth, kernelHeight);
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