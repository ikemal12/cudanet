#include "perf.h"
#include "conv2d.h"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>


__global__ void conv2d_kernel_naive(const float* input, const float *kernel, float* output,
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


void conv2d_naive(const float* input, int inputHeight, int inputWidth,
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

    conv2d_kernel_naive<<<gridSize, blockSize>>>(d_input, d_kernel, d_output,
                                                 inputWidth, inputHeight,
                                                 kernelWidth, kernelHeight);

    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, sizeof(float) * outputWidth * outputHeight, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

void benchmark_convolution(int inputHeight, int inputWidth, 
                          int kernelHeight, int kernelWidth,
                          int iterations) {
    
    std::cout << "\n=== CONVOLUTION BENCHMARK ===\n";
    std::cout << "Input: " << inputWidth << "x" << inputHeight << " | Kernel: " << kernelWidth << "x" << kernelHeight << "\n";
    
    int inputSize = inputWidth * inputHeight;
    int kernelSize = kernelWidth * kernelHeight;
    int outputWidth = inputWidth - kernelWidth + 1;
    int outputHeight = inputHeight - kernelHeight + 1;
    int outputSize = outputWidth * outputHeight;
    
    float* input = new float[inputSize];
    float* kernel = new float[kernelSize];
    float* output_naive = new float[outputSize];
    float* output_optimized = new float[outputSize];
    
    for (int i = 0; i < inputSize; ++i) input[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < kernelSize; ++i) kernel[i] = static_cast<float>(rand()) / RAND_MAX;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    conv2d_naive(input, inputHeight, inputWidth, kernel, kernelHeight, kernelWidth, output_naive);
    conv2d(input, inputHeight, inputWidth, kernel, kernelHeight, kernelWidth, output_optimized);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        conv2d_naive(input, inputHeight, inputWidth, kernel, kernelHeight, kernelWidth, output_naive);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float naive_time;
    cudaEventElapsedTime(&naive_time, start, stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        conv2d(input, inputHeight, inputWidth, kernel, kernelHeight, kernelWidth, output_optimized);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float optimized_time;
    cudaEventElapsedTime(&optimized_time, start, stop);
    
    long long ops_per_conv = (long long)outputWidth * outputHeight * kernelWidth * kernelHeight * 2; 
    double gops_naive = (double)(ops_per_conv * iterations) / (naive_time * 1e6);
    double gops_optimized = (double)(ops_per_conv * iterations) / (optimized_time * 1e6);
    
    long long bytes_per_conv = (long long)(inputSize + kernelSize + outputSize) * sizeof(float);
    double gb_naive = (double)(bytes_per_conv * iterations) / (naive_time * 1e6);
    double gb_optimized = (double)(bytes_per_conv * iterations) / (optimized_time * 1e6);
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Naive:     " << naive_time << " ms | " << gops_naive << " GOPS | " << gb_naive << " GB/s\n";
    std::cout << "Optimized: " << optimized_time << " ms | " << gops_optimized << " GOPS | " << gb_optimized << " GB/s\n";
    std::cout << "Speedup: " << naive_time / optimized_time << "x\n";
    
    delete[] input; delete[] kernel; delete[] output_naive; delete[] output_optimized;
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

void benchmark_relu(int height, int width, int iterations) {
    std::cout << "\n=== RELU BENCHMARK ===\n";
    std::cout << "Size: " << width << "x" << height << " (" << width*height << " elements)\n";
    
    int size = height * width;
    float* input = new float[size];
    float* output = new float[size];
    
    for (int i = 0; i < size; ++i) input[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    relu(input, height, width, output); // warmup
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        relu(input, height, width, output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    double elements_per_sec = (double)(size * iterations) / (elapsed_time * 1e-3);
    double gb_per_sec = (double)(size * iterations * 2 * sizeof(float)) / (elapsed_time * 1e6);
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Time: " << elapsed_time << " ms | " << std::scientific << elements_per_sec << std::fixed << " elem/s | " << gb_per_sec << " GB/s\n";
    
    delete[] input; delete[] output;
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

void benchmark_maxpool(int inputHeight, int inputWidth, 
                      int poolSize, int stride, int iterations) {
    
    std::cout << "\n=== MAXPOOL BENCHMARK ===\n";
    std::cout << "Input: " << inputWidth << "x" << inputHeight << " | Pool: " << poolSize << "x" << poolSize << "\n";
    
    int inputSize = inputWidth * inputHeight;
    int outputWidth = (inputWidth - poolSize) / stride + 1;
    int outputHeight = (inputHeight - poolSize) / stride + 1;
    int outputSize = outputWidth * outputHeight;
    
    float* input = new float[inputSize];
    float* output = new float[outputSize];
    
    for (int i = 0; i < inputSize; ++i) input[i] = static_cast<float>(rand()) / RAND_MAX;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    maxpool2d(input, inputHeight, inputWidth, poolSize, stride, output);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        maxpool2d(input, inputHeight, inputWidth, poolSize, stride, output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    long long ops_per_pool = (long long)outputSize * poolSize * poolSize;
    double gops = (double)(ops_per_pool * iterations) / (elapsed_time * 1e6);
    double gb_per_sec = (double)((inputSize + outputSize) * iterations * sizeof(float)) / (elapsed_time * 1e6);
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Time: " << elapsed_time << " ms | " << gops << " GOPS | " << gb_per_sec << " GB/s\n";
    
    delete[] input; delete[] output;
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

void benchmark_cnn_layers(int inputHeight, int inputWidth, 
                          int kernelHeight, int kernelWidth,
                          int iterations) {

    benchmark_convolution(inputHeight, inputWidth, kernelHeight, kernelWidth, iterations);
    
    int conv_h = inputHeight - kernelHeight + 1;
    int conv_w = inputWidth - kernelWidth + 1;
    benchmark_relu(conv_h, conv_w, iterations * 5);
   
    benchmark_maxpool(conv_h, conv_w, 2, 2, iterations);
}

void benchmark_batch_operations(int batchSize, int inputHeight, int inputWidth, 
                               int kernelHeight, int kernelWidth, int iterations) {
    
    std::cout << "\n=== BATCH OPERATIONS BENCHMARK ===\n";
    std::cout << "Batch size: " << batchSize << " | Input: " << inputWidth << "x" << inputHeight << "\n";
    
    int inputSize = inputWidth * inputHeight;
    int outputWidth = inputWidth - kernelWidth + 1;
    int outputHeight = inputHeight - kernelHeight + 1;
    int outputSize = outputWidth * outputHeight;
    
    float* batch_input = new float[batchSize * inputSize];
    float* kernel = new float[kernelWidth * kernelHeight];
    float* batch_output_conv = new float[batchSize * outputSize];
    float* batch_output_relu = new float[batchSize * outputSize];
    
    for (int i = 0; i < batchSize * inputSize; ++i) batch_input[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < kernelWidth * kernelHeight; ++i) kernel[i] = static_cast<float>(rand()) / RAND_MAX;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    conv2d_batch(batch_input, batchSize, inputHeight, inputWidth, kernel, kernelHeight, kernelWidth, batch_output_conv); // warmup
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        conv2d_batch(batch_input, batchSize, inputHeight, inputWidth, kernel, kernelHeight, kernelWidth, batch_output_conv);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float batch_conv_time;
    cudaEventElapsedTime(&batch_conv_time, start, stop);
    relu_batch(batch_output_conv, batchSize, outputHeight, outputWidth, batch_output_relu); // warmup
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations * 5; ++i) {
        relu_batch(batch_output_conv, batchSize, outputHeight, outputWidth, batch_output_relu);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float batch_relu_time;
    cudaEventElapsedTime(&batch_relu_time, start, stop);
  
    double images_per_sec_conv = (double)(batchSize * iterations) / (batch_conv_time * 1e-3);
    double images_per_sec_relu = (double)(batchSize * iterations * 5) / (batch_relu_time * 1e-3);
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Batch Conv: " << batch_conv_time/iterations << " ms/batch | " << images_per_sec_conv << " images/sec\n";
    std::cout << "Batch ReLU: " << batch_relu_time/(iterations*5) << " ms/batch | " << images_per_sec_relu << " images/sec\n";
    
    delete[] batch_input; delete[] kernel; delete[] batch_output_conv; delete[] batch_output_relu;
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

void benchmark_multichannel_operations(int inputChannels, int outputChannels, 
                                     int inputHeight, int inputWidth, 
                                     int kernelHeight, int kernelWidth, int iterations) {
    
    std::cout << "\n=== MULTICHANNEL BENCHMARK ===\n";
    std::cout << "Channels: " << inputChannels << "->" << outputChannels << " | Size: " << inputWidth << "x" << inputHeight << "\n";
    
    int inputSize = inputChannels * inputHeight * inputWidth;
    int kernelSize = outputChannels * inputChannels * kernelHeight * kernelWidth;
    int outputWidth = inputWidth - kernelWidth + 1;
    int outputHeight = inputHeight - kernelHeight + 1;
    int outputSize = outputChannels * outputHeight * outputWidth;
    
    float* input = new float[inputSize];
    float* kernel = new float[kernelSize];
    float* output = new float[outputSize];
    
    for (int i = 0; i < inputSize; ++i) input[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < kernelSize; ++i) kernel[i] = static_cast<float>(rand()) / RAND_MAX;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    conv2d_multichannel(input, inputChannels, inputHeight, inputWidth, kernel, outputChannels, kernelHeight, kernelWidth, output); // warmup
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        conv2d_multichannel(input, inputChannels, inputHeight, inputWidth, kernel, outputChannels, kernelHeight, kernelWidth, output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    long long ops_per_conv = (long long)outputSize * inputChannels * kernelHeight * kernelWidth * 2;
    double gops = (double)(ops_per_conv * iterations) / (elapsed_time * 1e6);
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Multichannel Conv: " << elapsed_time/iterations << " ms | " << gops << " GOPS\n";
    
    delete[] input; delete[] kernel; delete[] output;
    cudaEventDestroy(start); cudaEventDestroy(stop);
}
