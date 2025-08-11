#pragma once


void conv2d_naive(const float* input, int inputHeight, int inputWidth,
                 const float* kernel, int kernelHeight, int kernelWidth,
                 float* output);


void benchmark_cnn_layers(int inputHeight, int inputWidth, 
                          int kernelHeight, int kernelWidth,
                          int iterations = 100);

void benchmark_convolution(int inputHeight, int inputWidth, 
                          int kernelHeight, int kernelWidth,
                          int iterations = 100);

void benchmark_relu(int height, int width, int iterations = 1000);

void benchmark_maxpool(int inputHeight, int inputWidth, 
                      int poolSize, int stride, int iterations = 100);
