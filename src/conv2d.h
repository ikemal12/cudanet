#pragma once

void conv2d(const float* input, int inputHeight, int inputWidth,
            const float* kernel, int kernelHeight, int kernelWidth,
            float* output);

void relu(const float* input, int height, int width, float* output);

void maxpool2d(const float* input, int inputHeight, int inputWidth, 
               int poolSize, int stride, float* output);

void conv2d_multichannel(const float* input, int inputChannels, int inputHeight, int inputWidth,
                        const float* kernel, int outputChannels, int kernelHeight, int kernelWidth,
                        float* output);

void relu_multichannel(const float* input, int channels, int height, int width, float* output);

void maxpool2d_multichannel(const float* input, int channels, int inputHeight, int inputWidth,
                           int poolSize, int stride, float* output);

void conv2d_batch(const float* input, int batchSize, int inputHeight, int inputWidth,
                  const float* kernel, int kernelHeight, int kernelWidth,
                  float* output);

void relu_batch(const float* input, int batchSize, int height, int width, float* output);

void maxpool2d_batch(const float* input, int batchSize, int inputHeight, int inputWidth,
                     int poolSize, int stride, float* output);
