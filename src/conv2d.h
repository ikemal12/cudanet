#pragma once

void conv2d(const float* input, int inputHeight, int inputWidth,
            const float* kernel, int kernelHeight, int kernelWidth,
            float* output);


void relu(const float* input, int height, int width, float* output);


void maxpool2d(const float* input, int inputHeight, int inputWidth, 
               int poolSize, int stride, float* output);
