#pragma once

void conv2d(const float* input, int inputHeight, int inputWidth,
            const float* kernel, int kernelHeight, int kernelWidth,
            float* output);
