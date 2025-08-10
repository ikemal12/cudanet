#include <iostream>
#include <vector>
#include <iomanip>
#include "conv2d.h"

void print_matrix(const float* mat, int height, int width, const std::string& name) {
    std::cout << name << ":\n";
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(2) << mat[i * width + j] << " ";
        }
        std::cout << "\n";
    }
}


int main() {
    // Example input image: 6x6 
    int image_h = 6, image_w = 6;
    std::vector<float> input = {
        -1,  2, -3,  4,  5,  1,
         6, -2,  8,  9, -1,  3,
        -4, 12, 13, -5, 15,  2,
        16, -3, 18, 19,  2, -1,
        -2, 22, 23, -6, 25,  4,
         1, -1,  3,  2, -2,  5
    };

    // Edge detection kernel 
    int kernel_h = 3, kernel_w = 3;
    std::vector<float> kernel = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };

    int conv_h = image_h - kernel_h + 1;
    int conv_w = image_w - kernel_w + 1;
    std::vector<float> conv_output(conv_h * conv_w);

    conv2d(
        input.data(), image_h, image_w,
        kernel.data(), kernel_h, kernel_w,
        conv_output.data()
    );

    std::vector<float> relu_output(conv_h * conv_w);
    relu(conv_output.data(), conv_h, conv_w, relu_output.data());

    
    int pool_size = 2;
    int stride = 2;
    int pool_h = (conv_h - pool_size) / stride + 1;
    int pool_w = (conv_w - pool_size) / stride + 1;
    std::vector<float> pool_output(pool_h * pool_w);
    maxpool2d(relu_output.data(), conv_h, conv_w, pool_size, stride, pool_output.data());

    
    print_matrix(input.data(), image_h, image_w, "Input (6x6)");
    std::cout << "\n";
    print_matrix(kernel.data(), kernel_h, kernel_w, "Kernel (Edge Detection)");
    std::cout << "\n";
    print_matrix(conv_output.data(), conv_h, conv_w, "After Convolution (4x4)");
    std::cout << "\n";
    print_matrix(relu_output.data(), conv_h, conv_w, "After ReLU (4x4)");
    std::cout << "\n";
    print_matrix(pool_output.data(), pool_h, pool_w, "After MaxPooling (2x2)");
    return 0;
}