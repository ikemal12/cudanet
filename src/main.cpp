#include <iostream>
#include <vector>
#include <iomanip>
#include "conv2d.cu"

void print_matrix(const float* mat, int height, int width) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(2) << mat[i * width * j] << " ";
        }
        std::cout << "\n";
    }
}


int main() {
    // example input image: 5x5
    int image_h = 5, image_w = 5;
    std::vector<float> input = {
        1,  2,  3,  4,  5,
        6,  7,  8,  9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };

    // example kernel: 3x3 blur 
    int kernel_h = 3, kernel_w = 3;
    std::vector<float> kernel = {
        1.0f / 9, 1.0f / 9, 1.0f / 9,
        1.0f / 9, 1.0f / 9, 1.0f / 9,
        1.0f / 9, 1.0f / 9, 1.0f / 9
    };

    int output_h = image_h - kernel_h + 1;
    int output_w = image_w - kernel_w + 1;
    std::vector<float> output(output_h * output_w);

    conv2d(
        input.data(), image_h, image_w,
        kernel.data(), kernel_h, kernel_w,
        output.data()
    );

    std::cout << "Input:\n";
    print_matrix(input.data(), image_h, image_w);

    std::cout << "\nKernel:\n";
    print_matrix(kernel.data(), kernel_h, kernel_w);

    std::cout << "\nOutput:\n";
    print_matrix(output.data(), output_h, output_w);    
    return 0;
}