#include "test.h"
#include "conv2d.h"
#include <iostream>
#include <cmath>
#include <cassert>

bool test_conv2d() {

    float input[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8, 
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    
    float kernel[9] = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };
    
    float output[4]; 
    conv2d(input, 4, 4, kernel, 3, 3, output);
    
   
    bool pass = true;
    for (int i = 0; i < 4; ++i) {
        if (std::isnan(output[i]) || std::isinf(output[i])) {
            pass = false;
            break;
        }
    }
    
    std::cout << "Conv2D test: " << (pass ? "PASS" : "FAIL") << "\n";
    std::cout << "Output: [" << output[0] << ", " << output[1] << ", " << output[2] << ", " << output[3] << "]\n";
    return pass;
}

bool test_relu() {

    float input[6] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    float output[6];
    float expected[6] = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    
    relu(input, 2, 3, output);
    
    bool pass = true;
    for (int i = 0; i < 6; ++i) {
        if (std::abs(output[i] - expected[i]) > 1e-6) {
            pass = false;
            break;
        }
    }
    
    std::cout << "ReLU test: " << (pass ? "PASS" : "FAIL") << "\n";
    std::cout << "Input:  [";
    for (int i = 0; i < 6; ++i) std::cout << input[i] << (i < 5 ? ", " : "");
    std::cout << "]\n";
    std::cout << "Output: [";
    for (int i = 0; i < 6; ++i) std::cout << output[i] << (i < 5 ? ", " : "");
    std::cout << "]\n";
    return pass;
}

bool test_maxpool() {

    float input[16] = {
        1, 3, 2, 4,
        5, 6, 8, 7,
        9, 2, 1, 3,
        4, 5, 6, 8
    };
    
    float output[4]; // 2x2 output with 2x2 pooling, stride 2
    maxpool2d(input, 4, 4, 2, 2, output);
    float expected[4] = {6, 8, 9, 8};
    
    bool pass = true;
    for (int i = 0; i < 4; ++i) {
        if (std::abs(output[i] - expected[i]) > 1e-6) {
            pass = false;
            break;
        }
    }
    
    std::cout << "MaxPool test: " << (pass ? "PASS" : "FAIL") << "\n";
    std::cout << "Output: [" << output[0] << ", " << output[1] << ", " << output[2] << ", " << output[3] << "]\n";
    std::cout << "Expected: [" << expected[0] << ", " << expected[1] << ", " << expected[2] << ", " << expected[3] << "]\n";
    return pass;
}

bool test_cnn_pipeline() {
    float input[36];
    for (int i = 0; i < 36; ++i) input[i] = static_cast<float>(i % 10 - 5); 
    float kernel[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0}; 
    
    // Step 1: Convolution (6x6 → 4x4)
    float conv_output[16];
    conv2d(input, 6, 6, kernel, 3, 3, conv_output);
    
    // Step 2: ReLU (4x4 → 4x4)
    float relu_output[16];
    relu(conv_output, 4, 4, relu_output);
    
    // Step 3: MaxPool (4x4 → 2x2)
    float final_output[4];
    maxpool2d(relu_output, 4, 4, 2, 2, final_output);
    
 
    bool pass = true;
    for (int i = 0; i < 4; ++i) {
        if (std::isnan(final_output[i]) || std::isinf(final_output[i]) || final_output[i] < 0) {
            pass = false;
            break;
        }
    }
    
    std::cout << "CNN Pipeline test: " << (pass ? "PASS" : "FAIL") << "\n";
    std::cout << "Final output: [" << final_output[0] << ", " << final_output[1] << ", " << final_output[2] << ", " << final_output[3] << "]\n";
    return pass;
}

void run_all_tests() {
    int passed = 0;
    int total = 5;
    
    if (test_conv2d()) passed++;
    if (test_relu()) passed++;
    if (test_maxpool()) passed++;
    if (test_cnn_pipeline()) passed++;
    if (test_batch_operations()) passed++;
    
    std::cout << "\n======= TEST RESULTS =======\n";
    std::cout << "Passed: " << passed << "/" << total << " tests\n";
    std::cout << "Status: " << (passed == total ? "ALL TESTS PASSED ✓" : "SOME TESTS FAILED ✗") << "\n";
}

bool test_batch_operations() {
    int batchSize = 2;
    int height = 4, width = 4;
    
    float input[32] = {
        // Batch 1
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        // Batch 2  
        -1, -2, 3, 4, -5, 6, 7, -8, 9, -10, 11, 12, -13, 14, 15, -16
    };
    
    float relu_output[32];
    relu_batch(input, batchSize, height, width, relu_output);
    
    bool relu_pass = true;
    for (int i = 0; i < 32; ++i) {
        float expected = fmaxf(0.0f, input[i]);
        if (std::abs(relu_output[i] - expected) > 1e-6) {
            relu_pass = false;
            break;
        }
    }
    
    float pool_output[8]; 
    maxpool2d_batch(input, batchSize, height, width, 2, 2, pool_output);
    
    bool pool_pass = true;
    for (int i = 0; i < 8; ++i) {
        if (std::isnan(pool_output[i]) || std::isinf(pool_output[i])) {
            pool_pass = false;
            break;
        }
    }
    
    bool batch_pass = relu_pass && pool_pass;
    std::cout << "Batch ReLU: " << (relu_pass ? "PASS" : "FAIL") << "\n";
    std::cout << "Batch MaxPool: " << (pool_pass ? "PASS" : "FAIL") << "\n";
    std::cout << "Batch Operations: " << (batch_pass ? "PASS" : "FAIL") << "\n";
    
    return batch_pass;
}
