#include "../Headerfiles/decoder.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include "decoder.hpp"

// ============ Constructor & Destructor ============

Decoder::Decoder() {
    // Weights initialized as empty vectors
}

Decoder::~Decoder() {
    // Vector cleanup handled automatically
}

// ============ Load Weights ============

void Decoder::loadWeights(WeightLoader& loader) {
 
    // Linear_C[0]: projects 32 -> 256
    proj_w_[0] = loader.loadWeight("decode_head_linear_c_0_proj_weight.bin", 256 * 32);
    proj_b_[0] = loader.loadWeight("decode_head_linear_c_0_proj_bias.bin", 256);

    // Linear_C[1]: projects 64 -> 256
    proj_w_[1] = loader.loadWeight("decode_head_linear_c_1_proj_weight.bin", 256 * 64);
    proj_b_[1] = loader.loadWeight("decode_head_linear_c_1_proj_bias.bin", 256);

    // Linear_C[2]: projects 160 -> 256
    proj_w_[2] = loader.loadWeight("decode_head_linear_c_2_proj_weight.bin", 256 * 160);
    proj_b_[2] = loader.loadWeight("decode_head_linear_c_2_proj_bias.bin", 256);

    // Linear_C[3]: projects 256 -> 256
    proj_w_[3] = loader.loadWeight("decode_head_linear_c_3_proj_weight.bin", 256 * 256);
    proj_b_[3] = loader.loadWeight("decode_head_linear_c_3_proj_bias.bin", 256);

    // Fuse convolution 1x1: [256, 1024, 1, 1] (NO bias)
    fuse_w_ = loader.loadWeight("decode_head_linear_fuse_weight.bin", 256 * 1024);

    // BatchNorm 
    bn_w_ = loader.loadWeight("decode_head_batch_norm_weight.bin", 256);
    bn_b_ = loader.loadWeight("decode_head_batch_norm_bias.bin", 256);
    bn_running_mean_ = loader.loadWeight("decode_head_batch_norm_running_mean_buffer.bin", 256);
    bn_running_var_ = loader.loadWeight("decode_head_batch_norm_running_var_buffer.bin", 256);

    // Classifier convolution 1x1: [19, 256, 1, 1]
    cls_w_ = loader.loadWeight("decode_head_classifier_weight.bin", 19 * 256);
    cls_b_ = loader.loadWeight("decode_head_classifier_bias.bin", 19);
}

// ============ Forward Pass ============

std::vector<float> Decoder::forward(const std::vector<std::vector<float>>& encoder_outputs) {
    
    // Extract encoder outputs
    const std::vector<float>& enc0 = encoder_outputs[0]; // 128×128×32
    const std::vector<float>& enc1 = encoder_outputs[1]; // 64×64×64
    const std::vector<float>& enc2 = encoder_outputs[2]; // 32×32×160
    const std::vector<float>& enc3 = encoder_outputs[3]; // 16×16×256

    // Encoder output dimensions
    const int H0 = 128, W0 = 128, C0 = 32;
    const int H1 = 64, W1 = 64, C1 = 64;
    const int H2 = 32, W2 = 32, C2 = 160;
    const int H3 = 16, W3 = 16, C3 = 256;

    // Target resolution for fusion
    const int H_fuse = 128, W_fuse = 128;

    // ========== STEP 1: Linear Projections ==========

    std::vector<float> proj0 = projectTokens(enc0, H0, W0, C0, proj_w_[0], proj_b_[0]);
    std::vector<float> proj1 = projectTokens(enc1, H1, W1, C1, proj_w_[1], proj_b_[1]);
    std::vector<float> proj2 = projectTokens(enc2, H2, W2, C2, proj_w_[2], proj_b_[2]);
    std::vector<float> proj3 = projectTokens(enc3, H3, W3, C3, proj_w_[3], proj_b_[3]);
    
    // ========== STEP 2: Upsampling to 128×128 ==========
    
    std::vector<float> up0 = proj0;
    std::vector<float> up1 = upsampleBilinear(proj1, H1, W1, H_fuse, W_fuse, 256);
    std::vector<float> up2 = upsampleBilinear(proj2, H2, W2, H_fuse, W_fuse, 256);
    std::vector<float> up3 = upsampleBilinear(proj3, H3, W3, H_fuse, W_fuse, 256);
    
    // ========== STEP 3: Concatenation ==========

    int HW_fuse = H_fuse * W_fuse;
    std::vector<float> fused(HW_fuse * 1024);

    for (int i = 0; i < HW_fuse; ++i) {
        
        // up3
        for (int c = 0; c < 256; ++c) {
            fused[i * 1024 + c] = up3[i * 256 + c];
       }
        // up2
        for (int c = 0; c < 256; ++c) {
            fused[i * 1024 + 256 + c] = up2[i * 256 + c];
        }

        // up1
        for (int c = 0; c < 256; ++c) {
            fused[i * 1024 + 512 + c] = up1[i * 256 + c];
        }
        // up0 
        for (int c = 0; c < 256; ++c) {
            fused[i * 1024 + 768 + c] = up0[i * 256 + c];
        }
    }
    
    // ========== STEP 4: Fuse Convolution 1×1 ==========

    std::vector<float> fuse_out = conv1x1(fused, H_fuse, W_fuse, 1024, 256, fuse_w_, nullptr);
    
    // ========== STEP 5: BatchNorm ==========

    std::vector<float> bn_out = batchNormInference(fuse_out, H_fuse, W_fuse, 256, bn_w_, bn_b_, bn_running_mean_, bn_running_var_);
   
    // ========== STEP 6: ReLU ==========

    std::vector<float> relu_out = bn_out;
    for (size_t i = 0; i < relu_out.size(); ++i) {
        if (relu_out[i] < 0.0f) {
            relu_out[i] = 0.0f;
        }
    }

    // ========== STEP 7: Classifier Convolution 1×1 ==========

    std::vector<float> logits = conv1x1(relu_out, H_fuse, W_fuse, 256, 19, cls_w_, &cls_b_);

    return logits;
}

// ============ Helper Functions ============

std::vector<float> Decoder::projectTokens(const std::vector<float>& tokens, int H, int W, int C_in, const std::vector<float>& weight, const std::vector<float>& bias) {

    int HW = H * W;
    int C_out = 256;
    std::vector<float> output(HW * C_out);

    // For each spatial position
    for (int i = 0; i < HW; ++i) {
        // For each output channel
        for (int j = 0; j < C_out; ++j) {
            double sum = static_cast<double>(bias[j]);
            
            // Matrix multiply: tokens[i, :] @ weight[j, :]
            for (int k = 0; k < C_in; ++k) {
                int token_idx = i * C_in + k;
                // weight[j, k]
                int weight_idx = j * C_in + k;
                sum += static_cast<double>(tokens[token_idx]) * static_cast<double>(weight[weight_idx]);
            }
            output[i * C_out + j] = static_cast<float>(sum);
        }
    }

    return output;
}

std::vector<float> Decoder::upsampleBilinear( const std::vector<float>& tokens, int H_in, int W_in, int H_out, int W_out, int C) {

    std::vector<float> output(H_out * W_out * C);

    // Scale factors 
    float scale_h = static_cast<float>(H_in) / static_cast<float>(H_out);
    float scale_w = static_cast<float>(W_in) / static_cast<float>(W_out);

    for (int h_out = 0; h_out < H_out; ++h_out) {
        for (int w_out = 0; w_out < W_out; ++w_out) {
            
            float h_in_f = (h_out + 0.5f) * scale_h - 0.5f;
            float w_in_f = (w_out + 0.5f) * scale_w - 0.5f;

            h_in_f = std::max(0.0f, std::min(h_in_f, static_cast<float>(H_in - 1)));
            w_in_f = std::max(0.0f, std::min(w_in_f, static_cast<float>(W_in - 1)));

            int h_in = static_cast<int>(h_in_f);
            int w_in = static_cast<int>(w_in_f);

            float h_frac = h_in_f - h_in;
            float w_frac = w_in_f - w_in;

            int h_in_next = std::min(h_in + 1, H_in - 1);
            int w_in_next = std::min(w_in + 1, W_in - 1);

            for (int c = 0; c < C; ++c) {
                // Bilinear interpolation
                float v00 = tokens[h_in * W_in * C + w_in * C + c];
                float v01 = tokens[h_in * W_in * C + w_in_next * C + c];
                float v10 = tokens[h_in_next * W_in * C + w_in * C + c];
                float v11 = tokens[h_in_next * W_in * C + w_in_next * C + c];

                float v0 = v00 * (1.0f - w_frac) + v01 * w_frac;
                float v1 = v10 * (1.0f - w_frac) + v11 * w_frac;
                float v = v0 * (1.0f - h_frac) + v1 * h_frac;

                output[h_out * W_out * C + w_out * C + c] = v;
            }
        }
    }

    return output;
}

std::vector<float> Decoder::conv1x1( const std::vector<float>& input, int H, int W, int inC, int outC, const std::vector<float>& weight, const std::vector<float>* bias) {

    int HW = H * W;
    std::vector<float> output(HW * outC);

    // For each spatial position
    for (int i = 0; i < HW; ++i) {
        // For each output channel
        for (int j = 0; j < outC; ++j) {
            double sum = (bias != nullptr) ? static_cast<double>((*bias)[j]) : 0.0;
            for (int k = 0; k < inC; ++k) {
                int input_idx = i * inC + k;
                int weight_idx = j * inC + k;
                sum += static_cast<double>(input[input_idx]) * static_cast<double>(weight[weight_idx]);
            }
            output[i * outC + j] = static_cast<float>(sum);
        }
    }

    return output;
}

std::vector<float> Decoder::fuseConv1x1( const std::vector<float>& input, int H, int W, const std::vector<float>& weight) {

    const int inC = 1024;
    const int outC = 256;
    int HW = H * W;

    std::vector<float> output(HW * outC);

    // For each spatial position
    for (int i = 0; i < HW; ++i) {
        // For each output channel
        for (int oc = 0; oc < outC; ++oc) {
            double sum = 0.0;  
            for (int ic = 0; ic < inC; ++ic) {
                int input_idx = i * inC + ic;
                int weight_idx = oc * inC + ic;
                sum += static_cast<double>(input[input_idx]) * static_cast<double>(weight[weight_idx]);
            }
            output[i * outC + oc] = static_cast<float>(sum);
        }
    }
    
    return output;
}


std::vector<float> Decoder::batchNormInference( const std::vector<float>& input, int H, int W, int C, const std::vector<float>& gamma, const std::vector<float>& beta, const std::vector<float>& running_mean, const std::vector<float>& running_var) {

    std::vector<float> output(input.size());
    int HW = H * W;
    const float eps = 1e-5f;

    // Formula: y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta
    for (int c = 0; c < C; ++c) {
        float mean = running_mean[c];
        float var = running_var[c];
        float std_inv = 1.0f / std::sqrt(var + eps);
        float scale = gamma[c] * std_inv;
        float shift = beta[c] - mean * scale;

        for (int i = 0; i < HW; ++i) {
            int idx = i * C + c;
            output[idx] = input[idx] * scale + shift;
        }
    }

    return output;
}

// ============ Class Distribution ============

void Decoder::printClassDistribution(const std::vector<std::vector<int>>& predictions) const {
    if (predictions.empty() || predictions[0].empty()) {
        std::cout << "[WARNING] Empty predictions for class distribution" << std::endl;
        return;
    }

    int H = predictions.size();
    int W = predictions[0].size();
    int total_pixels = H * W;

    // Count predictions for each class
    std::vector<int> class_counts(19, 0);
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int class_id = predictions[h][w];
            if (class_id >= 0 && class_id < 19) {
                class_counts[class_id]++;
            }
        }
    }

    // Print class distribution
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "CLASS DISTRIBUTION STATISTICS" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Total pixels: " << total_pixels << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(8) << "Class"
              << std::setw(12) << "Count"
              << std::setw(12) << "Percentage"
              << std::setw(20) << "Visualization" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // Class names for Cityscapes dataset
    const char* class_names[] = {
        "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain",
        "sky", "person", "rider", "car", "truck", "bus",
        "train", "motorcycle", "bicycle"
    };

    for (int c = 0; c < 19; ++c) {
        double percentage = (class_counts[c] * 100.0) / total_pixels;
        std::cout << std::left << std::setw(8) << c
                  << std::setw(12) << class_counts[c]
                  << std::setw(12) << std::fixed << std::setprecision(2) << percentage << "%"
                  << std::endl;
    }
}
