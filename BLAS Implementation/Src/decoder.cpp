#include "../Headerfiles/decoder.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <cblas.h>
using namespace std;

// ============ Constructor & Destructor ============

Decoder::Decoder() {
    // Weights initialized as empty vectors
}

Decoder::~Decoder() {
    // Vector cleanup handled automatically
}

// ============ Load Weights ============
void Decoder::loadWeights(WeightLoader& loader) {
    cout << "Loading decoder weights..." << endl;

    proj_w_[0] = loader.loadWeight("decode_head_linear_c_0_proj_weight.bin", 256 * 32);
    proj_b_[0] = loader.loadWeight("decode_head_linear_c_0_proj_bias.bin", 256);

    proj_w_[1] = loader.loadWeight("decode_head_linear_c_1_proj_weight.bin", 256 * 64);
    proj_b_[1] = loader.loadWeight("decode_head_linear_c_1_proj_bias.bin", 256);

    proj_w_[2] = loader.loadWeight("decode_head_linear_c_2_proj_weight.bin", 256 * 160);
    proj_b_[2] = loader.loadWeight("decode_head_linear_c_2_proj_bias.bin", 256);

    proj_w_[3] = loader.loadWeight("decode_head_linear_c_3_proj_weight.bin", 256 * 256);
    proj_b_[3] = loader.loadWeight("decode_head_linear_c_3_proj_bias.bin", 256);

    fuse_w_ = loader.loadWeight("decode_head_linear_fuse_weight.bin", 256 * 1024);

    bn_w_ = loader.loadWeight("decode_head_batch_norm_weight.bin", 256);
    bn_b_ = loader.loadWeight("decode_head_batch_norm_bias.bin", 256);
    bn_running_mean_ = loader.loadWeight("decode_head_batch_norm_running_mean_buffer.bin", 256);
    bn_running_var_ = loader.loadWeight("decode_head_batch_norm_running_var_buffer.bin", 256);

    cls_w_ = loader.loadWeight("decode_head_classifier_weight.bin", 19 * 256);
    cls_b_ = loader.loadWeight("decode_head_classifier_bias.bin", 19);

    cout << "Decoder weights loaded successfully." << endl;
}

// ============ Forward Pass ============

vector<float> Decoder::forward(const vector<vector<float>>& encoder_outputs) {
    cout << "Running decoder forward pass..." << endl;

    const vector<float>& enc0 = encoder_outputs[0];
    const vector<float>& enc1 = encoder_outputs[1];
    const vector<float>& enc2 = encoder_outputs[2];
    const vector<float>& enc3 = encoder_outputs[3];
    
    const int H0 = 64, W0 = 64, C0 = 32;
    const int H1 = 32, W1 = 32, C1 = 64;
    const int H2 = 16, W2 = 16, C2 = 160;
    const int H3 = 8, W3 = 8, C3 = 256;

    const int H_fuse = 64, W_fuse = 64;

    // Linear Projections
    vector<float> proj0 = projectTokens(enc0, H0, W0, C0, proj_w_[0], proj_b_[0]);
    vector<float> proj1 = projectTokens(enc1, H1, W1, C1, proj_w_[1], proj_b_[1]);
    vector<float> proj2 = projectTokens(enc2, H2, W2, C2, proj_w_[2], proj_b_[2]);
    vector<float> proj3 = projectTokens(enc3, H3, W3, C3, proj_w_[3], proj_b_[3]);
    
    // Upsampling to 128×128
    vector<float> up0 = proj0;
    vector<float> up1 = upsampleBilinear(proj1, H1, W1, H_fuse, W_fuse, 256);
    vector<float> up2 = upsampleBilinear(proj2, H2, W2, H_fuse, W_fuse, 256);
    vector<float> up3 = upsampleBilinear(proj3, H3, W3, H_fuse, W_fuse, 256);
    
    // Concatenation
    int HW_fuse = H_fuse * W_fuse;
    vector<float> fused(HW_fuse * 1024);

    for (int i = 0; i < HW_fuse; ++i) {
        for (int c = 0; c < 256; ++c) {
            fused[i * 1024 + c] = up3[i * 256 + c];
        }
        for (int c = 0; c < 256; ++c) {
            fused[i * 1024 + 256 + c] = up2[i * 256 + c];
        }
        for (int c = 0; c < 256; ++c) {
            fused[i * 1024 + 512 + c] = up1[i * 256 + c];
        }
        for (int c = 0; c < 256; ++c) {
            fused[i * 1024 + 768 + c] = up0[i * 256 + c];
        }
    }
    
    // Fuse Convolution 1×1
    vector<float> fuse_out = conv1x1(fused, H_fuse, W_fuse, 1024, 256, fuse_w_, nullptr);
    
    // BatchNorm
    vector<float> bn_out = batchNormInference(fuse_out, H_fuse, W_fuse, 256,
                                                    bn_w_, bn_b_, bn_running_mean_, bn_running_var_);
   
    // ReLU
    vector<float> relu_out = bn_out;
    for (size_t i = 0; i < relu_out.size(); ++i) {
        if (relu_out[i] < 0.0f) {
            relu_out[i] = 0.0f;
        }
    }

    // Classifier Convolution 1×1
    vector<float> logits = conv1x1(relu_out, H_fuse, W_fuse, 256, 19, cls_w_, &cls_b_);

    cout << "Decoder forward pass complete." << endl;

    return logits;
}

// ============ Helper Functions ============

vector<float> Decoder::projectTokens(
    const vector<float>& tokens,
    int H, int W, int C_in,
    const vector<float>& weight,
    const vector<float>& bias) {

    int HW = H * W;
    int C_out = 256;
    vector<float> output(HW * C_out);

    // Use OpenBLAS for matrix multiplication
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        HW,
        C_out,
        C_in,
        1.0f,
        tokens.data(),
        C_in,
        weight.data(),
        C_in,
        0.0f,
        output.data(),
        C_out
    );
    
    // Add bias
    for (int i = 0; i < HW; i++) {
        for (int c = 0; c < C_out; c++) {
            output[i * C_out + c] += bias[c];
        }
    }

    return output;
}

vector<float> Decoder::upsampleBilinear(
    const vector<float>& tokens,
    int H_in, int W_in,
    int H_out, int W_out,
    int C) {

    vector<float> output(H_out * W_out * C);

    float scale_h = static_cast<float>(H_in) / static_cast<float>(H_out);
    float scale_w = static_cast<float>(W_in) / static_cast<float>(W_out);

    for (int h_out = 0; h_out < H_out; ++h_out) {
        for (int w_out = 0; w_out < W_out; ++w_out) {
            float h_in_f = (h_out + 0.5f) * scale_h - 0.5f;
            float w_in_f = (w_out + 0.5f) * scale_w - 0.5f;

            h_in_f = max(0.0f, min(h_in_f, static_cast<float>(H_in - 1)));
            w_in_f = max(0.0f, min(w_in_f, static_cast<float>(W_in - 1)));

            int h_in = static_cast<int>(h_in_f);
            int w_in = static_cast<int>(w_in_f);

            float h_frac = h_in_f - h_in;
            float w_frac = w_in_f - w_in;

            int h_in_next = min(h_in + 1, H_in - 1);
            int w_in_next = min(w_in + 1, W_in - 1);

            for (int c = 0; c < C; ++c) {
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

vector<float> Decoder::conv1x1(
    const vector<float>& input,
    int H, int W, int inC, int outC,
    const vector<float>& weight,
    const vector<float>* bias) {

    int HW = H * W;
    vector<float> output(HW * outC);

    // Use OpenBLAS for 1x1 convolution (matrix multiplication)
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        HW,
        outC,
        inC,
        1.0f,
        input.data(),
        inC,
        weight.data(),
        inC,
        0.0f,
        output.data(),
        outC
    );
    
    // Add bias if present
    if (bias != nullptr) {
        for (int i = 0; i < HW; ++i) {
            for (int j = 0; j < outC; ++j) {
                output[i * outC + j] += (*bias)[j];
            }
        }
    }

    return output;
}

vector<float> Decoder::batchNormInference(
    const vector<float>& input,
    int H, int W, int C,
    const vector<float>& gamma,
    const vector<float>& beta,
    const vector<float>& running_mean,
    const vector<float>& running_var) {

    vector<float> output(input.size());
    int HW = H * W;
    const float eps = 1e-5f;

    for (int c = 0; c < C; ++c) {
        float mean = running_mean[c];
        float var = running_var[c];
        float std_inv = 1.0f / sqrt(var + eps);
        float scale = gamma[c] * std_inv;
        float shift = beta[c] - mean * scale;

        for (int i = 0; i < HW; ++i) {
            int idx = i * C + c;
            output[idx] = input[idx] * scale + shift;
        }
    }

    return output;
}

/*void Decoder::printClassDistribution(const vector<vector<int>>& predictions) const {
    if (predictions.empty() || predictions[0].empty()) {
        cout << "[WARNING] Empty predictions for class distribution" << endl;
        return;
    }

    int H = predictions.size();
    int W = predictions[0].size();
    int total_pixels = H * W;

    vector<int> class_counts(19, 0);
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int class_id = predictions[h][w];
            if (class_id >= 0 && class_id < 19) {
                class_counts[class_id]++;
            }
        }
    }

    cout << "\n" << string(80, '=') << endl;
    cout << "CLASS DISTRIBUTION STATISTICS" << endl;
    cout << string(80, '=') << endl;
    cout << "Total pixels: " << total_pixels << endl;
    cout << string(80, '-') << endl;

    const char* class_names[] = {
        "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain",
        "sky", "person", "rider", "car", "truck", "bus",
        "train", "motorcycle", "bicycle"
    };

    for (int c = 0; c < 19; ++c) {
        if (class_counts[c] > 0) {
            double percentage = (class_counts[c] * 100.0) / total_pixels;
            cout << left << setw(8) << c
                      << setw(20) << class_names[c]
                      << setw(12) << class_counts[c]
                      << fixed << setprecision(2) << percentage << "%"
                      << endl;
        }
    }
}*/
