#include "../Headerfiles/encoder.hpp"
#include <iostream>
#include <fstream>
#include <cstring>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <algorithm>
using namespace std;

Encoder::Encoder() {
    // Initialize patch embeddings
    patch_embed_0_ = new PatchEmbedding(0, 3, 32, 7, 4);
    patch_embed_1_ = new PatchEmbedding(1, 32, 64, 3, 2);
    patch_embed_2_ = new PatchEmbedding(2, 64, 160, 3, 2);
    patch_embed_3_ = new PatchEmbedding(3, 160, 256, 3, 2);

    // Initialize transformer blocks
    tb_0_0_ = new TransformerBlock(0, 0, 32, 4, 8);
    tb_0_1_ = new TransformerBlock(0, 1, 32, 4, 8);

    tb_1_0_ = new TransformerBlock(1, 0, 64, 4, 4);
    tb_1_1_ = new TransformerBlock(1, 1, 64, 4, 4);

    tb_2_0_ = new TransformerBlock(2, 0, 160, 4, 2);
    tb_2_1_ = new TransformerBlock(2, 1, 160, 4, 2);

    tb_3_0_ = new TransformerBlock(3, 0, 256, 4, 1);
    tb_3_1_ = new TransformerBlock(3, 1, 256, 4, 1);
}

Encoder::~Encoder() {
    delete patch_embed_0_;
    delete patch_embed_1_;
    delete patch_embed_2_;
    delete patch_embed_3_;
    delete tb_0_0_;
    delete tb_0_1_;
    delete tb_1_0_;
    delete tb_1_1_;
    delete tb_2_0_;
    delete tb_2_1_;
    delete tb_3_0_;
    delete tb_3_1_;
}

void Encoder::loadWeights(WeightLoader& loader) {
    cout << "Loading encoder weights..." << endl;
    
    auto pe0 = loader.loadPatchEmbedWeights(0);
    patch_embed_0_->loadWeights(pe0);
    
    auto pe1 = loader.loadPatchEmbedWeights(1);
    patch_embed_1_->loadWeights(pe1);
    
    auto pe2 = loader.loadPatchEmbedWeights(2);
    patch_embed_2_->loadWeights(pe2);
    
    auto pe3 = loader.loadPatchEmbedWeights(3);
    patch_embed_3_->loadWeights(pe3);
    
    for (int stage = 0; stage < 4; ++stage) {
        for (int block = 0; block < 2; ++block) {
            auto tb_weights = loader.loadTransformerBlockWeights(stage, block);
            
            if (stage == 0) {
                if (block == 0) tb_0_0_->loadWeights(tb_weights);
                else tb_0_1_->loadWeights(tb_weights);
            } else if (stage == 1) {
                if (block == 0) tb_1_0_->loadWeights(tb_weights);
                else tb_1_1_->loadWeights(tb_weights);
            } else if (stage == 2) {
                if (block == 0) tb_2_0_->loadWeights(tb_weights);
                else tb_2_1_->loadWeights(tb_weights);
            } else if (stage == 3) {
                if (block == 0) tb_3_0_->loadWeights(tb_weights);
                else tb_3_1_->loadWeights(tb_weights);
            }
        }
    }
    
    for (int stage = 0; stage < 4; ++stage) {
        auto ln_weights = loader.loadStageLayerNormWeights(stage);
        stage_ln_weight_[stage] = ln_weights.weight;
        stage_ln_bias_[stage] = ln_weights.bias;
    }
    
    cout << "Encoder weights loaded successfully." << endl;
}

vector<float> Encoder::stageLayerNorm(
    const vector<float>& input,
    const vector<float>& weight,
    const vector<float>& bias,
    int H, int W, int C) {
    
    vector<float> output = input;
    int HW = H * W;
    const float eps = 1e-5f;
    
    #pragma omp parallel for
    for (int idx = 0; idx < HW; ++idx) {
        int base = idx * C;
        
        float mean = 0.0;
        for (int c = 0; c < C; ++c) {
            mean += static_cast<float>(input[base + c]);
        }
        mean /= C;
        
        float var = 0.0;
        for (int c = 0; c < C; ++c) {
            float diff = static_cast<float>(input[base + c]) - mean;
            var += diff * diff;
        }
        var /= C;
        
        float std = sqrt(var + static_cast<float>(eps));
        
        for (int c = 0; c < C; ++c) {
            float normalized = (static_cast<float>(input[base + c]) - mean) / std;
            output[base + c] = static_cast<float>(
                normalized * static_cast<float>(weight[c]) + 
                static_cast<double>(bias[c]));
        }
    }
    
    return output;
}

vector<float> Encoder::hwcToNCHW(
    const vector<float>& hwc, int H, int W, int C)
{
    vector<float> nchw(C * H * W);

    #pragma omp parallel for
    for (int c = 0; c < C; ++c) {
        int c_offset = c * H * W;

        for (int h = 0; h < H; ++h) {
            int hwc_base = h * W * C;
            int hw_offset = h * W;

            for (int w = 0; w < W; ++w) {
                nchw[c_offset + hw_offset + w] =
                    hwc[hwc_base + w * C + c];
            }
        }
    }

    return nchw;
}

vector<float> Encoder::tokensToNCHW(
    const vector<float>& tokens, int H, int W, int C)
{
    vector<float> nchw(C * H * W);

    #pragma omp parallel for
    for (int c = 0; c < C; ++c) {
        int c_offset = c * H * W;

        for (int h = 0; h < H; ++h) {
            int hw_offset = h * W;

            for (int w = 0; w < W; ++w) {
                int token = hw_offset + w;

                nchw[c_offset + hw_offset + w] =
                    tokens[token * C + c];
            }
        }
    }

    return nchw;
}
vector<float> Encoder::processStage(int stage, const vector<float>& input,
                                        int H_in, int W_in, int C_in) {
    PatchEmbedding* pe = nullptr;
    TransformerBlock* tb0 = nullptr;
    TransformerBlock* tb1 = nullptr;
    
    if (stage == 0) {
        pe = patch_embed_0_;
        tb0 = tb_0_0_;
        tb1 = tb_0_1_;
    } else if (stage == 1) {
        pe = patch_embed_1_;
        tb0 = tb_1_0_;
        tb1 = tb_1_1_;
    } else if (stage == 2) {
        pe = patch_embed_2_;
        tb0 = tb_2_0_;
        tb1 = tb_2_1_;
    } else if (stage == 3) {
        pe = patch_embed_3_;
        tb0 = tb_3_0_;
        tb1 = tb_3_1_;
    }
    
    // Patch embedding expects NCHW input, outputs HWC tokens
    vector<float> x = pe->forward(input, H_in, W_in);
    
    int H = pe->getOutputHeight();
    int W = pe->getOutputWidth();
    int C = pe->getEmbedDim();
    
    // Transformer blocks operate on HWC tokens
    x = tb0->forward(x, H, W);
    x = tb1->forward(x, H, W);
    
    // Apply stage layer normalization
    x = stageLayerNorm(x, stage_ln_weight_[stage], stage_ln_bias_[stage], H, W, C);
    
    return x;
}

vector<vector<float>> Encoder::forward(const vector<float>& input) {
    encoder_outputs_.clear();
    
    cout << "Running encoder forward pass..." << endl;
    
    // Convert input from HWC to NCHW
    vector<float> input_nchw = hwcToNCHW(input, 256, 256, 3);
    
    // Stage 0: 256x256x3 -> 64x64x32
    vector<float> enc0 = processStage(0, input_nchw, 256, 256, 3);
    encoder_outputs_.push_back(enc0);
    
    // Stage 1: 64x64x32 -> 64x64x64
    vector<float> enc0_nchw = tokensToNCHW(enc0, 64, 64, 32);
    vector<float> enc1 = processStage(1, enc0_nchw, 64, 64, 32);
    encoder_outputs_.push_back(enc1);
    
    // Stage 2: 64x64x64 -> 32x32x160
    vector<float> enc1_nchw = tokensToNCHW(enc1, 32, 32, 64);
    vector<float> enc2 = processStage(2, enc1_nchw, 32, 32, 64);
    encoder_outputs_.push_back(enc2);
    
    // Stage 3: 32x32x160 -> 16x16x256
    vector<float> enc2_nchw = tokensToNCHW(enc2, 16, 16, 160);
    vector<float> enc3 = processStage(3, enc2_nchw, 16, 16, 160);
    encoder_outputs_.push_back(enc3);
    
    cout << "Encoder forward pass complete." << endl;
    
    return encoder_outputs_;
}
