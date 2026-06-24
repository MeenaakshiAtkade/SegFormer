#ifndef WEIGHT_LOADER_HPP
#define WEIGHT_LOADER_HPP

#include <string>
#include <vector>
#include <iostream>

struct PatchEmbedWeights {
    std::vector<float> proj_weight;
    std::vector<float> proj_bias;
    std::vector<float> ln_weight;
    std::vector<float> ln_bias;
    
    int in_channels;
    int embed_dim;
    int patch_size;
    int stride;
};

struct StageLayerNormWeights {
    std::vector<float> weight;
    std::vector<float> bias;
};

struct TransformerBlockWeights {
    // LayerNorm 1
    std::vector<float> ln1_weight;
    std::vector<float> ln1_bias;
    
    // Attention weights
    std::vector<float> attn_q_weight;
    std::vector<float> attn_q_bias;
    std::vector<float> attn_k_weight;
    std::vector<float> attn_k_bias;
    std::vector<float> attn_v_weight;
    std::vector<float> attn_v_bias;
    std::vector<float> attn_sr_weight;   
    std::vector<float> attn_sr_bias;
    std::vector<float> attn_ln_weight;   
    std::vector<float> attn_ln_bias;
    std::vector<float> attn_proj_weight; 
    std::vector<float> attn_proj_bias;
    
    // LayerNorm 2
    std::vector<float> ln2_weight;
    std::vector<float> ln2_bias;
    
    // MLP weights
    std::vector<float> mlp_fc1_weight;
    std::vector<float> mlp_fc1_bias;
    std::vector<float> mlp_dwconv_weight;
    std::vector<float> mlp_dwconv_bias;
    std::vector<float> mlp_fc2_weight;
    std::vector<float> mlp_fc2_bias;
    
    int embed_dim;
    int mlp_hidden_dim;
    int sr_ratio;
};

class WeightLoader {
public:
    WeightLoader(const std::string& weights_dir);
    
    std::vector<float> loadWeight(const std::string& filename, int expected_count = -1);
    void printWeightStats(const std::vector<float>& weights, const std::string& name, int sample_count = 16);
    
    PatchEmbedWeights loadPatchEmbedWeights(int stage);
    TransformerBlockWeights loadTransformerBlockWeights(int stage, int block);
    StageLayerNormWeights loadStageLayerNormWeights(int stage);
    
private:
    std::string weights_dir_;
    const int embed_dims_[4] = {32, 64, 160, 256};
    const int patch_sizes_[4] = {7, 3, 3, 3};
    const int strides_[4] = {4, 2, 2, 2};
    const int mlp_ratios_[4] = {4, 4, 4, 4};
    const int sr_ratios_[4] = {8, 4, 2, 1};
};

#endif // WEIGHT_LOADER_HPP
