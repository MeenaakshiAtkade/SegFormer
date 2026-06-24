#include "../Headerfiles/weight_loader.hpp"
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>

WeightLoader::WeightLoader(const std::string& weights_dir) 
    : weights_dir_(weights_dir) {
    if (weights_dir_.back() != '/' && weights_dir_.back() != '\\') {
        weights_dir_ += "/";
    }
}

std::vector<float> WeightLoader::loadWeight(const std::string& filename, int expected_count) {
    std::string full_path = weights_dir_ + filename;
    std::ifstream file(full_path, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("[ERROR] Could not open weight file: " + full_path);
    }
    
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    size_t count = file_size / sizeof(float);
    
    if (expected_count > 0 && count != static_cast<size_t>(expected_count)) {
        throw std::runtime_error("[ERROR] Weight count mismatch in " + filename + ". Expected: " + std::to_string(expected_count) + ", Got: " + std::to_string(count));
    }
    
    std::vector<float> weights(count);
    file.read(reinterpret_cast<char*>(weights.data()), file_size);
    file.close();
    
    return weights;
}

PatchEmbedWeights WeightLoader::loadPatchEmbedWeights(int stage) {
    PatchEmbedWeights weights;
    
    weights.in_channels = (stage == 0) ? 3 : embed_dims_[stage - 1];
    weights.embed_dim = embed_dims_[stage];
    weights.patch_size = patch_sizes_[stage];
    weights.stride = strides_[stage];
    
    int proj_weight_count = weights.embed_dim * weights.in_channels * 
                           weights.patch_size * weights.patch_size;
    
    std::string stage_str = std::to_string(stage);
    
    // Load weights using exact naming from model_info.txt
    weights.proj_weight = loadWeight("pe_" + stage_str + "_proj_weight.bin", proj_weight_count);
    weights.proj_bias = loadWeight("pe_" + stage_str + "_proj_bias.bin", weights.embed_dim);
    weights.ln_weight = loadWeight("pe_" + stage_str + "_layer_norm_weight.bin", weights.embed_dim);
    weights.ln_bias = loadWeight("pe_" + stage_str + "_layer_norm_bias.bin", weights.embed_dim);
    
    return weights;
}
StageLayerNormWeights WeightLoader::loadStageLayerNormWeights(int stage) {
    StageLayerNormWeights weights;
    
    // Dimensions based on stage
    int embed_dim = 0;
    if (stage == 0) embed_dim = 32;
    else if (stage == 1) embed_dim = 64;
    else if (stage == 2) embed_dim = 160;
    else if (stage == 3) embed_dim = 256;
    
    // Load weight and bias
    std::string weight_file = "encoder_layer_norm_" + std::to_string(stage) + "_weight.bin";
    std::string bias_file = "encoder_layer_norm_" + std::to_string(stage) + "_bias.bin";

    weights.weight = loadWeight(weight_file, embed_dim);
    weights.bias = loadWeight(bias_file, embed_dim);
        
    return weights;
}

TransformerBlockWeights WeightLoader::loadTransformerBlockWeights(int stage, int block) {
    TransformerBlockWeights weights;
    
    weights.embed_dim = embed_dims_[stage];
    weights.mlp_hidden_dim = weights.embed_dim * mlp_ratios_[stage];
    weights.sr_ratio = sr_ratios_[stage];
    
    // Naming: encoder_block_{stage}_{block}_{layer}
    std::string prefix = "encoder_block_" + std::to_string(stage) + "_" + std::to_string(block) + "_";
    
    // Load LayerNorm 1
    weights.ln1_weight = loadWeight(prefix + "layer_norm_1_weight.bin", weights.embed_dim);
    weights.ln1_bias = loadWeight(prefix + "layer_norm_1_bias.bin", weights.embed_dim);
    
    // Load Attention weights (Q, K, V)
    weights.attn_q_weight = loadWeight(prefix + "attention_query_weight.bin", weights.embed_dim * weights.embed_dim);
    weights.attn_q_bias = loadWeight(prefix + "attention_query_bias.bin", weights.embed_dim);
    
    weights.attn_k_weight = loadWeight(prefix + "attention_key_weight.bin", weights.embed_dim * weights.embed_dim);
    weights.attn_k_bias = loadWeight(prefix + "attention_key_bias.bin", weights.embed_dim);
    
    weights.attn_v_weight = loadWeight(prefix + "attention_value_weight.bin", weights.embed_dim * weights.embed_dim);
    weights.attn_v_bias = loadWeight(prefix + "attention_value_bias.bin", weights.embed_dim);
    
    // Load Spatial Reduction weights (only for stages 0, 1, 2 with sr_ratio > 1)
    if (weights.sr_ratio > 1) {
        int sr_weight_count = weights.embed_dim * weights.embed_dim * weights.sr_ratio * weights.sr_ratio;
        weights.attn_sr_weight = loadWeight(prefix + "attention_sr_weight.bin", sr_weight_count);
        weights.attn_sr_bias = loadWeight(prefix + "attention_sr_bias.bin", weights.embed_dim);
        weights.attn_ln_weight = loadWeight(prefix + "attention_layer_norm_weight.bin", weights.embed_dim);
        weights.attn_ln_bias = loadWeight(prefix + "attention_layer_norm_bias.bin", weights.embed_dim);
    } else {
        // Stage 3 has no spatial reduction, initialize empty vectors
        weights.attn_sr_weight.clear();
        weights.attn_sr_bias.clear();
        weights.attn_ln_weight.clear();
        weights.attn_ln_bias.clear();
    }
    
    // Load Attention output projection
    weights.attn_proj_weight = loadWeight(prefix + "attention_output_dense_weight.bin", weights.embed_dim * weights.embed_dim);
    weights.attn_proj_bias = loadWeight(prefix + "attention_output_dense_bias.bin", weights.embed_dim);
    
    // Load LayerNorm 2
    weights.ln2_weight = loadWeight(prefix + "layer_norm_2_weight.bin", weights.embed_dim);
    weights.ln2_bias = loadWeight(prefix + "layer_norm_2_bias.bin", weights.embed_dim);
    
    // Load MLP weights
    weights.mlp_fc1_weight = loadWeight(prefix + "mlp_dense1_weight.bin", weights.mlp_hidden_dim * weights.embed_dim);
    weights.mlp_fc1_bias = loadWeight(prefix + "mlp_dense1_bias.bin", weights.mlp_hidden_dim);
    
    weights.mlp_dwconv_weight = loadWeight(prefix + "mlp_dwconv_dwconv_weight.bin", weights.mlp_hidden_dim * 9);
    weights.mlp_dwconv_bias = loadWeight(prefix + "mlp_dwconv_dwconv_bias.bin", weights.mlp_hidden_dim);
    
    weights.mlp_fc2_weight = loadWeight(prefix + "mlp_dense2_weight.bin", weights.embed_dim * weights.mlp_hidden_dim);
    weights.mlp_fc2_bias = loadWeight(prefix + "mlp_dense2_bias.bin", weights.embed_dim);
    
    return weights;
}
