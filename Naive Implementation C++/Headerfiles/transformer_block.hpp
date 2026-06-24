#ifndef TRANSFORMER_BLOCK_HPP
#define TRANSFORMER_BLOCK_HPP

#include <vector>
#include <string>
#include "weight_loader.hpp"

class TransformerBlock {
public:
    TransformerBlock(int stage, int block, int embed_dim, int mlp_ratio, int sr_ratio);
    
    void loadWeights(const TransformerBlockWeights& weights);
    std::vector<float> forward(const std::vector<float>& input, int H, int W);
    
private:
    int stage_;
    int block_;
    int embed_dim_;
    int mlp_hidden_dim_;
    int sr_ratio_;
    int num_heads_;
    int head_dim_;
    
    // Attention weights
    std::vector<float> attn_q_weight_;
    std::vector<float> attn_q_bias_;
    std::vector<float> attn_k_weight_;
    std::vector<float> attn_k_bias_;
    std::vector<float> attn_v_weight_;
    std::vector<float> attn_v_bias_;
    std::vector<float> attn_sr_weight_;
    std::vector<float> attn_sr_bias_;
    std::vector<float> attn_ln_weight_;
    std::vector<float> attn_ln_bias_;
    std::vector<float> attn_proj_weight_;
    std::vector<float> attn_proj_bias_;
    
    // LayerNorm weights
    std::vector<float> ln1_weight_;
    std::vector<float> ln1_bias_;
    std::vector<float> ln2_weight_;
    std::vector<float> ln2_bias_;
    
    // MLP weights
    std::vector<float> mlp_fc1_weight_;
    std::vector<float> mlp_fc1_bias_;
    std::vector<float> mlp_dwconv_weight_;
    std::vector<float> mlp_dwconv_bias_;
    std::vector<float> mlp_fc2_weight_;
    std::vector<float> mlp_fc2_bias_;
    
    bool weights_loaded_;
    
    std::vector<float> layerNorm(const std::vector<float>& input, const std::vector<float>& weight, const std::vector<float>& bias);
    std::vector<float> attention(const std::vector<float>& input, int H, int W);
    std::vector<float> mlp(const std::vector<float>& input, int H, int W);
    std::vector<float> spatialReduction(const std::vector<float>& input, int H, int W, int sr_ratio);
    std::vector<float> fullyConnected(const std::vector<float>& input, const std::vector<float>& weight, const std::vector<float>& bias, int in_features, int out_features);
    std::vector<float> depthwiseConv3x3(const std::vector<float>& input, int H, int W);
    std::vector<float> gelu(const std::vector<float>& input);
    void softmax(std::vector<float>& data, int num_rows, int num_cols);
};

#endif // TRANSFORMER_BLOCK_HPP
