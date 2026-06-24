#ifndef TRANSFORMER_BLOCK_HPP
#define TRANSFORMER_BLOCK_HPP

#include <vector>
#include <string>
using namespace std;

struct TransformerBlockWeights {
    int embed_dim;
    int mlp_hidden_dim;
    int sr_ratio;
    
    vector<float> ln1_weight;
    vector<float> ln1_bias;
    vector<float> ln2_weight;
    vector<float> ln2_bias;
    
    vector<float> attn_q_weight;
    vector<float> attn_q_bias;
    vector<float> attn_k_weight;
    vector<float> attn_k_bias;
    vector<float> attn_v_weight;
    vector<float> attn_v_bias;
    
    vector<float> attn_sr_weight;
    vector<float> attn_sr_bias;
    vector<float> attn_ln_weight;
    vector<float> attn_ln_bias;
    
    vector<float> attn_proj_weight;
    vector<float> attn_proj_bias;
    
    vector<float> mlp_fc1_weight;
    vector<float> mlp_fc1_bias;
    vector<float> mlp_dwconv_weight;
    vector<float> mlp_dwconv_bias;
    vector<float> mlp_fc2_weight;
    vector<float> mlp_fc2_bias;
};

class TransformerBlock {
public:
    TransformerBlock(int stage, int block, int embed_dim, int mlp_ratio, int sr_ratio);
    
    void loadWeights(const TransformerBlockWeights& weights);
    vector<float> forward(const vector<float>& input, int H, int W);
    
private:
    int stage_;
    int block_;
    int embed_dim_;
    int mlp_hidden_dim_;
    int sr_ratio_;
    int num_heads_;
    int head_dim_;
    bool weights_loaded_;
    
    // Weights
    vector<float> ln1_weight_;
    vector<float> ln1_bias_;
    vector<float> ln2_weight_;
    vector<float> ln2_bias_;
    
    vector<float> attn_q_weight_;
    vector<float> attn_q_bias_;
    vector<float> attn_k_weight_;
    vector<float> attn_k_bias_;
    vector<float> attn_v_weight_;
    vector<float> attn_v_bias_;
    
    vector<float> attn_sr_weight_;
    vector<float> attn_sr_bias_;
    vector<float> attn_ln_weight_;
    vector<float> attn_ln_bias_;
    
    vector<float> attn_proj_weight_;
    vector<float> attn_proj_bias_;
    
    vector<float> mlp_fc1_weight_;
    vector<float> mlp_fc1_bias_;
    vector<float> mlp_dwconv_weight_;
    vector<float> mlp_dwconv_bias_;
    vector<float> mlp_fc2_weight_;
    vector<float> mlp_fc2_bias_;
    
    // Helper functions
    vector<float> layerNorm(const vector<float>& input, const vector<float>& weight, const vector<float>& bias);
    
    vector<float> spatialReduction(const vector<float>& input, int H, int W, int sr_ratio);
    
    vector<float> fullyConnected(const vector<float>& input, const vector<float>& weight, const vector<float>& bias, int in_features, int out_features);
    
    void softmax(vector<float>& data, int num_rows, int num_cols);
    
    vector<float> attention(const vector<float>& input, int H, int W);
    vector<float> mlp(const vector<float>& input, int H, int W);
};

#endif // TRANSFORMER_BLOCK_HPP
