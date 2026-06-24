#include "../Headerfiles/transformer_block.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <limits>

// -------------------- Utility --------------------
static inline float gelu_scalar(float x) {
    constexpr float INV_SQRT_2PI = 0.797f;
    return 0.5f * x * (1.0f + std::tanh(
        INV_SQRT_2PI * (x + 0.0447f * x * x * x)));
}

// -------------------- Constructor --------------------
TransformerBlock::TransformerBlock(int stage, int block, int embed_dim, int mlp_ratio, int sr_ratio)
    : stage_(stage),block_(block), embed_dim_(embed_dim), mlp_hidden_dim_(embed_dim * mlp_ratio), sr_ratio_(sr_ratio), num_heads_(embed_dim == 32 ? 1 : embed_dim == 64 ? 2 : embed_dim == 160 ? 5 : 8), head_dim_(embed_dim / num_heads_), weights_loaded_(false) {
}
// -------------------- Load Weights --------------------
void TransformerBlock::loadWeights(const TransformerBlockWeights& weights) {
    ln1_weight_ = weights.ln1_weight;
    ln1_bias_ = weights.ln1_bias;
    ln2_weight_ = weights.ln2_weight;
    ln2_bias_ = weights.ln2_bias;
    
    attn_q_weight_ = weights.attn_q_weight;
    attn_q_bias_ = weights.attn_q_bias;
    attn_k_weight_ = weights.attn_k_weight;
    attn_k_bias_ = weights.attn_k_bias;
    attn_v_weight_ = weights.attn_v_weight;
    attn_v_bias_ = weights.attn_v_bias;
    
    attn_sr_weight_ = weights.attn_sr_weight;
    attn_sr_bias_ = weights.attn_sr_bias;
    attn_ln_weight_ = weights.attn_ln_weight;
    attn_ln_bias_ = weights.attn_ln_bias;
    
    attn_proj_weight_ = weights.attn_proj_weight;
    attn_proj_bias_ = weights.attn_proj_bias;
    
    mlp_fc1_weight_ = weights.mlp_fc1_weight;
    mlp_fc1_bias_ = weights.mlp_fc1_bias;
    mlp_dwconv_weight_ = weights.mlp_dwconv_weight;
    mlp_dwconv_bias_ = weights.mlp_dwconv_bias;
    mlp_fc2_weight_ = weights.mlp_fc2_weight;
    mlp_fc2_bias_ = weights.mlp_fc2_bias;
    
    weights_loaded_ = true;
}

// -------------------- Layer Normalization --------------------

std::vector<float> TransformerBlock::layerNorm(const std::vector<float>& input, const std::vector<float>& weight, const std::vector<float>& bias) {
    
    std::vector<float> output = input;
    int C = weight.size();
    int HW = input.size() / C;
    const float eps = 1e-5f;
    
    // Normalize each token (spatial position)
    for (int idx = 0; idx < HW; ++idx) {
        int base = idx * C;
        
        double mean = 0.0;
        for (int c = 0; c < C; ++c) {
            mean += static_cast<double>(input[base + c]);
        }
        mean /= C;
        
        double var = 0.0;
        for (int c = 0; c < C; ++c) {
            double diff = static_cast<double>(input[base + c]) - mean;
            var += diff * diff;
        }
        var /= C;
        
        double std = std::sqrt(var + static_cast<double>(eps));
        
        for (int c = 0; c < C; ++c) {
            double normalized = (static_cast<double>(input[base + c]) - mean) / std;
            output[base + c] = static_cast<float>(
                normalized * static_cast<double>(weight[c]) + 
                static_cast<double>(bias[c]));
        }
    }
    
    return output;
}

// -------------------- Spatial Reduction --------------------
std::vector<float> TransformerBlock::spatialReduction(const std::vector<float>& input, int H, int W, int sr_ratio) {
    
    if (sr_ratio == 1) {
        return input;
    }
    
    int H_reduced = H / sr_ratio;
    int W_reduced = W / sr_ratio;
    int C = embed_dim_;
    
    // HWC tokens to NCHW for convolution
    std::vector<float> nchw(C * H * W);
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int c = 0; c < C; ++c) {
                int hwc_idx = h * W * C + w * C + c;
                int nchw_idx = c * H * W + h * W + w;
                nchw[nchw_idx] = input[hwc_idx];
            }
        }
    }
    
    // Spatial reduction convolution (NCHW input)
    std::vector<float> reduced_nchw(C * H_reduced * W_reduced, 0.0f);
    
    for (int oc = 0; oc < C; ++oc) {
        for (int oh = 0; oh < H_reduced; ++oh) {
            for (int ow = 0; ow < W_reduced; ++ow) {
                double sum = static_cast<double>(attn_sr_bias_[oc]);
                
                for (int ic = 0; ic < C; ++ic) {
                    for (int kh = 0; kh < sr_ratio; ++kh) {
                        for (int kw = 0; kw < sr_ratio; ++kw) {
                            int ih = oh * sr_ratio + kh;
                            int iw = ow * sr_ratio + kw;
                            
                            if (ih < H && iw < W) {
                                int input_idx = ic * H * W + ih * W + iw;
                                float input_val = nchw[input_idx];
                                
                                int weight_idx = oc * (C * sr_ratio * sr_ratio) + ic * (sr_ratio * sr_ratio) + kh * sr_ratio + kw;
                                sum += static_cast<double>(input_val) * static_cast<double>(attn_sr_weight_[weight_idx]);
                            }
                        }
                    }
                }
                
                int out_idx = oc * H_reduced * W_reduced + oh * W_reduced + ow;
                reduced_nchw[out_idx] = static_cast<float>(sum);
            }
        }
    }
    
    // Convert NCHW back to HWC tokens
    std::vector<float> reduced_hwc(H_reduced * W_reduced * C);
    for (int h = 0; h < H_reduced; ++h) {
        for (int w = 0; w < W_reduced; ++w) {
            for (int c = 0; c < C; ++c) {
                int nchw_idx = c * H_reduced * W_reduced + h * W_reduced + w;
                int hwc_idx = h * W_reduced * C + w * C + c;
                reduced_hwc[hwc_idx] = reduced_nchw[nchw_idx];
            }
        }
    }
    
    // Apply LayerNorm
    reduced_hwc = layerNorm(reduced_hwc, attn_ln_weight_, attn_ln_bias_);
    
    return reduced_hwc;
}

// -------------------- Fully Connected --------------------
std::vector<float> TransformerBlock::fullyConnected(const std::vector<float>& input, const std::vector<float>& weight, const std::vector<float>& bias, int in_features, int out_features) {
    
    int HW = input.size() / in_features;
    std::vector<float> output(HW * out_features);
    
    for (int i = 0; i < HW; ++i) {
        for (int j = 0; j < out_features; ++j) {
            double sum = static_cast<double>(bias[j]);
            for (int k = 0; k < in_features; ++k) {
                sum += static_cast<double>(input[i * in_features + k]) * static_cast<double>(weight[j * in_features + k]);
            }
            output[i * out_features + j] = static_cast<float>(sum);
        }
    }
    
    return output;
}

// -------------------- Softmax --------------------
void TransformerBlock::softmax(std::vector<float>& data, int num_rows, int num_cols) {
    for (int i = 0; i < num_rows; ++i) {
        float max_val = *std::max_element(data.begin() + i * num_cols, data.begin() + (i + 1) * num_cols);
        double sum = 0.0;
        
        for (int j = 0; j < num_cols; ++j) {
            data[i * num_cols + j] = std::exp(data[i * num_cols + j] - max_val);
            sum += static_cast<double>(data[i * num_cols + j]);
        }
        
        for (int j = 0; j < num_cols; ++j) {
            data[i * num_cols + j] = static_cast<float>(static_cast<double>(data[i * num_cols + j]) / sum);
        }
    }
}

// -------------------- Attention --------------------
std::vector<float> TransformerBlock::attention(const std::vector<float>& input, int H, int W) {
    
    const int N = H * W;
    const int head_dim = embed_dim_ / num_heads_;
    
    std::vector<float> Q = fullyConnected(input, attn_q_weight_, attn_q_bias_, embed_dim_, embed_dim_);
    
    // Spatial reduction for K,V
    std::vector<float> reduced_input = spatialReduction(input, H, W, sr_ratio_);
    int H_reduced = H / sr_ratio_;
    int W_reduced = W / sr_ratio_;
    int N_reduced = H_reduced * W_reduced;
    
    // K,V: (N_reduced, embed_dim)
    std::vector<float> K = fullyConnected(reduced_input, attn_k_weight_, attn_k_bias_, embed_dim_, embed_dim_);
    std::vector<float> V = fullyConnected(reduced_input, attn_v_weight_, attn_v_bias_, embed_dim_, embed_dim_);

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    std::vector<float> attn_output(N * embed_dim_, 0.0f);
    
    // Process each head separately
    for (int h = 0; h < num_heads_; ++h) {
        
        std::vector<float> attn_scores(N * N_reduced);
        
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N_reduced; ++j) {
                double sum = 0.0;
                for (int d = 0; d < head_dim; ++d) {
                    int q_idx = i * embed_dim_ + h * head_dim + d;
                    int k_idx = j * embed_dim_ + h * head_dim + d;
                    sum += static_cast<double>(Q[q_idx]) * static_cast<double>(K[k_idx]);
                }
                attn_scores[i * N_reduced + j] = static_cast<float>(sum * scale);
            }
        }
        
        // Softmax over keys
        softmax(attn_scores, N, N_reduced);
        
        for (int i = 0; i < N; ++i) {
            for (int d = 0; d < head_dim; ++d) {
                double sum = 0.0;
                for (int j = 0; j < N_reduced; ++j) {
                    int v_idx = j * embed_dim_ + h * head_dim + d;
                    sum += static_cast<double>(attn_scores[i * N_reduced + j]) * static_cast<double>(V[v_idx]);
                }
                attn_output[i * embed_dim_ + h * head_dim + d] = static_cast<float>(sum);
            }
        }
    }
    
    // Project concatenated heads
    attn_output = fullyConnected(attn_output, attn_proj_weight_, attn_proj_bias_, embed_dim_, embed_dim_);
    
    return attn_output;
}

// -------------------- MLP --------------------

std::vector<float> TransformerBlock::mlp(const std::vector<float>& input, int H, int W) {

    const int N = H * W;
    const int C_hidden = mlp_hidden_dim_;
    const int C_in = embed_dim_;

    // Fully Connected 1: (N, C_in) -> (N, C_hidden)
    std::vector<float> x1 = fullyConnected(input, mlp_fc1_weight_, mlp_fc1_bias_, C_in, C_hidden);

    // Convert HWC tokens to NCHW for depthwise conv
    std::vector<float> nchw(C_hidden * H * W);
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int token_idx = h * W + w;
            for (int c = 0; c < C_hidden; ++c) {
                int hwc_idx = token_idx * C_hidden + c;
                int nchw_idx = c * H * W + h * W + w;
                nchw[nchw_idx] = x1[hwc_idx];
            }
        }
    }

    // Depthwise Conv 3×3 with GELU
    std::vector<float> dw_nchw(C_hidden * H * W);
    for (int c = 0; c < C_hidden; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                double sum = static_cast<double>(mlp_dwconv_bias_[c]);
                
                for (int kh = -1; kh <= 1; ++kh) {
                    for (int kw = -1; kw <= 1; ++kw) {
                        int ih = h + kh;
                        int iw = w + kw;
                        
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            int kernel_idx = (kh + 1) * 3 + (kw + 1);
                            int input_idx = c * H * W + ih * W + iw;
                            sum += static_cast<double>(nchw[input_idx]) * static_cast<double>(mlp_dwconv_weight_[c * 9 + kernel_idx]);
                        }
                    }
                }
                
                int out_idx = c * H * W + h * W + w;
                dw_nchw[out_idx] = gelu_scalar(static_cast<float>(sum));
            }
        }
    }

    // Convert NCHW back to HWC tokens
    std::vector<float> x2(N * C_hidden);
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int token_idx = h * W + w;
            for (int c = 0; c < C_hidden; ++c) {
                int nchw_idx = c * H * W + h * W + w;
                int hwc_idx = token_idx * C_hidden + c;
                x2[hwc_idx] = dw_nchw[nchw_idx];
            }
        }
    }

    // Fully Connected
    std::vector<float> out = fullyConnected(x2, mlp_fc2_weight_, mlp_fc2_bias_, C_hidden, C_in);

    return out;
}

// -------------------- Forward --------------------
std::vector<float> TransformerBlock::forward( const std::vector<float>& input, int H, int W) {
    
    if (!weights_loaded_) {
        throw std::runtime_error("[ERROR] Weights not loaded for TransformerBlock " + std::to_string(stage_) + "_" + std::to_string(block_));
    }
    
    const int N = H * W;
    std::string prefix = "s" + std::to_string(stage_) + "_b" + std::to_string(block_);
        
    // Layer Normalization 
    std::vector<float> x_norm1 = layerNorm(input, ln1_weight_, ln1_bias_);
        
    // Attention
    std::vector<float> attn_out = attention(x_norm1, H, W);
        
    // Residual 
    std::vector<float> x1(N * embed_dim_);
    for (int i = 0; i < N * embed_dim_; ++i) {
        x1[i] = input[i] + attn_out[i];
    }
        
    // Layer Normalization
    std::vector<float> x_norm2 = layerNorm(x1, ln2_weight_, ln2_bias_);
    
    // MLP
    std::vector<float> mlp_out = mlp(x_norm2, H, W);
        
    // Residual 
    std::vector<float> out(N * embed_dim_);
    for (int i = 0; i < N * embed_dim_; ++i) {
        out[i] = x1[i] + mlp_out[i];
    }
        
    return out;
}
