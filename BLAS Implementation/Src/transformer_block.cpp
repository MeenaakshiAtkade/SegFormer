#include "../Headerfiles/transformer_block.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <cblas.h>
#include <omp.h>
using namespace std;

// -------------------- Utility --------------------
static inline float gelu_scalar(float x) {
    constexpr float INV_SQRT_2PI = 0.797f;
    return 0.5f * x * (1.0f + tanh(INV_SQRT_2PI * (x + 0.0447f * x * x * x)));
}

// -------------------- Constructor --------------------
TransformerBlock::TransformerBlock( int stage, int block, int embed_dim, int mlp_ratio, int sr_ratio) : stage_(stage), block_(block), embed_dim_(embed_dim), mlp_hidden_dim_(embed_dim * mlp_ratio), sr_ratio_(sr_ratio), num_heads_(embed_dim == 32 ? 1 : embed_dim == 64 ? 2 : embed_dim == 160 ? 5 : 8), head_dim_(embed_dim / num_heads_), weights_loaded_(false) 
{
    
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
vector<float> TransformerBlock::layerNorm(
    const vector<float>& input,
    const vector<float>& weight,
    const vector<float>& bias) {
    
    vector<float> output = input;
    int C = weight.size();
    int HW = input.size() / C;
    const float eps = 1e-5f;
    
    // Normalize each token (spatial position)
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
            double normalized = (static_cast<float>(input[base + c]) - mean) / std;
            output[base + c] = static_cast<float>(
                normalized * static_cast<float>(weight[c]) + 
                static_cast<float>(bias[c]));
        }
    }
    
    return output;
}

// -------------------- Spatial Reduction --------------------
vector<float> TransformerBlock::spatialReduction( const vector<float>& input, int H, int W, int sr_ratio) {

    if (sr_ratio == 1) return input;
    int H_reduced = H / sr_ratio;
    int W_reduced = W / sr_ratio;
    int C = embed_dim_;

    vector<float> nchw(C * H * W);

    // HWC to NCHW
    #pragma omp parallel for collapse(2)
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int c = 0; c < C; ++c) {
                int hwc_idx = h * W * C + w * C + c;
                int nchw_idx = c * H * W + h * W + w;
                nchw[nchw_idx] = input[hwc_idx];
            }
        }
    }

    vector<float> reduced_nchw(C * H_reduced * W_reduced, 0.0f);

    // OpenMP structure
    #pragma omp parallel for collapse(3)
    for (int oc = 0; oc < C; ++oc) {
        for (int oh = 0; oh < H_reduced; ++oh) {
            for (int ow = 0; ow < W_reduced; ++ow) {

                float sum = attn_sr_bias_[oc];

                for (int ic = 0; ic < C; ++ic) {
                    for (int kh = 0; kh < sr_ratio; ++kh) {
                        for (int kw = 0; kw < sr_ratio; ++kw) {

                            int ih = oh * sr_ratio + kh;
                            int iw = ow * sr_ratio + kw;

                            if (ih < H && iw < W) {
                                int input_idx = ic * H * W + ih * W + iw;

                                int weight_idx =
                                    oc * (C * sr_ratio * sr_ratio) +
                                    ic * (sr_ratio * sr_ratio) +
                                    kh * sr_ratio + kw;

                                sum += nchw[input_idx] *
                                       attn_sr_weight_[weight_idx];
                            }
                        }
                    }
                }

                int out_idx = oc * H_reduced * W_reduced +
                              oh * W_reduced + ow;

                reduced_nchw[out_idx] = sum;
            }
        }
    }

    // NCHW to HWC
    vector<float> reduced_hwc(H_reduced * W_reduced * C);

    #pragma omp parallel for collapse(2)
    for (int h = 0; h < H_reduced; ++h) {
        for (int w = 0; w < W_reduced; ++w) {
            for (int c = 0; c < C; ++c) {
                int nchw_idx = c * H_reduced * W_reduced +
                               h * W_reduced + w;
                int hwc_idx = h * W_reduced * C + w * C + c;

                reduced_hwc[hwc_idx] = reduced_nchw[nchw_idx];
            }
        }
    }

    return layerNorm(reduced_hwc, attn_ln_weight_, attn_ln_bias_);
}

// -------------------- Fully Connected --------------------

vector<float> TransformerBlock::fullyConnected(const vector<float>& input, const vector<float>& weight, const vector<float>& bias, int in_features, int out_features) {
    
    int HW = input.size() / in_features;
    vector<float> output(HW * out_features);
    
    // Matrix multiplication: output = input @ weight^T
    cblas_sgemm(
        CblasRowMajor,      
        CblasNoTrans,       
        CblasTrans,         
        HW,               
        out_features,       
        in_features,        
        1.0f,               
        input.data(),       
        in_features,       
        weight.data(),      
        in_features,        
        0.0f,               
        output.data(),      
        out_features        
    );
    
    // Add bias
    for (int i = 0; i < HW; ++i) {
        for (int j = 0; j < out_features; ++j) {
            output[i * out_features + j] += bias[j];
        }
    }
    
    return output;
}

// -------------------- Softmax --------------------
void TransformerBlock::softmax(vector<float>& data, int num_rows, int num_cols) {
    for (int i = 0; i < num_rows; ++i) {
        float max_val = *max_element(data.begin() + i * num_cols,
                                         data.begin() + (i + 1) * num_cols);
        float sum = 0.0;
        
        for (int j = 0; j < num_cols; ++j) {
            data[i * num_cols + j] = exp(data[i * num_cols + j] - max_val);
            sum += static_cast<float>(data[i * num_cols + j]);
        }
        
        for (int j = 0; j < num_cols; ++j) {
            data[i * num_cols + j] = static_cast<float>(
                static_cast<float>(data[i * num_cols + j]) / sum);
        }
    }
}

// -------------------- Attention --------------------
vector<float> TransformerBlock::attention(
    const vector<float>& input, int H, int W) {
    
    const int N = H * W;
    const int head_dim = embed_dim_ / num_heads_;

    // Q: (N, embed_dim)
    vector<float> Q = fullyConnected(input, attn_q_weight_, attn_q_bias_, embed_dim_, embed_dim_);

    // Spatial reduction
    vector<float> reduced_input = spatialReduction(input, H, W, sr_ratio_);
    int H_reduced = H / sr_ratio_;
    int W_reduced = W / sr_ratio_;
    int N_reduced = H_reduced * W_reduced;

    // K, V
    vector<float> K = fullyConnected(reduced_input, attn_k_weight_, attn_k_bias_, embed_dim_, embed_dim_);
    vector<float> V = fullyConnected(reduced_input, attn_v_weight_, attn_v_bias_, embed_dim_, embed_dim_);

    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    vector<float> attn_output(N * embed_dim_, 0.0f);

    // ======== PER HEAD ========
    for (int h = 0; h < num_heads_; ++h) {

        float* Q_h = new float[N * head_dim];
        float* K_h = new float[N_reduced * head_dim];
        float* V_h = new float[N_reduced * head_dim];

        // Extract head slices
        #pragma omp parallel for
        for (int i = 0; i < N; ++i)
            for (int d = 0; d < head_dim; ++d)
                Q_h[i * head_dim + d] = Q[i * embed_dim_ + h * head_dim + d];

        #pragma omp parallel for
        for (int i = 0; i < N_reduced; ++i)
            for (int d = 0; d < head_dim; ++d) {
                K_h[i * head_dim + d] = K[i * embed_dim_ + h * head_dim + d];
                V_h[i * head_dim + d] = V[i * embed_dim_ + h * head_dim + d];
            }

        // ======== GEMM: Q × K^T ========
        vector<float> attn_scores(N * N_reduced);

        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            N,                  
            N_reduced,          
            head_dim,           
            scale,
            Q_h,
            head_dim,
            K_h,
            head_dim,
            0.0f,
            attn_scores.data(),
            N_reduced
        );

        // Softmax
        softmax(attn_scores, N, N_reduced);

        // ======== GEMM: attn × V ========
        float* out_h = new float[N * head_dim];

        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            N,
            head_dim,
            N_reduced,
            1.0f,
            attn_scores.data(),
            N_reduced,
            V_h,
            head_dim,
            0.0f,
            out_h,
            head_dim
        );

        #pragma omp parallel for
        for (int i = 0; i < N; ++i)
            for (int d = 0; d < head_dim; ++d)
                attn_output[i * embed_dim_ + h * head_dim + d] = out_h[i * head_dim + d];
        delete[] Q_h;
        delete[] K_h;
        delete[] V_h;
        delete[] out_h;
    }

    // Final projection
    attn_output = fullyConnected(attn_output, attn_proj_weight_, attn_proj_bias_, embed_dim_, embed_dim_);

    return attn_output;
}

// -------------------- MLP --------------------
// Input/Output: HWC token format
vector<float> TransformerBlock::mlp(
    const vector<float>& input, int H, int W) {

    const int N = H * W;
    const int C_hidden = mlp_hidden_dim_;
    const int C_in = embed_dim_;

    // FC1: (N, C_in) to (N, C_hidden)
    vector<float> x1 = fullyConnected(input, mlp_fc1_weight_, mlp_fc1_bias_, C_in, C_hidden);

    // Convert HWC tokens to NCHW for depthwise conv
    vector<float> nchw(C_hidden * H * W);
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int token_idx = h * W + w;
            #pragma omp parallel for
            for (int c = 0; c < C_hidden; ++c) {
                int hwc_idx = token_idx * C_hidden + c;
                int nchw_idx = c * H * W + h * W + w;
                nchw[nchw_idx] = x1[hwc_idx];
            }
        }
    }

    // Depthwise Conv 3×3 with GELU (operates on NCHW)
    vector<float> dw_nchw(C_hidden * H * W);
    for (int c = 0; c < C_hidden; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                float sum = static_cast<float>(mlp_dwconv_bias_[c]);
                
                for (int kh = -1; kh <= 1; ++kh) {
                    for (int kw = -1; kw <= 1; ++kw) {
                        int ih = h + kh;
                        int iw = w + kw;
                        
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            int kernel_idx = (kh + 1) * 3 + (kw + 1);
                            int input_idx = c * H * W + ih * W + iw;
                            sum += static_cast<float>(nchw[input_idx]) * 
                                   static_cast<float>(mlp_dwconv_weight_[c * 9 + kernel_idx]);
                        }
                    }
                }
                
                int out_idx = c * H * W + h * W + w;
                dw_nchw[out_idx] = gelu_scalar(static_cast<float>(sum));
            }
        }
    }

    // Convert NCHW back to HWC tokens
    vector<float> x2(N * C_hidden);
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

    // FC2: (N, C_hidden) -> (N, C_in) 
    vector<float> out = fullyConnected(x2, mlp_fc2_weight_, mlp_fc2_bias_, C_hidden, C_in);

    return out;
}

// -------------------- Forward --------------------
vector<float> TransformerBlock::forward( const vector<float>& input, int H, int W) {
    if (!weights_loaded_) {
        throw runtime_error("[ERROR] Weights not loaded for TransformerBlock " + to_string(stage_) + "_" + to_string(block_));
    }
    
    const int N = H * W;
        
    // LN1
    vector<float> x_norm1 = layerNorm(input, ln1_weight_, ln1_bias_);
        
    // Attention
    vector<float> attn_out = attention(x_norm1, H, W);
        
    // Residual 1
    vector<float> x1(N * embed_dim_);
    for (int i = 0; i < N * embed_dim_; ++i) {
        x1[i] = input[i] + attn_out[i];
    }
        
    // LN2
    vector<float> x_norm2 = layerNorm(x1, ln2_weight_, ln2_bias_);
    
    // MLP
    vector<float> mlp_out = mlp(x_norm2, H, W);
        
    // Residual 2
    vector<float> out(N * embed_dim_);
    for (int i = 0; i < N * embed_dim_; ++i) {
        out[i] = x1[i] + mlp_out[i];
    }
        
    return out;
}
