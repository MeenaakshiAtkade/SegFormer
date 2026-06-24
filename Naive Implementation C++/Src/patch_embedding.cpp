#include "../Headerfiles/patch_embedding.hpp"
#include <fstream>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <iostream>

PatchEmbedding::PatchEmbedding(int stage, int in_channels, int embed_dim, int patch_size, int stride) : stage_(stage), in_channels_(in_channels), embed_dim_(embed_dim), patch_size_(patch_size), stride_(stride), output_height_(0), output_width_(0) {}

void PatchEmbedding::loadWeights(const PatchEmbedWeights& w) {
    proj_weight_ = w.proj_weight; // [out, in, kH, kW]
    proj_bias_   = w.proj_bias;
    ln_weight_   = w.ln_weight;
    ln_bias_     = w.ln_bias;
}

std::vector<float> PatchEmbedding::conv2d_nchw( const std::vector<float>& input, int H, int W)
{
    const int pad = (patch_size_ - 1) / 2;

    output_height_ = (H + 2 * pad - patch_size_) / stride_ + 1;
    output_width_  = (W + 2 * pad - patch_size_) / stride_ + 1;

    std::vector<float> out(embed_dim_ * output_height_ * output_width_, 0.f);

    // Conv2d with NCHW layout
    for (int oc = 0; oc < embed_dim_; ++oc) {
        for (int oh = 0; oh < output_height_; ++oh) {
            for (int ow = 0; ow < output_width_; ++ow) {

                double sum = static_cast<double>(proj_bias_[oc]);

                for (int ic = 0; ic < in_channels_; ++ic) {
                    for (int kh = 0; kh < patch_size_; ++kh) {
                        for (int kw = 0; kw < patch_size_; ++kw) {

                            int ih = oh * stride_ + kh - pad;
                            int iw = ow * stride_ + kw - pad;

                            if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                                continue;
                                
                            int in_idx = ic * H * W + ih * W + iw;
                            int w_idx = oc * (in_channels_ * patch_size_ * patch_size_) + ic * (patch_size_ * patch_size_) + kh * patch_size_ + kw;

                            sum += static_cast<double>(input[in_idx]) * static_cast<double>(proj_weight_[w_idx]);
                        }
                    }
                }
                int out_idx = oc * output_height_ * output_width_ + oh * output_width_ + ow;
                out[out_idx] = static_cast<float>(sum);
            }
        }
    }

    return out;
}

std::vector<float> PatchEmbedding::layerNorm(const std::vector<float>& x)
{
    const int H = output_height_;
    const int W = output_width_;
    const int C = embed_dim_;
    const float eps = 1e-5f;

    std::vector<float> out(H * W * C);

    // Normalize per spatial position (h,w) across channels
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {

            // Compute mean across channels for this spatial position
            double mean = 0.0;
            for (int c = 0; c < C; ++c) {
                int nchw_idx = c * H * W + h * W + w;
                mean += static_cast<double>(x[nchw_idx]);
            }
            mean /= C;

            // Compute variance across channels for this spatial position
            double var = 0.0;
            for (int c = 0; c < C; ++c) {
                int nchw_idx = c * H * W + h * W + w;
                double diff = static_cast<double>(x[nchw_idx]) - mean;
                var += diff * diff;
            }
            var /= C;

            double inv_std = 1.0 / std::sqrt(var + static_cast<double>(eps));

            // Normalize 
            for (int c = 0; c < C; ++c) {
                int nchw_idx = c * H * W + h * W + w;
                int hwc_idx = h * W * C + w * C + c;
                
                double v = static_cast<double>(x[nchw_idx]);
                double normalized = (v - mean) * inv_std;
                
                out[hwc_idx] = static_cast<float>( normalized * static_cast<double>(ln_weight_[c]) + static_cast<double>(ln_bias_[c]));
            }
        }
    }

    return out;
}

std::vector<float> PatchEmbedding::forward(const std::vector<float>& input, int H, int W)
{
    // Conv2d:
    auto conv_out = conv2d_nchw(input, H, W);
    
    // LayerNorm
    auto token_out = layerNorm(conv_out);
    
    return token_out;
}
