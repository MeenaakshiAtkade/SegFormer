#ifndef PATCH_EMBEDDING_HPP
#define PATCH_EMBEDDING_HPP

#include <vector>
#include <string>
#include "weight_loader.hpp"

class PatchEmbedding {
public:
    PatchEmbedding(int stage, int in_channels, int embed_dim, int patch_size, int stride);

    void loadWeights(const PatchEmbedWeights& weights);
    std::vector<float> forward(const std::vector<float>& input, int H, int W);

    int getOutputHeight() const { return output_height_; }
    int getOutputWidth() const { return output_width_; }
    int getEmbedDim() const { return embed_dim_; }

private:
    int stage_;
    int in_channels_;
    int embed_dim_;
    int patch_size_;
    int stride_;
    int output_height_;
    int output_width_;

    std::vector<float> proj_weight_; 
    std::vector<float> proj_bias_;
    std::vector<float> ln_weight_;
    std::vector<float> ln_bias_;

    std::vector<float> conv2d_nchw(const std::vector<float>& input, int H, int W);
    std::vector<float> layerNorm(const std::vector<float>& input);
};

#endif
