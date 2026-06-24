#ifndef WEIGHT_LOADER_HPP
#define WEIGHT_LOADER_HPP

#include <string>
#include <vector>
#include <iostream>
#include "transformer_block.hpp"
using namespace std;

struct PatchEmbedWeights {
    vector<float> proj_weight;
    vector<float> proj_bias;
    vector<float> ln_weight;
    vector<float> ln_bias;
    
    int in_channels;
    int embed_dim;
    int patch_size;
    int stride;
};

struct StageLayerNormWeights {
    vector<float> weight;
    vector<float> bias;
};

class WeightLoader {
public:
    WeightLoader(const string& weights_dir);
    vector<float> loadWeight(const string& filename, int expected_count = -1);    
    PatchEmbedWeights loadPatchEmbedWeights(int stage);
    TransformerBlockWeights loadTransformerBlockWeights(int stage, int block);
    StageLayerNormWeights loadStageLayerNormWeights(int stage);
    
private:
    string weights_dir_;
    const int embed_dims_[4] = {32, 64, 160, 256};
    const int patch_sizes_[4] = {7, 3, 3, 3};
    const int strides_[4] = {4, 2, 2, 2};
    const int mlp_ratios_[4] = {4, 4, 4, 4};
    const int sr_ratios_[4] = {8, 4, 2, 1};
};

#endif // WEIGHT_LOADER_HPP