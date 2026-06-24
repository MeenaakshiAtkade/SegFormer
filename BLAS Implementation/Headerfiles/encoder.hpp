#ifndef ENCODER_HPP
#define ENCODER_HPP

#include <vector>
#include <string>
#include "patch_embedding.hpp"
#include "transformer_block.hpp"
#include "weight_loader.hpp"

class Encoder {
public:
    Encoder();
    ~Encoder();
    
    void loadWeights(WeightLoader& loader);
    std::vector<std::vector<float>> forward(const std::vector<float>& input);
    
    const std::vector<std::vector<float>>& getEncoderOutputs() const {
        return encoder_outputs_;
    }

private:
    // Patch embeddings for each stage
    PatchEmbedding* patch_embed_0_;
    PatchEmbedding* patch_embed_1_;
    PatchEmbedding* patch_embed_2_;
    PatchEmbedding* patch_embed_3_;
    
    // Transformer blocks for each stage (2 blocks per stage)
    TransformerBlock* tb_0_0_;
    TransformerBlock* tb_0_1_;
    TransformerBlock* tb_1_0_;
    TransformerBlock* tb_1_1_;
    TransformerBlock* tb_2_0_;
    TransformerBlock* tb_2_1_;
    TransformerBlock* tb_3_0_;
    TransformerBlock* tb_3_1_;
    
    // Stage layer normalization weights 
    std::vector<float> stage_ln_weight_[4];
    std::vector<float> stage_ln_bias_[4];
    
    // Store encoder outputs for each stage
    std::vector<std::vector<float>> encoder_outputs_;
    
    // Helper functions
    std::vector<float> hwcToNCHW(const std::vector<float>& hwc, int H, int W, int C);
    std::vector<float> tokensToNCHW(const std::vector<float>& tokens, int H, int W, int C);
    std::vector<float> stageLayerNorm(const std::vector<float>& input, const std::vector<float>& weight, const std::vector<float>& bias, int H, int W, int C);
    std::vector<float> processStage(int stage, const std::vector<float>& input, int H_in, int W_in, int C_in);
};

#endif // ENCODER_HPP
