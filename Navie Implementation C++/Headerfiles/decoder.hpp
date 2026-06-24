#ifndef DECODER_HPP
#define DECODER_HPP

#include <vector>
#include <map>
#include "weight_loader.hpp"

/**
 * SegFormer B0 Decoder 
 * 
 * Architecture:
 * 1. Linear projections: enc[0:4] with channels [32,64,160,256] -> all to 256
 * 2. Bilinear upsample all to 128×128 spatial resolution
 * 3. Concatenate: 128×128×(256+256+256+256) = 128×128×1024
 * 4. Fuse Conv 1×1: 128×128×1024 -> 128×128×256 (NO bias)
 * 5. BatchNorm (inference mode with running stats): normalize with learnable gamma/beta
 * 6. ReLU activation
 * 7. Classifier Conv 1×1: 128×128×256 -> 128×128×19
 */

class Decoder {
public:
    Decoder();
    ~Decoder();

    //Weights 
    void loadWeights(WeightLoader& loader);

    //Forward pass
    std::vector<float> forward(const std::vector<std::vector<float>>& encoder_outputs);

     //Class distribution statistics 
    void printClassDistribution(const std::vector<std::vector<int>>& predictions) const;

private:
    // ============ Weights ============
    
    // Linear projections
    std::vector<float> proj_w_[4];  
    std::vector<float> proj_b_[4];  

    // Fuse convolution 
    std::vector<float> fuse_w_;     

    // BatchNorm 
    std::vector<float> bn_w_;           
    std::vector<float> bn_b_;            
    std::vector<float> bn_running_mean_; 
    std::vector<float> bn_running_var_;  

    // Classifier 
    std::vector<float> cls_w_;      
    std::vector<float> cls_b_;      

    // ============ Helper Functions ============

    //Project tokens from input channels to 256 channels
    std::vector<float> projectTokens(const std::vector<float>& tokens, int H, int W, int C_in, const std::vector<float>& weight, const std::vector<float>& bias); 

    //Upsample using bilinear interpolation 
    std::vector<float> upsampleBilinear(const std::vector<float>& tokens, int H_in, int W_in, int H_out, int W_out, int C);

    //1x1 Convolution 
    std::vector<float> conv1x1(const std::vector<float>& input, int H, int W, int inC, int outC, const std::vector<float>& weight, const std::vector<float>* bias);   

    std::vector<float> fuseConv1x1(const std::vector<float>& input, int H, int W, const std::vector<float>& weight);

    //BatchNorm 
    std::vector<float> batchNormInference(const std::vector<float>& input, int H, int W, int C, const std::vector<float>& gamma, const std::vector<float>& beta, const std::vector<float>& running_mean, const std::vector<float>& running_var);
};

#endif  // DECODER_HPP
