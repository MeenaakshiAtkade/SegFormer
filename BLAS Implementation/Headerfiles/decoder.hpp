#ifndef DECODER_HPP
#define DECODER_HPP

#include <vector>
#include <map>
#include "weight_loader.hpp"
using namespace std;

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

    //Load all decoder weights from the weight loader
    void loadWeights(WeightLoader& loader);

    //Forward pass through decoder
    std::vector<float> forward(const std::vector<std::vector<float>>& encoder_outputs);

    //Print class distribution 
    void printClassDistribution(const std::vector<std::vector<int>>& predictions) const;

private:
    // ============ Weights ============
    
    // Linear projections: weight[256, C_in]
    vector<float> proj_w_[4];  
    vector<float> proj_b_[4];  

    // Fuse convolution 1x1: weight[256, 1024, 1, 1] (NO bias)
    vector<float> fuse_w_;     

    // BatchNorm parameters 
    vector<float> bn_w_;            
    vector<float> bn_b_;            
    vector<float> bn_running_mean_; 
    vector<float> bn_running_var_;  

    // Classifier convolution 1x1: weight[19, 256, 1, 1]
    vector<float> cls_w_;      
    vector<float> cls_b_;      

    // ============ Helper Functions ============
     
    //Project tokens from input channels to 256 channels
    vector<float> projectTokens(
        const vector<float>& tokens,
        int H, int W, int C_in,
        const vector<float>& weight,    
        const vector<float>& bias);   

    
    //Upsample using bilinear interpolation 
    vector<float> upsampleBilinear(const vector<float>& tokens, int H_in, int W_in, int H_out, int W_out, int C);

    //1x1 Convolution 
    vector<float> conv1x1(const vector<float>& input, int H, int W, int inC, int outC, const vector<float>& weight, const vector<float>* bias);     

    //1x1 Fuse Convolution 
    vector<float> fuseConv1x1(const vector<float>& input, int H, int W, const vector<float>& weight);

    //BatchNorm in inference mode 
    vector<float> batchNormInference( const vector<float>& input, int H, int W, int C, const vector<float>& gamma, const vector<float>& beta, const vector<float>& running_mean, const vector<float>& running_var);  
};

#endif  // DECODER_HPP
