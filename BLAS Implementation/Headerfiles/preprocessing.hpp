#ifndef PREPROCESSING_HPP
#define PREPROCESSING_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class Preprocessor {
public:
    Preprocessor(int target_height, int target_width);
    
    cv::Mat preprocess(const cv::Mat& input_image);
    
private:
    int target_height_;
    int target_width_;
    
    // ImageNet normalization parameters (RGB order)
    const std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
    const std::vector<float> std_ = {0.229f, 0.224f, 0.225f};
    
    cv::Mat resizeImage(const cv::Mat& image) const;
    cv::Mat normalize(const cv::Mat& image) const;
    std::vector<float> convertToChannelFirst(const cv::Mat& image) const;
};

#endif // PREPROCESSING_HPP
