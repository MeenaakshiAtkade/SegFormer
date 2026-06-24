#include "../Headerfiles/preprocessing.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iomanip>
#include <cmath>
using namespace std;

Preprocessor::Preprocessor(int target_height, int target_width)
    : target_height_(target_height), target_width_(target_width) {
}

cv::Mat Preprocessor::resizeImage(const cv::Mat& image) const {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(target_width_, target_height_), 0, 0, cv::INTER_LINEAR);
    return resized;
}

cv::Mat Preprocessor::normalize(const cv::Mat& image) const {
    cv::Mat normalized;
    image.convertTo(normalized, CV_32FC3, 1.0 / 255.0);
    
    // Convert BGR to RGB (OpenCV loads images as BGR)
    cv::cvtColor(normalized, normalized, cv::COLOR_BGR2RGB);
    
    // Apply ImageNet normalization: (pixel/255 - mean) / std
    vector<cv::Mat> channels(3);
    cv::split(normalized, channels);
    
    for (int c = 0; c < 3; ++c) {
        channels[c] = (channels[c] - mean_[c]) / std_[c];
    }
    
    cv::merge(channels, normalized);
    return normalized;
}

cv::Mat Preprocessor::preprocess(const cv::Mat& input_image) {
     
    // Resize
    cv::Mat resized = resizeImage(input_image);
    
    // Normalize
    cv::Mat normalized = normalize(resized);
     
    return normalized;
}
