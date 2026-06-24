#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include "preprocessing.hpp"
#include "patch_embedding.hpp"
#include "transformer_block.hpp"
#include "encoder.hpp"
#include "decoder.hpp"
#include "weight_loader.hpp"
#include "visualization.hpp"

int main(int argc, char* argv[]) {

    const std::string weights_dir = "Add Path for Pre Trained Weights Folder";
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    std::string image_path = argv[1];

    cv::Mat img_bgr = cv::imread(image_path);

    if (img_bgr.empty()) {
        std::cerr << "[ERROR] Failed to load image: " << image_path << std::endl;
        return -1;
    }

    int H_orig = img_bgr.rows;
    int W_orig = img_bgr.cols;
    /*
    // ================= CAMERA CAPTURE =================
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open camera." << std::endl;
        return -1;
    }

    cv::Mat img_bgr;
    cap >> img_bgr;

    if (img_bgr.empty()) {
        std::cerr << "[ERROR] Failed to capture image." << std::endl;
        return -1;
    }

    int H_orig = img_bgr.rows;
    int W_orig = img_bgr.cols;
    */

    // ================= PREPROCESSING =================
    Preprocessor preprocessor(512, 512);
    cv::Mat img_normalized_mat = preprocessor.preprocess(img_bgr);

    int H = img_normalized_mat.rows;
    int W = img_normalized_mat.cols;
    int C = img_normalized_mat.channels();

    std::vector<float> img_hwc(H * W * C);

    for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w) {
            cv::Vec3f v = img_normalized_mat.at<cv::Vec3f>(h, w);
            for (int c = 0; c < C; ++c)
                img_hwc[h * W * C + w * C + c] = v[c];
        }

    // ================= LOAD WEIGHTS =================
    WeightLoader loader(weights_dir);

    // ================= ENCODER =================
    Encoder encoder;
    encoder.loadWeights(loader);
    auto encoder_outputs = encoder.forward(img_hwc);

    // ================= DECODER =================
    Decoder decoder;
    decoder.loadWeights(loader);
    std::vector<float> logits128_flat = decoder.forward(encoder_outputs);

    const int outH = 128, outW = 128, outC = 19;

    std::vector<std::vector<std::vector<float>>> logits_128_hwc(outH, std::vector<std::vector<float>>(outW, std::vector<float>(outC)));

    for (int h = 0; h < outH; ++h)
        for (int w = 0; w < outW; ++w) {
            int base = (h * outW + w) * outC;
            for (int c = 0; c < outC; ++c)
                logits_128_hwc[h][w][c] = logits128_flat[base + c];
        }

    // ================= BILINEAR UPSAMPLE =================
    std::vector<std::vector<std::vector<float>>> logits_full(H_orig, std::vector<std::vector<float>>(W_orig, std::vector<float>(outC)));

    for (int c = 0; c < outC; ++c) {

        cv::Mat channel_128(outH, outW, CV_32F);
        for (int h = 0; h < outH; ++h)
            for (int w = 0; w < outW; ++w)
                channel_128.at<float>(h, w) = logits_128_hwc[h][w][c];

        cv::Mat channel_full;
        cv::resize(channel_128, channel_full, cv::Size(W_orig, H_orig), 0, 0, cv::INTER_LINEAR);

        for (int h = 0; h < H_orig; ++h)
            for (int w = 0; w < W_orig; ++w)
                logits_full[h][w][c] = channel_full.at<float>(h, w);
    }

    // ================= ARGMAX =================
    std::vector<std::vector<int>> predictions(
        H_orig, std::vector<int>(W_orig, 0));

    for (int h = 0; h < H_orig; ++h)
        for (int w = 0; w < W_orig; ++w) {
            float maxVal = logits_full[h][w][0];
            int maxClass = 0;

            for (int c = 1; c < outC; ++c)
                if (logits_full[h][w][c] > maxVal) {
                    maxVal = logits_full[h][w][c];
                    maxClass = c;
                }

            predictions[h][w] = maxClass;
        }

    // ================= SAVE OUTPUT =================
    Visualizer::saveFinalOutputs(img_bgr, predictions, "cpp_final");

    std::cout << "Segmentation complete.\n";
    return 0;
}
