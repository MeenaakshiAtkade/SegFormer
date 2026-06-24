#include "Headerfiles/preprocessing.hpp"
#include "Headerfiles/encoder.hpp"
#include "Headerfiles/decoder.hpp"
#include "Headerfiles/weight_loader.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <iomanip>
using namespace std;

// Color map for 19 Cityscapes classes (BGR format for OpenCV)
const cv::Vec3b CLASS_COLORS[19] = {
    {128, 64, 128},   // 0: road
    {232, 35, 244},   // 1: sidewalk
    {70, 70, 70},     // 2: building
    {156, 102, 102},  // 3: wall
    {153, 153, 190},  // 4: fence
    {153, 153, 153},  // 5: pole
    {30, 170, 250},   // 6: traffic light
    {0, 220, 220},    // 7: traffic sign
    {35, 142, 107},   // 8: vegetation
    {152, 251, 152},  // 9: terrain
    {180, 130, 70},   // 10: sky
    {60, 20, 220},    // 11: person
    {0, 0, 255},      // 12: rider
    {142, 0, 0},      // 13: car
    {70, 0, 0},       // 14: truck
    {100, 60, 0},     // 15: bus
    {100, 80, 0},     // 16: train
    {230, 0, 0},      // 17: motorcycle
    {32, 11, 119}     // 18: bicycle
};

const char* CLASS_NAMES[19] = {
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle"
};

vector<vector<int>> argmaxLogits(const vector<float>& logits, int H, int W, int num_classes) {
    vector<vector<int>> predictions(H, vector<int>(W));
    
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int max_class = 0;
            float max_val = logits[h * W * num_classes + w * num_classes + 0];
            
            for (int c = 1; c < num_classes; ++c) {
                float val = logits[h * W * num_classes + w * num_classes + c];
                if (val > max_val) {
                    max_val = val;
                    max_class = c;
                }
            }
            
            predictions[h][w] = max_class;
        }
    }
    
    return predictions;
}

cv::Mat createSegmentationImage(const vector<vector<int>>& predictions) {
    int H = predictions.size();
    int W = predictions[0].size();
    
    cv::Mat seg_img(H, W, CV_8UC3);
    
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int class_id = predictions[h][w];
            seg_img.at<cv::Vec3b>(h, w) = CLASS_COLORS[class_id];
        }
    }
    
    return seg_img;
}

int main(int argc, char** argv)
{
    string weights_dir = "Add Path for Pre Trained Weights Folder";

    WeightLoader loader(weights_dir);

    Encoder encoder;
    encoder.loadWeights(loader);

    Decoder decoder;
    decoder.loadWeights(loader);

    Preprocessor preprocessor(256, 256);

    cv::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        cerr << "Failed to open camera." << endl;
        return -1;
    }

    cout << "SEMANTIC SEGMENTATION INFERENCE" << endl;
        cv::Mat input_image;
        cap >> input_image;

        if (input_image.empty())
        {
            cerr << "Failed to capture frame." << endl;
        }

        cv::Mat preprocessed =
            preprocessor.preprocess(input_image);

        vector<float> input_data(256 * 256 * 3);

        for (int h = 0; h < 256; ++h)
        {
            for (int w = 0; w < 256; ++w)
            {
                cv::Vec3f pixel =
                    preprocessed.at<cv::Vec3f>(h, w);

                int idx = h * 256 * 3 + w * 3;

                input_data[idx + 0] = pixel[0];
                input_data[idx + 1] = pixel[1];
                input_data[idx + 2] = pixel[2];
            }
        }

        vector<vector<float>> encoder_outputs =
            encoder.forward(input_data);

        vector<float> logits =
            decoder.forward(encoder_outputs);
    
    int H_out = 64;
    int W_out = 64;
    int num_classes = 19;

// -------- Step 1: Convert logits vector → cv::Mat --------
    // Shape: [H, W, C]
    vector<cv::Mat> channels(num_classes);

    for (int c = 0; c < num_classes; ++c) {
        channels[c] = cv::Mat(H_out, W_out, CV_32F);

        for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {
                int idx = h * W_out * num_classes + w * num_classes + c;
                channels[c].at<float>(h, w) = logits[idx];
            }
        }
    }

// -------- Step 2: Resize EACH channel (bilinear) --------
    vector<cv::Mat> resized_channels(num_classes);

    for (int c = 0; c < num_classes; ++c) {
        cv::resize(channels[c], resized_channels[c], input_image.size(),0, 0, cv::INTER_LINEAR);  
}

// -------- Step 3: Argmax AFTER resizing --------
    int H_final = input_image.rows;
    int W_final = input_image.cols;

    vector<vector<int>> predictions(H_final, vector<int>(W_final));

    for (int h = 0; h < H_final; ++h) {
        for (int w = 0; w < W_final; ++w) {

            int best_class = 0;
            float max_val = resized_channels[0].at<float>(h, w);

            for (int c = 1; c < num_classes; ++c) {
                float val = resized_channels[c].at<float>(h, w);
                if (val > max_val) {
                    max_val = val;
                    best_class = c;
                }
            }

            predictions[h][w] = best_class;
        }
    }

// -------- Step 4: Create segmentation image --------
    cv::Mat seg_image = createSegmentationImage(predictions);
    string seg_filename = "/home/ankit/Pictures/SegformerImageInference/build/cpp_final_segmentation.png";
    cv::imwrite(seg_filename, seg_image);
    std::this_thread::sleep_for(std::chrono::seconds(2));

    return 0;
}
