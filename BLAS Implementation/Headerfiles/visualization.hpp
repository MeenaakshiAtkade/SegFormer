#ifndef VISUALIZATION_HPP
#define VISUALIZATION_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include <iomanip>

class Visualizer {
public:

    static cv::Mat tensorToGrayImage(const std::vector<std::vector<std::vector<float>>>& hwc, int channel = 0) {
        int H = hwc.size();
        int W = hwc[0].size();
        cv::Mat img(H, W, CV_8UC1);

        float minVal = FLT_MAX, maxVal = -FLT_MAX;

        // Find min/max of selected channel
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                if ((int)hwc[h][w].size() > channel) {
                    float val = hwc[h][w][channel];
                    minVal = std::min(minVal, val);
                    maxVal = std::max(maxVal, val);
                }
            }
        }

        // Normalize to [0, 255]
        float range = maxVal - minVal;
        if (range < 1e-6) range = 1.0f;

        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                if ((int)hwc[h][w].size() > channel) {
                    float val = hwc[h][w][channel];
                    float normalized = (val - minVal) / range;
                    img.at<uchar>(h, w) = (uint8_t)(normalized * 255.0f);
                }
            }
        }

        return img;
    }

    //Convert flattened HWC tensor to 3D vector for visualization
    static std::vector<std::vector<std::vector<float>>> flattenedToHWC(
        const std::vector<float>& flat, int H, int W, int C) {
        
        std::vector<std::vector<std::vector<float>>> hwc(
            H, std::vector<std::vector<float>>(W, std::vector<float>(C)));
        
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int base = (h * W + w) * C;
                for (int c = 0; c < C; c++) {
                    hwc[h][w][c] = flat[base + c];
                }
            }
        }
        
        return hwc;
    }

    static cv::Mat applyColorMap(const cv::Mat& gray) {
        cv::Mat colored;
        cv::applyColorMap(gray, colored, cv::COLORMAP_VIRIDIS);
        return colored;
    }

    static void saveFeatureMap(const std::vector<std::vector<std::vector<float>>>& hwc,
                              const std::string& filename, int channel = 0) {
        // Convert to grayscale
        cv::Mat gray = tensorToGrayImage(hwc, channel);
        cv::Mat colored;
        cv::applyColorMap(gray, colored, cv::COLORMAP_VIRIDIS);

        // Save colored version
        cv::imwrite(filename, colored);
        std::cout << "[VIZ] Saved: " << filename << " (" << colored.rows << "x" << colored.cols 
                  << ") [COLORIZED with Viridis]" << std::endl;
    }

    //Save feature map from flattened tensor
    static void saveFeatureMapFlat(const std::vector<float>& flat, int H, int W, int C, const std::string& filename, int channel = 0) {
        auto hwc = flattenedToHWC(flat, H, W, C);
        saveFeatureMap(hwc, filename, channel);
    }

    static void saveEncoderStage(const std::vector<std::vector<std::vector<float>>>& enc_hwc, const std::string& stage_name) {
        int H = enc_hwc.size();
        int W = enc_hwc[0].size();
        int C = enc_hwc[0][0].size();

        std::cout << "[VIZ] Saving " << stage_name << " (" << H << "x" << W << "x" << C << ")" << std::endl;

        // Save first 4 channels as separate images
        for (int ch = 0; ch < std::min(4, C); ch++) {
            std::string filename = "cpp_" + stage_name + "_ch" + std::to_string(ch) + ".jpg";
            saveFeatureMap(enc_hwc, filename, ch);
        }
    }

    //Save decoder stage features (from flattened tensor)
    static void saveDecoderStage(const std::vector<float>& flat, int H, int W, int C, const std::string& stage_name) {
        std::cout << "[VIZ] Saving decoder " << stage_name << " (" << H << "x" << W << "x" << C << ")" << std::endl;

        auto hwc = flattenedToHWC(flat, H, W, C);

        // Save first 2 channels (or 4 for larger channel counts)
        int num_channels = (C > 100) ? 2 : std::min(4, C);
        for (int ch = 0; ch < num_channels; ch++) {
            std::string filename = "cpp_decoder_" + stage_name + "_ch" + std::to_string(ch) + ".jpg";
            saveFeatureMap(hwc, filename, ch);
        }
    }

    static cv::Mat predictionsToColored(const std::vector<std::vector<int>>& predictions) {
        int H = predictions.size();
        int W = predictions[0].size();

        std::vector<cv::Vec3b> palette = {
            cv::Vec3b(128, 64, 128),    // 0: road
            cv::Vec3b(244, 35, 232),    // 1: sidewalk
            cv::Vec3b(70, 70, 70),      // 2: building
            cv::Vec3b(102, 102, 156),   // 3: wall
            cv::Vec3b(190, 153, 153),   // 4: fence
            cv::Vec3b(153, 153, 153),   // 5: pole
            cv::Vec3b(250, 170, 30),    // 6: traffic light
            cv::Vec3b(220, 220, 0),     // 7: traffic sign
            cv::Vec3b(107, 142, 35),    // 8: vegetation
            cv::Vec3b(152, 251, 152),   // 9: terrain
            cv::Vec3b(70, 130, 180),    // 10: sky
            cv::Vec3b(220, 20, 60),     // 11: person
            cv::Vec3b(255, 0, 0),       // 12: rider
            cv::Vec3b(0, 0, 142),       // 13: car
            cv::Vec3b(0, 0, 70),        // 14: truck
            cv::Vec3b(0, 60, 100),      // 15: bus
            cv::Vec3b(0, 80, 100),      // 16: train
            cv::Vec3b(0, 0, 230),       // 17: motorcycle
            cv::Vec3b(119, 11, 32)      // 18: bicycle
        };

        cv::Mat colored(H, W, CV_8UC3);

        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int cls = predictions[h][w];
                if (cls >= 0 && cls < (int)palette.size()) {
                    colored.at<cv::Vec3b>(h, w) = palette[cls];
                } else {
                    colored.at<cv::Vec3b>(h, w) = cv::Vec3b(0, 0, 0); 
                }
            }
        }

        return colored;
    }

    // Compute class distribution (counts) from prediction map
    static std::vector<int> computeClassDistribution(const std::vector<std::vector<int>>& predictions, int num_classes = 19) {
        std::vector<int> counts(num_classes, 0);
        for (const auto& row : predictions) {
            for (int v : row) {
                if (v >= 0 && v < num_classes) {
                    counts[v]++;
                }
            }
        }
        return counts;
    }
    /*
    static void saveClassDistributionCSV(const std::vector<int>& counts, const std::string& filename) {
        int total = 0;
        for (int c : counts) total += c;

        std::ofstream ofs(filename);
        if (!ofs.is_open()) {
            std::cerr << " Failed to open class distribution file: " << filename << std::endl;
            return;
        }

        ofs << "class,count,percentage\n";
        for (size_t i = 0; i < counts.size(); ++i) {
            double pct = (total > 0) ? (100.0 * counts[i] / total) : 0.0;
            ofs << i << "," << counts[i] << "," << std::fixed << std::setprecision(2) << pct << "\n";
        }

        ofs.close();
        std::cout << " Saved class distribution CSV: " << filename << std::endl;
    }*/

    //Save final outputs:
    static void saveFinalOutputs(const cv::Mat& originalBGR, const std::vector<std::vector<int>>& predictions, const std::string& out_prefix) {
        
        // Colored segmentation
        cv::Mat colored = predictionsToColored(predictions);

        // Resize colored to match original if needed
        if (colored.rows != originalBGR.rows || colored.cols != originalBGR.cols) {
            cv::Mat colored_resized;
            cv::resize(colored, colored_resized, originalBGR.size(), 0, 0, cv::INTER_NEAREST);
            colored = colored_resized;
        }

        // Save segmentation
        std::string seg_file = out_prefix + "_segmentation.png";
        cv::imwrite(seg_file, colored);
        std::cout << " Saved segmentation: " << seg_file << std::endl;

        // Save class distribution
        //int palette_size = 19;
        //std::vector<int> counts = computeClassDistribution(predictions, palette_size);
        //std::string csv_file = out_prefix + "_class_distribution.csv";
        //saveClassDistributionCSV(counts, csv_file);
    }

};

#endif // VISUALIZATION_HPP
