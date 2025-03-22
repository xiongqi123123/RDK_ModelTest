#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>
#include "detector.hpp"

// 标注数据处理类
class GroundTruthLoader {
public:
    // 从JSON文件加载标注数据
    static bool LoadFromJson(const std::string& label_path, 
                            std::vector<BBoxInfo>& gt_boxes,
                            const std::vector<std::string>& class_names);
};

// 结果保存类
class ResultSaver {
public:
    // 保存推理结果到JSON文件和图像
    static bool SaveResults(const std::string& output_path,
                          const std::string& task_id,
                          const std::string& model_name,
                          const std::string& task_type,
                          const cv::Mat& output_img,
                          const std::vector<std::vector<cv::Rect2d>>& bboxes,
                          const std::vector<std::vector<float>>& scores,
                          const std::vector<std::vector<int>>& indices,
                          const std::vector<std::string>& class_names,
                          const float x_scale,
                          const float y_scale,
                          const int x_shift,
                          const int y_shift,
                          InferenceResult& result);
};
