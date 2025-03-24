#include "config_json.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

// 从YOLO格式文件加载标注数据
bool GroundTruthLoader::LoadFromJson(const std::string& label_path,
                                   std::vector<BBoxInfo>& gt_boxes,
                                   const std::vector<std::string>& class_names) {
    // 检查标注文件路径是否为空
    if (label_path.empty()) {
        printf("Label path is empty, skipping ground truth data loading\n");
        return false;
    }
    
    // 尝试打开标注文件
    std::ifstream file(label_path);
    if (!file.is_open()) {
        printf("Failed to open label file: %s\n", label_path.c_str());
        return false;
    }
    
    // 清空之前的标注数据
    gt_boxes.clear();
    
    // 读取YOLO格式文件
    std::string line;
    int line_count = 0;
    
    while (std::getline(file, line)) {
        line_count++;
        std::istringstream iss(line);
        int class_id;
        float x_center, y_center, width, height;
        
        if (!(iss >> class_id >> x_center >> y_center >> width >> height)) {
            printf("Warning: Invalid format at line %d: %s\n", line_count, line.c_str());
            continue;
        }
        
        // YOLO格式的坐标是归一化的（0-1范围），需要记录原始值用于后续计算
        BBoxInfo box;
        box.class_id = class_id;
        box.x = x_center;
        box.y = y_center;
        box.width = width;
        box.height = height;
        box.confidence = 1.0f; // 标注数据的置信度设为1
        
        // 设置类别名称
        if (class_id < static_cast<int>(class_names.size())) {
            box.class_name = class_names[class_id];
        } else {
            box.class_name = "class" + std::to_string(class_id);
        }
        
        gt_boxes.push_back(box);
    }
    
    file.close();
    printf("Loaded %zu ground truth boxes from YOLO format file: %s\n", gt_boxes.size(), label_path.c_str());
    return !gt_boxes.empty();
}

// 保存推理结果到JSON文件和图像
bool ResultSaver::SaveResults(const std::string& output_path,
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
                            InferenceResult& result,
                            const std::string& image_name) {
    std::string image_key = image_name;
    size_t last_slash = image_key.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        image_key = image_key.substr(last_slash + 1);
    }
    size_t last_dot = image_key.find_last_of(".");
    if (last_dot != std::string::npos) {
        image_key = image_key.substr(0, last_dot);
    }
    
    std::string image_path = output_path + "result_" + image_key + ".jpg";
    if (!cv::imwrite(image_path, output_img)) {
        std::cerr << "Failed to save result image: " << image_path << std::endl;
        return false;
    }
    
    result.result_path = image_path;
    std::cout << "Result image saved to: " << image_path << std::endl;
    std::string json_path = output_path + "result_" + task_id + ".json";
    nlohmann::json result_json;
    bool json_exists = false;
    std::ifstream json_file_in(json_path);
    if (json_file_in.is_open()) {
        try {
            json_file_in >> result_json;
            json_exists = true;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to parse existing JSON file: " << e.what() << std::endl;
            result_json = nlohmann::json();
        }
        json_file_in.close();
    }
    
    if (!json_exists) {
        result_json["task_id"] = task_id;
        result_json["task_type"] = task_type;
        result_json["model_name"] = model_name;
        result_json["performance"] = {
            {"fps", result.fps},
            {"preprocess_time", result.preprocess_time},
            {"inference_time", result.inference_time},
            {"postprocess_time", result.postprocess_time},
            {"total_time", result.total_time}
        };
        if (task_type == "detection") {
            result_json["metrics"] = {
                {"mAP50", result.mAP50},
                {"mAP50-95", result.mAP50_95},
                {"precision", result.precision},
                {"recall", result.recall}
            };
        } else if (task_type == "classification") {
            result_json["metrics"] = {
                {"accuracy_top1", result.acc1},
                {"accuracy_top5", result.acc5}
            };
        }
    } else {
        // 更新性能指标（使用平均值）
        auto& perf = result_json["performance"];
        int img_count = result_json.contains("processed_images") ? result_json["processed_images"].get<int>() : 0;
        float weight = 1.0f / (img_count + 1);
        
        perf["fps"] = perf["fps"].get<float>() * (1.0f - weight) + result.fps * weight;
        perf["preprocess_time"] = perf["preprocess_time"].get<float>() * (1.0f - weight) + result.preprocess_time * weight;
        perf["inference_time"] = perf["inference_time"].get<float>() * (1.0f - weight) + result.inference_time * weight;
        perf["postprocess_time"] = perf["postprocess_time"].get<float>() * (1.0f - weight) + result.postprocess_time * weight;
        perf["total_time"] = perf["total_time"].get<float>() * (1.0f - weight) + result.total_time * weight;
        result_json["processed_images"] = img_count + 1;
    }
    
    if (task_type == "detection") {
        nlohmann::json detections = nlohmann::json::array();
        for (size_t cls_id = 0; cls_id < bboxes.size(); cls_id++) {
            for (size_t i = 0; i < indices[cls_id].size(); i++) {
                int idx = indices[cls_id][i];
                
                // 获取原始图像中的坐标
                float x1 = (bboxes[cls_id][idx].x - x_shift) / x_scale;
                float y1 = (bboxes[cls_id][idx].y - y_shift) / y_scale;
                float width = bboxes[cls_id][idx].width / x_scale;
                float height = bboxes[cls_id][idx].height / y_scale;
                float confidence = scores[cls_id][idx];
                
                nlohmann::json detection;
                detection["bbox"] = {
                    {"x", x1},
                    {"y", y1},
                    {"width", width},
                    {"height", height}
                };
                detection["class_id"] = cls_id;
                detection["class_name"] = (cls_id < class_names.size()) ? 
                                        class_names[cls_id] : "class" + std::to_string(cls_id);
                detection["confidence"] = confidence;
                
                detections.push_back(detection);
            }
        }
        result_json["detections"][image_key] = detections;
    } else if (task_type == "classification") {
        nlohmann::json classifications = nlohmann::json::array();
        int top_count = std::min(5, static_cast<int>(indices[0].size()));
        
        for (int i = 0; i < top_count; i++) {
            int idx = indices[0][i];
            float confidence = scores[0][i];
            
            nlohmann::json classification;
            classification["rank"] = i + 1;
            classification["class_id"] = idx;
            classification["class_name"] = (idx < static_cast<int>(class_names.size())) ? 
                                       class_names[idx] : "class" + std::to_string(idx);
            classification["confidence"] = confidence;
            
            classifications.push_back(classification);
        }
        
        result_json["classifications"][image_key] = classifications;
    }
    
    std::ofstream json_file_out(json_path);
    if (json_file_out.is_open()) {
        json_file_out << std::setw(4) << result_json << std::endl;
        json_file_out.close();
        std::cout << "Result JSON saved to: " << json_path << std::endl;
    } else {
        std::cerr << "Failed to save JSON results: " << json_path << std::endl;
        return false;
    }
    
    return true;
}
