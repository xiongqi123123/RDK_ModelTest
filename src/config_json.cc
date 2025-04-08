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
            result_json["processed_images"] = 1;
        } else if (task_type == "classification") {
            result_json["metrics"] = {
                {"top1_accuracy", result.acc1},
                {"top5_accuracy", result.acc5}
            };
            result_json["processed_images"] = 1;
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
        
        // 更新metrics部分
        if (task_type == "detection") {
            // // 更新检测相关的metrics（如果result中有值且大于0，则使用result的值）
            // auto& metrics = result_json["metrics"];
            // if (result.mAP50 > 0) metrics["mAP50"] = result.mAP50;
            // if (result.mAP50_95 > 0) metrics["mAP50-95"] = result.mAP50_95;
            // if (result.precision > 0) metrics["precision"] = result.precision;
            // if (result.recall > 0) metrics["recall"] = result.recall;
            // 更新累积指标
            int total_images = result_json.contains("processed_images") ? 
                             result_json["processed_images"].get<int>() : 0;
            total_images++;
            
            auto& accum = result_json["accumulated_metrics"];
            float sum_mAP50 = accum["sum_mAP50"].get<float>() + result.mAP50;
            float sum_mAP50_95 = accum["sum_mAP50_95"].get<float>() + result.mAP50_95;
            float sum_precision = accum["sum_precision"].get<float>() + result.precision;
            float sum_recall = accum["sum_recall"].get<float>() + result.recall;
            
            // 更新累积和
            accum["sum_mAP50"] = sum_mAP50;
            accum["sum_mAP50_95"] = sum_mAP50_95;
            accum["sum_precision"] = sum_precision;
            accum["sum_recall"] = sum_recall;
            
            // 对于检测任务，使用processed_images作为唯一的计数器
            int processed_count = result_json.contains("processed_images") ? 
                                 result_json["processed_images"].get<int>() : 0;
            processed_count++;
            
            // 计算新的平均指标
            result_json["metrics"]["mAP50"] = sum_mAP50 / processed_count;
            result_json["metrics"]["mAP50-95"] = sum_mAP50_95 / processed_count;
            result_json["metrics"]["precision"] = sum_precision / processed_count;
            result_json["metrics"]["recall"] = sum_recall / processed_count;
            
            // 更新计数
            result_json["processed_images"] = processed_count;
        } else if (task_type == "classification") {
            int img_count = result_json.contains("processed_images") ? result_json["processed_images"].get<int>() : 0;
            auto& metrics = result_json["metrics"];
            float current_acc = result.acc1 > 0.5f ? 1.0f : 0.0f;
            float prev_acc = metrics["top1_accuracy"].get<float>();
            // (旧平均值 * 旧样本数 + 新值) / 新样本数
            float new_acc = (prev_acc * img_count + current_acc) / (img_count + 1);
            metrics["top1_accuracy"] = new_acc;
            float current_acc5 = result.acc5 > 0.5f ? 1.0f : 0.0f;
            float prev_acc5 = metrics["top5_accuracy"].get<float>();
            float new_acc5 = (prev_acc5 * img_count + current_acc5) / (img_count + 1);
            metrics["top5_accuracy"] = new_acc5;
        }
        
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
