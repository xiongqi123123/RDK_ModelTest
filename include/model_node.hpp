#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>
#include "detector.hpp"

class ModelConfig {
public:
    ModelConfig() = default;
    ~ModelConfig() = default;

    bool loadFromJson(const std::string& json_path);
    
    // 获取配置参数
    std::string getTaskId() const { return task_id_; }
    std::string getTaskType() const { return task_type_; }
    std::string getModelName() const { return model_name_; }
    std::string getModelPath() const { return model_path_; }
    int getClassesNum() const { return classes_num_; }
    std::vector<std::string> getClassesLabels() const { return classes_labels_; }
    std::string getImagePath() const { return image_path_; }
    std::string getOutputPath() const { return output_path_; }

private:
    std::string task_id_;
    std::string task_type_;
    std::string model_name_;
    std::string model_path_;
    int classes_num_;
    std::vector<std::string> classes_labels_;
    std::string image_path_;
    std::string output_path_;
};
