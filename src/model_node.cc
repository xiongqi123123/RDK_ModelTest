#include "model_node.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>

bool ModelConfig::loadFromJson(const std::string& json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        std::cerr << "无法打开配置文件: " << json_path << std::endl;
        return false;
    }
    
    try {
        nlohmann::json config;
        file >> config;
        
        task_id_ = config["task_id"];
        task_type_ = config["task_type"];
        model_name_ = config["model_name"];
        model_path_ = config["model_path"];
        classes_num_ = config["classes_num"];
        
        // 解析类别标签数组
        classes_labels_.clear();
        for (const auto& label : config["classes_labels"]) {
            classes_labels_.push_back(label);
        }
        
        image_path_ = config["image_path"];
        label_path_ = config["label_path"];
        
        // 设置输出路径，如果未指定则使用当前目录
        if (config.contains("output_path")) {
            output_path_ = config["output_path"];
            // 确保路径以"/"结尾
            if (!output_path_.empty() && output_path_.back() != '/') {
                output_path_ += '/';
            }
        } else {
            output_path_ = "./";
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "解析JSON配置文件失败: " << e.what() << std::endl;
        return false;
    }
}

void printUsage(const char* programName) {
    std::cout << "用法: " << programName << " [配置文件路径]" << std::endl;
    std::cout << "示例: " << programName << " dataset/input/input.json" << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc < 2){
        printf("invalid arguments!\n");
        printf("usage: ./model_node configFilePath\n");
        return -1;
    }

    std::string config_file = argv[1];
    ModelConfig config;
    if (!config.loadFromJson(config_file)){
        printf("load config from json failed!\n");
        return -1;
    }

    printf("Configuration:\n");
    printf("  task_id: %s\n", config.getTaskId().c_str());
    printf("  task_type: %s\n", config.getTaskType().c_str());
    printf("  model_name: %s\n", config.getModelName().c_str());
    printf("  model_path: %s\n", config.getModelPath().c_str());
    printf("  classes_num: %d\n", config.getClassesNum());
    printf("  classes_labels:");
    for (const auto& label : config.getClassesLabels()) {
        printf(" %s", label.c_str());
    }
    printf("\n");
    printf("  image_path: %s\n", config.getImagePath().c_str());
    printf("  label_path: %s\n", config.getLabelPath().c_str());
    printf("  output_path: %s\n", config.getOutputPath().c_str());

    // 初始化检测器
    BPU_Detect detector(config.getModelName(), config.getTaskType(), config.getModelPath(), config.getClassesNum());
    
    // 设置类别名称
    detector.SetClassNames(config.getClassesLabels());
    
    // 设置任务ID和输出路径
    detector.SetTaskId(config.getTaskId());
    detector.SetOutputPath(config.getOutputPath());
    
    // 设置标注文件路径
    detector.SetLabelPath(config.getLabelPath());

    // 读取输入图像
    cv::Mat input_image = cv::imread(config.getImagePath());
    if (input_image.empty()) {
        printf("Failed to read input image: %s\n", config.getImagePath().c_str());
        return -1;
    }

    // 执行推理
    InferenceResult result;
    cv::Mat output_image;
    if (!detector.Model_Inference(input_image, output_image, result)) {
        printf("Model inference failed\n");
        return -1;
    }

    // 输出性能指标
    printf("Performance metrics:\n");
    printf("  FPS: %.2f\n", result.fps);
    printf("  Preprocess time: %.2f ms\n", result.preprocess_time);
    printf("  Inference time: %.2f ms\n", result.inference_time);
    printf("  Postprocess time: %.2f ms\n", result.postprocess_time);
    printf("  Total time: %.2f ms\n", result.total_time);
    printf("  Precision: %.4f\n", result.precision);
    printf("  Recall: %.4f\n", result.recall);
    printf("  mAP@0.5: %.4f\n", result.mAP50);
    printf("  mAP@0.5:0.95: %.4f\n", result.mAP50_95);
    printf("  Result image saved to: %s\n", result.result_path.c_str());

    // 释放资源
    detector.Model_Release();

    return 0;
}
    