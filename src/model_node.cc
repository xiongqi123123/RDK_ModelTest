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

int main(int argc, char** argv) {
    // 检查命令行参数
    std::string config_path;
    if (argc < 2) {
        std::cerr << "错误: 未提供配置文件路径" << std::endl;
        printUsage(argv[0]);
        return 1;
    } else {
        config_path = argv[1];
    }
    
    // 加载配置文件
    std::cout << "正在加载配置文件: " << config_path << std::endl;
    ModelConfig config;
    if (!config.loadFromJson(config_path)) {
        std::cerr << "加载配置文件失败" << std::endl;
        return 1;
    }
    
    // 打印配置信息
    std::cout << "配置信息:" << std::endl;
    std::cout << "- 任务ID: " << config.getTaskId() << std::endl;
    std::cout << "- 任务类型: " << config.getTaskType() << std::endl;
    std::cout << "- 模型名称: " << config.getModelName() << std::endl;
    std::cout << "- 模型路径: " << config.getModelPath() << std::endl;
    std::cout << "- 类别数量: " << config.getClassesNum() << std::endl;
    std::cout << "- 类别标签: ";
    for (const auto& label : config.getClassesLabels()) {
        std::cout << label << " ";
    }
    std::cout << std::endl;
    std::cout << "- 图像路径: " << config.getImagePath() << std::endl;
    std::cout << "- 输出路径: " << config.getOutputPath() << std::endl;
    
    // 创建检测器实例
    BPU_Detect detector(
        config.getModelName(),
        config.getTaskType(),
        config.getModelPath(),
        config.getClassesNum()
    );
    
    // 设置类别名称
    detector.SetClassNames(config.getClassesLabels());
    
    // 初始化模型
    if (!detector.Model_Init()) {
        std::cerr << "模型初始化失败" << std::endl;
        return 1;
    }
    
    // 读取输入图像
    cv::Mat input_img = cv::imread(config.getImagePath());
    if (input_img.empty()) {
        std::cerr << "无法读取图像: " << config.getImagePath() << std::endl;
        return 1;
    }
    
    // 准备输出图像
    cv::Mat output_img;
    
    // 执行推理
    if (!detector.Model_Inference(input_img, output_img)) {
        std::cerr << "模型推理失败" << std::endl;
        return 1;
    }
    
    // 保存结果图像
    std::string output_path = config.getOutputPath() + "result_" + config.getTaskId() + ".jpg";
    cv::imwrite(output_path, output_img);
    std::cout << "结果图像已保存至: " << output_path << std::endl;
    
    // 释放模型资源
    detector.Model_Release();
    
    std::cout << "推理完成" << std::endl;
    return 0;
}
    