#include "model_node.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <vector>
#include <dirent.h>  // POSIX目录访问
#include <sys/stat.h> // 文件状态
#include <algorithm>  // transform
#include <unistd.h>   // access
#include <string.h>   // strrchr
#include <signal.h>   // 信号处理
#include <cmath>      // isnan, isinf

// 全局变量，用于信号处理
volatile sig_atomic_t stop_processing = 0;

// 信号处理函数
void signal_handler(int sig) {
    printf("\nReceived signal %d, cleaning up and exiting...\n", sig);
    stop_processing = 1;
}

// 判断路径是否为目录
bool isDirectory(const std::string& path) {
    struct stat s;
    if (stat(path.c_str(), &s) == 0) {
        return S_ISDIR(s.st_mode);
    }
    return false;
}

// 判断文件是否存在
bool fileExists(const std::string& path) {
    return access(path.c_str(), F_OK) != -1;
}

// 获取文件扩展名
std::string getFileExtension(const std::string& path) {
    const char* ext = strrchr(path.c_str(), '.');
    if (ext == nullptr) {
        return "";
    }
    return std::string(ext);
}

// 获取文件名(不含扩展名)
std::string getFileBaseName(const std::string& path) {
    // 找到最后一个路径分隔符
    size_t last_slash = path.find_last_of("/\\");
    std::string filename = (last_slash == std::string::npos) ? path : path.substr(last_slash + 1);
    
    // 找到最后一个点(扩展名开始处)
    size_t last_dot = filename.find_last_of(".");
    if (last_dot == std::string::npos) {
        return filename;
    }
    return filename.substr(0, last_dot);
}

// 获取文件夹中的所有图片文件
std::vector<std::string> getImageFiles(const std::string& folder_path) {
    std::vector<std::string> image_files;
    DIR* dir = opendir(folder_path.c_str());
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string name = entry->d_name;
            // 跳过 . 和 ..
            if (name == "." || name == "..") {
                continue;
            }
            
            std::string full_path = folder_path;
            if (full_path.back() != '/') {
                full_path += '/';
            }
            full_path += name;
            
            // 检查是否为常规文件
            struct stat s;
            if (stat(full_path.c_str(), &s) == 0 && S_ISREG(s.st_mode)) {
                std::string ext = getFileExtension(name);
                // 转换为小写
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                    image_files.push_back(full_path);
                }
            }
        }
        closedir(dir);
    }
    return image_files;
}

// 根据图片路径获取对应的标签路径
std::string getLabelPath(const std::string& image_path, const std::string& label_folder) {
    // 获取图片文件名(不含扩展名)
    std::string image_name = getFileBaseName(image_path);
    // 尝试不同的标签文件扩展名
    std::vector<std::string> label_extensions = {".txt", ".json", ".xml"};
    
    std::string base_path = label_folder;
    if (base_path.back() != '/') {
        base_path += '/';
    }
    
    for (const auto& ext : label_extensions) {
        std::string label_path = base_path + image_name + ext;
        if (fileExists(label_path)) {
            return label_path;
        }
    }
    return ""; // 如果没找到对应的标签文件，返回空字符串
}

bool ModelConfig::loadFromJson(const std::string& json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open configuration file: " << json_path << std::endl;
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
        std::cerr << "Failed to parse JSON configuration file: " << e.what() << std::endl;
        return false;
    }
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [configuration file path]" << std::endl;
    std::cout << "Example: " << programName << " dataset/input/input.json" << std::endl;
}

// 安全打印浮点数
void safePrintFloat(const char* prefix, float value) {
    if (std::isnan(value)) {
        printf("%s NaN\n", prefix);
    } else if (std::isinf(value)) {
        printf("%s Infinity\n", prefix);
    } else if (value < -1e10 || value > 1e10) {
        printf("%s Invalid value\n", prefix);
    } else {
        printf("%s %.4f\n", prefix, value);
    }
}

int main(int argc, char *argv[])
{
    // 设置信号处理
    signal(SIGINT, signal_handler);  // Ctrl+C
    signal(SIGTERM, signal_handler); // 终止信号
    signal(SIGSEGV, signal_handler); // 段错误

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

    // 检查输入路径是文件还是文件夹
    std::vector<std::string> image_files;
    std::vector<std::string> label_files;
    
    if (isDirectory(config.getImagePath())) {
        // 如果是文件夹，获取所有图片文件
        image_files = getImageFiles(config.getImagePath());
        printf("Found %zu images in folder: %s\n", image_files.size(), config.getImagePath().c_str());
        
        // 如果标签路径也是文件夹，获取对应的标签文件
        if (isDirectory(config.getLabelPath())) {
            for (const auto& image_path : image_files) {
                std::string label_path = getLabelPath(image_path, config.getLabelPath());
                if (!label_path.empty()) {
                    label_files.push_back(label_path);
                } else {
                    printf("Warning: No label file found for image: %s\n", image_path.c_str());
                    label_files.push_back(""); // 添加空标签路径占位
                }
            }
        } else {
            // 如果标签路径是单个文件，所有图片共用这个标签文件
            label_files.resize(image_files.size(), config.getLabelPath());
        }
    } else {
        // 如果是单个文件
        image_files.push_back(config.getImagePath());
        label_files.push_back(config.getLabelPath());
    }

    // 为每张图片创建一个任务ID
    int success_count = 0;
    int fail_count = 0;

    // 处理每张图片
    for (size_t i = 0; i < image_files.size() && !stop_processing; i++) {
        // 每次创建新的检测器实例以避免内存累积
        BPU_Detect detector(config.getModelName(), config.getTaskType(), config.getModelPath(), config.getClassesNum());
        
        // 设置任务ID（使用原始任务ID加索引，确保唯一性）
        std::string current_task_id = config.getTaskId() + "_" + std::to_string(i+1);
        
        try {
            printf("\nProcessing image %zu/%zu: %s\n", i + 1, image_files.size(), image_files[i].c_str());
            
            // 设置类别名称
            detector.SetClassNames(config.getClassesLabels());
            
            // 设置任务ID和输出路径
            detector.SetTaskId(current_task_id);
            detector.SetOutputPath(config.getOutputPath());
            
            // 设置当前图片的标签路径
            if (!label_files[i].empty()) {
                detector.SetLabelPath(label_files[i]);
            }

            // 读取输入图像
            cv::Mat input_image = cv::imread(image_files[i]);
            if (input_image.empty()) {
                printf("Failed to read input image: %s\n", image_files[i].c_str());
                fail_count++;
                continue;
            }

            // 执行推理
            InferenceResult result;
            cv::Mat output_image;
            if (!detector.Model_Inference(input_image, output_image, result)) {
                printf("Model inference failed for image: %s\n", image_files[i].c_str());
                fail_count++;
                continue;
            }

            // 安全打印性能指标
            printf("Performance metrics for image %zu:\n", i + 1);
            printf("  FPS: %.2f\n", result.fps);
            printf("  Preprocess time: %.2f ms\n", result.preprocess_time);
            printf("  Inference time: %.2f ms\n", result.inference_time);
            printf("  Postprocess time: %.2f ms\n", result.postprocess_time);
            printf("  Total time: %.2f ms\n", result.total_time);
            
            // 使用安全打印函数打印可能存在问题的浮点数
            safePrintFloat("  Precision:", result.precision);
            safePrintFloat("  Recall:", result.recall);
            safePrintFloat("  mAP@0.5:", result.mAP50);
            safePrintFloat("  mAP@0.5:0.95:", result.mAP50_95);
            
            printf("  Result image saved to: %s\n", result.result_path.c_str());
            success_count++;
            
        } catch (const std::exception& e) {
            printf("Exception occurred while processing image %s: %s\n", 
                  image_files[i].c_str(), e.what());
            fail_count++;
        } catch (...) {
            printf("Unknown exception occurred while processing image %s\n", 
                  image_files[i].c_str());
            fail_count++;
        }
        
        // 释放资源
        detector.Model_Release();
        
        // 每处理10张图片输出一次进度
        if (i % 10 == 9 || i == image_files.size() - 1) {
            printf("\nProgress: %zu/%zu (%.1f%%), Success: %d, Failed: %d\n", 
                  i + 1, image_files.size(), 
                  (i + 1) * 100.0 / image_files.size(),
                  success_count, fail_count);
        }
        
        // 短暂等待，让系统有时间释放资源
        usleep(100000); // 100ms
    }

    printf("\nProcessing complete. Total: %zu, Success: %d, Failed: %d\n", 
          image_files.size(), success_count, fail_count);

    return 0;
}
    