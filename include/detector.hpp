#ifndef DETECTOR_H
#define DETECTOR_H

// C/C++ Standard Librarys
#include <iostream>     // 输入输出流
#include <vector>      // 向量容器
#include <algorithm>   // 算法库
#include <chrono>      // 时间相关功能
#include <iomanip>     // 输入输出格式控制
#include <omp.h>
// Thrid Party Librarys
#include <opencv2/opencv.hpp>      // OpenCV主要头文件
#include <opencv2/dnn/dnn.hpp>     // OpenCV深度学习模块

// RDK BPU libDNN API
#include "dnn/hb_dnn.h"           // BPU基础功能
#include "dnn/hb_dnn_ext.h"       // BPU扩展功能
#include "dnn/plugin/hb_dnn_layer.h"    // BPU层定义
#include "dnn/plugin/hb_dnn_plugin.h"   // BPU插件
#include "dnn/hb_sys.h"           // BPU系统功能

// JSON处理
#include <nlohmann/json.hpp>

// 错误检查宏
#define RDK_CHECK_SUCCESS(value, errmsg)                        \
    do                                                          \
    {                                                          \
        auto ret_code = value;                                  \
        if (ret_code != 0)                                      \
        {                                                       \
            std::cout << errmsg << ", error code:" << ret_code; \
            return ret_code;                                    \
        }                                                       \
    } while (0);

// 默认参数定义
#define DEFAULT_NMS_THRESHOLD 0.45f //NMS的阈值, 默认0.45
#define DEFAULT_SCORE_THRESHOLD 0.25f // 置信度阈值, 默认0.25
#define DEFAULT_NMS_TOP_K 300 // NMS选取的前K个框数, 默认300
#define DEFAULT_FONT_SIZE 1.0f // 绘制标签的字体尺寸, 默认1.0
#define DEFAULT_FONT_THICKNESS 1.0f // 绘制标签的字体粗细, 默认 1.0
#define DEFAULT_LINE_SIZE 2.0f // 绘制矩形框的线宽, 默认2.0

#define REG 16  // 离散化程度的超参数
// 模型类型枚举
enum ModelType {
    YOLOV5 = 0,   // YOLOv5系列模型
    YOLO11 = 1,   // YOLO11系列模型
    YOLOV8 = 2,   // YOLOv8系列模型
    YOLOV8_SEG = 3, // YOLOv8分割模型
    YOLO11_SEG = 4, // YOLO11分割模型
    FCN = 5,      // FCN语义分割模型
    UNKNOWN = -1  // 未知模型类型
};

// 结果结构体
struct InferenceResult {
    float fps;                   // 帧率 (frames per second)
    float preprocess_time;       // 前处理时间 (ms)
    float inference_time;        // 推理时间 (ms)
    float postprocess_time;      // 后处理时间 (ms)
    float total_time;            // 总时间 (ms)
    float precision;             // 精确率
    float recall;                // 召回率
    float mAP50;                 // mAP@0.5
    float mAP50_95;              // mAP@0.5:0.95
    float acc1;                  // Top-1准确率 (分类任务)
    float acc5;                  // Top-5准确率 (分类任务)
    std::string result_path;     // 结果图像路径
};

struct BBoxInfo {
    float x;
    float y;
    float width;
    float height;
    int class_id;
    std::string class_name;
    float confidence;
};

struct detection_result_t {
    cv::Rect bbox;
    float score;
    int class_id;
    cv::Mat mask;
};

class BPU_Detect{
    public:
        BPU_Detect(const std::string& model_name = "",
                    const std::string& task_type = "",
                    const std::string& model_path = "",
                    const int classes_num = 0,
                    const float nms_threshold = DEFAULT_NMS_THRESHOLD,
                    const float score_threshold = DEFAULT_SCORE_THRESHOLD,
                    const int nms_top_k = DEFAULT_NMS_TOP_K);
        ~BPU_Detect();

        bool Model_Init();
        bool Model_Inference(const cv::Mat& input_img, cv::Mat& output_img, InferenceResult& result, const std::string& image_name);
        bool Model_Release();
        void SetClassNames(const std::vector<std::string>& class_names) { class_names_ = class_names; }
        void SetOutputPath(const std::string& output_path) { output_path_ = output_path; }
        void SetTaskId(const std::string& task_id) { task_id_ = task_id; }
        void SetLabelPath(const std::string& label_path) { label_path_ = label_path; }
    private:
        bool Model_Load();
        bool Model_Anchor_Init();
        bool Model_Output_Order();
        bool Model_Info_check();
        bool Model_Preprocess(const cv::Mat& input_img);
        bool Model_Detector();
        bool Model_Detection_Postprocess();
        bool Model_Detection_Postprocess_YOLO11(); // YOLO11专用后处理
        bool Model_Detection_Postprocess_YOLOV8(); // YOLOv8专用后处理
        bool Model_Segmentation_Postprocess_YOLOV8(); // YOLOv8-Seg专用后处理
        bool Model_Segmentation_Postprocess_YOLO11(); // YOLO11-Seg专用后处理
        bool Model_Segmentation_Postprocess_FCN(); // FCN专用后处理
        void Model_Process_FeatureMap(hbDNNTensor& output_tensor, 
                                     int feature_h, 
                                     int feature_w, 
                                     const std::vector<std::vector<float>>& anchors, 
                                     float conf_thres);
        void Model_Process_FeatureMap_YOLO11(hbDNNTensor& cls_tensor, 
                                   hbDNNTensor& bbox_tensor,
                                   int feature_h, 
                                   int feature_w); // YOLO11专用特征图处理
        void Model_Process_FeatureMap_YOLOV8(hbDNNTensor& output_tensor,
                                   int feature_h,
                                   int feature_w,
                                   float stride); // YOLOv8专用特征图处理
        void Model_Process_FeatureMap_YOLOV8_SEG(
                                   int scale_idx,
                                   std::vector<cv::Rect2d>& decoded_bboxes_all,
                                   std::vector<float>& decoded_scores_all,
                                   std::vector<int>& decoded_classes_all,
                                   std::vector<std::vector<float>>& decoded_mces_all); // YOLOv8-Seg专用特征图处理
        void Model_Process_FeatureMap_YOLO11_SEG(
                                   int scale_idx,
                                   std::vector<cv::Rect2d>& decoded_bboxes_all,
                                   std::vector<float>& decoded_scores_all,
                                   std::vector<int>& decoded_classes_all,
                                   std::vector<std::vector<float>>& decoded_mces_all); // YOLO11-Seg专用特征图处理
        bool Model_Classification_Postprocess();
        bool Model_Postprocess();
        void Model_Draw();
        void Model_Print() const;
        bool Model_Result_Save(InferenceResult& result, const std::string& image_name);
        void CalculateMetrics(InferenceResult& result);
        bool LoadGroundTruthData();
        float CalculateIoU(const BBoxInfo& box1, const BBoxInfo& box2);
        ModelType DetermineModelType(const std::string& model_name);

        void CalculateFeatureMapSizes(int input_height, int input_width);

        // 特征图尺寸变量
        int H_8_;    // 特征图高度 - 小目标
        int W_8_;    // 特征图宽度 - 小目标 
        int H_16_;   // 特征图高度 - 中目标
        int W_16_;   // 特征图宽度 - 中目标
        int H_32_;   // 特征图高度 - 大目标
        int W_32_;   // 特征图宽度 - 大目标
        int H_4_;    // 特征图高度 - 分割
        int W_4_;    // 特征图宽度 - 分割

        // 计算特征图尺寸的函数


        std::string model_name_;      // 模型名称
        std::string task_type_;       // 任务类型
        std::string model_path_;      // 模型文件路径
        std::string task_id_;         // 任务ID
        std::string output_path_;     // 输出路径
        int classes_num_;             // 类别数量
        float nms_threshold_;         // NMS阈值
        float score_threshold_;       // 置信度阈值
        int nms_top_k_;              // NMS保留的最大框数
        bool is_initialized_;         // 初始化状态标志
        float font_size_;            // 绘制文字大小
        float font_thickness_;       // 绘制文字粗细
        float line_size_;            // 绘制线条粗细
        hbDNNTaskHandle_t task_handle_;          // 推理任务句柄
        ModelType model_type_;        // 模型类型
        int output_count_;            // 输出tensor数量

        float total_inference_time_;
        float total_preprocess_time_;
        float total_postprocess_time_;
        float total_time_;

        std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0,     
                                 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 
                                 116.0, 90.0, 156.0, 198.0, 373.0, 326.0};// 初始化anchors

        std::vector<std::vector<float>> s_anchors_;
        std::vector<std::vector<float>> m_anchors_;
        std::vector<std::vector<float>> l_anchors_;

        int input_h_;// 输入高度
        int input_w_;// 输入宽度
        int output_order_[10]; // 输出顺序映射，YOLOv8-Seg和YOLO11-Seg有10个输出

        float x_scale_;                          // X方向缩放比例
        float y_scale_;                          // Y方向缩放比例
        int x_shift_;                            // X方向偏移量
        int y_shift_;                            // Y方向偏移量
        cv::Mat resized_img_;                    // 缩放后的图像
        cv::Mat input_img_;                       // 输入图像
        cv::Mat output_img_;                      // 输出图像
        hbDNNTensor input_tensor_;               // 输入tensor

        hbPackedDNNHandle_t packed_dnn_handle_;
        hbDNNHandle_t dnn_handle_;// 模型句柄
        hbDNNTensorProperties input_properties_; // 输入tensor属性

        hbDNNTensor* output_tensors_;// 输出tensor数组

        // 检测结果存储
        std::vector<std::vector<cv::Rect2d>> bboxes_;  // 每个类别的边界框
        std::vector<std::vector<float>> scores_;       // 每个类别的得分
        std::vector<std::vector<int>> indices_;        // NMS后的索引
        std::vector<std::string> class_names_;         // 类别名称
        std::string label_path_;
        std::vector<BBoxInfo> gt_boxes_; // Ground truth boxes from label data

        // 分割相关数据
        std::vector<cv::Mat> masks_; // 存储每个最终检测实例的二值掩码 (尺寸与原图相同)
        cv::Mat proto_;             // 存储原型掩码 (来自模型输出 H/4 x W/4 x 32)
        std::vector<std::vector<float>> mask_coeffs_; // 存储每个通过NMS的检测框的32个掩码系数
        
        // YOLO11-Seg/YOLO8-Seg的掩码系数数量，默认为32
        const int MCES_ = 32;

        // FCN语义分割相关数据
        cv::Mat semantic_mask_;     // 语义分割结果掩码
        std::vector<cv::Vec3b> color_map_; // 类别颜色映射表
        
};

#endif // DETECTOR_H