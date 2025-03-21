// C/C++ Standard Librarys
#include <iostream>     // 输入输出流
#include <vector>      // 向量容器
#include <algorithm>   // 算法库
#include <chrono>      // 时间相关功能
#include <iomanip>     // 输入输出格式控制

// Thrid Party Librarys
#include <opencv2/opencv.hpp>      // OpenCV主要头文件
#include <opencv2/dnn/dnn.hpp>     // OpenCV深度学习模块

// RDK BPU libDNN API
#include "dnn/hb_dnn.h"           // BPU基础功能
#include "dnn/hb_dnn_ext.h"       // BPU扩展功能
#include "dnn/plugin/hb_dnn_layer.h"    // BPU层定义
#include "dnn/plugin/hb_dnn_plugin.h"   // BPU插件
#include "dnn/hb_sys.h"           // BPU系统功能

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

// 特征图尺寸宏定义
#define H_8 80    // 特征图高度 - 小目标
#define W_8 80    // 特征图宽度 - 小目标 
#define H_16 40   // 特征图高度 - 中目标
#define W_16 40   // 特征图宽度 - 中目标
#define H_32 20   // 特征图高度 - 大目标
#define W_32 20   // 特征图宽度 - 大目标


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
        bool Model_Inference(const cv::Mat& input_img, cv::Mat& output_img);
        bool Model_Release();
        void SetClassNames(const std::vector<std::string>& class_names) { class_names_ = class_names; }
    private:
        bool Model_Load();
        bool Model_Anchor_Init();
        bool Model_Output_Order();
        bool Model_Info_check();
        bool Model_Preprocess(const cv::Mat& input_img);
        bool Model_Detector();
        bool Model_Detection_Postprocess();
        void Model_Process_FeatureMap(hbDNNTensor& output_tensor, 
                                     int feature_h, 
                                     int feature_w, 
                                     const std::vector<std::vector<float>>& anchors, 
                                     float conf_thres);
        bool Model_Classification_Postprocess();
        bool Model_Postprocess();
        void Model_Draw();
        void Model_Print() const;
        bool Model_Result_Save();

        std::string model_name_;      // 模型名称
        std::string task_type_;       // 任务类型
        std::string model_path_;      // 模型文件路径
        int classes_num_;             // 类别数量
        float nms_threshold_;         // NMS阈值
        float score_threshold_;       // 置信度阈值
        int nms_top_k_;              // NMS保留的最大框数
        bool is_initialized_;         // 初始化状态标志
        float font_size_;            // 绘制文字大小
        float font_thickness_;       // 绘制文字粗细
        float line_size_;            // 绘制线条粗细

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
        int output_order_[3];// 输出顺序映射

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

        hbDNNTaskHandle_t task_handle_;          // 推理任务句柄

        // 检测结果存储
        std::vector<std::vector<cv::Rect2d>> bboxes_;  // 每个类别的边界框
        std::vector<std::vector<float>> scores_;       // 每个类别的得分
        std::vector<std::vector<int>> indices_;        // NMS后的索引
        std::vector<std::string> class_names_;         // 类别名称
};