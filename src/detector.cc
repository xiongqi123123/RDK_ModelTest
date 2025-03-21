#include "detector.hpp"
BPU_Detect::BPU_Detect(const std::string& model_name,
                       const std::string& task_type,
                       const std::string& model_path,
                       const int classes_num,
                       const float nms_threshold,
                       const float score_threshold,
                       const int nms_top_k)
    : model_name_(model_name),
      task_type_(task_type),
      model_path_(model_path),
      classes_num_(classes_num),
      nms_threshold_(nms_threshold),
      score_threshold_(score_threshold),
      nms_top_k_(nms_top_k),
      is_initialized_(false),
      font_size_(DEFAULT_FONT_SIZE),
      font_thickness_(DEFAULT_FONT_THICKNESS),
      line_size_(DEFAULT_LINE_SIZE),
      task_handle_(nullptr)
{
    Model_Init();
}

BPU_Detect::~BPU_Detect()
{
    if(is_initialized_)
    {
        Model_Release();
    }
}

bool BPU_Detect::Model_Anchor_Init()
{
    for(int i = 0; i < 3; i++) {
        s_anchors_.push_back({anchors[i*2], anchors[i*2+1]});
        m_anchors_.push_back({anchors[i*2+6], anchors[i*2+7]});
        l_anchors_.push_back({anchors[i*2+12], anchors[i*2+13]});
    }
    return true;
}

bool BPU_Detect::Model_Load()
{
    const char* model_file_name = model_path_.c_str(); //获取文件路径字符指针
    if(model_file_name == nullptr)
    {
        std::cout << "model file name is nullptr" << std::endl;
        return false;
    }
    RDK_CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&packed_dnn_handle_, &model_file_name, 1),
        "Initialize model from file failed");//调用模型加载API
    return true;

}

bool BPU_Detect::Model_Output_Order()
{
    if (task_type_ == "detection"){
    // 初始化默认顺序
    output_order_[0] = 0;  // 默认第1个输出
    output_order_[1] = 1;  // 默认第2个输出
    output_order_[2] = 2;  // 默认第3个输出
    // 定义期望的输出特征图尺寸和通道数
    int32_t expected_shapes[3][3] = {
        {H_8,  W_8,  3 * (5 + classes_num_)},   // 小目标特征图: H/8 x W/8
        {H_16, W_16, 3 * (5 + classes_num_)},   // 中目标特征图: H/16 x W/16
        {H_32, W_32, 3 * (5 + classes_num_)}    // 大目标特征图: H/32 x W/32
    };
    // 遍历每个期望的输出尺度
    for(int i = 0; i < 3; i++) {
        // 遍历实际的输出节点
        for(int j = 0; j < 3; j++) {
            hbDNNTensorProperties output_properties;// 获取当前输出节点的属性
            RDK_CHECK_SUCCESS(
                hbDNNGetOutputTensorProperties(&output_properties, dnn_handle_, j),
                "Get output tensor properties failed");
            // 获取实际的特征图尺寸和通道数
            int32_t actual_h = output_properties.validShape.dimensionSize[1];
            int32_t actual_w = output_properties.validShape.dimensionSize[2];
            int32_t actual_c = output_properties.validShape.dimensionSize[3];
            // 如果实际尺寸和通道数与期望的匹配
            if(actual_h == expected_shapes[i][0] && 
            actual_w == expected_shapes[i][1] && 
            actual_c == expected_shapes[i][2]) {
                output_order_[i] = j;// 记录正确的输出顺序
                break;
                }
            }
        }
        // 打印输出顺序映射信息
        std::cout << "\n============ Output Order Mapping ============" << std::endl;
        std::cout << "Small object  (1/" << 8  << "): output[" << output_order_[0] << "]" << std::endl;
        std::cout << "Medium object (1/" << 16 << "): output[" << output_order_[1] << "]" << std::endl;
        std::cout << "Large object  (1/" << 32 << "): output[" << output_order_[2] << "]" << std::endl;
        std::cout << "==========================================\n" << std::endl;
    }
    return true;
}

bool BPU_Detect::Model_Info_check()
{
    const char** model_name_list; //创建模型列表变量
    int model_count = 0; //创建模型打包数量变量
    RDK_CHECK_SUCCESS(
        hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle_),
        "hbDNNGetModelNameList failed");
    if(model_count > 1) {
    std::cout << "Model count: " << model_count << std::endl;
    std::cout << "Please check the model count!" << std::endl;
    return false;
    }
    model_name_ = model_name_list[0];

    RDK_CHECK_SUCCESS(
    hbDNNGetModelHandle(&dnn_handle_, packed_dnn_handle_, model_name_.c_str()),
    "hbDNNGetModelHandle failed");

    // 获取输入信息
    int32_t input_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetInputCount(&input_count, dnn_handle_),
        "hbDNNGetInputCount failed");

    if(input_count > 1){
        std::cout << "模型输入节点大于1，请检查！" << std::endl;
        return false;
    }

    RDK_CHECK_SUCCESS(
        hbDNNGetInputTensorProperties(&input_properties_, dnn_handle_, 0),
        "hbDNNGetInputTensorProperties failed");

    //检查模型的输入类型
    if(input_properties_.validShape.numDimensions == 4){
        std::cout << "输入tensor类型: HB_DNN_IMG_TYPE_NV12" << std::endl;
    }
    else{
        std::cout << "输入tensor类型不是HB_DNN_IMG_TYPE_NV12，请检查！" << std::endl;
        return false;
    }

    //检查模型的输入数据排布
    if(input_properties_.tensorType == 1){
        std::cout << "输入tensor数据排布: HB_DNN_LAYOUT_NCHW" << std::endl;
    }
    else{
        std::cout << "输入tensor数据排布不是HB_DNN_LAYOUT_NCHW，请检查！" << std::endl;
        return false;
    }

    // 检查模型输入Tensor数据的valid shape
    input_h_ = input_properties_.validShape.dimensionSize[2];
    input_w_ = input_properties_.validShape.dimensionSize[3];
    if (input_properties_.validShape.numDimensions == 4)
    {
        std::cout << "输入的尺寸为: (" << input_properties_.validShape.dimensionSize[0];
        std::cout << ", " << input_properties_.validShape.dimensionSize[1];
        std::cout << ", " << input_h_;
        std::cout << ", " << input_w_ << ")" << std::endl;
        if (task_type_ == "detection"){
            if(input_h_ == 640 && input_w_ == 640){
                std::cout << "输入尺寸为640x640，符合检测任务要求" << std::endl;
            }
            else{
                std::cout << "输入尺寸不符合检测任务要求，请检查！" << std::endl;
                return false;
            }
        }
        else if(task_type_ == "classification"){
            if(input_h_ == 224 && input_w_ == 224){
                std::cout << "输入尺寸为224x224，符合分类任务要求" << std::endl;
            }
            else{
                std::cout << "输入尺寸不符合分类任务要求，请检查！" << std::endl;
                return false;
            }
        }
    }
    else{
        std::cout << "输入尺寸不符合要求，请检查！" << std::endl;
        return false;
    }

    int32_t output_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetOutputCount(&output_count, dnn_handle_),
        "hbDNNGetOutputCount failed");

    output_tensors_ = new hbDNNTensor[output_count];
    memset(output_tensors_, 0, sizeof(hbDNNTensor) * output_count);  // 初始化为0

    if (!Model_Output_Order()){
        std::cout << "输出顺序映射调整失败，请检查！" << std::endl;
        return false;
    }

    return true;
}

bool BPU_Detect::Model_Preprocess(const cv::Mat& input_img)
{
    // 使用letterbox方式进行预处理
    x_scale_ = std::min(1.0f * input_h_ / input_img.rows, 1.0f * input_w_ / input_img.cols);
    y_scale_ = x_scale_;

    int new_w = input_img.cols * x_scale_;
    x_shift_ = (input_w_ - new_w) / 2;
    int x_other = input_w_ - new_w - x_shift_;

    int new_h = input_img.rows * y_scale_;
    y_shift_ = (input_h_ - new_h) / 2;
    int y_other = input_h_ - new_h - y_shift_;

    cv::resize(input_img, resized_img_, cv::Size(new_w, new_h));
    cv::copyMakeBorder(resized_img_, resized_img_, y_shift_, y_other, 
                    x_shift_, x_other, cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));
    // 转换为NV12格式
    cv::Mat yuv_mat;
    cv::cvtColor(resized_img_, yuv_mat, cv::COLOR_BGR2YUV_I420);

    // 准备输入tensor
    hbSysAllocCachedMem(&input_tensor_.sysMem[0], int(3 * input_h_ * input_w_ / 2));
    uint8_t* yuv = yuv_mat.ptr<uint8_t>();
    uint8_t* ynv12 = (uint8_t*)input_tensor_.sysMem[0].virAddr;
    // 计算UV部分的高度和宽度，以及Y部分的大小
    int uv_height = input_h_ / 2;
    int uv_width = input_w_ / 2;
    int y_size = input_h_ * input_w_;
    // 将Y分量数据复制到输入张量
    memcpy(ynv12, yuv, y_size);
    // 获取NV12格式的UV分量位置
    uint8_t* nv12 = ynv12 + y_size;
    uint8_t* u_data = yuv + y_size;
    uint8_t* v_data = u_data + uv_height * uv_width;
    // 将U和V分量交替写入NV12格式
    for(int i = 0; i < uv_width * uv_height; i++) {
        *nv12++ = *u_data++;
        *nv12++ = *v_data++;
    }
    // 将内存缓存清理，确保数据准备好可以供模型使用
    hbSysFlushMem(&input_tensor_.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);// 清除缓存，确保数据同步

    return true;
}

bool BPU_Detect::Model_Detector()
{
    // 初始化任务句柄为nullptr
    task_handle_ = nullptr;
    // 初始化输入tensor属性
    input_tensor_.properties = input_properties_;
    // 获取输出tensor属性
    for(int i = 0; i < 3; i++) {
        hbDNNTensorProperties output_properties;
        RDK_CHECK_SUCCESS(
            hbDNNGetOutputTensorProperties(&output_properties, dnn_handle_, i),
            "Get output tensor properties failed");
        output_tensors_[i].properties = output_properties;

        // 为输出分配内存
        int out_aligned_size = output_properties.alignedByteSize;
        RDK_CHECK_SUCCESS(
            hbSysAllocCachedMem(&output_tensors_[i].sysMem[0], out_aligned_size),
            "Allocate output memory failed");
    }

    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
    RDK_CHECK_SUCCESS(
            hbDNNInfer(&task_handle_, &output_tensors_, &input_tensor_, dnn_handle_, &infer_ctrl_param),
            "Model inference failed");
    RDK_CHECK_SUCCESS(
        hbDNNWaitTaskDone(task_handle_, 0),
        "Wait task done failed");
        
    return true;

}

// 特征图处理辅助函数
void BPU_Detect::Model_Process_FeatureMap(hbDNNTensor& output_tensor, 
                                  int height, int width,
                                  const std::vector<std::vector<float>>& anchors,
                                  float conf_thres_raw) {
    // 检查量化类型
    if (output_tensor.properties.quantiType != NONE) {
        std::cout << "Output tensor quantization type should be NONE!" << std::endl;
        return;
    }
    
    // 刷新内存
    hbSysFlushMem(&output_tensor.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    
    // 获取输出数据指针
    auto* raw_data = reinterpret_cast<float*>(output_tensor.sysMem[0].virAddr);
    
    // 遍历特征图的每个位置
    for(int h = 0; h < height; h++) {
        for(int w = 0; w < width; w++) {
            for(const auto& anchor : anchors) {
                // 获取当前位置的预测数据
                float* cur_raw = raw_data;
                raw_data += (5 + classes_num_);
                
                // 条件概率过滤
                if(cur_raw[4] < conf_thres_raw) continue;
                
                // 找到最大类别概率
                int cls_id = 5;
                int end = classes_num_ + 5;
                for(int i = 6; i < end; i++) {
                    if(cur_raw[i] > cur_raw[cls_id]) {
                        cls_id = i;
                    }
                }
                
                // 计算最终得分
                float score = 1.0f / (1.0f + std::exp(-cur_raw[4])) * 
                            1.0f / (1.0f + std::exp(-cur_raw[cls_id]));
                
                // 得分过滤
                if(score < score_threshold_) continue;
                cls_id -= 5;
                
                // 解码边界框
                float stride = input_h_ / height;
                float center_x = ((1.0f / (1.0f + std::exp(-cur_raw[0]))) * 2 - 0.5f + w) * stride;
                float center_y = ((1.0f / (1.0f + std::exp(-cur_raw[1]))) * 2 - 0.5f + h) * stride;
                float bbox_w = std::pow((1.0f / (1.0f + std::exp(-cur_raw[2]))) * 2, 2) * anchor[0];
                float bbox_h = std::pow((1.0f / (1.0f + std::exp(-cur_raw[3]))) * 2, 2) * anchor[1];
                float bbox_x = center_x - bbox_w / 2.0f;
                float bbox_y = center_y - bbox_h / 2.0f;
                
                // 保存检测结果
                bboxes_[cls_id].push_back(cv::Rect2d(bbox_x, bbox_y, bbox_w, bbox_h));
                scores_[cls_id].push_back(score);
            }
        }
    }
}


bool BPU_Detect::Model_Detection_Postprocess()
{
    // 清空并预分配存储空间
    bboxes_.clear();  // 清空边界框
    scores_.clear();  // 清空得分
    indices_.clear(); // 清空索引
    
    bboxes_.resize(classes_num_);
    scores_.resize(classes_num_);
    indices_.resize(classes_num_);

    float conf_thres_raw = -log(1 / score_threshold_ - 1);

    Model_Process_FeatureMap(output_tensors_[output_order_[0]], H_8, W_8, s_anchors_, conf_thres_raw);
    Model_Process_FeatureMap(output_tensors_[output_order_[1]], H_16, W_16, m_anchors_, conf_thres_raw);
    Model_Process_FeatureMap(output_tensors_[output_order_[2]], H_32, W_32, l_anchors_, conf_thres_raw);

    for(int i = 0; i < classes_num_; i++) {
        if(!bboxes_[i].empty()) {
            cv::dnn::NMSBoxes(bboxes_[i], scores_[i], score_threshold_, 
                        nms_threshold_, indices_[i], 1.f, nms_top_k_);
        }
    }

    return true;
}

bool BPU_Detect::Model_Classification_Postprocess()
{
    // 获取输出tensor
    // TODO: 实现分类任务的后处理逻辑
    return true;
}


bool BPU_Detect::Model_Postprocess()
{
    if(task_type_ == "detection"){
        if(!Model_Detection_Postprocess()){
            std::cout << "Detection postprocess failed" << std::endl;
            return false;
        }
    }
    else if(task_type_ == "classification"){
        if(!Model_Classification_Postprocess()){
            std::cout << "Classification postprocess failed" << std::endl;
            return false;
        }
    }
    return true;
}

void BPU_Detect::Model_Draw(){
    if (task_type_ == "detection"){
        for(int cls_id = 0; cls_id < classes_num_; cls_id++) {
            if(!indices_[cls_id].empty()) {
                for(size_t i = 0; i < indices_[cls_id].size(); i++) {
                    int idx = indices_[cls_id][i];
                    float x1 = (bboxes_[cls_id][idx].x - x_shift_) / x_scale_;
                    float y1 = (bboxes_[cls_id][idx].y - y_shift_) / y_scale_;
                    float x2 = x1 + (bboxes_[cls_id][idx].width) / x_scale_;
                    float y2 = y1 + (bboxes_[cls_id][idx].height) / y_scale_;
                    float score = scores_[cls_id][idx];
                    
                    // 绘制边界框
                    cv::rectangle(output_img_, cv::Point(x1, y1), cv::Point(x2, y2), 
                                cv::Scalar(255, 0, 0), line_size_);
                    
                    // 绘制标签
                    std::string text = (cls_id < static_cast<int>(class_names_.size()) ? class_names_[cls_id] : "class" + std::to_string(cls_id)) + 
                                    ": " + std::to_string(static_cast<int>(score * 100)) + "%";
                    cv::putText(output_img_, text, cv::Point(x1, y1 - 5), 
                            cv::FONT_HERSHEY_SIMPLEX, font_size_, 
                            cv::Scalar(0, 0, 255), font_thickness_, cv::LINE_AA);
                }
            }
        }
    }
    else if(task_type_ == "classification"){
        // TODO: 分类任务结果绘制
    }
}

void BPU_Detect::Model_Print() const {
    if(task_type_ == "detection"){
        int total_detections = 0;
        for(int cls_id = 0; cls_id < classes_num_; cls_id++) {
            total_detections += indices_[cls_id].size();
        }
        std::cout << "\n============ Detection Results ============" << std::endl;
        std::cout << "Total detections: " << total_detections << std::endl;
        
        for(int cls_id = 0; cls_id < classes_num_; cls_id++) {
            if(!indices_[cls_id].empty()) {
                std::cout << "\nClass: " << class_names_[cls_id] << std::endl;
                std::cout << "Number of detections: " << indices_[cls_id].size() << std::endl;
                std::cout << "Details:" << std::endl;
                
                for(size_t i = 0; i < indices_[cls_id].size(); i++) {
                    int idx = indices_[cls_id][i];
                    float x1 = (bboxes_[cls_id][idx].x - x_shift_) / x_scale_;
                    float y1 = (bboxes_[cls_id][idx].y - y_shift_) / y_scale_;
                    float x2 = x1 + (bboxes_[cls_id][idx].width) / x_scale_;
                    float y2 = y1 + (bboxes_[cls_id][idx].height) / y_scale_;
                    float score = scores_[cls_id][idx];
                    
                    // 打印每个检测框的详细信息
                    std::cout << "  Detection " << i + 1 << ":" << std::endl;
                    std::cout << "    Position: (" << x1 << ", " << y1 << ") to (" << x2 << ", " << y2 << ")" << std::endl;
                    std::cout << "    Confidence: " << std::fixed << std::setprecision(2) << score * 100 << "%" << std::endl;
                }
            }
        }
        std::cout << "========================================\n" << std::endl;
    }
    else if(task_type_ == "classification"){
        
        // TODO: 分类任务结果打印

    }
}

bool BPU_Detect::Model_Init(){
    if(is_initialized_) {
        std::cout << "Model already initialized!" << std::endl;
        return true;
    }
    if(!Model_Load()) {
        std::cout << "Failed to load model!" << std::endl;
        return false;
    }
    if(!Model_Info_check()) {
        std::cout << "Failed to get model info!" << std::endl;
        return false;
    }
    
    // 初始化锚点
    if(!Model_Anchor_Init()) {
        std::cout << "Failed to initialize anchors!" << std::endl;
        return false;
    }
    
    // 预分配结果数组空间
    bboxes_.resize(classes_num_);
    scores_.resize(classes_num_);
    indices_.resize(classes_num_);
    
    // 如果用户未提供类别名称，则使用默认名称
    if(class_names_.empty()) {
        class_names_.resize(classes_num_);
        for(int i = 0; i < classes_num_; i++) {
            class_names_[i] = "class" + std::to_string(i);
        }
    }
    
    is_initialized_ = true;
    return true;
}

bool BPU_Detect::Model_Inference(const cv::Mat& input_img, cv::Mat& output_img){
    if(!is_initialized_) {
        std::cout << "Please initialize first!" << std::endl;
        return false;
    }
    
    // 保存输入和输出图像
    input_img.copyTo(input_img_);
    input_img.copyTo(output_img);
    output_img_ = output_img;
    
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    if(!Model_Preprocess(input_img)) {
        return false;
    }
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    total_preprocess_time_ = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end - preprocess_start).count();

    auto inference_start = std::chrono::high_resolution_clock::now();
    if(!Model_Detector()) {
        return false;
    }
    auto inference_end = std::chrono::high_resolution_clock::now();
    total_inference_time_ = std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - inference_start).count();

    auto postprocess_start = std::chrono::high_resolution_clock::now();
    if(!Model_Postprocess()) {
        return false;
    }
    auto postprocess_end = std::chrono::high_resolution_clock::now();
    total_postprocess_time_ = std::chrono::duration_cast<std::chrono::milliseconds>(postprocess_end - postprocess_start).count();

    total_time_ = total_preprocess_time_ + total_inference_time_ + total_postprocess_time_;
    std::cout << "Total time: " << total_time_ << "ms" << std::endl;
    Model_Draw();
    Model_Print();
    output_img = output_img_; // 确保输出图像被正确更新
    return true;
}

// 释放资源实现
bool BPU_Detect::Model_Release() {
    if(!is_initialized_) {
        return true;
    }
    
    // 释放任务
    if(task_handle_) {
        hbDNNReleaseTask(task_handle_);
        task_handle_ = nullptr;
    }
    
    try {
        // 释放输入内存
        if(input_tensor_.sysMem[0].virAddr) {
            hbSysFreeMem(&(input_tensor_.sysMem[0]));
        }
        
        // 释放输出内存
        for(int i = 0; i < 3; i++) {
            if(output_tensors_ && output_tensors_[i].sysMem[0].virAddr) {
                hbSysFreeMem(&(output_tensors_[i].sysMem[0]));
            }
        }
        
        if(output_tensors_) {
            delete[] output_tensors_;
            output_tensors_ = nullptr;
        }
        
        // 释放模型
        if(packed_dnn_handle_) {
            hbDNNRelease(packed_dnn_handle_);
            packed_dnn_handle_ = nullptr;
        }
    } catch(const std::exception& e) {
        std::cout << "Exception during release: " << e.what() << std::endl;
    }
    
    is_initialized_ = false;
    return true;
}

