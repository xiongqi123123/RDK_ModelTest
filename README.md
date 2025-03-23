
## 项目简介
本项目用于在RDK BPU硬件上验证和部署深度学习模型，支持通过JSON配置文件进行模型推理任务配置。

## 功能特点
- 支持从命令行指定JSON配置文件
- 支持解析JSON配置文件中的模型参数
- 支持目标检测任务（如YOLOv5、YOLO11、YOLOv8等）
- 支持图像分类任务（如ResNet、EfficientFormerV2、MobileNetV4、RepViT等）
- 支持模型性能评估并生成JSON格式的结果报告
- 使用BPU硬件加速推理

## 依赖项
- OpenCV
- nlohmann/json (JSON解析库)
- RDK BPU库

## 编译方法
```bash
mkdir build
cd build
cmake ..
make
```

## 使用方法
1. 准备好配置文件（JSON格式）
2. 运行程序，传入配置文件路径
```bash
./d_modelsquare_test path/to/config.json
```

## 配置文件格式

### 目标检测任务配置示例
```json
{
    "task_id": "1",
    "task_type": "detection",
    "model_name": "yolov5s",
    "model_path": "models/yolov5s.bin",
    "classes_num": 1,
    "classes_labels":["tennis"],
    "image_path": "images/tennis.jpg",
    "label_path": "labels/tennis.txt",
    "output_path": "output/"
}
```

### 图像分类任务配置示例
```json
{
    "task_id": "1",
    "task_type": "classification",
    "model_name": "MobileNetv4",
    "model_path": "models/MobileNetv4.bin",
    "classes_num": 1000,
    "classes_labels":["tench", "goldfish", "..."],
    "image_path": "images/sample.jpg",
    "label_path": "",
    "output_path": "output/"
}
```

## 参数说明
- task_id: 任务ID
- task_type: 任务类型（detection、classification等）
- model_name: 模型名称
- model_path: 模型文件路径
- classes_num: 类别数量
- classes_labels: 类别标签数组
- image_path: 测试图像路径
- label_path: 标签文件路径（用于计算评估指标，可选）
- output_path: 输出结果保存路径

## 结果输出
程序会在指定的输出目录生成两种文件：
1. 可视化结果图像（result_{task_id}.jpg）
2. JSON格式的结果文件（result_{task_id}.json）

### 检测任务JSON结果示例
```json
{
    "task_id": "1",
    "task_type": "detection",
    "model_name": "yolov5s",
    "performance": {
        "fps": 35.0,
        "preprocess_time": 3.0,
        "inference_time": 25.0,
        "postprocess_time": 2.0,
        "total_time": 30.0
    },
    "metrics": {
        "mAP50": 0.95,
        "mAP50-95": 0.75,
        "precision": 0.92,
        "recall": 0.88
    },
    "detections": [...]
}
```

### 分类任务JSON结果示例
```json
{
    "task_id": "1",
    "task_type": "classification",
    "model_name": "MobileNetv4",
    "performance": {
        "fps": 166.67,
        "preprocess_time": 2.0,
        "inference_time": 4.0,
        "postprocess_time": 0.0,
        "total_time": 6.0
    },
    "metrics": {
        "accuracy_top1": 0.85,
        "accuracy_top5": 0.95
    },
    "classifications": [...]
}
```

## 示例
```bash
./d_modelsquare_test dataset/input/input_yolov5.json  # 运行YOLOv5检测
./d_modelsquare_test dataset/input/input_yolo11.json  # 运行YOLO11检测
./d_modelsquare_test dataset/input/input_mobilenet.json  # 运行MobileNet分类
```

## 常见问题
1. 如何添加新的模型？
   - 将模型文件放入models目录
   - 修改配置文件中的model_path和相关参数
   - 对于新的模型架构，可能需要调整detector.cc中的相关处理逻辑

2. 支持哪些任务类型？
   - 目前支持detection（目标检测）：YOLOv5、YOLO11、YOLOv8
   - 已支持classification（图像分类）：ResNet、EfficientFormerV2、MobileNetV4、RepViT等

3. 如何获取评估指标？
   - 检测任务：提供YOLO格式的标注文件作为label_path
   - 分类任务：模型会自动计算Top-1和Top-5准确率
   - 所有评估指标会保存在JSON结果文件中