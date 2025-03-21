# d-modelsquare-test

模型广场的板端验证仓库

## 项目简介
本项目用于在RDK BPU硬件上验证和部署深度学习模型，支持通过JSON配置文件进行模型推理任务配置。

## 功能特点
- 支持从命令行指定JSON配置文件
- 支持解析JSON配置文件中的模型参数
- 支持目标检测任务（如YOLOv5）
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
./main path/to/config.json
```

## 配置文件格式
```json
{
    "task_id": "1",
    "task_type": "detection",
    "model_name": "yolov5s",
    "model_path": "models/yolov5s.onnx",
    "classes_num": 1,
    "classes_labels":["tennis"],
    "image_path": "images/tennis.jpg"
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

## 示例
```bash
./main dataset/input/input.json
```

## 常见问题
1. 如何添加新的模型？
   - 将模型文件放入models目录
   - 修改配置文件中的model_path和相关参数

2. 支持哪些任务类型？
   - 目前支持detection（目标检测）
   - 未来计划支持classification（分类）、segmentation（分割）等