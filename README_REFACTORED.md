# 早期谣言检测系统 - 重构版本

## 📁 项目结构

```
D2-master/
├── run.py                  # 原始单文件版本（保持不变）
├── main_refactored.py      # 重构后的主程序入口
├── config.py              # 配置模块：超参数和常量设置
├── data_processor.py       # 数据处理模块：文本预处理、图数据处理
├── models.py              # 模型定义模块：网络架构定义
├── trainer.py             # 训练模块：训练和评估逻辑
├── utils.py               # 工具模块：辅助函数
└── data/                  # 数据目录
    └── twitter16/
        ├── source_tweets.txt
        ├── label.txt
        └── tree/
```

## 🔧 模块说明

### 1. config.py - 配置模块
- **功能**: 集中管理所有超参数和常量
- **包含**: 模型参数、训练参数、损失权重等
- **优势**: 便于调参和配置管理

### 2. data_processor.py - 数据处理模块
- **功能**: 完整的数据处理流水线
- **包含**: 
  - 文本清洗和分词 (`clean_str_cut`)
  - Word2Vec模型训练 (`load_word2vec_model`)
  - 节点特征提取 (`get_node_features`)
  - 传播树解析 (`parse_tree_file`)
  - 级联分割 (`split_cascade_by_percentage`)
  - PyTorch Geometric格式转换 (`convert_to_pytorch_geometric`)
  - 标签处理 (`add_numeric_labels_to_data`)
- **主函数**: `load_and_process_data()` - 一站式数据处理

### 3. models.py - 模型定义模块
- **功能**: 定义所有神经网络模型
- **包含**:
  - `SharedEncoder`: 共享图神经网络编码器
  - `CascadePredictor`: 级联链接预测器
  - `RumorDetector`: 谣言检测器
  - `EndToEndModel`: 端到端联合模型
- **工厂函数**: `create_models()` - 便于模型实例化

### 4. trainer.py - 训练模块
- **功能**: 训练和评估逻辑
- **包含**:
  - `train_epoch()`: 单轮训练
  - `evaluate_model()`: 模型评估
  - `train_single_fold()`: 单折训练
  - `cross_validation_training()`: K折交叉验证
- **特色**: 支持早停、梯度裁剪、学习率调度

### 5. utils.py - 工具模块
- **功能**: 辅助函数集合
- **包含**:
  - `custom_negative_sampling()`: 自定义负采样
  - `setup_device()`: 设备设置（GPU/CPU）
  - `safe_model_to_device()`: 安全的模型设备转移
  - `print_model_info()`: 模型信息打印

### 6. main.py - 主程序
- **功能**: 协调所有模块，执行完整流程
- **流程**:
  1. 数据加载和处理
  2. 设备设置
  3. 模型创建
  4. 交叉验证训练
  5. 结果展示

## 🚀 使用方法

### 运行重构版本
```bash
python main.py
```



