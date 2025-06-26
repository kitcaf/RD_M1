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

### 6. main_refactored.py - 主程序
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
python main_refactored.py
```

### 运行原始版本（对比）
```bash
python run.py
```

## ✨ 重构优势

### 1. **模块化设计**
- 代码按功能分离，便于维护和扩展
- 每个模块职责单一，降低耦合度

### 2. **可重用性**
- 模块可以独立使用和测试
- 便于在其他项目中复用

### 3. **可配置性**
- 所有参数集中在config.py中
- 便于实验和调参

### 4. **可读性**
- 添加了详细的中文注释
- 函数和类有清晰的文档说明

### 5. **可扩展性**
- 新功能可以通过添加新模块实现
- 不影响现有代码结构

### 6. **错误处理**
- 增强的错误处理和用户反馈
- 更好的调试体验

## 📊 功能保持一致

重构后的代码与原始 `run.py` 功能完全一致：
- 相同的模型架构
- 相同的训练流程
- 相同的评估指标
- 相同的实验设置

## 🔄 迁移指南

如果要从原始版本迁移到重构版本：

1. **数据**: 无需改动，使用相同的数据格式
2. **配置**: 在 `config.py` 中调整参数
3. **自定义**: 在对应模块中添加新功能
4. **运行**: 使用 `main_refactored.py` 替代 `run.py`

## 📝 注意事项

1. 确保所有依赖包已安装
2. 数据文件路径需要正确设置
3. GPU环境会自动检测，无GPU时自动使用CPU
4. 建议在运行前检查数据文件的完整性

## 🤝 贡献

欢迎通过以下方式贡献：
- 报告bug
- 建议新功能
- 提交代码改进
- 完善文档
