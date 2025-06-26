# -*- coding: utf-8 -*-
"""
配置文件模块
包含所有超参数和常量设置
"""

# 模型超参数
VECTOR_SIZE = 300           # Word2Vec向量维度
WINDOW_SIZE = 5             # Word2Vec窗口大小
MIN_COUNT = 1               # Word2Vec最小词频
WORKERS = 4                 # Word2Vec训练线程数

# 网络结构参数
IN_CHANNELS = VECTOR_SIZE + 5   # 输入通道数（Word2Vec维度 + 额外特征）
HIDDEN_CHANNELS = 512           # 隐藏层通道数
OUT_CHANNELS = 4                # 输出类别数（false, true, unverified, non-rumor）
DROPOUT = 0.25                  # Dropout概率

# 训练参数
LEARNING_RATE = 1e-4        # 学习率
WEIGHT_DECAY = 5e-5         # 权重衰减
STEP_SIZE = 10              # 学习率调度器步长
GAMMA = 0.5                 # 学习率衰减因子
K_FOLDS = 10                # K折交叉验证的折数
NUM_EPOCHS = 200            # 最大训练轮数
BATCH_SIZE = 64             # 批次大小
PATIENCE = 10               # 早停耐心值

# 损失函数权重
LOSS_RATIO_LINK_PRED = 0.4              # 链接预测损失权重
LOSS_RATIO_RUMOR = 1.0 - LOSS_RATIO_LINK_PRED  # 谣言检测损失权重

# 训练优化参数
MAX_GRAD_NORM = 1.0         # 梯度裁剪阈值
PERCENTAGE = 0.75           # 早期级联分割百分比
