# -*- coding: utf-8 -*-
"""
模型架构详细解释脚本
通过可视化和逐步演示来理解D2模型的完整工作流程
"""

import torch
import numpy as np
from torch_geometric.data import Data
import torch_geometric.transforms as T

# 假设我们已经有了模型定义（实际运行时需要导入）
# from models import SharedEncoder, CascadePredictor, RumorDetector, EndToEndModel

def explain_model_architecture():
    """
    详细解释D2模型的架构和数据流
    """
    print("=" * 80)
    print("🏗️  D2早期谣言检测模型架构详细解释")
    print("=" * 80)
    
    # === 1. 模型架构概览 ===
    print("\n📋 1. 模型架构概览")
    print("=" * 50)
    print("""
    整个模型由四个核心组件构成：
    
    ┌─────────────────────────────────────────────────────────────┐
    │                    D2 End-to-End Model                     │
    │                                                             │
    │  ┌─────────────────┐    ┌────────────────┐                 │
    │  │  SharedEncoder  │ ──▶│ CascadePredictor│ ──┐             │
    │  │   (GNN核心)     │    │   (链接预测)     │   │             │
    │  │                 │    └────────────────┘   │             │
    │  │                 │                         │ 图重构       │
    │  │                 │ ──▶┌────────────────┐   │             │
    │  │                 │    │ RumorDetector  │ ◀─┘             │
    │  └─────────────────┘    │  (谣言分类)     │                 │
    │                         └────────────────┘                 │
    └─────────────────────────────────────────────────────────────┘
    """)
    
    # === 2. SharedEncoder详细解释 ===
    print("\n🧠 2. SharedEncoder (共享编码器) - 模型的大脑")
    print("=" * 50)
    print("""
    SharedEncoder是整个模型的核心，它的作用是：
    
    🎯 主要功能：
    • 将原始节点特征 (Word2Vec + 度数特征) 编码成高维语义表示
    • 通过GraphSAGE学习节点在传播图中的结构化嵌入
    • 为两个下游任务提供统一的节点表示
    
    🔧 技术细节：
    • 使用两层GraphSAGE卷积层 (305维 → 512维 → 512维)
    • 每层都有BatchNorm + ReLU + Dropout
    • 添加跳跃连接 (ResNet风格) 防止梯度消失
    • 最后通过MLP进行非线性变换
    
    📊 数据流：
    输入: [num_nodes, 305] (Word2Vec 300维 + 度数特征 5维)
      ↓ GraphSAGE Layer 1
    [num_nodes, 512] + BatchNorm + ReLU + Dropout  
      ↓ GraphSAGE Layer 2  
    [num_nodes, 512] + BatchNorm + ReLU + Dropout
      ↓ 跳跃连接 + MLP
    输出: [num_nodes, 512] (高质量节点嵌入)
    """)
    
    # === 3. 为什么需要SharedEncoder？ ===
    print("\n💡 3. 为什么需要SharedEncoder？")
    print("=" * 50)
    print("""
    🤔 核心问题：两个任务如何联合训练？
    
    传统方法的问题：
    ❌ 独立训练两个模型 → 无法共享信息
    ❌ 简单堆叠模型 → 梯度传播困难
    ❌ 特征不一致 → 难以融合
    
    SharedEncoder的解决方案：
    ✅ 统一特征空间：两个任务使用相同的节点表示
    ✅ 共享参数：编码器参数被两个任务共同优化
    ✅ 信息传递：链接预测的结构信息帮助谣言检测
    ✅ 正则化效果：多任务学习防止过拟合
    
    🔗 联合训练机制：
    • SharedEncoder的参数同时接收两个任务的梯度
    • CascadePredictor学到的传播模式→帮助RumorDetector理解传播语义
    • RumorDetector学到的语义信息→帮助CascadePredictor预测合理链接
    """)

def demonstrate_data_flow():
    """
    通过具体例子演示数据流动过程
    """
    print("\n🌊 4. 数据流动详细演示")
    print("=" * 50)
    
    # 创建示例数据
    print("📝 假设我们有一个传播级联的例子：")
    print("""
    传播级联：
    推文A (t=0) → 推文B (t=5) → 推文C (t=8)
                ↓
               推文D (t=6) → 推文E (t=12)
    
    时序分割 (75%): t_threshold = 0.75 * 12 = 9
    • 早期图 (观察到的): A→B, A→D, B→C, D (t ≤ 9)  
    • 预测图 (待预测的): D→E (t > 9)
    """)
    
    # 模拟数据结构
    num_nodes = 5  # A, B, C, D, E
    in_channels = 305  # Word2Vec(300) + 度数特征(5)
    
    print(f"\n📊 数据结构：")
    print(f"• 节点数量: {num_nodes}")
    print(f"• 节点特征维度: {in_channels}")
    print(f"• 早期边: [[0,1], [0,3], [1,2]] (A→B, A→D, B→C)")
    print(f"• 预测边: [[3,4]] (D→E, 待预测)")
    
    # 创建示例数据
    x = torch.randn(num_nodes, in_channels)  # 节点特征
    edge_index = torch.tensor([[0, 0, 1], [1, 3, 2]], dtype=torch.long)  # 早期边
    pred_edge_index = torch.tensor([[3], [4]], dtype=torch.long)  # 预测边
    
    print(f"\n🔢 张量形状：")
    print(f"• x (节点特征): {x.shape}")
    print(f"• edge_index (早期边): {edge_index.shape}")  
    print(f"• pred_edge_index (预测边): {pred_edge_index.shape}")

def explain_cascade_predictor():
    """
    详细解释CascadePredictor的工作原理
    """
    print("\n🔮 5. CascadePredictor (级联预测器) 详解")
    print("=" * 50)
    print("""
    🎯 任务：预测"哪些边会在未来出现"
    
    🔧 工作流程：
    
    步骤1: 编码 (Encode)
    输入: 节点特征x + 早期边edge_index  
    ↓ SharedEncoder  
    输出: 节点嵌入z [num_nodes, 512]
    
    步骤2: 解码 (Decode)  
    输入: 节点嵌入z + 待预测边pred_edge_index
    ↓ 提取源节点和目标节点嵌入
    z_src = z[pred_edge_index[0]]  # [num_pred_edges, 512]
    z_dst = z[pred_edge_index[1]]  # [num_pred_edges, 512]
    ↓ 组合嵌入
    z_combined = cat([z_src, z_dst], dim=1)  # [num_pred_edges, 2, 512]
    ↓ 多头注意力 (4个头)
    attn_output = MultiheadAttention(z_combined, z_combined, z_combined)
    ↓ Sigmoid激活
    链接概率 = sigmoid(attn_output.sum())  # [num_pred_edges]
    
    🧠 多头注意力的作用：
    • 捕获节点间的复杂交互模式
    • 学习不同类型的传播关系 (转发、评论、点赞等)
    • 提供可解释的注意力权重
    
    📈 输出解释：
    • 每个待预测边都有一个概率值 [0,1]
    • 概率 > 0.5 表示"这条边很可能在未来出现"
    • 概率 ≤ 0.5 表示"这条边不太可能出现"
    """)

def explain_rumor_detector():
    """
    详细解释RumorDetector的工作原理
    """
    print("\n🕵️ 6. RumorDetector (谣言检测器) 详解")
    print("=" * 50)
    print("""
    🎯 任务：基于"重构的完整传播图"进行谣言分类
    
    🔧 工作流程：
    
    步骤1: 图重构
    输入: 早期边 + 预测边(概率>0.5)
    重构边 = cat([early_edges, predicted_edges[prob>0.5]], dim=1)
    → 得到"推断的完整传播图"
    
    步骤2: 节点编码
    输入: 节点特征 + 重构边
    ↓ SharedEncoder (复用相同编码器!)
    输出: 节点嵌入 [num_nodes, 512]
    
    步骤3: 图级表示学习
    ↓ SAGPooling自适应池化 (保留最重要的50%节点)
    重要节点嵌入 [num_important_nodes, 512]
    ↓ BatchNorm归一化
    ↓ 全局平均池化 (每个图聚合为单一向量)
    图级表示 [batch_size, 512]
    
    步骤4: 分类
    ↓ 全连接层 [512] → [4]
    ↓ LogSoftmax
    类别概率 [batch_size, 4]  # {false, true, unverified, non-rumor}
    
    🎯 关键创新：
    • 使用"重构图"而非"早期图"进行分类
    • 自适应池化自动发现关键传播节点
    • 端到端优化，链接预测质量直接影响分类性能
    """)

def explain_joint_training():
    """
    解释联合训练的核心机制
    """
    print("\n🤝 7. 联合训练机制详解")
    print("=" * 50)
    print("""
    🧩 核心问题：两个任务如何相互帮助？
    
    📈 损失函数设计：
    总损失 = 0.4 × 链接预测损失 + 0.6 × 谣言检测损失
    
    链接预测损失 = BCELoss(预测概率, 真实边标签)
    谣言检测损失 = CrossEntropyLoss(分类输出, 真实谣言标签)
    
    🔄 反向传播流程：
    
    1. 前向传播：
       输入 → SharedEncoder → 分支到两个任务 → 两个损失
    
    2. 反向传播：
       ┌─ 链接预测梯度 ─┐
       │                ↓
       └─▶ SharedEncoder ◀─── 同时接收两个梯度
                        ↑
       ┌─ 谣言检测梯度 ─┘
    
    3. 参数更新：
       SharedEncoder的参数 = f(链接预测梯度, 谣言检测梯度)
    
    💡 相互促进机制：
    
    CascadePredictor → RumorDetector:
    • 预测出的链接提供更完整的传播图
    • 传播结构信息帮助理解谣言传播模式
    • 例如：谣言通常有"爆发式传播"特征
    
    RumorDetector → CascadePredictor:  
    • 谣言语义信息指导合理的链接预测
    • 真实新闻和谣言的传播模式不同
    • 例如：谣言链接通常更随机，真实新闻更有序
    
    🔧 技术实现：
    • 共享编码器确保特征空间一致
    • 交替优化两个任务 (每个batch同时训练)
    • 梯度累积和反向传播自动实现信息共享
    """)

def explain_innovation():
    """
    解释模型的创新点
    """
    print("\n🚀 8. 模型创新点总结")
    print("=" * 50)
    print("""
    🎯 核心创新：
    
    1. 时序建模创新：
       • 75%/25% 时间分割策略
       • 模拟真实早期检测场景
       • 从"部分观察"预测"完整传播"
    
    2. 图重构机制：
       • 不是基于静态图分类
       • 而是基于"预测重构的动态图"分类  
       • 更接近真实传播过程
    
    3. 联合学习框架：
       • 传播预测 + 内容分析的有机结合
       • 结构信息 + 语义信息的深度融合
       • 两个任务相互促进，共同提升
    
    4. 端到端优化：
       • 整个pipeline可微分
       • 避免传统方法的error propagation
       • 全局最优而非局部最优
    
    🔬 与传统方法对比：
    
    传统方法：
    ❌ 基于手工特征 (转发数、粉丝数等)
    ❌ 独立的文本分析 + 网络分析
    ❌ 静态图分析，忽略时序演化
    
    D2模型：
    ✅ 端到端学习，自动特征提取
    ✅ 联合优化，结构语义融合
    ✅ 动态图重构，时序感知
    ✅ 早期检测，实用价值高
    """)

def run_architecture_explanation():
    """
    运行完整的架构解释
    """
    explain_model_architecture()
    demonstrate_data_flow()
    explain_cascade_predictor()
    explain_rumor_detector()
    explain_joint_training()
    explain_innovation()
    
    print("\n" + "=" * 80)
    print("🎉 模型架构解释完成！")
    print("🔑 关键要点：")
    print("• SharedEncoder是两个任务的桥梁，提供统一的节点表示")
    print("• CascadePredictor预测传播，RumorDetector利用预测结果分类")
    print("• 联合训练通过共享参数实现信息互传和相互促进")
    print("• 端到端优化保证全局最优，避免error propagation")
    print("=" * 80)

if __name__ == "__main__":
    run_architecture_explanation()
