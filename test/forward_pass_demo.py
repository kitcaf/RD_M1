# -*- coding: utf-8 -*-
"""
实际模型前向传播演示脚本
通过具体代码和数据演示模型的完整工作流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data

def create_mock_models():
    """
    创建简化版的模型组件用于演示
    """
    class MockSharedEncoder(nn.Module):
        def __init__(self, in_channels=305, hidden_channels=512):
            super().__init__()
            self.conv1 = nn.Linear(in_channels, hidden_channels)  # 简化的"GraphSAGE"
            self.conv2 = nn.Linear(hidden_channels, hidden_channels)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
        
        def forward(self, x, edge_index):
            print(f"    📥 SharedEncoder输入: x.shape={x.shape}")
            
            x_1 = F.relu(self.conv1(x))
            print(f"    🔄 第1层GraphSAGE: {x_1.shape}")
            
            x_2 = F.relu(self.conv2(x_1))
            print(f"    🔄 第2层GraphSAGE: {x_2.shape}")
            
            x_out = x_1 + self.mlp(x_2)  # 跳跃连接
            print(f"    📤 SharedEncoder输出: {x_out.shape}")
            return x_out
    
    class MockCascadePredictor(nn.Module):
        def __init__(self, shared_encoder, hidden_channels=512):
            super().__init__()
            self.shared_encoder = shared_encoder
            self.attention = nn.MultiheadAttention(hidden_channels, 4, batch_first=True)
        
        def forward(self, x, edge_index, pred_edge_index):
            print(f"\n  🔮 CascadePredictor工作流程:")
            print(f"    📥 输入边索引: {pred_edge_index}")
            
            # 编码节点
            z = self.shared_encoder(x, edge_index)
            
            # 解码链接
            src, dst = pred_edge_index
            z_src = z[src].unsqueeze(1)
            z_dst = z[dst].unsqueeze(1)
            print(f"    🔗 源节点嵌入: {z_src.shape}")
            print(f"    🔗 目标节点嵌入: {z_dst.shape}")
            
            z_combined = torch.cat([z_src, z_dst], dim=1)
            print(f"    🤝 组合嵌入: {z_combined.shape}")
            
            attn_output, attn_weights = self.attention(z_combined, z_combined, z_combined)
            print(f"    🧠 注意力输出: {attn_output.shape}")
            
            link_probs = torch.sigmoid(attn_output[:, 0, :].sum(dim=-1))
            print(f"    📊 链接概率: {link_probs}")
            return link_probs
    
    class MockRumorDetector(nn.Module):
        def __init__(self, shared_encoder, hidden_channels=512, out_channels=4):
            super().__init__()
            self.shared_encoder = shared_encoder
            self.fc = nn.Linear(hidden_channels, out_channels)
        
        def forward(self, x, edge_index, batch):
            print(f"\n  🕵️ RumorDetector工作流程:")
            
            # 使用共享编码器
            z = self.shared_encoder(x, edge_index)
            
            # 简化的池化（实际是SAGPooling + global_mean_pool）
            pooled = z.mean(dim=0, keepdim=True)  # 简化为全局平均
            print(f"    🏊 池化后: {pooled.shape}")
            
            # 分类
            logits = self.fc(pooled)
            probs = F.log_softmax(logits, dim=-1)
            print(f"    📈 分类概率: {probs}")
            return probs
    
    class MockEndToEndModel(nn.Module):
        def __init__(self, shared_encoder, cascade_predictor, rumor_detector):
            super().__init__()
            self.shared_encoder = shared_encoder
            self.cascade_predictor = cascade_predictor
            self.rumor_detector = rumor_detector
        
        def forward(self, data):
            print(f"\n🚀 EndToEndModel前向传播:")
            print(f"  📥 输入数据: x={data.x.shape}, edge_index={data.edge_index.shape}")
            
            # 步骤1: 链接预测
            print(f"\n  === 步骤1: 链接预测 ===")
            pred_edges = self.cascade_predictor(data.x, data.edge_index, data.pred_edge_index)
            
            # 步骤2: 图重构
            print(f"\n  === 步骤2: 图重构 ===")
            significant_edges = data.pred_edge_index[:, pred_edges > 0.5]
            print(f"    🔍 预测概率>0.5的边: {significant_edges.shape[1]}条")
            
            reconstructed_edge_index = torch.cat([data.edge_index, significant_edges], dim=1)
            print(f"    🏗️ 重构图边数: 原始{data.edge_index.shape[1]} + 预测{significant_edges.shape[1]} = {reconstructed_edge_index.shape[1]}")
            
            # 步骤3: 谣言检测  
            print(f"\n  === 步骤3: 谣言检测 ===")
            rumor_output = self.rumor_detector(data.x, reconstructed_edge_index, None)
            
            return rumor_output, pred_edges
    
    # 创建模型实例
    shared_encoder = MockSharedEncoder()
    cascade_predictor = MockCascadePredictor(shared_encoder)
    rumor_detector = MockRumorDetector(shared_encoder)
    end_to_end_model = MockEndToEndModel(shared_encoder, cascade_predictor, rumor_detector)
    
    return end_to_end_model, shared_encoder, cascade_predictor, rumor_detector

def create_sample_data():
    """
    创建示例数据
    """
    print("📊 创建示例传播数据:")
    print("  传播场景: 推文A→推文B→推文C, 推文A→推文D→推文E")
    print("  时序分割: 前75%为早期图，后25%为预测目标")
    
    # 节点特征 (5个节点, 305维特征)
    num_nodes = 5
    x = torch.randn(num_nodes, 305)  # Word2Vec(300) + 度数特征(5)
    
    # 早期观察到的边 (前75%时间)
    edge_index = torch.tensor([
        [0, 0, 1],  # 源节点: A, A, B  
        [1, 3, 2]   # 目标节点: B, D, C
    ], dtype=torch.long)  # A→B, A→D, B→C
    
    # 待预测的边 (后25%时间)
    pred_edge_index = torch.tensor([
        [3],  # 源节点: D
        [4]   # 目标节点: E  
    ], dtype=torch.long)  # D→E
    
    # 谣言标签 (0:false, 1:true, 2:unverified, 3:non-rumor)
    label = torch.tensor([1], dtype=torch.long)  # 假设这是真谣言
    
    # 创建PyG数据对象
    data = Data(
        x=x,
        edge_index=edge_index,
        pred_edge_index=pred_edge_index,
        label=label
    )
    
    print(f"  ✅ 节点特征: {x.shape}")
    print(f"  ✅ 早期边: {edge_index.shape} {edge_index.tolist()}")
    print(f"  ✅ 预测边: {pred_edge_index.shape} {pred_edge_index.tolist()}")
    print(f"  ✅ 标签: {label.item()} (1=真谣言)")
    
    return data

def demonstrate_forward_pass():
    """
    演示完整的前向传播过程
    """
    print("=" * 80)
    print("🎭 D2模型前向传播演示")
    print("=" * 80)
    
    # 创建数据和模型
    data = create_sample_data()
    model, shared_encoder, cascade_predictor, rumor_detector = create_mock_models()
    
    print(f"\n🏃 开始前向传播...")
    
    # 执行前向传播
    with torch.no_grad():
        rumor_output, pred_edges = model(data)
    
    # 解释结果
    print(f"\n📋 最终结果:")
    print(f"  🔗 链接预测概率: {pred_edges.numpy():.4f}")
    print(f"      → 概率>0.5: {'是' if pred_edges.item() > 0.5 else '否'}")
    print(f"      → 解释: D→E这条边{'很可能' if pred_edges.item() > 0.5 else '不太可能'}在未来出现")
    
    print(f"  🕵️ 谣言分类概率: {rumor_output.numpy()}")
    predicted_class = rumor_output.argmax().item()
    class_names = ['假谣言', '真谣言', '未验证', '非谣言']
    print(f"      → 预测类别: {predicted_class} ({class_names[predicted_class]})")
    print(f"      → 真实类别: {data.label.item()} ({class_names[data.label.item()]})")

def explain_shared_encoder_role():
    """
    详细解释SharedEncoder的作用机制
    """
    print(f"\n🧠 SharedEncoder作用机制详解:")
    print("=" * 50)
    
    print("""
    🎯 为什么SharedEncoder是联合训练的关键？
    
    1. 参数共享机制:
       • CascadePredictor和RumorDetector都使用同一个SharedEncoder
       • 当CascadePredictor更新时，SharedEncoder参数改变
       • 这个改变会直接影响RumorDetector的性能
       • 反之亦然，形成相互影响的闭环
    
    2. 梯度传播路径:
       损失1 (链接预测) → CascadePredictor → SharedEncoder ← RumorDetector ← 损失2 (谣言检测)
                                              ↑
                                         参数同时接收两个梯度
    
    3. 特征空间统一:
       • 两个任务在相同的512维嵌入空间中工作
       • 链接预测学到的"传播模式"特征
       • 谣言检测学到的"语义内容"特征
       • 在SharedEncoder中融合成"传播-语义"联合特征
    
    4. 信息互传机制:
       CascadePredictor → SharedEncoder → RumorDetector:
       "这个传播模式很像谣言的爆发式传播"
       
       RumorDetector → SharedEncoder → CascadePredictor:  
       "这个内容是谣言，传播应该更随机和广泛"
    """)

def demonstrate_training_step():
    """
    演示一个训练步骤
    """
    print(f"\n🏋️ 联合训练步骤演示:")
    print("=" * 50)
    
    # 创建数据和模型
    data = create_sample_data()
    model, shared_encoder, cascade_predictor, rumor_detector = create_mock_models()
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("📝 训练步骤:")
    print("  1. 前向传播")
    rumor_output, pred_edges = model(data)
    
    print("  2. 计算损失")
    # 链接预测损失 (二分类)
    link_target = torch.tensor([1.0])  # 假设D→E确实发生了
    link_loss = F.binary_cross_entropy(pred_edges, link_target)
    
    # 谣言检测损失 (多分类)  
    rumor_loss = F.nll_loss(rumor_output, data.label)
    
    # 总损失
    total_loss = 0.4 * link_loss + 0.6 * rumor_loss
    
    print(f"    🔗 链接预测损失: {link_loss.item():.4f}")
    print(f"    🕵️ 谣言检测损失: {rumor_loss.item():.4f}")  
    print(f"    📊 总损失: {total_loss.item():.4f}")
    
    print("  3. 反向传播")
    optimizer.zero_grad()
    total_loss.backward()
    
    print("  4. 梯度分析")
    for name, param in shared_encoder.named_parameters():
        if param.grad is not None:
            print(f"    {name}: 梯度范数 = {param.grad.norm().item():.6f}")
    
    print("  5. 参数更新")
    optimizer.step()
    
    print("  ✅ 训练步骤完成！SharedEncoder参数已更新，影响两个任务。")

if __name__ == "__main__":
    demonstrate_forward_pass()
    explain_shared_encoder_role()
    demonstrate_training_step()
