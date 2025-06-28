# -*- coding: utf-8 -*-
"""
模型定义模块
包含共享编码器、级联预测器、谣言检测器和端到端模型

就3个模型 + 一个共享编码器

SharedEncoder、
CascadePredictor、
RumorDetector、
EndToEndModel

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, SAGPooling, global_mean_pool, BatchNorm
from config import DROPOUT, HIDDEN_CHANNELS


class SharedEncoder(torch.nn.Module):
    """
    共享编码器
    使用GraphSAGE进行图神经网络编码，包含批归一化、Dropout和跳跃连接
    GraphSAGE是一种流行的GNN架构，这里也可以替换成GCN、GAT或其他GNN架构
    """
    
    def __init__(self, in_channels, hidden_channels):
        """
        初始化共享编码器
        
        参数:
            in_channels (int): 输入特征维度
            hidden_channels (int): 隐藏层特征维度
        """
        super(SharedEncoder, self).__init__()
        
        # 第一层GraphSAGE卷积
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.dropout1 = torch.nn.Dropout(p=DROPOUT)
        
        # 第二层GraphSAGE卷积
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.batch_norm2 = BatchNorm(hidden_channels)
        self.dropout2 = torch.nn.Dropout(p=DROPOUT)
        
        # 多层感知机用于非线性变换
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )

    def forward(self, x, edge_index):
        """
        前向传播
        
        参数:
            x (Tensor): 节点特征矩阵
            edge_index (Tensor): 边索引
            
        返回:
            Tensor: 编码后的节点表示
        """
        # 第一层带跳跃连接
        x_1 = self.conv1(x, edge_index)
        x_1 = self.batch_norm1(x_1).relu()
        x_1 = self.dropout1(x_1)

        # 第二层带跳跃连接
        x_2 = self.conv2(x_1, edge_index)
        x_2 = self.batch_norm2(x_2).relu()
        x_2 = self.dropout2(x_2)

        # 添加跳跃连接和MLP非线性变换
        x_out = x_1 + self.mlp(x_2)
        return x_out


class CascadePredictor(torch.nn.Module):
    """
    级联预测器（预测未来的传播情况 --- 基于有限观察预测未来发展）
    用于预测级联中的链接，使用共享编码器和多头注意力机制

    注意：是边分类器(Link Classifier)，不是图生成器。基于二分类的监督学习
    
    输入：早期图数据，监督数据-候选数据（正样本-未来的边、负样本-随机的边）

    输出：候选数据的所有边的概率

    """
    
    def __init__(self, shared_encoder):
        """
        初始化级联预测器
        
        参数:
            shared_encoder (SharedEncoder): 共享编码器实例
        """
        super(CascadePredictor, self).__init__()
        self.shared_encoder = shared_encoder
        
        # 多头注意力机制
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=HIDDEN_CHANNELS, 
            num_heads=4, 
            batch_first=True
        )

    def encode(self, x, edge_index):
        """
        基于早期图(edge_index)编码节点特征
        
        参数:
            x (Tensor): 节点特征
            edge_index (Tensor): 边索引
            
        返回:
            Tensor: 编码后的节点嵌入
        """
        # 基于早期图使用GraphSAGE进行编码
        return self.shared_encoder(x, edge_index)

    def decode(self, z, edge_index):
        """
        解码链接概率
        
        参数:
            z (Tensor): 节点嵌入
            edge_index (Tensor): 边索引
            
        返回:
            Tensor: 链接存在的概率
        """
        src, dst = edge_index
        
        # 获取源节点和目标节点的嵌入
        z_src = z[src].unsqueeze(1)  # shape (num_edges, 1, hidden_channels)
        z_dst = z[dst].unsqueeze(1)  # shape (num_edges, 1, hidden_channels)
        
        # 组合源节点和目标节点嵌入 # 通过多头注意力机制融合源节点和目标节点特征
        z_combined = torch.cat([z_src, z_dst], dim=1)  # shape (num_edges, 2, hidden_channels)
        attn_output, _ = self.attention(z_combined, z_combined, z_combined)
        
        # 输出sigmoid概率，表示该边存在的可能性
        return torch.sigmoid(attn_output[:, 0, :].sum(dim=-1))

    def forward(self, x, edge_index, edge_index_pred):
        """
        前向传播
        
        参数:
            x (Tensor): 节点特征
            edge_index (Tensor): 当前图的边索引
            edge_index_pred (Tensor): 待预测的边索引
            
        返回:
            Tensor: 预测链接的概率
        """
        # 1. 基于早期图(edge_index)编码节点特征
        z = self.encode(x, edge_index) # 使用SharedEncoder处理early_graph

        # 2. 对候选边集合(edge_index_pred)进行预测
        return self.decode(z, edge_index_pred)  # 输出每条候选边的存在概率


class RumorDetector(torch.nn.Module):
    """
    谣言检测器
    使用共享编码器、自适应图池化和全连接层进行谣言分类
    """
    
    def __init__(self, shared_encoder, out_channels):
        """
        初始化谣言检测器
        
        参数:
            shared_encoder (SharedEncoder): 共享编码器实例
            out_channels (int): 输出类别数
        """
        super(RumorDetector, self).__init__()
        self.shared_encoder = shared_encoder
        
        # 自适应图池化
        self.pool = SAGPooling(HIDDEN_CHANNELS, ratio=0.5)
        self.batch_norm = BatchNorm(HIDDEN_CHANNELS)
        
        # 分类器
        self.fc = torch.nn.Linear(HIDDEN_CHANNELS, out_channels)

    def forward(self, x, edge_index, batch):
        """
        前向传播
        
        参数:
            x (Tensor): 节点特征
            edge_index (Tensor): 边索引
            batch (Tensor): 批次信息
            
        返回:
            Tensor: 谣言分类的对数概率
        """
        # 使用共享编码器
        x = self.shared_encoder(x, edge_index)
        
        # 自适应图池化
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, batch=batch)
        x = self.batch_norm(x)
        
        # 全局平均池化
        x = global_mean_pool(x, batch)
        
        # 分类输出
        return F.log_softmax(self.fc(x), dim=-1)


class EndToEndModel(torch.nn.Module):
    """
    端到端模型
    结合链接预测和谣言检测的完整模型
    """
    
    def __init__(self, shared_encoder, link_pred_model, rumor_detect_model):
        """
        初始化端到端模型
        
        参数:
            shared_encoder (SharedEncoder): 共享编码器实例
            link_pred_model (CascadePredictor): 链接预测模型
            rumor_detect_model (RumorDetector): 谣言检测模型
        """
        super(EndToEndModel, self).__init__()
        self.shared_encoder = shared_encoder
        self.link_pred_model = link_pred_model
        self.rumor_detect_model = rumor_detect_model

    def forward(self, data):
        """
        前向传播
        
        参数:
            data (Data): PyTorch Geometric数据对象
            
        返回:
            tuple: (谣言检测输出, 链接预测输出)
        """
        # 链接预测任务（输出所有候选边的概率）
        pred_edges = self.link_pred_model(
            data.x, data.edge_index, data.pred_edge_index
        )

        # 只保留预测概率大于0.5的边 基于概率阈值(0.5)筛选边，
        # 早期数据 + 预测边 == 构建重构图
        reconstructed_edge_index = torch.cat([
            data.edge_index, 
            data.pred_edge_index[:, pred_edges > 0.5]
        ], dim=1)

        # 将重构图传给 RumorDetector，进行谣言预测
        rumor_out = self.rumor_detect_model(
            data.x, reconstructed_edge_index, data.batch
        )

        return rumor_out, pred_edges


def create_models(in_channels, hidden_channels, out_channels):
    """
    创建模型的工厂函数
    
    参数:
        in_channels (int): 输入特征维度
        hidden_channels (int): 隐藏层特征维度
        out_channels (int): 输出类别数
        
    返回:
        tuple: (端到端模型, 共享编码器, 链接预测模型, 谣言检测模型)
    """
    # 初始化共享编码器
    shared_encoder = SharedEncoder(in_channels, hidden_channels)

    # 初始化链接预测模型和谣言检测模型 --- 
    link_pred_model = CascadePredictor(shared_encoder)
    rumor_detect_model = RumorDetector(shared_encoder, out_channels)

    # 初始化端到端模型
    model = EndToEndModel(shared_encoder, link_pred_model, rumor_detect_model)
    
    return model, shared_encoder, link_pred_model, rumor_detect_model
