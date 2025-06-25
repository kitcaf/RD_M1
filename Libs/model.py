import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import (GATConv, GCNConv, SAGEConv, global_max_pool,
                                global_mean_pool)



class CascadeGNNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, gnn_type: str = 'GCN'):
        """
        初始化一个指定类型的GNN层（GCN、GAT或SAGE）。
        
        参数:
            in_channels (int): 输入特征维度。
            out_channels (int): 输出特征维度。
            gnn_type (str): GNN层的类型（'GCN'、'GAT'或'SAGE'）。
        """
        super(CascadeGNNLayer, self).__init__()
        if gnn_type == 'GCN':
            self.conv = GCNConv(in_channels, out_channels)  # 图卷积网络层
        elif gnn_type == 'GAT':
            self.conv = GATConv(in_channels, out_channels, heads=1)  # 图注意力网络层
        elif gnn_type == 'SAGE':
            self.conv = SAGEConv(in_channels, out_channels)  # GraphSAGE卷积层
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")  # 不支持的GNN类型

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        GNN层的前向传播。
        
        参数:
            x (Tensor): 节点特征矩阵。
            edge_index (Tensor): 边索引张量。
            
        返回:
            Tensor: GNN层处理后的输出特征矩阵。
        """
        return self.conv(x, edge_index)  # 应用图卷积操作


class CascadePredictionModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        """
        级联预测模型，用于生成缺失节点的嵌入。
        
        参数:
            in_channels (int): 输入特征维度。
            hidden_channels (int): 隐藏层特征维度。
        """
        super(CascadePredictionModel, self).__init__()
        self.gnn1 = GCNConv(in_channels, hidden_channels)  # 第一个GCN层
        self.gnn2 = GCNConv(hidden_channels, hidden_channels)  # 第二个GCN层
        self.output_layer = Linear(hidden_channels, in_channels)  # 输出层，将隐藏特征映射回原始维度

    def forward(self, data, num_missing_nodes: int) -> torch.Tensor:
        """
        前向传播，预测级联中缺失节点的嵌入。
        
        参数:
            data (Data): 包含节点特征和边索引的PyG数据对象。
            num_missing_nodes (int): 需要预测的缺失节点数量。
            
        返回:
            Tensor: 缺失节点的预测嵌入。
        """
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gnn1(x, edge_index))  # 第一层GCN后应用ReLU激活
        x = F.relu(self.gnn2(x, edge_index))  # 第二层GCN后应用ReLU激活

        # 为缺失节点生成嵌入
        predicted_embeddings = self.output_layer(
            torch.randn(num_missing_nodes, x.size(1)).to(x.device)  # 随机初始化缺失节点特征
        )
        predicted_embeddings = torch.concat([x, predicted_embeddings], dim=0)  # 连接原始节点和预测节点的嵌入
        return predicted_embeddings


class RumorDetectionModel(nn.Module):
    def __init__(self, num_features: int, hidden_channels: int, num_classes: int, gnn_type: str = 'GCN', pooling: str = 'mean'):
        """
        级联中的谣言检测模型，结合多个GNN层和池化操作。
        
        参数:
            num_features (int): 每个节点的输入特征数量。
            hidden_channels (int): 隐藏层维度。
            num_classes (int): 输出类别数量。
            gnn_type (str): 使用的GNN层类型（'GCN'、'GAT'或'SAGE'）。
            pooling (str): 池化方法（'mean'或'max'）。
        """
        super(RumorDetectionModel, self).__init__()
        self.gnn1 = CascadeGNNLayer(num_features, hidden_channels, gnn_type)  # 第一个GNN层
        self.gnn2 = CascadeGNNLayer(num_features, hidden_channels, gnn_type)  # 第二个GNN层
        self.fc_fusion = nn.Linear(4 * hidden_channels, hidden_channels)  # 融合层，将不同关系的表示融合
        self.pooling = pooling
        self.fc = nn.Sequential(  # 分类器
            nn.Linear(hidden_channels, hidden_channels),  # 全连接层
            nn.ReLU(),  # ReLU激活函数
            nn.Dropout(0.5),  # Dropout正则化
            nn.Linear(hidden_channels, num_classes)  # 输出层
        )

    def forward(self, data) -> torch.Tensor:
        """
        谣言检测模型的前向传播。
        
        参数:
            data (Data): 包含节点特征和边索引的PyG数据对象。
            
        返回:
            Tensor: 谣言检测的输出logits。
        """
        x, edge_index = data.x, data.edge_index
        x_s = self.gnn1(x, edge_index)  # 第一个GNN层处理
        x_d = self.gnn2(x, edge_index)  # 第二个GNN层处理

        # 融合不同关系的表示
        x_interaction = torch.cat([x_s, x_d, x_s * x_d, x_s - x_d], dim=1)  # 连接不同的交互特征
        x_fused = self.fc_fusion(x_interaction)  # 通过全连接层融合

        # 全局池化
        if self.pooling == 'mean':
            x_pooled = global_mean_pool(x_fused, data.batch)  # 平均池化
        elif self.pooling == 'max':
            x_pooled = global_max_pool(x_fused, data.batch)  # 最大池化
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")  # 不支持的池化类型

        return self.fc(x_pooled)  # 通过分类器得到最终输出


class SequentialEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        基于LSTM的序列编码器，用于捕获序列信息。
        
        参数:
            input_dim (int): 输入特征维度。
            hidden_dim (int): LSTM中隐藏状态的维度。
        """
        super(SequentialEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)  # LSTM层，batch_first=True表示输入形状为(batch, seq, feature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        序列编码的前向传播。
        
        参数:
            x (Tensor): 输入张量，形状为(batch_size, seq_length, input_dim)。
            
        返回:
            Tensor: 序列的编码表示。
        """
        _, (h_n, _) = self.lstm(x)  # LSTM返回输出和隐藏状态
        return h_n[-1]  # 返回最后一层的隐藏状态


class FullRumorDetectionPipeline(nn.Module):
    def __init__(self, num_features: int, hidden_channels: int, num_classes: int, gnn_type: str = 'GCN', pooling: str = 'mean'):
        """
        完整的谣言检测流水线，包括级联预测、谣言检测和序列编码。
        
        参数:
            num_features (int): 每个节点的输入特征维度。
            hidden_channels (int): 隐藏层维度。
            num_classes (int): 谣言检测的输出类别数量。
            gnn_type (str): GNN层类型（'GCN'、'GAT'或'SAGE'）。
            pooling (str): 池化方法（'mean'或'max'）。
        """
        super(FullRumorDetectionPipeline, self).__init__()
        self.prediction_model = CascadePredictionModel(
            num_features, hidden_channels)  # 级联预测模型
        self.detection_model = RumorDetectionModel(
            hidden_channels, hidden_channels, num_classes, gnn_type, pooling)  # 谣言检测模型
        self.seq_encoder = SequentialEncoder(hidden_channels, hidden_channels)  # 序列编码器

    def forward(self, early_data, full_data) -> torch.Tensor:
        """
        完整谣言检测流水线的前向传播。
        
        参数:
            early_data (Data): 表示早期级联数据的PyG数据对象。
            full_data (Data): 表示完整级联数据的PyG数据对象。
            
        返回:
            Tensor: 编码后的序列表示。
        """
        # 步骤1: 级联预测
        num_missing_nodes = full_data.x.size(0) - early_data.x.size(0)  # 计算缺失节点数量
        predicted_features = self.prediction_model(
            early_data, num_missing_nodes)  # 预测缺失节点的特征

        # 步骤2: 连接早期级联和预测嵌入
        full_data.x = torch.cat([early_data.x, predicted_features], dim=0)  # 组合特征
        detection_output = self.detection_model(full_data)  # 进行谣言检测

        # 步骤3: 序列编码
        sequence_rep = self.seq_encoder(detection_output)  # 对检测输出进行序列编码

        return sequence_rep  # 返回最终表示
