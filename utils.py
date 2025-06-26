# -*- coding: utf-8 -*-
"""
工具函数模块
包含负采样、设备设置等辅助功能
"""

import torch
from torch_geometric.utils import negative_sampling


def custom_negative_sampling(edge_index, num_nodes, num_neg_samples):
    """
    自定义负采样函数，确保负样本与正样本无重叠
    
    参数:
        edge_index (Tensor): 正样本边索引
        num_nodes (int): 图中节点总数
        num_neg_samples (int): 负样本数量
        
    返回:
        Tensor: 负样本边索引
    """
    # 生成负样本
    negative_edges = negative_sampling(
        edge_index=edge_index, 
        num_nodes=num_nodes, 
        num_neg_samples=num_neg_samples
    )
    
    # 转换为集合以便去重
    pos_edges_set = set([tuple(edge) for edge in edge_index.cpu().numpy().T])
    neg_edges_set = set([tuple(edge) for edge in negative_edges.cpu().numpy().T])
    
    # 确保负样本不与正样本重叠
    neg_edges_set = neg_edges_set - pos_edges_set
    negative_edges = torch.tensor(list(neg_edges_set), dtype=torch.long).T
    
    return negative_edges


def setup_device():
    """
    设置训练设备（GPU或CPU）
    
    返回:
        torch.device: 可用的计算设备
    """
    # 更安全的设备设置
    if torch.cuda.is_available():
        print(f"CUDA可用: {torch.cuda.is_available()}")
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        device = torch.device('cuda:0')  # 明确使用第一个GPU
    else:
        print("CUDA不可用，使用CPU")
        device = torch.device('cpu')
    
    print(f"使用设备: {device}")
    
    # 清理CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return device


def safe_model_to_device(model, device):
    """
    安全地将模型移动到指定设备
    
    参数:
        model (nn.Module): 要移动的模型
        device (torch.device): 目标设备
        
    返回:
        tuple: (模型, 实际使用的设备)
    """
    try:
        model = model.to(device)
        print("模型成功移动到设备")
        return model, device
    except RuntimeError as e:
        print(f"将模型移动到设备时出错: {e}")
        print("回退到CPU")
        fallback_device = torch.device('cpu')
        model = model.to(fallback_device)
        return model, fallback_device


def print_model_info(model):
    """
    打印模型信息
    
    参数:
        model (nn.Module): 要分析的模型
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"模型结构:")
    print(model)
