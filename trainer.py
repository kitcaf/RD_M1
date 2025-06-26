# -*- coding: utf-8 -*-
"""
训练模块
包含训练函数、评估函数和交叉验证逻辑
"""

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader

from config import (
    LEARNING_RATE, WEIGHT_DECAY, GAMMA, MAX_GRAD_NORM, 
    LOSS_RATIO_LINK_PRED, LOSS_RATIO_RUMOR, K_FOLDS, 
    NUM_EPOCHS, BATCH_SIZE, PATIENCE
)
from utils import custom_negative_sampling


def train_epoch(model, data_loader, optimizer, scheduler, device):
    """
    训练一个epoch
    
    参数:
        model (nn.Module): 要训练的模型
        data_loader (DataLoader): 训练数据加载器
        optimizer (Optimizer): 优化器
        scheduler (lr_scheduler): 学习率调度器
        device (torch.device): 计算设备
        
    返回:
        dict: 包含损失值、准确率和F1分数的字典
    """
    model.train()
    total_loss, all_true_labels, all_pred_labels = 0, [], []
    all_link_true, all_link_pred = [], []

    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()

        # 生成负样本用于链接预测
        neg_edge_index = custom_negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.pred_edge_index.shape[1]
        ).to(device)
        
        # 创建边标签（正样本为1，负样本为0）
        edge_labels = torch.cat([
            torch.ones(data.pred_edge_index.shape[1]), 
            torch.zeros(neg_edge_index.shape[1])
        ]).to(device)
        
        # 合并正负样本边
        data.pred_edge_index = torch.cat([data.pred_edge_index, neg_edge_index], dim=1)
        data.edge_index = data.edge_index.long()
        data.pred_edge_index = data.pred_edge_index.long()

        # 前向传播
        out, pred_edges = model(data)
        
        # 计算损失
        link_loss = nn.BCELoss()(pred_edges, edge_labels)  # 链接预测损失
        rumor_loss = nn.CrossEntropyLoss()(out, data.label)  # 谣言检测损失
        
        # 调整损失比例，更加重视链接预测
        loss = LOSS_RATIO_LINK_PRED * link_loss + LOSS_RATIO_RUMOR * rumor_loss
        
        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        
        optimizer.step()
        scheduler.step(loss)
        total_loss += loss.item()

        # 收集预测结果用于评估
        _, predicted_labels = out.max(dim=1)
        all_true_labels.extend(data.label.cpu().numpy())
        all_pred_labels.extend(predicted_labels.cpu().numpy())
        all_link_true.extend(edge_labels.cpu().numpy())
        all_link_pred.extend((pred_edges > 0.5).cpu().numpy())

        # 清理内存
        del data, out, pred_edges, edge_labels, neg_edge_index
        torch.cuda.empty_cache()

    # 计算评估指标
    rumor_acc = accuracy_score(all_true_labels, all_pred_labels)
    rumor_f1 = f1_score(all_true_labels, all_pred_labels, average="weighted")

    return {
        "loss": total_loss / len(data_loader), 
        "acc": rumor_acc, 
        "f1": rumor_f1
    }


def evaluate_model(model, data_loader, device):
    """
    评估模型性能
    
    参数:
        model (nn.Module): 要评估的模型
        data_loader (DataLoader): 测试数据加载器
        device (torch.device): 计算设备
        
    返回:
        dict: 包含准确率和F1分数的字典
    """
    model.eval()
    all_true_labels, all_pred_labels = [], []
    all_link_true, all_link_pred = [], []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)

            # 为评估生成负样本
            neg_edge_index = custom_negative_sampling(
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=data.pred_edge_index.shape[1]
            ).to(device)
            
            # 合并正负样本
            data.pred_edge_index = torch.cat([data.pred_edge_index, neg_edge_index], dim=1)
            edge_labels = torch.cat([
                torch.ones(data.pred_edge_index.shape[1] // 2), 
                torch.zeros(neg_edge_index.shape[1])
            ]).to(device)

            # 前向传播
            out, pred_edges = model(data)
            
            # 收集预测结果
            _, predicted_labels = out.max(dim=1)
            all_true_labels.extend(data.label.cpu().numpy())
            all_pred_labels.extend(predicted_labels.cpu().numpy())
            all_link_true.extend(edge_labels.cpu().numpy())
            all_link_pred.extend((pred_edges > 0.5).cpu().numpy())

            # 清理内存
            del data, out, pred_edges, edge_labels, neg_edge_index
            torch.cuda.empty_cache()

    # 计算评估指标
    rumor_acc = accuracy_score(all_true_labels, all_pred_labels)
    rumor_f1 = f1_score(all_true_labels, all_pred_labels, average="weighted")

    return {"acc": rumor_acc, "f1": rumor_f1}


def train_single_fold(model, train_dataset, test_dataset, device):
    """
    训练单个折的数据
    
    参数:
        model (nn.Module): 要训练的模型
        train_dataset (list): 训练数据集
        test_dataset (list): 测试数据集
        device (torch.device): 计算设备
        
    返回:
        dict: 最终的测试结果
    """
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=GAMMA
    )

    # 早停相关变量
    best_f1 = 0
    patience_counter = 0

    # 训练循环
    for epoch in range(1, NUM_EPOCHS + 1):
        # 训练一个epoch
        train_results = train_epoch(model, train_loader, optimizer, scheduler, device)
        
        print(f"Epoch {epoch} | 训练损失: {train_results['loss']:.4f}")
        print(f"训练准确率: {train_results['acc']:.4f}, 训练F1: {train_results['f1']:.4f}")
        
        # 评估模型
        test_results = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch} | 测试准确率: {test_results['acc']:.4f}, 测试F1: {test_results['f1']:.4f}")

        # 早停检查
        if test_results['f1'] > best_f1:
            best_f1 = test_results['f1']
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"早停于第 {epoch} 轮")
            break

    return test_results


def cross_validation_training(model_factory, pyg_dataset, device):
    """
    执行K折交叉验证训练
    
    参数:
        model_factory (callable): 创建模型的工厂函数
        pyg_dataset (list): 完整的数据集
        device (torch.device): 计算设备
        
    返回:
        tuple: (平均准确率, 平均F1分数)
    """
    # 设置K折交叉验证
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    avg_acc, avg_f1 = 0, 0

    # 对每一折进行训练和评估
    for fold, (train_idx, test_idx) in enumerate(kf.split(pyg_dataset)):
        print(f"\n=== 第 {fold + 1}/{K_FOLDS} 折 ===")
        
        # 分割数据
        train_dataset = [pyg_dataset[i] for i in train_idx]
        test_dataset = [pyg_dataset[i] for i in test_idx]
        
        # 创建新的模型实例（避免折间干扰）
        model, _, _, _ = model_factory()
        model = model.to(device)
        
        # 训练当前折
        test_results = train_single_fold(model, train_dataset, test_dataset, device)
        
        # 累计结果
        avg_acc += test_results['acc']
        avg_f1 += test_results['f1']
        
        print(f"第 {fold + 1} 折结果 - 准确率: {test_results['acc']:.4f}, F1: {test_results['f1']:.4f}")

    # 计算平均值
    avg_acc /= K_FOLDS
    avg_f1 /= K_FOLDS
    
    print(f"\n=== 交叉验证最终结果 ===")
    print(f"平均测试准确率: {avg_acc:.4f}, 平均测试F1: {avg_f1:.4f}")
    
    return avg_acc, avg_f1
