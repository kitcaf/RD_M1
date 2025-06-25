import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch_geometric.utils import negative_sampling
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

def combined_loss(predicted_features, actual_features, rumor_output, rumor_labels, alpha=0.5):
    """
    计算级联预测和谣言检测任务的组合损失。
    
    参数:
        predicted_features (Tensor): 级联中缺失节点的预测特征。
        actual_features (Tensor): 缺失节点的真实特征。
        rumor_output (Tensor): 谣言检测的模型输出logits。
        rumor_labels (Tensor): 谣言检测的真实标签。
        alpha (float): 平衡预测和谣言检测损失的权重。
        
    返回:
        Tensor: 加权组合损失。
    """
    prediction_loss = F.mse_loss(predicted_features, actual_features)  # 预测特征的均方误差损失
    rumor_loss = F.cross_entropy(rumor_output, rumor_labels)  # 谣言检测的交叉熵损失
    return alpha * prediction_loss + (1 - alpha) * rumor_loss  # 组合损失

def train_model(prediction_model, detection_model, early_loader, full_loader, optimizer, epochs=10, alpha=0.5, patience=5, clip_value=1.0):
    """
    训练级联预测和谣言检测模型，使用早停策略。
    
    参数:
        prediction_model (nn.Module): 预测缺失级联节点的模型。
        detection_model (nn.Module): 在级联中检测谣言的模型。
        early_loader (DataLoader): 提供早期级联数据的DataLoader。
        full_loader (DataLoader): 提供完整级联数据的DataLoader。
        optimizer (Optimizer): 模型训练的优化器。
        epochs (int): 训练轮数。
        alpha (float): 组合损失的权重。
        patience (int): 早停的耐心值。
        clip_value (float): 梯度裁剪的最大值。
    """
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)  # 余弦退火学习率调度器
    scaler = GradScaler('cuda')  # 梯度缩放器，用于混合精度训练
    best_f1 = 0
    patience_counter = 0
    best_model_weights = None

    for epoch in range(epochs):
        prediction_model.train()  # 设置预测模型为训练模式
        detection_model.train()  # 设置检测模型为训练模式
        total_loss = 0
        all_preds, all_labels = [], []

        for early_data, full_data in zip(early_loader, full_loader):
            optimizer.zero_grad()  # 清除梯度

            with autocast('cuda'):  # 使用自动混合精度
                # 级联预测步骤: 预测节点数就是原序列节点 - 早期序列节点
                num_missing_nodes = full_data.x.size(0) - early_data.x.size(0)  # 计算缺失节点数
                predicted_features = prediction_model(early_data, num_missing_nodes)  # 预测缺失节点特征
                full_embeddings = torch.cat([early_data.x, predicted_features], dim=0)  # 连接早期特征和预测特征

                # 谣言检测步骤
                full_data.x = full_embeddings  # 更新完整数据的特征
                rumor_output = detection_model(full_data)  # 进行谣言检测

                # 计算组合损失（级联预测模型 + 谣言预测模型组合loss) 端到端训练模型
                loss = combined_loss(predicted_features, full_data.x[-num_missing_nodes:], rumor_output, full_data.y, alpha)
            
            # 使用梯度缩放进行反向传播
            scaler.scale(loss).backward()  # 缩放损失并反向传播
            torch.nn.utils.clip_grad_norm_(list(prediction_model.parameters()) + list(detection_model.parameters()), clip_value)  # 梯度裁剪
            scaler.step(optimizer)  # 优化器步进
            scaler.update()  # 更新缩放因子
            total_loss += loss.item()  # 累加损失

            # 收集预测和标签
            all_preds.extend(rumor_output.argmax(dim=1).detach().cpu().numpy())  # 获取预测类别
            all_labels.extend(full_data.y.cpu().numpy())  # 获取真实标签

        scheduler.step()  # 更新学习率
        
        # 计算轮次指标
        f1 = f1_score(all_labels, all_preds, average='micro')  # 计算F1分数
        accuracy = accuracy_score(all_labels, all_preds)  # 计算准确率
        precision = precision_score(all_labels, all_preds, average='micro')  # 计算精确率
        recall = recall_score(all_labels, all_preds, average='micro')  # 计算召回率

        # 记录和早停
        print(f"\nEpoch [{epoch+1}/{epochs}]\nLoss: {total_loss:.4f}\n"
              f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}\n"
              f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n" + "="*50)

        if f1 > best_f1:  # 如果当前F1分数更好
            best_f1 = f1  # 更新最佳F1分数
            patience_counter = 0  # 重置耐心计数器
            best_model_weights = {
                'prediction_model': prediction_model.state_dict(),
                'detection_model': detection_model.state_dict()
            }  # 保存最佳模型权重
            print("New best model saved.")
        else:
            patience_counter += 1  # 增加耐心计数器
            if patience_counter >= patience:  # 如果达到耐心限制
                print("Early stopping: Validation metrics did not improve.")
                break  # 早停

    if best_model_weights:  # 如果有最佳模型权重
        prediction_model.load_state_dict(best_model_weights['prediction_model'])  # 加载最佳预测模型
        detection_model.load_state_dict(best_model_weights['detection_model'])  # 加载最佳检测模型
        print("Best model weights restored.")

def test_model(prediction_model, detection_model, early_loader, full_loader, alpha=0.5):
    """
    在测试数据上评估训练好的模型并报告指标。
    
    参数:
        prediction_model (nn.Module): 级联预测模型。
        detection_model (nn.Module): 谣言检测模型。
        early_loader (DataLoader): 早期级联测试数据的DataLoader。
        full_loader (DataLoader): 完整级联测试数据的DataLoader。
        alpha (float): 评估期间组合损失函数的权重。
    """
    prediction_model.eval()  # 设置预测模型为评估模式
    detection_model.eval()  # 设置检测模型为评估模式
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():  # 不计算梯度
        for early_data, full_data in zip(early_loader, full_loader):
            with autocast('cuda'):  # 使用自动混合精度
                num_missing_nodes = full_data.x.size(0) - early_data.x.size(0)  # 计算缺失节点数
                predicted_features = prediction_model(early_data, num_missing_nodes)  # 预测缺失节点特征
                full_embeddings = torch.cat([early_data.x, predicted_features], dim=0)  # 连接特征
                
                full_data.x = full_embeddings  # 更新完整数据的特征
                rumor_output = detection_model(full_data)  # 进行谣言检测

                loss = combined_loss(predicted_features, full_data.x[-num_missing_nodes:], rumor_output, full_data.y, alpha)  # 计算损失
                total_loss += loss.item()  # 累加损失

                all_preds.extend(rumor_output.argmax(dim=1).cpu().numpy())  # 获取预测类别
                all_labels.extend(full_data.y.cpu().numpy())  # 获取真实标签

    # 计算测试指标
    f1 = f1_score(all_labels, all_preds, average='micro')  # 计算F1分数
    accuracy = accuracy_score(all_labels, all_preds)  # 计算准确率
    precision = precision_score(all_labels, all_preds, average='micro')  # 计算精确率
    recall = recall_score(all_labels, all_preds, average='micro')  # 计算召回率

    print("\nTest Results\n" + "="*50 + f"\nLoss: {total_loss:.4f}\n"
          f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}\n" + "="*50)
