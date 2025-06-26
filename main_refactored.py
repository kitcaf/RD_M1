# -*- coding: utf-8 -*-
"""
重构后的主程序文件
调用各个模块完成早期谣言检测的端到端训练和评估
"""

import os
import sys

# 导入自定义模块
from config import IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS
from data_processor import load_and_process_data
from models import create_models
from trainer import cross_validation_training
from utils import setup_device, safe_model_to_device, print_model_info


def main():
    """
    主函数：协调整个训练和评估流程
    """
    print("=" * 60)
    print("早期谣言检测系统 - 重构版本")
    print("=" * 60)
    
    # 设置数据集路径
    dataset_path = "e:/desk/diffuse/diff_true/seqToSeq/code/D2-master/D2-master/data/twitter16"
    
    # 数据处理阶段
    print("\n 开始数据处理...")
    try:
        pyg_dataset = load_and_process_data(dataset_path)
    except Exception as e:
        print(f" 数据处理失败: {e}")
        sys.exit(1)
    
    # 设备设置
    print("\n 设置计算设备...")
    device = setup_device()
    
    # 模型创建
    print("\n 创建模型...")
    
    def model_factory():
        """模型工厂函数，用于在交叉验证中创建新的模型实例"""
        return create_models(IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS)
    
    # 创建一个模型实例用于显示信息
    model, shared_encoder, link_pred_model, rumor_detect_model = model_factory()
    
    # 安全地将模型移动到设备
    model, actual_device = safe_model_to_device(model, device)
    
    # 打印模型信息
    print_model_info(model)
    
    # 开始交叉验证训练
    print(f"\n 开始 {10} 折交叉验证训练...")
    print("=" * 60)
    
    try:
        # 执行交叉验证
        avg_acc, avg_f1 = cross_validation_training(
            model_factory=lambda: create_models(IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS),
            pyg_dataset=pyg_dataset,
            device=actual_device
        )
        
        # 显示最终结果
        print("\n" + "=" * 60)
        print("🎉 训练完成！最终结果:")
        print(f"📊 平均准确率: {avg_acc:.4f}")
        print(f"📈 平均F1分数: {avg_f1:.4f}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
