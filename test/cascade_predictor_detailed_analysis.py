"""
CascadePredictor 详细原理分析脚本

本脚本深入解析 CascadePredictor 的真实工作机制，解答以下关键问题：
1. CascadePredictor 的输入是什么？
2. 它如何处理前75%和后25%的传播数据？
3. 它真正预测的是什么？
4. 监督信号从哪里来？
5. 训练过程是如何进行的？

重点分析：CascadePredictor 并不是生成完整的未来传播图，
而是对给定的候选边进行二分类判断（是否会在未来出现）
"""
import numpy as np
import networkx as nx
import torch
from datetime import datetime

def demonstrate_cascade_predictor_principle():
    """展示CascadePredictor的核心原理"""
    
    print("=" * 80)
    print("CascadePredictor 详细原理分析")
    print("=" * 80)
    
    # 1. 模拟一个完整的传播图
    print("\n1. 模拟一个完整传播级联的时间演化")
    print("-" * 50)
    
    # 创建一个带时间戳的传播图
    full_cascade = nx.DiGraph()
    
    # 添加节点和边（带时间戳）
    edges_with_time = [
        ('root', 'user1', 1.0),    # 时间=1
        ('root', 'user2', 2.0),    # 时间=2
        ('user1', 'user3', 3.0),   # 时间=3
        ('user2', 'user4', 4.0),   # 时间=4
        ('user1', 'user5', 5.0),   # 时间=5 (这里开始是后25%时间)
        ('user3', 'user6', 6.0),   # 时间=6
        ('user4', 'user7', 7.0),   # 时间=7
        ('user5', 'user8', 8.0),   # 时间=8
    ]
    
    for src, dst, time in edges_with_time:
        full_cascade.add_edge(src, dst, time=time)
    
    print(f"完整传播图: {len(full_cascade.edges())} 条边")
    print(f"时间范围: 1.0 到 8.0")
    
    # 2. 按时间分割传播图（前75% vs 后25%）
    print("\n2. 时间分割：前75% vs 后25%")
    print("-" * 50)
    
    max_time = 8.0
    time_threshold = max_time * 0.75  # 75%时间点 = 6.0
    
    # 前75%时间的边（early_graph）
    early_edges = [(src, dst) for src, dst, data in full_cascade.edges(data=True) 
                   if data['time'] <= time_threshold]
    
    # 后25%时间的边（prediction_graph）
    future_edges = [(src, dst) for src, dst, data in full_cascade.edges(data=True) 
                    if data['time'] > time_threshold]
    
    print(f"前75%时间边 (≤{time_threshold}): {early_edges}")
    print(f"后25%时间边 (>{time_threshold}): {future_edges}")
    
    # 3. CascadePredictor的真实输入
    print("\n3. CascadePredictor的真实输入分析")
    print("-" * 50)
    
    print("输入1: 前75%传播图的节点特征和边")
    print(f"   - 节点特征: 每个节点的Word2Vec嵌入 (300维)")
    print(f"   - 早期边: {early_edges}")
    
    print("\n输入2: 待预测的候选边")
    print("   - 正样本: 后25%时间的真实边")
    print(f"     {future_edges}")
    print("   - 负样本: 随机采样的不存在边")
    
    # 生成负样本示例
    all_nodes = list(full_cascade.nodes())
    candidate_negative_edges = []
    for i, src in enumerate(all_nodes):
        for j, dst in enumerate(all_nodes):
            if i != j and not full_cascade.has_edge(src, dst):
                candidate_negative_edges.append((src, dst))
    
    # 随机选择一些负样本
    np.random.seed(42)
    negative_samples = np.random.choice(len(candidate_negative_edges), 
                                      size=len(future_edges), replace=False)
    sampled_negative_edges = [candidate_negative_edges[i] for i in negative_samples]
    
    print(f"     {sampled_negative_edges[:len(future_edges)]}")
    
    # 4. CascadePredictor的输出
    print("\n4. CascadePredictor的输出")
    print("-" * 50)
    
    all_candidate_edges = future_edges + sampled_negative_edges[:len(future_edges)]
    print("对每条候选边输出一个概率值 (0-1之间):")
    
    # 模拟预测概率
    np.random.seed(42)
    predicted_probs = np.random.rand(len(all_candidate_edges))
    
    for i, (edge, prob) in enumerate(zip(all_candidate_edges, predicted_probs)):
        edge_type = "正样本" if edge in future_edges else "负样本"
        print(f"   边 {edge}: 概率={prob:.3f} ({edge_type})")
    
    # 5. 监督信号和训练过程
    print("\n5. 监督信号和训练过程")
    print("-" * 50)
    
    print("真实标签:")
    true_labels = [1 if edge in future_edges else 0 for edge in all_candidate_edges]
    for i, (edge, label, prob) in enumerate(zip(all_candidate_edges, true_labels, predicted_probs)):
        print(f"   边 {edge}: 真实标签={label}, 预测概率={prob:.3f}")
    
    # 计算BCE损失
    epsilon = 1e-8
    bce_loss = -np.mean([
        label * np.log(prob + epsilon) + (1 - label) * np.log(1 - prob + epsilon)
        for label, prob in zip(true_labels, predicted_probs)
    ])
    
    print(f"\nBCE损失: {bce_loss:.4f}")
    
    # 6. 关键洞察
    print("\n6. 关键洞察和误区澄清")
    print("-" * 50)
    
    print("✓ CascadePredictor的本质:")
    print("   - 这是一个二分类器，不是图生成器")
    print("   - 输入: 前75%传播图 + 候选边列表")
    print("   - 输出: 每条候选边在后25%时间出现的概率")
    
    print("\n✗ 常见误解:")
    print("   - 误解1: 它会生成完整的未来传播图")
    print("   - 误解2: 它会自动发现所有可能的未来边")
    print("   - 误解3: 它直接输出一个新的图结构")
    
    print("\n✓ 实际工作流程:")
    print("   1. 给定前75%时间的传播图")
    print("   2. 提供候选边列表（真实未来边 + 负样本边）")
    print("   3. 对每条候选边预测其出现概率")
    print("   4. 使用BCE损失进行监督学习")
    
    print("\n✓ 训练数据构造:")
    print("   - 正样本: prediction_graph中的真实边（后25%时间）")
    print("   - 负样本: 通过负采样生成的不存在边")
    print("   - 标签: 正样本=1, 负样本=0")
    
    return {
        'early_edges': early_edges,
        'future_edges': future_edges,
        'negative_edges': sampled_negative_edges[:len(future_edges)],
        'predictions': predicted_probs,
        'labels': true_labels,
        'bce_loss': bce_loss
    }

def analyze_model_architecture():
    """分析模型架构的关键组件"""
    
    print("\n" + "=" * 80)
    print("模型架构关键组件分析")
    print("=" * 80)
    
    print("\n1. SharedEncoder (共享编码器)")
    print("-" * 50)
    print("   - 类型: GraphSAGE图神经网络")
    print("   - 功能: 将节点特征和图结构编码为节点嵌入")
    print("   - 输入: 节点特征(300维) + 边索引")
    print("   - 输出: 节点嵌入(128维)")
    print("   - 共享: 同时被CascadePredictor和RumorDetector使用")
    
    print("\n2. CascadePredictor.decode() 方法")
    print("-" * 50)
    print("   - 输入: 节点嵌入 + 候选边索引")
    print("   - 处理: 获取源节点和目标节点的嵌入")
    print("   - 机制: 多头注意力机制融合源节点和目标节点信息")
    print("   - 输出: 每条边的存在概率")
    
    print("\n3. 训练时的边标签构造")
    print("-" * 50)
    print("   代码逻辑 (trainer.py):")
    print("   ```python")
    print("   # 生成负样本")
    print("   neg_edge_index = custom_negative_sampling(...)")
    print("   ")
    print("   # 创建标签 (正样本=1, 负样本=0)")
    print("   edge_labels = torch.cat([")
    print("       torch.ones(data.pred_edge_index.shape[1]),  # 正样本")
    print("       torch.zeros(neg_edge_index.shape[1])        # 负样本")
    print("   ])")
    print("   ")
    print("   # 合并正负样本边")
    print("   data.pred_edge_index = torch.cat([")
    print("       data.pred_edge_index,  # 真实未来边")
    print("       neg_edge_index         # 负采样边")
    print("   ], dim=1)")
    print("   ```")

def compare_with_alternatives():
    """对比其他可能的级联预测方法"""
    
    print("\n" + "=" * 80)
    print("CascadePredictor vs 其他级联预测方法")
    print("=" * 80)
    
    approaches = [
        {
            "方法": "当前CascadePredictor",
            "类型": "边分类",
            "输入": "前75%图 + 候选边",
            "输出": "每条边的概率",
            "优点": "高效、可控",
            "缺点": "需要预定义候选边"
        },
        {
            "方法": "图生成模型",
            "类型": "完整图生成",
            "输入": "前75%图",
            "输出": "完整未来图",
            "优点": "自动发现新边",
            "缺点": "计算复杂、难以控制"
        },
        {
            "方法": "序列预测",
            "类型": "时间序列",
            "输入": "历史传播序列",
            "输出": "未来传播序列",
            "优点": "考虑时间动态",
            "缺点": "忽略图结构"
        },
        {
            "方法": "节点影响力预测",
            "类型": "节点排序",
            "输入": "当前图状态",
            "输出": "节点影响力分数",
            "优点": "解释性强",
            "缺点": "无法预测具体边"
        }
    ]
    
    for i, approach in enumerate(approaches, 1):
        print(f"\n{i}. {approach['方法']}")
        print(f"   类型: {approach['类型']}")
        print(f"   输入: {approach['输入']}")
        print(f"   输出: {approach['输出']}")
        print(f"   优点: {approach['优点']}")
        print(f"   缺点: {approach['缺点']}")

def main():
    """主函数"""
    print("开始CascadePredictor详细原理分析...")
    
    # 运行详细分析
    results = demonstrate_cascade_predictor_principle()
    
    # 分析模型架构
    analyze_model_architecture()
    
    # 对比其他方法
    compare_with_alternatives()
    
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    
    print("\n🎯 CascadePredictor的核心功能:")
    print("   给定前75%时间的传播图和候选边列表，")
    print("   预测每条候选边在后25%时间内出现的概率")
    
    print("\n📊 数据流:")
    print("   前75%传播图 → SharedEncoder → 节点嵌入")
    print("   节点嵌入 + 候选边 → CascadePredictor → 边概率")
    
    print("\n🏋️ 训练机制:")
    print("   监督信号 = 后25%真实边(标签=1) + 负采样边(标签=0)")
    print("   损失函数 = Binary Cross Entropy Loss")
    
    print("\n🔍 本质洞察:")
    print("   这是一个边级别的二分类任务，不是图生成任务")
    print("   模型学习的是'给定当前图状态，某条边未来出现的可能性'")
    
    print(f"\n✅ 分析完毕! 共处理 {len(results['early_edges'])} 条早期边，")
    print(f"   {len(results['future_edges'])} 条未来边，")
    print(f"   {len(results['negative_edges'])} 条负样本边")

if __name__ == "__main__":
    main()
