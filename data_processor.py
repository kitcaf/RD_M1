# -*- coding: utf-8 -*-
"""
数据处理模块
包含文本预处理、Word2Vec训练、图数据处理等功能
"""

import os
import glob
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from gensim.models import Word2Vec
import re
import jieba
from config import VECTOR_SIZE, WINDOW_SIZE, MIN_COUNT, WORKERS, PERCENTAGE


def clean_str_cut(string, task):
    """
    文本清洗和分词函数
    对所有数据集进行分词/字符串清洗，除了SST
    
    参数:
        string (str): 待处理的字符串
        task (str): 任务类型，如果是"weibo"则使用jieba分词
        
    返回:
        list: 分词后的单词列表
    """
    if task != "weibo":
        # 英文文本清洗
        string = re.sub(r"[^A-Za-z0-9(),!?#@\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)

    # 标点符号处理
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    # 根据任务类型选择分词方式
    words = list(jieba.cut(string.strip().lower())) if task == "weibo" else string.strip().lower().split()
    return words


def load_word2vec_model(tweets_path):
    """
    加载并训练Word2Vec模型用于文本嵌入
    
    参数:
        tweets_path (str): 推文文件路径
    返回:
        Word2Vec: 训练好的Word2Vec模型
    """
    sentences = []
    
    # 读取推文内容并预处理
    with open(tweets_path, 'r', encoding='utf-8') as file:
        for line in file:
            _, content = line.strip().split('\t')
            # 使用clean_str_cut进行文本预处理
            cleaned_content = clean_str_cut(content, task="twitter")
            sentences.append(cleaned_content)
    
    # 训练Word2Vec模型  训练Word2Vec：300维向量，窗口大小5，最小词频1
    word2vec = Word2Vec(sentences, vector_size=VECTOR_SIZE,
                        window=WINDOW_SIZE, min_count=MIN_COUNT, workers=WORKERS)
    return word2vec


def get_node_features(tweets_path, word2vec):
    """
    解析源推文文本并获取嵌入向量
    
    参数:
        tweets_path (str): 推文文件路径
        word2vec (Word2Vec): 训练好的Word2Vec模型
        
    返回:
        dict: 推文ID到嵌入向量的映射
    """
    node_features = {}
    
    with open(tweets_path, 'r', encoding='utf-8') as file:
        for line in file:
            tweet_id, content = line.strip().split('\t')
            # 清除无效词汇
            cleaned_content = clean_str_cut(content, task="twitter")
            
            # 通过平均词嵌入获取推文嵌入（300维）
            tweet_embedding = np.mean(
                [word2vec.wv[word] for word in cleaned_content if word in word2vec.wv], axis=0)
            
            # 处理NaN值，使用零向量（空推文的情况）
            if np.isnan(tweet_embedding).any():
                tweet_embedding = np.zeros(VECTOR_SIZE)
            
            # 每一个存储推文ID->向量映射
            node_features[tweet_id] = tweet_embedding
    
    return node_features


def parse_tree_file(file_path):
    """
    解析传播树文件
    
    参数:
        file_path (str): 传播树文件路径
        
    返回:
        nx.DiGraph: 有向图表示的传播树
    """
    G = nx.DiGraph()
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parent, child = line.strip().split('->')
            parent_data, child_data = eval(parent.strip()), eval(child.strip())
            # TODO 没有理解这里做什么
            # 加入节点[用户id, 推文id, 时间戳]
            G.add_node(tuple(parent_data))
            # 加入节点[用户id, 推文id, 时间戳]
            G.add_node(tuple(child_data))
            # 加入节点之间的连边
            G.add_edge(tuple(parent_data), tuple(child_data))
    
    return G


def split_cascade_by_percentage(graph, percentage=PERCENTAGE):
    """
    按时间百分比分割级联，生成早期序列和预测序列
    PERCENTAGE：75%


    参数:
        graph (nx.DiGraph): 传播图
        percentage (float): 分割百分比，默认使用配置中的值
        
    返回:
        dict: 包含初始图和预测图的字典
    """
    # 获取带时间戳的边
    edges_with_times = [(u, v, float(v[2])) for u, v in graph.edges()]
    
    if not edges_with_times:
        return {"initial_graph": graph.copy(), "prediction_graph": nx.DiGraph()}
    
    # 计算时间阈值
    max_time = max(t for _, _, t in edges_with_times)
    time_threshold = max_time * percentage
    
    # 分离早期边和预测边 前%75 和后25%时间的传播边 
    early_edges = [(u, v) for u, v, t in edges_with_times if t <= time_threshold]
    prediction_edges = [(u, v) for u, v, t in edges_with_times if t > time_threshold]
    
    # 创建子图
    early_graph = graph.edge_subgraph(early_edges).copy()
    prediction_graph = graph.edge_subgraph(prediction_edges).copy()
    
    return {"initial_graph": early_graph, "prediction_graph": prediction_graph}


def process_for_temporal_link_prediction(cascade_data, node_features):
    """
    处理级联数据用于时序链接预测
    
    参数:
        cascade_data (dict): 级联数据字典
       {"initial_graph": early_graph:nx.DiGraph, "prediction_graph": prediction_graph:nx.DiGraph}
        node_features (dict): 节点特征字典
        
    返回:
        dict: 处理后的数据集
    """
    processed_data = {}
    
    for tree_id, data in cascade_data.items():
        initial_graph, prediction_graph = data["initial_graph"], data["prediction_graph"]
        
        # 获取所有节点 1. 创建节点映射（字符串ID -> 数字索引）
        all_nodes = set(initial_graph.nodes).union(set(prediction_graph.nodes))
        node_mapping = {node: i for i, node in enumerate(all_nodes)}
        
        # 使用Word2Vec特征作为节点特征 构建节点特征矩阵
        node_features_matrix = np.array([
            node_features.get(str(node[0]), np.zeros(VECTOR_SIZE)) 
            for node in all_nodes
        ], dtype=np.float32)
        
        # 处理初始边和预测边
        initial_edges = np.array([
            (node_mapping[u], node_mapping[v], float(u[2])) 
            for u, v in initial_graph.edges
        ], dtype=np.float32)
        
        prediction_edges = np.array([
            (node_mapping[u], node_mapping[v], float(u[2])) 
            for u, v in prediction_graph.edges
        ], dtype=np.float32)
        
        processed_data[tree_id] = {
            "node_features": node_features_matrix,
            "node_mapping": node_mapping,
            "initial_edges": initial_edges,
            "prediction_edges": prediction_edges,
        }
    
    return processed_data


def convert_to_pytorch_geometric(dataset):
    """
    将数据集转换为PyTorch Geometric格式
    
    参数:
        dataset (dict): 处理后的数据集
        
    返回:
        list: PyTorch Geometric数据对象列表
    """
    pyg_dataset = []
    
    # 定义图变换
    transform = T.Compose([
        T.ToUndirected(),      # 转换为无向图
        T.AddSelfLoops(),      # 添加自环
        T.LocalDegreeProfile() # 添加局部度数特征
    ])
    
    for tree_id, data in dataset.items():
        num_nodes = len(data["node_mapping"])
        
        # 转换为张量
        node_features = torch.tensor(data["node_features"], dtype=torch.float32)
        initial_edges = torch.tensor(data["initial_edges"][:, :2], dtype=torch.long).T
        prediction_edges = torch.tensor(data["prediction_edges"][:, :2], dtype=torch.long).T
        
        # 创建PyG数据对象
        pyg_data = Data(
            x=node_features,  # 节点特征[N, 300]
            edge_index=initial_edges, # 早期边索引[2, E1]
            pred_edge_index=prediction_edges,  # 待预测边索引[2, E2]
            tweet_id=tree_id # 推文ID（后续会删除）
        )

        # 应用图变换：无向化+自环+度数特征
        pyg_dataset.append(transform(pyg_data))
    
    return pyg_dataset


def add_numeric_labels_to_data(pyg_dataset, label_path):
    """
    为PyG数据添加数值标签
    
    参数:
        pyg_dataset (list): PyTorch Geometric数据集
        label_path (str): 标签文件路径
        
    返回:
        list: 添加标签后的数据集
    """
    # 标签映射
    label_mapping = {
        'false': 0,      # 虚假谣言
        'true': 1,       # 已澄清谣言  
        'unverified': 2, # 未验证推文
        'non-rumor': 3   # 非谣言
    }
    
    # 读取标签文件
    label_dict = {}
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            label, tweet_id = line.strip().split(':')
            label_dict[tweet_id] = label_mapping[label]
    
    # 为每个数据对象添加标签
    for data in pyg_dataset:
        data.label = torch.tensor(
            label_dict.get(data.tweet_id, 2),  # 默认为unverified
            dtype=torch.long
        ).unsqueeze(0)
        del data.tweet_id  # 删除tweet_id属性
    
    return pyg_dataset


def load_and_process_data(dataset_path):
    """
    加载和处理完整数据集的主函数
    
    参数:
        dataset_path (str): 数据集根目录路径
        
    返回:
        list: 处理好的PyTorch Geometric数据集
    
    """
    # 构建文件路径-标签文件
    label_path = os.path.join(dataset_path, 'label.txt')
    # 传播树文件
    tree_dir_path = os.path.join(dataset_path, 'tree')
    # 推文（推文id, 推文内容）数据
    source_tweets_path = os.path.join(dataset_path, 'source_tweets.txt')
    
    # 检查文件是否存在
    print(f"检查文件:")
    print(f"数据集路径: {dataset_path}")
    print(f"源推文文件存在: {os.path.exists(source_tweets_path)}")
    print(f"标签文件存在: {os.path.exists(label_path)}")
    print(f"传播树目录存在: {os.path.exists(tree_dir_path)}")
    
    # 加载Word2Vec模型并解析推文
    print("正在训练Word2Vec模型...")
    # 训练word2vecmodel
    word2vec_model = load_word2vec_model(source_tweets_path)
    # 节点特征提取，通过训练好的word2vecmodel模型对推文形成推文内容嵌入向量
    node_features = get_node_features(source_tweets_path, word2vec_model)
    
    # 处理所有传播树文件
    print("正在处理传播树文件...")
    # 传播树解析
    tree_files = glob.glob(os.path.join(tree_dir_path, "*.txt"))
    graphs = {
        os.path.basename(file).replace(".txt", ""): parse_tree_file(file) 
        for file in tree_files
    }

    """
        graphs是一个传播图对象：记录每一个推文节点产生的推文传播图
        graphs：{
            推文id: DiGraph对象（论文中将DiGraph对象描述为推文id的级联）
            ...
        }
    """
    
    # 处理级联并按百分比分割
    print("正在分割级联数据...")

    processed_cascades = {
        tree_id: split_cascade_by_percentage(graph) 
        for tree_id, graph in graphs.items()
    }
    
    # 处理级联用于时序链接预测
    print("正在处理时序链接预测数据...")
    temporal_dataset = process_for_temporal_link_prediction(processed_cascades, node_features)
    
    # 转换为PyTorch Geometric格式
    print("正在转换为PyTorch Geometric格式...")
    pyg_dataset = convert_to_pytorch_geometric(temporal_dataset)
    
    # 添加标签
    print("正在添加标签...")
    pyg_dataset = add_numeric_labels_to_data(pyg_dataset, label_path)
    
    print(f"数据处理完成，共有 {len(pyg_dataset)} 个样本")
    return pyg_dataset
