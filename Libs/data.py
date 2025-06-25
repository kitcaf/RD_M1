import os
from typing import List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class CascadeDatasetProcessor:
    def __init__(self, cascade_file_path: str, label_file_path: str, max_nodes: int = 10, output_dir: str = './processed_data', test_size: float = 0.2):
        """
        初始化CascadeDatasetProcessor，用于处理级联和标签数据。

        参数:
            cascade_file_path (str): 级联数据文件路径。
            label_file_path (str): 标签数据文件路径。
            max_nodes (int): 早期级联中的最大节点数，用于分割。
            output_dir (str): 保存处理后数据的目录。
            test_size (float): 用于测试的数据比例。
        """
        self.cascade_file_path = cascade_file_path
        self.label_file_path = label_file_path
        self.max_nodes = max_nodes
        self.output_dir = output_dir
        self.test_size = test_size

        # 数据存储
        self.cascade_df = None
        self.label_df = None
        self.merged_data = None

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """
        从指定的文件路径加载级联和标签数据，并合并它们。

        标签文件应包含'label'和'root-id'，级联文件应包含级联中节点的详细信息。
        """
        # 加载标签数据
        label_data = []
        with open(self.label_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                label_data.append({'label': parts[0], 'root-id': parts[2]})
        self.label_df = pd.DataFrame(label_data)

        # 加载级联数据
        cascade_data = []
        with open(self.cascade_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                cascade_data.append({
                    'root-id': parts[0],
                    'parent-tweet': parts[1],
                    'current-tweet': parts[2],
                    'parent-number': int(parts[3]),
                    'text-length': int(parts[4]),
                    'index-counts': parts[5] if len(parts) > 5 else ''
                })
        self.cascade_df = pd.DataFrame(cascade_data)
        self.merged_data = pd.merge(
            self.cascade_df, self.label_df, on='root-id', how='left')  # 合并级联和标签数据

    def process_and_split_data(self) -> Tuple[Tuple[List[Data], List[Data]], Tuple[List[Data], List[Data]]]:
        """
        将合并的数据处理为早期和完整级联，并分割为训练和测试集。

        返回:
            Tuple: (train, test)集的早期和完整级联。
        """
        early_cascades, full_cascades = [], []

        for root_id, group in self.merged_data.groupby('root-id'):
            label = 1 if group['label'].iloc[0] == 'true' else 0  # 将标签转换为数值

            # 创建早期级联（限制为max_nodes）
            early_group = group.head(self.max_nodes)
            early_data = self._create_data_object(early_group, label)  # 创建早期级联的PyG数据对象
            early_cascades.append(early_data)

            # 创建完整级联
            full_data = self._create_data_object(group, label)  # 创建完整级联的PyG数据对象
            full_cascades.append(full_data)

        # 分割为训练和测试集
        early_train, early_test, full_train, full_test = train_test_split(
            early_cascades, full_cascades, test_size=self.test_size, random_state=42
        )

        # 保存处理后的数据
        torch.save(early_train, os.path.join(
            self.output_dir, 'early_train.pt'))
        torch.save(full_train, os.path.join(self.output_dir, 'full_train.pt'))
        torch.save(early_test, os.path.join(self.output_dir, 'early_test.pt'))
        torch.save(full_test, os.path.join(self.output_dir, 'full_test.pt'))

        return (early_train, full_train), (early_test, full_test)

    def _create_data_object(self, group: pd.DataFrame, label: int) -> Data:
        """
        从DataFrame组创建PyTorch Geometric数据对象。

        参数:
            group (pd.DataFrame): 级联中的节点组。
            label (int): 指示级联是否为谣言的标签。

        返回:
            Data: 级联的PyTorch Geometric数据对象。
        """
        node_features = torch.tensor(
            [[row['text-length']] for _, row in group.iterrows()], dtype=torch.float)  # 将文本长度作为节点特征
        edge_index = []
        node_index_map = {tweet: idx for idx,
                          tweet in enumerate(group['current-tweet'])}  # 创建推文ID到索引的映射

        for _, row in group.iterrows():
            parent_tweet = row['parent-tweet']
            if parent_tweet != 'None' and parent_tweet in node_index_map:
                src = node_index_map[parent_tweet]  # 父节点索引
                tgt = node_index_map[row['current-tweet']]  # 当前节点索引
                edge_index.append([src, tgt])  # 添加边

        # 将边列表转换为张量
        if edge_index:
            edge_index = torch.tensor(
                edge_index, dtype=torch.long).t().contiguous()  # 转置使形状为[2, num_edges]
            if edge_index.max() >= node_features.size(0):
                raise ValueError("edge_index contains out-of-bounds indices.")  # 检查边索引是否有效
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)  # 创建空边索引

        return Data(x=node_features, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))  # 创建PyG数据对象

    def create_data_loaders(self, data_tuple: Tuple[List[Data], List[Data]], batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """
        为训练和测试集创建DataLoaders。

        参数:
            data_tuple (Tuple[List[Data], List[Data]]): 训练和测试集的数据元组。
            batch_size (int): 每批样本数。

        返回:
            Tuple[DataLoader, DataLoader]: 训练和测试DataLoaders。
        """
        train_loader = DataLoader(
            data_tuple[0], batch_size=batch_size, shuffle=True)  # 训练数据加载器，打乱数据
        test_loader = DataLoader(
            data_tuple[1], batch_size=batch_size, shuffle=False)  # 测试数据加载器，不打乱数据
        return train_loader, test_loader


def test_data_processing_and_splitting():
    """
    测试数据处理和分割功能的函数。

    验证加载、处理、分割和DataLoader创建。
    """
    processor = CascadeDatasetProcessor(
        cascade_file_path='/home/hwxu/Projects/Research/NPU/WSDM/Input/Twitter16/data.TD_RvNN.vol_5000.txt',
        label_file_path='/home/hwxu/Projects/Research/NPU/WSDM/Input/Twitter16/Twitter16_label_all.txt',
        max_nodes=3,
        output_dir='/home/hwxu/Projects/Research/NPU/WSDM/Input/Twitter16'
    )

    # 加载并处理数据，分割为训练和测试
    processor.load_data()
    (early_train, full_train), (early_test,
                                full_test) = processor.process_and_split_data()

    # 创建DataLoaders
    train_loaders = processor.create_data_loaders((early_train, full_train))
    test_loaders = processor.create_data_loaders((early_test, full_test))

    # 输出验证信息
    print(
        f"Train Early Cascades: {len(early_train)}, Train Full Cascades: {len(full_train)}")
    print(
        f"Test Early Cascades: {len(early_test)}, Test Full Cascades: {len(full_test)}")
    print(f"Sample Early Train Cascade: {early_train[0]}")
    print(f"Sample Full Train Cascade: {full_train[0]}")


if __name__ == "__main__":
    test_data_processing_and_splitting()  # 当作为主程序运行时，执行测试函数
