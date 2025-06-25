import torch
import yaml
from data import CascadeDatasetProcessor
from model import CascadePredictionModel, RumorDetectionModel
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from train import test_model, train_model
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_config(config_path: str = "./config.yaml") -> dict:
    """
    从YAML文件加载配置设置。

    参数:
        config_path (str): 配置YAML文件的路径。

    返回:
        dict: 配置字典。
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}")  # 配置文件未找到
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")  # YAML解析错误


def main():
    """
    主函数，用于加载配置、初始化数据集、模型，并训练/测试模型流水线。
    """
    # 加载配置文件
    config = load_config(
        '/home/hwxu/Projects/Research/NPU/WSDM/Libs/config.yaml'
    )

    # 初始化数据处理器
    processor = CascadeDatasetProcessor(
        cascade_file_path=config['data']['cascade_file_path'],  # 级联文件路径
        label_file_path=config['data']['label_file_path'],  # 标签文件路径
        max_nodes=config['data']['max_nodes'],  # 最大节点数
        output_dir=config['data']['output_dir']  # 输出目录
    )
    processor.load_data()  # 加载数据

    # 加载处理后的数据集
    early_cascades = torch.load(
        f"{config['data']['output_dir']}/early_cascades.pt"  # 早期级联数据
    )
    full_cascades = torch.load(
        f"{config['data']['output_dir']}/full_cascades.pt"  # 完整级联数据
    )

    # 将数据集分割为训练和测试
    train_size = int(config['data']['train_split'] * len(early_cascades))  # 计算训练集大小
    train_early, test_early = early_cascades[:
                                             train_size], early_cascades[train_size:]  # 分割早期级联数据
    train_full, test_full = full_cascades[:
                                          train_size], full_cascades[train_size:]  # 分割完整级联数据

    # 为训练和测试分割创建DataLoader
    train_early_loader = DataLoader(
        train_early, batch_size=config['data']['batch_size'], shuffle=True  # 训练数据加载器，打乱数据
    )
    train_full_loader = DataLoader(
        train_full, batch_size=config['data']['batch_size'], shuffle=True  # 训练数据加载器，打乱数据
    )
    test_early_loader = DataLoader(
        test_early, batch_size=config['data']['batch_size'], shuffle=False  # 测试数据加载器，不打乱数据
    )
    test_full_loader = DataLoader(
        test_full, batch_size=config['data']['batch_size'], shuffle=False  # 测试数据加载器，不打乱数据
    )

    # 模型配置和初始化
    in_channels = early_cascades[0].x.size(
        1) if 'in_channels' not in config['model'] else config['model']['in_channels']  # 输入通道数
    hidden_channels = config['model']['hidden_channels']  # 隐藏层通道数
    num_classes = config['model']['num_classes']  # 类别数
    gnn_type = config['model'].get('gnn_type', 'GCN')  # GNN类型

    prediction_model = CascadePredictionModel(in_channels, hidden_channels)  # 初始化级联预测模型
    detection_model = RumorDetectionModel(
        in_channels, hidden_channels, num_classes, gnn_type  # 初始化谣言检测模型
    )

    # 初始化优化器
    lr = config['training']['learning_rate']  # 学习率
    optimizer = AdamW(
        list(prediction_model.parameters()) + list(detection_model.parameters()), lr=lr  # 优化器
    )

    # 训练模型
    print("Starting model training...")  # 开始模型训练
    train_model(
        prediction_model,
        detection_model,
        train_early_loader,
        train_full_loader,
        optimizer,
        epochs=config['training']['epochs'],  # 训练轮数
        alpha=config['training']['alpha']  # 损失平衡系数
    )

    # 测试模型
    print("Evaluating model performance:")  # 评估模型性能
    test_model(
        prediction_model,
        detection_model,
        test_early_loader,
        test_full_loader
    )


if __name__ == "__main__":
    main()  # 当作为主程序运行时，执行主函数
