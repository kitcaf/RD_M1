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
    Loads configuration settings from a YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        dict: Configuration dictionary.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


def main():
    """
    Main function to load configuration, initialize dataset, models, 
    and train/test the model pipeline.
    """
    # Load configuration file
    config = load_config(
        '/home/hwxu/Projects/Research/NPU/WSDM/Libs/config.yaml'
    )

    # Initialize data processor
    processor = CascadeDatasetProcessor(
        cascade_file_path=config['data']['cascade_file_path'],
        label_file_path=config['data']['label_file_path'],
        max_nodes=config['data']['max_nodes'],
        output_dir=config['data']['output_dir']
    )
    processor.load_data()

    # Load processed datasets
    early_cascades = torch.load(
        f"{config['data']['output_dir']}/early_cascades.pt"
    )
    full_cascades = torch.load(
        f"{config['data']['output_dir']}/full_cascades.pt"
    )

    # Split dataset into training and testing
    train_size = int(config['data']['train_split'] * len(early_cascades))
    train_early, test_early = early_cascades[:
                                             train_size], early_cascades[train_size:]
    train_full, test_full = full_cascades[:
                                          train_size], full_cascades[train_size:]

    # Create DataLoader for train and test splits
    train_early_loader = DataLoader(
        train_early, batch_size=config['data']['batch_size'], shuffle=True
    )
    train_full_loader = DataLoader(
        train_full, batch_size=config['data']['batch_size'], shuffle=True
    )
    test_early_loader = DataLoader(
        test_early, batch_size=config['data']['batch_size'], shuffle=False
    )
    test_full_loader = DataLoader(
        test_full, batch_size=config['data']['batch_size'], shuffle=False
    )

    # Model configuration and initialization
    in_channels = early_cascades[0].x.size(
        1) if 'in_channels' not in config['model'] else config['model']['in_channels']
    hidden_channels = config['model']['hidden_channels']
    num_classes = config['model']['num_classes']
    gnn_type = config['model'].get('gnn_type', 'GCN')

    prediction_model = CascadePredictionModel(in_channels, hidden_channels)
    detection_model = RumorDetectionModel(
        in_channels, hidden_channels, num_classes, gnn_type
    )

    # Initialize optimizer
    lr = config['training']['learning_rate']
    optimizer = AdamW(
        list(prediction_model.parameters()) + list(detection_model.parameters()), lr=lr
    )

    # Training the model
    print("Starting model training...")
    train_model(
        prediction_model,
        detection_model,
        train_early_loader,
        train_full_loader,
        optimizer,
        epochs=config['training']['epochs'],
        alpha=config['training']['alpha']
    )

    # Testing the model
    print("Evaluating model performance:")
    test_model(
        prediction_model,
        detection_model,
        test_early_loader,
        test_full_loader
    )


if __name__ == "__main__":
    main()
