import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import (GATConv, GCNConv, SAGEConv, global_max_pool,
                                global_mean_pool)


class CascadeGNNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, gnn_type: str = 'GCN'):
        """
        Initializes a single GNN layer with a specified type (GCN, GAT, or SAGE).

        Args:
            in_channels (int): Input feature dimension.
            out_channels (int): Output feature dimension.
            gnn_type (str): Type of GNN layer ('GCN', 'GAT', or 'SAGE').
        """
        super(CascadeGNNLayer, self).__init__()
        if gnn_type == 'GCN':
            self.conv = GCNConv(in_channels, out_channels)
        elif gnn_type == 'GAT':
            self.conv = GATConv(in_channels, out_channels, heads=1)
        elif gnn_type == 'SAGE':
            self.conv = SAGEConv(in_channels, out_channels)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GNN layer.

        Args:
            x (Tensor): Node feature matrix.
            edge_index (Tensor): Edge index tensor.

        Returns:
            Tensor: Output feature matrix after GNN layer.
        """
        return self.conv(x, edge_index)


class CascadePredictionModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        """
        Cascade prediction model for generating missing node embeddings.

        Args:
            in_channels (int): Input feature dimension.
            hidden_channels (int): Hidden layer feature dimension.
        """
        super(CascadePredictionModel, self).__init__()
        self.gnn1 = GCNConv(in_channels, hidden_channels)
        self.gnn2 = GCNConv(hidden_channels, hidden_channels)
        self.output_layer = Linear(hidden_channels, in_channels)

    def forward(self, data, num_missing_nodes: int) -> torch.Tensor:
        """
        Forward pass to predict embeddings for missing nodes in the cascade.

        Args:
            data (Data): PyG data object containing node features and edge index.
            num_missing_nodes (int): Number of missing nodes to predict.

        Returns:
            Tensor: Predicted embeddings for missing nodes.
        """
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gnn1(x, edge_index))
        x = F.relu(self.gnn2(x, edge_index))

        # Generate embeddings for missing nodes
        predicted_embeddings = self.output_layer(
            torch.randn(num_missing_nodes, x.size(1)).to(x.device)
        )
        predicted_embeddings = torch.concat([x, predicted_embeddings], dim=0)
        return predicted_embeddings


class RumorDetectionModel(nn.Module):
    def __init__(self, num_features: int, hidden_channels: int, num_classes: int, gnn_type: str = 'GCN', pooling: str = 'mean'):
        """
        Model for rumor detection in cascades, combining multiple GNN layers and pooling.

        Args:
            num_features (int): Number of input features per node.
            hidden_channels (int): Hidden layer dimension.
            num_classes (int): Number of output classes.
            gnn_type (str): Type of GNN layer to use ('GCN', 'GAT', or 'SAGE').
            pooling (str): Pooling method ('mean' or 'max').
        """
        super(RumorDetectionModel, self).__init__()
        self.gnn1 = CascadeGNNLayer(num_features, hidden_channels, gnn_type)
        self.gnn2 = CascadeGNNLayer(num_features, hidden_channels, gnn_type)
        self.fc_fusion = nn.Linear(4 * hidden_channels, hidden_channels)
        self.pooling = pooling
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, data) -> torch.Tensor:
        """
        Forward pass for rumor detection model.

        Args:
            data (Data): PyG data object with node features and edge index.

        Returns:
            Tensor: Output logits for rumor detection.
        """
        x, edge_index = data.x, data.edge_index
        x_s = self.gnn1(x, edge_index)
        x_d = self.gnn2(x, edge_index)

        # Fuse representations of different relations
        x_interaction = torch.cat([x_s, x_d, x_s * x_d, x_s - x_d], dim=1)
        x_fused = self.fc_fusion(x_interaction)

        # Global pooling
        if self.pooling == 'mean':
            x_pooled = global_mean_pool(x_fused, data.batch)
        elif self.pooling == 'max':
            x_pooled = global_max_pool(x_fused, data.batch)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")

        return self.fc(x_pooled)


class SequentialEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        LSTM-based sequential encoder for capturing sequence information.

        Args:
            input_dim (int): Dimension of input features.
            hidden_dim (int): Dimension of hidden state in LSTM.
        """
        super(SequentialEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sequence encoding.

        Args:
            x (Tensor): Input tensor with shape (batch_size, seq_length, input_dim).

        Returns:
            Tensor: Encoded representation for the sequence.
        """
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


class FullRumorDetectionPipeline(nn.Module):
    def __init__(self, num_features: int, hidden_channels: int, num_classes: int, gnn_type: str = 'GCN', pooling: str = 'mean'):
        """
        Complete pipeline for rumor detection, including cascade prediction, rumor detection, and sequence encoding.

        Args:
            num_features (int): Input feature dimension for each node.
            hidden_channels (int): Hidden layer dimension.
            num_classes (int): Number of output classes for rumor detection.
            gnn_type (str): Type of GNN layer ('GCN', 'GAT', or 'SAGE').
            pooling (str): Pooling method ('mean' or 'max').
        """
        super(FullRumorDetectionPipeline, self).__init__()
        self.prediction_model = CascadePredictionModel(
            num_features, hidden_channels)
        self.detection_model = RumorDetectionModel(
            hidden_channels, hidden_channels, num_classes, gnn_type, pooling)
        self.seq_encoder = SequentialEncoder(hidden_channels, hidden_channels)

    def forward(self, early_data, full_data) -> torch.Tensor:
        """
        Forward pass for the full rumor detection pipeline.

        Args:
            early_data (Data): PyG data object representing early cascade data.
            full_data (Data): PyG data object representing full cascade data.

        Returns:
            Tensor: Encoded sequence representation.
        """
        # Step 1: Cascade Prediction
        num_missing_nodes = full_data.x.size(0) - early_data.x.size(0)
        predicted_features = self.prediction_model(
            early_data, num_missing_nodes)

        # Step 2: Concatenate early cascade and predicted embeddings
        full_data.x = torch.cat([early_data.x, predicted_features], dim=0)
        detection_output = self.detection_model(full_data)

        # Step 3: Sequence Encoding
        sequence_rep = self.seq_encoder(detection_output)

        return sequence_rep
