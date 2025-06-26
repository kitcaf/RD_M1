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
        Initializes the CascadeDatasetProcessor for handling cascade and label data.

        Args:
            cascade_file_path (str): Path to the cascade data file.
            label_file_path (str): Path to the label data file.
            max_nodes (int): Max nodes in early cascade, used for splitting.
            output_dir (str): Directory to save processed data.
            test_size (float): Ratio of data to be used for testing.
        """
        self.cascade_file_path = cascade_file_path
        self.label_file_path = label_file_path
        self.max_nodes = max_nodes
        self.output_dir = output_dir
        self.test_size = test_size

        # Data storage
        self.cascade_df = None
        self.label_df = None
        self.merged_data = None

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """
        Loads cascade and label data from specified file paths and merges them.

        The label file should have a 'label' and 'root-id', and cascade file should
        contain details for nodes in the cascade.
        """
        # Load label data
        label_data = []
        with open(self.label_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                label_data.append({'label': parts[0], 'root-id': parts[2]})
        self.label_df = pd.DataFrame(label_data)

        # Load cascade data
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
            self.cascade_df, self.label_df, on='root-id', how='left')

    def process_and_split_data(self) -> Tuple[Tuple[List[Data], List[Data]], Tuple[List[Data], List[Data]]]:
        """
        Processes the merged data into early and full cascades and splits into train and test sets.

        Returns:
            Tuple: (train, test) sets of early and full cascades.
        """
        early_cascades, full_cascades = [], []

        for root_id, group in self.merged_data.groupby('root-id'):
            label = 1 if group['label'].iloc[0] == 'true' else 0

            # Create early cascade (limited to max_nodes)
            early_group = group.head(self.max_nodes)
            early_data = self._create_data_object(early_group, label)
            early_cascades.append(early_data)

            # Create full cascade
            full_data = self._create_data_object(group, label)
            full_cascades.append(full_data)

        # Split into training and testing sets
        early_train, early_test, full_train, full_test = train_test_split(
            early_cascades, full_cascades, test_size=self.test_size, random_state=42
        )

        # Save processed data
        torch.save(early_train, os.path.join(
            self.output_dir, 'early_train.pt'))
        torch.save(full_train, os.path.join(self.output_dir, 'full_train.pt'))
        torch.save(early_test, os.path.join(self.output_dir, 'early_test.pt'))
        torch.save(full_test, os.path.join(self.output_dir, 'full_test.pt'))

        return (early_train, full_train), (early_test, full_test)

    def _create_data_object(self, group: pd.DataFrame, label: int) -> Data:
        """
        Creates a PyTorch Geometric Data object from the DataFrame group.

        Args:
            group (pd.DataFrame): Group of nodes in a cascade.
            label (int): Label indicating whether the cascade is rumor or not.

        Returns:
            Data: PyTorch Geometric Data object for the cascade.
        """
        node_features = torch.tensor(
            [[row['text-length']] for _, row in group.iterrows()], dtype=torch.float)
        edge_index = []
        node_index_map = {tweet: idx for idx,
                          tweet in enumerate(group['current-tweet'])}

        for _, row in group.iterrows():
            parent_tweet = row['parent-tweet']
            if parent_tweet != 'None' and parent_tweet in node_index_map:
                src = node_index_map[parent_tweet]
                tgt = node_index_map[row['current-tweet']]
                edge_index.append([src, tgt])

        # Convert edge list to tensor
        if edge_index:
            edge_index = torch.tensor(
                edge_index, dtype=torch.long).t().contiguous()
            if edge_index.max() >= node_features.size(0):
                raise ValueError("edge_index contains out-of-bounds indices.")
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return Data(x=node_features, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))

    def create_data_loaders(self, data_tuple: Tuple[List[Data], List[Data]], batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """
        Creates DataLoaders for training and test sets.

        Args:
            data_tuple (Tuple[List[Data], List[Data]]): Data tuples for training and test sets.
            batch_size (int): Number of samples per batch.

        Returns:
            Tuple[DataLoader, DataLoader]: Training and test DataLoaders.
        """
        train_loader = DataLoader(
            data_tuple[0], batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(
            data_tuple[1], batch_size=batch_size, shuffle=False)
        return train_loader, test_loader


def test_data_processing_and_splitting():
    """
    Function to test the data processing and splitting functionality.

    Verifies loading, processing, splitting, and DataLoader creation.
    """
    processor = CascadeDatasetProcessor(
        cascade_file_path='/home/hwxu/Projects/Research/NPU/WSDM/Input/Twitter16/data.TD_RvNN.vol_5000.txt',
        label_file_path='/home/hwxu/Projects/Research/NPU/WSDM/Input/Twitter16/Twitter16_label_all.txt',
        max_nodes=3,
        output_dir='/home/hwxu/Projects/Research/NPU/WSDM/Input/Twitter16'
    )

    # Load and process data, split into training and testing
    processor.load_data()
    (early_train, full_train), (early_test,
                                full_test) = processor.process_and_split_data()

    # Create DataLoaders
    train_loaders = processor.create_data_loaders((early_train, full_train))
    test_loaders = processor.create_data_loaders((early_test, full_test))

    # Output information for verification
    print(
        f"Train Early Cascades: {len(early_train)}, Train Full Cascades: {len(full_train)}")
    print(
        f"Test Early Cascades: {len(early_test)}, Test Full Cascades: {len(full_test)}")
    print(f"Sample Early Train Cascade: {early_train[0]}")
    print(f"Sample Full Train Cascade: {full_train[0]}")


if __name__ == "__main__":
    test_data_processing_and_splitting()
