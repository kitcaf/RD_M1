import os
import glob
import numpy as np
import networkx as nx
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.nn import GCNConv, GraphConv, SAGPooling, global_mean_pool, BatchNorm, SAGEConv
from torch_geometric.data import Data
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from gensim.models import Word2Vec
import re
import jieba

#忽略这个文件

# Hyperparameters
VECTOR_SIZE = 300
WINDOW_SIZE = 5
MIN_COUNT = 1
WORKERS = 4
IN_CHANNELS = VECTOR_SIZE + 5
HIDDEN_CHANNELS = 512
OUT_CHANNELS = 4
DROPOUT = 0.25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-5
STEP_SIZE = 10
GAMMA = 0.5
K_FOLDS = 10
NUM_EPOCHS = 200
LOSS_RATIO_LINK_PRED = 0.4
LOSS_RATIO_RUMOR = 1.0 - LOSS_RATIO_LINK_PRED
MAX_GRAD_NORM = 1.0
PERCENTAGE = 0.75
BATCH_SIZE = 64
PATIENCE = 10
# Clean and tokenize string


def clean_str_cut(string, task):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    if task != "weibo":
        string = re.sub(r"[^A-Za-z0-9(),!?#@\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)

    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    words = list(jieba.cut(string.strip().lower())
                 ) if task == "weibo" else string.strip().lower().split()
    return words

# Load word2vec model for text embeddings
def load_word2vec_model(tweets_path):
    sentences = []
    with open(tweets_path, 'r') as file:
        for line in file:
            _, content = line.strip().split('\t')
            # Preprocessing text using clean_str_cut
            cleaned_content = clean_str_cut(content, task="twitter")
            sentences.append(cleaned_content)
    # Train a word2vec model
    word2vec = Word2Vec(sentences, vector_size=VECTOR_SIZE,
                        window=WINDOW_SIZE, min_count=MIN_COUNT, workers=WORKERS)
    return word2vec

# Parse source tweet texts and get embeddings
def get_node_features(tweets_path, word2vec):
    node_features = {}
    with open(tweets_path, 'r') as file:
        for line in file:
            tweet_id, content = line.strip().split('\t')
            cleaned_content = clean_str_cut(content, task="twitter")
            # Average word embeddings to get the tweet embedding
            tweet_embedding = np.mean(
                [word2vec.wv[word] for word in cleaned_content if word in word2vec.wv], axis=0)
            if np.isnan(tweet_embedding).any():
                # Handle NaN values by using zero vector
                tweet_embedding = np.zeros(VECTOR_SIZE)
            node_features[tweet_id] = tweet_embedding
    return node_features

# Parse the tree file
def parse_tree_file(file_path):
    G = nx.DiGraph()
    with open(file_path, 'r') as file:
        for line in file:
            parent, child = line.strip().split('->')
            parent_data, child_data = eval(parent.strip()), eval(child.strip())
            G.add_node(tuple(parent_data))
            G.add_node(tuple(child_data))
            G.add_edge(tuple(parent_data), tuple(child_data))
    return G

# Split cascade by percentage 对原序列划分 早期序列 
def split_cascade_by_percentage(graph, percentage):
    edges_with_times = [(u, v, float(v[2])) for u, v in graph.edges()]
    max_time = max(
        t for _, _, t in edges_with_times) if edges_with_times else 0
    time_threshold = max_time * percentage
    early_edges = [(u, v)
                   for u, v, t in edges_with_times if t <= time_threshold]
    prediction_edges = [(u, v)
                        for u, v, t in edges_with_times if t > time_threshold]
    early_graph = graph.edge_subgraph(early_edges).copy()
    prediction_graph = graph.edge_subgraph(prediction_edges).copy()
    return {"initial_graph": early_graph, "prediction_graph": prediction_graph}

# Process cascades for temporal link prediction
def process_for_temporal_link_prediction(cascade_data, node_features):
    processed_data = {}
    for tree_id, data in cascade_data.items():
        initial_graph, prediction_graph = data["initial_graph"], data["prediction_graph"]
        all_nodes = set(initial_graph.nodes).union(set(prediction_graph.nodes))
        node_mapping = {node: i for i, node in enumerate(all_nodes)}
        # Use word2vec features as node features
        node_features_matrix = np.array([node_features.get(
            str(node[0]), np.zeros(VECTOR_SIZE)) for node in all_nodes], dtype=np.float32)
        initial_edges = np.array([(node_mapping[u], node_mapping[v], float(
            u[2])) for u, v in initial_graph.edges], dtype=np.float32)
        prediction_edges = np.array([(node_mapping[u], node_mapping[v], float(
            u[2])) for u, v in prediction_graph.edges], dtype=np.float32)
        processed_data[tree_id] = {
            "node_features": node_features_matrix,
            "node_mapping": node_mapping,
            "initial_edges": initial_edges,
            "prediction_edges": prediction_edges,
        }
    return processed_data

# Convert dataset to PyTorch Geometric format
def convert_to_pytorch_geometric(dataset):
    pyg_dataset = []
    transform = T.Compose(
        [T.ToUndirected(), T.AddSelfLoops(), T.LocalDegreeProfile()])
    for tree_id, data in dataset.items():
        num_nodes = len(data["node_mapping"])
        node_features = torch.tensor(
            data["node_features"], dtype=torch.float32)
        initial_edges = torch.tensor(
            data["initial_edges"][:, :2], dtype=torch.long).T
        prediction_edges = torch.tensor(
            data["prediction_edges"][:, :2], dtype=torch.long).T
        pyg_data = Data(x=node_features, edge_index=initial_edges,
                        pred_edge_index=prediction_edges, tweet_id=tree_id)
        pyg_dataset.append(transform(pyg_data))
    return pyg_dataset

# Add numeric labels to PyG data


def add_numeric_labels_to_data(pyg_dataset, label_path):
    label_mapping = {'false': 0, 'true': 1, 'unverified': 2, 'non-rumor': 3}
    label_dict = {}
    with open(label_path, 'r') as f:
        for line in f:
            label, tweet_id = line.strip().split(':')
            label_dict[tweet_id] = label_mapping[label]
    for data in pyg_dataset:
        data.label = torch.tensor(label_dict.get(
            data.tweet_id, 2), dtype=torch.long).unsqueeze(0)
        del data.tweet_id
    return pyg_dataset

# Model definitions


class SharedEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(SharedEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.dropout1 = torch.nn.Dropout(p=DROPOUT)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.batch_norm2 = BatchNorm(hidden_channels)
        self.dropout2 = torch.nn.Dropout(p=DROPOUT)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )

    def forward(self, x, edge_index):
        # First layer with skip connection
        x_1 = self.conv1(x, edge_index)
        x_1 = self.batch_norm1(x_1).relu()
        x_1 = self.dropout1(x_1)

        # Second layer with skip connection
        x_2 = self.conv2(x_1, edge_index)
        x_2 = self.batch_norm2(x_2).relu()
        x_2 = self.dropout2(x_2)

        # Adding skip connections with MLP for non-linear transformation
        x_out = x_1 + self.mlp(x_2)
        return x_out


class CascadePredictor(torch.nn.Module):
    def __init__(self, shared_encoder):
        super(CascadePredictor, self).__init__()
        self.shared_encoder = shared_encoder
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=HIDDEN_CHANNELS, num_heads=4, batch_first=True)

    def encode(self, x, edge_index):
        return self.shared_encoder(x, edge_index)

    def decode(self, z, edge_index):
        src, dst = edge_index
        z_src = z[src].unsqueeze(1)  # shape (num_edges, 1, hidden_channels)
        z_dst = z[dst].unsqueeze(1)  # shape (num_edges, 1, hidden_channels)
        # shape (num_edges, 2, hidden_channels)
        z_combined = torch.cat([z_src, z_dst], dim=1)
        attn_output, _ = self.attention(z_combined, z_combined, z_combined)
        return torch.sigmoid(attn_output[:, 0, :].sum(dim=-1))

    def forward(self, x, edge_index, edge_index_pred):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_index_pred)


class RumorDetector(torch.nn.Module):
    def __init__(self, shared_encoder, out_channels):
        super(RumorDetector, self).__init__()
        self.shared_encoder = shared_encoder
        self.pool = SAGPooling(HIDDEN_CHANNELS, ratio=0.5)
        self.batch_norm = BatchNorm(HIDDEN_CHANNELS)
        self.fc = torch.nn.Linear(HIDDEN_CHANNELS, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.shared_encoder(x, edge_index)
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, batch=batch)
        x = self.batch_norm(x)
        x = global_mean_pool(x, batch)
        return F.log_softmax(self.fc(x), dim=-1)


class EndToEndModel(torch.nn.Module):
    def __init__(self, shared_encoder, link_pred_model, rumor_detect_model):
        super(EndToEndModel, self).__init__()
        self.shared_encoder = shared_encoder
        self.link_pred_model = link_pred_model
        self.rumor_detect_model = rumor_detect_model

    def forward(self, data):
        # 链接预测任务
        pred_edges = self.link_pred_model(
            data.x, data.edge_index, data.pred_edge_index)

        # 从链接预测的注意力层输出构建推断图
        reconstructed_edge_index = torch.cat(
            [data.edge_index, data.pred_edge_index[:, pred_edges > 0.5]], dim=1
        )

        # 使用谣言检测模型进行分类
        rumor_out = self.rumor_detect_model(
            data.x, reconstructed_edge_index, data.batch)

        return rumor_out, pred_edges


# Negative sampling with no positive overlap


def custom_negative_sampling(edge_index, num_nodes, num_neg_samples):
    negative_edges = negative_sampling(
        edge_index=edge_index, num_nodes=num_nodes, num_neg_samples=num_neg_samples)
    pos_edges_set = set([tuple(edge) for edge in edge_index.cpu().numpy().T])
    neg_edges_set = set([tuple(edge)
                        for edge in negative_edges.cpu().numpy().T])
    neg_edges_set = neg_edges_set - pos_edges_set
    negative_edges = torch.tensor(list(neg_edges_set), dtype=torch.long).T
    return negative_edges

# Training function
def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss, all_true_labels, all_pred_labels = 0, [], []
    all_link_true, all_link_pred = [], []

    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()

        neg_edge_index = custom_negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.pred_edge_index.shape[1]
        ).to(device)
        edge_labels = torch.cat([torch.ones(data.pred_edge_index.shape[1]), torch.zeros(
            neg_edge_index.shape[1])]).to(device)
        data.pred_edge_index = torch.cat(
            [data.pred_edge_index, neg_edge_index], dim=1)
        data.edge_index = data.edge_index.long()
        data.pred_edge_index = data.pred_edge_index.long()

        out, pred_edges = model(data)
        link_loss = torch.nn.BCELoss()(pred_edges, edge_labels)
        rumor_loss = torch.nn.CrossEntropyLoss()(out, data.label)
        # Adjusted loss ratio to give more emphasis on link prediction
        loss = LOSS_RATIO_LINK_PRED * link_loss + LOSS_RATIO_RUMOR * rumor_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step(loss)
        total_loss += loss.item()

        _, predicted_labels = out.max(dim=1)
        all_true_labels.extend(data.label.cpu().numpy())
        all_pred_labels.extend(predicted_labels.cpu().numpy())
        all_link_true.extend(edge_labels.cpu().numpy())
        all_link_pred.extend((pred_edges > 0.5).cpu().numpy())

        del data, out, pred_edges, edge_labels, neg_edge_index
        torch.cuda.empty_cache()

    rumor_acc = accuracy_score(all_true_labels, all_pred_labels)
    rumor_f1 = f1_score(all_true_labels, all_pred_labels, average="weighted")

    return {"loss": total_loss / len(data_loader), "acc": rumor_acc, "f1": rumor_f1}

# Evaluation function


def evaluate(model, data_loader, device):
    model.eval()
    all_true_labels, all_pred_labels = [], []
    all_link_true, all_link_pred = [], []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)

            # Generate negative samples for evaluation
            neg_edge_index = custom_negative_sampling(
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=data.pred_edge_index.shape[1]
            ).to(device)
            data.pred_edge_index = torch.cat(
                [data.pred_edge_index, neg_edge_index], dim=1)
            edge_labels = torch.cat([torch.ones(
                data.pred_edge_index.shape[1] // 2), torch.zeros(neg_edge_index.shape[1])]).to(device)

            out, pred_edges = model(data)
            _, predicted_labels = out.max(dim=1)
            all_true_labels.extend(data.label.cpu().numpy())
            all_pred_labels.extend(predicted_labels.cpu().numpy())
            all_link_true.extend(edge_labels.cpu().numpy())
            all_link_pred.extend((pred_edges > 0.5).cpu().numpy())

            del data, out, pred_edges, edge_labels, neg_edge_index
            torch.cuda.empty_cache()

    rumor_acc = accuracy_score(all_true_labels, all_pred_labels)
    rumor_f1 = f1_score(all_true_labels, all_pred_labels, average="weighted")

    return {"acc": rumor_acc, "f1": rumor_f1}

# Main function to train and evaluate


def main():
    dataset_path = "/home/hwxu/Projects/Research/NPU/WSDM/Input/Twitter16"
    label_path = os.path.join(dataset_path, 'label.txt')
    tree_dir_path = os.path.join(dataset_path, 'tree')
    source_tweets_path = os.path.join(dataset_path, 'source_tweets.txt')

    # Load word2vec model and parse tweets
    word2vec_model = load_word2vec_model(source_tweets_path)
    node_features = get_node_features(source_tweets_path, word2vec_model)

    # Process all tree files
    tree_files = glob.glob(os.path.join(tree_dir_path, "*.txt"))
    graphs = {os.path.basename(file).replace(
        ".txt", ""): parse_tree_file(file) for file in tree_files}

    # Process cascades and split by percentage
    processed_cascades = {
        tree_id: split_cascade_by_percentage(graph, PERCENTAGE) for tree_id, graph in graphs.items()
    }

    # Process cascades for temporal link prediction
    temporal_dataset = process_for_temporal_link_prediction(
        processed_cascades, node_features)
    pyg_dataset = convert_to_pytorch_geometric(temporal_dataset)
    pyg_dataset = add_numeric_labels_to_data(pyg_dataset, label_path)

    # Cross-validation setup
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    # 初始化共享编码器
    shared_encoder = SharedEncoder(IN_CHANNELS, HIDDEN_CHANNELS)

    # 初始化链接预测模型和谣言检测模型
    link_pred_model = CascadePredictor(shared_encoder)
    rumor_detect_model = RumorDetector(shared_encoder, OUT_CHANNELS)

    # 初始化端到端模型
    model = EndToEndModel(shared_encoder, link_pred_model,
                          rumor_detect_model).to(device)

    avg_acc, avg_f1 = 0, 0
    best_f1 = 0
    patience_counter = 0
    for fold, (train_idx, test_idx) in enumerate(kf.split(pyg_dataset)):
        print(f"Fold {fold + 1}/{K_FOLDS}")
        train_dataset = [pyg_dataset[i] for i in train_idx]
        test_dataset = [pyg_dataset[i] for i in test_idx]
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=GAMMA,)

        for epoch in range(1, NUM_EPOCHS + 1):
            train_results = train(model, train_loader,
                                  optimizer, scheduler, device)
            patience_counter = 0
            print(f"Epoch {epoch} | Train Loss: {train_results['loss']:.4f}")
            print(
                f"Train Accuracy: {train_results['acc']:.4f}, Train F1: {train_results['f1']:.4f}")
            test_results = evaluate(model, test_loader, device)
            print(
                f"Epoch {epoch} | Test Accuracy: {test_results['acc']:.4f}, Test F1: {test_results['f1']:.4f}")

            # Early stopping
            if test_results['f1'] > best_f1:
                best_f1 = test_results['f1']
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        avg_acc += test_results['acc']
        avg_f1 += test_results['f1']

    avg_acc /= K_FOLDS
    avg_f1 /= K_FOLDS
    print(
        f"Average Test Accuracy: {avg_acc:.4f}, Average Test F1: {avg_f1:.4f}")


if __name__ == "__main__":
    main()
