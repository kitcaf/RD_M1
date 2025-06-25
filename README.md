# D$^2$: Customizing Two-Stage Graph Neural Networks for Early Rumor Detection through Cascade Diffusion Prediction

## Abstract

Early rumor detection is crucial for mitigating the widespread dissemination of misinformation. Existing methods predominantly rely on complete rumor diffusion graphs, which are challenging to obtain in real-world scenarios, complicating early detection efforts. To address this challenge, we propose **$D^2$**, a two-stage framework for early rumor **D**etection, integrating cascade **D**iffusion prediction. This framework aims to enhance early rumor detection by incorporating diffusion prediction capabilities. Specifically, a dynamic heterogeneous graph neural network (GNN) is developed to jointly model users' social and propagation graphs, enabling accurate prediction of potential diffusion paths using limited observed data within short time windows. The inferred diffusion paths are then integrated with early-stage data, and GNNs are employed for graph classification. However, the varying data distributions across different social media platforms necessitate extensive tuning to optimize GNN architectures. To facilitate the detection of rumor diffusion graphs at the initial stages, a search space is designed across four dimensions—*aggregation*, *merge*, *readout*, and *sequence* functions—encompassing various GNN architectures. Subsequently, D$^2$ employs an efficient differentiable search algorithm to identify high-performance GNNs within this search space. Experimental results on real social media datasets demonstrate that this approach significantly improves both the accuracy and robustness of early rumor detection.

## Project Structure

The project consists of several key components:

- **data.py**: Defines the `CascadeDatasetProcessor` for loading, preprocessing, and splitting datasets.
- **model.py**: Contains implementations of the `CascadePredictionModel` and `RumorDetectionModel`.
- **train.py**: Training and evaluation scripts for model training, testing, and performance metrics calculation.
- **config.yaml**: Configuration file to customize dataset paths, model parameters, and training settings.
- **main.py**: Main script to execute the data processing, model initialization, training, and testing pipeline.

## Installation

To set up the environment for the pipeline:

1. **Clone the repository**:

   ```sh
   git clone https://github.com/cgao-comp/D2.git
   cd D2
   ```

2. **Create a virtual environment** (recommended):

   ```sh
   python3 -m venv D2
   source D2/bin/activate
   ```

3. **Install dependencies**:

   ```sh
   pip install -r requirements.txt
   ```

## Usage

To train and test the rumor detection pipeline, run the `main.py` script after configuring parameters in `config.yaml`.

```sh
python main.py
```

### Parameters

Customize the pipeline by adjusting `config.yaml`. Key parameters include:

- **Dataset paths**: Specify paths to your dataset and labels.
- **Batch size** and **learning rate** for training.
- **GNN type**: Choose between GCN, GAT, or SAGE layers.
- **Model hyperparameters**: Adjust `in_channels`, `hidden_channels`, and number of classes.
  
## Model Architecture

The pipeline comprises two main components:

1. **Cascade Prediction Model**: Predicts embeddings for missing nodes in partial cascades using a configurable GNN model (e.g., GCN).
2. **Rumor Detection Model**: Detects rumors using the cascade structure, with a fusion layer for feature interaction and options for **mean** or **max** pooling.

Both models are configurable via `config.yaml` to allow flexibility in architectural choices.

## Datasets

This project supports loading and processing datasets with cascade structures and rumor labels. Use `CascadeDatasetProcessor` to prepare data for model input. Datasets should include:

- **Cascade information**: Nodes and edges representing user interactions.
- **Label information**: Binary or multi-class labels indicating rumor or non-rumor cascades.

### Preprocessing

The `data.py` script preprocesses data, splits it into training and testing sets, and serializes it for reuse.

## Evaluation

The pipeline is evaluated using common metrics for rumor detection:

- **Accuracy**, **Precision**, **Recall**, and **F1-score** are calculated to assess performance on test sets.

## Results

Using the pipeline on social network datasets, the models achieve robust performance, demonstrating improvements in rumor detection accuracy and generalization.

## Configuration Example (config.yaml)

```yaml
data:
  cascade_file_path: "/path/to/cascade_data.txt"
  label_file_path: "/path/to/label_data.txt"
  max_nodes: 32
  output_dir: "/path/to/output_dir"
  batch_size: 32
  train_split: 0.6

model:
  in_channels: 1
  hidden_channels: 128
  num_classes: 4
  gnn_type: "GCN"

training:
  learning_rate: 0.001
  epochs: 10
  alpha: 0.5
  optimizer: "Adam"
```

## Acknowledgments

This work is based on sociological theories of social influence and advanced graph neural network methodologies. We acknowledge the works that inspired this model, including **RvNN** ([GitHub Repository](https://github.com/majingCUHK/Rumor_RvNN)) and **SANE** ([GitHub Repository](https://github.com/LARS-research/SANE)).

## License

This project is licensed under the MIT License.

## Citation

If you use this pipeline in your research, please cite our project as follows:

```bibtex
@article{rumordetection2025,
  title={D$^2$: Customizing Two-Stage Graph Neural Networks for Early Rumor Detection through Cascade Diffusion Prediction},
  author={Haowei Xu, Chao Gao, Xianghua Li, Zhen Wang},
  journal={The 18th ACM International Conference on Web Search and Data Mining},
  year={2025}
}
```
