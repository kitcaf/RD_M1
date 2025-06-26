import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch_geometric.utils import negative_sampling
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

def combined_loss(predicted_features, actual_features, rumor_output, rumor_labels, alpha=0.5):
    """
    Computes a combined loss for the cascade prediction and rumor detection tasks.
    
    Args:
        predicted_features (Tensor): Predicted features for the missing nodes in the cascade.
        actual_features (Tensor): True features for the missing nodes in the cascade.
        rumor_output (Tensor): Model output logits for rumor detection.
        rumor_labels (Tensor): Ground truth labels for rumor detection.
        alpha (float): Weight for balancing prediction and rumor detection losses.
        
    Returns:
        Tensor: Weighted combined loss.
    """
    prediction_loss = F.mse_loss(predicted_features, actual_features)
    rumor_loss = F.cross_entropy(rumor_output, rumor_labels)
    return alpha * prediction_loss + (1 - alpha) * rumor_loss

def train_model(prediction_model, detection_model, early_loader, full_loader, optimizer, epochs=10, alpha=0.5, patience=5, clip_value=1.0):
    """
    Trains both the cascade prediction and rumor detection models with early stopping.
    
    Args:
        prediction_model (nn.Module): Model for predicting missing cascade nodes.
        detection_model (nn.Module): Model for detecting rumors within the cascade.
        early_loader (DataLoader): DataLoader providing early cascade data.
        full_loader (DataLoader): DataLoader providing full cascade data.
        optimizer (Optimizer): Optimizer for model training.
        epochs (int): Number of training epochs.
        alpha (float): Weight for combined loss.
        patience (int): Patience for early stopping.
        clip_value (float): Maximum value for gradient clipping.
    """
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda')
    best_f1 = 0
    patience_counter = 0
    best_model_weights = None

    for epoch in range(epochs):
        prediction_model.train()
        detection_model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for early_data, full_data in zip(early_loader, full_loader):
            optimizer.zero_grad()

            with autocast('cuda'):
                # Cascade Prediction Step
                num_missing_nodes = full_data.x.size(0) - early_data.x.size(0)
                predicted_features = prediction_model(early_data, num_missing_nodes)
                full_embeddings = torch.cat([early_data.x, predicted_features], dim=0)

                # Rumor Detection Step
                full_data.x = full_embeddings
                rumor_output = detection_model(full_data)

                # Calculate Combined Loss
                loss = combined_loss(predicted_features, full_data.x[-num_missing_nodes:], rumor_output, full_data.y, alpha)
            
            # Backpropagation with Gradient Scaling
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(list(prediction_model.parameters()) + list(detection_model.parameters()), clip_value)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            # Collect Predictions and Labels
            all_preds.extend(rumor_output.argmax(dim=1).detach().cpu().numpy())
            all_labels.extend(full_data.y.cpu().numpy())

        scheduler.step()
        
        # Epoch Metrics Calculation
        f1 = f1_score(all_labels, all_preds, average='micro')
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='micro')
        recall = recall_score(all_labels, all_preds, average='micro')

        # Logging and Early Stopping
        print(f"\nEpoch [{epoch+1}/{epochs}]\nLoss: {total_loss:.4f}\n"
              f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}\n"
              f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n" + "="*50)

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            best_model_weights = {
                'prediction_model': prediction_model.state_dict(),
                'detection_model': detection_model.state_dict()
            }
            print("New best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping: Validation metrics did not improve.")
                break

    if best_model_weights:
        prediction_model.load_state_dict(best_model_weights['prediction_model'])
        detection_model.load_state_dict(best_model_weights['detection_model'])
        print("Best model weights restored.")

def test_model(prediction_model, detection_model, early_loader, full_loader, alpha=0.5):
    """
    Evaluates the trained models on the test data and reports metrics.
    
    Args:
        prediction_model (nn.Module): Model for cascade prediction.
        detection_model (nn.Module): Model for rumor detection.
        early_loader (DataLoader): DataLoader for early cascade test data.
        full_loader (DataLoader): DataLoader for full cascade test data.
        alpha (float): Weight for the combined loss function during evaluation.
    """
    prediction_model.eval()
    detection_model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for early_data, full_data in zip(early_loader, full_loader):
            with autocast('cuda'):
                num_missing_nodes = full_data.x.size(0) - early_data.x.size(0)
                predicted_features = prediction_model(early_data, num_missing_nodes)
                full_embeddings = torch.cat([early_data.x, predicted_features], dim=0)
                
                full_data.x = full_embeddings
                rumor_output = detection_model(full_data)

                loss = combined_loss(predicted_features, full_data.x[-num_missing_nodes:], rumor_output, full_data.y, alpha)
                total_loss += loss.item()

                all_preds.extend(rumor_output.argmax(dim=1).cpu().numpy())
                all_labels.extend(full_data.y.cpu().numpy())

    # Calculate Test Metrics
    f1 = f1_score(all_labels, all_preds, average='micro')
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='micro')
    recall = recall_score(all_labels, all_preds, average='micro')

    print("\nTest Results\n" + "="*50 + f"\nLoss: {total_loss:.4f}\n"
          f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}\n" + "="*50)
