import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class MicrostructureCNN(nn.Module):
    """
    Deeper 1D-CNN for iceberg detection from market micro-structure sequences.
    """
    def __init__(self, in_channels: int, seq_len: int = 20):
        super().__init__()
        self.seq_len = seq_len
        
        # Extremely simple, robust encoder to prevent overfitting on L3 noise
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return torch.sigmoid(self.fc1(x))

class MicrostructureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_cnn_model(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    in_channels: int, seq_len: int,
    epochs: int = 15, batch_size: int = 64, lr: float = 0.001,
    device: str = "cpu"
) -> Tuple[MicrostructureCNN, Dict[str, List[float]]]:
    train_ds = MicrostructureDataset(X_train, y_train)
    val_ds = MicrostructureDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    model = MicrostructureCNN(in_channels, seq_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    best_val_auc = 0.0
    best_state = None
    history = {"train_loss": [], "val_auc": []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch).squeeze(-1)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                preds = model(x_batch.to(device)).squeeze(-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.numpy())
        
        from sklearn.metrics import roc_auc_score
        if len(set(all_labels)) < 2:
            val_auc = 0.5
        else:
            val_auc = roc_auc_score(all_labels, all_preds)

        avg_loss = total_loss / max(len(train_loader), 1)
        history["train_loss"].append(float(avg_loss))
        history["val_auc"].append(float(val_auc))
            
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict()
            
    if best_state:
        model.load_state_dict(best_state)
    return model, history
