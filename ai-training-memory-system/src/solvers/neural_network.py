"""
Neural network model for mathematical problem solving.
Feedforward network that learns from vector embeddings.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import numpy as np


class MathSolverNN(nn.Module):
    """
    Feedforward neural network for solving mathematical problems.
    
    Architecture:
    - Input Layer: 384 nodes (vector dimension)
    - Hidden Layer 1: 256 nodes + ReLU + Dropout(0.3)
    - Hidden Layer 2: 128 nodes + ReLU + Dropout(0.3)
    - Hidden Layer 3: 64 nodes + ReLU
    - Output Layer: 1 node (predicted numerical answer)
    """
    
    def __init__(self, input_dim: int = 384):
        """
        Initialize the neural network.
        
        Args:
            input_dim: Dimension of input vectors (default: 384)
        """
        super(MathSolverNN, self).__init__()
        
        self.input_dim = input_dim
        
        # Network layers
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 64)
        
        self.fc4 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 384)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.relu(self.fc3(x))
        
        x = self.fc4(x)
        
        return x


class NeuralNetworkSolver:
    """
    Wrapper class for training and using the neural network solver.
    """
    
    def __init__(self, input_dim: int = 384, learning_rate: float = 0.001):
        """
        Initialize the neural network solver.
        
        Args:
            input_dim: Dimension of input vectors
            learning_rate: Learning rate for optimizer
        """
        self.model = MathSolverNN(input_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.input_dim = input_dim
        self.trained = False
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> dict:
        """
        Train the neural network.
        
        Args:
            X_train: Training embeddings of shape (n_samples, 384)
            y_train: Training labels of shape (n_samples,)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            
        Returns:
            Dictionary with training history
        """
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        
        # Split into train and validation
        n_samples = len(X_train)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_train_split = X_tensor[train_indices]
        y_train_split = y_tensor[train_indices]
        X_val = X_tensor[val_indices] if n_val > 0 else None
        y_val = y_tensor[val_indices] if n_val > 0 else None
        
        history = {'train_loss': [], 'val_loss': []}
        
        self.model.train()
        
        for epoch in range(epochs):
            # Training
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(X_train_split), batch_size):
                batch_X = X_train_split[i:i+batch_size]
                batch_y = y_train_split[i:i+batch_size]
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = self.criterion(val_outputs, y_val)
                    history['val_loss'].append(val_loss.item())
                self.model.train()
        
        self.trained = True
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input embeddings of shape (n_samples, 384)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            predictions = self.model(X_tensor)
            return predictions.numpy().flatten()
    
    def save_model(self, filepath: str):
        """Save model weights to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'trained': self.trained
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model weights from file."""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.trained = checkpoint.get('trained', False)
