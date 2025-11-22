"""
Deep learning models for mmWave human activity and person recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MMWavePointNet(nn.Module):
    """
    PointNet-based architecture for mmWave point cloud processing.
    Handles variable-sized point clouds and temporal sequences.
    """
    
    def __init__(self, num_persons: int, num_actions: int, 
                 input_features: int = 4, hidden_dim: int = 128):
        """
        Initialize the model.
        
        Args:
            num_persons: Number of different persons to classify
            num_actions: Number of different actions to classify
            input_features: Number of features per point (x, y, z, doppler/intensity)
            hidden_dim: Hidden dimension size
        """
        super(MMWavePointNet, self).__init__()
        
        self.num_persons = num_persons
        self.num_actions = num_actions
        
        # Point-wise feature extraction (shared across all points)
        self.point_feat = nn.Sequential(
            nn.Conv1d(input_features, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Global feature extraction (max pooling across points)
        self.global_feat = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Temporal processing with LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Classification heads
        lstm_output_dim = hidden_dim * 2  # Bidirectional
        
        self.person_classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_persons)
        )
        
        self.action_classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, num_frames, num_points, features)
            
        Returns:
            person_logits: (batch, num_persons)
            action_logits: (batch, num_actions)
        """
        batch_size, num_frames, num_points, num_features = x.shape
        
        # Process each frame
        frame_features = []
        for t in range(num_frames):
            frame = x[:, t, :, :]  # (batch, num_points, features)
            
            # Transpose for Conv1d: (batch, features, num_points)
            frame = frame.transpose(1, 2)
            
            # Extract point-wise features
            point_feat = self.point_feat(frame)  # (batch, 256, num_points)
            
            # Global max pooling across points
            global_feat = torch.max(point_feat, dim=2)[0]  # (batch, 256)
            
            # Apply global feature network
            global_feat = self.global_feat(global_feat)  # (batch, hidden_dim)
            
            frame_features.append(global_feat)
        
        # Stack frame features: (batch, num_frames, hidden_dim)
        sequence = torch.stack(frame_features, dim=1)
        
        # Process temporal sequence with LSTM
        lstm_out, _ = self.lstm(sequence)  # (batch, num_frames, hidden_dim*2)
        
        # Use the last time step for classification
        final_feat = lstm_out[:, -1, :]  # (batch, hidden_dim*2)
        
        # Classify person and action
        person_logits = self.person_classifier(final_feat)
        action_logits = self.action_classifier(final_feat)
        
        return person_logits, action_logits


class MMWave3DCNN(nn.Module):
    """
    3D CNN architecture for voxelized mmWave data.
    Processes spatial-temporal voxel grids.
    """
    
    def __init__(self, num_persons: int, num_actions: int,
                 input_channels: int = 1, hidden_dim: int = 128):
        """
        Initialize the 3D CNN model.
        
        Args:
            num_persons: Number of different persons to classify
            num_actions: Number of different actions to classify
            input_channels: Number of input channels
            hidden_dim: Hidden dimension size
        """
        super(MMWave3DCNN, self).__init__()
        
        self.num_persons = num_persons
        self.num_actions = num_actions
        
        # 3D Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool3d((4, 4, 4))
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4 * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Classification heads
        self.person_classifier = nn.Linear(hidden_dim, num_persons)
        self.action_classifier = nn.Linear(hidden_dim, num_actions)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input voxel grid (batch, channels, depth, height, width)
            
        Returns:
            person_logits: (batch, num_persons)
            action_logits: (batch, num_actions)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.fc(x)
        
        # Classify
        person_logits = self.person_classifier(x)
        action_logits = self.action_classifier(x)
        
        return person_logits, action_logits

