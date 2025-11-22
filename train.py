"""
Training script for mmWave human activity and person recognition.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import MMWaveDataLoader, MMWaveDataset
from model import MMWavePointNet, MMWave3DCNN


class MMWaveTorchDataset(Dataset):
    """PyTorch Dataset wrapper for mmWave data."""
    
    def __init__(self, data_structure, max_frames=None, max_points=None, preload=True, device='cpu'):
        self.data_structure = data_structure
        self.loader = MMWaveDataLoader(".")
        self.max_frames = max_frames
        self.max_points = max_points
        self.preload = preload
        self.device = device
        self.cache = {}
        
        # Preload all data if requested
        if preload:
            print("Preloading dataset into memory...")
            for idx in tqdm(range(len(data_structure)), desc="Preloading"):
                self._load_and_cache(idx)
        
    def _load_and_cache(self, idx):
        """Load sequence and cache it."""
        if idx in self.cache:
            return
        
        sample = self.data_structure[idx]
        sequence = self.loader.load_sequence(sample['path'], frame_segment=sample.get('frame_segment'))
        
        # Limit frames and points if specified
        if self.max_frames and sequence.shape[0] > self.max_frames:
            sequence = sequence[:self.max_frames]
        if self.max_points and sequence.shape[1] > self.max_points:
            sequence = sequence[:, :self.max_points]
        
        # Pad frames if needed
        if self.max_frames and sequence.shape[0] < self.max_frames:
            padding = np.zeros((self.max_frames - sequence.shape[0], 
                              sequence.shape[1], sequence.shape[2]))
            sequence = np.vstack([sequence, padding])
        
        # Pad points if needed
        if self.max_points and sequence.shape[1] < self.max_points:
            padding = np.zeros((sequence.shape[0], 
                              self.max_points - sequence.shape[1], 
                              sequence.shape[2]))
            sequence = np.concatenate([sequence, padding], axis=1)
        
        self.cache[idx] = {
            'sequence': torch.FloatTensor(sequence),
            'person_id': sample['person_id'],
            'action_id': sample['action_id']
        }
        
    def __len__(self):
        return len(self.data_structure)
    
    def __getitem__(self, idx):
        if self.preload:
            return self.cache[idx]
        
        sample = self.data_structure[idx]
        sequence = self.loader.load_sequence(sample['path'], frame_segment=sample.get('frame_segment'))
        
        # Limit frames and points if specified
        if self.max_frames and sequence.shape[0] > self.max_frames:
            sequence = sequence[:self.max_frames]
        if self.max_points and sequence.shape[1] > self.max_points:
            sequence = sequence[:, :self.max_points]
        
        # Pad frames if needed
        if self.max_frames and sequence.shape[0] < self.max_frames:
            padding = np.zeros((self.max_frames - sequence.shape[0], 
                              sequence.shape[1], sequence.shape[2]))
            sequence = np.vstack([sequence, padding])
        
        # Pad points if needed
        if self.max_points and sequence.shape[1] < self.max_points:
            padding = np.zeros((sequence.shape[0], 
                              self.max_points - sequence.shape[1], 
                              sequence.shape[2]))
            sequence = np.concatenate([sequence, padding], axis=1)
        
        return {
            'sequence': torch.FloatTensor(sequence),
            'person_id': sample['person_id'],
            'action_id': sample['action_id']
        }


def collate_fn(batch):
    """Custom collate function for batching - minimal CPU work."""
    sequences = torch.stack([item['sequence'] for item in batch])
    person_ids = torch.LongTensor([item['person_id'] for item in batch])
    action_ids = torch.LongTensor([item['action_id'] for item in batch])
    
    return {
        'sequence': sequences,
        'person_id': person_ids,
        'action_id': action_ids
    }


class Trainer:
    """Training manager for mmWave recognition models."""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_person_acc': [],
            'val_person_acc': [],
            'train_action_acc': [],
            'val_action_acc': []
        }
    
    def train_epoch(self, dataloader, optimizer, criterion_person, criterion_action):
        """Train for one epoch."""
        import time
        self.model.train()
        total_loss = 0
        correct_person = 0
        correct_action = 0
        total_samples = 0
        
        pbar = tqdm(dataloader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            # Time the data transfer
            sequences = batch['sequence'].to(self.device)
            person_labels = batch['person_id'].to(self.device)
            action_labels = batch['action_id'].to(self.device)
            torch.cuda.synchronize()  # Ensure transfer is complete
            
            optimizer.zero_grad()
            
            # Forward pass
            person_logits, action_logits = self.model(sequences)
            
            # Compute losses
            loss_person = criterion_person(person_logits, person_labels)
            loss_action = criterion_action(action_logits, action_labels)
            loss = loss_person + loss_action
            
            # Backward pass
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            
            # Statistics
            total_loss += loss.item()
            _, person_pred = person_logits.max(1)
            _, action_pred = action_logits.max(1)
            correct_person += person_pred.eq(person_labels).sum().item()
            correct_action += action_pred.eq(action_labels).sum().item()
            total_samples += sequences.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'person_acc': f'{100.*correct_person/total_samples:.2f}%',
                'action_acc': f'{100.*correct_action/total_samples:.2f}%'
            })
        
        avg_loss = total_loss / len(dataloader)
        person_acc = 100. * correct_person / total_samples
        action_acc = 100. * correct_action / total_samples
        
        return avg_loss, person_acc, action_acc
    
    def validate(self, dataloader, criterion_person, criterion_action):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct_person = 0
        correct_action = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                sequences = batch['sequence'].to(self.device)
                person_labels = batch['person_id'].to(self.device)
                action_labels = batch['action_id'].to(self.device)
                
                # Forward pass
                person_logits, action_logits = self.model(sequences)
                
                # Compute losses
                loss_person = criterion_person(person_logits, person_labels)
                loss_action = criterion_action(action_logits, action_labels)
                loss = loss_person + loss_action
                
                # Statistics
                total_loss += loss.item()
                _, person_pred = person_logits.max(1)
                _, action_pred = action_logits.max(1)
                correct_person += person_pred.eq(person_labels).sum().item()
                correct_action += action_pred.eq(action_labels).sum().item()
                total_samples += sequences.size(0)
        
        avg_loss = total_loss / len(dataloader)
        person_acc = 100. * correct_person / total_samples
        action_acc = 100. * correct_action / total_samples
        
        return avg_loss, person_acc, action_acc
    
    def train(self, train_loader, val_loader, num_epochs, learning_rate=0.001):
        """Full training loop."""
        criterion_person = nn.CrossEntropyLoss()
        criterion_action = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=5)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_person_acc, train_action_acc = self.train_epoch(
                train_loader, optimizer, criterion_person, criterion_action
            )
            
            # Validate
            val_loss, val_person_acc, val_action_acc = self.validate(
                val_loader, criterion_person, criterion_action
            )
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_person_acc'].append(train_person_acc)
            self.history['val_person_acc'].append(val_person_acc)
            self.history['train_action_acc'].append(train_action_acc)
            self.history['val_action_acc'].append(val_action_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train Person Acc: {train_person_acc:.2f}% | Val Person Acc: {val_person_acc:.2f}%")
            print(f"Train Action Acc: {train_action_acc:.2f}% | Val Action Acc: {val_action_acc:.2f}%")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                print("Saved best model!")
        
        return self.history
    
    def plot_history(self, save_path='training_history.png'):
        """Plot training history."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Person accuracy
        axes[1].plot(self.history['train_person_acc'], label='Train')
        axes[1].plot(self.history['val_person_acc'], label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Person Recognition Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        # Action accuracy
        axes[2].plot(self.history['train_action_acc'], label='Train')
        axes[2].plot(self.history['val_action_acc'], label='Validation')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Accuracy (%)')
        axes[2].set_title('Action Recognition Accuracy')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")


def prepare_dataset(data_dirs, person_labels, action_labels, frame_segments=None):
    """
    Prepare dataset structure from directories.
    
    Args:
        data_dirs: List of paths to directories containing frame*.bin files
        person_labels: List of person labels
        action_labels: List of action labels
        frame_segments: List of frame segment strings (e.g., '1-7', '8-15') or None
    
    Returns:
        data_structure, person_encoder, action_encoder
    """
    # Encode labels
    person_encoder = LabelEncoder()
    action_encoder = LabelEncoder()
    
    person_ids = person_encoder.fit_transform(person_labels)
    action_ids = action_encoder.fit_transform(action_labels)
    
    # Create data structure
    data_structure = []
    for i, path in enumerate(data_dirs):
        data_structure.append({
            'path': path,
            'person_id': person_ids[i],
            'action_id': action_ids[i],
            'frame_segment': frame_segments[i] if frame_segments else None
        })
    
    return data_structure, person_encoder, action_encoder


def main():
    """Main training function."""
    
    # Specify which actions to include in training
    #target_actions = {'A11', 'A13', 'A14', 'A17', 'A18'}
    target_actions = {}

    # Scan data directories from your folder structure
    # Define base path to your data folders (adjust this path as needed)
    base_path = Path('t:/Niki/Documents/School')  # Change this to your data root
    csv_path = base_path / 'MMFi_action_segments - MMFi_action_segments.csv'
    
    # Load CSV to get frame segment information
    df = pd.read_csv(csv_path, header=None)
    
    # Parse CSV to build frame segment mapping
    # Format: "E##,S##,A##,segments"
    segments_map = {}
    for idx, row in df.iterrows():
        first_col = row[0]
        if isinstance(first_col, str) and first_col.startswith('E'):
            parts = first_col.split(',')
            if len(parts) >= 4:
                episode, session, action = parts[0], parts[1], parts[2]
                key = f'{episode}_{session}_{action}'
                
                # Extract frame segments from remaining columns
                segments = []
                for col_idx in range(3, len(row)):
                    if pd.notna(row[col_idx]):
                        seg_str = str(row[col_idx]).strip()
                        if seg_str and seg_str != '':
                            segments.append(seg_str)
                
                segments_map[key] = segments
    
    # Collect all directories containing frame*.bin files
    data_dirs = []
    person_labels = []
    action_labels = []
    frame_segments = []
    
    # Scan for directories matching pattern E*/S*/A*/
    for episode_dir in sorted(base_path.glob('E*/S*/A*')):
        # Extract labels from path structure
        parts = episode_dir.parts
        episode = parts[-3]  # E01, E02, etc.
        session = parts[-2]   # S01, S02, etc.
        action = parts[-1]    # A01, A02, etc.
        
        # Skip if action is not in target list (only if target_actions is not empty)
        if target_actions and action not in target_actions:
            continue
        
        # Check if directory contains frame*.bin files in mmwave subdirectory
        mmwave_dir = episode_dir / 'mmwave'
        if mmwave_dir.exists() and any(mmwave_dir.glob('frame*.bin')):
            key = f'{episode}_{session}_{action}'
            
            # Get frame segments for this action from CSV
            if key in segments_map:
                # Use full action (not individual segments) to preserve action context
                # Each segment becomes a different instance but loads the full action
                for segment in segments_map[key]:
                    data_dirs.append(str(mmwave_dir))
                    person_labels.append(episode)  # Use only episode as person identifier
                    action_labels.append(action)
                    frame_segments.append(None)  # Don't use frame segmentation
            else:
                # If not in CSV, treat entire action folder as one sample
                data_dirs.append(str(mmwave_dir))
                person_labels.append(episode)
                action_labels.append(action)
                frame_segments.append(None)  # No specific segment
    
    if not data_dirs:
        print(f"No data directories found at {base_path}")
        print("Please ensure your data path is correct and contains frame*.bin files")
        return
    
    print("Preparing dataset...")
    print(f"Sample paths being loaded:")
    for i, (path, seg) in enumerate(zip(data_dirs[:3], frame_segments[:3])):  # Show first 3
        print(f"  {i}: {path} | Segment: {seg}")
    
    data_structure, person_encoder, action_encoder = prepare_dataset(
        data_dirs, person_labels, action_labels, frame_segments
    )
    
    # Save label encoders
    np.save('person_encoder_classes.npy', person_encoder.classes_)
    np.save('action_encoder_classes.npy', action_encoder.classes_)
    
    print(f"Number of persons: {len(person_encoder.classes_)}")
    print(f"Number of actions: {len(action_encoder.classes_)}")
    print(f"Total samples: {len(data_structure)}")
    print(f"Person classes: {person_encoder.classes_}")
    print(f"Action classes: {action_encoder.classes_}")
    
    # For demonstration with single sample, we'll duplicate it
    # In practice, you should have multiple samples for training
    if len(data_structure) == 1:
        print("\nWarning: Only one sample available. Duplicating for demonstration.")
        print("Please add more data samples for proper training!")
        data_structure = data_structure * 10  # Duplicate for demo
    
    # Split into train and validation
    train_data, val_data = train_test_split(data_structure, test_size=0.2, 
                                            random_state=42)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Create datasets
    # Use 128 frames for good balance between computation and I/O
    train_dataset = MMWaveTorchDataset(train_data, max_frames=128, max_points=150, preload=False, device='cpu')
    val_dataset = MMWaveTorchDataset(val_data, max_frames=128, max_points=150, preload=False, device='cpu')
    
    # Debug: Check a sample
    print("\nDebug: Checking first sample...")
    sample = train_dataset[0]
    print(f"  Sample shape: {sample['sequence'].shape}")
    print(f"  Sample min/max: {sample['sequence'].min():.4f} / {sample['sequence'].max():.4f}")
    print(f"  Sample contains NaN: {torch.isnan(sample['sequence']).any()}")
    print(f"  Person ID: {sample['person_id']}, Action ID: {sample['action_id']}")
    
    # Create dataloaders
    # Windows doesn't support num_workers well, so use 0
    # Use larger batch size to maximize GPU utilization
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, 
                             collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,
                           collate_fn=collate_fn, num_workers=0, pin_memory=True)
    
    # Create model
    num_persons = len(person_encoder.classes_)
    num_actions = len(action_encoder.classes_)
    
    print("\nInitializing model...")
    model = MMWavePointNet(
        num_persons=num_persons,
        num_actions=num_actions,
        input_features=4,
        hidden_dim=512
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    trainer = Trainer(model, device=device)
    
    # Train
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        learning_rate=0.001
    )
    
    # Plot results
    trainer.plot_history()
    
    # Save final model
    torch.save(model.state_dict(), 'final_model.pth')
    
    # Save model config
    config = {
        'num_persons': num_persons,
        'num_actions': num_actions,
        'input_features': 4,
        'hidden_dim': 512,
        'max_frames': 128,
        'max_points': 150
    }
    
    with open('model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\nTraining complete!")
    print("Saved files:")
    print("  - best_model.pth (best validation model)")
    print("  - final_model.pth (final model)")
    print("  - model_config.json (model configuration)")
    print("  - person_encoder_classes.npy (person label mapping)")
    print("  - action_encoder_classes.npy (action label mapping)")
    print("  - training_history.png (training plots)")


if __name__ == '__main__':
    main()

