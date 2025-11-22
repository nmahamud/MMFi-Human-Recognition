"""
Inference script for mmWave human activity and person recognition.
"""

import numpy as np
import torch
import json
from pathlib import Path

from data_loader import MMWaveDataLoader
from model import MMWavePointNet


class MMWavePredictor:
    """Predictor for mmWave recognition."""
    
    def __init__(self, model_path='best_model.pth', config_path='model_config.json',
                 person_classes_path='person_encoder_classes.npy',
                 action_classes_path='action_encoder_classes.npy',
                 device=None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model weights
            config_path: Path to model configuration
            person_classes_path: Path to person label encoder classes
            action_classes_path: Path to action label encoder classes
            device: Device to use (cuda/cpu)
        """
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load label encoders
        self.person_classes = np.load(person_classes_path, allow_pickle=True)
        self.action_classes = np.load(action_classes_path, allow_pickle=True)
        
        # Initialize model
        self.model = MMWavePointNet(
            num_persons=self.config['num_persons'],
            num_actions=self.config['num_actions'],
            input_features=self.config['input_features'],
            hidden_dim=self.config['hidden_dim']
        )
        
        # Load weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Data loader
        self.loader = MMWaveDataLoader(".")
        
        print(f"Model loaded successfully!")
        print(f"Persons: {list(self.person_classes)}")
        print(f"Actions: {list(self.action_classes)}")
    
    def preprocess_sequence(self, sequence):
        """
        Preprocess sequence to match training format.
        
        Args:
            sequence: numpy array (num_frames, num_points, features)
            
        Returns:
            Preprocessed tensor ready for model
        """
        max_frames = self.config['max_frames']
        max_points = self.config['max_points']
        
        # Limit frames and points
        if sequence.shape[0] > max_frames:
            sequence = sequence[:max_frames]
        if sequence.shape[1] > max_points:
            sequence = sequence[:, :max_points]
        
        # Pad frames if needed
        if sequence.shape[0] < max_frames:
            padding = np.zeros((max_frames - sequence.shape[0], 
                              sequence.shape[1], sequence.shape[2]))
            sequence = np.vstack([sequence, padding])
        
        # Pad points if needed
        if sequence.shape[1] < max_points:
            padding = np.zeros((sequence.shape[0], 
                              max_points - sequence.shape[1], 
                              sequence.shape[2]))
            sequence = np.concatenate([sequence, padding], axis=1)
        
        # Convert to tensor and add batch dimension
        tensor = torch.FloatTensor(sequence).unsqueeze(0)  # (1, frames, points, features)
        
        return tensor
    
    def predict(self, data_path):
        """
        Make prediction on a data directory.
        
        Args:
            data_path: Path to directory containing frame*.bin files
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        # Load sequence
        sequence = self.loader.load_sequence(data_path)
        
        # Preprocess
        input_tensor = self.preprocess_sequence(sequence).to(self.device)
        
        # Predict
        with torch.no_grad():
            person_logits, action_logits = self.model(input_tensor)
            
            # Get probabilities
            person_probs = torch.softmax(person_logits, dim=1).cpu().numpy()[0]
            action_probs = torch.softmax(action_logits, dim=1).cpu().numpy()[0]
            
            # Get predictions
            person_idx = person_probs.argmax()
            action_idx = action_probs.argmax()
        
        # Format results
        results = {
            'person': {
                'label': self.person_classes[person_idx],
                'confidence': float(person_probs[person_idx]),
                'all_probabilities': {
                    self.person_classes[i]: float(person_probs[i])
                    for i in range(len(self.person_classes))
                }
            },
            'action': {
                'label': self.action_classes[action_idx],
                'confidence': float(action_probs[action_idx]),
                'all_probabilities': {
                    self.action_classes[i]: float(action_probs[i])
                    for i in range(len(self.action_classes))
                }
            }
        }
        
        return results
    
    def predict_live(self, sequence_array):
        """
        Make prediction on a numpy array directly.
        
        Args:
            sequence_array: numpy array (num_frames, num_points, features)
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        # Preprocess
        input_tensor = self.preprocess_sequence(sequence_array).to(self.device)
        
        # Predict
        with torch.no_grad():
            person_logits, action_logits = self.model(input_tensor)
            
            # Get probabilities
            person_probs = torch.softmax(person_logits, dim=1).cpu().numpy()[0]
            action_probs = torch.softmax(action_logits, dim=1).cpu().numpy()[0]
            
            # Get predictions
            person_idx = person_probs.argmax()
            action_idx = action_probs.argmax()
        
        # Format results
        results = {
            'person': {
                'label': self.person_classes[person_idx],
                'confidence': float(person_probs[person_idx]),
                'all_probabilities': {
                    self.person_classes[i]: float(person_probs[i])
                    for i in range(len(self.person_classes))
                }
            },
            'action': {
                'label': self.action_classes[action_idx],
                'confidence': float(action_probs[action_idx]),
                'all_probabilities': {
                    self.action_classes[i]: float(action_probs[i])
                    for i in range(len(self.action_classes))
                }
            }
        }
        
        return results


def main():
    """Example usage."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_data_directory> [model_path] [config_path]")
        print("\nExamples:")
        print("  python inference.py mmwave")
        print("  python inference.py mmwave best_model.pth model_config.json")
        print("  python inference.py mmwave demo_model.pth demo_config.json")
        return
    
    data_path = sys.argv[1]
    
    if not Path(data_path).exists():
        print(f"Error: Path {data_path} does not exist!")
        return
    
    # Check for model files
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    config_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Auto-detect available model files
    if model_path is None:
        if Path('best_model.pth').exists():
            model_path = 'best_model.pth'
            config_path = 'model_config.json'
        elif Path('demo_model.pth').exists():
            model_path = 'demo_model.pth'
            config_path = 'demo_config.json'
        else:
            print("Error: No model files found!")
            print("Please train a model first using: python train.py")
            print("Or run the demo: python demo_single_sample.py")
            return
    
    print(f"Loading model from: {model_path}")
    print(f"Loading config from: {config_path}")
    
    predictor = MMWavePredictor(
        model_path=model_path,
        config_path=config_path,
        person_classes_path=model_path.replace('_model.pth', '_person_classes.npy'),
        action_classes_path=model_path.replace('_model.pth', '_action_classes.npy')
    )
    
    print(f"\nMaking prediction on: {data_path}")
    results = predictor.predict(data_path)
    
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    print(f"\nPerson: {results['person']['label']}")
    print(f"Confidence: {results['person']['confidence']*100:.2f}%")
    print("\nAll person probabilities:")
    for label, prob in results['person']['all_probabilities'].items():
        print(f"  {label}: {prob*100:.2f}%")
    
    print(f"\nAction: {results['action']['label']}")
    print(f"Confidence: {results['action']['confidence']*100:.2f}%")
    print("\nAll action probabilities:")
    for label, prob in results['action']['all_probabilities'].items():
        print(f"  {label}: {prob*100:.2f}%")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()

