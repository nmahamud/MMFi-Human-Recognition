"""
Inference script for mmWave human activity and person recognition.

Example Command Line Usage:
---------------------------
1. Auto-detect sample:
   python inference.py

2. Specify a data folder:
   python inference.py "t:/Niki/Documents/School/E01/S01/A01/mmwave"

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
    
    # ==========================================
    # CONFIGURATION
    # ==========================================
    # Base path to data (same as in train.py)
    base_path = Path('t:/Niki/Documents/School')
    
    # Model configuration
    model_file = 'best_model.pth'
    config_file = 'model_config.json'
    
    # Select a specific sample to test (or None to pick first found)
    # Set to None to scan for the first available sample
    # or set like: target_sample = ('E01', 'S01', 'A01') 
    target_sample = None 
    # ==========================================
    
    # Determine data path
    data_path = None
    
    if len(sys.argv) > 1:
        # Use command line argument if provided
        data_path = sys.argv[1]
        if len(sys.argv) > 2:
            model_file = sys.argv[2]
        if len(sys.argv) > 3:
            config_file = sys.argv[3]
    else:
        # Use configuration from file
        print(f"No arguments provided. Using base path: {base_path}")
        
        if target_sample:
            # Construct path from target sample
            episode, subject, action = target_sample
            # Try different depths depending on folder structure
            # Structure from train.py: base_path / E.. / S.. / A.. / mmwave
            potential_path = base_path / episode / subject / action / 'mmwave'
            if potential_path.exists():
                data_path = str(potential_path)
            else:
                # Try without mmwave subdir
                potential_path = base_path / episode / subject / action
                if potential_path.exists():
                    data_path = str(potential_path)
        else:
            # Scan for first available sample
            print("Scanning for available samples...")
            found = False
            # Look for E*/S*/A* structure
            for episode_dir in sorted(base_path.glob('E*')):
                if not episode_dir.is_dir(): continue
                for subject_dir in sorted(episode_dir.glob('S*')):
                    if not subject_dir.is_dir(): continue
                    for action_dir in sorted(subject_dir.glob('A*')):
                        if not action_dir.is_dir(): continue
                        
                        # Check for mmwave subdir
                        mmwave_dir = action_dir / 'mmwave'
                        if mmwave_dir.exists() and any(mmwave_dir.glob('frame*.bin')):
                            data_path = str(mmwave_dir)
                            found = True
                            break
                        
                        # Check if action_dir itself contains frames
                        if any(action_dir.glob('frame*.bin')):
                            data_path = str(action_dir)
                            found = True
                            break
                    if found: break
                if found: break
    
    if not data_path:
        print("Usage: python inference.py <path_to_data_directory> [model_path] [config_path]")
        print("\nOr configure 'base_path' in the script.")
        print(f"Could not find any valid data in {base_path}")
        return
    
    if not Path(data_path).exists():
        print(f"Error: Path {data_path} does not exist!")
        return
    
    # Check for model files
    if not Path(model_file).exists():
        # Try to find it in the same directory as the script
        script_dir = Path(__file__).parent
        if (script_dir / model_file).exists():
            model_file = str(script_dir / model_file)
            config_file = str(script_dir / config_file)
        elif Path('demo_model.pth').exists():
            print("Model not found, falling back to demo model...")
            model_file = 'demo_model.pth'
            config_file = 'demo_config.json'
        else:
            print(f"Error: Model file {model_file} not found!")
            print("Please train a model first using: python train.py")
            return
            
    model_path = model_file
    config_path = config_file
    
    print(f"Loading model from: {model_path}")
    print(f"Loading config from: {config_path}")
    
    # Set class file paths
    if 'best_model.pth' in model_path or 'final_model.pth' in model_path:
        person_classes_path = 'person_encoder_classes.npy'
        action_classes_path = 'action_encoder_classes.npy'
    else:
        # Fallback for other naming conventions (like demo_model.pth -> demo_person_classes.npy)
        person_classes_path = model_path.replace('_model.pth', '_person_classes.npy')
        action_classes_path = model_path.replace('_model.pth', '_action_classes.npy')

    predictor = MMWavePredictor(
        model_path=model_path,
        config_path=config_path,
        person_classes_path=person_classes_path,
        action_classes_path=action_classes_path
    )
    
    print(f"\nMaking prediction on: {data_path}")
    
    # Extract ground truth from path if possible
    gt_person = None
    gt_action = None
    path_parts = Path(data_path).parts
    
    # Look for pattern E../S../A..
    for i in range(len(path_parts)):
        part = path_parts[i]
        if part.startswith('E') and part[1:].isdigit():
            # Found Episode
            if i+2 < len(path_parts):
                possible_subject = path_parts[i+1]
                possible_action = path_parts[i+2]
                
                if (possible_subject.startswith('S') and possible_subject[1:].isdigit() and
                    possible_action.startswith('A') and possible_action[1:].isdigit()):
                    
                    # In the updated train.py, we use Subject (Sxx) as the person ID
                    # and Action (Axx) as the action ID
                    gt_person = possible_subject
                    gt_action = possible_action
                    print(f"Ground Truth (from path): Person={gt_person}, Action={gt_action}")
                    break

    results = predictor.predict(data_path)
    
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    # Person Result
    is_person_correct = ""
    if gt_person:
        is_person_correct = " [CORRECT]" if results['person']['label'] == gt_person else f" [WRONG, expected {gt_person}]"
    
    print(f"\nPerson: {results['person']['label']}{is_person_correct}")
    print(f"Confidence: {results['person']['confidence']*100:.2f}%")
    print("\nAll person probabilities:")
    for label, prob in results['person']['all_probabilities'].items():
        print(f"  {label}: {prob*100:.2f}%")
    
    # Action Result
    is_action_correct = ""
    if gt_action:
        is_action_correct = " [CORRECT]" if results['action']['label'] == gt_action else f" [WRONG, expected {gt_action}]"
        
    print(f"\nAction: {results['action']['label']}{is_action_correct}")
    print(f"Confidence: {results['action']['confidence']*100:.2f}%")
    print("\nAll action probabilities:")
    for label, prob in results['action']['all_probabilities'].items():
        print(f"  {label}: {prob*100:.2f}%")
    
    # Save results to file
    output_file = 'prediction_results.json'
    
    # Add metadata to results
    results['data_path'] = data_path
    results['model_path'] = model_path
    
    # Convert numpy types to native python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=convert_to_serializable)
    
    print(f"\nResults saved to: {output_file}")
    
    # Also save a readable text report
    report_file = 'prediction_report.txt'
    with open(report_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"PREDICTION RESULTS (Source: {data_path})\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Person: {results['person']['label']}\n")
        f.write(f"Confidence: {results['person']['confidence']*100:.2f}%\n")
        f.write("\nAll person probabilities:\n")
        for label, prob in results['person']['all_probabilities'].items():
            f.write(f"  {label}: {prob*100:.2f}%\n")
        
        f.write(f"\nAction: {results['action']['label']}\n")
        f.write(f"Confidence: {results['action']['confidence']*100:.2f}%\n")
        f.write("\nAll action probabilities:\n")
        for label, prob in results['action']['all_probabilities'].items():
            f.write(f"  {label}: {prob*100:.2f}%\n")
        f.write("\n" + "="*60 + "\n")
        
    print(f"Report saved to: {report_file}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()

