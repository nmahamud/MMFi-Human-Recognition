"""
Visualize the model architecture and create diagrams.
"""

import torch
from model import MMWavePointNet
import json


def print_model_architecture():
    """Print a detailed model architecture."""
    
    print("="*80)
    print("MMWAVE POINTNET ARCHITECTURE")
    print("="*80)
    
    # Load config if available
    try:
        with open('demo_config.json', 'r') as f:
            config = json.load(f)
    except:
        config = {
            'num_persons': 1,
            'num_actions': 1,
            'input_features': 4,
            'hidden_dim': 64
        }
    
    model = MMWavePointNet(
        num_persons=config['num_persons'],
        num_actions=config['num_actions'],
        input_features=config['input_features'],
        hidden_dim=config['hidden_dim']
    )
    
    print("\n[INPUT]")
    print("  Shape: (batch_size, num_frames, num_points, 4)")
    print("  Features per point: [x, y, z, intensity/doppler]")
    
    print("\n[FRAME PROCESSING] (Per-Frame Feature Extraction)")
    print("  For each frame:")
    print("    1. Point-wise Convolution Layers:")
    print("       Conv1D(4 -> 64)  + BatchNorm + ReLU")
    print("       Conv1D(64 -> 128) + BatchNorm + ReLU")
    print("       Conv1D(128 -> 256) + BatchNorm + ReLU")
    print("    2. Global Max Pooling: (batch, 256, num_points) -> (batch, 256)")
    print("    3. Global Feature Network:")
    print(f"       Linear(256 -> {config['hidden_dim']}) + BatchNorm + ReLU + Dropout(0.3)")
    
    print(f"\n[TEMPORAL PROCESSING]")
    print(f"  Stack frame features: (batch, num_frames, {config['hidden_dim']})")
    print(f"  Bidirectional LSTM:")
    print(f"    - Input size: {config['hidden_dim']}")
    print(f"    - Hidden size: {config['hidden_dim']}")
    print(f"    - Num layers: 2")
    print(f"    - Dropout: 0.3")
    print(f"    - Output: (batch, num_frames, {config['hidden_dim']*2})")
    print(f"  Take last time step: (batch, {config['hidden_dim']*2})")
    
    print(f"\n[PERSON CLASSIFICATION HEAD]")
    print(f"  Linear({config['hidden_dim']*2} -> {config['hidden_dim']}) + ReLU + Dropout(0.3)")
    print(f"  Linear({config['hidden_dim']} -> {config['num_persons']})")
    print(f"  Output: Person logits (batch, {config['num_persons']})")
    
    print(f"\n[ACTION CLASSIFICATION HEAD]")
    print(f"  Linear({config['hidden_dim']*2} -> {config['hidden_dim']}) + ReLU + Dropout(0.3)")
    print(f"  Linear({config['hidden_dim']} -> {config['num_actions']})")
    print(f"  Output: Action logits (batch, {config['num_actions']})")
    
    print("\n[OUTPUT]")
    print("  Person prediction: Softmax(person_logits)")
    print("  Action prediction: Softmax(action_logits)")
    
    # Count parameters by section
    print("\n" + "="*80)
    print("PARAMETER COUNT")
    print("="*80)
    
    point_feat_params = sum(p.numel() for p in model.point_feat.parameters())
    global_feat_params = sum(p.numel() for p in model.global_feat.parameters())
    lstm_params = sum(p.numel() for p in model.lstm.parameters())
    person_params = sum(p.numel() for p in model.person_classifier.parameters())
    action_params = sum(p.numel() for p in model.action_classifier.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Point-wise Feature Extractor:  {point_feat_params:>10,} parameters")
    print(f"Global Feature Network:        {global_feat_params:>10,} parameters")
    print(f"LSTM (Temporal):               {lstm_params:>10,} parameters")
    print(f"Person Classifier:             {person_params:>10,} parameters")
    print(f"Action Classifier:             {action_params:>10,} parameters")
    print(f"{'-'*80}")
    print(f"Total:                         {total_params:>10,} parameters")
    
    # Model summary
    print("\n" + "="*80)
    print("DETAILED MODEL STRUCTURE")
    print("="*80)
    print(model)
    
    # Save to file
    with open('model_architecture.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("MMWAVE POINTNET ARCHITECTURE\n")
        f.write("="*80 + "\n")
        f.write(str(model))
        f.write(f"\n\nTotal Parameters: {total_params:,}\n")
    
    print("\n" + "="*80)
    print(f"Architecture details saved to: model_architecture.txt")
    print("="*80)


def visualize_data_flow():
    """Create a text-based data flow diagram."""
    
    print("\n" + "="*80)
    print("DATA FLOW DIAGRAM")
    print("="*80)
    
    diagram = """
    
INPUT: mmWave Radar Frames
|
|  Shape: (batch, 297 frames, ~100 points, 4 features)
|  Each point: [x, y, z, intensity]
|
+-----------------------------------------------------+
|                                                     |
v                                                     v
FRAME 1                  ...                    FRAME 297
(~100 points, 4 features)                       (~100 points, 4 features)
|                                               |
|  [Point-wise Processing]                     |  [Point-wise Processing]
|  +-----------------+                          |  +-----------------+
|  | Conv1D(4->64)   |                          |  | Conv1D(4->64)   |
|  | Conv1D(64->128) |                          |  | Conv1D(64->128) |
|  | Conv1D(128->256)|                          |  | Conv1D(128->256)|
|  +-----------------+                          |  +-----------------+
|           |                                   |           |
|  [Global Max Pooling]                         |  [Global Max Pooling]
|           |                                   |           |
|    (256 features)                             |    (256 features)
|           |                                   |           |
|  [Dense Layer]                                |  [Dense Layer]
|           |                                   |           |
|    (128 features)                             |    (128 features)
|           |                                   |           |
+-----------+-----------------------------------+-----------+
            |
            |  Frame Features: (batch, 297, 128)
            |
            v
    +-------------------+
    |  Bidirectional    |
    |  LSTM (2 layers)  |
    |                   |
    |  Hidden: 128      |
    |  Output: 256      |
    +-------------------+
            |
            |  Take last time step: (batch, 256)
            |
            +----------------------------+-----------------------------+
            |                            |                             |
            v                            v                             v
    +--------------+           +--------------+           +--------------+
    |    Shared    |           |   Person     |           |   Action     |
    |  Features    |---------> |  Classifier  |           |  Classifier  |
    |  (256 dim)   |           |              |           |              |
    +--------------+           |  Dense(128)  |           |  Dense(128)  |
                               |  Dense(N)    |           |  Dense(M)    |
                               +--------------+           +--------------+
                                      |                          |
                                      v                          v
                               Person Prediction         Action Prediction
                               (e.g., "Person1")        (e.g., "A11")
                               Confidence: 95.2%         Confidence: 98.4%

KEY FEATURES:
  + Handles variable number of points per frame
  + Permutation invariant (point order doesn't matter)
  + Temporal modeling with LSTM
  + Multi-task learning (person + action)
  + End-to-end trainable

"""
    
    print(diagram)
    
    with open('data_flow_diagram.txt', 'w') as f:
        f.write(diagram)
    
    print("="*80)
    print("Data flow diagram saved to: data_flow_diagram.txt")
    print("="*80)


def main():
    print_model_architecture()
    visualize_data_flow()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nThe model uses a PointNet-inspired architecture that:")
    print("  1. Processes each frame's point cloud independently")
    print("  2. Extracts spatial features using point-wise convolutions")
    print("  3. Aggregates features across points using max pooling")
    print("  4. Models temporal dynamics using bidirectional LSTM")
    print("  5. Classifies both person and action simultaneously")
    print("\nThis design is well-suited for mmWave radar data because:")
    print("  - It handles variable numbers of points per frame")
    print("  - It's robust to point ordering")
    print("  - It captures both spatial and temporal patterns")
    print("  - It learns task-specific features end-to-end")


if __name__ == '__main__':
    main()

