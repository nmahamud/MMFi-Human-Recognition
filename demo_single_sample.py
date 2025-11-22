"""
Demo script for training and testing with a single sample.
This demonstrates the complete workflow even with limited data.
"""

import os
import numpy as np
import torch
from data_loader import MMWaveDataLoader
from model import MMWavePointNet
import json

def main():
    print("="*70)
    print("MM-WAVE HUMAN RECOGNITION - DEMO")
    print("="*70)
    
    # Step 1: Load sample data
    print("\n[1/5] Loading sample data...")
    loader = MMWaveDataLoader(".")
    
    try:
        sequence = loader.load_sequence("mmwave")
        print(f"[OK] Loaded sequence: {sequence.shape}")
        print(f"  - Frames: {sequence.shape[0]}")
        print(f"  - Max points per frame: {sequence.shape[1]}")
        print(f"  - Features per point: {sequence.shape[2]}")
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Create a simple model
    print("\n[2/5] Creating model...")
    
    # For demo purposes with single sample
    num_persons = 1  # Person1
    num_actions = 1  # A11
    
    model = MMWavePointNet(
        num_persons=num_persons,
        num_actions=num_actions,
        input_features=4,
        hidden_dim=64  # Smaller for demo
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Model created with {num_params:,} parameters")
    
    # Step 3: Test forward pass
    print("\n[3/5] Testing model forward pass...")
    
    # Prepare input (limit size for demo)
    max_frames = min(100, sequence.shape[0])
    max_points = min(100, sequence.shape[1])
    
    input_seq = sequence[:max_frames, :max_points, :]
    
    # Pad if needed
    if input_seq.shape[0] < max_frames:
        padding = np.zeros((max_frames - input_seq.shape[0], 
                          input_seq.shape[1], input_seq.shape[2]))
        input_seq = np.vstack([input_seq, padding])
    
    if input_seq.shape[1] < max_points:
        padding = np.zeros((input_seq.shape[0], 
                          max_points - input_seq.shape[1], 
                          input_seq.shape[2]))
        input_seq = np.concatenate([input_seq, padding], axis=1)
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.FloatTensor(input_seq).unsqueeze(0)
    print(f"  Input shape: {input_tensor.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        person_logits, action_logits = model(input_tensor)
    
    print(f"[OK] Forward pass successful!")
    print(f"  Person logits: {person_logits.shape}")
    print(f"  Action logits: {action_logits.shape}")
    
    # Step 4: Demonstrate inference
    print("\n[4/5] Demonstrating inference...")
    
    person_probs = torch.softmax(person_logits, dim=1).numpy()[0]
    action_probs = torch.softmax(action_logits, dim=1).numpy()[0]
    
    print(f"  Person prediction: Person1 (confidence: {person_probs[0]*100:.2f}%)")
    print(f"  Action prediction: A11 (confidence: {action_probs[0]*100:.2f}%)")
    
    # Step 5: Save demo model
    print("\n[5/5] Saving demo model...")
    
    # Save model
    torch.save(model.state_dict(), 'demo_model.pth')
    
    # Save config
    config = {
        'num_persons': num_persons,
        'num_actions': num_actions,
        'input_features': 4,
        'hidden_dim': 64,
        'max_frames': max_frames,
        'max_points': max_points
    }
    
    with open('demo_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save label encoders
    np.save('demo_person_classes.npy', np.array(['Person1']))
    np.save('demo_action_classes.npy', np.array(['A11']))
    
    print(f"[OK] Demo model saved:")
    print(f"  - demo_model.pth")
    print(f"  - demo_config.json")
    print(f"  - demo_person_classes.npy")
    print(f"  - demo_action_classes.npy")
    
    # Summary
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\nThe model architecture is working correctly!")
    print("\nTo train a real model with multiple samples:")
    print("1. Organize your data into separate directories for each person-action pair")
    print("2. Update the data_dirs, person_labels, and action_labels in train.py")
    print("3. Run: python train.py")
    print("\nFor more information, see README.md")
    
    return True


if __name__ == '__main__':
    success = main()
    if success:
        exit(0)
    else:
        exit(1)

