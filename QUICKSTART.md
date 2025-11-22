# Quick Start Guide

This guide will help you get started quickly with training a model on your mmWave data.

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Inspect Your Data

First, let's understand your data structure:

```bash
python inspect_data.py mmwave
```

This will show you:
- Number of frames
- Points per frame statistics
- Recommended model parameters
- Visualization plots

**Output files**:
- `data_statistics.png` - Shows points distribution
- `point_cloud_visualization.png` - 3D visualization of sample frames

## Step 3: Test the System (Single Sample Demo)

Run the demo to verify everything works:

```bash
python demo_single_sample.py
```

This will:
- Load your mmwave data
- Create a model
- Test forward pass
- Save a demo model

✅ If this works, your setup is correct!

## Step 4: Organize Your Dataset

If you have multiple data samples, organize them properly:

### Option A: Manual Organization

Create this structure:
```
dataset/
  ├── Person1_A11/
  │   ├── frame001.bin
  │   ├── frame002.bin
  │   └── ...
  ├── Person1_A12/
  │   └── ...
  └── Person2_A11/
      └── ...
```

### Option B: Use the Organization Script

```bash
python organize_data.py \
    --dirs mmwave person2_a11 person1_a12 \
    --persons Person1 Person2 Person1 \
    --actions A11 A11 A12 \
    --output dataset
```

## Step 5: Update Training Configuration

Edit `train.py` and update these lines:

```python
# Option 1: Manual specification
data_dirs = [
    'dataset/Person1_A11',
    'dataset/Person1_A12',
    'dataset/Person2_A11',
    # ... add more
]
person_labels = ['Person1', 'Person1', 'Person2', ...]
action_labels = ['A11', 'A12', 'A11', ...]

# Option 2: Load from dataset structure (if you used organize_data.py)
import json
with open('dataset/dataset_structure.json', 'r') as f:
    dataset_structure = json.load(f)

data_dirs = [item['path'] for item in dataset_structure]
person_labels = [item['person'] for item in dataset_structure]
action_labels = [item['action'] for item in dataset_structure]
```

## Step 6: Train Your Model

```bash
python train.py
```

**Training will**:
- Split data into train/validation (80/20)
- Train for 50 epochs
- Save the best model based on validation loss
- Generate training plots

**Output files**:
- `best_model.pth` - Best model weights
- `final_model.pth` - Final model after all epochs
- `model_config.json` - Model configuration
- `person_encoder_classes.npy` - Person label mapping
- `action_encoder_classes.npy` - Action label mapping
- `training_history.png` - Training curves

⏱️ **Training time**: Depends on data size and hardware
- CPU: ~1-2 minutes per epoch (small dataset)
- GPU: Much faster!

## Step 7: Run Inference

Test your trained model:

```bash
python inference.py mmwave
```

Or test on any other data directory:

```bash
python inference.py path/to/test/data
```

**Example output**:
```
===========================================================
PREDICTION RESULTS
===========================================================

Person: Person1
Confidence: 95.23%

Action: A11
Confidence: 98.45%
===========================================================
```

## Step 8: Use in Your Code

```python
from inference import MMWavePredictor

# Load model
predictor = MMWavePredictor(
    model_path='best_model.pth',
    config_path='model_config.json'
)

# Predict from directory
results = predictor.predict('path/to/data')

# Or predict from numpy array
import numpy as np
sequence = np.load('my_sequence.npy')  # Shape: (frames, points, 4)
results = predictor.predict_live(sequence)

# Use predictions
person = results['person']['label']
action = results['action']['label']
person_conf = results['person']['confidence']
action_conf = results['action']['confidence']

print(f"{person} is doing {action}")
print(f"Confidence: {person_conf:.1%} / {action_conf:.1%}")
```

## Troubleshooting

### "Only one sample available"

You need more data! The current setup only has one sample (mmwave directory). To properly train:

1. Collect more data samples (different persons, different actions)
2. Organize them into separate directories
3. Update train.py with all directories and labels

### "CUDA out of memory"

Reduce batch size or model size:

```python
# In train.py
train_loader = DataLoader(train_dataset, batch_size=2, ...)  # Reduce from 4

# Or reduce model size
model = MMWavePointNet(..., hidden_dim=64)  # Reduce from 128
```

### "Poor accuracy"

This is expected with limited data. To improve:

1. **Collect more data**: At least 10-20 samples per person-action combination
2. **More training**: Increase epochs to 100+
3. **Data augmentation**: Add noise, rotation, scaling
4. **Check data quality**: Use `inspect_data.py` to verify

### Model predictions are always the same

This happens with:
- Only one class in training data
- Insufficient training data
- Model not trained enough

**Solution**: Add more diverse samples to your dataset.

## Next Steps

1. **Collect more data**: The model needs multiple samples of each person doing each action
2. **Experiment with hyperparameters**: Try different learning rates, hidden dimensions
3. **Data augmentation**: Add transformations to increase effective dataset size
4. **Evaluate thoroughly**: Use confusion matrices, per-class accuracy
5. **Deploy**: Integrate the trained model into your application

## Recommended Dataset Size

For good performance:

| Metric | Minimum | Recommended | Excellent |
|--------|---------|-------------|-----------|
| Persons | 2 | 5-10 | 20+ |
| Actions | 2 | 5-15 | 27+ |
| Samples per class | 5 | 20 | 50+ |
| **Total samples** | **20** | **200** | **1000+** |

## Getting More Data

If you're using the MM-Fi dataset, download additional samples:

1. Visit: https://github.com/ybhbingo/MMFi_dataset
2. Download data for multiple persons and actions
3. Extract and organize using `organize_data.py`
4. Train with the full dataset

## Support

- Check `README.md` for detailed documentation
- Review model architecture in `model.py`
- Examine data loading in `data_loader.py`
- See training logic in `train.py`

Good luck with your mmWave recognition project! 🎯

