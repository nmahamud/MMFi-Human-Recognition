# mmWave Human Activity and Person Recognition

This project implements a deep learning pipeline for recognizing both the person and the action they are performing using mmWave radar sensor data from the MM-Fi dataset.

## Features

- **Dual-Task Learning**: Simultaneously predicts person identity and action type
- **PointNet-based Architecture**: Handles variable-sized point clouds efficiently
- **LSTM Temporal Processing**: Captures temporal dynamics across frames
- **Flexible Data Loading**: Supports MM-Fi binary format
- **Complete Pipeline**: Training, validation, and inference scripts

## Model Architecture

The system uses a **MMWavePointNet** model that combines:

1. **Point-wise Feature Extraction**: Conv1D layers to process individual radar points
2. **Global Feature Aggregation**: Max pooling across spatial points
3. **Temporal Modeling**: Bidirectional LSTM for sequence processing
4. **Dual Classification Heads**: Separate classifiers for person and action

```
Input: (batch, frames, points, features)
  ↓
Point-wise Conv1D (per frame)
  ↓
Max Pooling (spatial aggregation)
  ↓
Bidirectional LSTM (temporal)
  ↓
├─→ Person Classifier
└─→ Action Classifier
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

Requirements:
- Python 3.7+
- PyTorch 1.12+
- NumPy
- scikit-learn
- matplotlib
- tqdm

## Data Format

The system expects mmWave data in binary format (`.bin` files) organized as:

```
dataset/
  ├── person1_action11/
  │   ├── frame001.bin
  │   ├── frame002.bin
  │   └── ...
  ├── person1_action12/
  │   └── ...
  └── person2_action11/
      └── ...
```

Each `.bin` file contains point cloud data with format:
- Each point: `[x, y, z, doppler/intensity]` (4 float32 values)

## Usage

### 1. Prepare Your Dataset

Organize your data into separate directories for each person-action combination. Update the `train.py` script with your data paths:

```python
data_dirs = [
    'mmwave',           # Person1, Action A11
    'person1_a12',      # Person1, Action A12
    'person2_a11',      # Person2, Action A11
    # ... add more
]
person_labels = ['Person1', 'Person1', 'Person2', ...]
action_labels = ['A11', 'A12', 'A11', ...]
```

### 2. Train the Model

```bash
python train.py
```

This will:
- Load and preprocess the data
- Split into train/validation sets
- Train the model for 50 epochs
- Save the best model and training history
- Generate training plots

Output files:
- `best_model.pth` - Best performing model
- `final_model.pth` - Final model after all epochs
- `model_config.json` - Model configuration
- `person_encoder_classes.npy` - Person label mappings
- `action_encoder_classes.npy` - Action label mappings
- `training_history.png` - Training curves

### 3. Run Inference

```bash
python inference.py <path_to_data_directory>
```

Example:
```bash
python inference.py mmwave
```

Output:
```
===========================================================
PREDICTION RESULTS
===========================================================

Person: Person1
Confidence: 95.23%

All person probabilities:
  Person1: 95.23%
  Person2: 4.77%

Action: A11
Confidence: 98.45%

All action probabilities:
  A11: 98.45%
  A12: 1.55%

===========================================================
```

### 4. Use in Your Code

```python
from inference import MMWavePredictor

# Initialize predictor
predictor = MMWavePredictor(
    model_path='best_model.pth',
    config_path='model_config.json'
)

# Predict from directory
results = predictor.predict('path/to/data')

# Or predict from numpy array
import numpy as np
sequence = np.random.randn(100, 50, 4)  # (frames, points, features)
results = predictor.predict_live(sequence)

print(f"Person: {results['person']['label']} "
      f"({results['person']['confidence']*100:.2f}%)")
print(f"Action: {results['action']['label']} "
      f"({results['action']['confidence']*100:.2f}%)")
```

## Model Configuration

Key parameters in `train.py`:

```python
# Data parameters
max_frames = 100        # Maximum frames per sequence
max_points = 50         # Maximum points per frame
input_features = 4      # Features per point (x, y, z, doppler)

# Model parameters
hidden_dim = 128        # Hidden layer dimension
num_persons = N         # Number of unique persons
num_actions = M         # Number of unique actions

# Training parameters
batch_size = 4
num_epochs = 50
learning_rate = 0.001
```

## Project Structure

```
.
├── data_loader.py      # Data loading and preprocessing
├── model.py            # Neural network architectures
├── train.py            # Training script
├── inference.py        # Inference/prediction script
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Customization

### Adding New Architectures

The `model.py` includes an alternative 3D CNN architecture (`MMWave3DCNN`) that works with voxelized representations. To use it:

1. Enable voxelization in the data loader
2. Change the model in `train.py`:

```python
from model import MMWave3DCNN

model = MMWave3DCNN(
    num_persons=num_persons,
    num_actions=num_actions,
    input_channels=1,
    hidden_dim=128
)
```

### Adjusting Hyperparameters

Modify these in `train.py` based on your dataset:

- **Learning rate**: Decrease if loss oscillates, increase if converging too slowly
- **Batch size**: Increase with more GPU memory
- **Hidden dimension**: Increase for more complex patterns
- **Max frames/points**: Adjust based on your data statistics

## Performance Tips

1. **Data Augmentation**: Add random rotations, scaling, or point dropout
2. **Class Balancing**: Use weighted loss if classes are imbalanced
3. **Early Stopping**: Monitor validation loss to prevent overfitting
4. **Ensemble Methods**: Combine multiple models for better accuracy

## Troubleshooting

**Out of Memory Error**:
- Reduce `batch_size`
- Reduce `max_frames` or `max_points`
- Reduce `hidden_dim`

**Poor Accuracy**:
- Collect more training data
- Increase model capacity (`hidden_dim`)
- Train for more epochs
- Check data quality and labels

**Slow Training**:
- Use GPU if available (`cuda`)
- Reduce `max_frames` or `max_points`
- Increase `batch_size` (if memory allows)

## MM-Fi Dataset

This project is designed for the MM-Fi public dataset:
- **Source**: [MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset](https://github.com/ybhbingo/MMFi_dataset)
- **Actions**: 27 activities (A1-A27)
- **Persons**: 40+ participants
- **Modality**: mmWave radar point clouds

## License

This project is provided as-is for research and educational purposes.

## Citation

If you use this code, please cite the MM-Fi dataset:

```bibtex
@inproceedings{mmfi2023,
  title={MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing},
  author={Yang, Jianfei and Huang, He and Zhou, Yunjiao and Chen, Xinyan and Xu, Yuecong and Yuan, Shenghai and Zou, Han and Lu, Chris Xiaoxuan and Xie, Lihua},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023}
}
```

## Contact

For questions or issues, please open an issue on the project repository.

