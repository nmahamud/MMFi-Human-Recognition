# File Index - mmWave Human Recognition System

## Core System Files (Python)

### Data Processing
- **`data_loader.py`** (190 lines)
  - `MMWaveDataLoader` class for loading binary radar frames
  - `MMWaveDataset` class for dataset management
  - Handles variable-sized point clouds and corrupt data
  - Supports voxel representation conversion

### Model Architecture
- **`model.py`** (200 lines)
  - `MMWavePointNet` - Primary model (PointNet + LSTM)
  - `MMWave3DCNN` - Alternative 3D CNN model
  - Dual-task classification (person + action)
  - Configurable hidden dimensions

### Training Pipeline
- **`train.py`** (300 lines)
  - `MMWaveTorchDataset` - PyTorch Dataset wrapper
  - `Trainer` class with training loop
  - Automatic train/validation split
  - Model checkpointing and history plotting
  - Main training function with data preparation

### Inference & Deployment
- **`inference.py`** (250 lines)
  - `MMWavePredictor` class for predictions
  - Load trained models with config
  - Predict from directories or numpy arrays
  - Confidence scores and probabilities
  - Command-line interface

## Utility Scripts (Python)

### Data Analysis
- **`inspect_data.py`** (200 lines)
  - Analyze binary frame files
  - Display statistics (min, max, mean, std)
  - Detect data format (3, 4, 5, or 6 features)
  - Generate visualizations:
    - `data_statistics.png` - Point distribution plots
    - `point_cloud_visualization.png` - 3D scatter plots
  - Provide configuration recommendations

### Demo & Testing
- **`demo_single_sample.py`** (150 lines)
  - Quick system demonstration
  - Load sample data
  - Test model forward pass
  - Save demo model files:
    - `demo_model.pth`
    - `demo_config.json`
    - `demo_person_classes.npy`
    - `demo_action_classes.npy`

### Data Organization
- **`organize_data.py`** (150 lines)
  - Structure dataset for training
  - Copy/organize frame files
  - Create dataset manifest (JSON)
  - Generate training code snippets
  - Command-line interface with examples

### Architecture Visualization
- **`visualize_architecture.py`** (220 lines)
  - Display model architecture
  - Show data flow diagram
  - Count parameters by component
  - Generate text files:
    - `model_architecture.txt` - Detailed model structure
    - `data_flow_diagram.txt` - Visual data flow

## Documentation Files (Markdown)

### User Guides
- **`README.md`** (400 lines)
  - Comprehensive system documentation
  - Installation instructions
  - Model architecture explanation
  - Usage examples and API reference
  - Troubleshooting guide
  - MM-Fi dataset information

- **`QUICKSTART.md`** (300 lines)
  - Step-by-step tutorial
  - Quick start for beginners
  - Common commands and examples
  - Dataset size recommendations
  - Troubleshooting tips

- **`PROJECT_SUMMARY.md`** (500 lines)
  - Project overview and goals
  - Complete file descriptions
  - Current status and limitations
  - Usage workflows
  - Example use cases
  - Next steps and enhancements

- **`FILE_INDEX.md`** (This file)
  - Complete file listing
  - File descriptions and purposes
  - Line counts and organization

## Configuration Files

### Dependencies
- **`requirements.txt`** (5 lines)
  ```
  numpy>=1.21.0
  torch>=1.12.0
  scikit-learn>=1.0.0
  matplotlib>=3.5.0
  tqdm>=4.64.0
  ```

## Generated Files (During Demo)

### Model Files
- **`demo_model.pth`** (968 KB)
  - Demo model weights (untrained)
  - PyTorch state dict format

- **`demo_config.json`** (7 lines)
  ```json
  {
    "num_persons": 1,
    "num_actions": 1,
    "input_features": 4,
    "hidden_dim": 64,
    "max_frames": 100,
    "max_points": 100
  }
  ```

- **`demo_person_classes.npy`** (numpy array)
  - Person label encodings: ['Person1']

- **`demo_action_classes.npy`** (numpy array)
  - Action label encodings: ['A11']

### Visualization Files
- **`data_statistics.png`**
  - Points per frame over time (line plot)
  - Distribution of points per frame (histogram)

- **`point_cloud_visualization.png`**
  - 3D scatter plots of first 3 frames
  - Color-coded by intensity

- **`model_architecture.txt`**
  - Detailed PyTorch model structure
  - Parameter counts

- **`data_flow_diagram.txt`**
  - ASCII art data flow diagram
  - Shows input to output pipeline

## Expected Files After Training

When you run `python train.py`, these files will be created:

- **`best_model.pth`** - Best model based on validation loss
- **`final_model.pth`** - Model after all epochs
- **`model_config.json`** - Training configuration
- **`person_encoder_classes.npy`** - Person labels
- **`action_encoder_classes.npy`** - Action labels
- **`training_history.png`** - Loss and accuracy plots

## Directory Structure

```
MMFi-Human-Recognition/
│
├── Core System (Python)
│   ├── data_loader.py          # Data loading & preprocessing
│   ├── model.py                # Neural network models
│   ├── train.py                # Training pipeline
│   └── inference.py            # Prediction & deployment
│
├── Utilities (Python)
│   ├── inspect_data.py         # Data analysis
│   ├── demo_single_sample.py   # Quick demo
│   ├── organize_data.py        # Dataset organization
│   └── visualize_architecture.py # Model visualization
│
├── Documentation (Markdown)
│   ├── README.md               # Main documentation
│   ├── QUICKSTART.md           # Quick start guide
│   ├── PROJECT_SUMMARY.md      # Project overview
│   └── FILE_INDEX.md           # This file
│
├── Configuration
│   └── requirements.txt        # Python dependencies
│
├── Generated Files (Demo)
│   ├── demo_model.pth          # Demo model weights
│   ├── demo_config.json        # Demo configuration
│   ├── demo_person_classes.npy # Demo person labels
│   ├── demo_action_classes.npy # Demo action labels
│   ├── data_statistics.png     # Data analysis plots
│   ├── point_cloud_visualization.png # 3D visualizations
│   ├── model_architecture.txt  # Model structure
│   └── data_flow_diagram.txt   # Data flow diagram
│
├── Data (Your mmWave Data)
│   └── mmwave/
│       ├── frame001.bin        # 297 binary frame files
│       ├── frame002.bin
│       └── ...
│
└── Runtime (Python Cache)
    └── __pycache__/            # Python bytecode cache

```

## File Size Summary

**Total Python Code**: ~1,660 lines
- Core system: ~940 lines
- Utilities: ~720 lines

**Total Documentation**: ~1,200 lines
- User guides: ~1,200 lines

**Total Size**: ~2,860 lines of code/documentation

## Quick Command Reference

### Analysis & Setup
```bash
python inspect_data.py mmwave              # Analyze data
python demo_single_sample.py               # Run demo
python visualize_architecture.py           # Show architecture
```

### Dataset Organization
```bash
python organize_data.py \
    --dirs mmwave person2_a11 \
    --persons Person1 Person2 \
    --actions A11 A11 \
    --output dataset
```

### Training & Inference
```bash
python train.py                            # Train model
python inference.py mmwave                 # Make predictions
python inference.py mmwave best_model.pth model_config.json
```

## Development Status

### ✅ Completed
- [x] Data loading and preprocessing
- [x] Model architecture implementation
- [x] Training pipeline
- [x] Inference system
- [x] Data analysis tools
- [x] Demo and testing
- [x] Comprehensive documentation
- [x] Visualization tools

### 📋 Pending (User Tasks)
- [ ] Collect more data samples
- [ ] Organize dataset
- [ ] Train model with full dataset
- [ ] Evaluate performance
- [ ] Deploy to production

## Usage Priority

**For First-Time Users:**
1. Read `QUICKSTART.md`
2. Run `python demo_single_sample.py`
3. Run `python inspect_data.py mmwave`
4. Follow steps in `QUICKSTART.md`

**For Development:**
1. Review `model.py` for architecture
2. Review `train.py` for training logic
3. Modify hyperparameters as needed
4. Check `data_loader.py` for data format

**For Deployment:**
1. Use `inference.py` as reference
2. Load model with `MMWavePredictor`
3. Call `predict()` or `predict_live()`
4. Process results dictionary

## Contact & Support

For questions about specific files:
- **Data issues**: Check `data_loader.py` and `inspect_data.py`
- **Model issues**: Check `model.py` and `visualize_architecture.py`
- **Training issues**: Check `train.py` and `QUICKSTART.md`
- **Inference issues**: Check `inference.py` and examples in `README.md`

---

**Last Updated**: Created during initial project setup
**Version**: 1.0
**Python Version**: 3.7+
**PyTorch Version**: 1.12+

