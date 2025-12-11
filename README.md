# Two-Stage mmWave → Skeleton → Action Recognition Pipeline

This repository implements a **two-stage deep learning pipeline** for human action recognition using **mmWave radar point clouds**.  

- **Stage 1 – PointNet++ Regressor**: Converts raw mmWave point clouds into 3D human skeletons (17 joints × 3 coordinates).  
- **Stage 2 – ST‑GCN Classifier**: Processes skeleton sequences to classify human actions.

---

Dataset located here: https://drive.google.com/file/d/1Ucur9abUMbZIAybi76D1JmG5yXIPi8en/view?usp=sharing

## Project Structure

```
.
├── dataset.py                # Custom PyTorch dataset (MMFiDataset)
├── model_stage1.py           # PointNet++ skeleton regression model
├── model_stage2.py           # ST-GCN action classification model
├── train.py                  # Training script for both stages
├── test.py                   # Evaluation script (MPJPE + accuracy)
├── requirements.txt          # Python dependencies
├── CleanTheCSV.ipynb         # Data cleaning/preprocessing notebook
├── MMFi_action_segments_rmA1_2_3_6_len10to30.csv  # Segment metadata
└── MMFi.zip                  # mmWave dataset folder (unzipped as ./MMFi)
```

---

## Installation

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Dataset

Unzip the dataset archive to the repository root:

```
unzip MMFi.zip -d ./MMFi
```

Each session folder should contain:

```
MMFi/
├── E01/
│   ├── S01/
│   │   ├── A01/
│   │   │   ├── mmwave/frame001.bin ...
│   │   │   └── ground_truth.npy
│   │   └── ...
```

---

## Training

Train each stage sequentially:

```bash
# Stage 1: mmWave → Skeleton
python train.py --stage 1 --data_root ./MMFi --csv_file MMFi_action_segments_rmA1_2_3_6_len10to30.csv

# Stage 2: Skeleton → Action
python train.py --stage 2 --data_root ./MMFi --csv_file MMFi_action_segments_rmA1_2_3_6_len10to30.csv
```

Best model checkpoints are saved as:
```
stage1_best.pth
stage2_best.pth
```

---

## Testing

Evaluate either stage or the full pipeline:

```bash
# Evaluate both stages
python test.py --mode both --data_root ./MMFi --csv_file MMFi_action_segments_rmA1_2_3_6_len10to30.csv     --stage1_ckpt stage1_best.pth --stage2_ckpt stage2_best.pth

# Evaluate Stage‑1 only (MPJPE)
python test.py --mode stage1 --csv_file MMFi_action_segments_rmA1_2_3_6_len10to30.csv

# Evaluate Stage‑2 only using ground‑truth skeletons
python test.py --mode stage2 --use_gt_for_stage2 --csv_file MMFi_action_segments_rmA1_2_3_6_len10to30.csv
```

---

## Outputs

- **Stage 1 metrics:** Mean Per Joint Position Error (MPJPE)  
- **Stage 2 metrics:** Overall & per‑action accuracy  

Checkpoints and logs are stored in the current directory.

---

## Notes

- Works on both CPU and GPU (CUDA auto‑detected).
- Handles unbalanced classes via weighted sampling.
- Compatible with Python 3.10+ and PyTorch 2.x.

 
