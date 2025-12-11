import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import your modules
from dataset import MMFiDataset
from model_stage1 import PointNet2Regressor, MPJPELoss
from model_stage2 import STGCNActionClassifier

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS_STAGE1 = 30
EPOCHS_STAGE2 = 100
LR_STAGE1 = 0.001
LR_STAGE2 = 0.001

# Anti-Jitter Smoothing Function
def smooth_predictions(skeletons, window_size=5):
    B, T, V, C = skeletons.shape
    data = skeletons.view(B, T, -1).permute(0, 2, 1) # [B, 51, T]
    pad = window_size // 2
    smoothed = torch.nn.functional.avg_pool1d(
        data, kernel_size=window_size, stride=1, padding=pad, count_include_pad=False
    )
    smoothed = smoothed[:, :, :T]
    return smoothed.permute(0, 2, 1).view(B, T, V, C)

def get_dataloaders(args):
    # 1. Load full CSV
    full_df = pd.read_csv(args.csv_file)
    
    # 2. Split into Train (80%) and Val (20%)
    # Stratify ensures we have equal distribution of actions in both sets
    train_df, val_df = train_test_split(
        full_df, test_size=0.2, random_state=42, stratify=full_df['Action']
    )
    
    # 3. Create Datasets
    train_dataset = MMFiDataset(data_root=args.data_root, csv_file=args.csv_file, dataframe=train_df, points_per_frame=512)
    val_dataset = MMFiDataset(data_root=args.data_root, csv_file=args.csv_file, dataframe=val_df, points_per_frame=512)
    
    # 4. Sampler for Training Only (Handle Class Imbalance)
    train_weights = train_dataset.get_weights()
    train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
    
    # 5. Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, 
        num_workers=NUM_WORKERS, drop_last=True
    )
    
    # Validation loader doesn't need sampler, just shuffle=False
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, drop_last=False
    )
    
    print(f"Data Split: {len(train_dataset)} Training | {len(val_dataset)} Validation")
    return train_loader, val_loader

# -----------------------------------------------------------------------------
# Stage 1 Training Loop (Regression)
# -----------------------------------------------------------------------------
def train_stage1(args):
    print("\n[Stage 1] Training PointNet++ Regressor (mmWave -> Skeleton)...")
    train_loader, val_loader = get_dataloaders(args)
    
    model = PointNet2Regressor(num_joints=17).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR_STAGE1)
    criterion = MPJPELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS_STAGE1):
        # --- TRAINING ---
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS_STAGE1} [Train]")
        
        for mmwave_seq, gt_skel_seq, _ in progress_bar:
            B, T, N, C = mmwave_seq.shape
            inputs = mmwave_seq.view(-1, N, C).to(DEVICE)
            targets = gt_skel_seq.view(-1, 17, 3).to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'MPJPE': f"{loss.item():.4f}"})
            
        # --- VALIDATION ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for mmwave_seq, gt_skel_seq, _ in val_loader:
                B, T, N, C = mmwave_seq.shape
                inputs = mmwave_seq.view(-1, N, C).to(DEVICE)
                targets = gt_skel_seq.view(-1, 17, 3).to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train MPJPE: {avg_train:.4f} | Val MPJPE: {avg_val:.4f}")
        
        # Save Best Model based on VALIDATION score
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "stage1_best.pth")
            print(f"   -> New Best Model Saved! (Val MPJPE: {best_val_loss:.4f})")
            
        torch.save(model.state_dict(), "stage1_latest.pth")

# -----------------------------------------------------------------------------
# Stage 2 Training Loop (Classification)
# -----------------------------------------------------------------------------
def train_stage2(args):
    print("\n[Stage 2] Training ST-GCN Classifier (Skeleton -> Action)...")
    TRAIN_ON_GROUND_TRUTH = False 
    
    train_loader, val_loader = get_dataloaders(args)
    
    # 1. Load Pre-trained Stage 1 Model
    regressor = PointNet2Regressor(num_joints=17).to(DEVICE)
    if not TRAIN_ON_GROUND_TRUTH:
        if os.path.exists("stage1_best.pth"):
            print("-> Loading Best Stage 1 Model (stage1_best.pth)")
            regressor.load_state_dict(torch.load("stage1_best.pth"))
        else:
            print("Error: No Stage 1 checkpoint found!")
            return
        regressor.eval()
        for param in regressor.parameters(): param.requires_grad = False
        
    # 2. Initialize Stage 2
    classifier = STGCNActionClassifier(num_classes=27).to(DEVICE)
    optimizer = optim.Adam(classifier.parameters(), lr=LR_STAGE2)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_acc = 0.0

    for epoch in range(EPOCHS_STAGE2):
        # --- TRAINING ---
        classifier.train()
        train_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS_STAGE2} [Train]")
        for mmwave_seq, gt_skel_seq, labels in progress_bar:
            B, T, N, C = mmwave_seq.shape
            
            if TRAIN_ON_GROUND_TRUTH:
                skeletons = gt_skel_seq.to(DEVICE)
            else:
                with torch.no_grad():
                    mmwave_input = mmwave_seq.to(DEVICE)
                    flat_input = mmwave_input.view(-1, N, C)
                    flat_skel = regressor(flat_input)
                    skeletons = flat_skel.view(B, T, 17, 3)
                    # Apply Smoothing
                    skeletons = smooth_predictions(skeletons, window_size=5)

            # Center/Normalize Skeletons
            center_joint = skeletons[:, :, 0:1, :]
            skeletons_rel = skeletons - center_joint
            
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = classifier(skeletons_rel)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar.set_postfix({'Acc': f"{100 * correct / total:.2f}%"})

        # --- VALIDATION ---
        classifier.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for mmwave_seq, gt_skel_seq, labels in val_loader:
                B, T, N, C = mmwave_seq.shape
                
                if TRAIN_ON_GROUND_TRUTH:
                    skeletons = gt_skel_seq.to(DEVICE)
                else:
                    mmwave_input = mmwave_seq.to(DEVICE)
                    flat_input = mmwave_input.view(-1, N, C)
                    flat_skel = regressor(flat_input)
                    skeletons = flat_skel.view(B, T, 17, 3)
                    skeletons = smooth_predictions(skeletons, window_size=5)

                center_joint = skeletons[:, :, 0:1, :]
                skeletons_rel = skeletons - center_joint
                
                labels = labels.to(DEVICE)
                outputs = classifier(skeletons_rel)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        scheduler.step(avg_val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), "stage2_best.pth")
            print(f"   -> New Best Model Saved! (Val Acc: {best_val_acc:.2f}%)")
            
        torch.save(classifier.state_dict(), "stage2_latest.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2])
    parser.add_argument('--data_root', type=str, default='./MMFi')
    parser.add_argument('--csv_file', type=str, default='MMFi_action_segments_rmA1_2_3_6_len10to30.csv')
    args = parser.parse_args()
    
    if args.stage == 1:
        train_stage1(args)
    elif args.stage == 2:
        train_stage2(args)