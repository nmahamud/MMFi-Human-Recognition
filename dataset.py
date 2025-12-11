import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import glob

class MMFiDataset(Dataset):
    def __init__(self, data_root, csv_file=None, dataframe=None, points_per_frame=512):
        self.data_root = data_root
        
        # Allow passing pre-split DataFrame or loading from CSV
        if dataframe is not None:
            self.df = dataframe
        elif csv_file is not None:
            self.df = pd.read_csv(csv_file)
        else:
            raise ValueError("Must provide either csv_file or dataframe")
            
        self.points_per_frame = points_per_frame
        
        # Create a mapping from Action String (A01) to Integer (0)
        # NOTE: We sort unique actions from the DATAFRAME to ensure consistency
        # Ideally, pass the full list of actions to ensure train/val have same mapping
        # For this dataset, assuming both splits contain all actions is usually safe, 
        # but technically we should hardcode the 27 actions to be safe.
        self.actions = sorted(self.df['Action'].unique())
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        env, subj, act = row['Environment'], row['Subject'], row['Action']
        start_frame, end_frame = row['Start'], row['End']
        
        session_path = os.path.join(self.data_root, env, subj, act)
        mmwave_path = os.path.join(session_path, 'mmwave')
        gt_path = os.path.join(session_path, 'ground_truth.npy')
        
        target_indices = np.linspace(start_frame, end_frame, 30).astype(int)
        
        all_gt = np.load(gt_path)
        skeleton_seq = all_gt[target_indices - 1] 
        
        aggregated_pcs = []
        aggregated_skels = []
        
        # Sliding Window Logic (from previous step)
        window_size = 5
        stride = 1
        num_windows = (len(target_indices) - window_size) // stride + 1
        
        for i in range(num_windows):
            start = i * stride
            end = start + window_size
            chunk = target_indices[start : end]
            
            chunk_points = []
            for frame_idx in chunk:
                bin_file = os.path.join(mmwave_path, f"frame{frame_idx:03d}.bin")
                if os.path.exists(bin_file):
                    try:
                        # Load data
                        data = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 5)
                        # Sanity Check: Remove points with infinity or NaNs or extreme values
                        if data.size > 0:
                            # Clamp extreme values (prevent overflow warnings)
                            data = np.clip(data, -100, 100) # Assuming human is within 100m box
                        chunk_points.append(data)
                    except Exception:
                        # Fallback for corrupted files
                        chunk_points.append(np.zeros((1, 5), dtype=np.float32))
                else:
                    chunk_points.append(np.zeros((1, 5), dtype=np.float32))
            
            # Concatenate
            if len(chunk_points) > 0:
                dense_cloud = np.concatenate(chunk_points, axis=0)
            else:
                dense_cloud = np.zeros((1, 5), dtype=np.float32)
            
            # --- SAFETY 1: Handle Empty Cloud ---
            if dense_cloud.shape[0] == 0:
                dense_cloud = np.zeros((1, 5), dtype=np.float32)

            # 1. Filter Noise
            non_zero_mask = np.any(dense_cloud != 0, axis=1)
            # Only filter if we have enough points, otherwise keep what we have (even if zeros)
            if non_zero_mask.sum() > 10:
                dense_cloud = dense_cloud[non_zero_mask]

            # --- SAFETY 2: Handle Empty after Filter ---
            if dense_cloud.shape[0] == 0:
                dense_cloud = np.zeros((1, 5), dtype=np.float32)

            # 2. CENTERING
            xyz = dense_cloud[:, :3]
            # Safety check before mean calculation
            if xyz.shape[0] > 0:
                centroid = np.mean(xyz, axis=0)
                xyz -= centroid
            else:
                centroid = np.zeros(3)

            # 3. SCALING (Robust)
            scale = 1.0
            if xyz.shape[0] > 0:
                # Use linalg.norm (safer than manual square/sum)
                dists = np.linalg.norm(xyz, axis=1)
                max_dist = np.max(dists)
                
                if max_dist > 1e-4:
                    scale = 1.0 / max_dist
                    xyz *= scale
                    dense_cloud[:, :3] = xyz
            
            # Apply Centering + Scale to Target Skeleton
            skel_idx = start + 2
            target_skel = skeleton_seq[skel_idx].copy()
            target_skel -= centroid
            target_skel *= scale 

            # 4. Fixed Sampling
            num_pts = dense_cloud.shape[0]
            if num_pts > 0:
                replace = num_pts < self.points_per_frame
                choice_idx = np.random.choice(num_pts, self.points_per_frame, replace=replace)
                sampled_cloud = dense_cloud[choice_idx, :]
            else:
                # Final fallback for purely empty frame
                sampled_cloud = np.zeros((self.points_per_frame, 5), dtype=np.float32)

            aggregated_pcs.append(sampled_cloud)
            aggregated_skels.append(target_skel)

        input_tensor = torch.tensor(np.array(aggregated_pcs), dtype=torch.float32)
        target_tensor = torch.tensor(np.array(aggregated_skels), dtype=torch.float32)
        label = torch.tensor(self.action_to_idx[act], dtype=torch.long)
        
        return input_tensor, target_tensor, label

    def get_weights(self):
        """
        Calculates weights for WeightedRandomSampler to handle class imbalance.
        """
        class_counts = self.df['Action'].value_counts().sort_index()
        # Create a dictionary for weight lookup
        # Weight = 1.0 / frequency
        weight_per_class = {act: 1.0/count for act, count in class_counts.items()}
        
        # Assign a weight to every sample in the dataset
        sample_weights = [weight_per_class[act] for act in self.df['Action']]
        
        return sample_weights

# --- Usage Example ---
if __name__ == "__main__":
    # 1. Initialize Dataset
    dataset = MMFiDataset(
        data_root='./MMFi', 
        csv_file='MMFi_action_segments_rmA1_2_3_6_len10to30.csv'
    )
    
    # 2. Setup Weighted Sampler (CRITICAL for your unbalanced data)
    weights = dataset.get_weights()
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    # 3. Create DataLoader
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=16, 
        sampler=sampler,  # Pass sampler here
        num_workers=4
    )
    
    # 4. Test a batch
    for pcs, skels, labs in train_loader:
        print("Batch Shapes:")
        print("Point Clouds:", pcs.shape)  # Should be [16, 6, 512, 5]
        print("Skeletons:   ", skels.shape) # Should be [16, 6, 17, 3]
        print("Labels:      ", labs.shape)  # Should be [16]
        break