"""
Data loader for MM-Fi mmWave sensor data.
Handles loading and preprocessing of binary radar frames.
"""

import numpy as np
import os
from pathlib import Path
from typing import Tuple, List, Dict
import struct


class MMWaveDataLoader:
    """Load and preprocess mmWave radar data from binary files."""
    
    def __init__(self, data_root: str, normalize: bool = True):
        """
        Initialize the data loader.
        
        Args:
            data_root: Root directory containing mmWave data
            normalize: Whether to normalize point cloud features
        """
        self.data_root = Path(data_root)
        self.normalize = normalize
        
        # Normalization statistics (can be computed from data or use defaults)
        # These are reasonable defaults for mmWave radar data in meters
        # You can update these after running compute_normalization_stats()
        self.feature_means = np.array([0.0, 0.0, 0.0, 0.0])  # x, y, z, intensity
        self.feature_stds = np.array([1.0, 1.0, 1.0, 1.0])   # will be updated
        self.stats_computed = False
    
    def compute_normalization_stats(self, frame_dirs: list, max_samples: int = 1000):
        """
        Compute normalization statistics from a sample of the dataset.
        
        Args:
            frame_dirs: List of directories containing frame*.bin files
            max_samples: Maximum number of frames to sample for statistics
        """
        all_points = []
        sample_count = 0
        
        for frame_dir in frame_dirs:
            frame_dir = Path(frame_dir)
            frame_files = sorted(frame_dir.glob('frame*.bin'))
            
            for frame_file in frame_files:
                if sample_count >= max_samples:
                    break
                    
                # Load without normalization for stats computation
                old_normalize = self.normalize
                self.normalize = False
                points = self.load_frame(str(frame_file))
                self.normalize = old_normalize
                
                if len(points) > 0:
                    all_points.append(points)
                    sample_count += 1
            
            if sample_count >= max_samples:
                break
        
        if all_points:
            all_points = np.vstack(all_points)
            self.feature_means = np.mean(all_points, axis=0)
            self.feature_stds = np.std(all_points, axis=0)
            # Avoid division by zero
            self.feature_stds = np.where(self.feature_stds < 1e-6, 1.0, self.feature_stds)
            self.stats_computed = True
            
            print(f"Normalization statistics computed from {sample_count} frames:")
            print(f"  Means: {self.feature_means}")
            print(f"  Stds:  {self.feature_stds}")
        
        return self.feature_means, self.feature_stds
    
    def set_normalization_stats(self, means: np.ndarray, stds: np.ndarray):
        """
        Manually set normalization statistics.
        
        Args:
            means: Mean values for each feature [x, y, z, intensity]
            stds: Standard deviation for each feature
        """
        self.feature_means = np.array(means)
        self.feature_stds = np.array(stds)
        self.feature_stds = np.where(self.feature_stds < 1e-6, 1.0, self.feature_stds)
        self.stats_computed = True
    
    def normalize_points(self, points: np.ndarray) -> np.ndarray:
        """
        Normalize point cloud features to zero mean and unit variance.
        
        Args:
            points: Point cloud data (num_points, features)
            
        Returns:
            Normalized points
        """
        if len(points) == 0:
            return points
        
        # Apply z-score normalization: (x - mean) / std
        normalized = (points - self.feature_means) / self.feature_stds
        
        return normalized
        
    def load_frame(self, frame_path: str) -> np.ndarray:
        """
        Load a single frame from binary file.
        
        The MM-Fi dataset typically stores point cloud data with format:
        Each point: [x, y, z, doppler] or [x, y, z, intensity]
        
        Args:
            frame_path: Path to the binary frame file
            
        Returns:
            numpy array of shape (num_points, features)
        """
        try:
            # Read binary file
            with open(frame_path, 'rb') as f:
                data = f.read()
            
            # Try to parse as float32 values (common format)
            # Typically: x, y, z, doppler/intensity per point
            num_floats = len(data) // 4
            
            if num_floats == 0:
                return np.zeros((0, 4))
                
            values = struct.unpack(f'{num_floats}f', data)
            points_array = np.array(values)
            
            # Try different common point cloud formats
            # MM-Fi dataset can have different formats per frame
            points = None
            for n_features in [4, 5, 6, 3]:  # Try common formats
                if num_floats % n_features == 0:
                    points = points_array.reshape(-1, n_features)
                    # Ensure we have exactly 4 features (x, y, z, intensity)
                    if n_features > 4:
                        points = points[:, :4]  # Take first 4 columns
                    elif n_features < 4:
                        # Pad with zeros to get 4 features
                        padding = np.zeros((points.shape[0], 4 - n_features))
                        points = np.hstack([points, padding])
                    break
            
            if points is None:
                # If no standard format fits, try to make it work
                # Truncate to nearest multiple of 4
                truncated_len = (num_floats // 4) * 4
                if truncated_len > 0:
                    points = points_array[:truncated_len].reshape(-1, 4)
                else:
                    return np.zeros((0, 4))
            
            # Filter out invalid/corrupt points (extremely large values)
            # These can occur due to data corruption or invalid readings
            valid_mask = np.all(np.abs(points) < 1e10, axis=1)
            points = points[valid_mask]
            
            # Replace NaN and Inf values with 0
            points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply normalization if enabled
            if self.normalize and len(points) > 0:
                points = self.normalize_points(points)
            
            return points
            
        except Exception as e:
            print(f"Error loading {frame_path}: {e}")
            return np.zeros((0, 4))
    
    def load_sequence(self, frame_dir: str, frame_segment: str = None) -> np.ndarray:
        """
        Load a sequence of frames from a directory.
        
        Args:
            frame_dir: Directory containing frame*.bin files
            frame_segment: Optional segment string like '1-7' to load only specific frames
            
        Returns:
            numpy array of shape (num_frames, max_points, features)
        """
        frame_dir = Path(frame_dir)
        frame_files = sorted(frame_dir.glob('frame*.bin'))
        
        if not frame_files:
            raise ValueError(f"No frame files found in {frame_dir}")
        
        # If frame_segment is specified, extract the frame range
        frame_indices = None
        if frame_segment:
            try:
                parts = str(frame_segment).split('-')
                if len(parts) == 2:
                    start, end = int(parts[0]), int(parts[1])
                    # Handle reversed ranges by swapping
                    if start > end:
                        start, end = end, start
                    # Convert to 0-indexed
                    frame_indices = set(range(start - 1, end))
            except (ValueError, AttributeError):
                # If parsing fails, load all frames
                pass
        
        # Load all frames or only specified segment
        frames = []
        for i, frame_file in enumerate(frame_files):
            # If we have frame_indices, only load frames in that range
            if frame_indices is not None:
                if i not in frame_indices:
                    continue
            frame_data = self.load_frame(str(frame_file))
            frames.append(frame_data)
        
        if not frames:
            # If no frames match segment, load all frames as fallback
            frames = []
            for frame_file in frame_files:
                frame_data = self.load_frame(str(frame_file))
                frames.append(frame_data)
        
        # Find max number of points across all frames
        max_points = max(len(frame) for frame in frames)
        
        if max_points == 0:
            max_points = 1  # Avoid empty arrays
        
        # Pad frames to same length
        num_features = frames[0].shape[1] if len(frames[0]) > 0 else 4
        padded_frames = []
        
        for frame in frames:
            if len(frame) == 0:
                padded_frame = np.zeros((max_points, num_features))
            else:
                num_points = len(frame)
                if num_points < max_points:
                    # Pad with zeros
                    padding = np.zeros((max_points - num_points, num_features))
                    padded_frame = np.vstack([frame, padding])
                else:
                    padded_frame = frame[:max_points]  # Truncate if needed
                    
            padded_frames.append(padded_frame)
        
        return np.array(padded_frames)
    
    def create_voxel_representation(self, points: np.ndarray, 
                                   grid_size: Tuple[int, int, int] = (32, 32, 32),
                                   spatial_range: Tuple[float, float] = (-3, 3)) -> np.ndarray:
        """
        Convert point cloud to voxel grid representation.
        
        Args:
            points: Point cloud data (num_points, features)
            grid_size: Size of voxel grid (x, y, z)
            spatial_range: Min and max spatial coordinates (meters)
            
        Returns:
            Voxel grid of shape grid_size
        """
        if len(points) == 0:
            return np.zeros(grid_size)
        
        voxel_grid = np.zeros(grid_size)
        
        # Extract spatial coordinates (x, y, z)
        xyz = points[:, :3]
        
        # Normalize to grid coordinates
        min_coord, max_coord = spatial_range
        xyz_normalized = (xyz - min_coord) / (max_coord - min_coord)
        xyz_grid = (xyz_normalized * np.array(grid_size)).astype(int)
        
        # Clip to valid range
        xyz_grid = np.clip(xyz_grid, 0, np.array(grid_size) - 1)
        
        # Fill voxel grid (use intensity or doppler if available)
        for i, (x, y, z) in enumerate(xyz_grid):
            if points.shape[1] > 3:
                voxel_grid[x, y, z] = max(voxel_grid[x, y, z], points[i, 3])
            else:
                voxel_grid[x, y, z] += 1
        
        return voxel_grid


class MMWaveDataset:
    """Dataset handler for organizing multiple samples."""
    
    def __init__(self, data_structure: List[Dict[str, any]]):
        """
        Initialize dataset.
        
        Args:
            data_structure: List of dicts with keys: 'path', 'person_id', 'action_id'
        """
        self.data_structure = data_structure
        self.loader = MMWaveDataLoader(".")
        
    def __len__(self):
        return len(self.data_structure)
    
    def __getitem__(self, idx):
        sample = self.data_structure[idx]
        sequence = self.loader.load_sequence(sample['path'])
        return {
            'sequence': sequence,
            'person_id': sample['person_id'],
            'action_id': sample['action_id'],
            'path': sample['path']
        }

