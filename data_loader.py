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
    
    def __init__(self, data_root: str):
        """
        Initialize the data loader.
        
        Args:
            data_root: Root directory containing mmWave data
        """
        self.data_root = Path(data_root)
        
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
            
            return points
            
        except Exception as e:
            print(f"Error loading {frame_path}: {e}")
            return np.zeros((0, 4))
    
    def load_sequence(self, frame_dir: str) -> np.ndarray:
        """
        Load a sequence of frames from a directory.
        
        Args:
            frame_dir: Directory containing frame*.bin files
            
        Returns:
            numpy array of shape (num_frames, max_points, features)
        """
        frame_dir = Path(frame_dir)
        frame_files = sorted(frame_dir.glob('frame*.bin'))
        
        if not frame_files:
            raise ValueError(f"No frame files found in {frame_dir}")
        
        # Load all frames
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

