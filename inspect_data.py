"""
Script to inspect mmWave binary data and understand its structure.
"""

import os
import struct
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def inspect_binary_file(file_path):
    """Inspect a single binary file."""
    with open(file_path, 'rb') as f:
        data = f.read()
    
    file_size = len(data)
    
    print(f"\nFile: {file_path}")
    print(f"Size: {file_size} bytes")
    
    # Try different interpretations
    if file_size % 4 == 0:
        num_floats = file_size // 4
        print(f"Number of float32 values: {num_floats}")
        
        if num_floats > 0:
            values = struct.unpack(f'{num_floats}f', data)
            values_array = np.array(values)
            
            print(f"\nStatistics:")
            print(f"  Min: {values_array.min():.4f}")
            print(f"  Max: {values_array.max():.4f}")
            print(f"  Mean: {values_array.mean():.4f}")
            print(f"  Std: {values_array.std():.4f}")
            
            # Check if divisible by common point formats
            for n_features in [3, 4, 5, 6]:
                if num_floats % n_features == 0:
                    num_points = num_floats // n_features
                    print(f"\nPossible format: {num_points} points × {n_features} features")
                    
                    if n_features == 4:
                        # Assume [x, y, z, doppler/intensity]
                        points = values_array.reshape(-1, 4)
                        print(f"\nAssuming [x, y, z, intensity] format:")
                        print(f"  X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
                        print(f"  Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
                        print(f"  Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
                        print(f"  Intensity range: [{points[:, 3].min():.3f}, {points[:, 3].max():.3f}]")
                        
                        return points
    
    return None


def inspect_dataset(data_dir):
    """Inspect entire dataset."""
    data_dir = Path(data_dir)
    frame_files = sorted(data_dir.glob('frame*.bin'))
    
    if not frame_files:
        print(f"No frame files found in {data_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"DATASET INSPECTION: {data_dir}")
    print(f"{'='*70}")
    print(f"\nTotal frames: {len(frame_files)}")
    
    # Inspect first few frames
    print(f"\nInspecting first 3 frames:")
    all_points = []
    
    for i, frame_file in enumerate(frame_files[:3]):
        points = inspect_binary_file(str(frame_file))
        if points is not None:
            all_points.append(points)
    
    # Analyze sequence statistics
    if frame_files:
        print(f"\n{'='*70}")
        print("SEQUENCE STATISTICS")
        print(f"{'='*70}")
        
        num_points_per_frame = []
        
        for frame_file in frame_files:
            try:
                with open(frame_file, 'rb') as f:
                    data = f.read()
                num_floats = len(data) // 4
                if num_floats % 4 == 0:
                    num_points = num_floats // 4
                    num_points_per_frame.append(num_points)
            except:
                pass
        
        if num_points_per_frame:
            points_array = np.array(num_points_per_frame)
            print(f"\nPoints per frame:")
            print(f"  Min: {points_array.min()}")
            print(f"  Max: {points_array.max()}")
            print(f"  Mean: {points_array.mean():.2f}")
            print(f"  Median: {np.median(points_array):.2f}")
            
            # Recommendations
            print(f"\n{'='*70}")
            print("RECOMMENDATIONS FOR MODEL CONFIGURATION")
            print(f"{'='*70}")
            print(f"\nBased on the data analysis:")
            print(f"  max_frames: {len(frame_files)} (or adjust as needed)")
            print(f"  max_points: {int(points_array.max())} (covers all frames)")
            print(f"  max_points (efficient): {int(np.percentile(points_array, 95))} (covers 95% of frames)")
            print(f"  input_features: 4 (x, y, z, intensity/doppler)")
            
            # Visualize distribution
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(num_points_per_frame)
            plt.xlabel('Frame Number')
            plt.ylabel('Number of Points')
            plt.title('Points per Frame Over Time')
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.hist(num_points_per_frame, bins=30, edgecolor='black')
            plt.xlabel('Number of Points')
            plt.ylabel('Frequency')
            plt.title('Distribution of Points per Frame')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('data_statistics.png')
            print(f"\nVisualization saved to: data_statistics.png")
    
    # Visualize 3D point cloud (if available)
    if all_points:
        try:
            fig = plt.figure(figsize=(15, 5))
            
            for i, points in enumerate(all_points):
                ax = fig.add_subplot(1, 3, i+1, projection='3d')
                
                # Plot points
                scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                                   c=points[:, 3], cmap='viridis', s=10)
                
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                ax.set_title(f'Frame {i+1} ({len(points)} points)')
                
                plt.colorbar(scatter, ax=ax, label='Intensity')
            
            plt.tight_layout()
            plt.savefig('point_cloud_visualization.png')
            print(f"Point cloud visualization saved to: point_cloud_visualization.png")
        except Exception as e:
            print(f"\nCould not create 3D visualization: {e}")


def main():
    """Main function."""
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'mmwave'
    
    if not Path(data_dir).exists():
        print(f"Error: Directory {data_dir} does not exist!")
        return
    
    inspect_dataset(data_dir)
    
    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print("\n1. Review the statistics and recommendations above")
    print("2. Update train.py with appropriate max_frames and max_points")
    print("3. Organize your data into separate directories for each person-action combination")
    print("4. Update the data_dirs, person_labels, and action_labels in train.py")
    print("5. Run: python train.py")


if __name__ == '__main__':
    main()

