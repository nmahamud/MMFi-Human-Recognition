"""
Helper script to organize mmWave data for training.
Helps users structure their dataset properly.
"""

import os
import shutil
from pathlib import Path
import argparse


def organize_dataset(source_dirs, person_labels, action_labels, output_dir='dataset'):
    """
    Organize multiple data directories into a structured dataset.
    
    Args:
        source_dirs: List of source directories containing frame*.bin files
        person_labels: List of person labels corresponding to each source directory
        action_labels: List of action labels corresponding to each source directory
        output_dir: Output directory to create organized dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Organizing dataset into: {output_dir}")
    print("="*70)
    
    dataset_structure = []
    
    for i, (source_dir, person, action) in enumerate(zip(source_dirs, person_labels, action_labels)):
        source_path = Path(source_dir)
        
        if not source_path.exists():
            print(f"Warning: {source_dir} does not exist, skipping...")
            continue
        
        # Create target directory
        target_name = f"{person}_{action}"
        target_path = output_path / target_name
        target_path.mkdir(exist_ok=True)
        
        # Copy or link files
        frame_files = list(source_path.glob('frame*.bin'))
        
        if not frame_files:
            print(f"Warning: No frame files in {source_dir}, skipping...")
            continue
        
        print(f"\n[{i+1}] {source_dir} -> {target_name}")
        print(f"    Person: {person}, Action: {action}")
        print(f"    Frames: {len(frame_files)}")
        
        for frame_file in frame_files:
            target_file = target_path / frame_file.name
            if not target_file.exists():
                shutil.copy2(frame_file, target_file)
        
        dataset_structure.append({
            'path': str(target_path),
            'person': person,
            'action': action,
            'num_frames': len(frame_files)
        })
        
        print(f"    ✓ Copied {len(frame_files)} frames")
    
    # Save dataset structure
    import json
    structure_file = output_path / 'dataset_structure.json'
    with open(structure_file, 'w') as f:
        json.dump(dataset_structure, f, indent=2)
    
    print("\n" + "="*70)
    print(f"Dataset organized successfully!")
    print(f"Total samples: {len(dataset_structure)}")
    print(f"Structure saved to: {structure_file}")
    
    # Generate training code snippet
    print("\n" + "="*70)
    print("Add this to your train.py:")
    print("="*70)
    print("\n# Load dataset structure")
    print(f"with open('{structure_file}', 'r') as f:")
    print("    dataset_structure = json.load(f)")
    print("\ndata_dirs = [item['path'] for item in dataset_structure]")
    print("person_labels = [item['person'] for item in dataset_structure]")
    print("action_labels = [item['action'] for item in dataset_structure]")
    
    return dataset_structure


def main():
    parser = argparse.ArgumentParser(
        description='Organize mmWave data for training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Example 1: Organize existing data
    python organize_data.py \\
        --dirs mmwave person2_a11 person1_a12 \\
        --persons Person1 Person2 Person1 \\
        --actions A11 A11 A12 \\
        --output dataset

    # Example 2: Create dataset structure file
    python organize_data.py \\
        --dirs data/p1_walk data/p1_run data/p2_walk \\
        --persons Person1 Person1 Person2 \\
        --actions Walk Run Walk
        """
    )
    
    parser.add_argument('--dirs', nargs='+', required=True,
                       help='Source directories containing frame*.bin files')
    parser.add_argument('--persons', nargs='+', required=True,
                       help='Person labels (one per directory)')
    parser.add_argument('--actions', nargs='+', required=True,
                       help='Action labels (one per directory)')
    parser.add_argument('--output', default='dataset',
                       help='Output directory (default: dataset)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.dirs) != len(args.persons) or len(args.dirs) != len(args.actions):
        print("Error: Number of dirs, persons, and actions must match!")
        return
    
    # Organize dataset
    organize_dataset(args.dirs, args.persons, args.actions, args.output)


if __name__ == '__main__':
    main()

