#!/usr/bin/env python3
"""
Script to count particles in simulation frames.
"""
import sys
import os
from pathlib import Path

# Add repo to path
REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

try:
    from sim_reader.data import DataReader
except ImportError as e:
    print(f"Error importing sim_reader: {e}", file=sys.stderr)
    print("Make sure you're running in the correct Python environment", file=sys.stderr)
    sys.exit(1)


def count_particles_in_range(sim_config_path, initial_step, final_step, skip_steps=1):
    """
    Count total particles across a range of frames.
    
    Args:
        sim_config_path: Path to sim_config.yaml
        initial_step: Starting frame (or -1 for first available)
        final_step: Ending frame (or -1 for last available)
        skip_steps: Number of frames to skip between counts
        
    Returns:
        total_particles: Total number of particles across all processed frames
        num_frames: Number of frames processed
        avg_particles: Average particles per frame
    """
    reader = DataReader(sim_config_path)
    
    # Get available frames
    frames_dir = reader.frames_dir
    base_name = reader.data_info['base_name']
    base_sep = reader.data_info['base_sep']
    extension = reader.data_info['extension']
    
    # Find all available frame files
    import glob
    pattern = os.path.join(frames_dir, f"{base_name}{base_sep}*.{extension}")
    frame_files = sorted(glob.glob(pattern))
    
    if not frame_files:
        return 0, 0, 0.0
    
    # Extract frame numbers
    frame_numbers = []
    for f in frame_files:
        basename = os.path.basename(f)
        try:
            # Extract number from filename
            # Remove base_name and extension, keeping what's between
            num_str = basename.replace(base_name, '', 1).replace(f'.{extension}', '')
            # Remove any remaining separators
            num_str = num_str.strip(base_sep)
            if num_str:
                frame_numbers.append(int(num_str))
        except ValueError:
            continue
    
    frame_numbers = sorted(frame_numbers)
    
    if not frame_numbers:
        return 0, 0, 0.0
    
    # Determine actual range
    if initial_step == -1:
        initial_step = frame_numbers[0]
    if final_step == -1:
        final_step = frame_numbers[-1]
    
    # Count particles in range
    total_particles = 0
    num_frames = 0
    
    for step in range(initial_step, final_step + 1, skip_steps):
        if step not in frame_numbers:
            continue
            
        try:
            coords = reader.get_step(step, attribute='coords')
            num_particles = coords.shape[0]
            total_particles += num_particles
            num_frames += 1
        except Exception as e:
            print(f"Warning: Could not read frame {step}: {e}", file=sys.stderr)
            continue
    
    avg_particles = total_particles / num_frames if num_frames > 0 else 0.0
    
    return total_particles, num_frames, avg_particles


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: count_particles.py <sim_config_path> <initial_step> <final_step> [skip_steps]")
        sys.exit(1)
    
    sim_config_path = sys.argv[1]
    initial_step = int(sys.argv[2])
    final_step = int(sys.argv[3])
    skip_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    
    total, num_frames, avg = count_particles_in_range(
        sim_config_path, initial_step, final_step, skip_steps
    )
    
    # Output format: total_particles,num_frames,avg_particles
    print(f"{total},{num_frames},{avg:.2f}")
