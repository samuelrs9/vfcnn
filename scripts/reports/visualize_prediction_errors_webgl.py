#!/usr/bin/env python3
"""
Script to visualize prediction errors with WebGL sphere rendering.
Uses native WebGL capabilities for better 3D particle visualization.
"""

import os
import yaml
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


def load_particles(file_path, header_lines=1):
    """Load particles from a file."""
    if not os.path.exists(file_path):
        return None
    try:
        data = np.loadtxt(file_path, skiprows=header_lines)
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_labels(file_path):
    """Load labels (ground truth or prediction)."""
    if not os.path.exists(file_path):
        return None
    try:
        labels = np.loadtxt(file_path, dtype=int)
        if labels.ndim == 0:
            labels = np.array([labels])
        return labels
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def get_simulation_name(gt_config_path):
    """Extract simulation name from gt_config path."""
    parts = Path(gt_config_path).parts
    for i, part in enumerate(parts):
        if part == 'simulations' and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def find_last_step(sim_config, pred_dir, pred_dir_name, pred_base_name, pred_extension):
    """Find the last step of the simulation based on available prediction files."""
    pred_boundary_dir = os.path.join(pred_dir, pred_dir_name)
    if not os.path.exists(pred_boundary_dir):
        return None
    
    # List all prediction files
    files = os.listdir(pred_boundary_dir)
    steps = []
    for f in files:
        if f.startswith(pred_base_name) and f.endswith(f".{pred_extension}"):
            try:
                # Extract step number from filename
                step_str = f.replace(pred_base_name + '.', '').replace(f'.{pred_extension}', '')
                steps.append(int(step_str))
            except ValueError:
                continue
    
    if steps:
        return max(steps)
    return sim_config.get('data', {}).get('final_step', 500)


def load_sim_config(sim_dir):
    """Load simulation configuration."""
    sim_config_path = os.path.join(sim_dir, 'sim_config.yaml')
    if os.path.exists(sim_config_path):
        with open(sim_config_path, 'r') as f:
            return yaml.safe_load(f)
    return None


def visualize_errors(particles, gt_labels, pred_labels, sim_name, output_path, step):
    """
    Visualiza falsos positivos e falsos negativos em 3D.
    
    - Falsos Positivos (FP): predição diz 'boundary' (1), GT diz 'interior' (0)
    - Falsos Negativos (FN): predição diz 'interior' (0), GT diz 'boundary' (1)
    - True Positives (TP): ambos dizem 'boundary' (1)
    - True Negatives (TN): ambos dizem 'interior' (0)
    """
    if particles is None or gt_labels is None or pred_labels is None:
        print(f"Incomplete data for {sim_name}, step {step}")
        return
    
    # Ensure sizes match
    min_len = min(len(particles), len(gt_labels), len(pred_labels))
    particles = particles[:min_len]
    gt_labels = gt_labels[:min_len]
    pred_labels = pred_labels[:min_len]
    
    # Extract x, y, z coordinates (assuming first 3 columns are coordinates)
    x = particles[:, 0]
    y = particles[:, 1]
    z = particles[:, 2]
    
    # Identify error types
    false_positives = (pred_labels == 1) & (gt_labels == 0)
    false_negatives = (pred_labels == 0) & (gt_labels == 1)
    true_positives = (pred_labels == 1) & (gt_labels == 1)
    true_negatives = (pred_labels == 0) & (gt_labels == 0)
    
    # Count errors
    n_fp = np.sum(false_positives)
    n_fn = np.sum(false_negatives)
    n_tp = np.sum(true_positives)
    n_tn = np.sum(true_negatives)
    total = len(particles)
    
    print(f"\n{sim_name} - Step {step}:")
    print(f"  Total particles: {total}")
    print(f"  True Positives (TP): {n_tp} ({100*n_tp/total:.2f}%)")
    print(f"  True Negatives (TN): {n_tn} ({100*n_tn/total:.2f}%)")
    print(f"  False Positives (FP): {n_fp} ({100*n_fp/total:.2f}%)")
    print(f"  False Negatives (FN): {n_fn} ({100*n_fn/total:.2f}%)")
    
    # Create subplots with Plotly
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            f'{sim_name} - Step {step}<br>Overview - {total:,} particles',
            f'False Positives: {n_fp}<br>(Pred=boundary, GT=interior)',
            f'False Negatives: {n_fn}<br>(Pred=interior, GT=boundary)'
        ),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        horizontal_spacing=0.05
    )
    
    # Subplot 1: Overview with all points
    # Subsampling for better performance (use only a fraction of TN particles)
    subsample_tn = max(1, n_tn // 10000) if n_tn > 10000 else 1
    
    if n_tn > 0:
        tn_indices = np.where(true_negatives)[0][::subsample_tn]
        fig.add_trace(
            go.Scatter3d(
                x=x[tn_indices], y=y[tn_indices], z=z[tn_indices],
                mode='markers',
                marker=dict(
                    size=2,
                    color='rgba(200, 200, 200, 0.4)',
                    symbol='circle',
                    line=dict(width=0)
                ),
                name=f'TN (correct interior): {n_tn}',
                showlegend=True,
                hovertemplate='<b>TN</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    if n_tp > 0:
        fig.add_trace(
            go.Scatter3d(
                x=x[true_positives], y=y[true_positives], z=z[true_positives],
                mode='markers',
                marker=dict(
                    size=3,
                    color='rgba(34, 139, 34, 0.7)',  # Forest green
                    symbol='circle',
                    line=dict(width=0)
                ),
                name=f'TP (correct boundary): {n_tp}',
                showlegend=True,
                hovertemplate='<b>TP</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    if n_fp > 0:
        fig.add_trace(
            go.Scatter3d(
                x=x[false_positives], y=y[false_positives], z=z[false_positives],
                mode='markers',
                marker=dict(
                    size=5,
                    color='rgba(220, 20, 60, 0.8)',  # Crimson
                    symbol='diamond',
                    line=dict(width=0.5, color='rgba(139, 0, 0, 0.9)')
                ),
                name=f'FP (error): {n_fp}',
                showlegend=True,
                hovertemplate='<b>FALSE POSITIVE</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    if n_fn > 0:
        fig.add_trace(
            go.Scatter3d(
                x=x[false_negatives], y=y[false_negatives], z=z[false_negatives],
                mode='markers',
                marker=dict(
                    size=5,
                    color='rgba(30, 144, 255, 0.8)',  # Dodger blue
                    symbol='diamond',
                    line=dict(width=0.5, color='rgba(0, 0, 139, 0.9)')
                ),
                name=f'FN (error): {n_fn}',
                showlegend=True,
                hovertemplate='<b>FALSE NEGATIVE</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Subplot 2: Only False Positives
    if n_fp > 0:
        fig.add_trace(
            go.Scatter3d(
                x=x[false_positives], y=y[false_positives], z=z[false_positives],
                mode='markers',
                marker=dict(
                    size=6,
                    color='rgba(220, 20, 60, 0.85)',
                    symbol='diamond',
                    line=dict(width=0.5, color='rgba(139, 0, 0, 0.9)')
                ),
                name='False Positives',
                showlegend=False,
                hovertemplate='<b>FALSE POSITIVE</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
    
    # Subplot 3: Only False Negatives
    if n_fn > 0:
        fig.add_trace(
            go.Scatter3d(
                x=x[false_negatives], y=y[false_negatives], z=z[false_negatives],
                mode='markers',
                marker=dict(
                    size=6,
                    color='rgba(30, 144, 255, 0.85)',
                    symbol='diamond',
                    line=dict(width=0.5, color='rgba(0, 0, 139, 0.9)')
                ),
                name='False Negatives',
                showlegend=False,
                hovertemplate='<b>FALSE NEGATIVE</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
            ),
            row=1, col=3
        )
    
    # Update layout with WebGL rendering and better 3D settings
    camera_settings = dict(
        eye=dict(x=1.5, y=1.5, z=1.3),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0)
    )
    
    scene_settings = dict(
        xaxis=dict(
            title='X',
            backgroundcolor='rgb(230, 230,230)',
            gridcolor='white',
            showbackground=True,
            zerolinecolor='white'
        ),
        yaxis=dict(
            title='Y',
            backgroundcolor='rgb(230, 230,230)',
            gridcolor='white',
            showbackground=True,
            zerolinecolor='white'
        ),
        zaxis=dict(
            title='Z',
            backgroundcolor='rgb(230, 230,230)',
            gridcolor='white',
            showbackground=True,
            zerolinecolor='white'
        ),
        aspectmode='data',
        camera=camera_settings
    )
    
    fig.update_layout(
        title_text=f'Error Analysis - {sim_name} - Step {step} (WebGL Enhanced)',
        width=1800,
        height=600,
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        scene=scene_settings,
        scene2=scene_settings,
        scene3=scene_settings,
        hovermode='closest'
    )
    
    # Save as interactive HTML
    fig.write_html(output_path)
    
    print(f"  Visualization saved to: {output_path}")


def process_comparison(comparison, output_dir):
    """Process a prediction comparison."""
    gt_config_file = comparison['gt_config_file']
    pred_config_files = comparison['pred_config_files']
    
    # Find the "no_coarse" prediction
    pred_config_no_coarse = None
    for pred_file in pred_config_files:
        if 'no_coarse' in pred_file:
            pred_config_no_coarse = pred_file
            break
    
    if pred_config_no_coarse is None:
        print(f"Prediction 'no_coarse' not found for {gt_config_file}")
        return
    
    # Load configurations
    with open(gt_config_file, 'r') as f:
        gt_config = yaml.safe_load(f)
    
    with open(pred_config_no_coarse, 'r') as f:
        pred_config = yaml.safe_load(f)
    
    # Get directories
    sim_dir = os.path.dirname(gt_config_file)
    pred_dir = os.path.dirname(pred_config_no_coarse)
    sim_name = get_simulation_name(gt_config_file)
    
    # Load sim_config to get the last step
    sim_config = load_sim_config(sim_dir)
    if sim_config is None:
        print(f"Could not load sim_config.yaml for {sim_name}")
        return
    
    # Get prediction information to find the last available step
    pred_dir_name = pred_config.get('boundary', {}).get('dir', 'boundary')
    pred_base_name = pred_config.get('boundary', {}).get('base_name', 'labels')
    pred_extension = pred_config.get('boundary', {}).get('extension', 'txt')
    
    last_step = find_last_step(sim_config, pred_dir, pred_dir_name, pred_base_name, pred_extension)
    if last_step is None:
        print(f"Could not determine last step for {sim_name}")
        return
    
    # Data paths
    data_config = sim_config.get('data', {})
    frames_dir = os.path.join(sim_dir, data_config.get('frames_dir', 'frames'))
    base_name = data_config.get('base_name', 'pdata')
    extension = data_config.get('extension', 'dat')
    header_lines = data_config.get('header_lines', 1)
    
    # Load particles
    particle_file = os.path.join(frames_dir, f"{base_name}.{last_step}.{extension}")
    print(f"  Loading particles from: {particle_file}")
    particles = load_particles(particle_file, header_lines)
    if particles is None:
        print(f"  ERROR: Could not load particles from {particle_file}")
        return
    
    # Load ground truth
    gt_dir_name = gt_config.get('boundary', {}).get('dir', 'gt_hdp=1.73')
    gt_base_name = gt_config.get('boundary', {}).get('base_name', 'gt')
    gt_extension = gt_config.get('boundary', {}).get('extension', 'dat')
    
    # Try to find the correct gt directory if the specified one doesn't exist
    gt_dir = os.path.join(sim_dir, gt_dir_name)
    if not os.path.exists(gt_dir):
        # Try common alternatives
        alternatives = ['gt_hdp=1.73', 'gt_hdp=2.0', 'gt']
        for alt in alternatives:
            alt_dir = os.path.join(sim_dir, alt)
            if os.path.exists(alt_dir):
                gt_dir_name = alt
                gt_dir = alt_dir
                print(f"  Using alternative GT directory: {gt_dir_name}")
                break
    
    gt_file = os.path.join(sim_dir, gt_dir_name, f"{gt_base_name}.{last_step}.{gt_extension}")
    print(f"  Loading ground truth from: {gt_file}")
    gt_labels = load_labels(gt_file)
    if gt_labels is None:
        print(f"  ERROR: Could not load ground truth from {gt_file}")
        return
    
    # Load prediction
    pred_dir_name = pred_config.get('boundary', {}).get('dir', 'boundary')
    pred_base_name = pred_config.get('boundary', {}).get('base_name', 'labels')
    pred_extension = pred_config.get('boundary', {}).get('extension', 'txt')
    pred_file = os.path.join(pred_dir, pred_dir_name, f"{pred_base_name}.{last_step}.{pred_extension}")
    print(f"  Loading prediction from: {pred_file}")
    pred_labels = load_labels(pred_file)
    if pred_labels is None:
        print(f"  ERROR: Could not load prediction from {pred_file}")
        return
    
    # Create visualization
    output_path = os.path.join(output_dir, f"{sim_name}_step_{last_step}_errors.html")
    visualize_errors(particles, gt_labels, pred_labels, sim_name, output_path, last_step)


def main():
    # Configuration
    config_file = '/work1/voxel-fluid-net/scripts/configs/reports/compare_predictions.yaml'
    output_dir = '/work1/voxel-fluid-net/reports/prediction_errors_webgl'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    comparisons = config.get('comparisons', [])
    
    print(f"Processing {len(comparisons)} comparisons...")
    print(f"Output will be saved to: {output_dir}\n")
    
    # Process each comparison
    for i, comparison in enumerate(comparisons, 1):
        print(f"\n{'='*60}")
        print(f"Processing comparison {i}/{len(comparisons)}")
        print(f"{'='*60}")
        process_comparison(comparison, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Processing completed!")
    print(f"Images saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
