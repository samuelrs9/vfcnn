#!/usr/bin/env python3
"""
Script to compare coarse predictions against ground truth across all frames.

This script reads a list of coarse predictions with their corresponding ground truths,
iterates over all frames, and calculates:
  - Percentage of particles labeled as 'interior' or 'undefined' by the prediction
  - Number and percentage of boundary particles (GT) incorrectly classified as interior

Author: VFNet Team
Last modified: 2026-06-20
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sim_reader.data import DataReader
from sim_reader.config import ConfigReader


def check_cache(pred_config_file, pred_type):
    """
    Check if cached metrics exist for the prediction.
    
    Args:
        pred_config_file: Path to prediction config
        pred_type: Type of prediction being analyzed
        
    Returns:
        dict with cached metrics if exists, None otherwise
    """
    pred_dir = os.path.dirname(pred_config_file)
    cache_file = os.path.join(pred_dir, 'coarse_metrics_cache.npz')
    
    if os.path.exists(cache_file):
        try:
            data = np.load(cache_file, allow_pickle=True)
            # Check if cache is for the same prediction type
            if 'prediction_type' in data and str(data['prediction_type']) == pred_type:
                return {
                    'total_particles': int(data['total_particles']),
                    'interior_count': int(data['interior_count']),
                    'undefined_count': int(data['undefined_count']),
                    'gt_boundary_count': int(data.get('gt_boundary_count', 0)),
                    'gt_boundary_as_interior_count': int(data.get('gt_boundary_as_interior_count', 0))
                }
        except Exception as e:
            print(f"  Warning: Could not load cache: {e}")
            return None
    
    return None


def save_cache(pred_config_file, pred_type, metrics):
    """
    Save computed metrics to cache.
    
    Args:
        pred_config_file: Path to prediction config
        pred_type: Type of prediction
        metrics: Dictionary with computed metrics
    """
    pred_dir = os.path.dirname(pred_config_file)
    cache_file = os.path.join(pred_dir, 'coarse_metrics_cache.npz')
    
    try:
        np.savez(
            cache_file,
            prediction_type=pred_type,
            total_particles=metrics['total_particles'],
            interior_count=metrics['interior_count'],
            undefined_count=metrics['undefined_count'],
            gt_boundary_count=metrics['gt_boundary_count'],
            gt_boundary_as_interior_count=metrics['gt_boundary_as_interior_count']
        )
    except Exception as e:
        print(f"  Warning: Could not save cache: {e}")


def get_coarse_prediction_labels(data_reader, step, pred_config_file, 
                                  pred_type, threshold=None):
    """
    Get the predicted labels for a given step and prediction type.
    
    Args:
        data_reader: DataReader object for the simulation
        step: Frame/step number
        pred_config_file: Path to the prediction config file
        pred_type: Type of prediction (e.g., 'pred_combined_product', 'pred_density')
        threshold: Optional threshold to apply to measures
        
    Returns:
        labels: Array of labels (0=interior, 1=undefined)
    """
    pred_config = ConfigReader(pred_config_file)
    pred_section = pred_config.get_section(pred_type)
    
    # Check if this is a prediction section or a measure section
    if pred_type.startswith('pred_'):
        # This is already a prediction - read labels directly
        labels = data_reader.get_step_labels(step, pred_config_file, section=pred_type)
    else:
        # This is a measure section - need to apply threshold
        measures = data_reader.get_step_measures(step, pred_config_file, section=pred_type)
        
        # Get threshold from config or use provided threshold
        if threshold is None:
            # Try to find threshold in a corresponding pred section
            pred_type_name = f"pred_{pred_type}"
            if pred_config.has_section(pred_type_name):
                pred_sect = pred_config.get_section(pred_type_name)
                # Look for common threshold names
                for key in ['combined_threshold', 'density_threshold', 'centroid_threshold', 'threshold']:
                    if key in pred_sect:
                        threshold = pred_sect[key]
                        break
        
        if threshold is None:
            raise ValueError(f"No threshold found for measure type '{pred_type}'. "
                           f"Please specify threshold in config or pred_config.yaml")
        
        # Apply threshold: values below threshold are labeled as boundary (1)
        # values above threshold are labeled as interior (0)
        labels = (measures < threshold).astype(int)
    
    return labels


def analyze_coarse_predictions(config_file):
    """
    Main function to compare coarse predictions against ground truth.
    
    Args:
        config_file: Path to the YAML configuration file
    """
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    comparisons = config['comparisons']
    pred_type = config['prediction_type']
    threshold = config.get('threshold', None)
    report_config = config.get('report', {})
    output_path = report_config.get('output_path', './reports/coarse_prediction_comparison')
    
    print(f"\n{'='*80}")
    print(f"Coarse Prediction Comparison Report")
    print(f"{'='*80}")
    print(f"Prediction type: {pred_type}")
    if threshold is not None:
        print(f"Threshold: {threshold}")
    print(f"Number of comparisons: {len(comparisons)}")
    print(f"{'='*80}\n")
    
    results = []
    
    for comparison in comparisons:
        gt_config_file = comparison['gt_config_file']
        pred_config_files = comparison.get('pred_config_files', [])
        
        if not isinstance(pred_config_files, list):
            pred_config_files = [pred_config_files]
        
        # Get simulation name from GT directory
        gt_dir = os.path.dirname(gt_config_file)
        sim_name = os.path.basename(gt_dir)
        
        # Derive sim_config_file from gt_config_file directory
        sim_config_file = os.path.join(gt_dir, 'sim_config.yaml')
        
        # Initialize data reader
        data_reader = DataReader(sim_config_file)
        
        # Process each prediction
        for pred_config_file in pred_config_files:
            # Get prediction name
            pred_name = os.path.basename(os.path.dirname(pred_config_file))
            
            print(f"\nProcessing: {sim_name} - {pred_name}")
            
            # Check if cached metrics exist
            cached_metrics = check_cache(pred_config_file, pred_type)
            
            if cached_metrics is not None:
                print(f"  Using cached metrics for: {pred_name}")
                
                total_particles = cached_metrics['total_particles']
                total_interior = cached_metrics['interior_count']
                total_undefined = cached_metrics['undefined_count']
                total_gt_boundary = cached_metrics['gt_boundary_count']
                total_gt_boundary_as_interior = cached_metrics['gt_boundary_as_interior_count']
                
                # Get frame info for display
                initial_step = 0
                final_step = data_reader.data_info['final_step']
                total_steps = final_step + 1
            else:
                # Get frame range
                initial_step = 0
                final_step = data_reader.data_info['final_step']
                # TEST: Limit to first 50 frames for faster testing
                # final_step = min(final_step, 49)
                total_steps = final_step + 1
                
                print(f"  Frames: {initial_step} to {final_step} ({total_steps} frames)")
                print(f"  Computing metrics...")
                
                # Counters
                total_particles = 0
                total_interior = 0
                total_undefined = 0
                
                # Counters for GT comparison
                total_gt_boundary = 0
                total_gt_boundary_as_interior = 0
                
                # Process each frame
                for step in tqdm(range(initial_step, final_step + 1), 
                                desc=f"  Processing frames", leave=False):
                    try:
                        # Get prediction labels
                        pred_labels = get_coarse_prediction_labels(
                            data_reader, step, pred_config_file, pred_type, threshold
                        )
                        
                        # Count prediction labels (0=interior, 1=undefined/boundary)
                        n_particles = len(pred_labels)
                        n_interior = np.sum(pred_labels == 0)
                        n_undefined = np.sum(pred_labels == 1)
                        
                        total_particles += n_particles
                        total_interior += n_interior
                        total_undefined += n_undefined
                        
                        # Get GT labels and compare
                        try:
                            gt_labels = data_reader.get_step_labels(step, gt_config_file, section='boundary')
                            
                            # GT: 0=interior, 1=boundary
                            # Pred: 0=interior, 1=undefined
                            # Find GT boundary particles classified as interior by prediction
                            gt_boundary_mask = (gt_labels == 1)
                            pred_interior_mask = (pred_labels == 0)
                            
                            n_gt_boundary = np.sum(gt_boundary_mask)
                            n_gt_boundary_as_interior = np.sum(gt_boundary_mask & pred_interior_mask)
                            
                            total_gt_boundary += n_gt_boundary
                            total_gt_boundary_as_interior += n_gt_boundary_as_interior
                            
                        except Exception as e:
                            # GT not available for this step
                            pass
                        
                    except Exception as e:
                        print(f"\n  Warning: Error processing step {step}: {e}")
                        continue
                
                # Save metrics to cache
                save_cache(pred_config_file, pred_type, {
                    'total_particles': total_particles,
                    'interior_count': total_interior,
                    'undefined_count': total_undefined,
                    'gt_boundary_count': total_gt_boundary,
                    'gt_boundary_as_interior_count': total_gt_boundary_as_interior
                })
            
            # Calculate percentages
            pct_interior = (total_interior / total_particles * 100) if total_particles > 0 else 0
            pct_undefined = (total_undefined / total_particles * 100) if total_particles > 0 else 0
            pct_boundary_loss = (total_gt_boundary_as_interior / total_gt_boundary * 100) if total_gt_boundary > 0 else 0
            
            print(f"  Total particles: {total_particles:,}")
            print(f"  Interior: {total_interior:,} ({pct_interior:.2f}%)")
            print(f"  Undefined: {total_undefined:,} ({pct_undefined:.2f}%)")
            if total_gt_boundary > 0:
                print(f"  GT Boundary particles: {total_gt_boundary:,}")
                print(f"  GT Boundary classified as Interior: {total_gt_boundary_as_interior:,} ({pct_boundary_loss:.2f}%)")
            
            results.append({
                'simulation': sim_name,
                'gt_config': gt_config_file,
                'prediction': pred_config_file,
                'prediction_type': pred_type,
                'threshold': threshold,
                'total_frames': total_steps,
                'total_particles': total_particles,
                'interior_count': total_interior,
                'undefined_count': total_undefined,
                'interior_percentage': pct_interior,
                'undefined_percentage': pct_undefined,
                'gt_boundary_count': total_gt_boundary,
                'gt_boundary_as_interior_count': total_gt_boundary_as_interior,
                'boundary_loss_percentage': pct_boundary_loss
            })
    
    # Save results
    os.makedirs(output_path, exist_ok=True)
    
    df = pd.DataFrame(results)
    csv_file = os.path.join(output_path, 'coarse_prediction_comparison.csv')
    df.to_csv(csv_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"Report saved to: {csv_file}")
    print(f"{'='*80}\n")
    
    # Print summary table
    print("\nSummary:")
    summary_cols = ['simulation', 'prediction', 'interior_percentage', 'undefined_percentage', 'boundary_loss_percentage']
    print(df[summary_cols].to_string(index=False))


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python compare_coarse_predictions.py <config_file.yaml>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    if not os.path.exists(config_file):
        print(f"Error: Config file not found: {config_file}")
        sys.exit(1)
    
    analyze_coarse_predictions(config_file)
    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()
