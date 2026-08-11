#!/usr/bin/env python3
"""
Script to compute accuracy metrics comparing predictions to ground truth.
"""

import argparse
import os
import sys

# Add vfnet to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vfnet.report import Reports
from sim_reader.data import DataReader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute accuracy metrics for prediction vs ground truth"
    )
    parser.add_argument(
        "sim_path",
        help="Path to simulation directory"
    )
    parser.add_argument(
        "pred_config",
        help="Path to prediction config file"
    )
    parser.add_argument(
        "--gt-config",
        default=None,
        help="Path to ground truth config file (default: sim_path/gt_config.yaml)"
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV file for metrics (default: pred_dir/metrics_report.csv)"
    )
    parser.add_argument(
        "--skip-normal",
        action="store_true",
        help="Skip normal regression metrics (only compute boundary classification)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine gt_config path
    if args.gt_config is None:
        # Try gt_config_hdp=1.73.yaml first, fallback to gt_config.yaml
        gt_config_173 = os.path.join(args.sim_path, 'gt_config_hdp=1.73.yaml')
        gt_config_default = os.path.join(args.sim_path, 'gt_config.yaml')
        
        if os.path.exists(gt_config_173):
            gt_config = gt_config_173
        else:
            gt_config = gt_config_default
    else:
        gt_config = args.gt_config
    
    if not os.path.exists(gt_config):
        print(f"ERROR: Ground truth config not found: {gt_config}")
        return 1
    
    if not os.path.exists(args.pred_config):
        print(f"ERROR: Prediction config not found: {args.pred_config}")
        return 1
    
    # Determine output CSV path
    if args.output_csv is None:
        pred_dir = os.path.dirname(args.pred_config)
        output_csv = os.path.join(pred_dir, 'metrics_report.csv')
    else:
        output_csv = args.output_csv
    
    output_dir = os.path.dirname(output_csv)
    
    # DataReader expects sim_config.yaml path, not directory
    sim_config_path = os.path.join(args.sim_path, 'sim_config.yaml')
    if not os.path.exists(sim_config_path):
        print(f"ERROR: sim_config.yaml not found at: {sim_config_path}")
        return 1
    
    print(f"Simulation config: {sim_config_path}")
    print(f"Ground truth config: {gt_config}")
    print(f"Prediction config: {args.pred_config}")
    print(f"Output CSV: {output_csv}")
    
    # Initialize data reader with sim_config.yaml path
    print("\n[1/4] Initializing data reader...")
    data_reader = DataReader(sim_config_path)
    
    # Initialize reports
    print("[2/4] Initializing reports...")
    reports = Reports(data_reader)
    
    # Compute classification metrics for boundary
    print("\n[3/4] Computing Boundary Classification Metrics...")
    print(f"  - Reading ground truth from: {gt_config}")
    print(f"  - Reading predictions from: {args.pred_config}")
    try:
        print("  - Calling classification_metrics...")
        import time
        start_time = time.time()
        
        reports.classification_metrics(
            pred_configs=(gt_config, args.pred_config),
            sections=('boundary', 'boundary'),
            output_dir=output_dir,
            extension='csv',
            plot_metrics=False,
            print_metrics=True,
            return_metrics=False
        )
        
        elapsed = time.time() - start_time
        print(f"  ✓ Classification metrics completed in {elapsed:.1f}s")
        
        # Find the generated metrics file and create a standard name link
        import glob
        pred_dir = os.path.dirname(args.pred_config)
        metrics_files = glob.glob(os.path.join(pred_dir, '*_metrics_report.csv'))
        
        if metrics_files:
            # Use the most recent one
            actual_metrics_file = max(metrics_files, key=os.path.getmtime)
            standard_metrics_file = os.path.join(pred_dir, 'metrics_report.csv')
            
            # Copy to standard name
            import shutil
            shutil.copy2(actual_metrics_file, standard_metrics_file)
            print(f"  ✓ Metrics saved to: {standard_metrics_file}")
        else:
            print(f"  ⚠ WARNING: Could not find generated metrics file in {pred_dir}")
            
    except Exception as e:
        print(f"\n✗ ERROR computing boundary metrics: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Compute regression metrics for normal (if available and not skipped)
    if args.skip_normal:
        print("\n[4/4] Skipping Normal Regression Metrics (--skip-normal enabled)")
    else:
        print("\n[4/4] Computing Normal Regression Metrics...")
        print("  WARNING: This may be slow if iterating over many frames")
        print("  TIP: Use --skip-normal to only compute boundary metrics")
        try:
            print("  - Calling compute_regression_metrics...")
            start_time = time.time()
            
            reports.compute_regression_metrics(
                gt_config_file=gt_config,
                pred_config_file=args.pred_config,
                section='normal',
                comparative_label=1,
                initial_step=-1,
                final_step=-1,
                device='cpu',
                print_metrics=True,
                return_metrics=False
            )
            
            elapsed = time.time() - start_time
            print(f"  ✓ Normal regression metrics completed in {elapsed:.1f}s")
        except Exception as e:
            print(f"\n✗ Could not compute normal metrics (may not be available): {e}")
    
    print("\n" + "="*60)
    print("✓ Accuracy metrics computation complete!")
    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
