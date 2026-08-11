#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import yaml


REPO_DIR = Path(__file__).resolve().parents[1]
for path in (REPO_DIR, REPO_DIR / "vfnet", REPO_DIR / "voxel-cloud-net"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


VFLUID_KEYS = {
    "tasks",
    "features",
    "approach",
    "search_radius",
    "grid_length",
    "image_size",
    "border_size",
    "enable_plot",
    "coarse_prediction",
}

PREDICT_KEYS = {
    "predict_id",
    "batch_size",
    "coarse_threshold",
    "base_name",
    "extension",
    "device",
    "debug",
    "report_extension",
    "initial_step",
    "final_step",
    "skip_steps",
    "grid_offset",
    "pred_dir",
    "return_prediction",
    "resolution_based_on_mean_distance",
    "hdp",
    "model_path",
    "decision_threshold",
    "post_processing",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run direct VoxelFluid predictions from a YAML config."
    )
    parser.add_argument("config", help="Path to the YAML prediction config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the prediction jobs without running inference.",
    )
    return parser.parse_args()


def load_config(config_file):
    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Prediction config must be a YAML mapping.")
    if "jobs" not in config or not isinstance(config["jobs"], list):
        raise ValueError("Prediction config must define a 'jobs' list.")
    return config


def merge_dicts(base, override):
    merged = dict(base or {})
    merged.update(override or {})
    return merged


def require_absolute_path(path, field_name):
    if not path:
        raise ValueError(f"'{field_name}' is required.")
    if not os.path.isabs(path):
        raise ValueError(f"'{field_name}' must be an absolute path: {path}")
    return path


def configure_tensorflow(config):
    import tensorflow as tf

    tf.config.run_functions_eagerly(True)

    if not config.get("gpu_memory_growth", False):
        return

    physical_devices = tf.config.list_physical_devices("GPU")
    if not physical_devices:
        print("No GPU found; continuing without GPU memory-growth configuration.")
        return

    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


def build_jobs(config):
    default_fluid = config.get("voxel_fluid", {})
    default_prediction = config.get("prediction", {})
    jobs = []

    for raw_job in config["jobs"]:
        job = dict(raw_job)
        sim_path = require_absolute_path(job.pop("sim_path", None), "sim_path")
        model_path = require_absolute_path(job.pop("model_path", None), "model_path")
        sim_config = job.pop("sim_config", "sim_config.yaml")
        model_config = job.pop("model_config", "model_config.yaml")

        vfluid_kwargs = merge_dicts(default_fluid, job.pop("voxel_fluid", {}))
        predict_kwargs = merge_dicts(default_prediction, job.pop("prediction", {}))
        for key in list(job):
            if key in VFLUID_KEYS:
                vfluid_kwargs[key] = job.pop(key)
            elif key in PREDICT_KEYS:
                predict_kwargs[key] = job.pop(key)

        # Extract post_processing fields and merge with predict_kwargs
        post_processing = predict_kwargs.pop("post_processing", {})
        if post_processing:
            predict_kwargs["extract_mesh"] = post_processing.get("extract_mesh", False)
            predict_kwargs["poisson_recon_path"] = post_processing.get("poisson_recon_path", None)

        predict_kwargs["model_config_file"] = os.path.join(model_path, model_config)
        predict_kwargs["model_path"] = model_path

        jobs.append(
            {
                "name": job.pop("name", "prediction"),
                "sim_path": sim_path,
                "model_path": model_path,
                "skip_if_exists": job.pop("skip_if_exists", None),
                "vfluid_kwargs": {
                    **vfluid_kwargs,
                    "data_config_file": os.path.join(sim_path, sim_config),
                },
                "predict_kwargs": predict_kwargs,
            }
        )

        if job:
            unknown = ", ".join(sorted(job))
            raise ValueError(f"Unknown job keys: {unknown}")

    return jobs


def print_job(index, total, job):
    print("=========================================")
    print(f"Prediction job {index}/{total}: {job['name']}")
    print(f"Sim path: {job['sim_path']}")
    print(f"Model path: {job['model_path']}")
    print(f"Sim config: {job['vfluid_kwargs']['data_config_file']}")
    print(f"Model config: {job['predict_kwargs']['model_config_file']}")
    if job["predict_kwargs"].get("model_path"):
        print(f"Model checkpoint: {job['predict_kwargs']['model_path']}")
    if job["predict_kwargs"].get("pred_dir"):
        print(f"Prediction dir: {job['predict_kwargs']['pred_dir']}")


def run_jobs(jobs, dry_run=False):
    if not dry_run:
        from vfnet.base import VoxelFluid

    for index, job in enumerate(jobs, start=1):
        print_job(index, len(jobs), job)
        if dry_run:
            continue

        skip_if_exists = job.get("skip_if_exists")
        if skip_if_exists and os.path.exists(skip_if_exists):
            print(f"Prediction exists: {skip_if_exists}")
            continue

        vfluid = VoxelFluid(**job["vfluid_kwargs"])
        vfluid.predict_offline(**job["predict_kwargs"])


def main():
    args = parse_args()
    config = load_config(args.config)
    jobs = build_jobs(config)

    if not args.dry_run:
        configure_tensorflow(config.get("tensorflow", {}))

    run_jobs(jobs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
