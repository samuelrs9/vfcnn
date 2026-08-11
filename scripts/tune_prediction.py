#!/usr/bin/env python3
import argparse
import glob
import os
import sys
from decimal import Decimal
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
    "extract_mesh",
    "decision_threshold",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VoxelFluid prediction tuning jobs from a YAML config."
    )
    parser.add_argument("config", help="Path to the YAML tuning config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the tuning jobs without running inference.",
    )
    return parser.parse_args()


def load_config(config_file):
    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Prediction tuning config must be a YAML mapping.")
    if "jobs" not in config or not isinstance(config["jobs"], list):
        raise ValueError("Prediction tuning config must define a 'jobs' list.")
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


def decimal_range(start, stop, step):
    value = Decimal(str(start))
    stop = Decimal(str(stop))
    step = Decimal(str(step))
    values = []
    while value < stop:
        values.append(float(value))
        value += step
    return values


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


def expand_tuning_values(tuning):
    tuning_type = tuning.get("type")
    if tuning_type == "hdp":
        values = tuning.get("values")
        if values is None:
            values = decimal_range(
                tuning["start"],
                tuning["stop"],
                tuning.get("step", 0.1),
            )
        return [{"hdp": value} for value in values]

    if tuning_type == "decision_threshold":
        values = tuning.get("values")
        if values is None:
            values = decimal_range(
                tuning["start"],
                tuning["stop"],
                tuning.get("step", 0.05),
            )
        return [{"decision_threshold": value} for value in values]

    if tuning_type == "checkpoints":
        checkpoint_glob = tuning.get("checkpoint_glob", "checkpoints/*")
        if not os.path.isabs(checkpoint_glob):
            checkpoint_glob = os.path.join(tuning["model_dir"], checkpoint_glob)
        checkpoints = sorted(glob.glob(checkpoint_glob))
        if not checkpoints:
            if tuning.get("allow_missing_glob", False):
                checkpoints = [checkpoint_glob]
            else:
                raise FileNotFoundError(f"No checkpoints found for glob: {checkpoint_glob}")
        return [
            {
                "model_path": checkpoint,
                "checkpoint_name": os.path.basename(checkpoint.rstrip(os.sep)),
            }
            for checkpoint in checkpoints
        ]

    raise ValueError(f"Unsupported tuning type: {tuning_type}")


def build_base_job(config, raw_job):
    job = dict(raw_job)
    data_dir = require_absolute_path(job.pop("data_dir", None), "data_dir")
    model_dir = require_absolute_path(job.pop("model_dir", None), "model_dir")
    sim_config = job.pop("sim_config", "sim_config.yaml")
    model_config = job.pop("model_config", "model_config.yaml")

    vfluid_kwargs = merge_dicts(config.get("voxel_fluid", {}), job.pop("voxel_fluid", {}))
    predict_kwargs = merge_dicts(config.get("prediction", {}), job.pop("prediction", {}))

    for key in list(job):
        if key in VFLUID_KEYS:
            vfluid_kwargs[key] = job.pop(key)
        elif key in PREDICT_KEYS:
            predict_kwargs[key] = job.pop(key)

    tuning = job.pop("tuning")
    tuning["model_dir"] = model_dir

    predict_kwargs["model_config_file"] = os.path.join(model_dir, model_config)

    base_job = {
        "name": job.pop("name", "prediction_tuning"),
        "data_dir": data_dir,
        "model_dir": model_dir,
        "skip_existing": job.pop("skip_existing", False),
        "vfluid_kwargs": {
            **vfluid_kwargs,
            "data_config_file": os.path.join(data_dir, sim_config),
        },
        "predict_kwargs": predict_kwargs,
        "tuning": tuning,
    }

    if job:
        unknown = ", ".join(sorted(job))
        raise ValueError(f"Unknown job keys: {unknown}")

    return base_job


def build_jobs(config, dry_run=False):
    jobs = []
    for raw_job in config["jobs"]:
        base_job = build_base_job(config, raw_job)
        if dry_run and base_job["tuning"].get("type") == "checkpoints":
            base_job["tuning"]["allow_missing_glob"] = True

        for values in expand_tuning_values(base_job["tuning"]):
            job = {
                "name": base_job["name"],
                "data_dir": base_job["data_dir"],
                "model_dir": base_job["model_dir"],
                "skip_existing": base_job["skip_existing"],
                "vfluid_kwargs": dict(base_job["vfluid_kwargs"]),
                "predict_kwargs": dict(base_job["predict_kwargs"]),
                "tuning_values": values,
            }

            job["predict_kwargs"].update(
                {key: value for key, value in values.items() if key in PREDICT_KEYS}
            )

            pred_dir = job["predict_kwargs"].get("pred_dir")
            if pred_dir:
                job["predict_kwargs"]["pred_dir"] = pred_dir.format(**values)

            jobs.append(job)

    return jobs


def print_job(index, total, job):
    print("=========================================")
    print(f"Tuning job {index}/{total}: {job['name']}")
    print(f"Data dir: {job['data_dir']}")
    print(f"Model dir: {job['model_dir']}")
    print(f"Sim config: {job['vfluid_kwargs']['data_config_file']}")
    print(f"Model config: {job['predict_kwargs']['model_config_file']}")
    if job["predict_kwargs"].get("model_path"):
        print(f"Model checkpoint: {job['predict_kwargs']['model_path']}")
    print(f"Tuning values: {job['tuning_values']}")
    if job["predict_kwargs"].get("pred_dir"):
        print(f"Prediction dir: {job['predict_kwargs']['pred_dir']}")


def run_jobs(jobs, dry_run=False):
    if not dry_run:
        from vfnet.base import VoxelFluid

    for index, job in enumerate(jobs, start=1):
        print_job(index, len(jobs), job)
        if dry_run:
            continue

        pred_dir = job["predict_kwargs"].get("pred_dir")
        if job["skip_existing"] and pred_dir and os.path.exists(pred_dir):
            print(f"Prediction exists: {pred_dir}")
            continue

        vfluid = VoxelFluid(**job["vfluid_kwargs"])
        vfluid.predict_offline(**job["predict_kwargs"])


def main():
    args = parse_args()
    config = load_config(args.config)
    jobs = build_jobs(config, dry_run=args.dry_run)

    if not args.dry_run:
        configure_tensorflow(config.get("tensorflow", {}))

    run_jobs(jobs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
