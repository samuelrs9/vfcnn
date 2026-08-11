#!/usr/bin/env python3
"""
CLI para o módulo de preparação de dataset (HDF5) do VFCNN.

Consolida os exemplos antigos de `scripts/create_dataset.py` (removido) em um
único script guiado por configuração `.yaml`, seguindo o mesmo padrão de
`scripts/train.py` e `scripts/predict.py`.

Uso:
    python scripts/build_datasets.py scripts/configs/dataset/example.yaml
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

REPO_DIR = Path(__file__).resolve().parents[1]
for path in (REPO_DIR,):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


VFLUID_KEYS = {
    "data_config_file",
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
BUILD_DATASET_KEYS = {
    "gt_config_file",
    "labels",
    "based_on",
    "dataset_id",
    "initial_step",
    "final_step",
    "skip_steps",
    "batch_size",
    "max_batches",
    "random_ratio",
    "vel_threshold",
    "max_selected_per_bin",
    "balanced_selection",
    "plot_distribution",
    "resolution_based_on_mean_distance",
    "hdp",
    "output_dir",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build VFCNN HDF5 datasets from a YAML config."
    )
    parser.add_argument("config", help="Path to the YAML dataset-build config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the dataset-build jobs without running them.",
    )
    return parser.parse_args()


def load_config(config_file):
    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Dataset config must be a YAML mapping.")
    if "jobs" not in config or not isinstance(config["jobs"], list):
        raise ValueError("Dataset config must define a 'jobs' list.")
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


def split_job_kwargs(job, defaults, keys, section):
    kwargs = merge_dicts(defaults.get(section, {}), job.pop(section, {}))
    for key in list(job):
        if key in keys:
            kwargs[key] = job.pop(key)
    return kwargs


def build_jobs(config):
    defaults = {
        "vfluid": config.get("vfluid", {}),
        "build_dataset": config.get("build_dataset", {}),
    }
    jobs = []

    for raw_job in config["jobs"]:
        job = dict(raw_job)
        vfluid = split_job_kwargs(job, defaults, VFLUID_KEYS, "vfluid")
        build_dataset = split_job_kwargs(job, defaults, BUILD_DATASET_KEYS, "build_dataset")

        vfluid["data_config_file"] = require_absolute_path(
            vfluid.get("data_config_file"), "data_config_file"
        )
        build_dataset["gt_config_file"] = require_absolute_path(
            build_dataset.get("gt_config_file"), "gt_config_file"
        )

        jobs.append(
            {
                "name": job.pop("name", "build_dataset"),
                "vfluid": vfluid,
                "build_dataset": build_dataset,
            }
        )

        if job:
            unknown = ", ".join(sorted(job))
            raise ValueError(f"Unknown job keys: {unknown}")

    return jobs


def run_job(job):
    from vfnet.base import VoxelFluid

    vfluid = VoxelFluid(**job["vfluid"])
    vfluid.build_dataset(**job["build_dataset"])


def print_job(index, total, job):
    print("=========================================")
    print(f"Dataset build job {index}/{total}: {job['name']}")
    print(f"Approach: {job['vfluid'].get('approach')}")
    print(f"Data config: {job['vfluid'].get('data_config_file')}")
    print(f"GT config: {job['build_dataset'].get('gt_config_file')}")
    print(f"Dataset id: {job['build_dataset'].get('dataset_id')}")


def main():
    args = parse_args()
    config = load_config(args.config)
    jobs = build_jobs(config)

    for index, job in enumerate(jobs, start=1):
        print_job(index, len(jobs), job)
        if not args.dry_run:
            run_job(job)


if __name__ == "__main__":
    main()
