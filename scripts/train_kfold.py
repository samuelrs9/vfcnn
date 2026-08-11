#!/usr/bin/env python3
import argparse
import inspect
import os
import sys
from pathlib import Path

import yaml


REPO_DIR = Path(__file__).resolve().parents[1]
for path in (REPO_DIR, REPO_DIR / "vfnet", REPO_DIR / "voxel-cloud-net"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mix K-Fold datasets and train VoxelFluid models from YAML."
    )
    parser.add_argument("config", help="Path to the YAML K-Fold training config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the K-Fold training jobs without running training.",
    )
    return parser.parse_args()


def load_config(config_file):
    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("K-Fold training config must be a YAML mapping.")
    if "jobs" not in config or not isinstance(config["jobs"], list):
        raise ValueError("K-Fold training config must define a 'jobs' list.")
    return config


def require_absolute_path(path, field_name):
    if not path:
        raise ValueError(f"'{field_name}' is required.")
    if not os.path.isabs(path):
        raise ValueError(f"'{field_name}' must be an absolute path: {path}")
    return path


def normalize_device(kwargs):
    if kwargs.get("device") == "cuda":
        kwargs = dict(kwargs)
        kwargs["_configured_device"] = "cuda"
        kwargs["device"] = "gpu"
    return kwargs


def configure_tensorflow(config):
    import tensorflow as tf

    tf.config.run_functions_eagerly(config.get("run_eagerly", True))

    if not config.get("gpu_memory_growth", False):
        return

    physical_devices = tf.config.list_physical_devices("GPU")
    if not physical_devices:
        print("No GPU found; continuing without GPU memory-growth configuration.")
        return

    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


def make_manager(manager_type, kwargs):
    from vfnet.models import SparseVFCNNManager, VFRWCNN

    if manager_type == "regionwise":
        return VFRWCNN(**kwargs)
    if manager_type == "sparse_regionwise":
        return SparseVFCNNManager(**kwargs)
    raise ValueError(f"Unsupported K-Fold manager type: {manager_type}")


def make_model(vfcnn, config):
    from vfnet.cnn_models.normal_region_3d import Models25 as NRCNN25
    from vfnet.cnn_models.normal_region_3d import Models31 as NRCNN31
    from vfnet.cnn_models.sparse_models import SparseVoxelizedFluidCNN

    model_type = config.get("type", "factory")
    if model_type == "factory":
        model = eval(config["factory"], {"NRCNN31": NRCNN31, "NRCNN25": NRCNN25})()
        vfcnn.set_model(model=model)
        return
    if model_type == "sparse_voxelized_fluid_cnn":
        model = SparseVoxelizedFluidCNN(
            tasks=vfcnn.tasks,
            num_input_features=config.get("num_input_features", 1),
            grid_size=config.get("grid_size", [2048, 2048, 2048]),
            architecture=config.get("architecture", "v3_4"),
        )
        vfcnn.set_model(model=model)
        getattr(vfcnn.model, config.get("build_method", "build_model"))()
        return
    raise ValueError(f"Unsupported model type: {model_type}")


def build_jobs(config):
    jobs = []
    for raw_job in config["jobs"]:
        job = dict(raw_job)
        manager = dict(config.get("manager", {}))
        manager.update(job.pop("manager", {}))
        training = dict(config.get("training", {}))
        training.update(job.pop("training", {}))
        training = normalize_device(training)

        data_dir = require_absolute_path(job.pop("data_dir"), "data_dir")
        mixed_dataset_dir = require_absolute_path(
            job.pop("mixed_dataset_dir"), "mixed_dataset_dir"
        )
        dataset_config_file = os.path.join(
            mixed_dataset_dir, job.pop("mixed_dataset_config", "dataset_config_v2.yaml")
        )
        dataset_configs = [
            require_absolute_path(path, "dataset_configs[]")
            for path in job.pop("dataset_configs")
        ]

        jobs.append(
            {
                "name": job.pop("name", "kfold_training"),
                "manager_type": manager.pop("type"),
                "tasks": manager.pop("tasks"),
                "data_dir": data_dir,
                "mixed_dataset_dir": mixed_dataset_dir,
                "dataset_config_file": dataset_config_file,
                "dataset_configs": dataset_configs,
                "fold_id": job.pop("fold_id"),
                "model": job.pop("model", config.get("model", {})),
                "train_pipeline": job.pop("train_pipeline", config.get("train_pipeline", {})),
                "val_pipeline": job.pop("val_pipeline", config.get("val_pipeline", {})),
                "training": training,
            }
        )

        if manager:
            unknown = ", ".join(sorted(manager))
            raise ValueError(f"Unknown manager keys: {unknown}")
        if job:
            unknown = ", ".join(sorted(job))
            raise ValueError(f"Unknown job keys: {unknown}")

    return jobs


def run_job(job):
    manager_for_mix = make_manager(job["manager_type"], {"tasks": job["tasks"]})
    if not os.path.exists(job["mixed_dataset_dir"]):
        manager_for_mix.mix_hdf5_datasets(
            job["dataset_configs"], job["mixed_dataset_dir"], job["fold_id"]
        )

    vfcnn = make_manager(
        job["manager_type"],
        {
            "tasks": job["tasks"],
            "data_dir": job["data_dir"],
            "dataset_config_file": job["dataset_config_file"],
        },
    )
    make_model(vfcnn, job["model"])
    call_with_supported_kwargs(vfcnn.train_input_pipeline, job["train_pipeline"])
    call_with_supported_kwargs(vfcnn.val_input_pipeline, job["val_pipeline"])

    if job["model"].get("summary", False):
        vfcnn.model.summary()

    method = job["training"].pop("method", "train_model")
    call_with_supported_kwargs(getattr(vfcnn, method), job["training"])


def call_with_supported_kwargs(func, kwargs):
    signature = inspect.signature(func)
    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return func(**supported)


def print_job(index, total, job):
    print("=========================================")
    print(f"K-Fold training job {index}/{total}: {job['name']}")
    print(f"Manager: {job['manager_type']}")
    print(f"Data dir: {job['data_dir']}")
    print(f"Mixed dataset dir: {job['mixed_dataset_dir']}")
    print(f"Dataset config: {job['dataset_config_file']}")
    print(f"Fold id: {job['fold_id']}")
    print(f"Train id: {job['training'].get('train_id')}")
    configured_device = job["training"].get("_configured_device")
    if configured_device:
        print(f"Device: {configured_device} (mapped to {job['training'].get('device')})")
    else:
        print(f"Device: {job['training'].get('device')}")


def main():
    args = parse_args()
    config = load_config(args.config)
    jobs = build_jobs(config)

    if not args.dry_run:
        configure_tensorflow(config.get("tensorflow", {}))

    for index, job in enumerate(jobs, start=1):
        print_job(index, len(jobs), job)
        if not args.dry_run:
            run_job(job)


if __name__ == "__main__":
    main()
