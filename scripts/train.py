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


MANAGER_KEYS = {"tasks", "data_dir", "dataset_config_file"}
TRAIN_PIPELINE_KEYS = {
    "trainset_file",
    "batch_size",
    "train_size",
    "buffer_size_factor",
    "data_augmentation",
    "debug_mode",
}
VAL_PIPELINE_KEYS = {
    "valset_file",
    "batch_size",
    "val_size",
    "buffer_size_factor",
    "data_augmentation",
    "debug_mode",
}
TRAIN_MODEL_KEYS = {
    "train_id",
    "num_epochs",
    "learning_rate",
    "patience_max",
    "device",
    "debug_mode",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train VoxelFluid models from a YAML config."
    )
    parser.add_argument("config", help="Path to the YAML training config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the training jobs without running training.",
    )
    return parser.parse_args()


def load_config(config_file):
    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Training config must be a YAML mapping.")
    if "jobs" not in config or not isinstance(config["jobs"], list):
        raise ValueError("Training config must define a 'jobs' list.")
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


def split_job_kwargs(job, defaults, keys, section):
    kwargs = merge_dicts(defaults.get(section, {}), job.pop(section, {}))
    for key in list(job):
        if key in keys:
            kwargs[key] = job.pop(key)
    return kwargs


def build_jobs(config):
    defaults = {
        "manager": config.get("manager", {}),
        "model": config.get("model", {}),
        "train_pipeline": config.get("train_pipeline", {}),
        "val_pipeline": config.get("val_pipeline", {}),
        "training": config.get("training", {}),
    }
    jobs = []

    for raw_job in config["jobs"]:
        job = dict(raw_job)
        manager = split_job_kwargs(job, defaults, MANAGER_KEYS, "manager")
        model = merge_dicts(defaults["model"], job.pop("model", {}))
        train_pipeline = split_job_kwargs(
            job, defaults, TRAIN_PIPELINE_KEYS, "train_pipeline"
        )
        val_pipeline = split_job_kwargs(job, defaults, VAL_PIPELINE_KEYS, "val_pipeline")
        training = split_job_kwargs(job, defaults, TRAIN_MODEL_KEYS, "training")

        manager["data_dir"] = require_absolute_path(manager.get("data_dir"), "data_dir")
        manager["dataset_config_file"] = require_absolute_path(
            manager.get("dataset_config_file"), "dataset_config_file"
        )
        training = normalize_device(training)

        jobs.append(
            {
                "name": job.pop("name", "training"),
                "manager": manager,
                "model": model,
                "train_pipeline": train_pipeline,
                "val_pipeline": val_pipeline,
                "training": training,
            }
        )

        if job:
            unknown = ", ".join(sorted(job))
            raise ValueError(f"Unknown job keys: {unknown}")

    return jobs


def make_manager(config):
    from vfnet.models import SparseVFCNNManager, VFRWCNN

    manager_type = config.pop("type")
    if manager_type == "regionwise":
        return VFRWCNN(**config)
    if manager_type == "sparse_regionwise":
        return SparseVFCNNManager(**config)
    raise ValueError(f"Unsupported manager type: {manager_type}")


def make_model(vfcnn, config):
    from vfnet.cnn_models.normal_region_3d import Models25 as NRCNN25
    from vfnet.cnn_models.normal_region_3d import Models31 as NRCNN31
    from vfnet.cnn_models.sparse_models import SparseVoxelizedFluidCNN

    model_type = config.get("type", "template")

    if model_type == "template":
        kwargs = {"from_template": config["template"]}
        if "pretrained_model_config_file" in config:
            kwargs["pretrained_model_config_file"] = config["pretrained_model_config_file"]
        if "transfer_learning" in config:
            kwargs["transfer_learning"] = config["transfer_learning"]
        vfcnn.set_model(**kwargs)
        return

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
        build_method = config.get("build_method", "build_model")
        getattr(vfcnn.model, build_method)()
        return

    raise ValueError(f"Unsupported model type: {model_type}")


def train_job(job):
    vfcnn = make_manager(dict(job["manager"]))
    make_model(vfcnn, job["model"])

    call_with_supported_kwargs(vfcnn.train_input_pipeline, job["train_pipeline"])
    call_with_supported_kwargs(vfcnn.val_input_pipeline, job["val_pipeline"])

    if job["model"].get("summary", False):
        vfcnn.model.summary()
    if job["model"].get("plot", False):
        vfcnn.model.plot_model()

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
    print(f"Training job {index}/{total}: {job['name']}")
    print(f"Manager: {job['manager']['type']}")
    print(f"Data dir: {job['manager']['data_dir']}")
    print(f"Dataset config: {job['manager']['dataset_config_file']}")
    print(f"Model: {job['model'].get('type', 'template')}")
    if job["model"].get("template"):
        print(f"Template: {job['model']['template']}")
    if job["model"].get("factory"):
        print(f"Factory: {job['model']['factory']}")
    print(f"Train id: {job['training'].get('train_id')}")
    print(f"Epochs: {job['training'].get('num_epochs')}")
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
            train_job(job)


if __name__ == "__main__":
    main()
