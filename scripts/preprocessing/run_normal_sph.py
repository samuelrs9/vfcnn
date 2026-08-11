import os
import glob
import sys
from random import random
import numpy as np
import argparse
import yaml
import subprocess
import shutil

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_dir))

from vfnet.base import VoxelFluid
from vfnet.models import VFRWCNN

from vfnet.cnn_models.normal_region_3d import Models31 as NRCNN31
from vfnet.cnn_models.normal_region_3d import Models25 as NRCNN25

from vfnet.preprocessing import Curves2D, DataProcessing, PreprocessSimulation
from vfnet.report import Reports

from sim_reader.data import DataReader
from sim_reader.config import ConfigReader

from metrics.classification import Report

from vfnet.plots import Plots2D


if __name__=="__main__":
    #data_dir = "data/3D/big/fountain_3d_big_res"
    data_dir = "/work1/Doutorado/data/3D/static/rocker-arm"
    normal_dir = f"{data_dir}/sph/normal_sph_spline"
    sim_config_file = f"{data_dir}/sim_config_hdp=2.0.yaml"
    gt_config_file = f"{data_dir}/gt_config.yaml"

    # Roda o ground-truth de normal usando o operador SPH
    print("Running normal ground-truth in the simulation...")
    if not os.path.exists(normal_dir):
        preprocessing = PreprocessSimulation(sim_config_file)
        preprocessing.compute_normals_sph(
            gt_config_file = gt_config_file,
            section = 'boundary',
            search_radius = 2.0,
            use_only_boundary=False,
            initial_step = 0,
            final_step = 0,
            enable_plot = False,
            save = True,
            extension = 'csv',
            output_dir = normal_dir,
            kernel_type='cubic_spline'
        )
    print("Done!")