"""
Pós-processamento de predições do VFCNN: conversão de resultados de predição
(HDF5/npz/CSV) para formatos utilizados em visualização (CSV, PLY).

Migrado de `vfnet/preprocessing.py` (`DataProcessing.pred_to_csv`,
`DataProcessing.npz_to_csv`, `DataProcessing.export_to_ply` e
`DataProcessing.export_to_dat`), conforme REFACTOR_PLAN.md.
"""
import os

import numpy as np
import pandas as pd
import open3d as o3d

from sim_reader.config import ConfigReader
from sim_reader.data import DataReader


class PostProcessing:

    def __init__(self, config_file=None):
        """
        Construtor.

        Args:
            config_file: arquivo de configuração (.yaml) da simulação.
        """
        if config_file is not None:
            self.config_file = config_file
            self.data_dir = os.path.dirname(config_file)
            self.data_reader = DataReader(config_file)

    def pred_to_csv(self, pred_config_files, output_dir=None, initial_step=None, final_step=np.inf):
        """
        Cria arquivos csv com dados de previsões.
        Útil para visualização no paraview.
        Última atualização: 23/07/2023.
        """
        os.makedirs(output_dir, exist_ok=True)

        steps = self.data_reader.find_available_steps()

        if len(steps) == 0:
            print("No steps found!")
            return

        for step in steps:
            if step < initial_step:
                continue

            if step > final_step:
                break

            print('Step', step)
            particles = self.data_reader.get_step(step)
            array = particles
            columns = []
            columns += self.data_reader.data_info['coords']

            for key in pred_config_files:

                pred_config_file = pred_config_files[key]
                pred_config = ConfigReader(pred_config_file)

                pred_dict = pred_config.get_all_sections()
                pred_sections = pred_dict['general']['prediction_sections']

                if type(pred_sections) is str:
                    pred_sections = pred_sections.split(" ")

                for section in pred_sections:

                    try:
                        pred = self.data_reader.get_step_measures(
                            step, pred_config_file, section=section)

                        column_names = pred_dict[section]['column_names']
                        array = np.concatenate([array, pred.reshape(-1, len(column_names))], axis=-1)
                        columns += [f'{key}_{name}' for name in column_names]

                    except FileNotFoundError as error:
                        print("Prediction not found:", error)
                        continue
                    except ValueError as error:
                        print("ValueError:", error)
                        continue

            pred_file = os.path.join(output_dir, f'pred.{step}.csv')
            df = pd.DataFrame(array, columns=columns)
            df.to_csv(pred_file, index=False, header=True)

    def export_to_ply(self, pred_config_file, output_dir=None, initial_step=None, final_step=np.inf, replace=False):
        """
        Convert the predictions (boundary and normal) to ply format.
        That's util to create an output compatible with the Poisson Reconstruction algorithm.
        """
        os.makedirs(output_dir, exist_ok=True)
        steps = self.data_reader.find_available_steps()

        if len(steps) == 0:
            print("No steps found!")
            return

        for step in steps:
            print('Step', step)

            if step < initial_step:
                continue

            if step > final_step:
                break

            ply_file = os.path.join(output_dir, f"boundary.{step}.ply")
            if os.path.exists(ply_file) and not replace:
                print(f"Frame {step} has already been processed!")
                continue

            try:
                boundary = self.data_reader.get_step_measures(step, pred_config_file, section='boundary').astype(bool)
                normals = self.data_reader.get_step_measures(step, pred_config_file, section='normal')[boundary]
                coords = self.data_reader.get_step(step)[boundary]

            except FileNotFoundError as error:
                print("Prediction not found:", error)
                continue
            except ValueError as error:
                print("ValueError:", error)
                continue
            except Exception as error:
                print(f"Skipping frame: {error}")
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords)
            pcd.normals = o3d.utility.Vector3dVector(normals)

            o3d.io.write_point_cloud(ply_file, pcd)

    def export_to_dat(self, pred_config_file, output_dir=None, initial_step=None, final_step=np.inf):
        """
        Convert the predictions (boundary and normal) to dat.
        That's util to create an output compatible with the Poisson Reconstruction algorithm.
        """
        os.makedirs(output_dir, exist_ok=True)
        steps = self.data_reader.find_available_steps()

        if len(steps) == 0:
            print("No steps found!")
            return

        for step in steps:
            print('Step', step)

            if step < initial_step:
                continue

            if step > final_step:
                break

            ply_file = os.path.join(output_dir, f"boundary.{step}.ply")
            if os.path.exists(ply_file):
                print(f"Frame {step} has already been processed!")
                continue

            try:
                boundary = self.data_reader.get_step_measures(step, pred_config_file, section='boundary').astype(bool)
                normals = self.data_reader.get_step_measures(step, pred_config_file, section='normal')[boundary]
                coords = self.data_reader.get_step(step)[boundary]

            except FileNotFoundError as error:
                print("Prediction not found:", error)
                continue
            except ValueError as error:
                print("ValueError:", error)
                continue
            except Exception as error:
                print(f"Skipping frame: {error}")
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords)
            pcd.normals = o3d.utility.Vector3dVector(normals)

            o3d.io.write_point_cloud(ply_file, pcd)

    def npz_to_csv(self, npz_file):
        """
        Converte arquivo compactado numpy para csv. As variáveis em npz
        precisam ter as mesmas dimensões.
        """
        npz = dict(np.load(npz_file))
        columns = []
        array = []
        for key in npz:
            columns.append(key)
            array.append(npz[key])
        csv_file = npz_file.replace('npz', 'csv')
        df = pd.DataFrame(np.array(array).T.round(3), columns=columns)
        df.to_csv(csv_file, index=False, header=True)
