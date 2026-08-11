"""
Utilitários diversos de análise/pré-processamento de simulações que não
fazem parte do fluxo principal (preprocessamento -> dataset -> treino -> predição),
mas são úteis como funções auxiliares.

Migrado de `vfnet/preprocessing.py` (`PreprocessSimulation.compute_normals_pca` e
`PreprocessSimulation.ressample_simulation`), conforme REFACTOR_PLAN.md.
"""
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
from sklearn.decomposition import PCA
from tqdm import tqdm

try:
    from tf_kdtree.neighbors import KDTree
except Exception:
    print('Tf KDTree não foi carregada corretamente!')


def compute_normals_pca(data_reader, data_dir, gt_config_file, section, search_radius=2.0,
    use_only_boundary=False, initial_step=0, final_step=-1, skip_steps=1,
    enable_plot=False, pause=0.1, save=False, base_name='normal', extension='txt',
    output_dir=None):
    """
    Calcula os vetores normais das partículas de fronteira usando PCA.
    Última modificação: 08/04/2022.

    Args:
        data_reader: instância de `DataReader` com a simulação carregada.
        data_dir: diretório da simulação.
        gt_config_file: arquivo de configuração do ground-truth (.yaml).
        section: seção do arquivo de configuração com os rótulos de fronteira.
        search_radius: raio de busca (múltiplo da distância inicial entre partículas).
        use_only_boundary: se verdadeiro, usa apenas partículas de fronteira na busca de vizinhos.
        initial_step: passo inicial.
        final_step: passo final (-1 para usar o último passo disponível).
        skip_steps: intervalo entre passos processados.
        enable_plot: se verdadeiro, plota as normais calculadas (apenas 2D).
        pause: pausa (segundos) entre plots.
        save: se verdadeiro, salva os resultados em disco.
        base_name: nome base dos arquivos de saída.
        extension: extensão dos arquivos de saída ('npy', 'txt' ou 'csv').
        output_dir: diretório de saída (default: `<data_dir>/normal_pca`).
    """
    if save:
        if output_dir is None:
            normal_dir = os.path.join(data_dir, 'normal_pca')
        else:
            normal_dir = output_dir
        os.makedirs(normal_dir, exist_ok=True)

    point_distance = data_reader.properties_info['dp']
    kernel_length = data_reader.properties_info['h']
    spatial_dimensions = data_reader.properties_info['dimensions']

    real_search_radius = search_radius * point_distance

    if final_step == -1:
        final_step = data_reader.data_info['final_step']

    steps = np.arange(initial_step, final_step + 1, skip_steps)
    time_per_step = np.zeros(steps.shape[0])

    for k, step in enumerate(steps):
        print('Step', step)

        t = time.time()

        particles = data_reader.get_step(step)

        gt_labels = data_reader.get_step_labels(step, gt_config_file, section=section)
        gt_labels = gt_labels == 1

        full_kdtree = KDTree(particles, device='cpu')
        _, dists = full_kdtree.query(particles, knn=2)

        normal = np.zeros(particles.shape)

        boundary_particles = particles[gt_labels]

        if use_only_boundary:
            boundary_kdtree = KDTree(boundary_particles, device='cpu')
            neighbors, _ = boundary_kdtree.query_radius(boundary_particles, real_search_radius)
        else:
            neighbors, _ = full_kdtree.query_radius(boundary_particles, real_search_radius)

        boundary_normal = np.zeros(boundary_particles.shape)
        for i in range(neighbors.shape[0]):
            pca = PCA()
            if use_only_boundary:
                pca.fit(boundary_particles[neighbors[i]])
            else:
                pca.fit(particles[neighbors[i]])
            boundary_normal[i] = pca.components_[-1]

            p1 = boundary_particles[i] + real_search_radius * boundary_normal[i]
            p2 = boundary_particles[i] - real_search_radius * boundary_normal[i]

            n_p1p2, _ = full_kdtree.query_radius(np.array([p1, p2]), real_search_radius)

            if len(n_p1p2[0]) > len(n_p1p2[1]):
                boundary_normal[i] = -1 * boundary_normal[i]

        normal[gt_labels] = boundary_normal

        time_per_step[k] = time.time() - t
        print(f' --> time: {time_per_step[k]} s')

        if enable_plot and spatial_dimensions == 2:
            plt.cla()
            plt.scatter(particles[:, 0], particles[:, 1])
            X = boundary_particles[:, 0]
            Y = boundary_particles[:, 1]
            U = boundary_normal[:, 0]
            V = boundary_normal[:, 1]
            plt.quiver(X, Y, U, V)
            plt.axis('equal')
            plt.pause(pause)

        if save:
            normal_file = os.path.join(normal_dir, f'{base_name}.{step}.{extension}')
            if extension == 'npy':
                np.save(normal_file, normal)
            elif extension == 'txt':
                np.savetxt(normal_file, normal, fmt='%.6f')
            elif extension == 'csv':
                if spatial_dimensions == 2:
                    col_normal = ['nx', 'ny']
                    columns = ['label', 'x', 'y'] + col_normal
                elif spatial_dimensions == 3:
                    col_normal = ['nx', 'ny', 'nz']
                    columns = ['label', 'x', 'y', 'z'] + col_normal
                array = np.concatenate([gt_labels[:, np.newaxis], particles, normal], axis=-1)
                df = pd.DataFrame(array, columns=columns)
                df.to_csv(normal_file, index=False, header=True)

    if save:
        times_file = os.path.join(normal_dir, 'times_per_step.csv')
        times = np.concatenate(
            [np.array(steps)[:, np.newaxis], time_per_step[:, np.newaxis]], axis=-1)
        df_t = pd.DataFrame(times, columns=['step', 'time (s)'])
        df_t.to_csv(times_file, index=False, header=True)

        with open(gt_config_file, 'r', encoding='utf-8') as file:
            gt_config = yaml.safe_load(file) or {}

        if 'normal' not in gt_config:
            gt_config['normal'] = {
                'dir': 'normal_pca',
                'base_name': 'normal',
                'extension': extension,
                'columns': ' '.join(col_normal),
                'method': 'pca',
                'initial_distance': point_distance,
                'search_radius': search_radius,
                'comments': (
                    'search radius is a multiplicative factor of the initial distance '
                    'between particles the true search radius is calculated as '
                    'search_radius*initial_distance'),
            }

            with open(gt_config_file, 'w', encoding='utf-8') as file:
                yaml.dump(gt_config, file)


def ressample_simulation(data_reader, gt_config_file, section='boundary', label=1,
    initial_step=0, final_step=-1, save=False, save_num_particles=False,
    enable_plot=False, base_name='pdata', extension='txt', output_dir=None):
    """
    Reamostra a simulação criando uma camada de partículas extras na
    fronteira do fluido.
    Última modificação: 06/06/2022.

    Args:
        data_reader: instância de `DataReader` com a simulação carregada.
        gt_config_file: arquivo de configuração do ground-truth (.yaml).
        section: seção do arquivo de configuração com os rótulos de fronteira.
        label: rótulo (valor) considerado como fronteira.
        initial_step: passo inicial.
        final_step: passo final (-1 para usar o último passo disponível).
        save: se verdadeiro, salva os resultados em disco.
        save_num_particles: se verdadeiro, inclui o número de partículas no cabeçalho do arquivo.
        enable_plot: se verdadeiro, plota o resultado (apenas 2D).
        base_name: nome base dos arquivos de saída.
        extension: extensão dos arquivos de saída ('npy', 'txt'/'dat' ou 'csv').
        output_dir: diretório de saída da simulação reamostrada.
    """
    frames_dir = os.path.join(output_dir, 'frames')
    if os.path.exists(frames_dir) and save:
        return

    if not os.path.exists(gt_config_file):
        raise FileNotFoundError("Ground-truth config file not found!")

    point_distance = data_reader.properties_info['dp']
    spatial_dimensions = data_reader.properties_info['dimensions']

    if spatial_dimensions == 2:
        threshold = 0.7
    elif spatial_dimensions == 3:
        threshold = 0.8

    threshold_point_distance = threshold * point_distance

    if final_step == -1:
        final_step = data_reader.data_info['final_step']

    steps = range(initial_step, final_step + 1)

    if spatial_dimensions == 2:
        particles_sample = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    elif spatial_dimensions == 3:
        particles_sample = np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])

    header_lines = 0
    for step in tqdm(steps, desc='Running resampling...'):
        particles = data_reader.get_step(step, 'coords')
        velocity = data_reader.get_step(step, 'velocity')

        gt_labels = data_reader.get_step_labels(step, gt_config_file, section=section)
        gt_labels = gt_labels == label

        boundary_particles = particles[gt_labels]
        velocity_ressampling = velocity[gt_labels]

        particles_sample_r = point_distance * particles_sample
        particles_ressampling = (
            tf.repeat(boundary_particles, particles_sample.shape[0], axis=0)
            + tf.tile(particles_sample_r, (boundary_particles.shape[0], 1))
        )

        velocity_ressampling = np.tile(velocity_ressampling, [1, particles_sample.shape[0]])
        velocity_ressampling = velocity_ressampling.reshape(-1, particles_sample.shape[1])

        # Verifica região de segurança em relação as partículas da simulação original
        kdtree = KDTree(particles, device='cpu')
        _, dists_nearest_particle_0 = kdtree.query(particles_ressampling, knn=1)

        safe_region = dists_nearest_particle_0 > threshold_point_distance
        safe_region = tf.reshape(safe_region, (-1,))

        particles_ressampling = particles_ressampling[safe_region]
        velocity_ressampling = velocity_ressampling[safe_region]

        # Verifica região de segurança em relação as partículas da reamostragem
        kdtree_aux = KDTree(particles_ressampling, device='cpu')
        neighbors_aux, dists_aux = kdtree_aux.query(particles_ressampling, knn=7)
        dists_aux = dists_aux[:, 1:]

        safe_region_aux = np.ones(particles_ressampling.shape[0], dtype=bool)
        for neighbors in neighbors_aux:
            k, vk = neighbors[0], neighbors[1:]
            if not safe_region_aux[k]:
                continue
            remove = dists_aux[k] < threshold_point_distance
            safe_region_aux[vk[remove]] = False
        particles_ressampling = particles_ressampling[safe_region_aux]
        velocity_ressampling = velocity_ressampling[safe_region_aux]

        particles = np.concatenate([particles, particles_ressampling], axis=0)
        velocity = np.concatenate([velocity, velocity_ressampling], axis=0)

        if enable_plot and spatial_dimensions == 2:
            plt.cla()
            plt.scatter(particles[:, 0], particles[:, 1])
            plt.scatter(particles_ressampling[:, 0], particles_ressampling[:, 1])
            plt.legend(['original particle', 'particle ressampling'])
            _ = plt.axis('equal')
            plt.pause(2)

        if save:
            os.makedirs(frames_dir, exist_ok=True)
            frame_file = os.path.join(frames_dir, f'{base_name}.{step}.{extension}')
            if extension == 'npy':
                np.save(frame_file, particles)
            elif extension in ('txt', 'dat'):
                header_lines = 0
                with open(frame_file, 'a') as f:
                    if save_num_particles:
                        header_lines += 1
                        np.savetxt(f, particles.shape[0:1], fmt='%d')
                    np.savetxt(f, particles, fmt='%.9f')
            elif extension == 'csv':
                if spatial_dimensions == 2:
                    columns = ['x', 'y', 'vx', 'vy', 'ressampling']
                elif spatial_dimensions == 3:
                    columns = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ressampling']
                ressampling = np.zeros((particles.shape[0], 1))
                ressampling[-particles_ressampling.shape[0]:] = 1
                array = np.concatenate([particles, velocity, ressampling], axis=-1)
                df = pd.DataFrame(array, columns=columns)
                df.to_csv(frame_file, index=False, header=True)

    if save:
        # Salva o arquivo de configuração da simulação reamostrada
        sim_config = {
            'simulation_properties': {
                'dp': data_reader.properties_info['dp'],
                'h': data_reader.properties_info['h'],
                'mass': data_reader.properties_info['mass'],
                'dimensions': data_reader.properties_info['dimensions'],
                'limits': list(data_reader.properties_info['limits']),
            },
            'data': {
                'sim_name': output_dir.split(os.sep)[-1],
                'from': 'ressampling',
                'frames_dir': 'frames',
                'base_name': base_name,
                'extension': extension,
                'header_lines': header_lines,
                'initial_step': initial_step,
                'final_step': final_step,
                'coords': 'x / y',
                'velocity': 'vx / vy',
            },
        }
        if spatial_dimensions == 3:
            sim_config['data']['coords'] += ' / z'
            sim_config['data']['velocity'] += ' / vz'

        with open(os.path.join(output_dir, 'sim_config.yaml'), 'w', encoding='utf-8') as file:
            yaml.dump(sim_config, file)
