import os
import time
import h5py
import numpy as np
import tensorflow as tf

def create_voxel_mesh(initial_step=0,final_step=-1,skip_steps=10):
    """"
    Cria malhas de voxels.        
    Última atualização: 17/02/2022.
    
    Args:
        initial_step:
        final_step:
        skip_steps:
    """
    # Diretórios
    data_dir = self.data_reader.data_dir
    mesh_dir = os.path.join(data_dir,'voxel_mesh')
        
    # Cria os diretório de saída caso não existam
    if not os.path.exists(mesh_dir):
        os.mkdir(mesh_dir)            
        
    # Grid
    voxelization = SparseVoxelizer(
        limits = self.real_grid_limits,
        res = self.real_grid_length,
        image_size = self.image_size,
        data_reader = self.data_reader,
        enable_plot=False)
    
    if final_step == -1:
        final_step = self.data_reader.data_info['final_step']

    for step in range(initial_step,final_step+1,skip_steps):
        print('Step {}\n'.format(step))
        # Carrega as particulas do passo corrente
        particles = self.data_reader.get_step(step)            
        mesh_file = os.path.join(mesh_dir,f'voxels.{step}.obj')
        
        t0 = time.time() 
        voxelization.set_points(particles)
        #gt_labels = self.data_reader.get_step_labels_config(step,gt_config_file) # Carrega o ground-truth
        voxelization.create_voxel_mesh(mesh_file)

        print(f' --> Total time: { time.time() - t0:.4f} s\n')

def array_to_csv(file_path,array,columns):
    """
    Salva um array numpy em arquivo csv.
    Última atualização: 21/10/2021.
    
    Args:
        file:
        array:
        columns:
    """
    df = pd.DataFrame(array,columns=columns)
    df.to_csv(file_path,index=False,header=True)

def check_hdf5_dataset(dataset_hdf5):
    """ 
    Verifica as propriedades de um dataset no formato hdf5 das abordagens 1 e 2.
    Última modificação: 16/02/2022. 
    
    Args:
        dataset_hdf5:
        
    """
    print(f'\nDataset file: {dataset_hdf5}')
    with h5py.File(dataset_hdf5) as f:
        # Atributos
        print('Attributes: ')
        attrs = list(f.attrs.keys())
        if len(attrs)==0:
            print("\tthere are no attributes!")
        else:
            for key in attrs:
                print(f'\t{key}: {f.attrs[key]}')
        # Datasets
        print('Datasets arrays: ')
        dataset_arrays = list(f.keys())
        if len(dataset_arrays)==0:
            print("\tthere are no dataset arrays!")
        else:
            for key in dataset_arrays:
                print(f'\t{key} shape: {f[key].shape}')

def padding_ragged(inputs,num_output_points):
    
    @tf.function
    def compute_padding(input):
        if input.shape[0] >= num_output_points:
            output = tf.slice(input,[0,0],[num_output_points,-1])
        else:
            paddings = tf.constant([[0,num_output_points.numpy()-input.shape[0]],[0,0]])
            output = tf.pad(input,paddings,"CONSTANT")
        return output
    
    outputs = tf.map_fn(
        compute_padding,
        inputs,
        #parallel_iterations=4,
        fn_output_signature = tf.TensorSpec(
        (num_output_points, inputs.shape[-1]), dtype=tf.float32)
    )

    return outputs


def trilinear_interpolate(grid_3d: tf.Tensor,sampling_points: tf.Tensor,
                name: str = "trilinear_interpolate",device: str="gpu") -> tf.Tensor:
  """Trilinear interpolation on a 3D regular grid.
  Args:
    grid_3d: A tensor with shape `[A1, ..., An, H, W, D, C]` where H, W, D are
      height, width, depth of the grid and C is the number of channels.
    sampling_points: A tensor with shape `[A1, ..., An, M, 3]` where M is the
      number of sampling points. Sampling points outside the grid are projected
      in the grid borders.
    name:  A name for this op that defaults to "trilinear_interpolate".
  Returns:
    A tensor of shape `[A1, ..., An, M, C]`
  """
  with tf.device(device):
    grid_3d = tf.convert_to_tensor(value=grid_3d)
    sampling_points = tf.convert_to_tensor(value=sampling_points)

    voxel_cube_shape = tf.shape(input=grid_3d)[-4:-1]
    sampling_points.set_shape(sampling_points.shape)
    batch_dims = tf.shape(input=sampling_points)[:-2]
    num_points = tf.shape(input=sampling_points)[-2]

    bottom_left = tf.floor(sampling_points)
    top_right = bottom_left + 1
    bottom_left_index = tf.cast(bottom_left, tf.int32)
    top_right_index = tf.cast(top_right, tf.int32)
    x0_index, y0_index, z0_index = tf.unstack(bottom_left_index, axis=-1)
    x1_index, y1_index, z1_index = tf.unstack(top_right_index, axis=-1)
    index_x = tf.concat([x0_index, x1_index, x0_index, x1_index,
                         x0_index, x1_index, x0_index, x1_index], axis=-1)
    index_y = tf.concat([y0_index, y0_index, y1_index, y1_index,
                         y0_index, y0_index, y1_index, y1_index], axis=-1)
    index_z = tf.concat([z0_index, z0_index, z0_index, z0_index,
                         z1_index, z1_index, z1_index, z1_index], axis=-1)
    indices = tf.stack([index_x, index_y, index_z], axis=-1)
    clip_value = tf.convert_to_tensor(
        value=[voxel_cube_shape - 1], dtype=indices.dtype)
    indices = tf.clip_by_value(indices, 0, clip_value)
    content = tf.gather_nd(
        params=grid_3d, indices=indices, batch_dims=tf.size(input=batch_dims))
        
    distance_to_bottom_left = sampling_points - bottom_left
    distance_to_top_right = top_right - sampling_points
    x_x0, y_y0, z_z0 = tf.unstack(distance_to_bottom_left, axis=-1)
    x1_x, y1_y, z1_z = tf.unstack(distance_to_top_right, axis=-1)
    weights_x = tf.concat([x1_x, x_x0, x1_x, x_x0,
                           x1_x, x_x0, x1_x, x_x0], axis=-1)
    weights_y = tf.concat([y1_y, y1_y, y_y0, y_y0,
                           y1_y, y1_y, y_y0, y_y0], axis=-1)
    weights_z = tf.concat([z1_z, z1_z, z1_z, z1_z,
                           z_z0, z_z0, z_z0, z_z0], axis=-1)
    weights = tf.expand_dims(weights_x * weights_y * weights_z, axis=-1)

    interpolated_values = weights * content
    
    return tf.add_n(tf.split(interpolated_values, [num_points] * 8, -2))
