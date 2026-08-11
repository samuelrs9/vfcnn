import os
import glob
import configparser

def convert_value(value):
    # Convert space-separated values to lists
    if ' ' in value:
        items = value.split()
        if all(item.isdigit() for item in items):
            return [int(item) for item in items]
        elif all(is_float(item) for item in items):
            return [float(item) for item in items]
        else:
            return [item for item in items]
    # Convert comma-separated values to lists
    if ',' in value:
        items = value.split(',')
        return [item.strip() for item in items]
    # Convert single values to int or float if possible
    if value.isdigit():
        return int(value)
    elif is_float(value):
        return float(value)
    return value

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def process_ini(input_path, output_path):
    config = configparser.ConfigParser()
    config.read(input_path)

    new_config = configparser.ConfigParser()

    for section in config.sections():
        new_config.add_section(section)
        for key, value in config.items(section):
            new_value = convert_value(value)
            if isinstance(new_value, list):
                new_value_str = "[" + ", ".join(repr(item) if isinstance(item, str) else str(item) for item in new_value) + "]"
                new_config.set(section, key, new_value_str)
            else:
                if isinstance(new_value, str):
                    new_config.set(section, key, repr(new_value))
                else:
                    new_config.set(section, key, str(new_value))

    with open(output_path, 'w') as configfile:
        new_config.write(configfile)


if __name__ == "__main__":

    working_dir = "/home/samuel/Doutorado/voxel-fluid-net"
    
    # Parâmetros do dataset
    batch_size = 100000
    search_radius = 2.0
    grid_length = 0.1
    dataset_id = 2

    #config_str = '10000_1.50_1'
    config_str = f'{batch_size}_{search_radius:.2f}_{grid_length:.1f}_{dataset_id}'

    # Arquivos de configuração de datasets
    config_file_0 = os.path.join(working_dir,f"data/3D/big/ddb_3d_big_res/sparse_regionwise_approach/datasets/kfold5/dataset_{config_str}/dataset_config.ini")
    config_file_1 = os.path.join(working_dir,f"data/3D/big/inlet_collision_3d_big_res/sparse_regionwise_approach/datasets/kfold5/dataset_{config_str}/dataset_config.ini")
    config_file_2 = os.path.join(working_dir,f"data/3D/big/db_blocks_3d_big_res/sparse_regionwise_approach/datasets/kfold5/dataset_{config_str}/dataset_config.ini")
    config_file_3 = os.path.join(working_dir,f"data/3D/big/inlet_vortex_3d_big_res/sparse_regionwise_approach/datasets/kfold5/dataset_{config_str}/dataset_config.ini")
    config_file_4 = os.path.join(working_dir,f"data/3D/big/fountain_3d_big_res/sparse_regionwise_approach/datasets/kfold5/dataset_{config_str}/dataset_config.ini")
    #input_files = [config_file_0, config_file_1, config_file_2, config_file_3, config_file_4]

    input_files = [
        "data/3D/big/inlet_vortex_3d_big_res/regionwise_approach/predictions/kfold2/pred_31_9_3.10_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_74/pred_config.ini",
        "data/3D/big/inlet_vortex_3d_big_res/regionwise_approach/predictions/kfold2/pred_31_9_3.10_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_74/pred_config.ini",
        "data/3D/big/db_blocks_3d_big_res/regionwise_approach/predictions/kfold2/pred_31_9_3.10_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_74/pred_config.ini",
        "data/3D/big/inlet_collision_3d_big_res/regionwise_approach/predictions/kfold2/pred_31_9_3.10_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_74/pred_config.ini",
        "data/3D/big/ddb_3d_big_res/regionwise_approach/predictions/kfold2/pred_31_9_3.10_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_74/pred_config.ini"   
    ]

    #input_files = glob.glob("data/3D/static/*/sparse_regionwise_approach/predictions/kfold3_h=1dp/*/pred_config.ini")
    
    for input_file in input_files:
        output_file = input_file.replace('.ini', '_v2.ini')
        process_ini(input_file, output_file)
