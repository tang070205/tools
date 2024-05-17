from ase.io import read, write
import numpy as np
import os
import shutil
import subprocess
from ovito.io import import_file, export_file
from ovito.modifiers import WrapPeriodicImagesModifier

def create_train_folders():
    if not os.path.exists('train_folders'):
        os.makedirs('train_folders')
    for i in range(1, 51):
        folder_name = f'train-{i}'
        folder_path = os.path.join('train_folders', folder_name)
        os.makedirs(folder_path)

def split_train_xyz():
    with open('dump.xyz', 'r') as file:
        lines = file.readlines()
    group_size = 58   #一个结构的行数，按实际修改
    num_groups = len(lines) // group_size
    for i in range(num_groups):
        start_index = i * group_size
        end_index = start_index + group_size
        group_lines = lines[start_index:end_index]
        group_filename = f'train_group{i + 1}.xyz'
        group_filepath = os.path.join('train_folders', f'train-{i + 1}', group_filename)
        with open(group_filepath, 'w') as group_file:
            group_file.writelines(group_lines)

def convert_train_xyz_to_poscar():
    train_folder_path = os.path.join(os.getcwd(), 'train_folders')
    folders = os.listdir(train_folder_path)
    for folder_name in folders:
        folder_path = os.path.join(train_folder_path, folder_name)
        xyz_files = [f for f in os.listdir(folder_path) if f.endswith('.xyz')]
        for xyz_file in xyz_files:
            xyz_filepath = os.path.join(folder_path, xyz_file)
            poscar_filepath = os.path.join(folder_path, xyz_file.replace('.xyz', '.vasp'))
            pipeline = import_file(xyz_filepath)
            pipeline.modifiers.append(WrapPeriodicImagesModifier())
            export_file(pipeline, poscar_filepath, 'vasp')
            new_poscar_filepath = os.path.join(folder_path, 'POSCAR')
            os.rename(poscar_filepath, new_poscar_filepath)

create_train_folders()
split_train_xyz()
convert_train_xyz_to_poscar()

original_cwd = os.getcwd()  
for j in range(1, 51):
    folder_name = f'train-{j}' 
    folder_path = os.path.join('train_folders', folder_name)  
    shutil.copyfile('INCAR-single', os.path.join(folder_path, 'INCAR'))  #INCAR-single是计算单点能的INCAR
    os.chdir(folder_path) 
    vaspkit_command = "vaspkit -task 102 -kpr 0.04"  # 此处采用vaspkit生成KPOINTS和POTCAR，
    subprocess.run(vaspkit_command, shell=True)  
    os.chdir(original_cwd)
