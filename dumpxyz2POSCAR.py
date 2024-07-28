from ase.io import read, write
import sys
import os
import shutil
import subprocess

def main(number_structures):
    original_cwd = os.getcwd()

    with open("dump.xyz", "r") as input_file:
        first_line = input_file.readline()
        try:
            structure_lines = int(first_line) + 2
        except ValueError:
            print("第一行内容不能转换为整数")
        total_lines = sum(1 for _ in input_file) + 1
        print(structure_lines)
        print(total_lines)
        structures_count = total_lines // structure_lines
        print(structures_count)

    output_lines = []
    with open("dump.xyz", "r") as input_file:
        lines = input_file.readlines()
        for i in range(number_structures):
            start_index = structure_lines * (structures_count // number_structures * (i+1) - 1)
            output_lines += lines[start_index:start_index + structure_lines]

    with open("activate-learning.xyz", "w") as output_file:
        output_file.writelines(output_lines)

    def create_train_folders():
        if not os.path.exists('train_folders'):
            os.makedirs('train_folders')
        for i in range(number_structures):
            folder_name = f'activate-learning-{i+1}'
            folder_path = os.path.join('train_folders', folder_name)
            os.makedirs(folder_path)

    def split_xyz():
        with open('dump.xyz', 'r') as file:
            lines = file.readlines()  
        num_groups = len(lines) // structure_lines
        for i in range(num_groups):
            start_index = i * structure_lines
            end_index = start_index + structure_lines
            group_lines = lines[start_index:end_index]
            group_filepath = os.path.join('train_folders', f'activate-learning-{i + 1}', f'activate-learning-{i + 1}.xyz')
            with open(group_filepath, 'w') as group_file:
                group_file.writelines(group_lines)

    def convert_xyz_to_poscar():
        train_folder_path = os.path.join(os.getcwd(), 'train_folders')
        folders = os.listdir(train_folder_path)
        for folder_name in folders:
            folder_path = os.path.join(train_folder_path, folder_name)
            os.chdir(folder_path)
            xyz_file = next((f for f in os.listdir(folder_path) if f.endswith('.xyz')), None)
            write("POSCAR", read(xyz_file, format="extxyz"))

    create_train_folders()
    split_xyz()
    convert_xyz_to_poscar()

    os.chdir(original_cwd)
    for j in range(number_structures):
        folder_name = f'activate-learning-{j+1}' 
        folder_path = os.path.join('train_folders', folder_name)  
        shutil.copyfile('INCAR-single', os.path.join(folder_path, 'INCAR'))  #INCAR-single是计算单点能的INCAR
        os.chdir(folder_path) 
        vaspkit_command = "vaspkit -task 102 -kpr 0.03"  # 此处采用vaspkit生成KPOINTS和POTCAR，
        subprocess.run(vaspkit_command, shell=True)  
        os.chdir(original_cwd)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 dump-xyz2POSCAR.py <number_structures>")
        sys.exit(1)

    number_structures = int(sys.argv[1])
    main(number_structures)
