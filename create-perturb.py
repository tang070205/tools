"""
    Purpose:
        对已有不同原型结构, 利用dpdatapython包生成晶格+原子坐标都微扰(perturbed_system) 
    Notice:
        准备好原始构型POSCAR-*、INCAR-single, KPOINT和POTCAR由vaspkit生成, 
        然后python3 create-perturb.py 25即可对一个初始结构生成25个微扰结构和文件夹
"""

import os
import shutil
import subprocess
import dpdata
import sys
import glob

def main(perturbations):
    original_cwd = os.getcwd()
    poscar_files = glob.glob('POSCAR-*')  # Find all POSCAR- prefixed files

    for poscar_file in poscar_files:
        base_name = os.path.splitext(os.path.basename(poscar_file))[0]
        perturb_directory = f'perturb-{base_name}'
        os.makedirs(perturb_directory, exist_ok=True)
        shutil.copyfile(poscar_file, os.path.join(perturb_directory, 'CONTCAR'))
        os.chdir(perturb_directory)
        # Generate specified number of perturbed structures and folders
        perturbed_system = dpdata.System('CONTCAR').perturb(pert_num=perturbations,
                                                            cell_pert_fraction=0.03,
                                                            atom_pert_distance=0.15,
                                                            atom_pert_style='uniform')
        for j in range(perturbations):  
            train_directory = f'train-{j+1}'
            os.makedirs(train_directory, exist_ok=True)
            poscar_filename = f'POSCAR{j+1}'
            perturbed_system.to_vasp_poscar(poscar_filename, frame_idx=j)
            shutil.move(poscar_filename, os.path.join(train_directory, 'POSCAR'))
        os.chdir('..')
    os.chdir(original_cwd)

    for folder_name in os.listdir(original_cwd):
        folder_path = os.path.join(original_cwd, folder_name)
        if os.path.isdir(folder_path):
            for subfolder_name in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder_name)
                if os.path.isdir(subfolder_path):
                    shutil.copy("INCAR-single", os.path.join(subfolder_path, 'INCAR'))
                    os.chdir(subfolder_path)
                    vaspkit_command = "vaspkit -task 102 -kpr 0.03"  # K-Spacing set to 0.04
                    subprocess.run(vaspkit_command, shell=True)
                    os.chdir(original_cwd)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 perturb.py <number_of_perturbations>")
        sys.exit(1)

    perturbations = int(sys.argv[1])
    main(perturbations)
