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

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 perturb.py <number_of_perturbations>")
        sys.exit(1)
if __name__ == "__main__":
    main()

perturbations = int(sys.argv[1])
original_cwd = os.getcwd()
perturb_dic = f'perturb'
os.makedirs(perturb_dic, exist_ok=True)
shutil.copyfile(poscar_file, os.path.join(perturb_dic, 'CONTCAR'))
os.chdir(perturb_dic)
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

perturb_dic = f'perturb'
folder_path = os.path.join(original_cwd, perturb_dic)
if os.path.isdir(folder_path):
    for subfolder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder_name)
        if os.path.isdir(subfolder_path):
            shutil.copy("INCAR-single", os.path.join(subfolder_path, 'INCAR'))
            os.chdir(subfolder_path)
            vaspkit_command = "vaspkit -task 102 -kpr 0.03" 
            subprocess.run(vaspkit_command, shell=True)
            os.chdir(original_cwd)

