import os
import shutil
import subprocess
import dpdata
import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 perturb.py <number_of_perturbations> <poscar_file>")
        print("Usage: 需要POSCAR.STRU,该文件是poscar_file通过atomkit转换得到")
        sys.exit(1)
if __name__ == "__main__":
    main()

poscar_file = sys.argv[2]
ntype = dpdata.System(poscar_file, fmt="vasp/poscar").get_ntypes()
with open(f"{poscar_file}.STRU", 'r') as file:
    upf_orb = ''.join([next(file) for _ in range(2 * ntype + 3)])

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
    d_poscar = dpdata.System(os.path.join(train_directory, 'POSCAR'), fmt="vasp/poscar")
    d_poscar.to("abacus/stru", os.path.join(train_directory, "STRU"))
    with open(os.path.join(train_directory, "STRU"), 'r') as file:
        lines = file.readlines()
    lines[:ntype+1] = upf_orb
    with open(os.path.join(train_directory, "STRU"), 'w') as file:
        file.writelines(lines)
os.chdir('..')

folder_path = os.path.join(original_cwd, perturb_dic)
if os.path.isdir(folder_path):
    for subfolder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder_name)
        if os.path.isdir(subfolder_path):
            shutil.copy("INPUT-scf", os.path.join(subfolder_path, 'INPUT'))
            shutil.copy("KPT-scf", os.path.join(subfolder_path, 'KPT'))
            os.chdir(original_cwd)

