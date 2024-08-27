import os
import shutil
import subprocess
import dpdata
import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: 需要自备INPUT-scf")
        print("Usage: python3 perturb.py <number_of_perturbations> <poscar_file>")
        sys.exit(1)
if __name__ == "__main__":
    main()

poscar_file = sys.argv[2]
subprocess.run(f"echo '1\n 101\n  175 {poscar_file}' | atomkit", shell=True)
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
            os.chdir(subfolder_path)
            atomkit_command = 'echo "3\n 301\n 0\n 101 STRU\n 0.03" | atomkit'
            subprocess.run(atomkit_command, shell=True)
            os.chdir(original_cwd)

