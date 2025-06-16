import os, sys, shutil, subprocess
import dpdata


def main():
    if len(sys.argv) != 4:
        print("Usage: 需要自备INPUT-scf/INCAR-scf文件")
        print("Usage: python3 perturb.py <number_of_perturbations> <poscar_file> abacus/vasp")
        sys.exit(1)
if __name__ == "__main__":
    main()

poscar_file = sys.argv[2]
if sys.argv[3] == "abacus":
    subprocess.run(f"echo '1\n 101\n  175 {poscar_file}' | atomkit", shell=True)
    ntype = dpdata.System(poscar_file, fmt="vasp/poscar").get_ntypes()
    with open(f"{poscar_file}.STRU", 'r') as file:
        upf_orb = ''.join([next(file) for _ in range(2 * ntype + 3)])
else:
    None

perturbations = int(sys.argv[1])
original_cwd = os.getcwd()
perturb_dic = f'perturb-{poscar_file}'
os.makedirs(perturb_dic, exist_ok=True)
shutil.copyfile(poscar_file, os.path.join(perturb_dic, 'POSCAR'))
os.chdir(perturb_dic)
perturbed_system = dpdata.System('POSCAR').perturb(pert_num=perturbations,           #生成个数
                                                    cell_pert_fraction = 0.03,       #晶胞扰动比例
                                                    atom_pert_distance = 0.1,        #原子扰动距离
                                                    atom_pert_style = 'uniform',     #原子扰动方式：uniform, normal, const
                                                    atom_pert_prob = 1.0,            #原子数扰动比例
                                                    #elem_pert_list = None            #指定元素扰动列表, 如['O', 'Si']     #可能在0.25版本中开始支持或根据2025/6/16的pr进行手动更改
                                                    )
for j in range(perturbations):
    train_directory = f'train-{poscar_file}-{j+1}'
    os.makedirs(train_directory, exist_ok=True)
    poscar_filename = f'POSCAR{j+1}'
    perturbed_system.to_vasp_poscar(poscar_filename, frame_idx=j)
    shutil.move(poscar_filename, os.path.join(train_directory, 'POSCAR'))
    if sys.argv[3] == "abacus":
        d_poscar = dpdata.System(os.path.join(train_directory, 'POSCAR'), fmt="vasp/poscar")
        d_poscar.to("abacus/stru", os.path.join(train_directory, "STRU"))
        with open(os.path.join(train_directory, "STRU"), 'r') as file:
            lines = file.readlines()
        lines[:ntype+1] = upf_orb
        with open(os.path.join(train_directory, "STRU"), 'w') as file:
            file.writelines(lines)
    else:
        None
os.chdir('..')

folder_path = os.path.join(original_cwd, perturb_dic)
if os.path.isdir(folder_path):
    for subfolder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder_name)
        if os.path.isdir(subfolder_path):
            if sys.argv[3] == "abacus":
                shutil.copy("INPUT-scf", os.path.join(subfolder_path, 'INPUT'))
                os.chdir(subfolder_path)
                subprocess.run('echo "3\n 301\n 0\n 101 STRU\n 0.03" | atomkit', shell=True)
            else:
                shutil.copy("INCAR-scf", os.path.join(subfolder_path, 'INCAR'))
                os.chdir(subfolder_path)
                subprocess.run('vaspkit -task 102 -kpr 0.03', shell=True)
            os.chdir(original_cwd)

