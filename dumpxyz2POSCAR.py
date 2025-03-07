from ase.io import read, write
import os, sys, shutil, subprocess

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 aimd_OUTCAR_xyz2POSCAR.py <xyz_file> <number_of_perturbations> vasp/abacus")
        sys.exit(1)
if __name__ == "__main__":
    main()

if sys.argv[3] == 'abacus':
    import dpdata
    subprocess.run(f"echo '1\n 101\n  175 POSCAR' | atomkit", shell=True)
    ntype = dpdata.System('POSCAR', fmt="vasp/poscar").get_ntypes()
    with open("POSCAR.STRU", 'r') as file:
         upf_orb = ''.join([next(file) for _ in range(2 * ntype + 3)])
else:
    None

number_structures = int(sys.argv[2])
original_cwd = os.getcwd()

with open(sys.argv[1], "r") as input_file:
    first_line = input_file.readline()
    try:
        structure_lines = int(first_line) + 2
    except ValueError:
        print("第一行内容不能转换为整数")
    total_lines = sum(1 for _ in input_file) + 1
    structures_count = total_lines // structure_lines

output_lines = []
with open(sys.argv[1], "r") as input_file:
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
    with open(sys.argv[1], 'r') as file:
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
        if sys.argv[1] == 'abacus':
            d_poscar = dpdata.System('POSCAR', fmt="vasp/poscar")
            d_poscar.to("abacus/stru", "STRU")
            with open("STRU", 'r') as file:
                lines = file.readlines()
                lines[:ntype+1] = upf_orb
                with open("STRU", 'w') as file:
                    file.writelines(lines)
        else: 
            None

create_train_folders()
split_xyz()
convert_xyz_to_poscar()

os.chdir(original_cwd)
for j in range(number_structures):
    folder_name = f'activate-learning-{j+1}' 
    folder_path = os.path.join('train_folders', folder_name)  
    if sys.argv[1] == 'abacus':
        shutil.copy("INPUT-scf", os.path.join(folder_path, 'INPUT'))
        os.chdir(folder_path)
        atomkit_command = 'echo "3\n 301\n 0\n 101 STRU\n 0.03" | atomkit'
        subprocess.run(atomkit_command, shell=True)
    else: 
        shutil.copy("INCAR-scf", os.path.join(folder_path, 'INCAR'))
        os.chdir(folder_path)
        vaspkit_command = "vaspkit -task 102 -kpr 0.03" 
        subprocess.run(vaspkit_command, shell=True) 
    os.chdir(original_cwd)

