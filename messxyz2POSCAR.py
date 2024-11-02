from ase.io import read, write
import os, sys, shutil, subprocess

def main():
    if len(sys.argv) != 2:
        print("Usage: python messxyz2POSCAR.py <xyz_file>")
        sys.exit(1)
if __name__ == "__main__":
    main()

original_cwd = os.getcwd()
strucs = read(sys.argv[1], ":")
num_strucs = len(strucs)
struc_lines = [0]
for atoms in strucs:
    struc_lines.append(struc_lines[-1] + len(atoms)+2)

def create_mess_folders():
    if not os.path.exists('mess_strucs_folders'):
        os.makedirs('mess_strucs_folders')
    for i in range(num_strucs):
        folder_name = f'activate-learning-{i+1}'
        folder_path = os.path.join('mess_strucs_folders', folder_name)
        os.makedirs(folder_path)

def split_xyz():
    with open(sys.argv[1], 'r') as file:
        lines = file.readlines()  
    for i in range(num_strucs):
        group_lines = lines[struc_lines[i]:struc_lines[i+1]]
        group_filepath = os.path.join('mess_strucs_folders', f'activate-learning-{i + 1}', f'activate-learning-{i + 1}.xyz')
        with open(group_filepath, 'w') as group_file:
            group_file.writelines(group_lines)

def convert_xyz_to_poscar():
    mess_folder_path = os.path.join(os.getcwd(), 'mess_strucs_folders')
    folders = os.listdir(mess_folder_path)
    for folder_name in folders:
        folder_path = os.path.join(mess_folder_path, folder_name)
        os.chdir(folder_path)
        xyz_file = next((f for f in os.listdir(folder_path) if f.endswith('.xyz')), None)
        write("POSCAR", read(xyz_file, format="extxyz"))

create_mess_folders()
split_xyz()
convert_xyz_to_poscar()

os.chdir(original_cwd)
for j in range(num_strucs):
    folder_name = f'activate-learning-{j+1}' 
    folder_path = os.path.join('mess_strucs_folders', folder_name)  
    shutil.copyfile('INCAR-single', os.path.join(folder_path, 'INCAR'))  #INCAR-single是计算单点能的INCAR
    os.chdir(folder_path) 
    vaspkit_command = "vaspkit -task 102 -kpr 0.03"  # 此处采用vaspkit生成KPOINTS和POTCAR，
    subprocess.run(vaspkit_command, shell=True)  
    os.chdir(original_cwd)

