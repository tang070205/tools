from ase.io import read, write
import numpy as np
import sys
import os
import shutil
import subprocess

#这部分使用了叠加态大佬的代码，往网址：https://zhuanlan.zhihu.com/p/397866915
label='aimd'
os.system("find . -name vasprun.xml > xmllist")
os.system("if [ -f 'screen_tmp' ]; then rm screen_tmp;fi")
os.system("if [ -f 'aimd.xyz' ]; then rm aimd.xyz;fi")
os.system("if [ -f 'dump.xyz' ]; then rm dump.xyz;fi")
os.system("if [ -d 'train_folders' ]; then rm -r train_folders; fi")
for line in open('xmllist'):
    xml=line.strip('\n')
    print(xml)
    try:
        b=read(xml,index=":")
    except:
        b=read(xml.replace("vasprun.xml","OUTCAR"),index=":")
        print(xml.replace("vasprun.xml","OUTCAR"))
    os.system("grep -B 1 E0 "+xml.replace('vasprun.xml','OSZICAR')+" |grep -E 'DAV|RMM' |awk '{if($2>=60) print 0; else print 1}'>screen_tmp")
    screen=np.loadtxt("screen_tmp")
    try:
        len(screen)
    except:
        screen=[screen]
    for ind,i in enumerate(screen):
        if(i==1):
            xx,yy,zz,yz,xz,xy=-b[ind].calc.results['stress']*b[ind].get_volume()
            b[ind].info['virial']= np.array([(xx, xy, xz), (xy, yy, yz), (xz, yz, zz)])
            del b[ind].calc.results['stress']
            b[ind].pbc=True
            b[ind].info['config_type']=label
            write("aimd.xyz",b[ind],append=True)
    os.system("rm screen_tmp")
os.system("rm xmllist")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 aimd_OUTCAR_xyz2POSCAR.py <number_of_perturbations>")
        sys.exit(1)
if __name__ == "__main__":
    main()

number_structures = int(sys.argv[1])
original_cwd = os.getcwd()

with open("aimd.xyz", "r") as input_file:
    first_line = input_file.readline()
    structure_lines = int(first_line) + 2
    total_lines = sum(1 for _ in input_file) + 1
    structures_count = total_lines // structure_lines

output_lines = []
with open("aimd.xyz", "r") as input_file:
    lines = input_file.readlines()
    for i in range(number_structures):
        start_index = structure_lines * (structures_count // number_structures * (i+1) - 1)
        output_lines += lines[start_index:start_index + structure_lines]

with open("dump.xyz", "w") as output_file:
    output_file.writelines(output_lines)

def create_train_folders():
    if not os.path.exists('train_folders'):
        os.makedirs('train_folders')
    for i in range(number_structures):
        folder_name = f'aimd-{i+1}'
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
        group_filepath = os.path.join('train_folders', f'aimd-{i + 1}', f'aimd-{i + 1}.xyz')
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
    folder_name = f'aimd-{j+1}' 
    folder_path = os.path.join('train_folders', folder_name)  
    shutil.copyfile('INCAR-single', os.path.join(folder_path, 'INCAR'))  #INCAR-single是计算单点能的INCAR
    os.chdir(folder_path) 
    vaspkit_command = "vaspkit -task 102 -kpr 0.03"  # 此处采用vaspkit生成KPOINTS和POTCAR，
    subprocess.run(vaspkit_command, shell=True)  
    os.chdir(original_cwd)


