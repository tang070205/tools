from ase.io import read, write
import numpy as np
import sys
import os
import shutil
import subprocess
from ovito.io import import_file, export_file
from ovito.modifiers import WrapPeriodicImagesModifier

if(len(sys.argv)==2):
    label=sys.argv[1]
else:
    label='low'
os.system("find . -name vasprun.xml > xmllist")
os.system("if [ -f 'screen_tmp' ]; then rm screen_tmp;fi")
os.system("if [ -f 'dump.xyz' ]; then rm dump.xyz;fi")
for line in open('xmllist'):
    xml=line.strip('\n')
    print(xml)
    try:
        b=read(xml,index=":")
    except:
        b=read(xml.replace("vasprun.xml","OUTCAR"),index=":")
        print(xml.replace("vasprun.xml","OUTCAR"))
    # check convergence for each ionic step
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

output_lines = []
with open("aimd.xyz", "r") as input_file:
    lines = input_file.readlines()
    for i in range(50):   #此部分是根据行数分离，大家可能需要计算机
        start_index = 28942 + i * 580
        output_lines += lines[start_index : start_index + 58]

with open("dump.xyz", "w") as output_file:
    output_file.writelines(output_lines)

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
