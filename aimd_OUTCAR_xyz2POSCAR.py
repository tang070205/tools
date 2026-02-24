from ase.io import read, write
import numpy as np
import os, sys, shutil, subprocess

#这部分使用了叠加态大佬的代码，往网址：https://zhuanlan.zhihu.com/p/397866915
label='aimd'
os.system("find . -name vasprun.xml > xmllist")
os.system("if [ -f 'screen_tmp' ]; then rm screen_tmp;fi")
os.system("if [ -f 'aimd.xyz' ]; then rm aimd.xyz;fi")
os.system("if [ -f 'aimd_subset*.xyz' ]; then rm aimd_subset*.xyz;fi")
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
aimd = read("aimd.xyz", index=":")
interval = len(aimd) // number_structures
first_indices = len(aimd) - interval * (number_structures - 1) - 1
for i in range(number_structures):
    subset = aimd[first_indices + i * interval]
    write(f"aimd_subset_{interval}.xyz", subset, append=True)

if not os.path.exists('aimd_subset_train_folders'):
    os.makedirs('aimd_subset_train_folders')
for i in range(number_structures):
    folder_name = f'aimd_subset_{i+1}'
    folder_path = os.path.join('aimd_subset_train_folders', folder_name)
    os.makedirs(folder_path)

aimd_subset = read("aimd_subset_*.xyz", index=":")
for i in range(number_structures):
    subset_filepath = os.path.join('aimd_subset_train_folders', f'aimd_subset_{i+1}', f'subset.xyz')
    write(subset_filepath, aimd_subset[i])

os.chdir(original_cwd)
for i in range(number_structures):
    folder_name = f'aimd_subset_{i+1}' 
    folder_path = os.path.join('aimd_subset_train_folders', folder_name)  
    shutil.copy('INCAR-scf', os.path.join(folder_path, 'INCAR'))  #INCAR-scf是计算单点能的INCAR
    os.chdir(folder_path)
    write("POSCAR", read('subset.xyz', format="extxyz"))
    vaspkit_command = "vaspkit -task 102 -kpr 0.03"  # 此处采用vaspkit生成KPOINTS和POTCAR
    subprocess.run(vaspkit_command, shell=True)  
    os.chdir(original_cwd)
