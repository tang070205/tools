from ase.io import read, write
import os, sys, shutil, subprocess

def main():
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print("Usage: python3 mdxyz2POSCAR.py <xyz_file> vasp/abacus (<number_of_perturbations>)")
        sys.exit(1)
if __name__ == "__main__":
    main()

original_cwd = os.getcwd()
infile = os.path.splitext(os.path.basename(sys.argv[1]))[0]
inxyz = read(sys.argv[1], index=":")
if len(sys.argv) == 3:
    os.system(f"cp {sys.argv[1]} {infile}_activate_learning.xyz")
    number_structures = len(inxyz)
else:
    number_structures = int(sys.argv[3])
    interval = len(inxyz) // number_structures
    first_indices = len(inxyz) - interval * (number_structures - 1) - 1
    for i in range(number_structures):
        subset = inxyz[first_indices + i * interval]
        write(f'{infile}_activate_learning.xyz', subset, append=True)

if not os.path.exists(f'{infile}_activate_learning_folders'):
    os.makedirs(f'{infile}_activate_learning_folders')
for i in range(number_structures):
    folder_name = f'activate_learning_{i+1}'
    folder_path = os.path.join(f'{infile}_activate_learning_folders', folder_name)
    os.makedirs(folder_path)

alxyz = read(f'{infile}_activate_learning.xyz', index=":")
for i in range(number_structures):
    al_filepath = os.path.join(f'{infile}_activate_learning_folders', f'activate_learning_{i+1}', 'activate_learning.xyz')
    write(al_filepath, alxyz[i])

os.chdir(original_cwd)
for i in range(number_structures):
    folder_name = f'activate_learning_{i+1}' 
    folder_path = os.path.join(f'{infile}_activate_learning_folders', folder_name)  
    if sys.argv[2] == 'abacus':
        shutil.copy("INPUT-scf", os.path.join(folder_path, 'INPUT'))
    else:
        shutil.copy('INCAR-scf', os.path.join(folder_path, 'INCAR'))
    os.chdir(folder_path)
    write("POSCAR", read('activate_learning.xyz', format="extxyz"))
    if sys.argv[2] == 'abacus':
        import dpdata
        subprocess.run(f"echo '1\n 101\n  175 POSCAR' | atomkit", shell=True)
        ntype = dpdata.System('POSCAR', fmt="vasp/poscar").get_ntypes()
        with open("POSCAR.STRU", 'r') as file:
            upf_orb = ''.join([next(file) for _ in range(2 * ntype + 3)])
        
        d_poscar = dpdata.System('POSCAR', fmt="vasp/poscar")
        d_poscar.to("abacus/stru", "STRU")
        with open("STRU", 'r') as file:
            lines = file.readlines()
            lines[:ntype+1] = upf_orb
            with open("STRU", 'w') as file:
                file.writelines(lines)
        commandkit = 'echo "3\n 301\n 0\n 101 STRU\n 0.03" | atomkit'
    else:
        commandkit = 'vaspkit -task 102 -kpr 0.03'
    subprocess.run(commandkit, shell=True)  
    os.chdir(original_cwd)
