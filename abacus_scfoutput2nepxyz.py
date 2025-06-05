import os, sys, json
from ase.io import read, write

def main():
    if len(sys.argv) != 3:
        print("Usage: python single-abacus2nep.py <dir> <xyz>")
        sys.exit(1)
if __name__ == "__main__":
    main()

def get_scf_info(input_file):
    scf_nmax = None
    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()
            if 'scf_nmax' in line:
                scf_nmax = int(line.split()[1])
    return scf_nmax

for root, dirs, files in os.walk(sys.argv[1]):
    scf_count = 0
    if "running_scf.log" in files:
        input_file, log_file = os.path.join(root, "INPUT"), os.path.join(root, "running_scf.log")
        scf_nmax = get_scf_info(input_file)
        with open(log_file, 'r') as file:
            for line in file:
                line = line.strip()
                scf_count += line.count("ALGORITHM")
        if scf_count == scf_nmax:
            print(f"Directory {root} has not completed the calculation or has not converged")
            continue
        atoms = read(log_file, format='abacus-out') #pip install git+https://gitlab.com/1041176461/ase-abacus.git
        natoms = len(atoms)
        cell = atoms.get_cell()[0] + atoms.get_cell()[1] + atoms.get_cell()[2]
        energy = atoms.get_potential_energy()
        if 'TOTAL-STRESS' in open(log_file, 'r').read():
            virial = [f"{-s:.10f}" for s in atoms.get_stress()*atoms.get_volume()]
        else:
            print("This structure does not calculate stress, please add cal_stress in INPUT")
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        forces = atoms.get_forces()
    else:
        input_file, json_file = os.path.join(root, "INPUT"), os.path.join(root, "abacus.json")
        scf_nmax = get_scf_info(input_file)
        with open(json_file, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        scf_count = len(data['output'][0]['scf'])
        if scf_count == scf_nmax:
            print(f"Directory {root} has not completed the calculation or has not converged")
            continue
        natoms = data['init']['natom']
        cell = data['output'][0]['cell'][0] + data['output'][0]['cell'][1] + data['output'][0]['cell'][2]
        symbols = data['init']['label']
        forces = data['output'][0]['force']
        energy = data['output'][0]['energy']
        positions = data['output'][0]['coordinate']
        if 'stress' in open(json_file, 'r').read():
            volume = np.abs(np.dot(cell[0], np.cross(cell[1], cell[2])))
            stress = [[s * (1602.1766208 / volume) for s in row] for row in data['output'][0]['stress']]
            virial = stress[0] +  stress[1] +  stress[2]
        else:
            print("This structure does not calculate stress, please add cal_stress in INPUT")

    with open(sys.argv[2], 'w') as f:
        f.write(f"{natoms}\n")
        if stress:
            if os.path.exists(os.path.join(root, "running_scf.log")):
                f.write(f'energy={energy} Lattice="{cell[0]} {cell[1]} {cell[2]} {cell[3]} {cell[4]} {cell[5]} {cell[6]} {cell[7]} {cell[8]}" Virial="{virial[0]} {virial[5]} {virial[4]} {virial[5]} {virial[1]} {virial[3]} {virial[4]} {virial[3]} {virial[2]}" config_type={root} Properties=species:S:1:pos:R:3:forces:R:3\n')
            else:
                f.write(f'energy={energy} Lattice="{cell[0]} {cell[1]} {cell[2]} {cell[3]} {cell[4]} {cell[5]} {cell[6]} {cell[7]} {cell[8]}" Virial="{virial[0]} {virial[1]} {virial[2]} {virial[3]} {virial[4]} {virial[5]} {virial[6]} {virial[7]} {virial[8]}" config_type={root} Properties=species:S:1:pos:R:3:forces:R:3\n')
        else:
            f.write(f'energy={energy} Lattice="{cell[0]} {cell[1]} {cell[2]} {cell[3]} {cell[4]} {cell[5]} {cell[6]} {cell[7]} {cell[8]}" config_type={root} Properties=species:S:1:pos:R:3:forces:R:3\n')
        for i in range(natoms):
            f.write(f"{symbols[i]}     {positions[i][0]:10.10f}     {positions[i][1]:10.10f}     {positions[i][2]:10.10f}     {forces[i][0]:10.10f}     {forces[i][1]:10.10f}     {forces[i][2]:10.10f}\n")  

'''
#需要保留到小数点更多位请使用上面代码, 仅针对log文件
for root, dirs, files in os.walk(sys.argv[1]):
    if "running_scf.log" in files:
        log_file_path = os.path.join(root, "running_scf.log")
        atoms = read(log_file_path, format='abacus-out') #pip install git+https://gitlab.com/1041176461/ase-abacus.git
    atoms.set_momenta(None)
    atoms.set_initial_magnetic_moments(None)
    if 'TOTAL-STRESS' in open(log_file, 'r').read():
        xx,yy,zz,yz,xz,xy = -atoms.get_stress()*atoms.get_volume()
        atoms.info['virial']= np.array([(xx, xy, xz), (xy, yy, yz), (xz, yz, zz)])
        del atoms.get_stress()
    else:
        print("This structure does not calculate stress, please add cal_stress in INPUT")
    atoms.info['config_type'] = {root}
    write(sys.argv[2], atoms, format='extxyz', append=True)
'''