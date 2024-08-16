import os, sys, glob, math, random
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python in2exyz.py *.in #按in文件顺序转换xyz文件")
        print("Usage: python in2exyz.py *.in ratio #同样转换并随机抽取ratio比例的xyz文件")
        sys.exit(1)
if __name__ == "__main__":
    main()

in_file = sys.argv[1]
file_prefix = os.path.splitext(os.path.basename(in_file))[0]
out_file = file_prefix + '.xyz' 
extract_file = file_prefix + '_extract.xyz' 
surplus_file = file_prefix + '_surplus.xyz' 

with open(in_file, 'r') as infile, open('perstrucatoms.out', 'w') as psa, open('envipos.out', 'w') as evp, open('lattice.out', 'w') as lat:
    num_strucs = int(infile.readline().strip())
    lines = infile.readlines()  
    for line1 in lines[0:num_strucs]:  
        atoms = line1.split()[0]
        psa.write(atoms + '\n')
    for line2 in lines[num_strucs:]: 
        if len(line2.split()) == 9: 
            lat.write(line2)
        else:
            evp.write(line2)

psa = np.loadtxt('perstrucatoms.out')
psa_lines = [1]
pos_lines = [0]
for value in psa:
    psa_lines.append(psa_lines[-1] + int(value+1))
    pos_lines.append(pos_lines[-1] + int(value))

with open('envipos.out', 'r') as infile, open('envi.out', 'w') as file_ev, open('position.out', 'w') as file_pos:
    for line_number, line in enumerate(infile, start=1):
        if line_number in psa_lines:
            file_ev.write(line)
        else:
            file_pos.write(line)

with open('envi.out', 'r') as infile, open('energy.out', 'w') as file_e, open('virial.out', 'w') as file_v:
    for line in infile.readlines():
        columns = line.split()
        columns[4], columns[6] = columns[6], columns[4]
        file_e.write(f"{columns[0]}\n")
        if len(columns) >= 2:
            file_v.write(f"{' '.join(columns[1:7])}\n")
        else:
            zeros = ['0'] * (7 - len(columns))
            file_v.write(f"{' '.join(columns[1:] + zeros)}\n")

def write_strucs(struc_lines, xyz_file, perstrucatoms, energy, virial, lattice, position):
    for i in struc_lines:
        energy_line = energy.readline().strip()
        virial_line = virial.readline().strip()
        lattice_line = lattice.readline().strip()
        xyz_file.write(perstrucatoms.readline())
        xyz_file.write(f"Energy={energy_line} Virial=\"{virial_line}\" Lattice=\"{lattice_line}\" Config_type=infile-{i+1} Weight=1 Properties=species:S:1:pos:R:3:force:R:3\n")
        for _ in range(pos_lines[i], pos_lines[i+1]):
            xyz_file.write(position.readline())
        
with open('perstrucatoms.out', 'r') as perstrucatoms, open('energy.out', 'r') as energy, open('lattice.out', 'r') as lattice, open('virial.out', 'r') as virial, open('position.out', 'r') as position, open(out_file, 'w') as out_xyz:
    if len(sys.argv) == 2:
        write_strucs(range(num_strucs), out_xyz, perstrucatoms, energy, virial, lattice, position)
    elif len(sys.argv) == 3:
        write_strucs(range(num_strucs), out_xyz, perstrucatoms, energy, virial, lattice, position)
        ratio = float(sys.argv[2])
        extract_frames = sorted(random.sample(range(num_strucs), math.floor(num_strucs * ratio)))
        surplus_frames = [index for index in range(num_strucs) if index not in extract_frames]
        with open(extract_file, 'w') as extract, open(surplus_file, 'w') as surplus:
            write_strucs(extract_frames, extract, perstrucatoms, energy, virial, lattice, position)
            write_strucs(surplus_frames, surplus, perstrucatoms, energy, virial, lattice, position)

out_files = glob.glob('*.out') 
for file in out_files:
    os.remove(file) 

