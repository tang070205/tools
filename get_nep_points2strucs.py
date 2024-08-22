import sys
import numpy as np
from ase.io import read, write

def main():
    if len(sys.argv) < 2:
        print("Usage: python get_energy_points.py direct <*.out> <*.xyz> x1_start x1_end x2_start x2_end...")
        print("Usage: python get_energy_points.py differ <*.out> <*.xyz> <dif_value-> <dif_value+>")
        sys.exit(1)
if __name__ == "__main__":
    main()

strucs = read(sys.argv[3], ":")
atom_counts = [0]
for atoms in strucs:
    atom_counts.append(atom_counts[-1] + len(atoms))
print(atom_counts)
fout = np.loadtxt(sys.argv[2])
if 'energy' in sys.argv[2]:
    dir_value = fout[:,1]
    dif_value = fout[:,1]-fout[:,0]
elif 'force' in sys.argv[2]:
    dir_value = fout[:,3:6]
    dif_value = fout[:,3:6]-fout[:,0:3]
elif 'virial' in sys.argv[2]:
    dir_value = fout[:,6:12]
    dif_value = fout[:,6:12]-fout[:,0:6]
elif 'stress' in sys.argv[2]:
    dir_value = fout[:,6:12]
    dif_value = fout[:,6:12]-fout[:,0:6]

def generate_struc_ids(di_ids, atom_counts):
    struc_ids = []
    for i, di_id in enumerate(di_ids):
        for j in range(len(atom_counts) - 1):
            if atom_counts[j] <= di_id < atom_counts[j+1]:
                struc_ids.append(j)
                break 
    return struc_ids
    
if sys.argv[1] == 'direct':
    direct_id = []
    for i in range(4, len(sys.argv), 2):
        small_value = float(sys.argv[i])
        large_value = float(sys.argv[i+1])
        dir_id = np.where((dir_value > small_value) & (dir_value < large_value))[0]
        direct_id.extend(dir_id)
    print(direct_id)
    if len(dir_value[0]) == 3:
        struc_id = sorted(set(generate_struc_ids(direct_id, atom_counts)))
    else:
        struc_id = sorted(set(direct_id))

elif sys.argv[1] == 'differ':
    small_value = float(sys.argv[4])
    large_value = float(sys.argv[5])
    differ_id = np.where((dif_value > large_value) | (dif_value < small_value))[0].tolist() 
    print(differ_id)
    if len(dif_value[0]) == 3:
        struc_id = sorted(set(generate_struc_ids(differ_id, atom_counts)))
    else:
        struc_id = sorted(set(differ_id))

print(struc_id)
write('deviate.xyz', [strucs[i] for i in struc_id], format='extxyz', write_results=False)
retain_id = [i for i in range(len(strucs)) if i not in struc_id]
write('reserve.xyz', [strucs[i] for i in retain_id], format='extxyz', write_results=False)