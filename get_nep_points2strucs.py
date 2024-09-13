import sys
import numpy as np
from ase.io import read, write

def main():
    if len(sys.argv) < 5:
        print("Usage: differ差值使用DFT-NEP")
        print("Usage: 目前有一个缺点,训练时如果不是fullbatch,需要将所有结构先预测一下,用那个out文件才能对的上xyz的结构顺序")
        print("Usage: python get_energy_points.py direct <*.xyz> <*.out> x1_start x1_end x2_start x2_end...")
        print("Usage: python get_energy_points.py differ <*.xyz> <*.out> <dif_value-> <dif_value+>")
        print("Usage: python get_energy_points.py rmse <*.xyz> <*.out> <max_rmse_strucs>")
        sys.exit(1)
if __name__ == "__main__":
    main()

strucs = read(sys.argv[2], ":")
atom_counts = [0]
for atoms in strucs:
    atom_counts.append(atom_counts[-1] + len(atoms))

fout = np.loadtxt(sys.argv[3])
if 'energy' in sys.argv[3]:
    dir_value = fout[:,1]
    dif_value = fout[:,1]-fout[:,0]
    rmse = np.sqrt((fout[:,0]-fout[:,1])**2)
elif 'force' in sys.argv[3]:
    dir_value = fout[:,3:6]
    dif_value = fout[:,3:6]-fout[:,0:3]
    rmse = np.sqrt((fout[:,3:6]-fout[:,0:3])**2)
    print(rmse)
elif 'virial' in sys.argv[3]:
    dir_value = fout[:,6:12]
    dif_value = fout[:,6:12]-fout[:,0:6]
    rmse = np.sqrt((fout[:, 0:6] - fout[:, 6:12])**2)
elif 'stress' in sys.argv[3]:
    dir_value = fout[:,6:12]
    dif_value = fout[:,6:12]-fout[:,0:6]
    rmse = np.sqrt((fout[:, 0:6] - fout[:, 6:12])**2)

def force_struc_ids(di_ids, atom_counts):
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
    if 'force' in sys.argv[3]:
        struc_id = sorted(set(force_struc_ids(direct_id, atom_counts)))
    else:
        struc_id = sorted(set(direct_id))

elif sys.argv[1] == 'differ':
    small_value = float(sys.argv[4])
    large_value = float(sys.argv[5])
    differ_id = np.where((dif_value > large_value) | (dif_value < small_value))[0].tolist() 
    if 'force' in sys.argv[3]:
        struc_id = sorted(set(force_struc_ids(differ_id, atom_counts)))
    else:
        struc_id = sorted(set(differ_id))

elif sys.argv[1] == 'rmse':
    max_rmse_strucs = int(sys.argv[4])
    if rmse.shape[1] > 1:
        max_rmse = sorted([(max(row), index) for index, row in enumerate(rmse)])[::-1]
    else:
        max_rmse = sorted([(row, index) for index, row in enumerate(rmse)])[::-1]
    rmse_id = [index for _, index in max_rmse]
    if 'force' in sys.argv[3]:
        struc_id = sorted(set(np.unique(force_struc_ids(rmse_id, atom_counts)[:max_rmse_strucs])))
    else:
        struc_id = sorted(set(rmse_id[:max_rmse_strucs]))
        
write('deviate.xyz', [strucs[i] for i in struc_id], format='extxyz', write_results=False)
retain_id = [i for i in range(len(strucs)) if i not in struc_id]
write('reserve.xyz', [strucs[i] for i in retain_id], format='extxyz', write_results=False)
