import sys
import numpy as np

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

with open(sys.argv[2], 'r') as file:
    struc_atom = []
    for line in file:
        atom_line = line.strip()
        if len(atom_line.split()) == 1 and atom_line.isdigit():
            struc_atom.append(int(atom_line))
atom_counts, struc_lines = [0], [0]
for i in range(len(struc_atom)):
    atom_counts.append(atom_counts[-1] + struc_atom[i])
    struc_lines.append(struc_lines[-1] + struc_atom[i]+2)

fout = np.loadtxt(sys.argv[3])
if 'energy' in sys.argv[3]:
    dir_value = fout[:,1]
    dif_value = fout[:,1]-fout[:,0]
    rmse = np.sqrt((fout[:,0]-fout[:,1])**2)
elif 'force' in sys.argv[3]:
    dir_value = fout[:,3:6]
    dif_value = fout[:,3:6]-fout[:,0:3]
    rmse = []
    for i in range(len(atom_counts) - 1):
        strucs_forces = fout[atom_counts[i]:atom_counts[i + 1], :]
        strucs_rmse = np.sqrt(np.mean((strucs_forces[:,3:6]-strucs_forces[:,0:3])**2))
        rmse.append(strucs_rmse)
elif 'virial' in sys.argv[3]:
    dir_value = fout[:,6:12]
    dif_value = fout[:,6:12]-fout[:,0:6]
    rmse = np.sqrt(np.sum((fout[:, 0:6] - fout[:, 6:12])**2, axis=1))
elif 'stress' in sys.argv[3]:
    dir_value = fout[:,6:12]
    dif_value = fout[:,6:12]-fout[:,0:6]
    rmse = np.sqrt(np.sum((fout[:, 0:6] - fout[:, 6:12])**2, axis=1))

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
    rmse_id =np.argsort(np.array(rmse))
    struc_id = sorted(set(rmse_id[-max_rmse_strucs:]))

with open(sys.argv[2], 'r') as file:
    lines = file.readlines()
with open('deviate.xyz', 'w') as file1:
    for i, j in enumerate(struc_id):
        file1.writelines(lines[struc_lines[j]:struc_lines[j+1]])
retain_id = [i for i in range(len(struc_atom)) if i not in struc_id]
with open('reserve.xyz', 'w') as file2:
    for i, j in enumerate(retain_id):
        file2.writelines(lines[struc_lines[j]:struc_lines[j+1]])


