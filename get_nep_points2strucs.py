import sys
import numpy as np

def main():
    if len(sys.argv) < 5:
        print("Usage: differ差值使用DFT-NEP")
        print("Usage: 目前有一个缺点,训练时如果不是fullbatch,需要将所有结构先预测一下,用那个out文件才能对的上xyz的结构顺序")
        print("Usage: python get_energy_points.py direct <*.xyz> <*.out> nep/dft <x1/y1_start> <x1/y1_end> <x2/y2_start> <x2/y2_end>...")
        print("Usage: python get_energy_points.py allin <*.xyz> <*.out> nep/dft <x/y_start> <x/y_end>")
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
half_columns = int(fout.shape[1]//2)
def calc_rmse(fout):
    if 'energy' in sys.argv[3]:
        return np.sqrt(np.mean((fout[:,:half_columns]-fout[:,half_columns:])**2))
    elif 'force' in sys.argv[3]:
        rmse = []
        for i in range(len(atom_counts) - 1):
            strucs_forces = fout[atom_counts[i]:atom_counts[i + 1], :]
            strucs_rmse = np.sqrt(np.mean((strucs_forces[:,:half_columns]-strucs_forces[:,half_columns:])**2))
            rmse.append(strucs_rmse)
        return rmse
    elif 'virial' or 'stress' in sys.argv[3]:
        return np.sqrt(np.sum((fout[:,:half_columns]-fout[:,half_columns:])**2, axis=1))

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
    dir_value = fout[:,:half_columns] if sys.argv[4] == 'nep' else fout[:,half_columns:]
    for i in range(5, len(sys.argv), 2):
        small_value = float(sys.argv[i])
        large_value = float(sys.argv[i+1])
        dir_id = np.where((dir_value > small_value) & (dir_value < large_value))[0]
        direct_id.extend(dir_id)
    if 'force' in sys.argv[4]:
        struc_id = sorted(set(force_struc_ids(direct_id, atom_counts)))
    else:
        struc_id = sorted(set(direct_id))

elif sys.argv[1] == 'allin':
    allin_id = []
    if 'force' in sys.argv[3]:
        for i in range(len(atom_counts) - 1):
            strucs_forces = fout[atom_counts[i]:atom_counts[i + 1], :]
            force_values = strucs_forces[:,:half_columns]if sys.argv[4] == 'nep' else strucs_forces[:,half_columns:]
            if np.all(np.logical_and(force_values > float(sys.argv[4]), force_values < float(sys.argv[5]))):
                allin_id.append(i)
    else:
        for idx, row in enumerate(fout):
            allin_values = fout[:,:half_columns]if sys.argv[4] == 'nep' else fout[:,half_columns:]
            if np.all(np.logical_and(allin_values > float(sys.argv[5]), allin_values < float(sys.argv[5]))):
                allin_id.append(idx)
    struc_id = sorted(set(allin_id))

elif sys.argv[1] == 'differ':
    dif_value = fout[:,:half_columns] - fout[:,half_columns:]
    small_value = float(sys.argv[4])
    large_value = float(sys.argv[5])
    differ_id = np.where((dif_value > large_value) | (dif_value < small_value))[0].tolist() 
    if 'force' in sys.argv[3]:
        struc_id = sorted(set(force_struc_ids(differ_id, atom_counts)))
    else:
        struc_id = sorted(set(differ_id))

elif sys.argv[1] == 'rmse':
    rmse = calc_rmse(fout)
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

