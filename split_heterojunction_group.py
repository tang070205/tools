from ase.io import write,read
import numpy as np
#除分组方向其他两方向长度要一致或者就是沿用第一个结构的两方向长度
dir = 'z'
merge_interface = 0
vacuums = [1, 1]
ucs = [ ('Si.vasp', 8, 8, 120, [8,20,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]),
        ('BAs.vasp', 9, 9, 21, [3,3,3,3,3,3,3]),
        ('AlN.vasp', 14, 8, 120, [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,20])]

def get_supercell(file, cx, cy, cz):
    uc = read(file)
    expanded_structure = uc * (cx, cy, cz)
    return expanded_structure

strucs, gatoms, ncounts, gsplit, gsplits, splits = [], [], [], [], [], []
for file, cx, cy, cz, _ in ucs:
    supercell = get_supercell(file, cx, cy, cz)
    strucs.append(supercell)

def add_vacandsite(dir, f_cell, c_struc, vacuum):
    dex = 0 if dir == 'x' else 1 if dir == 'y' else 2
    f_cell[dex][dex] += (c_struc.get_cell()[dex, dex] + vacuum)
    c_struc.positions[:, dex] += (f_cell[dex][dex] - c_struc.get_cell()[dex, dex])

final_struc = strucs[0]
final_cell = final_struc.get_cell().copy()
for i in range(1, len(strucs)):
    current_struc = strucs[i]
    add_vacandsite(dir, final_cell, current_struc, vacuums[i - 1])
    final_struc += current_struc
final_struc.set_cell(final_cell)

def get_natoms(file, cx, cy, cz):
    return len(file)*cy*cz if dir == 'x' else cx*len(file)*cz if dir == 'y' else cx*cy*len(file)

for file, cx, cy, cz, group_cyclical in ucs:
    atoms = get_natoms(file, cx, cy, cz)
    natoms = [atoms * count for count in group_cyclical]
    gatoms.append(natoms)

for i, natoms in enumerate(gatoms):
    if i == 0:
        ncounts.extend(natoms)
    else:
        if merge_interface == 1:
            ncounts[-1] += natoms[0]
            ncounts.extend(natoms[1:])
        else:
            ncounts.extend(natoms)


def get_cell_length(file):
    uc = read(file)
    return uc.cell[0][0] if dir == 'x' else uc.cell[1][1] if dir == 'y' else uc.cell[2][2]
def split_group(input_list, ucl):
    return [n * ucl for n in input_list]

for file, _, _, _, group_cyclical in ucs:
    ucl = get_cell_length(file)
    split = split_group(group_cyclical, ucl)
    gsplit.append(split)

for i, split in enumerate(gsplit):
    if i == 0:
        splits.extend(split)
    else:
        if merge_interface == 1:
            splits[-1] += split[0] + vacuums[i-1]
            splits.extend([x + vacuums[i-1] for x in split[1:]])
        else:
            splits.extend([x + vacuums[i-1] for x in split])

gsplits = [0] + list(np.cumsum(splits))
gsplits[:-1] = [x - 0.001 for x in gsplits[:-1]]
print("direction boundaries:", [round(l,2) for l in gsplits])
print("atoms per group:", ncounts)

group_id = []
for atom in final_struc:
    n = atom.position[-3] if dir == 'x' else atom.position[-2] if dir == 'y' else atom.position[-1]
    for i in range(len(gsplits)-1):
        if n >= gsplits[i] and n < gsplits[i + 1]:
            group_index = i
    group_id.append(group_index)
final_struc.arrays["group"] = np.array(group_id)

write("model.xyz", final_struc)