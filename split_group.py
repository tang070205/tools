from ase.io import write,read
import numpy as np

uc = read('POSCAR') 
cx, cy, cz = 15, 1, 1
struc = uc* (cx, cy, cz)
struc.set_pbc([True, True, True])

dic = 'x' #这里修改分组方向
group_cyclical = [3,3,3,3,3] #每组的周期数
ucl = uc.cell[0][0] if dic == 'x' else uc.cell[1][1] if dic == 'y' else uc.cell[2][2]
natoms = len(uc)*cy*cz if dic == 'x' else len(uc)*cx*cz if dic == 'y' else len(uc)*cy*cx
def split_group(input_list, ucl):
    return [n * ucl for n in input_list]
ncounts = [natoms * count for count in group_cyclical]
split = split_group(group_cyclical, ucl)
split = [0] + list(np.cumsum(split))
split[:-1] = [x - 0.001 for x in split[:-1]]
print("direction boundaries:", [round(l,2) for l in split])
print("atoms per group:", ncounts)

group_id = []
for atom in struc:
    n = atom.position[-3] if dic == 'x' else atom.position[-2] if dic == 'y' else atom.position[-1]
    for i in range(len(group_cyclical)):
        if n >= split[i] and n < split[i + 1]:
            group_index = i
    group_id.append(group_index)
struc.arrays["group"] = np.array(group_id)

write("model.xyz", struc)
