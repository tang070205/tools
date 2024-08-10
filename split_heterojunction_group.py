from ase.io import write,read
import numpy as np
#除分组方向其他两方向长度要一致
uc1 = read('POSCAR1') 
uc2 = read('POSCAR2') 
cx1, cy1, cz1 = 10, 2, 2
cx2, cy2, cz2 = 10, 2, 2
struc1 = uc1* (cx1, cy1, cz1)
struc2 = uc2* (cx2, cy2, cz2)
dic = 'x'
struc_cell = struc1.get_cell()
if dic == 'x':
    struc_cell[0][0] = struc1.get_cell()[0, 0] + struc2.get_cell()[0, 0]
    struc2.positions[:, 0] += struc1.get_cell()[0, 0]
elif dic == 'y':
    struc_cell[1][1] = struc1.get_cell()[1, 1] + struc2.get_cell()[1, 1]
    struc2.positions[:, 1] += struc1.get_cell()[1, 1]
elif dic == 'z':
    struc_cell[2][2] = struc1.get_cell()[2, 2] + struc2.get_cell()[2, 2]
    struc2.positions[:, -1] += struc1.get_cell()[2, 2] 
struc = struc1 + struc2
struc.set_cell(struc_cell)

group_cyclical_1 = [4,2,2,2] #每组的周期数
group_cyclical_2 = [2,2,2,4] #每组的周期数
ucl1 = uc1.cell[0][0] if dic == 'x' else uc1.cell[1][1] if dic == 'y' else uc1.cell[2][2]
ucl2 = uc2.cell[0][0] if dic == 'x' else uc2.cell[1][1] if dic == 'y' else uc2.cell[2][2]
natoms1 = len(uc1)*cy1*cz1 if dic == 'x' else len(uc1)*cx1*cz1 if dic == 'y' else len(uc1)*cy1*cx1
natoms2 = len(uc2)*cy2*cz2 if dic == 'x' else len(uc1)*cx2*cz2 if dic == 'y' else len(uc2)*cy2*cx2
def split_group(input_list, ucl):
    return [n * ucl for n in input_list]
ncounts1 = [natoms1 * count for count in group_cyclical_1]
ncounts2 = [natoms2 * count for count in group_cyclical_2]
ncounts = ncounts1
#ncounts[-1] = ncounts1[-1] + ncounts2[0]
ncounts.extend(ncounts2[0:]) #如果想要界面在一组里，把0改成1，并取消上一行的注释
split1 = split_group(group_cyclical_1, ucl1)
split2 = split_group(group_cyclical_2, ucl2)
split = split1
#split[-1] = split1[-1] + split2[0]
split.extend(split2[0:]) #同上
split = [-1] + list(np.cumsum(split))
print("direction boundaries:", [round(l,2) for l in split])
print("atoms per group:", ncounts)

group_id = []
for atom in struc:
    n = atom.position[-3] if dic == 'x' else atom.position[-2] if dic == 'y' else atom.position[-1]
    for i in range(len(group_cyclical_1) + len(group_cyclical_2)): #如果想要界面在一组里，range()里面要减1
        if n > split[i] and n < split[i + 1]:
            group_index = i
    group_id.append(group_index)
struc.arrays["group"] = np.array(group_id)

write("model.xyz", struc)