from ase.io import read, write
import numpy as np
import random
random.seed(42)

periodic_order = 0 #如果是1，每个子列表只能由一个数
num_structures = 1
uc = read('POSCAR')
cx, cy, cz = 4, 2, 1
atoms = uc * (cx, cy, cz)
#uc.arrays['spacegroup_kinds'] = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]
#atoms.arrays['spacegroup_kinds'] = np.tile(uc.arrays['spacegroup_kinds'], (cx * cy * cz))
target_symbols = ['Mo']  # , 'Si', 'N']
replacement_symbols_list = [['W']]  # , 'Ti']], ['Al', 'Ga', 'Ge'], ['S', 'Se', 'Te', 'P', 'As']]
num_replacements_list = [[8]]  # , 2], [2, 4, 3], [4, 5, 2, 1, 3]]
extend = None #atoms.arrays.get('spacegroup_kinds')
extend_infos = []
if extend is not None:
    print(uc.get_chemical_symbols())
    print(uc.arrays.get('spacegroup_kinds'))
    extend_infos = [[2]] #对应uc.arrays['spacegroup_kinds']里的数
if not extend_infos:
    extend_infos = [None] * len(target_symbols)
out_file = 'replace_structures.xyz'

old_symbols = atoms.get_chemical_symbols()
structures = []
for _ in range(num_structures):
    temp_atoms = atoms.copy()
    temp_symbols = temp_atoms.get_chemical_symbols()

    for target_symbol, replacements, nums, extend_info in zip(target_symbols, replacement_symbols_list, num_replacements_list, extend_infos):
        target_indices = [i for i, symbol in enumerate(temp_symbols) if symbol == target_symbol]
        if extend is not None:
            target_indices = [i for i in target_indices if extend[i] in extend_info]

        if len(target_indices) < sum(nums):
            raise ValueError(f"The number of symbol '{target_symbol}' is less than the number that needs to be replaced {sum(nums)}")

        if periodic_order == 1:
            num_per_extend = len(target_indices) // (cx*cy*cz)
            replacements_per_extend = sum(nums)  // (cx*cy*cz)

            grouped_indices = [target_indices[i:i + num_per_extend] for i in range(0, len(target_indices), num_per_extend)]
            selected_indices = []
            for group in grouped_indices:
                if len(group) > replacements_per_extend:
                    selected_indices.extend(random.sample(group, replacements_per_extend))
                else:
                    selected_indices.extend(group)
        else:
            selected_indices = target_indices

        random.shuffle(selected_indices)
        start_index = 0
        for replacement_symbol, num_replacements in zip(replacements, nums):
            replacement_indices = selected_indices[start_index:start_index + num_replacements]
            start_index += num_replacements

            new_symbols = temp_symbols[:]
            for index in replacement_indices:
                new_symbols[index] = replacement_symbol
            temp_atoms.set_chemical_symbols(new_symbols)
            temp_symbols = new_symbols

    if extend is not None:
        del temp_atoms.arrays['spacegroup_kinds']
    structures.append(temp_atoms)

write(out_file, structures, format='extxyz')