import os, glob, math
import numpy as np
from pylab import *


element_force = 0   # 0不画元素力，1画元素力
component = '0'   # '0'不画分量, 'force'画力分量, 'dipole'画偶极矩分量, 'virial'画virial分量等等(不包含BEC)
train_colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan'] #力的话各取前三个
test_colors = ['magenta', 'lime', 'teal', 'navy', 'olive', 'maroon']
def generate_colors(data):
    if three_six_component == 0 or data == 'energy' or data == 'bec':
        return 'deepskyblue', 'orange'   #不画三六分量，前是训练集颜色，后是测试集颜色
    else:
        if data in ['force', 'dipole']:
            return train_colors[:3], test_colors[:3]
        else:
            return train_colors, test_colors

def set_tick_params():
    tick_params(axis='x', which='both', direction='in', top=True, bottom=True)
    tick_params(axis='y', which='both', direction='in', left=True, right=True)

def get_counts2two(out_file):
    file_nums = int(out_file.shape[1]//2)
    new_nep, new_dft = out_file[:, :file_nums].flatten(), out_file[:, file_nums:].flatten()
    return np.column_stack((new_nep, new_dft))

def calc_r2_rmse(out_file):
    file_columns = int(out_file.shape[1]//2)
    numerator = np.sum((out_file[:, :file_columns] - out_file[:, file_columns:]) ** 2)
    denominator = np.sum((out_file[:, :file_columns] - np.mean(out_file[:, :file_columns])) ** 2)
    r2_data = 1.0 if denominator == 0 else 1 - (numerator / denominator)
    rmse_origin = np.sqrt(np.mean((out_file[:, :file_columns]-out_file[:, file_columns:])**2))
    rmse_data = rmse_origin * 1000 if rmse_origin < 1 else rmse_origin
    return rmse_origin, rmse_data, r2_data

def get_range(data, data_file):
    if data == 'energy':
        return np.floor(data_file.min() * 10) / 10, np.ceil(data_file.max() * 10) / 10
    else:
        return np.floor(data_file.min()), np.ceil(data_file.max())

def get_element_property(file, atoms_property):
    potential_file = 'gnep.txt' if os.path.exists('gnep.txt') else 'nep.txt'
    with open(potential_file, 'r') as txtfile:
        first_line = txtfile.readline().strip()
        elements = first_line.split()[2:]

    from ase.io import read
    atoms = read(f'{file}.xyz', index=':')
    atom_symbols = []
    for atom in atoms:
        atom_symbol = atom.get_chemical_symbols()
        atom_symbols.extend(atom_symbol)
    element_lists = {element: [] for element in elements}
    for symbol, atom_property in zip(atom_symbols, atoms_property):
        element_lists[symbol].append(atom_property)
    non_empty_elements_lists = {element: atom_property for element, atom_property in element_lists.items() if atom_property}
    return non_empty_elements_lists

def get_indices(data, marker='-1e+06'):
    def get_no_indices(path):
        idx = []
        with open(path) as f:
            for i, line in enumerate(f):
                *_, last = line.split()
                if last == marker:
                    idx.append(i)
            total = i + 1 if 'i' in locals() else 0
        return idx, total

    train_no_indices, test_no_indices, train_indices, test_indices = None, None, None, None
    train_length, test_length = 0, 0
    if lambda_v != 0:
        if os.path.exists(f'{data}_train.out'):
            train_no_indices, train_length = get_no_indices(f'{data}_train.out')
            train_indices = [i for i in range(train_length) if i not in train_no_indices]
            if len(train_no_indices) > 0:
                np.savetxt(f'train_no_{data}_indices.txt', train_no_indices, fmt='%d')
                print(f"Train set has {len(train_no_indices)} structures without {data}, saved to train_no_{data}_indices.txt")
                print("This index is only applicable to fullbatch training and prediction")
        if os.path.exists(f'{data}_test.out'):
            test_no_indices, test_length = get_no_indices(f'{data}_test.out')
            test_indices = [i for i in range(test_length) if i not in test_no_indices]
            if len(test_no_indices) > 0:
                np.savetxt(f'test_no_{data}_indices.txt', test_no_indices, fmt='%d')
                print(f"Test set has {len(test_no_indices)} structures without {data}, saved to test_no_{data}_indices.txt")
                print("This index is only applicable to fullbatch training and prediction")
    return train_no_indices, train_indices, train_length, test_no_indices, test_indices, test_length
train_novirial_indices, train_virial_indices, train_virial_length, test_novirial_indices, test_virial_indices, test_virial_length = get_indices('virial')

def plot_element_force():
    print(f'Plotting enery element forces...')
    if model_type == 'dipole' or model_type == 'polarizability':
        print('Element force plotting is not available for dipole or polarizability models.')
        return

    if not os.path.exists('force_test.out'):
        force_test = np.loadtxt('force_test.out')
        force_elements_test = get_element_property('test', force_test)
    force_train = np.loadtxt('force_train.out')
    force_elements_train = get_element_property('train', force_train)

    for element, force_element_train in force_elements_train.items():
        figure(figsize=(5.5, 5))
        if os.path.exists('force_test.out'):
            train_element_force = get_counts2two(np.array(force_element_train))
            test_element_force = get_counts2two(np.array(force_elements_test[element]))
            plot(train_element_force[:, 1], train_element_force[:, 0], '.', label=f'{element}-train', alpha=0.8, color='deepskyblue')
            plot(test_element_force[:, 1], test_element_force[:, 0], '.', label=f'{element}-test', alpha=0.8, color='orange')
        else:
            train_element_force = np.array(force_element_train)
            plot(train_element_force[:, 3], train_element_force[:, 0], '.', label=f'{element}-x', alpha=0.8, color='red')
            plot(train_element_force[:, 4], train_element_force[:, 1], '.', label=f'{element}-y', alpha=0.8, color='green')
            plot(train_element_force[:, 5], train_element_force[:, 2], '.', label=f'{element}-z', alpha=0.8, color='blue')
        
        train_min, train_max = get_range('force', train_element_force)
        test_min, test_max = get_range('force', test_element_force) if os.path.exists('force_test.out') else (None, None)
        if use_range == 0:
            range_min = train_min if test_min is None or train_min < test_min else test_min
            range_max = train_max if test_max is None or train_max > test_max else test_max
        elif use_range == 1:
            range_min, range_max = plot_range.get('force', (None, None))
        elif use_range == 2:
            range_min, range_max = plot_range.get('force', (None, None))
        xlim(range_min, range_max); xticks(fontsize=13)
        ylim(range_min, range_max); yticks(fontsize=13)
        plot(linspace(range_min, range_max), linspace(range_min, range_max), 'k--', zorder=0)
        set_tick_params()
        xlabel('DFT force (eV/Å)', fontsize=15)
        ylabel('NEP force (eV/Å)', fontsize=15)
        legend(frameon=False, fontsize=12, loc='upper left')
        tight_layout()
        title(f'The force of element {element}')
        savefig(f'{element}-force.png', dpi=300, bbox_inches='tight')
    pass

def plot_data_component(comp):
    print(f'Plotting {comp} components...')
    if lambda_v == 0 and (comp == 'virial' or comp == 'stress'):
        print('The virial/stress component plotting is not available when virial/stress is not used in training or all structures are not has virial/stress.')
        return
    color_train, color_test = generate_colors(comp)
    label_unit = units.get(comp, 'unknown unit')
    comps3, comps6 = ['x', 'y', 'z'], ['xx', 'yy', 'zz', 'xy', 'yz', 'xz']

    def plot_component_diagonals(data_t, hang, lie, start, line_i, pic, comps):
        subplot(hang, lie, start)
        plot(data_t[:, line_i + pic], data_t[:, line_i], '.', color=color_train[line_i % len(color_train)])
        data_lie = np.column_stack((data_t[:, line_i + pic], data_t[:, line_i]))
        range_min, range_max = get_range(comp, data_lie)
        plot(linspace(range_min, range_max), linspace(range_min, range_max), 'k--', zorder=0)
        xlim(range_min, range_max); xticks(fontsize=13)
        ylim(range_min, range_max); yticks(fontsize=13)
        xlabel(f"DFT {comp} ({label_unit})", fontsize=15)
        ylabel(f"NEP {comp} ({label_unit})", fontsize=15)
        legend([f'{comps[line_i]}'], frameon=False, fontsize=13, loc='upper left')
        set_tick_params()
        tight_layout()
        pass
    
    if comp in ('force', 'dipole'):
        picture_count=  3
        figure(figsize=(16.5,5))
    else:
        picture_count=  6
        figure(figsize=(16.5,10))
    if (comp == 'virial' or comp == 'stress') and (train_novirial_indices is not None and len(train_novirial_indices) < train_virial_length):
        globals()[f'{comp}_train'] = np.loadtxt(f'{comp}_train.out')[train_virial_indices]
        globals()[f'{comp}_test'] = np.loadtxt(f'{comp}_train.out')[test_virial_indices] if os.path.exists(f'{comp}_test.out') else None
    else:
        globals()[f'{comp}_train'] = np.loadtxt(f'{comp}_train.out')
        globals()[f'{comp}_test'] = np.loadtxt(f'{comp}_test.out') if os.path.exists(f'{comp}_test.out') else None
        
    if os.path.exists(f'{comp}_test.out'):
        data_test = np.loadtxt(f'{comp}_test.out')
    data_train = np.loadtxt(f'{comp}_train.out')

    for i in range(picture_count):
        if comp in ('force', 'dipole'):
            if os.path.exists(f'{comp}_test.out'):
                plot_component_diagonals(data_test, 1, 3, i+1, i, picture_count, comps3)
                savefig(f'{comp}-test-components.png', dpi=200)
            plot_component_diagonals(data_train, 1, 3, i+1, i, picture_count, comps3)
            savefig(f'{comp}-train-components.png', dpi=200)
        else:
            if os.path.exists(f'{comp}_test.out'):
                plot_component_diagonals(data_test, 2, 3, i+1, i, picture_count, comps6)
                savefig(f'{comp}-test-components.png', dpi=200)
            plot_component_diagonals(data_train, 2, 3, i+1, i, picture_count, comps6)
            savefig(f'{comp}-train-components.png', dpi=200)
    pass

if element_force == 1:
    plot_element_force()
elif component != '0':
    plot_data_component(component)