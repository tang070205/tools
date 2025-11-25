import os, glob, math
import numpy as np
from pylab import *

three_six_component = 0   # 0不画三六分量，1画三六分量
use_range = 0   # 0使用默认读取文件最大值个最小值作范围，1使用对角线范围，2使用坐标轴范围
element_force = 0   # 0不画元素力，1画元素力
component = '0'   # '0'不画分量, 'force'画力分量, 'dipole'画偶极矩分量, 'virial'画virial分量等等
charge_sign, charge_plot_method = 1, 'hist'  # -1是图里电荷感觉反了，1是没反, hist是histplot，kde是kdeplot
plot_range = {'energy': (-9, -8), 'force': (-20, 20), 'virial': (-10, 10), 
       'stress': (-10, 10), 'dipole': (-10, 10), 'polarizability': (-10, 10)}
train_colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan'] #力的话各取前三个
test_colors = ['magenta', 'lime', 'teal', 'navy', 'olive', 'maroon']
def generate_colors(data):
    if three_six_component == 0 or data == 'energy':
        return 'deepskyblue', 'orange'   #不画三六分量，前是训练集颜色，后是测试集颜色
    else:
        if data in ['force', 'dipole']:
            return train_colors[:3], test_colors[:3]
        else:
            return train_colors, test_colors

dipole_files, polar_files = glob.glob('dipole*'), glob.glob('polarizability*')
model_type = 'dipole' if dipole_files else 'polarizability' if polar_files else None
in_file = 'gnep.in' if os.path.exists('gnep.in') else 'nep.in'
lambda_v = 0.1
batch = 1 if os.path.exists('gnep.in') else 1000
with open(in_file) as file:
    for line in file:
        line = line.strip()
        if 'lambda_v' in line and not line.startswith('#'):
            lambda_v = float(line.split()[1])
        if 'batch' in line and not line.startswith('#'):
            batch = int(line.split()[1])
        if 'prediction' in line and not line.startswith('#'):
            batch = 1000000

def get_novirial_indices(path, marker='-1e+06'):
    idx = []
    with open(path) as f:
        for i, line in enumerate(f):
            *_, last = line.split()
            if last == marker:
                idx.append(i)
        total = i + 1 if 'i' in locals() else 0
    return idx, total

train_novirial_indices, test_novirial_indices = None, None
train_length, test_length = 0, 0
if lambda_v != 0 and element_force == 0:
    if os.path.exists('virial_train.out'):
        train_novirial_indices, train_length = get_novirial_indices('virial_train.out')
        train_indices = [i for i in range(train_length) if i not in train_novirial_indices]
        if len(train_novirial_indices) > 0:
            np.savetxt('train_no_virial_indices.txt', train_novirial_indices, fmt='%d')
            print(f"Train set has {len(train_novirial_indices)} structures without virial stress, saved to train_no_virial_indices.txt")
            print("This index is only applicable to fullbatch training and prediction")
    if os.path.exists('virial_test.out'):
        test_novirial_indices, test_length = get_novirial_indices('virial_test.out')
        test_indices = [i for i in range(test_length) if i not in test_novirial_indices]
        if len(test_novirial_indices) > 0:
            np.savetxt('test_no_virial_indices.txt', test_novirial_indices, fmt='%d')
            print(f"Test set has {len(test_novirial_indices)} structures without virial stress, saved to test_no_virial_indices.txt")
            print("This index is only applicable to fullbatch training and prediction")

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

def plot_value(values, color, data):
    columns = int(values.shape[1]//2)
    if three_six_component == 0 or data == 'energy':
        plot(values[:, 1], values[:, 0], '.', color=color)
    else:
        for i in range(columns):
            plot(values[:, i+columns], values[:, i], '.', color=color[i % len(color)])
    pass

units = {'force': 'eV/Å', 'stress': 'GPa', 'energy': 'eV/atom','virial': 'eV/atom', 'dipole': 'a.u./atom', 'polarizability': 'a.u./atom'}
munits = {'force': 'meV/Å', 'stress': 'MPa', 'energy': 'meV/atom','virial': 'meV/atom', 'dipole': 'ma.u./atom', 'polarizability': 'ma.u./atom'}
def get_unit(data, rmse_origin):
    return munits.get(data, 'unknown unit') if rmse_origin < 1 else units.get(data, 'unknown unit')

def generate_legs(data, types, prefixes):
    comps = {3: ['x', 'y', 'z'], 6: ['xx', 'yy', 'zz', 'xy', 'yz', 'xz']}
    if os.path.exists(f"{data}_test.out"):
        return {typ: [f"{prefix}_{comp}" for comp in comps[3 if typ in ['force', 'dipole'] else 6]] for typ in types for prefix in prefixes}
    else:
        return {typ: [f"{comp}" for comp in comps[3 if typ in ['force', 'dipole'] else 6]] for typ in types}
properties, prefixes = ['force', 'stress', 'virial', 'dipole', 'polarizability'], ['train', 'test']

def get_range(data, data_file):
    if data == 'energy':
        return np.floor(data_file.min() * 10) / 10, np.ceil(data_file.max() * 10) / 10
    else:
        return np.floor(data_file.min()), np.ceil(data_file.max())
        
def process_data(data, data_type):
    if three_six_component == 0:
        return globals().get(f"{data}_{data_type}") if data == 'energy' else get_counts2two(globals().get(f"{data}_{data_type}"))
    else:
        return globals().get(f"{data}_{data_type}")

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

def plot_loss():
    print('plotting loss...')
    loss = np.loadtxt('loss.out')
    label_Lgnep, label_Lnep = [r'$L_{\text{total}}$'], [r'$L_1$', r'$L_2$'] if loss[-1,2] != 0 else [r'$L_2$']
    label_ef, label_ef_train, label_ef_test = ['Energy', 'Force'], ['E-train', 'F-train'], ['E-test', 'F-test']
    if os.path.exists('gnep.in'):
        loss_L, learning_rate = loss[:, 1], loss[:, 8]
        loss_train, loss_train_v = loss[:, 2:4], loss[:, 2:5]
        loss_test, loss_test_v= loss[:, 5:7], loss[:, 5:8]
    else:
        loss_L = loss[:, 2:4] if loss[-1,2] != 0 else loss[:, 3]
        loss_train, loss_train_v = loss[:, 4:6], loss[:, 4:7]
        loss_test, loss_test_v = loss[:, 7:9], loss[:, 7:10]
    
    if loss.shape[1] < 7:
        loglog(loss_L)
        loglog(loss[:, 4])
        if os.path.exists('test.xyz'):
            loglog(loss[:, 5])
            legend(label_Lnep + [f'{model_type}-train', f'{model_type}-test'], ncol= 1, frameon=False, fontsize=13, loc='upper right')
        else:
            legend(label_Lnep + [f'{model_type}'], ncol= 3 if loss[-1,2] != 0 else 2, frameon=False, fontsize=13, loc='upper right')
    else: 
        if lambda_v == 0 or (train_novirial_indices is not None and len(train_novirial_indices) == train_length):
            loglog(loss_L)
            loglog(loss_train)
            if os.path.exists('test.xyz'):
                loglog(loss_test)
                loss_label = label_Lgnep + label_ef_train + label_ef_test if os.path.exists('gnep.in') else label_Lnep + label_ef_train + label_ef_test
                legend(loss_label, ncol=3, frameon=False, fontsize=10, loc='lower left')
            else:
                loss_label = label_Lgnep + label_ef if os.path.exists('gnep.in') else label_Lnep + label_ef
                legend(loss_label, ncol=1, frameon=False, fontsize=10, loc='lower left')
        else:
            loglog(loss_L)
            loglog(loss_train_v)
            if os.path.exists('test.xyz'):
                loglog(loss_test_v)
                loss_label_v = label_Lgnep + label_ef_train + ['V-train'] + label_ef_test + ['V-test'] if os.path.exists('gnep.in') else label_Lnep + label_ef_train + ['V-train'] + label_ef_test + ['V-test']
                legend(loss_label_v, ncol=2, frameon=False, fontsize=9, loc='lower left')
            else:
                loss_label_v = label_Lgnep + label_ef + ['Virial'] if os.path.exists('gnep.in') else label_Lnep + label_ef + ['Virial']
                legend(loss_label_v, ncol=1, frameon=False, fontsize=10, loc='lower left')

    set_tick_params()
    if os.path.exists('gnep.in'):
        xlabel('Epoch', fontsize=15); xticks(fontsize=12)
    else:
        xlabel('Generation/100', fontsize=15); xticks(fontsize=12)
    ylabel('Loss', fontsize=15); yticks(fontsize=12)
    tight_layout()
    pass

def plot_learning_rate():
    loss = np.loadtxt('loss.out')
    plot(range(1, len(loss) + 1), loss[:, 8])
    set_tick_params()
    #xlim(0, 1000)
    #ylim(0.9, 1)
    xlabel('Epoch', fontsize=15)
    ylabel('Learning Rate', fontsize=15)
    tight_layout()
    pass

def plot_diagonal(data):
    print(f'plotting {data} diagonal...')
    color_train, color_test = generate_colors(data)
    label_unit = units.get(data, 'unknown unit')
    train_legs, test_legs = generate_legs(data, properties, prefixes[0::2]), generate_legs(data, properties, prefixes[1::2])
    train_leg, test_leg = train_legs.get(data, 'unknown train_legs'), test_legs.get(data, 'unknown test_legs')

    if os.path.exists(f'{data}_train.out'):
        globals()[f'{data}_train'] = np.loadtxt(f'{data}_train.out')
    else:
        print(f'{data}_train.out does not exist')
        return
    if data == 'virial' or data == 'stress':
        if train_novirial_indices is not None and len(train_novirial_indices) < train_length:
            globals()[f'{data}_train'] = globals()[f'{data}_train'][train_indices]
    data_train = process_data(data, 'train')
    train_min, train_max = get_range(data, data_train)
    plot_value(data_train, color_train, data)
    origin_rmse_train, rmse_data_train, r2_data_train = calc_r2_rmse(data_train)

    if os.path.exists(f"{data}_test.out"):
        globals()[f'{data}_test'] = np.loadtxt(f'{data}_test.out')
        if data == 'virial' or data == 'stress':
            if test_novirial_indices is not None and len(test_novirial_indices) < test_length:
                globals()[f'{data}_test'] = globals()[f'{data}_test'][test_indices]
        data_test = process_data(data, 'test')
        test_min, test_max = get_range(data, data_test) 
        plot_value(data_test, color_test, data)
        origin_rmse_test, rmse_data_test, r2_data_test = calc_r2_rmse(data_test)
        unitest = get_unit(data, origin_rmse_test)
        if three_six_component == 0 or data == 'energy':
            legend([f'train RMSE= {rmse_data_train:.3f} {unitest}', f'test RMSE= {rmse_data_test:.3f} {unitest}'], frameon=False, fontsize=13)
            annotate(f'train R²= {r2_data_train:.5f}', xy=(0.55, 0.12), fontsize=14, xycoords='axes fraction', ha='left', va='top')
            annotate(f'test R²= {r2_data_test:.5f}', xy=(0.55, 0.07), fontsize=14, xycoords='axes fraction', ha='left', va='top')
        else:
            legend(train_leg+test_leg, frameon=False, fontsize=11, ncol=2, loc='upper left', bbox_to_anchor=(0, 0.9), columnspacing=0.2)
            annotate(f'train RMSE= {rmse_data_train:.3f} {unitest}', xy=(0.09, 0.97), fontsize=13, xycoords='axes fraction', ha='left', va='top')
            annotate(f'test RMSE= {rmse_data_test:.3f} {unitest}', xy=(0.09, 0.92), fontsize=13, xycoords='axes fraction', ha='left', va='top')
            annotate(f'train R²= {r2_data_train:.5f}', xy=(0.55, 0.12), fontsize=14, xycoords='axes fraction', ha='left', va='top')
            annotate(f'test R²= {r2_data_test:.5f}', xy=(0.55, 0.07), fontsize=14, xycoords='axes fraction', ha='left', va='top')
    else:
        test_min, test_max = None, None
        unitrain = get_unit(data, origin_rmse_train)
        if three_six_component == 0 or data == 'energy':
            legend([f'train RMSE= {rmse_data_train:.3f} {unitrain}'], frameon=False, fontsize=14)
            annotate(f'train R²= {r2_data_train:.5f}', xy=(0.55, 0.07), fontsize=14, xycoords='axes fraction', ha='left', va='top')
        else:
            legend(train_leg, frameon=False, fontsize=13, loc='upper left', bbox_to_anchor=(0, 0.95))
            annotate(f'train RMSE= {rmse_data_train:.3f} {unitrain}', xy=(0.11, 0.97), fontsize=14, xycoords='axes fraction', ha='left', va='top')
            annotate(f'train R²= {r2_data_train:.5f}', xy=(0.55, 0.07), fontsize=14, xycoords='axes fraction', ha='left', va='top')

    if use_range == 0:
        range_min = train_min if test_min is None or train_min < test_min else test_min
        range_max = train_max if test_max is None or train_max > test_max else test_max
    elif use_range == 1:
        range_min, range_max = plot_range.get(data, (None, None))
    elif use_range == 2:
        range_min, range_max = plot_range.get(data, (None, None))
    xlim(range_min, range_max); xticks(fontsize=13)
    ylim(range_min, range_max); yticks(fontsize=13)
    plot(linspace(range_min, range_max), linspace(range_min, range_max), 'k--', zorder=0)
    set_tick_params()
    xlabel(f"DFT {data} ({label_unit})", fontsize=15)
    ylabel(f"NEP {data} ({label_unit})", fontsize=15)
    tight_layout()
    pass

def plot_charge():
    if not os.path.exists('charge_train.out'):
        return
    else:
        print('Plotting charge...')
        charge_train = np.loadtxt('charge_train.out')
    if batch < len(energy_train):
        print('If it is not fullbatch, please use the predicted charge_ *. out file')
        return
    
    import seaborn as sns
    figure(figsize=(6,5))
    element_charges_train = get_element_property('train', charge_train * charge_sign) 
    if os.path.exists("charge_test.out"):
        charge_test = np.loadtxt('charge_test.out')
        element_charges_test = get_element_property('test', charge_test * charge_sign)
        for element_train, element_test in zip(element_charges_train.keys(), element_charges_test.keys()):
            if charge_plot_method == 'hist':
                sns.histplot(element_charges_train[element_train], bins=500, alpha=0.6, label=f'{element_train}-train', kde=True, line_kws={'lw': 1})
                sns.histplot(element_charges_test[element_test], bins=500, alpha=0.6, label=f'{element_test}-test', kde=True, line_kws={'lw': 1})
            else:
                sns.kdeplot(element_charges_train[element_train], bins=500, alpha=0.6, label=f'{element_train}-train', lw=1)
                sns.kdeplot(element_charges_test[element_test], bins=500, alpha=0.6, label=f'{element_test}-test', lw=1)
        legend(ncol=2, frameon=False, fontsize=12, loc='upper right')
    else:
        for element in element_charges_train.keys():
            if charge_plot_method == 'hist':
                sns.histplot(element_charges_train[element], bins=500, alpha=0.6, label=element, kde=True, line_kws={'lw': 1})
            else:
                sns.kdeplot(element_charges_train[element], bins=500, alpha=0.6, label=element, lw=1)
        legend(frameon=False, fontsize=12, loc='upper right')

    xlabel('Charge', fontsize=15); xticks(fontsize=12)
    ylabel('Frequency', fontsize=15); yticks(fontsize=12)
    #ylim(0, 1000)
    set_tick_params()
    tight_layout()
    savefig(f'nep-charge.png', dpi=200, bbox_inches='tight')
    pass

def plot_descriptor():
    print('Plotting descriptor...')
    if not os.path.exists('descriptor.out'):
        return
    else:
        descriptor = np.loadtxt('descriptor.out')
    from sklearn.decomposition import PCA
    reducer = PCA(n_components=2)
    reducer.fit(descriptor)
    proj = reducer.transform(descriptor)

    figure(figsize=(5.5,5))
    if len(descriptor) == len(energy_train):
        sc = scatter(proj[:, 0], proj[:, 1], c=energy_train[:,1], cmap='Blues', edgecolor='grey', alpha=0.8)
        cbar = colorbar(sc, cax=gca().inset_axes([0.73, 0.95, 0.23, 0.03]), orientation='horizontal')
        #cbar.set_ticks([sc.get_clim()[0], sc.get_clim()[1]])
        #cbar.set_ticklabels(['{:.1f}'.format(sc.get_clim()[0]), '{:.1f}'.format(sc.get_clim()[1])])
        cbar.set_label('E/atom (eV)')
        title('Descriptors for each structure')
    elif len(descriptor) == len(force_train):
        element_des = get_element_property('train', descriptor)
        for element in element_des.keys():
            scatter([i[0] for i in element_des[element]], [i[1] for i in element_des[element]], edgecolor='grey', alpha=0.8, label=element)
        legend(frameon=False, fontsize=10, loc='upper right')
        title('Descriptors for each atom')
    else:
        print('The number of descriptors does not match the number of train.xyz structures. Please delete descriptor.out or change the existing descriptor.out file name')
    xlabel('PC1')
    ylabel('PC2')
    set_tick_params()
    tight_layout()
    savefig(f'nep-descriptor.png', dpi=200, bbox_inches='tight')
    pass

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
    if (lambda_v == 0 or (train_novirial_indices is not None and len(train_novirial_indices) == train_length)) and (comp == 'virial' or comp == 'stress'):
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
    if (comp == 'virial' or comp == 'stress') and (train_novirial_indices is not None and len(train_novirial_indices) < train_length):
        globals()[f'{comp}_train'] = np.loadtxt(f'{comp}_train.out')[train_indices]
        globals()[f'{comp}_test'] = np.loadtxt(f'{comp}_train.out')[test_indices] if os.path.exists(f'{comp}_test.out') else None
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

def plot_diagonals(diag_types, hang, lie, start):
    for i, diag_type in enumerate(diag_types):
        subplot(hang, lie, i+start)
        plot_diagonal(diag_type)
    pass

def plot_base_picture():
    if model_type is not None:
        figure(figsize=(5.5,5))
        plot_diagonal(f'{model_type}')
        savefig(f'nep-{model_type}-diagonal.png', dpi=200)
    else:
        base_diag_types= ['energy', 'force', 'virial', 'stress']
        if lambda_v == 0 or (train_novirial_indices is not None and len(train_novirial_indices) == train_length):
            figure(figsize=(11,5))
            plot_diagonals(base_diag_types[:2], 1, 2, 1)
            savefig('nep-ef-diagonals.png', dpi=200)
        elif not os.path.exists('stress_train.out'):
            figure(figsize=(16.5,5))
            plot_diagonals(base_diag_types[:3], 1, 3, 1)
            savefig('nep-efv-diagonals.png', dpi=200)
        else:
            figure(figsize=(11,10))
            plot_diagonals(base_diag_types, 2, 2, 1)
            savefig('nep-efvs-diagonals.png', dpi=200)

if os.path.exists('loss.out'):
    print('NEP Train')
    if os.path.exists('gnep.in'):
        figure(figsize=(11,5))
        subplot(1,2,1)
        plot_loss()
        subplot(1,2,2)
        plot_learning_rate()
        savefig('gnep-loss-learning_rate.png', dpi=200)
    else:
        figure(figsize=(5.5,5))
        plot_loss()
        savefig('nep-loss.png', dpi=200)
    if element_force == 1:
        plot_element_force()
    elif component != '0':
        plot_data_component(component)
    else:
        plot_base_picture()
        plot_charge()
else:
    print('NEP Prediction')
    if element_force == 1:
        plot_element_force()
    elif component != '0':
        plot_data_component(component)
    else:
        plot_base_picture()
        plot_charge()
        plot_descriptor()

