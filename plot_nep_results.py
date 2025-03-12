import os, glob
import numpy as np
from pylab import *

three_six_component = 0   # 0不画三六分量，1画三六分量
use_range = 0   # 0使用默认读取文件最大值个最小值作范围，1使用对角线范围，2使用坐标轴范围
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

files = ['loss.out', 'energy_train.out', 'energy_test.out', 
         'force_train.out', 'force_test.out', 'virial_train.out', 'virial_test.out', 
         'stress_train.out', 'stress_test.out', 'dipole_train.out', 'dipole_test.out', 
         'polarizability_train.out', 'polarizability_test.out',
         'charge_train.out', 'charge_test.out', 'descriptor.out'] 
for file in files:
    if os.path.exists(file):
        vars()[file.split('.')[0]] = np.loadtxt(file)

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

with open('nep.in', 'r') as file:
    for line in file:
        line = line.strip()
        if 'type' in line:
            elements = line.split()[2:]
def get_indices(file):
    valid_indices = []
    element_indices = {element: [] for element in elements}
    with open(f'{file}.xyz', 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) > 1 and "Lattice" not in line:
                valid_indices.append(line)
    for i, line in enumerate(valid_indices):
        parts = line.split()
        if parts[0] in element_indices:
            element_indices[parts[0]].append(i)
    return element_indices

dipole_files, polar_files = glob.glob('dipole*'), glob.glob('polarizability*')
model_type = 'dipole' if dipole_files else 'polarizability' if polar_files else None
def plot_loss():
    if loss.shape[1] < 7:
        loglog(loss[:, 2:5])
        if os.path.exists('test.xyz'):
            loglog(loss[:, 5])
            legend([ r'$L_1$', r'$L_2$', f'{model_type}-train', f'{model_type}test'], ncol=4, frameon=False, fontsize=8.5, loc='upper right')
        else:
            legend([ r'$L_1$', r'$L_2$', f'{model_type}'], ncol=3, frameon=False, fontsize=10, loc='upper right')
    else: 
        if '-1e+06' in open('virial_train.out', 'r').read():
            loglog(loss[:, 2:6])
            if os.path.exists('test.xyz'):
                loglog(loss[:, 7:9])
                legend([r'$L_1$', r'$L_2$', 'E-train', 'F-train', 'E-test', 'F-test'], ncol=3, frameon=False, fontsize=10, loc='lower left')
            else:
                legend([r'$L_1$', r'$L_2$', 'Energy', 'Force'], ncol=4, frameon=False, fontsize=8, loc='upper right')
        else:
            loglog(loss[:, 2:7])
            if os.path.exists('test.xyz'):
                loglog(loss[:, 7:10])
                legend([r'$L_1$', r'$L_2$', 'E-train', 'F-train', 'V-train', 'E-test', 'F-test', 'V-test'], ncol=2, frameon=False, fontsize=8, loc='lower left')
            else:
                legend([r'$L_1$', r'$L_2$', 'Energy', 'Force', 'Virial'], ncol=5, frameon=False, fontsize=8, loc='upper right')
    tick_params(axis='x', which='both', direction='in', bottom=True)
    tick_params(axis='y', which='both', direction='in', left=True, right=True)
    xlabel('Generation/100')
    ylabel('Loss')
    tight_layout()
    pass

def plot_diagonal(data):
    color_train, color_test = generate_colors(data)
    def plot_value(values, color):
        columns = int(values.shape[1]//2)
        if three_six_component == 0 or data == 'energy':
            plot(values[:, 1], values[:, 0], '.', color=color)
        else:
            for i in range(columns):
                plot(values[:, i+columns], values[:, i], '.', color=color[i % len(color)])
    pass

    units = {'force': 'eV/Å', 'stress': 'GPa', 'energy': 'eV/atom','virial': 'eV/atom', 'dipole': 'a.u./atom', 'polarizability': 'a.u./atom'}
    munits = {'force': 'meV/Å', 'stress': 'MPa', 'energy': 'meV/atom','virial': 'meV/atom', 'dipole': 'ma.u./atom', 'polarizability': 'ma.u./atom'}
    label_unit = units.get(data, 'unknown unit')
    def get_unit(rmse_origin):
        return munits.get(data, 'unknown unit') if rmse_origin < 1 else units.get(data, 'unknown unit')
    def generate_dirs(types, prefixes):
        comps = {3: ['x', 'y', 'z'], 6: ['xx', 'yy', 'zz', 'xy', 'yz', 'xz']}
        if os.path.exists(f"{data}_test.out"):
            return {typ: [f"{prefix}_{comp}" for comp in comps[3 if typ in ['force', 'dipole'] else 6]] for typ in types for prefix in prefixes}
        else:
            return {typ: [f"{comp}" for comp in comps[3 if typ in ['force', 'dipole'] else 6]] for typ in types}
    properties, prefixes = ['force', 'stress', 'virial', 'dipole', 'polarizability'], ['train', 'test']
    train_dirs, test_dirs = generate_dirs(properties, prefixes[0::2]), generate_dirs(properties, prefixes[1::2])
    train_dir, test_dir = train_dirs.get(data, 'unknown train_dirs'), test_dirs.get(data, 'unknown test_dirs')

    def get_range(data_file):
        if data == 'energy':
            return np.floor(data_file.min() * 10) / 10, np.ceil(data_file.max() * 10) / 10
        else:
            return np.floor(data_file.min()), np.ceil(data_file.max())
    def process_data(data_type):
        if three_six_component == 0:
            return globals().get(f"{data}_{data_type}") if data == 'energy' else get_counts2two(globals().get(f"{data}_{data_type}"))
        else:
            return globals().get(f"{data}_{data_type}")
    data_train = process_data('train')
    train_min, train_max = get_range(data_train)
    plot_value(data_train, color_train)
    origin_rmse_train, rmse_data_train, r2_data_train = calc_r2_rmse(data_train)

    if os.path.exists(f"{data}_test.out"):
        data_test = process_data('test')
        test_min, test_max = get_range(data_test)
        plot_value(data_test, color_test)
        origin_rmse_test, rmse_data_test, r2_data_test = calc_r2_rmse(data_test)
        unitest = get_unit(origin_rmse_test)
        if three_six_component == 0 or data == 'energy':
            legend([f'train RMSE= {rmse_data_train:.3f} {unitest} R²= {r2_data_train:.3f}', 
                   f'test RMSE= {rmse_data_test:.3f} {unitest} R²= {r2_data_test:.3f}'], frameon=False, fontsize=10)
        else:
            legend(train_dir+test_dir, frameon=False, fontsize=9, ncol=2, loc='upper left', bbox_to_anchor=(0, 0.9))
            annotate(f'train RMSE= {rmse_data_train:.3f} {unitest} R²= {r2_data_train:.3f}', xy=(0.09, 0.97), fontsize=10, xycoords='axes fraction', ha='left', va='top')
            annotate(f'test RMSE= {rmse_data_test:.3f} {unitest} R²= {r2_data_test:.3f}', xy=(0.09, 0.92), fontsize=10, xycoords='axes fraction', ha='left', va='top')
    else:
        test_min, test_max = None, None
        unitrain = get_unit(origin_rmse_train)
        if three_six_component == 0 or data == 'energy':
            legend([f'train RMSE= {rmse_data_train:.3f} {unitrain} R²= {r2_data_train:.3f}'], frameon=False, fontsize=10)
        else:
            legend(train_dir, frameon=False, fontsize=10, loc='upper left', bbox_to_anchor=(0, 0.95))
            annotate(f'train RMSE= {rmse_data_train:.3f} {unitrain} R²= {r2_data_train:.3f}', xy=(0.11, 0.97), fontsize=10, xycoords='axes fraction', ha='left', va='top')
    
    if use_range == 0:
        range_min = train_min if test_min is None or train_min < test_min else test_min
        range_max = train_max if test_max is None or train_max > test_max else test_max
    elif use_range == 1:
        range_min, range_max = plot_range.get(data, (None, None))
    elif use_range == 2:
        range_min, range_max = plot_range.get(data, (None, None))
        xlim(range_min, range_max)
        ylim(range_min, range_max)
    plot(linspace(range_min, range_max), linspace(range_min, range_max), '-')
    tick_params(axis='x', which='both', direction='in', top=True, bottom=True)
    tick_params(axis='y', which='both', direction='in', left=True, right=True)
    xlabel(f"DFT {data} ({label_unit})")
    ylabel(f"NEP {data} ({label_unit})")
    tight_layout()
    pass

def plot_charge():
    import seaborn as sns
    def get_charge(file, charge_data):
        element_indices = get_indices(file)
        element_charges = {element: [] for element in elements}
        for element in elements:
            for idx in element_indices[element]:
                element_charges[element].append(-float(charge_data[idx]))
        return element_charges

    element_charges_train = get_charge('train', charge_train)
    if not os.path.exists("charge_test.out"):
        for element in elements:
            sns.histplot(element_charges_train[element], bins=500, alpha=0.6, label=element, kde=True, line_kws={'lw': 1})
    else:
        element_charges_test = get_charge('test', charge_test)
        for element in elements:
            sns.histplot(element_charges_train[element], bins=500, alpha=0.6, label=f'{element}-train', kde=True, line_kws={'lw': 1})
            sns.histplot(element_charges_test[element], bins=500, alpha=0.6, label=f'{element}-test', kde=True, line_kws={'lw': 1})

    tick_params(axis='x', which='both', direction='in', top=True, bottom=True)
    tick_params(axis='y', which='both', direction='in', left=True, right=True)
    legend(frameon=False, fontsize=10, loc='upper right')
    xlabel('Charge')
    ylabel('Frequency')
    tight_layout()
    pass

def plot_descriptor():
    from sklearn.decomposition import PCA
    reducer = PCA(n_components=2)
    reducer.fit(descriptor)
    proj = reducer.transform(descriptor)
    if len(descriptor) == len(energy_train):
        sc = scatter(proj[:, 0], proj[:, 1], c=energy_train[:,1], cmap='Blues', edgecolor='grey', alpha=0.8)
        cbar = colorbar(sc, cax=gca().inset_axes([0.73, 0.95, 0.23, 0.03]), orientation='horizontal')
        cbar.set_ticks([sc.get_clim()[0], sc.get_clim()[1]])
        cbar.set_ticklabels(['{:.1f}'.format(sc.get_clim()[0]), '{:.1f}'.format(sc.get_clim()[1])])
        cbar.set_label('E/atom (eV)')
        title('Descriptors for each structure')
    elif len(descriptor) == len(force_train):
        element_descriptors = {element: [] for element in elements}
        element_indices = get_indices('train')
        for element in elements:
            for idx in element_indices[element]:
                element_descriptors[element].append(proj[idx])
        for element in elements:
            scatter([i[0] for i in element_descriptors[element]], [i[1] for i in element_descriptors[element]], edgecolor='grey', alpha=0.8, label=element)
        legend(frameon=False, fontsize=10, loc='upper right')
        title('Descriptors for each atom')
    else:
        print('The number of descriptors does not match the number of train.xyz structures. Please delete descriptor.out or change the existing descriptor.out file name')
    xlabel('PC1')
    ylabel('PC2')
    tight_layout()
    pass

def plot_diagonals(diag_types, hang, lie, start):
    for i, diag_type in enumerate(diag_types):
        subplot(hang, lie, i+start)
        plot_diagonal(diag_type)
    pass
diag_types, type_vs = ['energy', 'force'], ['virial', 'stress']
if os.path.exists('loss.out'):
    print('NEP训练')
    figure(figsize=(5.5,5))
    plot_loss()
    savefig('nep-loss.png', dpi=200)
    if model_type is not None:
        figure(figsize=(5.5,5))
        plot_diagonal(f'{model_type}')
        savefig(f'nep-{model_type}.png', dpi=200)
    else:
        if '-1e+06' in open('virial_train.out', 'r').read():
            figure(figsize=(17,5))
            plot_diagonals(diag_types, 1, 2, 1)
            savefig('nep-ef-diagonals.png', dpi=200)
        elif not os.path.exists('stress_train.out'):
            diag_types_v = diag_types + [type_vs[0]]
            figure(figsize=(16.5,5))
            plot_diagonals(diag_types_v, 1, 3, 1)
            savefig('nep-efv-diagonals.png', dpi=200)
        else:
            figure(figsize=(11,10))
            diag_types_vs = diag_types + type_vs
            plot_diagonals(diag_types_vs, 2, 2, 1)
            savefig('nep-efvs-diagonals.png', dpi=200)
    if os.path.exists('charge_train.out'):
        figure(figsize=(5.5,5))
        plot_charge()
        savefig('nep-charge.png', dpi=200)
    else:
        None
else:
    print('NEP预测')
    if model_type is not None:
        figure(figsize=(5.5,5))
        plot_diagonal(f'{model_type}')
        savefig(f'nep-{model_type}.png', dpi=200)
    else:
        if '-1e+06' in open('virial_train.out', 'r').read():
            figure(figsize=(17,5))
            plot_diagonals(diag_types, 1, 2, 1)
            savefig('nep-ef-diagonals.png', dpi=200)
        elif not os.path.exists('stress_train.out'):
            diag_types_v = diag_types + [type_vs[0]]
            figure(figsize=(16.5,5))
            plot_diagonals(diag_types_v, 1, 3, 1)
            savefig('nep-efv-diagonals.png', dpi=200)
        else:
            figure(figsize=(11,10))
            diag_types_vs = diag_types + type_vs
            plot_diagonals(diag_types_vs, 2, 2, 1)
            savefig('nep-efvs-diagonals.png', dpi=200)
    if os.path.exists('charge_train.out'):
        figure(figsize=(5.5,5))
        plot_charge()
        savefig('nep-charge.png', dpi=200)
    else:
        None
    if os.path.exists('descriptor.out'):
        figure(figsize=(5.5,5))
        plot_descriptor()
        savefig('nep-descriptor.png', dpi=200)
    else:
        None