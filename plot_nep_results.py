import os, glob
import numpy as np
from pylab import *
from sklearn.metrics import r2_score

three_six_component = 0   # 0不画三六分量，1画三六分量
use_range = 0   # 0使用默认读取文件最大值个最小值作范围，1使用对角线范围，2使用坐标轴范围
plot_range = {'energy': (-9, -8), 'force': (-20, 20), 'virial': (-10, 10), 
       'stress': (-10, 10), 'dipole': (-10, 10), 'polarizability': (-10, 10)}  #对角线范围

def generate_colors(data):
    if three_six_component == 0:
        return 'deepskyblue', 'orange'   #不画三六分量，前是训练集颜色，后是测试集颜色
    else:
        #这里的colors可以随便更改，只要训练集和测试集的颜色不一样即可
        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'magenta', 'lime', 'teal', 'navy', 'olive', 'maroon']
        if data in ['force', 'dipole']:
            return colors[:3], colors[3:6]
        elif data == 'energy':
            return 'deepskyblue', 'orange'   #能量本身就只有两列
        else:
            return colors[:6], colors[6:]

files = ['loss.out', 'energy_train.out', 'energy_test.out', 
         'force_train.out', 'force_test.out', 'virial_train.out', 'virial_test.out', 
         'stress_train.out', 'stress_test.out', 'dipole_train.out', 'dipole_test.out', 
         'polarizability_train.out', 'polarizability_test.out'] 
for file in files:
    if os.path.exists(file):
        vars()[file.split('.')[0]] = np.loadtxt(file)

def get_counts2two(out_file):
    file_nums = int(out_file.shape[1]//2)
    new_nep, new_dft = out_file[:, :file_nums].flatten(), out_file[:, file_nums:].flatten()
    return np.column_stack((new_nep, new_dft))

def calc_r2_rmse(out_file):
    file_columns = out_file.shape[1]//2
    r2_file = r2_score(out_file[:, :file_columns], out_file[:, file_columns:])
    rmse_file = np.sqrt(np.mean((out_file[:, :file_columns]-out_file[:, file_columns:])**2))
    return rmse_file, r2_file

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
    munits = {'force': 'meV/Å', 'stress': 'MPa', 'energy': 'meV/atom','virial': 'meV/atom', 'dipole': 'a.u./atom', 'polarizability': 'a.u./atom'}
    label_unit = units.get(data, 'unknown unit')
    def get_unit(rmse_data):
        return munits.get(data, 'unknown unit') if rmse_data < 1 else units.get(data, 'unknown unit')

    def generate_dirs(types, prefixes):
        comps = {3: ['x', 'y', 'z'], 6: ['xx', 'yy', 'zz', 'xy', 'yz', 'xz']}
        if os.path.exists(f"{data}_test.out"):
            return {typ: [f"{prefix}_{comp}" for comp in comps[3 if typ in ['force', 'dipole'] else 6]] for typ in types for prefix in prefixes}
        else:
            return {typ: [f"{comp}" for comp in comps[3 if typ in ['force', 'dipole'] else 6]] for typ in types}
    properties, prefixes = ['force', 'stress', 'virial', 'dipole', 'polarizability'], ['train', 'test']
    train_dirs, test_dirs = generate_dirs(properties, prefixes[0::2]), generate_dirs(properties, prefixes[1::2])
    train_dir, test_dir = train_dirs.get(data, 'unknown train_dirs'), test_dirs.get(data, 'unknown test_dirs')

    def process_data(data_type):
        if three_six_component == 0:
            return globals().get(f"{data}_{data_type}") if data == 'energy' else get_counts2two(globals().get(f"{data}_{data_type}"))
        else:
            return globals().get(f"{data}_{data_type}")
    data_train = process_data('train')
    train_min, train_max = np.floor(np.min(data_train)), np.ceil(np.max(data_train))
    plot_value(data_train, color_train)
    rmse_data_train, r2_data_train = calc_r2_rmse(data_train)

    if os.path.exists(f"{data}_test.out"):
        data_test = process_data('test')
        test_min, test_max = np.floor(np.min(data_test)), np.ceil(np.max(data_test))
        plot_value(data_test, color_test)
        rmse_data_test, r2_data_test = calc_r2_rmse(data_test)
        unitest = get_unit(rmse_data_test)
        if three_six_component == 0 or data == 'energy':
            legend([f'train RMSE= {1000*rmse_data_train:.3f} {unitest} R²= {r2_data_train:.3f}', 
                   f'test RMSE= {1000*rmse_data_test:.3f} {unitest} R²= {r2_data_test:.3f}'], frameon=False, fontsize=10)
        else:
            legend(train_dir+test_dir, frameon=False, fontsize=9, ncol=2, loc='upper left', bbox_to_anchor=(0, 0.9))
            annotate(f'train RMSE= {1000*rmse_data_train:.3f} {unitest} R²= {r2_data_train:.3f}', xy=(0.09, 0.97), fontsize=10, xycoords='axes fraction', ha='left', va='top')
            annotate(f'test RMSE= {1000*rmse_data_test:.3f} {unitest} R²= {r2_data_test:.3f}', xy=(0.09, 0.92), fontsize=10, xycoords='axes fraction', ha='left', va='top')
    else:
        unitrain = get_unit(rmse_data_train)
        if three_six_component == 0 or data == 'energy':
            legend([f'train RMSE= {1000*rmse_data_train:.3f} {unitrain} R²= {r2_data_train:.3f}'], frameon=False, fontsize=10)
        else:
            legend(train_dir, frameon=False, fontsize=10, loc='upper left', bbox_to_anchor=(0, 0.95))
            annotate(f'train RMSE= {1000*rmse_data_train:.3f} {unitrain} R²= {r2_data_train:.3f}', xy=(0.11, 0.97), fontsize=10, xycoords='axes fraction', ha='left', va='top')
    
    if use_range == 0:
        range_min = train_min if train_min < test_min else test_min
        range_max = train_max if train_max > test_max else test_max
    elif use_range == 1:
        range_min, range_max = plot_range.get(data, (None, None))
    elif use_range == 2:
        range_min, range_max = plot_range.get(data, (None, None))
        xlim(range_min, range_max)
        ylim(range_min, range_max)
    plot(linspace(range_min, range_max), linspace(range_min, range_max), '-')
    xlabel(f"DFT {data} ({label_unit})")
    ylabel(f"NEP {data} ({label_unit})")
    tight_layout()
    pass

def plot_diagonals(diag_types, hang, lie, start):
    for i, diag_type in enumerate(diag_types):
        subplot(hang, lie, i+start)
        plot_diagonal(diag_type)
    pass
diag_types = ['energy', 'force']
if os.path.exists('loss.out'):
    print('NEP训练')
    if model_type is not None:
        figure(figsize=(11,5))
        subplot(1,2,1)
        plot_loss()
        subplot(1,2,2)
        plot_diagonal(f'{model_type}')
        savefig(f'nep-{model_type}.png', dpi=200, bbox_inches='tight')
    else:
        if '-1e+06' in open('virial_train.out', 'r').read():
            figure(figsize=(17,5))
            subplot(1,3,1)
            plot_loss()
            plot_diagonals(diag_types, 1, 3, 2)
        else:
            diag_types.append('virial')
            figure(figsize=(12,10))
            subplot(2,2,1)
            plot_loss()  
            plot_diagonals(diag_types, 2, 2, 2)
        savefig('nep.png', dpi=200, bbox_inches='tight')
else:
    print('NEP预测')
    if model_type is not None:
        figure(figsize=(5.5,5))
        plot_diagonal(f'{model_type}')
    else:
        if '-1e+06' in open('virial_train.out', 'r').read():
            figure(figsize=(11,5))
            plot_diagonals(diag_types, 1, 2, 1)
        elif not os.path.exists('stress_train.out'):
            figure(figsize=(17,5))
            diag_types.append('virial')
            plot_diagonals(diag_types, 1, 3, 1)
        else:
            figure(figsize=(11,10))
            diag_types.append('virial')
            diag_types.append('stress')
            plot_diagonals(diag_types, 2, 2, 1)
    savefig('nep-prediction.png', dpi=200, bbox_inches='tight')


