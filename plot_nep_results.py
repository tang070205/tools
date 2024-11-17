import os
import numpy as np
from pylab import *
from sklearn.metrics import r2_score

three_six_component = 0   # 0不画三六分量，1画三六分量
coord_range = {'energy': (-9, -8), 'force': (-20, 20), 'virial': (-10, 10), 
              'stress': (-10, 10), 'dipole': (-10, 10), 'polarizability': (-10, 10)}

def generate_colors(data):
    if three_six_component == 0:
        return 'deepskyblue', 'orange'
    else:
        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'magenta', 'lime', 'teal', 'navy', 'olive', 'maroon']
        if data in ['force', 'dipole']:
            return colors[:3], colors[3:6]
        elif data == 'energy':
            return 'deepskyblue', 'orange'
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
    new_nep = out_file[:, :file_nums].flatten()
    new_dft = out_file[:, file_nums:].flatten()
    return np.column_stack((new_nep, new_dft))

def calc_r2_rmse(out_file):
    file_columns = out_file.shape[1]//2
    r2_file = r2_score(out_file[:, :file_columns], out_file[:, file_columns:])
    rmse_file = np.sqrt(np.mean((out_file[:, :file_columns]-out_file[:, file_columns:])**2))
    return rmse_file, r2_file

def plot_loss():
    if os.path.exists('dipole_train.out'):
        loglog(loss[:, 1:5])
        if os.path.exists('test.xyz'):
            loglog(loss[:, 5])
            legend(['Tot', r'$L_1$', r'$L_2$', 'Dipole-train', 'Dipole-test'], ncol=2, frameon=False, fontsize=8, loc='upper right')
        else:
            legend(['Tot', r'$L_1$', r'$L_2$', 'Dipole'], ncol=2, frameon=False, fontsize=8, loc='upper right')
    elif os.path.exists('polarizability_train.out'):
        loglog(loss[:, 1:5])
        if os.path.exists('test.xyz'):
            loglog(loss[:, 5])
            legend(['Tot', r'$L_1$', r'$L_2$', 'Polarizability-train', 'Polarizability-test'], ncol=2, frameon=False, fontsize=8, loc='upper right')
        else:
            legend(['Tot', r'$L_1$', r'$L_2$', 'Polarizability'], ncol=2, frameon=False, fontsize=8, loc='upper right')
    else: 
        loglog(loss[:, 1:7])
        if os.path.exists('test.xyz'):
            loglog(loss[:, 7:10])
            legend(['Tot', r'$L_1$', r'$L_2$', 'E-train', 'F-train', 'V-train', 'E-test', 'F-test', 'V-test'], 
                    ncol=3, frameon=False, fontsize=8, loc='lower left')
        else:
            legend(['Tot', r'$L_1$', r'$L_2$', 'Energy', 'Force', 'Virial'], ncol=2, frameon=False, fontsize=8, loc='lower left')
    xlabel('Generation/100')
    ylabel('Loss')
    tight_layout()
    pass

def plot_diagonal(data):
    color_train, color_test = generate_colors(data)
    def plot_value(values, columns, color):
        if three_six_component == 0:
            plot(values[:, 1], values[:, 0], '.', color=color)
        else:
            if data == 'energy':
                plot(values[:, 1], values[:, 0], '.', color=color)
            else:
                for i in range(columns):
                    plot(values[:, i+columns], values[:, i], '.', color=color[i % len(color)])
    pass

    units = {'force': 'eV/Å', 'stress': 'GPa', 'energy': 'eV/atom','virial': 'eV/atom',
            'dipole': 'a.u./atom', 'polarizability': 'a.u./atom'}
    def generate_dirs(types, prefixes):
        components = {3: ['x', 'y', 'z'], 6: ['xx', 'yy', 'zz', 'xy', 'yz', 'xz']}
        if os.path.exists(f"{data}_test.out"):
            return {typ: [f"{prefix}_{comp}" for comp in components[3 if typ in ['force', 'dipole'] else 6]] for typ in types for prefix in prefixes}
        else:
            return {typ: [f"{comp}" for comp in components[3 if typ in ['force', 'dipole'] else 6]] for typ in types}
    types = ['force', 'stress', 'virial', 'dipole', 'polarizability']
    prefixes = ['train', 'test']
    train_dirs, test_dirs = generate_dirs(types, prefixes[0::2]), generate_dirs(types, prefixes[1::2])
    unit = units.get(data, 'unknown unit')
    train_dir = train_dirs.get(data, 'unknown train_dirs')
    test_dir = test_dirs.get(data, 'unknown test_dirs')

    if three_six_component == 0:
        if data == 'energy':
            data_train = globals().get(f"{data}_train")
        else:
            data_train_two = get_counts2two(globals().get(f"{data}_train"))
            data_train = data_train_two
    else:
        data_train = globals().get(f"{data}_train")
    train_columns = int(data_train.shape[1]//2)
    plot_value(data_train, train_columns, color_train)
    rmse_data_train, r2_data_train = calc_r2_rmse(data_train)

    if os.path.exists(f"{data}_test.out"):
        if three_six_component == 0:
            if data == 'energy':
                data_test = globals().get(f"{data}_test")
            else:
                data_test_two = get_counts2two(globals().get(f"{data}_test"))
                data_test = data_test_two
        else:
            data_test = globals().get(f"{data}_test")
        test_columns = int(data_test.shape[1]//2)
        plot_value(data_test, test_columns, color_test)
        rmse_data_test, r2_data_test = calc_r2_rmse(data_test)
        if three_six_component == 0 or data == 'energy':
            legend([f'train RMSE= {1000*rmse_data_train:.3f} {unit} R²= {r2_data_train:.3f}', 
                   f'test RMSE= {1000*rmse_data_test:.3f} {unit} R²= {r2_data_test:.3f}'], frameon=False, fontsize=10)
        else:
            legend(train_dir+test_dir, frameon=False, fontsize=8, ncol=2, loc='upper left')
            plt.annotate(f'train RMSE= {1000*rmse_data_train:.3f} {unit} R²= {r2_data_train:.3f}', xy=(0.95, 0.05), xycoords='axes fraction', ha='right', va='bottom')
            plt.annotate(f'test RMSE= {1000*rmse_data_test:.3f} {unit} R²= {r2_data_test:.3f}', xy=(0.95, 0.10), xycoords='axes fraction', ha='right', va='bottom')
    else:
        if three_six_component == 0 or data == 'energy':
            legend([f'train RMSE= {1000*rmse_data_train:.3f} {unit} R²= {r2_data_train:.3f}'], frameon=False, fontsize=10)
        else:
            legend(train_dir, frameon=False, fontsize=8, loc='upper left')
            plt.annotate(f'train RMSE= {1000*rmse_data_train:.3f} {unit} R²= {r2_data_train:.3f}', xy=(0.95, 0.05), xycoords='axes fraction', ha='right', va='bottom')
    data_min, data_max = coord_range.get(data, (None, None))
    plot(linspace(data_min, data_max), linspace(data_min, data_max), '-')
    xlabel(f"DFT {data} ({unit})")
    ylabel(f"NEP {data} ({unit})")
    tight_layout()
    pass

if os.path.exists('loss.out'):
    print('NEP训练')
    with open('nep.in', 'r') as file:
        for line in file:
            line = line.strip()
            if 'labbma_v' in line:
                lambda_v = float(line.split()[2])
            else:
                lambda_v = 0.1
    if os.path.exists('dipole_train.out'):
        figure(figsize=(10,5))
        subplot(1,2,1)
        plot_loss()
        subplot(1,2,2)
        plot_diagonal('dipole')
        savefig('nep-dipole.png', dpi=150, bbox_inches='tight')
    elif os.path.exists('polarizability_train.out'):
        plt.figure(figsize=(10,5))
        subplot(1,2,1)
        plot_loss()
        subplot(1,2,2)
        plot_diagonal('polarizability')
        savefig('nep-polarizability.png', dpi=150, bbox_inches='tight')
    else:
        if lambda_v > 0:
            figure(figsize=(12,10))
            subplot(2,2,1)
            plot_loss()
            subplot(2,2,2)
            plot_diagonal('energy')
            subplot(2,2,3)
            plot_diagonal('force')
            subplot(2,2,4)
            plot_diagonal('virial')
            #plot_diagonal('stress')
        else:
            figure(figsize=(14,5))
            subplot(1,3,1)
            plot_loss()  
            subplot(1,3,2)
            plot_diagonal('energy')
            subplot(1,3,3)
            plot_diagonal('force')
        savefig('nep.png', dpi=150, bbox_inches='tight')
else:
    print('NEP预测')
    if os.path.exists('dipole_train.out'):
        figure(figsize=(5,5))
        plot_diagonal('dipole')
    elif os.path.exists('polarizability_train.out'):
        figure(figsize=(5,5))
        plot_diagonal('polarizability')
    else:
        if not os.path.exists('stress_train.out'):
            figure(figsize=(14,5))
            subplot(1,3,1)
            plot_diagonal('energy')
            subplot(1,3,2)
            plot_diagonal('force')
            subplot(1,3,3)
            plot_diagonal('virial')
        else:
            figure(figsize=(12,10))
            subplot(2,2,1)
            plot_diagonal('energy')
            subplot(2,2,2)
            plot_diagonal('force')
            subplot(2,2,3)
            plot_diagonal('virial')
            subplot(2,2,4)
            plot_diagonal('stress')
    savefig('nep-prediction.png', dpi=150, bbox_inches='tight')
