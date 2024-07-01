import os
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from sklearn.metrics import r2_score

files = ['loss.out', 'energy_train.out', 'energy_test.out', 'force_train.out',
         'force_test.out', 'virial_train.out', 'virial_test.out', 'stress_train.out', 'stress_test.out']
for file in files:
    if os.path.exists(file):
        vars()[file.split('.')[0]] = np.loadtxt(file)

def calculate_r_squared(y_true, y_pred):
    return r2_score(y_true, y_pred)
color_train= 'deepskyblue'
color_test= 'orange'
legend_train = [plt.Line2D([0], [0], color=color_train, marker='.', markersize=6, lw=0, label='train'), frameon=False]
legend_train_test = [plt.Line2D([0], [0], color=color_train, marker='.', markersize=6, lw=0, label='train'),
                     plt.Line2D([0], [0], color='orange', marker='.', markersize=6, lw=0, label='test'), frameon=False]

def loss_train_code():
    loglog(loss[:, 1:7])
    xlabel('Generation/100')
    ylabel('Loss')
    legend(['Total', 'L1-regularization', 'L2-regularization', 'Energy-train', 'Force-train', 'Virial-train', frameon=False])
    tight_layout()
    pass

def loss_test_code():
    loglog(loss[:, 7:10])
    legend(['Total', 'L1-regularization', 'L2-regularization', 'Energy-train', 'Force-train', 'Virial-train', 'Energy-test', 'Force-test', 'Virial-test', frameon=False])
    pass

def energy_train_code():
    plot(energy_train[:, 1], energy_train[:, 0], '.', color=color_train)
    plot(linspace(-3.5,-2.5), linspace(-3.5,-2.5), '-')
    xlabel('DFT energy (eV/atom)')
    ylabel('NEP energy (eV/atom)')
    legend(handles=legend_train)
    rmse_energy = np.sqrt(np.mean((energy_train[:,0]-energy_train[:,1])**2))
    plt.title(f'RMSE = {1000* rmse_energy:.3f} meV/atom')
    R_energy = calculate_r_squared(energy_train[:, 0], energy_train[:, 1])
    plt.annotate(f'R² = {R_energy:.3f}',xy=(0.7, 0.4), xycoords='axes fraction')
    tight_layout()
    pass

def energy_test_code():
    plot(energy_test[:, 1], energy_test[:, 0], '.', color=color_test)
    legend(handles=legend_train_test)
    pass

def force_train_code():
    plot(force_train[:, 3:6], force_train[:, 0:3], '.', color=color_train)
    #plot(force_train[:, 3:6], force_train[:, 0:3], '.')
    plot(linspace(-10,10), linspace(-10,10), '-')
    xlabel('DFT force (eV/A)')
    ylabel('NEP force (eV/A)')
    legend(handles=legend_train)
    #legend(['train x direction', 'train y direction', 'train z direction'])
    force_diff = np.reshape(force_train[:,3:6]-force_train[:,0:3], (force_train.shape[0]*3, 1))
    rmse_force = np.sqrt(np.mean(force_diff**2))
    plt.title(f'RMSE = {1000* rmse_force:.3f} meV/A')
    R_force = calculate_r_squared(force_train[:, 3:6], force_train[:, 0:3])
    plt.annotate(f'R² = {R_force:.3f}',xy=(0.7, 0.4), xycoords='axes fraction')
    tight_layout()
    pass

def force_test_code():
    plot(force_test[:, 3:6], force_test[:, 0:3], '.', color=color_test)
    legend(handles=legend_train_test)
    #legend(['test x direction', 'test y direction', 'test z direction', 'train x direction', 'train y direction', 'train z direction'])
    pass

def virial_train_code():
    plot(virial_train[:, 6:12], virial_train[:, 0:6], '.', color=color_train)
    plot(linspace(-3,5), linspace(-3,5), '-')
    xlabel('DFT virial (eV/atom)')
    ylabel('NEP virial (eV/atom)')
    legend(handles=legend_train)
    rmse_virial = np.sqrt(np.mean((virial_train[:, 0:6] - virial_train[:, 6:12])**2))
    plt.title(f'RMSE = {1000* rmse_virial:.3f} meV/atom')
    R_virial = calculate_r_squared(virial_train[:, 0:6], virial_train[:, 6:12])
    plt.annotate(f'R² = {R_virial:.3f}',xy=(0.7, 0.4), xycoords='axes fraction')
    tight_layout()
    pass

def virial_test_code():
    plot(virial_test[:, 6:12], virial_test[:, 0:6], '.', color=color_test)
    legend(handles=legend_train_test)
    pass

def stress_train_code():
    plot(stress_train[:, 6:12], stress_train[:, 0:6], '.', color=color_train)
    plot(linspace(-10,20), linspace(-10,20), '-')
    xlabel('DFT stress (GPa)')
    ylabel('NEP stress (GPa)')
    legend(handles=legend_train)
    rmse_stress = np.sqrt(np.mean((stress_train[:, 0:6] - stress_train[:, 6:12])**2))
    plt.title(f'RMSE = {1000* rmse_stress:.3f} MPa')
    R_stress = calculate_r_squared(stress_train[:, 0:6], stress_train[:, 6:12])
    plt.annotate(f'R² = {R_stress:.3f}',xy=(0.7, 0.4), xycoords='axes fraction')
    tight_layout()
    pass

def stress_test_code():
    plot(stress_test[:, 6:12], stress_test[:, 0:6], '.', color=color_test)
    legend(handles=legend_train_test)
    pass

if os.path.exists('loss.out'):
    print('NEP训练')
    
    if not os.path.exists('test.xyz'):
        if not os.path.exists('stress_train.out'):
            plt.figure(figsize=(12,10))
            plt.subplot(2,2,1)
            loss_train_code()
            plt.subplot(2,2,2)
            energy_train_code()
            plt.subplot(2,2,3)
            force_train_code()
            plt.subplot(2,2,4)
            virial_train_code()
        else:
            plt.figure(figsize=(20,10))
            plt.subplot(2,3,1)
            loss_train_code()
            plt.subplot(2,3,2)
            energy_train_code()
            plt.subplot(2,3,3)
            force_train_code()
            plt.subplot(2,3,4)
            virial_train_code()
            plt.subplot(2,3,5)
            stress_train_code()
    else:
        if not os.path.exists('stress_train.out'):
            plt.figure(figsize=(12,10))
            plt.subplot(2,2,1)
            loss_train_code()
            loss_test_code()
            plt.subplot(2,2,2)
            energy_train_code()
            energy_test_code()
            plt.subplot(2,2,3)
            force_train_code()
            force_test_code()
            plt.subplot(2,2,4)
            virial_train_code()
            virial_test_code()
        else:
            plt.figure(figsize=(20,10))
            plt.subplot(2,3,1)
            loss_train_code()
            loss_test_code()
            plt.subplot(2,3,2)
            energy_train_code()
            energy_test_code()
            plt.subplot(2,3,3)
            force_train_code()
            force_test_code()
            plt.subplot(2,3,4)
            virial_train_code()
            virial_test_code()
            plt.subplot(2,3,5)
            stress_train_code()
            stress_test_code()
else:
    print('NEP预测')
    if not os.path.exists('stress_train.out'):
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        energy_train_code()
        plt.subplot(1,3,2)
        force_train_code()
        plt.subplot(1,3,3)
        virial_train_code()
    else:
        plt.figure(figsize=(10,8))
        plt.subplot(2,2,1)
        energy_train_code()
        plt.subplot(2,2,2)
        force_train_code()
        plt.subplot(2,2,3)
        virial_train_code()
        plt.subplot(2,2,4)
        stress_train_code()

plt.savefig('nep.png', dpi=150, bbox_inches='tight')
