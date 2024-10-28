import os
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from sklearn.metrics import r2_score

color_train= 'deepskyblue'
color_test= 'orange'
energy_min, energy_max = -8.9, -7.9
force_min, force_max = -45, 45
virial_min, virial_max = -4, 6
stress_min, stress_max = -25, 35

files = ['loss.out', 'energy_train.out', 'energy_test.out', 'force_train.out',
         'force_test.out', 'virial_train.out', 'virial_test.out', 'stress_train.out', 'stress_test.out']
for file in files:
    if os.path.exists(file):
        vars()[file.split('.')[0]] = np.loadtxt(file)

def get_counts2two(out_file):
    half_nums = int(out_file.shape[1]/2)
    new_nep = out_file[:, :half_nums].flatten()
    new_dft = out_file[:, half_nums:].flatten()
    return np.column_stack((new_nep, new_dft))

def calc_r2_rmse(out_file):
    r2_file = r2_score(out_file[:, 0], out_file[:, 1])
    rmse_file = np.sqrt(np.mean((out_file[:, 0]-out_file[:, 1])**2))
    return rmse_file, r2_file

def loss_code():
    loglog(loss[:, 1:7])
    xlabel('Generation/100')
    ylabel('Loss')
    if os.path.exists('test.xyz'):
        loglog(loss[:, 7:10])
        legend(['Tot', r'$L_1$', r'$L_2$', 'E-train', 'F-train', 'V-train', 'E-test', 'F-test', 'V-test'], 
               ncol=3, frameon=False, fontsize=7)
    else:
        legend(['Tot', r'$L_1$', r'$L_2$', 'Energy', 'Force', 'Virial'], ncol=2, frameon=False, fontsize=8)
    tight_layout()
    pass

def energy_code():
    plot(energy_train[:, 1], energy_train[:, 0], '.', color=color_train)
    rmse_e_train, r2_e_train = calc_r2_rmse(energy_train)
    if os.path.exists('energy_test.out'):
        plot(energy_test[:, 1], energy_test[:, 0], '.', color=color_test)
        rmse_e_test, r2_e_test = calc_r2_rmse(energy_test)
        legend([f'train RMSE= {1000*rmse_e_train:.3f} meV/atom R²= {r2_e_train:.3f}', 
               f'test RMSE= {1000*rmse_e_test:.3f} meV/atom R²= {r2_e_test:.3f}'], frameon=False, fontsize=9)
    else:
        legend([f'train RMSE = {1000*rmse_e_train:.3f} meV/atom R²= {r2_e_train:.3f}'], frameon=False, fontsize=9)
    plot(linspace(energy_min, energy_max), linspace(energy_min, energy_max), '-')
    xlabel('DFT energy (eV/atom)')
    ylabel('NEP energy (eV/atom)')
    tight_layout()
    pass

def force_code():
    force_train_two = get_counts2two(force_train)
    plot(force_train_two[:, 1], force_train_two[:, 0], '.', color=color_train)
    rmse_f_train, r2_f_train = calc_r2_rmse(force_train_two)
    if os.path.exists('force_test.out'):
        force_test_two = get_counts2two(force_test)
        plot(force_test_two[:, 1], force_test_two[:, 0], '.', color=color_test)
        rmse_f_test, r2_f_test = calc_r2_rmse(force_test_two)
        legend([f'train RMSE= {1000*rmse_f_train:.3f} meV/atom R²= {r2_f_train:.3f}', 
               f'test RMSE= {1000*rmse_f_test:.3f} meV/atom R²= {r2_f_test:.3f}'], frameon=False, fontsize=9)
    else:
        legend([f'train RMSE= {1000*rmse_f_train:.3f} meV/atom R²= {r2_f_train:.3f}'], frameon=False, fontsize=19)
    plot(linspace(force_min, force_max), linspace(force_min, force_max), '-')
    xlabel('DFT force (eV/A)')
    ylabel('NEP force (eV/A)')
    tight_layout()
    pass

def virial_code():
    virial_train_two = get_counts2two(virial_train)
    plot(virial_train_two[:, 1], virial_train_two[:, 0], '.', color=color_train)
    rmse_v_train, r2_v_train = calc_r2_rmse(virial_train_two)
    if os.path.exists('virial_test.out'):
        virial_test_two = get_counts2two(virial_test)
        plot(virial_test_two[:, 1], virial_test_two[:, 0], '.', color=color_test)
        rmse_v_test, r2_v_test = calc_r2_rmse(virial_test_two)
        legend([f'train RMSE= {1000*rmse_v_train:.3f} meV/atom R²= {r2_v_train:.3f}', 
               f'test RMSE= {1000*rmse_v_test:.3f} meV/atom R²= {r2_v_test:.3f}'], frameon=False, fontsize=9)
    else:
        legend([f'train RMSE= {1000*rmse_v_train:.3f} meV/atom R²= {r2_v_train:.3f}'], frameon=False, fontsize=9)
    plot(linspace(virial_min, virial_max), linspace(virial_min, virial_max), '-')
    xlabel('DFT virial (eV/atom)')
    ylabel('NEP virial (eV/atom)')
    tight_layout()
    pass

def stress_code():
    stress_train_two = get_counts2two(stress_train)
    plot(stress_train_two[:, 1], stress_train_two[:, 0], '.', color=color_train)
    rmse_s_train, r2_s_train = calc_r2_rmse(stress_train_two)
    if os.path.exists('stress_test.out'):  
        stress_test_two = get_counts2two(stress_test)
        plot(stress_test_two[:, 1], stress_test_two[:, 0], '.', color=color_test)
        rmse_s_test, r2_s_test = calc_r2_rmse(stress_test_two)
        legend([f'train RMSE= {1000*rmse_s_train:.3f} MPa R²= {r2_s_train:.3f}', 
               f'test RMSE= {1000*rmse_s_test:.3f} MPa R²= {r2_s_test:.3f}'], frameon=False, fontsize=9)
    else:
        legend([f'train RMSE= {1000*rmse_s_train:.3f} MPa R²= {r2_s_train:.3f}'], frameon=False, fontsize=9)
    plot(linspace(stress_min, stress_max), linspace(stress_min, stress_max), '-')
    xlabel('DFT stress (GPa)')
    ylabel('NEP stress (GPa)')
    tight_layout()
    pass

if os.path.exists('loss.out'):
    print('NEP训练')
    plt.figure(figsize=(10,8))
    plt.subplot(2,2,1)
    loss_code()
    plt.subplot(2,2,2)
    energy_code()
    plt.subplot(2,2,3)
    force_code()
    plt.subplot(2,2,4)
    virial_code()
    #stress_code()
    plt.savefig('nep.png', dpi=150, bbox_inches='tight')
else:
    print('NEP预测')
    if not os.path.exists('stress_train.out'):
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        energy_code()
        plt.subplot(1,3,2)
        force_code()
        plt.subplot(1,3,3)
        virialcode()
    else:
        plt.figure(figsize=(10,8))
        plt.subplot(2,2,1)
        energy_code()
        plt.subplot(2,2,2)
        force_code()
        plt.subplot(2,2,3)
        virial_code()
        plt.subplot(2,2,4)
        stress_code()
    plt.savefig('nep-prediction.png', dpi=150, bbox_inches='tight')
