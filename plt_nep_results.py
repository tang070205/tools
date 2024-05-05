import os
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

files = ['loss.out', 'energy_train.out', 'energy_test.out', 'force_train.out',
         'force_test.out', 'virial_train.out', 'virial_test.out',
         'stress_train.out', 'stress_test.out']
for file in files:
    if os.path.exists(file):
        vars()[file.split('.')[0]] = np.loadtxt(file)

rmse_energy = np.sqrt(np.mean((energy_train[:,0]-energy_train[:,1])**2))
force_diff = np.reshape(force_train[:,3:6]-force_train[:,0:3], (force_train.shape[0]*3, 1))
rmse_force = np.sqrt(np.mean(force_diff**2))
virial_train = virial_train[virial_train[:, 1] > -1000, :]
rmse_virial = np.sqrt(np.mean((virial_train[:, 0:5] - virial_train[:, 6:11])**2))

legend_train = [plt.Line2D([0], [0], color='deepskyblue', marker='.', markersize=6, lw=0, label='train')]
legend_train_test = [plt.Line2D([0], [0], color='deepskyblue', marker='.', markersize=6, lw=0, label='train'),plt.Line2D([0], [0], color='orange', marker='.', markersize=6, lw=0, label='test')]

def replace_this_with_your_code1():
    loglog(loss[:, 1:7])
    xlabel('Generation/100')
    ylabel('Loss')
    legend(['Total', 'L1-regularization', 'L2-regularization', 'Energy-train', 'Force-train', 'Virial-train'])
    tight_layout()
    pass

def replace_this_with_your_code2():
    plot(energy_train[:, 1], energy_train[:, 0], '.', color='deepskyblue')
    plot(linspace(-9,-6.5), linspace(-9,-6.5), '-')
    xlabel('DFT energy (eV/atom)')
    ylabel('NEP energy (eV/atom)')
    legend(handles=legend_train)
    plt.title(f'RMSE = {1000* rmse_energy:.3f} meV/atom')
    tight_layout()
    pass

def replace_this_with_your_code3():
    plot(force_train[:, 3:6], force_train[:, 0:3], '.', color='deepskyblue')
    #plot(force_train[:, 3:6], force_train[:, 0:3], '.')
    plot(linspace(-60,60), linspace(-60,60), '-')
    xlabel('DFT force (eV/A)')
    ylabel('NEP force (eV/A)')
    legend(handles=legend_train)
    #legend(['train x direction', 'train y direction', 'train z direction'])
    plt.title(f'RMSE = {1000* rmse_force:.3f} meV/A')
    tight_layout()
    pass

def replace_this_with_your_code4():
    plot(virial_train[:, 6:11], virial_train[:, 0:5], '.', color='deepskyblue')
    plot(linspace(-5,7), linspace(-5,7), '-')
    xlabel('DFT virial (eV/atom)')
    ylabel('NEP virial (eV/atom)')
    legend(handles=legend_train)
    plt.title(f'RMSE = {1000* rmse_virial:.3f} meV/atom')
    tight_layout()
    pass

def replace_this_with_your_code5():
    plot(stress_train[:, 6:11], stress_train[:, 0:5], '.', color='deepskyblue')
    plot(linspace(-35,55), linspace(-35,55), '-')
    xlabel('DFT virial (GPa)')
    ylabel('NEP virial (GPa)')
    legend(handles=legend_train)
    plt.title(f'RMSE = {1000* rmse_stress:.3f} mGPa')
    tight_layout()
    pass

if os.path.exists('loss.out'):
    print('NEP训练')
    
    if not os.path.exists('test.xyz'):
        if not os.path.exists('stress_train.out'):
            plt.figure(figsize=(12,10))
            plt.subplot(2,2,1)
            replace_this_with_your_code1()
            plt.subplot(2,2,2)
            replace_this_with_your_code2()
            plt.subplot(2,2,3)
            replace_this_with_your_code3()
            plt.subplot(2,2,4)
            replace_this_with_your_code4()
        else:
            rmse_stress = np.sqrt(np.mean((stress_train[:, 0:5] - stress_train[:, 6:11])**2))
            plt.figure(figsize=(20,10))
            plt.subplot(2,3,1)
            replace_this_with_your_code1()
            plt.subplot(2,3,2)
            replace_this_with_your_code2()
            plt.subplot(2,3,3)
            replace_this_with_your_code3()
            plt.subplot(2,3,4)
            replace_this_with_your_code4()
            plt.subplot(2,3,5)
            replace_this_with_your_code5()
    else:
        if not os.path.exists('stress_train.out'):
            plt.figure(figsize=(12,10))
            plt.subplot(2,2,1)
            replace_this_with_your_code1()
            loglog(loss[:, 7:10])
            legend(['Total', 'L1-regularization', 'L2-regularization', 'Energy-train', 'Force-train', 'Virial-train', 'Energy-test', 'Force-test', 'Virial-test'])
            plt.subplot(2,2,2)
            replace_this_with_your_code2()
            plot(energy_test[:, 1], energy_test[:, 0], '.', color='orange')
            legend(handles=legend_train_test)
            plt.subplot(2,2,3)
            replace_this_with_your_code3()
            plot(force_test[:, 3:6], force_test[:, 0:3], '.', color='orange')
            legend(handles=legend_train_test)
            #legend(['test x direction', 'test y direction', 'test z direction', 'train x direction', 'train y direction', 'train z direction'])
            plt.subplot(2,2,4)
            replace_this_with_your_code4()
            plot(virial_test[:, 6:11], virial_test[:, 0:5], '.', color='orange')
            legend(handles=legend_train_test)
        else:
            rmse_stress = np.sqrt(np.mean((stress_train[:, 0:5] - stress_train[:, 6:11])**2))
            plt.figure(figsize=(20,10))
            plt.subplot(2,3,1)
            replace_this_with_your_code1()
            loglog(loss[:, 7:10])
            legend(['Total', 'L1-regularization', 'L2-regularization', 'Energy-train', 'Force-train', 'Virial-train', 'Energy-test', 'Force-test', 'Virial-test'])
            plt.subplot(2,3,2)
            replace_this_with_your_code2()
            plot(energy_test[:, 1], energy_test[:, 0], '.', color='orange')
            legend(handles=legend_train_test)
            plt.subplot(2,3,3)
            replace_this_with_your_code3()
            plot(force_test[:, 3:6], force_test[:, 0:3], '.', color='orange')
            legend(handles=legend_train_test)
            #legend(['train x direction', 'train y direction', 'train z direction', 'test x direction', 'test y direction', 'test z direction'])
            plt.subplot(2,3,4)
            replace_this_with_your_code4()
            plot(virial_test[:, 6:11], virial_test[:, 0:5], '.', color='orange')
            legend(handles=legend_train_test)
            plt.subplot(2,3,5)
            replace_this_with_your_code5()
            plot(stress_test[:, 6:11], stress_test[:, 0:5], '.', color='orange')
            legend(handles=legend_train_test)
else:
    print('NEP预测')
    if not os.path.exists('stress_train.out'):
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        replace_this_with_your_code2()
        plt.subplot(1,3,2)
        replace_this_with_your_code3()
        plt.subplot(1,3,3)
        replace_this_with_your_code4()
    else:
        rmse_stress = np.sqrt(np.mean((stress_train[:, 0:5] - stress_train[:, 6:11])**2))
        plt.figure(figsize=(10,10))
        plt.subplot(2,2,1)
        replace_this_with_your_code2()
        plt.subplot(2,2,2)
        replace_this_with_your_code3()
        plt.subplot(2,2,3)
        replace_this_with_your_code4()
        plt.subplot(2,2,4)
        replace_this_with_your_code5()

plt.savefig('nep.png', dpi=150, bbox_inches='tight')

