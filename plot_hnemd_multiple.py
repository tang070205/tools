import os
import math
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from ase.io import write,read
from gpyumd.math import running_ave
from gpyumd.load import load_shc, load_kappa

au = read('2.xyz')
am = au * (130, 75, 1)
lx, ly, lz = am.cell.lengths()
am.center()
am.pbc = [True, True, False]
am
write("model.xyz", am, write_info = False)


num_kappas = 10    #执行几次
lines_per_file = 5000    #一次多少行
kappa = np.loadtxt('kappa.out', max_rows=num_kappas*lines_per_file)
file_datas = np.split(kappa, num_kappas)
t = np.arange(1, lines_per_file + 1) * 0.004

plt.figure(figsize=(17, 5))

def plot_running_avg(data, subplot_index, color, ylabel, ylimit, yticks, title_tag):
    ax = plt.subplot(1, 3, subplot_index)
    avg_data = np.zeros_like(data[0])
    for dataset in data:
        plot_data = running_ave(dataset, t)
        plt.plot(t, plot_data, color='C7', alpha=0.5)
        avg_data += plot_data
    avg_data /= num_kappas
    plt.plot(t, avg_data, color=color, linewidth=2)
    plt.annotate(f'{avg_data[-1]:.2f}', xy=(t[-1], avg_data[-1]), xytext=(-20, 5), textcoords='offset points', ha='center', va='bottom')
    
    plt.xlim([0, 20])
    plt.ylim([-50, ylimit])
    plt.gca().set_xticks(np.arange(0, 21, 5))
    plt.gca().set_yticks(np.arange(-50, ylimit+1, yticks))
    plt.xlabel('time (ns)')
    plt.ylabel(ylabel)
    plt.title(f'({title_tag})')

ki_data = [file_datas[i][:, 0] for i in range(num_kappas)]   #0为xi，2为yi
ko_data = [file_datas[i][:, 1] for i in range(num_kappas)]   #1为xo，3为yo
plot_running_avg(ki_data, 1, 'red', r'$\kappa_{in}$ W/m/K', 50, 20, 'a')
plot_running_avg(ko_data, 2, 'blue', r'$\kappa_{out}$ W/m/K', 100, 30, 'b')

plt.subplot(1, 3, 3)
plt.plot(t, running_ave(np.mean(np.array(ki_data), axis=0),t), 'red', label='in', linewidth=2)
plt.plot(t, running_ave(np.mean(np.array(ko_data), axis=0),t), 'blue', label='out', linewidth=2)
running_avg_k = running_ave(np.mean(np.array(ki_data), axis=0) + np.mean(np.array(ko_data), axis=0), t)
plt.plot(t, running_avg_k, 'black', label='total', linewidth=2)
plt.annotate(f'{running_avg_k[-1]:.2f}', xy=(t[-1], running_avg_k[-1]), xytext=(-20, -10), textcoords='offset points', ha='center', va='bottom')
plt.xlim([0, 20])
plt.ylim([-50, 100])
plt.gca().set_xticks(np.arange(0, 21, 5))
plt.gca().set_yticks(np.arange(-50, 1-1, 30))
plt.xlabel('time (ns)')
plt.ylabel(r'$\kappa_{total}$ W/m/K')
plt.title('(c)')
plt.legend(['in', 'out', 'total'])

plt.savefig('hnemd.png', dpi=150, bbox_inches='tight')

plt.figure(figsize=(6, 5))
kz_data = [file_datas[i][:, 4] for i in range(num_kappas)]
plt.plot(t, running_ave(np.mean(np.array(kz_data), axis=0),t), 'black', linewidth=2)
plt.xlim([0, 10])
plt.ylim([0, 500])
plt.gca().set_xticks(np.arange(0, 11, 2))
plt.gca().set_yticks(np.arange(0, 501, 100))
plt.xlabel('time (ns)')
plt.ylabel(r'$\kappa_{z}$ (W/m/K)')

plt.savefig('hnemd-z.png', dpi=150, bbox_inches='tight')






def process_files(output_filename):
    dir_path = os.getcwd()
    results = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file == 'shc.out':
                with open(os.path.join(root, file), 'r') as f:
                    array = [[float(num) for num in line.split()] for line in f]
                    results.append(array)

    avg_result = [[sum([results[k][j][i] for k in range(len(results))])/len(results) for i in range(len(results[0][0]))] for j in range(1499)]

    with open(output_filename, 'w') as file:
        for row in avg_result:
            file.write(' '.join(map(str, row)) + '\n')

if __name__ == "__main__":
    process_files('shc.out')

shc = load_shc(num_corr_points=250, num_omega=1000)['run0']
shc.keys()

l = am.cell.lengths()
Lx, Ly, Lz = l[0], l[1], l[2]
V = Lx * Ly * Lz
T = 300
Fe = 4.0e-6
calc_spectral_kappa(shc, driving_force=Fe, temperature=T, volume=V)
shc['kw'] = shc['kwi'] + shc['kwo']
shc['K'] = shc['Ki'] + shc['Ko']
Gc = np.load('Gc.npy')
shc.keys()

lambda_i = shc['kw']/Gc
length = np.logspace(1,6,100)
k_L = np.zeros_like(length)
for i, el in enumerate(length):
    k_L[i] = np.trapz(shc['kw']/(1+lambda_i/el), shc['nu'])

figure(figsize=(12,10))
subplot(2,2,1)
#set_fig_properties([gca()])
plot(shc['t'], shc['K']/Ly, linewidth=3)
xlim([-2, 2])
gca().set_xticks([-2, 0, 2])
ylim([-0.5, 1])
gca().set_yticks(np.arange(-0.5,1.1,0.5))
ylabel('K (eV/ps)')
xlabel('Correlation time (ps)')
title('(a)')

subplot(2,2,2)
#set_fig_properties([gca()])
plot(shc['nu'], shc['kw'],linewidth=3)
xlim([0, 5.8])
gca().set_xticks(range(0,6,1))
ylim([0, 180])
gca().set_yticks(range(0,181,30))
ylabel(r'$\kappa$($\omega$) (W/m/K/THz)')
xlabel(r'$\nu$ (THz)')
title('(b)')

subplot(2,2,3)
#set_fig_properties([gca()])
plot(shc['nu'], lambda_i,linewidth=3)
xlim([0, 5.8])
gca().set_xticks(range(0,6,1))
ylim([0, 24000])
gca().set_yticks(range(0,24001,6000))
ylabel(r'$\lambda$($\omega$) (nm)')
xlabel(r'$\nu$ (THz)')
title('(c)')

subplot(2,2,4)
#set_fig_properties([gca()])
semilogx(length/1000, k_L,linewidth=3)
xlim([1e-2, 1e3])
ylim([-10, 90])
gca().set_yticks(np.arange(-10,91,20))
ylabel(r'$\kappa$ (W/m/K)')
xlabel(r'L ($\mu$m)')
title('(d)')

tight_layout()
savefig('shc.png', dpi=300, bbox_inches='tight')
