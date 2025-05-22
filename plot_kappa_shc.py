import numpy as np
from pylab import *
from ase.io import read,write
import importlib.metadata

uc = read('POSCAR')
struc = uc*(10,10,10)
l = struc.cell.lengths()
struc.pbc = [True, True, True]
struc
write("model.xyz", struc)

with open('run.in', 'r') as file:
    for line in file:
        line = line.strip()
        if 'time_step' in line:
            time_step = float(line.split()[1])
        elif 'nvt' in line:
            T = int(line.split()[2])
        elif 'compute_shc' in line:
            num_corr_points = int(line.split()[2])
            max_corr_t = int(line.split()[1])*int(line.split()[2])/1000
            dic = 'x' if int(line.split()[3]) ==0 else 'y' if int(line.split()[3]) ==1 else 'z'
            freq_points = int(line.split()[4])
            num_omega = int(line.split()[5])
        elif 'compute_hnemd' in line:
            hnemd_sample = int(line.split()[1])
            Fex, Fey,Fez = line.split()[2], line.split()[3], line.split()[4]
print('驱动力方向：', dic)
print("请在run.in平衡阶段中添加dump_thermo命令")

kappa = np.loadtxt('kappa.out')
run_time = hnemd_sample * time_step * len(kappa)
t = np.arange(1, len(kappa) + 1) * time_step * 0.001
xlimit = int(run_time * 0.000001)

def set_tick_params():
    tick_params(axis='x', which='both', direction='in', top=True, bottom=True)
    tick_params(axis='y', which='both', direction='in', left=True, right=True)

def running_ave(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    scipy_version = importlib.metadata.version('scipy')
    if scipy_version < '1.14':
        from scipy.integrate import cumtrapz
        return cumtrapz(y, x, initial=0) / x
    else:
        from scipy.integrate import cumulative_trapezoid
        return cumulative_trapezoid(y, x, initial=0) / x 

if dic == 'x':
    ki_ra = running_ave(kappa[:,0],t)
    ko_ra = running_ave(kappa[:,1],t)
elif dic == 'y':
    ki_ra = running_ave(kappa[:,2],t)
    ko_ra = running_ave(kappa[:,3],t)
else:
    kz_ra = running_ave(kappa[:,4],t)

if dic == 'x' or dic == 'y':
    figure(figsize=(12,4))
    subplot(1,3,1)
    plot(t, kappa[:,2],color='C7',alpha=0.5)
    plot(t, ki_ra, linewidth=2)
    xlim([0, xlimit])
    gca().set_xticks(linspace(0,xlimit,3))
    ylim([0, 1000])
    gca().set_yticks(linspace(0,1000,6))
    xlabel('time (ns)')
    ylabel(r'$\kappa_{in}$ W/m/K')
    set_tick_params()
    title('(a)')

    subplot(1,3,2)
    plot(t, kappa[:,3],color='C7',alpha=0.5)
    plot(t, ko_ra, linewidth=2, color='C3')
    xlim([0, xlimit])
    gca().set_xticks(linspace(0,xlimit,3))
    ylim([0, 1000])
    gca().set_yticks(linspace(0,1000,6))
    xlabel('time (ns)')
    ylabel(r'$\kappa_{out}$ (W/m/K)')
    set_tick_params()
    title('(b)')

    subplot(1,3,3)
    plot(t, ki_ra, linewidth=2)
    plot(t, ko_ra, linewidth=2, color='C3')
    plot(t, ki_ra + ko_ra, linewidth=2, color='k')
    xlim([0, xlimit])
    gca().set_xticks(linspace(0,xlimit,3))
    ylim([0, 1000])
    gca().set_yticks(linspace(0,1000,6))
    xlabel('time (ns)')
    ylabel(r'$\kappa$ (W/m/K)')
    legend(['in', 'out', 'total'])
    set_tick_params()
    title('(c)')
else:
    figure(figsize=(5,5))
    plot(t, kz_ra, color='C3', linewidth=2)
    xlim([0, xlimit])
    gca().set_xticks(linspace(0,xlimit,3))
    ylim([0, 1000])
    gca().set_yticks(linspace(0,1000,6))
    xlabel('time (ns)')
    ylabel(r'$\kappa$ (W/m/K)')
    set_tick_params()

savefig('hnemd.png', dpi=150, bbox_inches='tight')

shc, thermo = np.loadtxt('shc.out'), np.loadtxt('thermo.out')
finalx, finaly, finalz = np.mean(thermo[-10:, -9], axis=0), np.mean(thermo[-10:, -5], axis=0), np.mean(thermo[-10:, -1], axis=0)
V = finalx * finaly * finalz
Vvcf = shc[:(2*num_corr_points-1), :]
shc_t, shc_Ki, shc_Ko = Vvcf[:, 0], Vvcf[:, 1], Vvcf[:, 2]
shc_nu = shc[-freq_points:, 0]/(2*pi)
Fe = Fex if dic == 'x' else Fey if dic == 'y' else Fez
shc_kwi = shc[-freq_points:, 1] * 1602.17662 / (float(Fe) * T * V) #convert = 1602.17662
shc_kwo = shc[-freq_points:, 2] * 1602.17662 / (float(Fe) * T * V)
shc_kw = shc_kwi + shc_kwo
shc_K = shc_Ki + shc_Ko 

Gc = np.load('Gc.npy')
lambda_i = np.abs(shc_kw/Gc)
length = np.logspace(1,6,100)
k_L = np.zeros_like(length)
numpy_version = importlib.metadata.version('numpy')
for i, el in enumerate(length):
    if numpy_version < '2':
        k_L[i] = np.trapz(shc_kw/(1+lambda_i/el), shc_nu)
    else:
        k_L[i] = np.trapezoid(shc_kw/(1+lambda_i/el), shc_nu)

figure(figsize=(12,10))
subplot(2,2,1)
L = finalx if dic == 'x' else finaly if dic == 'y' else finalz
plot(shc_t, shc_K/L, linewidth=3)
xlim([-max_corr_t, max_corr_t])
gca().set_xticks(linspace(-max_corr_t, max_corr_t, 5))
ylim([-1, 5])
gca().set_yticks(linspace(-1,5,7))
ylabel('K (eV/ps)')
xlabel('Correlation time (ps)')
set_tick_params()
title('(a)')

subplot(2,2,2)
plot(shc_nu, shc_kw,linewidth=3)
xlim([0, num_omega/(2*pi)])
gca().set_xticks(linspace(0,num_omega/(2*pi),6))
ylim([0, 200])
gca().set_yticks(linspace(0,200,5))
ylabel(r'$\kappa$($\omega$) (W/m/K/THz)')
xlabel(r'$\nu$ (THz)')
set_tick_params()
title('(b)')

subplot(2,2,3)
plot(shc_nu, lambda_i,linewidth=3)
xlim([0, num_omega/(2*pi)])
gca().set_xticks(linspace(0,num_omega/(2*pi),6))
ylim([0, 6000])
gca().set_yticks(linspace(0,6000,7))
ylabel(r'$\lambda$($\omega$) (nm)')
xlabel(r'$\nu$ (THz)')
set_tick_params()
title('(c)')

subplot(2,2,4)
semilogx(length/1000, k_L,linewidth=3)
xlim([1e-2, 1e3])
ylim([0, 3000])
gca().set_yticks(linspace(0,3000,7))
ylabel(r'$\kappa$ (W/m/K)')
xlabel(r'L ($\mu$m)')
set_tick_params()
title('(d)')

savefig('shc.png', dpi=150, bbox_inches='tight')