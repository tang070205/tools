import sys
import numpy as np
from ase.io import read
from pylab import *

with open('run.in', 'r') as file:
    for line in file:
        line = line.strip()
        if 'heat_lan' in line:
            t_start, t_end = int(line.split()[1]-line.split()[3]), int(line.split()[1]+line.split()[3])
            group_first, group_last = int(line.split()[5]), int(line.split()[6])
        elif 'compute_shc' in line:
            num_corr_points = int(line.split()[2])
            max_corr_t = int(line.split()[1]*line.split()[2]/1000)
            dic = 'x' if  int(line.split()[2]) == 0 else 'y' if  int(line.split()[2]) == 1 else 'z'
            num_omega = int(line.split()[4])

compute = np.loadtxt('compute.out')
T = compute[:, group_first:group_last+1]
Ein = compute[:, -2]
Eout = compute[:, -1]
temp_ave = np.mean(T[1+int(len(compute)/2):,:], axis=0)
Ns = 1000
t = np.arange(1,len(compute)+1) * Ns/1000000  # ns

figure(figsize=(10,5))
subplot(1,2,1)
group_idx = range(group_first, group_last+1)
plot(group_idx, temp_ave,linewidth=3,marker='o',markersize=10)
xlim([group_first, group_last])
gca().set_xticks(group_idx)
ylim([t_start, t_end])
gca().set_yticks(linspace(t_start, t_end+1,6))
xlabel('group index')
ylabel('T (K)')
title('(a)')

subplot(1,2,2)
plot(t, Ein/1000, 'C3', linewidth=3)
plot(t, Eout/1000, 'C0', linewidth=3, linestyle='--' )
compute_t = int(len(compute)*Ns/1000000)
xlim([0, compute_t])
gca().set_xticks(linspace(0,compute_t,6))
ylim([-10, 10])
gca().set_yticks(linspace(-10,10,5))
xlabel('t (ns)')
ylabel('Heat (keV)')
title('(b)')

savefig('compute.png')

deltaT = temp_ave[0] - temp_ave[-1]  # [K]
Q1 = (Ein[int(len(compute)/2)] - Ein[-1])/(len(compute)/2)/Ns*1000
Q2 = (Eout[-1] - Eout[int(len(compute)/2)])/(len(compute)/2)/Ns*1000
Q = np.mean([Q1, Q1])  # [eV/ps]
l = read('model.xyz').cell.lengths()
A = l[0]*l[2]/100  # [nm2]
G = 160*Q/deltaT/A  # [GW/m2/K]


shc = np.loadtxt('shc.out')
Lx, Ly, Lz = l[0], 3*1.42*10, l[2]
V = Lx * Ly * Lz
Vvcf = shc[:2 * num_corr_points - 1, :]
shc_t, shc_Ki, shc_Ko = Vvcf[:, 0], Vvcf[:, 1], Vvcf[:, 2]
shc_nu = shc[-num_omega:, 0]/(2*pi)
shc_jwi = shc[-num_omega:, 1]
shc_jwo = shc[-num_omega:, 2]
Gc = 1.6e4*(shc_jwi+shc_jwo)/V/deltaT

figure(figsize=(10,5))
subplot(1,2,1)
L = Lx if sys.argv[2] == 'x' else Ly if sys.argv[2] == 'y' else Lz
plot(shc_t, (shc_Ki+shc_Ko)/L, linewidth=2)
xlim([-max_corr_t, max_corr_t])
gca().set_xticks(linspace(-max_corr_t, max_corr_t, 5))
ylim([-4, 10])
gca().set_yticks(linspace(-4,10,9))
ylabel('K (eV/ps)')
xlabel('Correlation time (ps)')
title('(a)')

subplot(1,2,2)
plot(shc_nu, Gc, linewidth=2)
xlim([0, 50])
gca().set_xticks(linspace(0,50,6))
ylim([0, 0.35])
gca().set_yticks(linspace(0,0.35,8))
ylabel(r'$G$($\omega$) (GW/m$^2$/K/THz)')
xlabel(r'$\omega$/2$\pi$ (THz)')
title('(b)')

savefig('shc.png')
np.save('Gc.npy', Gc)