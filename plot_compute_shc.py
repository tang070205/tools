import sys, subprocess
import numpy as np
from pylab import *
from ase.io import read, write

uc = read('POSCAR') 
cx, cy, cz = 52, 30, 1
struc = uc* (cx, cy, cz)
struc.set_pbc([True, True, False])
l = struc.cell.lengths()

dic = 'x' #这里修改分组方向
group_cycl = [9,20,1,1,1,20] #每组的周期数
ucl = uc.cell[0][0] if dic == 'x' else uc.cell[1][1] if dic == 'y' else uc.cell[2][2]
natoms = len(uc)*cy*cz if dic == 'x' else len(uc)*cx*cz if dic == 'y' else len(uc)*cy*cx
def split_group(input_list, ucl):
    return [n * ucl for n in input_list]
ncounts = [natoms * count for count in group_cycl]
split = split_group(group_cycl, ucl)
split = [-1] + list(np.cumsum(split))
print("direction boundaries:", [round(l,2) for l in split])
print("atoms per group:", ncounts)

group_id = []
for atom in struc:
    n = atom.position[-3] if dic == 'x' else atom.position[-2] if dic == 'y' else atom.position[-1]
    for i in range(len(group_cycl)):
        if n > split[i] and n < split[i + 1]:
            group_index = i
    group_id.append(group_index)
struc.arrays["group"] = np.array(group_id)

write("model.xyz", struc)

def set_tick_params():
    tick_params(axis='x', which='both', direction='in', top=True, bottom=True)
    tick_params(axis='y', which='both', direction='in', left=True, right=True)

with open('run.in', 'r') as file:
    for line in file:
        line = line.strip()
        if 'time_step' in line:
            time_step = float(line.split()[1])
        elif 'heat_lan' in line:
            T = int(line.split()[2])
            delta_T = int(line.split()[4])
            group_start = int(line.split()[5])
            group_end = int(line.split()[6])
        elif 'compute_shc' in line:
            num_corr_points = int(line.split()[2])
            max_corr_t = int(line.split()[2])/500
            freq_points = int(line.split()[4])
            num_omega = int(line.split()[5])
            group_length = group_cycl[int(line.split()[8])+1]*ucl

compute = np.loadtxt('compute.out')
temp = compute[:, group_start:group_end+1]
Ein = compute[:, -2]
Eout = compute[:, -1]
temp_ave = np.mean(temp[1+int(len(compute)/2):,:], axis=0)
Ns_output = subprocess.run("grep 'compute' run.in -m 1 | awk '{print $3*$4}'", shell=True, capture_output=True, text=True)
Ns = int(Ns_output.stdout.strip())
t = np.arange(1,len(compute)+1) * Ns/1000000  # ns

figure(figsize=(10,5))
subplot(1,2,1)
group_idx = range(group_start, group_end+1)
plot(group_idx, temp_ave,linewidth=3,marker='o',markersize=10)
xlim([group_start, group_end])
gca().set_xticks(group_idx)
ylim([T-delta_T, T+delta_T])
gca().set_yticks(linspace(T-delta_T,T+delta_T,3))
xlabel('group index')
ylabel('T (K)')
set_tick_params()
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
set_tick_params()
title('(b)')

savefig('compute.png', dpi=150, bbox_inches='tight')

deltaT = temp_ave[0] - temp_ave[-1]  # [K]
Q1 = (Ein[int(len(compute)/2)] - Ein[-1])/(len(compute)/2)/Ns*1000
Q2 = (Eout[-1] - Eout[int(len(compute)/2)])/(len(compute)/2)/Ns*1000
Q = np.mean([Q1, Q2])  # [eV/ps]
A = l[0]*l[2]/100 if dic == 'y' else l[1]*l[2]/100 if dic == 'x' else l[0]*l[1]/100
G = 160*Q/deltaT/A  # [GW/m2/K]

shc = np.loadtxt('shc.out')
V = l[0]*group_length*l[2] if dic == 'y' else group_length*l[1]*l[2] if dic == 'x' else l[0]*l[1]*group_length
Vvcf = shc[:(2*num_corr_points-1), :]
shc_t, shc_Ki, shc_Ko = Vvcf[:, 0], Vvcf[:, 1], Vvcf[:, 2]
shc_nu = shc[-freq_points:, 0]/(2*pi)
shc_jwi, shc_jwo = shc[-freq_points:, 1], shc[-freq_points:, 2]
Gc = 1.6e4*(shc_jwi+shc_jwo)/V/deltaT

figure(figsize=(10,5))
subplot(1,2,1)
plot(shc_t, (shc_Ki+shc_Ko)/group_length, linewidth=2)
xlim([-max_corr_t, max_corr_t])
gca().set_xticks(linspace(-max_corr_t, max_corr_t, 5))
ylim([-4, 10])
gca().set_yticks(linspace(-4,10,9))
ylabel('K (eV/ps)')
xlabel('Correlation time (ps)')
set_tick_params()
title('(a)')

subplot(1,2,2)
plot(shc_nu, Gc, linewidth=2)
xlim([0, num_omega/(2*pi)])
gca().set_xticks(linspace(0,num_omega/(2*pi),6))
ylim([0, 0.35])
gca().set_yticks(linspace(0,0.35,8))
ylabel(r'$G$($\omega$) (GW/m$^2$/K/THz)')
xlabel(r'$\omega$/2$\pi$ (THz)')
set_tick_params()
title('(b)')

savefig('shc.png', dpi=150, bbox_inches='tight')
np.save('Gc.npy', Gc)