import numpy as np
from pylab import *
from ase.io import read, write

dic = 'x' #这里修改分组方向
uc = read('POSCAR') 
cx, cy, cz = 52, 30, 1
group_cycl = [9,20,1,1,1,20] #每组的周期数
Emax = 5 #compute图中(b)能量的范围最大最小值

struc = uc* (cx, cy, cz)
struc.set_pbc([True, True, False])
ucl = uc.cell[0][0] if dic == 'x' else uc.cell[1][1] if dic == 'y' else uc.cell[2][2]
natoms = len(uc)*cy*cz if dic == 'x' else len(uc)*cx*cz if dic == 'y' else len(uc)*cy*cx
def split_group(input_list, ucl):
    return [n * ucl for n in input_list]
ncounts = [natoms * count for count in group_cycl]
split = split_group(group_cycl, ucl)
split = [0] + list(np.cumsum(split))
split[:-1] = [x - 0.001 for x in split[:-1]]
print("direction boundaries:", [round(l,2) for l in split])
print("atoms per group:", ncounts)

group_id = []
for atom in struc:
    n = atom.position[-3] if dic == 'x' else atom.position[-2] if dic == 'y' else atom.position[-1]
    for i in range(len(group_cycl)):
        if n >= split[i] and n < split[i + 1]:
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
            group_start,group_end = int(line.split()[5]), int(line.split()[6])
        elif 'compute_shc' in line:
            num_corr_points = int(line.split()[2])
            max_corr_t = int(line.split()[1])*num_corr_points/1000
            freq_points, num_omega = int(line.split()[4]), int(line.split()[5])
        elif 'compute' in line:
            if len(line.split()) == 5:
                Ns = int(line.split()[3])*int(line.split()[2])
print('方向：', dic)
print("请在run.in平衡阶段中添加dump_thermo命令")

compute = np.loadtxt('compute.out')
temp = compute[:, group_start:group_end+1]
Ein, Eout = compute[:, -2], compute[:, -1]
temp_ave = np.mean(temp[1+int(len(compute)/2):,:], axis=0)
t = np.arange(1,len(compute)+1) * Ns/1000  # ps

shc, thermo = np.loadtxt('shc.out'), np.loadtxt('thermo.out')
finalx, finaly, finalz = np.mean(thermo[-10:, -9], axis=0), np.mean(thermo[-10:, -5], axis=0), np.mean(thermo[-10:, -1], axis=0)
deltaT = temp_ave[0] - temp_ave[-1]  # [K]
Q1 = (Ein[int(len(compute)/2)] - Ein[-1])/(len(compute)/2)/Ns*1000
Q2 = (Eout[-1] - Eout[int(len(compute)/2)])/(len(compute)/2)/Ns*1000
Q = np.mean([Q1, Q2])  # [eV/ps]
A = finalx*finaly/100 if dic == 'y' else finaly*finalz/100 if dic == 'x' else finalx*finalz/100
G = 160*Q/deltaT/A  # [GW/m2/K]

group_length = finalx/cx if dic == 'x' else finaly/cy if dic == 'y' else finalz/cz
V = group_length*finaly*finalz if dic == 'x' else finalx*group_length*finalz if dic == 'y' else finalx*finaly*group_length
Vvcf = shc[:(2*num_corr_points-1), :]
shc_t, shc_Ki, shc_Ko = Vvcf[:, 0], Vvcf[:, 1], Vvcf[:, 2]
shc_nu = shc[-freq_points:, 0]/(2*pi)
shc_jwi, shc_jwo = shc[-freq_points:, 1], shc[-freq_points:, 2]
Gc = 1.6e4*(shc_jwi+shc_jwo)/V/deltaT

figure(figsize=(10,10))
subplot(2,2,1)
group_idx = range(group_start, group_end+1)
plot(group_idx, temp_ave,linewidth=3,marker='o',markersize=10)
xlim([group_start, group_end])
if group_end - group_start > 10:
    gca().set_xticks(linspace(group_start, group_end, 6))
else:
    gca().set_xticks(group_idx)
ylim([T-delta_T, T+delta_T])
gca().set_yticks(linspace(T-delta_T,T+delta_T,3))
xlabel('group index')
ylabel('T (K)')
set_tick_params()
title('(a)')

subplot(2,2,2)
plot(t, Ein/1000, 'C3', linewidth=3)
plot(t, Eout/1000, 'C0', linewidth=3)#, linestyle='--')
compute_t = int(len(compute)*Ns/1000)
xlim([0, compute_t])
gca().set_xticks(linspace(0,compute_t,6))
ylim([-Emax, Emax])
gca().set_yticks(linspace(-Emax,Emax,5))
xlabel('t (ps)')
ylabel('Heat (keV)')
set_tick_params()
title('(b)')

subplot(2,2,3)
plot(shc_t, (shc_Ki+shc_Ko)/group_length, linewidth=2)
xlim([-max_corr_t, max_corr_t])
gca().set_xticks(linspace(-max_corr_t, max_corr_t, 5))
ylim([-4, 10])
gca().set_yticks(linspace(-4,10,8))
ylabel('K (eV/ps)')
xlabel('Correlation time (ps)')
set_tick_params()
title('(c)')

subplot(2,2,4)
plot(shc_nu, Gc, linewidth=2)
xlim([0, num_omega/(2*pi)])
gca().set_xticks(linspace(0,num_omega/(2*pi),6))
ylim([0, 0.35])
gca().set_yticks(linspace(0,0.35,8))
ylabel(r'$G$($\omega$) (GW/m$^2$/K/THz)')
xlabel(r'$\omega$/2$\pi$ (THz)')
set_tick_params()
title('(d)')

savefig('compute-shc.png', dpi=150, bbox_inches='tight')
np.save('Gc.npy', Gc)
