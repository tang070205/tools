import sys
import numpy as np
from ase.io import read
from pylab import *
from ase.io import read,write
from ase.build import graphene_nanoribbon

gnr = graphene_nanoribbon(100, 101, type='armchair', sheet=True, vacuum=3.35/2)
gnr.euler_rotate(theta=90)
lx, lz, ly = gnr.cell.lengths()
gnr.cell = gnr.cell.new((lx, ly, lz))
gnr.center()
gnr.pbc = [True, True, False]
gnr

group_cyclical = [1,20,10,10,10,10,10,10,20]
def split_group(input_list):
    return [n * 1.42 for n in input_list]
ncounts = [400 * count for count in group_cyclical]
split = split_group(group_cyclical)
split = [0] + list(np.cumsum(split)) + [ly]
print("y-direction boundaries:", [round(l,2) for l in split])
print("atoms per group:", ncounts)
group_id = []
for atom in gnr:
    n = atom.position[-2]
    for i in range(len(split)):
        if n > split[i] and n < split[i + 1]:
            group_index = i
    group_id.append(group_index)
gnr.arrays["group"] = np.array(group_id)
write("model.xyz", gnr)

compute = np.loadtxt('compute.out')
T = compute[:, 1, 9]
Ein = compute[:, -2]
Eout = compute[:, -1]
temp_ave = np.mean(T[1+int(len(compute)/2):,:], axis=0)
Ns = 1000
t = np.arange(1,len(compute)+1) * Ns/1000000  # ns

figure(figsize=(10,5))
subplot(1,2,1)
group_idx = range(1, 9)
plot(group_idx, temp_ave,linewidth=3,marker='o',markersize=10)
xlim([1,8])
gca().set_xticks(group_idx)
ylim([290, 310])
gca().set_yticks(linspace(290,310,6))
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
Vvcf = shc[:499, :]
shc_t, shc_Ki, shc_Ko = Vvcf[:, 0], Vvcf[:, 1], Vvcf[:, 2]
shc_nu = shc[-1000:, 0]/(2*pi)
shc_jwi, shc_jwo = shc[-1000:, 1], shc[-1000:, 2]
Gc = 1.6e4*(shc_jwi+shc_jwo)/V/deltaT

figure(figsize=(10,5))
subplot(1,2,1)
L = Lx if sys.argv[2] == 'x' else Ly if sys.argv[2] == 'y' else Lz
plot(shc_t, (shc_Ki+shc_Ko)/L, linewidth=2)
xlim([-0.5, 0.5])
gca().set_xticks(linspace(-0.5, 0.5, 5))
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