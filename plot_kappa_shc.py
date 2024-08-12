import sys
import numpy as np
from ase.build import graphene_nanoribbon
from ase.io import read,write
from pylab import *
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz #scipy<=1.13
#from scipy.integrate import cumulative_trapezoid #scipy>=1.14

gnr = graphene_nanoribbon(100, 60, type='armchair', sheet=True, vacuum=3.35/2)
gnr.euler_rotate(theta=90)
l = gnr.cell.lengths()
gnr.cell = gnr.cell.new((l[0], l[2], l[1]))
l = l[2]
gnr.center()
gnr.pbc = [True, True, False]
gnr

Ly = 3*1.42*10
split = [0, Ly, 255.6]
group_id = []
for atom in gnr:
    n =  atom.position[-2]
    for i in range(len(split)):
        if n > split[i] and n < split[i + 1]:
            group_index = i
    group_id.append(group_index)
gnr.arrays["group"] = np.array(group_id)
write("model.xyz", gnr)

run_time = 10000000
kappa = np.loadtxt('kappa.out')
t = np.arange(1, len(kappa) + 1) * 0.001 
xlimit = int(run_time / 1000000)

def running_ave(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return cumtrapz(y, x, initial=0) / x  #scipy<=1.13
    #return cumulative_trapezoid(y, x, initial=0) / x #scipy>=1.14
kappa_kxi_ra = running_ave(kappa[:,0],t)
kappa_kxo_ra = running_ave(kappa[:,1],t)
kappa_kyi_ra = running_ave(kappa[:,2],t)
kappa_kyo_ra = running_ave(kappa[:,3],t)
kappa_kz_ra = running_ave(kappa[:,4],t)

figure(figsize=(12,10))
subplot(2,2,1)
plot(t, kappa[:,2],color='C7',alpha=0.5)
plot(t, kappa_kyi_ra, linewidth=2)
xlim([0, 10])
gca().set_xticks(range(0,11,2))
ylim([-2000, 4000])
gca().set_yticks(range(-2000,4001,1000))
xlabel('time (ns)')
ylabel(r'$\kappa_{in}$ W/m/K')
title('(a)')

subplot(2,2,2)
plot(t, kappa[:,3],color='C7',alpha=0.5)
plot(t, kappa_kyo_ra, linewidth=2, color='C3')
xlim([0, 10])
gca().set_xticks(range(0,11,2))
ylim([0, 4000])
gca().set_yticks(range(0,4001,1000))
xlabel('time (ns)')
ylabel(r'$\kappa_{out}$ (W/m/K)')
title('(b)')

subplot(2,2,3)
plot(t, kappa_kyi_ra, linewidth=2)
plot(t, kappa_kyo_ra, linewidth=2, color='C3')
plot(t, kappa_kyi_ra + kappa_kyo_ra, linewidth=2, color='k')
xlim([0, 10])
gca().set_xticks(range(0,11,2))
ylim([0, 4000])
gca().set_yticks(range(0,4001,1000))
xlabel('time (ns)')
ylabel(r'$\kappa$ (W/m/K)')
legend(['in', 'out', 'total'])
title('(c)')


subplot(2,2,4)
plot(t, kappa_kyi_ra + kappa_kyo_ra,color='k', linewidth=2)
plot(t, kappa_kxi_ra + kappa_kxo_ra, color='C0', linewidth=2)
plot(t, kappa_kz_ra, color='C3', linewidth=2)
xlim([0, 10])
gca().set_xticks(range(0,11,2))
ylim([0, 4000])
gca().set_yticks(range(-2000,4001,1000))
xlabel('time (ns)')
ylabel(r'$\kappa$ (W/m/K)')
legend(['yy', 'xy', 'zy'])
title('(d)')

plt.savefig('hnemd.png', dpi=150, bbox_inches='tight')

shc = np.loadtxt('shc.out')
Lx, Lz = l[0], l[2]
V = Lx * Ly * Lz
Vvcf = shc[:499, :]
shc_t, shc_Ki, shc_Ko = Vvcf[:, 0], Vvcf[:, 1], Vvcf[:, 2]
shc_nu = shc[-1000:, 0]/(2*pi)
Fe, T = 1e-5, 300
shc_kwi = shc[-1000:, 1] * 1602.17662 / (float(Fe) * T * V) #convert = 1602.17662
shc_kwo = shc[-1000:, 2] * 1602.17662 / (float(Fe) * T * V)
shc_kw = shc_kwi + shc_kwo
shc_K = shc_Ki + shc_Ko
Gc = np.load('Gc.npy')
lambda_i = (shc_kw/Gc)
length = np.logspace(1,6,100)
k_L = np.zeros_like(length)
for i, el in enumerate(length):
    k_L[i] = np.trapz(shc_kw/(1+lambda_i/el), shc_nu)

figure(figsize=(12,10))
subplot(2,2,1)
plot(shc_t, shc_K/Ly, linewidth=3)
xlim([-0.5, 0.5])
gca().set_xticks(linspace(-0.5, 0.5, 5))
ylim([-1, 5])
gca().set_yticks(linspace(-1,5,7))
ylabel('K (eV/ps)')
xlabel('Correlation time (ps)')
title('(a)')

subplot(2,2,2)
plot(shc_nu, shc_kw,linewidth=3)
xlim([0, 50])
gca().set_xticks(linspace(0,50,6))
ylim([0, 200])
gca().set_yticks(linspace(0,200,5))
ylabel(r'$\kappa$($\omega$) (W/m/K/THz)')
xlabel(r'$\nu$ (THz)')
title('(b)')

subplot(2,2,3)
plot(shc_nu, lambda_i,linewidth=3)
xlim([0, 50])
gca().set_xticks(linspace(0,50,6))
ylim([0, 6000])
gca().set_yticks(linspace(0,6000,7))
ylabel(r'$\lambda$($\omega$) (nm)')
xlabel(r'$\nu$ (THz)')
title('(c)')

subplot(2,2,4)
semilogx(length/1000, k_L,linewidth=3)
xlim([1e-2, 1e3])
ylim([0, 3000])
gca().set_yticks(linspace(0,3000,7))
ylabel(r'$\kappa$ (W/m/K)')
xlabel(r'L ($\mu$m)')
title('(d)')

savefig('shc.png')
