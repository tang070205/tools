import sys
import numpy as np
from ase.io import read
from pylab import *
import importlib.metadata

def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_shc_multiple.py <number-of-runs>")
        sys.exit(1)
if __name__ == "__main__":
    main()

with open('run.in', 'r') as file:
    for line in file:
        line = line.strip()
        if 'compute_shc' in line:
            num_corr_points = int(line.split()[2])
            max_corr_t = int(line.split()[1])*int(line.split()[2])/1000
            num_omega = int(line.split()[4])
            dic = 'x' if int(line.split()[3]) !=0 else 'y' if int(line.split()[3]) !=0 else 'z'
        elif 'nvt' in line:
            T = int(line.split()[2])
        elif 'compute_hnemd' in line:
            Fex, Fey,Fez = line.split()[2], line.split()[3], line.split()[4]
print('驱动力方向：', dic)
shc_unanalyzed = np.loadtxt('shc.out')
shc = np.mean(np.split(shc_unanalyzed, int(sys.argv[1])), axis=0)

def set_tick_params():
    tick_params(axis='x', which='both', direction='in', top=True, bottom=True)
    tick_params(axis='y', which='both', direction='in', left=True, right=True)

l = read('model.xyz').cell.lengths()
Lx, Ly, Lz = l[0], l[1], l[2]
V = Lx * Ly * Lz
Vvcf = shc[:2 * num_corr_points - 1, :]
shc_t, shc_Ki, shc_Ko = Vvcf[:, 0], Vvcf[:, 1], Vvcf[:, 2]
shc_nu = shc[-num_omega:, 0]/(2*pi)
Fe = Fex if dic == 'x' else Fey if dic == 'y' else Fez
shc_kwi = shc[-num_omega:, 1] * 1602.17662 / (float(Fe) * T * V) #convert = 1602.17662
shc_kwo = shc[-num_omega:, 2] * 1602.17662 / (float(Fe) * T * V)
shc_kw = shc_kwi + shc_kwo
shc_K = shc_Ki + shc_Ko
Gc = np.load('Gc.npy')
mask = shc_nu <= 50 #可能你在run.in设置的频率要大一些，可以通过改变这个值来使图三取正部分以及图四光滑
mask_nu = shc_nu[mask]
mask_kw = shc_kw[mask]
lambda_i = (shc_kw/Gc)[mask]
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
L = Lx if dic == 'x' else Ly if dic == 'y' else Lz
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
plot(shc_nu[mask], shc_kw[mask],linewidth=3)
xlim([0, 50])
gca().set_xticks(linspace(0,50,6))
ylim([0, 200])
gca().set_yticks(linspace(0,200,5))
ylabel(r'$\kappa$($\omega$) (W/m/K/THz)')
xlabel(r'$\nu$ (THz)')
set_tick_params()
title('(b)')

subplot(2,2,3)
plot(mask_nu, lambda_i,linewidth=3)
xlim([0, 50])
gca().set_xticks(linspace(0,50,6))
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

savefig('shc-multiple.png', dpi=150, bbox_inches='tight')
