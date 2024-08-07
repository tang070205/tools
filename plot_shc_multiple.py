import sys
import numpy as np
from ase.io import read
from pylab import *

def main():
    if len(sys.argv) != 3:
        print("Usage: python sdf.py <number-of-runs> <direction>")
        sys.exit(1)
if __name__ == "__main__":
    main()

with open('run.in', 'r') as file:
    for line in file:
        line = line.strip()
        if 'compute_shc' in line:
            num_corr_points = int(line.split()[2])
            num_omega = int(line.split()[4])
        elif 'nvt' in line:
            T = int(line.split()[2])
        elif 'compute_hnemd' in line:
            Fe = line.split()[2] if sys.argv[2] == 'x' else line.split()[3] if sys.argv[2] == 'y' else line.split()[4]
one_lines = 2 * num_corr_points - 1 + num_omega
shc_unanalyzed = np.loadtxt('shc.out', max_rows = int(sys.argv[1]) * int(one_lines))
shc = np.mean(np.split(shc_unanalyzed, int(sys.argv[1])), axis=0)

l = read('model.xyz').cell.lengths()
Lx, Ly, Lz = l[0], l[1], l[2]
V = Lx * Ly * Lz
shc_t = shc[:2 * num_corr_points - 1, 0]
shc_Ki = shc[:2 * num_corr_points - 1, 1]
shc_Ko = shc[:2 * num_corr_points - 1, 2]
shc_nu = shc[-num_omega:, 0]/(2*pi)
shc_jwi = shc[-num_omega:, 1]
shc_jwo = shc[-num_omega:, 2]
shc_kwi = shc_jwi * 1602.17662 / (float(Fe) * T * V)
shc_kwo = shc_jwo * 1602.17662 / (float(Fe) * T * V)
shc_kw = shc_kwi + shc_kwo
shc_K = shc_Ki + shc_Ko
Gc = np.load('Gc.npy')
mask = shc_nu <= 4.2 #可能你设置的频率要大很多，可以通过改变这个值来使图三取正部分以及图四光滑
mask_nu = shc_nu[mask]
mask_kw = shc_kw[mask]
lambda_i = (shc_kw/Gc)[mask]
length = np.logspace(1,6,100)
k_L = np.zeros_like(length)
for i, el in enumerate(length):
    k_L[i] = np.trapz(mask_kw/(1+lambda_i/el), mask_nu)

figure(figsize=(12,10))
subplot(2,2,1)
L = Lx if sys.argv[2] == 'x' else Ly if sys.argv[2] == 'y' else Lz
plot(shc_t, shc_K/L, linewidth=3)
xlim([-2, 2])
gca().set_xticks([-2, -1, 0, 1, 2])
ylim([-0.1, 0.2])
gca().set_yticks([-0.1,0,0.1,0.2])
ylabel('K (eV/ps)')
xlabel('Correlation time (ps)')
title('(a)')

subplot(2,2,2)
plot(shc_nu[mask], shc_kw[mask],linewidth=3)
xlim([0, 4.2])
gca().set_xticks([0,1,2,3,4.2])
ylim([0, 2])
gca().set_yticks(range(0,3,1))
ylabel(r'$\kappa$($\omega$) (W/m/K/THz)')
xlabel(r'$\nu$ (THz)')
title('(b)')

subplot(2,2,3)
plot(mask_nu, lambda_i,linewidth=3)
xlim([0, 4.2])
gca().set_xticks([0,1,2,3,4.2])
ylim([0, 100])
gca().set_yticks(range(0,101,20))
ylabel(r'$\lambda$($\omega$) (nm)')
xlabel(r'$\nu$ (THz)')
title('(c)')

subplot(2,2,4)
semilogx(length/1000, k_L,linewidth=3)
xlim([1e-2, 1e3])
ylim([0, 5])
gca().set_yticks(range(0,6,1))
ylabel(r'$\kappa$ (W/m/K)')
xlabel(r'L ($\mu$m)')
title('(d)')

tight_layout()
savefig('shc.png')