import numpy as np
from pylab import *

with open('run.in', 'r') as file:
    for line in file:
        line = line.strip()
        if 'compute_hac' in line:
            one_lines = int(line.split()[2])/10

hac = np.loadtxt('hac.out')
#labels = ['t', 'jxijx', 'jxojx', 'jyijy', 'jyojy', 'jzjz', 'kxi', 'kxo', 'kyi', 'kyo', 'kz']
num_runs = int(len(hac)/one_lines) #这个可以手动改成确定的数
hac_split = np.split(hac, num_runs)
t = hac[:int(one_lines), 0]
hac_ave_i,hac_ave_o,ki_ave,ko_ave = np.zeros_like(t),np.zeros_like(t),np.zeros_like(t),np.zeros_like(t)

for array in hac_split:
    hac_ave_i += array[:, 1] + array[:, 3] 
    hac_ave_o += array[:, 2] + array[:, 4]
    ki_ave += array[:, 6] + array[:, 8]
    ko_ave += array[:, 7] + array[:, 9]
hac_ave_i /= hac_ave_i.max()
hac_ave_o /= hac_ave_o.max()
ki_ave /= 2*num_runs
ko_ave /= 2*num_runs

figure(figsize=(12,10))
subplot(2,2,1)
loglog(t, hac_ave_i, color='C3')
loglog(t, hac_ave_o, color='C0')
xlim([1e-1, 1e3])
ylim([1e-4, 1])
xlabel('Correlation Time (ps)')
ylabel('Normalized HAC')
title('(a)')

subplot(2,2,2)
for array in hac_split:
    plot(t, (array[:, 6] + array[:, 8])/2, color='C7',alpha=0.5)
plot(t, ki_ave, color='C3', linewidth=3)
xlim([0, 1000])
gca().set_xticks(linspace(0,1000,6))
ylim([0, 1500])
gca().set_yticks(linspace(0,1500,4))
xlabel('Correlation Time (ps)')
ylabel(r'$\kappa^{in}$ (W/m/K)')
title('(b)')

subplot(2,2,3)
for array in hac_split:
    plot(t, (array[:, 7] + array[:, 9])/2, color='C7',alpha=0.5)
plot(t, ko_ave, color='C0', linewidth=3)
xlim([0, 1000])
gca().set_xticks(linspace(0,1000,6))
ylim([0, 4000])
gca().set_yticks(linspace(0,4000,5))
xlabel('Correlation Time (ps)')
ylabel(r'$\kappa^{out}$ (W/m/K)')
title('(c)')

subplot(2,2,4)
plot(t, ko_ave, color='C0', linewidth=3)
plot(t, ki_ave, color='C3', linewidth=3)
plot(t, ki_ave + ko_ave, color='k', linewidth=3)
xlim([0, 1000])
gca().set_xticks(linspace(0,1000,6))
ylim([0, 4000])
gca().set_yticks(linspace(0,4000,5))
xlabel('Correlation Time (ps)')
ylabel(r'$\kappa$ (W/m/K)')
title('(d)')

savefig('emd.png')
