import sys
from pylab import *
import numpy as np
import importlib.metadata

def main():
    if len(sys.argv) != 3:
        print("Usage: python plot_kappa_multiple.py <number-of-runs> <one-run-time> ")
        sys.exit(1)
if __name__ == "__main__":
    main()

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
            Fex, Fey, Fez = line.split()[2], line.split()[3], line.split()[4]
print("请在run.in平衡阶段中添加dump_thermo命令")
print('驱动力方向：', dic)

def running_ave(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    scipy_version = importlib.metadata.version('scipy')
    if scipy_version < '1.14':
        from scipy.integrate import cumtrapz
        return cumtrapz(y, x, initial=0) / x
    else:
        from scipy.integrate import cumulative_trapezoid
        return cumulative_trapezoid(y, x, initial=0) / x 

def set_tick_params():
    tick_params(axis='x', which='both', direction='in', top=True, bottom=True)
    tick_params(axis='y', which='both', direction='in', left=True, right=True)

run_time = int(sys.argv[2])
output_num = int(run_time / hnemd_sample)
if len(sys.argv) == 3:
    run_num = int(sys.argv[1])
    kappa_raw = np.loadtxt('kappa.out', max_rows = run_num * output_num)
else:
    run_num = 1
    kappa_raw = np.loadtxt('kappa.out')
kappa = np.zeros((output_num, run_num))
t = np.arange(1, output_num + 1)
for i in range(run_num):
    k_run = kappa_raw[output_num * i:output_num * (i + 1), :]
    if dic == 'x':
        kappa[:, i] = running_ave(k_run[:,0],t) + running_ave(k_run[:,1],t)
    elif dic == 'y':
        kappa[:, i] = running_ave(k_run[:,2],t) + running_ave(k_run[:,3],t)
    else:
        kappa[:, i] = running_ave(k_run[:,4],t)
k_ave = np.average(kappa, axis=1)
k_std = []
for i in range(len(kappa)):
    k_std.append(np.std(kappa[i]) / np.sqrt(run_num))
print("k = " + format(k_ave[-1], ".3f") + " ± " + format(k_std[-1], ".3f") + "\n")

shc_raw, thermo = np.loadtxt('shc.out'), np.loadtxt('thermo.out')
finalx, finaly, finalz = np.mean(thermo[-10:, -9], axis=0), np.mean(thermo[-10:, -5], axis=0), np.mean(thermo[-10:, -1], axis=0)
V = finalx * finaly * finalz
shc_nu = shc_raw[-freq_points:, 0]/(2*pi)
shc_sample = 2*num_corr_points + freq_points - 1
shc = np.zeros((freq_points, run_num))
Fe = Fex if dic == 'x' else Fey if dic == 'y' else Fez
for i in range(run_num):
    shc_run = shc_raw[:shc_sample * run_num * (i + 1), :]
    shc_kwi = shc_run[-freq_points:, 1] * 1602.17662 / (float(Fe) * T * V)
    shc_kwo = shc_run[-freq_points:, 2] * 1602.17662 / (float(Fe) * T * V)
    shc_kw = shc_kwi + shc_kwo
    shc[:, i] = shc_kw
shc_ave = np.average(shc, axis=1)
shc_ave[shc_ave < 0] = 0
shc_correct = np.zeros((freq_points, run_num))
for i in range(freq_points):
    for j in range(run_num):
        r = 6.62607015e-34 / 1.38065e-23  # h/k_B, unit in K*s
        x = r * shc_nu[i] * 1e12 / T  # dimensionless
        shc_correct[i] = shc[i] * x ** 2 * exp(x) / (exp(x) - 1) ** 2
kappa_quantum = [np.trapezoid(shc_correct[:, i], shc_nu) for i in range(run_num)]
kappa_quantum_ave = np.average(kappa_quantum)
kappa_quantum_std = np.std(kappa_quantum) / np.sqrt(run_num)
print("k_quantum_corrected = " + format(kappa_quantum_ave, ".3f") + " ± " + format(kappa_quantum_std, ".3f") + "\n")

figure(figsize=(11, 5))
subplot(1, 2, 1)
plot(t * 0.001, k_ave, color="red", linewidth=3)
plot(t * 0.001, k_ave + k_std, color="black", linewidth=1.5, linestyle="--")
plot(t * 0.001, k_ave - k_std, color="black", linewidth=1.5, linestyle="--")
for j in range(run_num):
    plot(t * 0.001, kappa[:, j], color="red", linewidth=0.3)
xlim([0, 10])
gca().set_xticks(np.linspace(0, 10, 6))
ylim([0, 200])
gca().set_yticks(np.linspace(0, 200, 6))
set_tick_params()
subplots_adjust(wspace=0.1, hspace=0.15)
xlabel(r'Simulation time (ns)')
ylabel(r'$\kappa$ (WmK$^{-1}$)')
title('(a)')

subplot(1, 2, 2)
plot(shc_nu, shc_ave, linewidth=2, color='red', label="Classical Value")
plot(shc_nu, np.average(shc_correct, axis=1), linewidth=2, linestyle="--", color='blue', label="Quantum-Corrected Value")
xlim([0, 35])
gca().set_xticks(linspace(0, 35, 8))
ylim([0, 30])
gca().set_yticks(linspace(0, 30, 7))
set_tick_params()
legend(fontsize=10)
subplots_adjust(wspace=0.3)
ylabel(r'$\kappa$($\omega$) (W m$^{-1}$ K$^{-1}$ THz$^{-1}$)')
xlabel(r'$\omega$/2$\pi$ (THz)')
title('(b)')

savefig('kappa_shc.png', dpi=150, bbox_inches='tight')


