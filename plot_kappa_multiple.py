import sys, subprocess
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import importlib.metadata

def main():
    if len(sys.argv) != 3:
        print("Usage: python plot_kappa_multiple.py <number-of-runs> <one-run-time>")
        sys.exit(1)
if __name__ == "__main__":
    main()

with open('run.in', 'r') as file:
    for line in file:
        line = line.strip()
        if 'time_step' in line:
            time_step = float(line.split()[1])
        elif 'compute_hnemd' in line:
            hnemd_sample = int(line.split()[1])
            dic = 'x' if float(line.split()[2]) > 0 else 'y' if float(line.split()[3]) > 0 else 'z'
print('驱动力方向：', dic)

run_time = int(sys.argv[2])
one_lines = int(run_time / hnemd_sample)
kappa = np.loadtxt('kappa.out', max_rows = int(sys.argv[1]) * int(one_lines))
file_datas = np.split(kappa, int(sys.argv[1]))
t = np.arange(1, one_lines + 1) * 0.001 * time_step
xlimit = int(run_time * time_step)

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

if dic == 'x' or dic == 'y':
    plt.figure(figsize=(13, 4))
    def plot_running_avg(data, subplot_index, color, ylabel, y_start, y_end, title_tag):
        ax = plt.subplot(1, 3, subplot_index)
        avg_data = np.zeros_like(data[0])
        for dataset in data:
            plot_data = running_ave(dataset, t)
            plt.plot(t, plot_data, color='C7', alpha=0.5)
            avg_data += plot_data
        avg_data /= int(sys.argv[1])
        plt.plot(t, avg_data, color=color, linewidth=2)
        plt.annotate(f'{avg_data[-1]:.2f}', xy=(t[-1], avg_data[-1]), xytext=(-20, 5), textcoords='offset points', ha='center', va='bottom')
        plt.xlim([0, xlimit])
        plt.ylim([y_start, y_end])
        plt.gca().set_xticks(linspace(0, xlimit, 6))
        plt.gca().set_yticks(linspace(y_start, y_end, 5))
        plt.xlabel('time (ps)')
        plt.ylabel(ylabel)
        set_tick_params()
        plt.title(f'({title_tag})')

    if dic == 'x':
        ki_data = [file_datas[i][:, 0] for i in range(int(sys.argv[1]))]
        ko_data = [file_datas[i][:, 1] for i in range(int(sys.argv[1]))]
    elif dic == 'y':
        ki_data = [file_datas[i][:, 2] for i in range(int(sys.argv[1]))]
        ko_data = [file_datas[i][:, 3] for i in range(int(sys.argv[1]))]
    plot_running_avg(ki_data, 1, 'red', r'$\kappa_{in}$ W/m/K', -2000, 4000, 'a')
    plot_running_avg(ko_data, 2, 'blue', r'$\kappa_{out}$ W/m/K', 0, 4000, 'b')

    plt.subplot(1, 3, 3)
    plt.plot(t, running_ave(np.mean(np.array(ki_data), axis=0),t), 'red', label='in', linewidth=2)
    plt.plot(t, running_ave(np.mean(np.array(ko_data), axis=0),t), 'blue', label='out', linewidth=2)
    running_avg_k = running_ave(np.mean(np.array(ki_data), axis=0) + np.mean(np.array(ko_data), axis=0), t)
    plt.plot(t, running_avg_k, 'black', label='total', linewidth=2)
    plt.annotate(f'{running_avg_k[-1]:.2f}', xy=(t[-1], running_avg_k[-1]), xytext=(-20, 5), textcoords='offset points', ha='center', va='bottom')
    plt.xlim([0,xlimit])
    plt.ylim([-50, 100])
    plt.gca().set_xticks(linspace(0, xlimit, 6))
    plt.gca().set_yticks(linspace(-50, 200, 7))
    plt.xlabel('time (ps)')
    plt.ylabel(r'$\kappa_{total}$ W/m/K')
    set_tick_params()
    plt.title('(c)')
    plt.legend(['in', 'out', 'total'])
    plt.savefig(f'hnemd-{dic}-multiple.png', dpi=150, bbox_inches='tight')

elif dic == 'z':
    plt.figure(figsize=(5, 4))
    kz_data = [file_datas[i][:, 4] for i in range(int(sys.argv[1]))]
    for dataset in kz_data:
        plot_data = running_ave(dataset, t)
        plt.plot(t, plot_data, color='C7', alpha=0.5)
    plt.plot(t, running_ave(np.mean(np.array(kz_data), axis=0),t), 'black', linewidth=2)
    plt.xlim([0, xlimit])
    plt.ylim([0, 4000])
    plt.gca().set_xticks(linspace(0, xlimit, 6))
    plt.gca().set_yticks(linspace(0, 4000, 5))
    plt.xlabel('time (ps)')
    plt.ylabel(r'$\kappa_{z}$ (W/m/K)')
    set_tick_params()

    plt.savefig('hnemd-z-multiple.png', dpi=150, bbox_inches='tight')

