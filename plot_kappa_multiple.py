import sys
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

def main():
    if len(sys.argv) != 3:
        print("Usage: python sdf.py <number-of-runs> <direction>")
        sys.exit(1)
if __name__ == "__main__":
    main()

with open('run.in', 'r') as file:
    for line in file:
        line = line.strip()
        if 'time_step' in line:
            parts = line.split()
            time_step = int(parts[1])
        elif 'compute_hnemd' in line:
            parts = line.split()
            compute_hnemd = int(parts[1])
        elif 'run' in line:
            parts = line.split()
            run_value = int(parts[1])

one_lines = run_value / compute_hnemd
kappa = np.loadtxt('kappa.out', max_rows = int(sys.argv[1]) * int(one_lines))
file_datas = np.split(np.genfromtxt('kappa.out'), int(sys.argv[1]))
t = np.arange(1, one_lines + 1) * 0.001 * time_step
xlimit = int(run_value / 1000000 * time_step)

def running_ave(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return cumtrapz(y, x, initial=0) / x

if sys.argv[2] == 'x' or sys.argv[2] == 'y':
    plt.figure(figsize=(17, 5))
    def plot_running_avg(data, subplot_index, color, ylabel, ylimit, yticks, title_tag):
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
        plt.ylim([-50, ylimit])
        plt.gca().set_xticks(np.arange(0, xlimit+1, 1))
        plt.gca().set_yticks(np.arange(-50, ylimit+1, yticks))
        plt.xlabel('time (ns)')
        plt.ylabel(ylabel)
        plt.title(f'({title_tag})')

    if sys.argv[2] == 'x':
        ki_data = [file_datas[i][:, 0] for i in range(int(sys.argv[1]))]
        ko_data = [file_datas[i][:, 1] for i in range(int(sys.argv[1]))]
    elif sys.argv[2] == 'y':
        ki_data = [file_datas[i][:, 2] for i in range(int(sys.argv[1]))]
        ko_data = [file_datas[i][:, 3] for i in range(int(sys.argv[1]))]
    plot_running_avg(ki_data, 1, 'red', r'$\kappa_{in}$ W/m/K', 100, 30, 'a')
    plot_running_avg(ko_data, 2, 'blue', r'$\kappa_{out}$ W/m/K', 100, 30, 'b')

    plt.subplot(1, 3, 3)
    plt.plot(t, running_ave(np.mean(np.array(ki_data), axis=0),t), 'red', label='in', linewidth=2)
    plt.plot(t, running_ave(np.mean(np.array(ko_data), axis=0),t), 'blue', label='out', linewidth=2)
    running_avg_k = running_ave(np.mean(np.array(ki_data), axis=0) + np.mean(np.array(ko_data), axis=0), t)
    plt.plot(t, running_avg_k, 'black', label='total', linewidth=2)
    plt.annotate(f'{running_avg_k[-1]:.2f}', xy=(t[-1], running_avg_k[-1]), xytext=(-20, -10), textcoords='offset points', ha='center', va='bottom')
    plt.xlim([0,xlimit])
    plt.ylim([-50, 100])
    plt.gca().set_xticks(np.arange(0, xlimit+1, 1))
    plt.gca().set_yticks(np.arange(-50, 201, 50))
    plt.xlabel('time (ns)')
    plt.ylabel(r'$\kappa_{total}$ W/m/K')
    plt.title('(c)')
    plt.legend(['in', 'out', 'total'])

    plt.savefig(f'hnemd-{sys.argv[2]}.png', dpi=150, bbox_inches='tight')

elif sys.argv[2] == 'z':
    plt.figure(figsize=(6, 5))
    kz_data = [file_datas[i][:, 4] for i in range(int(sys.argv[1]))]
    for dataset in kz_data:
        plot_data = running_ave(dataset, t)
        plt.plot(t, plot_data, color='C7', alpha=0.5)
    plt.plot(t, running_ave(np.mean(np.array(kz_data), axis=0),t), 'black', linewidth=2)
    plt.xlim([0, xlimit])
    plt.ylim([0, 100])
    plt.gca().set_xticks(np.arange(0, xlimit+1, 1))
    plt.gca().set_yticks(np.arange(0, 101, 20))
    plt.xlabel('time (ns)')
    plt.ylabel(r'$\kappa_{z}$ (W/m/K)')

    plt.savefig('hnemd-z.png', dpi=150, bbox_inches='tight')

