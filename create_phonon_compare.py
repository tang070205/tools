from pylab import *
from ase.io import read,write
from gpyumd.atoms import GpumdAtoms
import pandas as pd
import numpy as np

struc_UC = read('POSCAR') #xyz、cif文件也可以
struc_UC = GpumdAtoms(struc_UC)
struc_UC.add_basis()
struc_UC
struc = struc_UC.repeat([1,1,1])
struc.wrap()
struc = struc.repeat([10,10,1])
write("model.xyz", struc)

struc.write_basis()
special_points = {'G': [0, 0, 0], 'M': [0.5, 0, 0], 'K': [0.3333, 0.3333, 0], 'G': [0, 0, 0]}
linear_path, sym_points, labels = struc_UC.write_kpoints(path='GMKG', npoints=400, special_points=special_points) 


def set_fig_properties(ax_list):
    tl = 10
    tw = 3
    tlm = 6

    for ax in ax_list:
        ax.tick_params(which='major', length=tl, width=tw)
        ax.tick_params(which='minor', length=tlm, width=tw)
        ax.tick_params(which='both', axis='both', direction='in', labelsize=18, right=True, top=True)

data = np.loadtxt("omega2.out")

for i in range(len(data)):
    for j in range(len(data[0])):
        data[i, j] = np.sqrt(abs(data[i, j])) / (2 * np.pi) * np.sign(data[i, j])
nu = data

""" #qe加这段，vasp不用
data = np.loadtxt("C.freq.gp")
x = data[:, 0]
y_columns = data[:, 1:]
new_data = []
for i in range(y_columns.shape[1]):
    matrix = np.column_stack((x, y_columns[:, i]))
    new_data.append(matrix)
final_matrix = np.concatenate(new_data)
np.savetxt("phonon_data.txt", final_matrix, comments='', fmt='%1.6f')
data = np.loadtxt("phonon_data.txt")
data[:, 1] = data[:, 1] / 33.35641
np.savetxt("phonon.out", data, comments='', fmt='%1.6f')
"""

data_vasp = np.loadtxt('phonon.out')
max_value = data_vasp[2:,0].max()
data_vasp[2:,0] = data_vasp[2:,0] / max_value * max(linear_path) #第一个数是phonon.out文件最大横坐标，第二个文件时omega2最大横坐标
with open('phonon.out', 'r') as file:
    lines = file.readlines()
values = lines[1].strip().split()

figure(figsize=(9, 8))
set_fig_properties([gca()])
plt.scatter(data_vasp[2:, 0], data_vasp[2:, 1], marker='.', edgecolors='C1', facecolors='none')
plot(linear_path, nu, color='C0', lw=1))
xlim([0, max(linear_path)])
for value in values[2:-1]:
    float_value = float(value)
    plt.axvline(float_value / max_value * max(linear_path), color='black', linestyle='--')
    print(float_value)  
gca().set_xticks(sym_points)
gca().set_xticklabels([r'$\Gamma$', 'M', 'K', '$\Gamma$'])
ylim([0, 32])  
gca().set_yticks(range(0, 32, 5))
ylabel(r'$\nu$ (THz)',fontsize=15)
legend(['DFT', 'NEP'])
savefig('phonon.png')


            
