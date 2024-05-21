from pylab import *
from ase.io import read,write
from gpyumd.atoms import GpumdAtoms
from gpyumd.load import load_omega2
import pandas as pd
import numpy as np

struc_UC = read('POSCAR') #xyz、cif文件也可以
struc_UC = GpumdAtoms(struc_UC)
struc_UC.add_basis()
struc_UC
struc = struc_UC.repeat([1,1,1])
struc.wrap()
struc = struc.repeat([10,10,1])
write("model.xyz", Si)

struc.write_basis()
special_points = {'G': [0, 0, 0], 'M': [0.5, 0.5, 0], 'K': [0.375, 0.375, 0.75], 'G': [0, 0, 0], 'L': [0.5, 0.5, 0.5]}
linear_path, sym_points, labels = struc_UC.write_kpoints(path='GMKGL', npoints=400, special_points=special_points) 


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

""" #qe需要加这段
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

data_vasp = pd.read_csv('phonon.out', delim_whitespace=True, header=None)
max_value = data_vasp[0].max()
data_vasp[0] = data_vasp[0] / max_value * max(linear_path) #第一个数是phonon.out文件最大横坐标，第二个文件时omega2最大横坐标

figure(figsize=(10, 8))
set_fig_properties([gca()])
plt.plot(data_vasp.iloc[:, 0], data_vasp.iloc[:, 1], linestyle='--', color='C1')
plot(linear_path, nu[:, 0], color='C0', lw=1)
plot(linear_path, nu[:, 1:], color='C0', lw=1)
xlim([0, max(linear_path)])
gca().set_xticks(sym_points)
gca().set_xticklabels([r'$\Gamma$', 'M', 'K', '$\Gamma$', 'L'])
ylim([0, 30])  
gca().set_yticks(linspace(0, 30, 6))
ylabel(r'$\nu$ (THz)',fontsize=30)
savefig('phonon.png')
            
