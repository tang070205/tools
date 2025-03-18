import os
import numpy as np
from pylab import *
from ase.io import read,write

cx,cy,cz = 20, 20, 1   # 超胞参数
npoints = 400   
special_points = {'G': [0, 0, 0], 'M': [0.5, 0, 0], 'K': [0.3333, 0.3333, 0], 'G': [0, 0, 0]}  # 高对称点坐标，同样vaspkit305提供的文件里有
points_path = ['GMKG']     #高对称点路径，可以写断点比如['GM', 'KG']，最后还要设置横坐标的标签gca().set_xticklabels([])

uc = read('POSCAR-unitcell') #xyz、cif文件也可以
struc = uc * (cx,cy,cz)
write("model.xyz", struc)

with open('basis.in', 'w') as f:
    f.write(f"{len(uc)}\n")
    for i, mass in enumerate(uc.get_masses()):
        f.write(f"{i} {mass}\n")
    for _ in range(cx*cy*cz): 
        for i in range(len(uc)):
            f.write(f"{i}\n")

npaths = len(points_path)
lengths = [len(path) - 1 for path in points_path]
path_ratio = [float(length / sum(lengths)) for length in lengths]
if npoints % sum(lengths) != 0:
    raise ValueError(f"npoints 建议是 {sum(lengths) * 100} 的倍数")

gpumd_kpts, kpaths, sym_points_list, labels_list = [], [], [], []
for i in range(npaths):
    path = uc.cell.bandpath(path=points_path[i], npoints=path_ratio[i]*npoints, special_points=special_points)
    kpath, sym_points, labels = path.get_linear_kpoint_axis()
    kpaths.append(kpath)
    sym_points_list.append(sym_points)
    labels_list.append(labels)
    kpts = np.matmul(path.kpts, uc.cell.reciprocal() * 2 * np.pi)
    kpts[np.abs(kpts) < 1e-15] = 0.0
    gpumd_kpts = kpts if i == 0 else np.vstack((gpumd_kpts, kpts))
np.savetxt('kpoints.in', gpumd_kpts, header=str(npoints), comments='', fmt='%g')

whole_kpaths, whole_sym_points = [], []
origin_kpaths, origin_sym_points = 0, 0
for i in range(len(kpaths)):
    if i == 0:
        whole_kpaths.extend(kpaths[i])
        origin_kpaths = kpaths[i][-1]
        whole_sym_points.extend(sym_points_list[i])
        origin_sym_points = sym_points_list[i][-1]
    else:
        adjusted_kpaths = [x + origin_kpaths for x in kpaths[i]]
        whole_kpaths.extend(adjusted_kpaths)
        origin_kpaths += kpaths[i][-1]
        adjusted_sym_points = [x + origin_sym_points for x in sym_points_list[i]]
        whole_sym_points.extend(adjusted_sym_points)
        origin_sym_points += sym_points_list[i][-1]
whole_sym_points = list(set(whole_sym_points))

data = np.loadtxt("omega2.out")
for i in range(len(data)):
    for j in range(len(data[0])):
        data[i, j] = np.sqrt(abs(data[i, j])) / (2 * np.pi) * np.sign(data[i, j])
nu = data

""" #qe加这段,vasp不用
data = np.loadtxt("*.freq.gp")
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
#没有phonopy-bandplot --gnuplot > phonon.out这样生成phonon.out
figure(figsize=(9, 8))
if os.path.exists('phonon.out'):
    data_vasp = np.loadtxt('phonon.out')
    vasp_path = data_vasp[:,0] / max(data_vasp[:,0]) * whole_kpaths[-1]
    scatter(vasp_path, data_vasp[:,1], marker='.', color='C1', linewidths=0 ,s=16)
plot(whole_kpaths, nu, color='C0', lw=1)
xlim([0, whole_kpaths[-1]])
for sym_point in whole_sym_points[1:-1]:
    plt.axvline(sym_point, color='black', linestyle='--')
gca().set_xticks(whole_sym_points)
#如果存在断点，这里需要修改，比如['GM', 'KG']这里要写[r'$\Gamma$', 'M|K', r'$\Gamma$']，或许有点顺序问题要改一下
gca().set_xticklabels([r'$\Gamma$', 'M', 'K', r'$\Gamma$'])
ylim([0, 30])
gca().set_yticks(linspace(0,30,7))
ylabel(r'$\nu$ (THz)',fontsize=15)
tick_params(axis='x', which='both', direction='in', top=True, bottom=True)
tick_params(axis='y', which='both', direction='in', left=True, right=True)
if os.path.exists('phonon.out'):
    legend(['DFT', 'NEP'])
else:
    legend(['NEP'])
savefig('phonon.png', dpi=150, bbox_inches='tight')
