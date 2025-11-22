import os, re
import numpy as np
from pylab import *
from ase.io import read,write

cx,cy,cz = 20, 20, 1   # 超胞参数
npoints = 300
special_points = {'G': [0, 0, 0], 'K': [0.3333, 0.3333, 0], 'M': [0.5, 0, 0]}  # 高对称点坐标，同样vaspkit305提供的文件里有
points_path = ['GMKG']     #高对称点路径，可以写断点比如['GM', 'KG']，最后还要设置横坐标的标签gca().set_xticklabels([])
dft_file = 'phonon.out'  # DFT计算得到的声子频率文件，没有phonopy-bandplot --gnuplot > phonon.out这样生成phonon.out
uc = read('POSCAR') #xyz、cif文件也可以
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

def generate_set_path(points_path):
    segs = []
    for seg in points_path:
        tokens = re.findall(r'([A-Za-z])(\d*)', seg.upper())
        segs.append([r'$\Gamma$' if L == 'G' else f'${L}_{{{N}}}$' if N else L for L, N in tokens])

    labels = segs[0][:]
    for i in range(1, len(segs)):
        prev_tail = labels[-1]
        curr_head = segs[i][0]
        labels[-1] = f'{prev_tail}|{curr_head}'
        labels.extend(segs[i][1:])

    return labels

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

data = np.loadtxt("omega2.out")
for i in range(len(data)):
    for j in range(len(data[0])):
        data[i, j] = np.sqrt(abs(data[i, j])) / (2 * np.pi) * np.sign(data[i, j])
nep = data

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

figure(figsize=(8, 7))
if os.path.exists(dft_file):
    dft = np.loadtxt(dft_file)
    idx0 = np.where(dft[:, 0] == 0.000000)[0]
    idx0 = np.append(idx0, len(dft))
    blocks = [dft[:, 1][idx0[i]:idx0[i+1]] for i in range(len(idx0)-1)]
    DFT = np.column_stack(blocks)
    dft_path = dft[idx0[0]:idx0[1],0] / dft[-1,0] * whole_kpaths[-1]
    plot(dft_path, DFT, color='C1', lw=1)

plot(whole_kpaths, nep, color='C0', lw=1)
xlim([0, whole_kpaths[-1]])
for sym_point in whole_sym_points[1:-1]:
    plt.axvline(sym_point, color='black', linestyle='--')
gca().set_xticks(whole_sym_points)
gca().set_xticklabels(generate_set_path(points_path))
ylim([0, 35])
gca().set_yticks(linspace(0,35,8))
ylabel(r'$\nu$ (THz)',fontsize=15)
tick_params(axis='x', which='both', direction='in', top=True, bottom=True)
tick_params(axis='y', which='both', direction='in', left=True, right=True)
if os.path.exists(dft_file):
    plot([], [], color='C1', lw=1, label='DFT')
    plot([], [], color='C0', lw=1, label='NEP')
    legend()
else:
    plot([], [], color='C0', lw=1, label='NEP')
    legend()
savefig('phonon.png', dpi=150, bbox_inches='tight')
