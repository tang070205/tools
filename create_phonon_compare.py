import numpy as np
from pylab import *
from ase.io import read,write

uc = read('POSCAR-unitcell') #xyz、cif文件也可以
cx,cy,cz = 20, 20, 1
struc = uc * (cx,cy,cz)
write("model.xyz", struc)

with open('basis.in', 'w') as f:
    f.write(f"{len(uc)}\n")
    for i, mass in enumerate(uc.get_masses()):
        f.write(f"{i} {mass}\n")
    for _ in range(cx*cy*cz): 
        for i in range(len(uc)):
            f.write(f"{i}\n")

special_points = {'G': [0, 0, 0], 'M': [0.5, 0, 0], 'K': [0.3333, 0.3333, 0], 'G': [0, 0, 0]}
path = uc.cell.bandpath(path='GMKG', npoints = 400, special_points=special_points) #这里的npoints不能在外面单独设置，要不可以和下面的400共用一个单独定义的npoints
kpath, sym_points, labels = path.get_linear_kpoint_axis()
gpumd_kpts = np.matmul(path.kpts, uc.cell.reciprocal() * 2 * np.pi)
gpumd_kpts[np.abs(gpumd_kpts) < 1e-15] = 0.0
np.savetxt('kpoints.in', gpumd_kpts, header=str(400), comments='', fmt='%g') #这里的400要和npoints等于的数一致

data = np.loadtxt("omega2.out")

for i in range(len(data)):
    for j in range(len(data[0])):
        data[i, j] = np.sqrt(abs(data[i, j])) / (2 * np.pi) * np.sign(data[i, j])
nu = data

""" #qe加这段,vasp不用
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
#没有phonopy-bandplot --gnuplot > phonon.out这样生成phonon.out，就不加这段和下面仨加vasp注释的，相当于只画gpumd的结果
data_vasp = np.loadtxt('phonon.out') #如果没有vasp的结果，就不加这段及下面画图那行
vasp_path = data_vasp[:,0] / max(data_vasp[:,0]) * max(kpath) #第一个数是phonon.out文件最大横坐标，第二个文件时omega2最大横坐标

figure(figsize=(9, 8))
plt.scatter(vasp_path, data_vasp[:,1], marker='.', edgecolors='C1', facecolors='none')#vasp
plot(kpath, nu, color='C0', lw=1)
xlim([0, max(kpath)])
for sym_point in sym_points[1:-1]:
    plt.axvline(sym_point, color='black', linestyle='--') 
gca().set_xticks(sym_points)
gca().set_xticklabels([r'$\Gamma$', 'M', 'K', '$\Gamma$'])
ylim([0, 6])  
gca().set_yticks(range(0, 7, 1))
ylabel(r'$\nu$ (THz)',fontsize=15)
legend(['DFT', 'NEP']) #只画gpumd就只用NEP图例即可，或者不加图例
savefig('phonon.png', dpi=150, bbox_inches='tight')


            