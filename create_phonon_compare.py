import os
from pylab import *
from ase.io import read

uc = read("model.xyz")   #必须是原胞
kpoints = ['GMKG']     #高对称点路径，可从vaspkit305等获取，可以写断点比如['GM', 'KG']
dft_file = 'phonon.out'
y_max = 10
# kpoints.in文件也可手写
with open('kpoints.in', 'w') as f:
    for kp in kpoints:
        path = uc.cell.bandpath(path=kp, npoints=0)
        for label in kp:
            k = path.special_points[label]
            f.write(f"{k[0]:.3f} {k[1]:.3f} {k[2]:.3f} {label}\n")
        f.write("\n")

with open("omega2.out", 'r') as f:
    first_line = f.readline().strip().lstrip('#').split()
    sym_points = first_line[:len(first_line)//2]
    sym_labels = first_line[len(first_line)//2:]
sym_points = [float(x) for x in sym_points]
sym_labels = [r'$\Gamma$' if x == 'G' else x for x in sym_labels]

omega2 = np.loadtxt("omega2.out")
npath = omega2[:, 0]
nu = omega2[:, 1:]
for i in range(len(nu)):
    for j in range(len(nu[0])):
        nu[i, j] = np.sqrt(abs(nu[i, j])) / (2 * np.pi) * np.sign(nu[i, j])

figure(figsize=(8,8))

if os.path.exists(dft_file):
    dft = np.loadtxt(dft_file)
    idx0 = np.where(dft[:, 0] == 0.000000)[0]
    idx0 = np.append(idx0, len(dft))
    blocks = [dft[:, 1][idx0[i]:idx0[i+1]] for i in range(len(idx0)-1)]
    DFT = np.column_stack(blocks)
    dft_path = dft[idx0[0]:idx0[1],0] / dft[-1,0] * npath [-1]
    plot(dft_path, DFT, color='C0', lw=2)

plot(npath, nu, color='C1',lw=2)
xlim([0, npath[-1]]);xticks(fontsize=18)
vlines(sym_points[1:-1], ymin=0, ymax=y_max, color='black', linestyle='--')
gca().set_xticks(sym_points)
gca().set_xticklabels(sym_labels)
ylim([0, y_max]);yticks(fontsize=18)
ylabel(r'$\nu$ (THz)', fontsize=18)
tick_params(axis='x', which='both', direction='in', top=True, bottom=True)
tick_params(axis='y', which='both', direction='in', left=True, right=True)
if os.path.exists(dft_file):
    plot([], [], color='C0', lw=2, label='DFT')
    plot([], [], color='C1', lw=2, label='NEP')
    legend()
    savefig('phonon_NEPvsDFT.png', dpi=300, bbox_inches='tight')
else:
    plot([], [], color='C1', lw=2, label='NEP')
    legend()
    savefig('phonon_NEP.png', dpi=300, bbox_inches='tight')
