import sys, os
import numpy as np
from pylab import *
from ase.io import read, write

#xyz_file和des_file顺序要对应，des_file可以没有(用"0"代替)
nep_name = "nep.txt"
xyz_file = ["train.xyz", "test.xyz"]
des_file = ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"]
all_color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5']
#all_color = ["grey", "MediumVioletRed", "MediumSpringGreen"]
strucs, des, proj , all_strucs, all_des = {}, {}, {}, [], []
for i, (xyz, descriptor) in enumerate(zip(xyz_file, des_file)):
    if os.path.exists(xyz):
        strucs[i] = read(xyz, ":")
        all_strucs.extend(strucs[i])
    else:
        print(f"File {xyz} does not exist.")
        sys.exit(1)

    if os.path.exists(descriptor):
        des[i] = np.loadtxt(descriptor)
    elif des_file[i] == "0":
        des_name = os.path.splitext(os.path.basename(xyz_file[i]))[0]
        if os.path.exists(f'descriptor-{des_name}.out'):
            des[i] = np.loadtxt(f'descriptor-{des_name}.out')
        else:
            from NepTrain.core.nep import *
            nep3=Nep3Calculator(nep_name)
            descriptor = [nep3.get_descriptors(j).mean(0) for j in strucs[i]]
            with open(f'descriptor-{des_name}.out', "w") as f:
                for descriptor in descriptor:
                    descriptor_str = " ".join(map(str, descriptor))
                    f.write(f"{descriptor_str}\n")
            des[i] = np.loadtxt(f'descriptor-{des_name}.out')
    all_des.append(des[i])

def main():
    if len(sys.argv) < 3:
        print("methon: pca umap tsne kpca svd isomap se ica fa mds grp srp")
        print("Usage: python select_pick_structure.py all <method> # Observe the position of all points")
        print("Usage: python select_pick_structure.py select <method> min_distance_1/max_select_1 min_distance_2/max_select_2 ......")
        print("Usage: python select_pick_structure.py pick <method> x1_start x1_end y1_start y1_end x2_start x2_end y2_start y2_end ......")
        sys.exit(1)
if __name__ == "__main__":
    main()

def get_structure_lines():
    struc_lines = [0]
    for atoms in all_strucs:
        struc_lines.append(struc_lines[-1] + len(atoms)+2)
    return struc_lines

def FarthestPointSample(new_data, now_data=[], min_distance=0.1, min_select=1, max_select=None, metric='euclidean'):
    from scipy.spatial.distance import cdist
    max_select = max_select or len(new_data)
    to_add = []
    if len(new_data) == 0:
        return to_add
    if len(now_data) == 0:
        to_add.append(0)
        now_data.append(new_data[0])
    distances = np.min(cdist(new_data, now_data, metric=metric), axis=1)
    while np.max(distances) > min_distance or len(to_add) < min_select:
        i = np.argmax(distances)
        to_add.append(i)
        if len(to_add) >= max_select:
            break
        distances = np.minimum(distances, cdist([new_data[i]], new_data, metric=metric)[0])
    return to_add

def pick_points(proj, range_x, range_y):
    pick_strucs = []
    for i, point in enumerate(proj):
        if range_x[0] <= point[0] <= range_x[1] and range_y[0] <= point[1] <= range_y[1]:
            pick_strucs.append(i)
    return pick_strucs

def get_indices(xyzfile):
    valid_indices = []
    element_indices = {element: [] for element in elements}
    with open(xyzfile, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) > 1 and "Lattice" not in line:
                valid_indices.append(line)
    for i, line in enumerate(valid_indices):
        parts = line.split()
        if parts[0] in element_indices:
            element_indices[parts[0]].append(i)
    return element_indices

def get_energies_per_atom(xyzfile):
    with open(xyzfile, 'r') as file:
        lines = file.readlines()
    struc_lines, energies_per_atom = [], []
    for line in lines:
        columns = line.strip().split()
        if len(columns) != 7:
            struc_lines.append(line)
    i = 0
    while i < len(struc_lines):
        num_atoms_line = struc_lines[i].strip()
        energy_line = struc_lines[i+1].strip()
        num_atoms = int(num_atoms_line)
        energy_value = float(energy_line.split('nergy=')[1].split()[0])
        energy_per_atom = energy_value / num_atoms
        energies_per_atom.append(energy_per_atom)
        i += 2
    return energies_per_atom

def set_tick_params():
    tick_params(axis='x', which='both', direction='in', top=True, bottom=True)
    tick_params(axis='y', which='both', direction='in', left=True, right=True)

if sys.argv[2] == "umap":
    from umap import UMAP  # pip install umap-learn
    reducer = UMAP(n_components=2, random_state=42, n_jobs=1)
elif sys.argv[2] == "pca":
    from sklearn.decomposition import PCA  # pip install scikit-learn 
    reducer = PCA(n_components=2)
elif sys.argv[2] == "tsne":
    from sklearn.manifold import TSNE  # pip install scikit-learn
    reducer = TSNE(n_components=2)
elif sys.argv[2] == "isomap":
    from sklearn.manifold import Isomap  # pip install scikit-learn 
    reducer = Isomap(n_components=2)
elif sys.argv[2] == "svd":
    from sklearn.decomposition import TruncatedSVD  # pip install scikit-learn 
    reducer = TruncatedSVD(n_components=2)
elif sys.argv[2] == "kpca":
    from sklearn.decomposition import KernelPCA  # pip install scikit-learn 
    reducer = KernelPCA(n_components=2)#, kernel='rbf', gamma=0.1)
elif sys.argv[2] == "mds":
    from sklearn.manifold import MDS
    reducer = MDS(n_components=2, random_state=42)
elif sys.argv[2] == "se":
    from sklearn.manifold import SpectralEmbedding  # pip install scikit-learn 
    reducer = SpectralEmbedding(n_components=2, random_state=42)
elif sys.argv[2] == "fa":
    from sklearn.decomposition import FactorAnalysis  # pip install scikit-learn 
    reducer = FactorAnalysis(n_components=2, random_state=42)
elif sys.argv[2] == "ica":
    from sklearn.decomposition import FastICA  # pip install scikit-learn 
    reducer = FastICA(n_components=2, max_iter=1000, random_state=42)
elif sys.argv[2] == "grp":
    from sklearn.random_projection import GaussianRandomProjection  # pip install scikit-learn 
    reducer = GaussianRandomProjection(n_components=2, random_state=42)
elif sys.argv[2] == "srp":
    from sklearn.random_projection import SparseRandomProjection  # pip install scikit-learn 
    reducer = SparseRandomProjection(n_components=2, random_state=42)
for i in range(len(des)):
    proj[i] = reducer.fit_transform(des[i])

if sys.argv[1] == "all":
    for i in range(len(des)):
        if len(des[i]) == len(strucs[i]):
            energy_train = get_energies_per_atom(xyz_file[i])
            sc = scatter(proj[i][:, 0], proj[i][:, 1], c=energy_train, cmap='Blues', edgecolor=all_color[i], alpha=0.8, label=os.path.splitext(os.path.basename(xyz_file[i]))[0])
        else:
            with open(nep_name, 'r') as file:
                first_line = file.readline().strip()
                elements = first_line.split()[2:]
            element_descriptors = {element: [] for element in elements}
            element_indices = get_indices(xyz_file[0])
            for element in elements:
                for idx in element_indices[element]:
                    element_descriptors[element].append(proj[idx])
            for element in elements:
                scatter([i[0] for i in element_descriptors[element]], [i[1] for i in element_descriptors[element]], edgecolor='grey', alpha=0.8, label=element)
            legend(frameon=False, fontsize=10, loc='upper right')
            title(f'Descriptors for each atom with {sys.argv[2]}')
    if len(des[0]) == len(strucs[0]):
        cbar = colorbar(sc, cax=gca().inset_axes([0.73, 0.95, 0.23, 0.03]), orientation='horizontal')
        #cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        cbar.set_ticks([sc.get_clim()[0], sc.get_clim()[1]])
        cbar.set_ticklabels(['{:.1f}'.format(sc.get_clim()[0]), '{:.1f}'.format(sc.get_clim()[1])])
        cbar.set_label('E/atom (eV)')
        legend(ncol=3, frameon=False, fontsize=5)
        title(f'Descriptors for each structure with {sys.argv[2]}')
    set_tick_params()
    xlabel('PC1')
    ylabel('PC2')
    tight_layout()
    savefig(f"all-points-{sys.argv[2]}.png", dpi=150, bbox_inches='tight')

elif sys.argv[1] == "select":
    struc_lines = get_structure_lines()
    min_distances = []
    counts = []
    for arg in sys.argv[3:]:
        if float(arg) < 1:
            min_distances.append(float(arg))
        else:
            counts.append(int(arg))
    select_values = min_distances + counts
    for select_value in select_values:
        if float(select_value) < 1:
            select_ids = sorted(set(FarthestPointSample(des[0], min_distance=select_value)))
        else:
            select_ids = sorted(set(FarthestPointSample(des[0], min_distance=0, max_select=select_value)))
        write(f"select_{select_value}.xyz", [strucs[0][j] for j in select_ids], format='extxyz')
        abandon_ids = [i for i in range(len(strucs[0])) if i not in select_ids]
        write(f"abandon_{select_value}.xyz", [strucs[0][j] for j in abandon_ids], format='extxyz')
        scatter(proj[0][:, 0], proj[0][:, 1], alpha=0.8, c="#8e9cff", label="All")
        selected_proj = array([proj[0][i] for i in select_ids])
        if float(select_value) < 1:
            scatter(selected_proj[:, 0], selected_proj[:, 1], alpha=0.7, c="#e26fff", label="min_distance={}".format(select_value))
        else:
            scatter(selected_proj[:, 0], selected_proj[:, 1], alpha=0.7, c="#e26fff", label="select_counts={}".format(select_value))
        set_tick_params()
        xlabel('PC1')
        ylabel('PC2')
        legend()
        savefig(f"select-{select_value}-{sys.argv[2]}.png", dpi=150, bbox_inches='tight')

elif sys.argv[1] == "pick":
    struc_lines = get_structure_lines()
    pick_ids = set()
    for i in range(3, len(sys.argv), 4):
        x_start, x_end, y_start, y_end = map(float, sys.argv[i:i+4])
        range_x = (x_start, x_end)
        range_y = (y_start, y_end)
        current_ids = pick_points(proj, range_x, range_y)
        pick_ids.update(current_ids)
    pick_ids = list(pick_ids)
    write(f"pick.xyz", [strucs[0][j] for j in pick_ids], format='extxyz')
    retain_ids = [i for i in range(len(strucs)) if i not in pick_ids]
    write(f"retain.xyz", [strucs[0][j] for j in retain_ids], format='extxyz')
    scatter(proj[pick_ids, 0], proj[pick_ids, 1], alpha=0.5, color='C1', label="Pick")
    scatter(proj[retain_ids, 0], proj[retain_ids, 1], alpha=0.5, color='C0', label="Retain")
    set_tick_params()
    xlabel('PC1')
    ylabel('PC2')
    legend()
    savefig(f"retain-pick-{sys.argv[2]}.png", dpi=150, bbox_inches='tight')
