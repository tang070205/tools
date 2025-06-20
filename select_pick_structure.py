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
strucs, des, all_strucs, append_des, extend_des, des_names = {}, {}, [], [], [], []
for i, (xyz, descriptor) in enumerate(zip(xyz_file, des_file)):
    if os.path.exists(xyz):
        strucs[i] = read(xyz, ":")
        all_strucs.extend(strucs[i])
    else:
        print(f"File {xyz} does not exist.")
        sys.exit(1)

    if os.path.exists(descriptor):
        des = np.loadtxt(descriptor)
    elif descriptor == "0":
        des_name = os.path.splitext(os.path.basename(xyz))[0]
        des_names.append(des_name)
        if os.path.exists(f'descriptor-{des_name}.out'):
            des = np.loadtxt(f'descriptor-{des_name}.out')
        else:
            from NepTrain.core.nep import *
            nep = Nep3Calculator(nep_name)
            descriptor = [nep.get_descriptors(j).mean(0) for j in strucs[i]]
            with open(f'descriptor-{des_name}.out', "w") as f:
                for descriptor in descriptor:
                    descriptor_str = " ".join(map(str, descriptor))
                    f.write(f"{descriptor_str}\n")
            des = np.loadtxt(f'descriptor-{des_name}.out')
    append_des.append(des)
    extend_des.append(des)

def main():
    if len(sys.argv) < 3:
        print("methon: pca umap tsne kpca svd isomap se ica fa mds grp srp")
        print("Usage: python select_pick_structure.py all <method> # Observe the position of all points")
        print("Usage: python select_pick_structure.py select <method> min_distance_1/max_select_1 min_distance_2/max_select_2 ......")
        print("Usage: python select_pick_structure.py pick <method> x1_start x1_end y1_start y1_end x2_start x2_end y2_start y2_end ......")
        sys.exit(1)
if __name__ == "__main__":
    main()

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

def get_energies_per_atom(xyzfile):
    strucs = read(xyzfile, format='xyz', index=':')
    energies_per_atom = []
    for atoms in strucs:
        energy_per_atom = atoms.get_potential_energy() / len(atoms)
        energies_per_atom.append(energy_per_atom)
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
    from sklearn.manifold import MDS  # pip install scikit-learn 
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

if sys.argv[1] == "all":
    for i in range(len(append_des)):
        if len(append_des[i]) == len(strucs[i]):
            proj = reducer.fit_transform(append_des[i])
            energy_train = get_energies_per_atom(xyz_file[i])
            sc = scatter(proj[:, 0], proj[:, 1], c=energy_train, cmap='Blues', edgecolor=all_color[i], alpha=0.8, label=des_names[i])
        else:
            element_descriptors, all_des_symbols = {element: [] for element in elements}, []
            for des_struc in all_strucs:
                des_symbol = des_struc.get_chemical_symbols()
                all_des_symbols.extend(des_symbol)
            for symbol, des in zip(all_des_symbols, extend_des):
                element_descriptors[symbol].append(des)
            element_des = {element: des for element, des in element_descriptors.items() if des}
            for element in element_des.keys():
                scatter([i[0] for i in element_des[element]], [i[1] for i in element_des[element]], edgecolor='grey', alpha=0.8, label=element)
            legend(frameon=False, fontsize=10, loc='upper right')
            title(f'Descriptors for each atom with {sys.argv[2]}')
    if len(all_des[0]) == len(strucs[0]):
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
    all_proj = reducer.fit_transform(extend_des)
    min_distances, counts = [], []
    for arg in sys.argv[3:]:
        if float(arg) < 1:
            min_distances.append(float(arg))
        else:
            counts.append(int(arg))
    select_values = min_distances + counts
    for select_value in select_values:
        if float(select_value) < 1:
            select_ids = sorted(set(FarthestPointSample(extend_des, min_distance=select_value)))
        else:
            select_ids = sorted(set(FarthestPointSample(extend_des, min_distance=0, max_select=select_value)))
        write(f"select_{select_value}.xyz", [all_strucs[j] for j in select_ids], format='extxyz')
        abandon_ids = [i for i in range(len(all_strucs)) if i not in select_ids]
        write(f"abandon_{select_value}.xyz", [all_strucs[j] for j in abandon_ids], format='extxyz')
        scatter(all_proj[:, 0], all_proj[:, 1], alpha=0.8, c="#8e9cff", label="All")
        selected_proj = array([all_proj[i] for i in select_ids])
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
    all_proj = reducer.fit_transform(extend_des)
    pick_ids = set()
    for i in range(3, len(sys.argv), 4):
        x_start, x_end, y_start, y_end = map(float, sys.argv[i:i+4])
        range_x = (x_start, x_end)
        range_y = (y_start, y_end)
        current_ids = pick_points(all_proj, range_x, range_y)
        pick_ids.update(current_ids)
    pick_ids = list(pick_ids)
    write(f"pick.xyz", [all_strucs[j] for j in pick_ids], format='extxyz')
    retain_ids = [i for i in range(len(all_strucs)) if i not in pick_ids]
    write(f"retain.xyz", [all_strucs[j] for j in retain_ids], format='extxyz')
    scatter(all_proj[pick_ids, 0], all_proj[pick_ids, 1], alpha=0.5, color='C1', label="Pick")
    scatter(all_proj[retain_ids, 0], all_proj[retain_ids, 1], alpha=0.5, color='C0', label="Retain")
    set_tick_params()
    xlabel('PC1')
    ylabel('PC2')
    legend()
    savefig(f"retain-pick-{sys.argv[2]}.png", dpi=150, bbox_inches='tight')