import sys
import numpy as np
from pylab import *
from ase.io import read
from sklearn.decomposition import PCA

xyz_file = "train.xyz"
nep_name = "nep.txt"
strucs = read(xyz_file, ":")

def main():
    if len(sys.argv) < 2:
        print("Usage: python sclect_pick_structure.py all #这个看所有点的位置情况")
        print("Usage: python sclect_pick_structure.py select min_distance_1/max_select_1 min_distance_2/max_select_2 ......")
        print("Usage: python sclect_pick_structure.py pick x1_start x1_end y1_start y1_end x2_start x2_end y2_start y2_end ......")
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

def get_indices(file):
    valid_indices = []
    element_indices = {element: [] for element in elements}
    with open(f'{file}.xyz', 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) > 1 and "Lattice" not in line:
                valid_indices.append(line)
    for i, line in enumerate(valid_indices):
        parts = line.split()
        if parts[0] in element_indices:
            element_indices[parts[0]].append(i)
    return element_indices

if os.path.exists('discriptor.out'):
    des = np.loadtxt('discriptor.out')
else:
    nep3=Nep3Calculator(nep_name)
    des = [nep3.get_descriptors(i).mean(0) for i in strucs]
reducer = PCA(n_components=2)
reducer.fit(des)
proj = reducer.transform(des)

with open(xyz_file, 'r') as file:
    lines = file.readlines()
if sys.argv[1] == "all":
    if os.path.exists('discriptor.out'):
        with open('nep.in', 'r') as file:
            for line in file:
                line = line.strip()
                if 'type' in line:
                    elements = line.split()[2:]
    else:
        None
    if len(des) == len(strucs):
        sc = scatter(proj[:, 0], proj[:, 1], c=energy_train[:,1], cmap='Blues', edgecolor='grey', alpha=0.8)
        cbar = colorbar(sc, cax=gca().inset_axes([0.75, 0.95, 0.23, 0.03]), orientation='horizontal')
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        cbar.set_label('E/atom (eV)')
        title('Descriptors for each structure')
    else:
        element_descriptors = {element: [] for element in elements}
        element_indices = get_indices('train')
        for element in elements:
            for idx in element_indices[element]:
                element_descriptors[element].append(proj[idx])
        for element in elements:
            scatter([i[0] for i in element_descriptors[element]], [i[1] for i in element_descriptors[element]], edgecolor='grey', alpha=0.8, label=element)
        legend(frameon=False, fontsize=10, loc='upper right')
        title('Descriptors for each atom')
    xlabel('PC1')
    ylabel('PC2')
    tight_layout()
    savefig("all-points.png", dpi=150, bbox_inches='tight')

elif sys.argv[1] == "select":
    struc_lines = [0]
    for atoms in strucs:
        struc_lines.append(struc_lines[-1] + len(atoms)+2)
    min_distances = []
    counts = []
    for arg in sys.argv[2:]:
        if float(arg) < 1:
            min_distances.append(float(arg))
        else:
            counts.append(int(arg))
    select_values = min_distances + counts
    for select_value in select_values:
        if float(select_value) < 1:
            selected_strucs = FarthestPointSample(des, min_distance=select_value)
        else:
            selected_strucs = FarthestPointSample(des, min_distance=0, max_select=select_value)
        with open(f"selected_{select_value}.xyz", 'w') as file1:
            for i, j in enumerate(selected_strucs):
                file1.writelines(lines[struc_lines[j]:struc_lines[j+1]])
        abandoned_strucs = [i for i in range(len(strucs)) if i not in selected_strucs]
        with open(f"abandoned_{select_value}.xyz", 'w') as file1:
            for i, j in enumerate(abandoned_strucs):
                file1.writelines(lines[struc_lines[j]:struc_lines[j+1]])
        
        scatter(proj[:, 0], proj[:, 1], alpha=0.8, c="#8e9cff", label="All")
        selected_proj = reducer.transform(np.array([des[i] for i in selected_strucs]))
        if float(select_value) < 1:
            scatter(selected_proj[:, 0], selected_proj[:, 1], alpha=0.7, c="#e26fff", label="min_distance={}".format(select_value))
        else:
            scatter(selected_proj[:, 0], selected_proj[:, 1], alpha=0.7, c="#e26fff", label="select_counts={}".format(select_value))
        xlabel('PC1')
        ylabel('PC2')
        legend()
        savefig(f"select_{select_value}.png", dpi=150, bbox_inches='tight')

elif sys.argv[1] == "pick":
    struc_lines = [0]
    for atoms in strucs:
        struc_lines.append(struc_lines[-1] + len(atoms)+2)
    picked_proj = set()
    for i in range(2, len(sys.argv), 4):
        x_start, x_end, y_start, y_end = map(float, sys.argv[i:i+4])
        range_x = (x_start, x_end)
        range_y = (y_start, y_end)
        current_proj = pick_points(proj, range_x, range_y)
        picked_proj.update(current_proj)
    picked_proj = list(picked_proj)
    with open("picked.xyz", 'w') as file1:
        for i in picked_proj:
            file1.writelines(lines[struc_lines[i]:struc_lines[i+1]])
    retained_strucs = [i for i in range(len(strucs)) if i not in picked_proj]
    with open("retained.xyz", 'w') as file1:
        for i in retained_strucs:
            file1.writelines(lines[struc_lines[i]:struc_lines[i+1]])
    scatter(proj[picked_proj, 0], proj[picked_proj, 1], alpha=0.5, color='C1', label="Picked")
    scatter(proj[retained_proj, 0], proj[retained_proj, 1], alpha=0.5, color='C0', label="Retained")
    legend()
    savefig("retain-pick.png", dpi=150, bbox_inches='tight')

