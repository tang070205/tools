import sys
import numpy as np
from ase.io import read, write
import matplotlib.pyplot as plt
from calorine.nep import get_descriptors
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

strucs = read("structures.xyz", ":")
nep_name = "nep.txt"

def main():
    if len(sys.argv) < 2:
        print("Usage: python sclect_pick_structure.py all #这个看所有点的位置情况")
        print("Usage: python sclect_pick_structure.py select min_distance_1 min_distance_2 ......")
        print("Usage: python sclect_pick_structure.py pick x1_start x1_end y1_start y1_end x2_start x2_end y2_start y2_end ......")
        sys.exit(1)
if __name__ == "__main__":
    main()

def FarthestPointSample(new_data, now_data=[], min_distance=0.1, min_select=1, max_select=None, metric='euclidean'):
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


des = np.array([np.mean(get_descriptors(i, model_filename=nep_name), axis=0) for i in strucs])
reducer = PCA(n_components=2)
reducer.fit(des)
proj = reducer.transform(des)


if sys.argv[1] == "all":
    plt.scatter(proj[:, 0], proj[:, 1], alpha=0.5, c="C0")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig("all-points.png", dpi=150, bbox_inches='tight')

elif sys.argv[1] == "select":
    min_distances = [float(arg) for arg in sys.argv[2:]]
    for min_distance in min_distances:
        selected_strucs = FarthestPointSample(des, min_distance=min_distance)
        write(f"selected_{min_distance}.xyz", [strucs[i] for i in selected_strucs], format='extxyz', write_results=False)
        abandoned_strucs = [i for i in range(len(strucs)) if i not in selected_strucs]
        write(f"abandoned_{min_distance}.xyz", [strucs[i] for i in abandoned_strucs], format='extxyz', write_results=False)
        plt.scatter(proj[:, 0], proj[:, 1], alpha=0.5, c="C0", label="All")
        selected_proj = reducer.transform(np.array([des[i] for i in selected_strucs]))
        plt.scatter(selected_proj[:, 0], selected_proj[:, 1], s=8, color='C1', label="Selected at {}".format(min_distance))
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.savefig(f"select_{min_distance}.png", dpi=150, bbox_inches='tight')

if sys.argv[1] == "pick":
    picked_proj = set()
    for i in range(2, len(sys.argv), 4):
        x_start, x_end, y_start, y_end = map(float, sys.argv[i:i+4])
        range_x = (x_start, x_end)
        range_y = (y_start, y_end)
        current_proj = pick_points(proj, range_x, range_y)
        picked_proj.update(current_proj)
    picked_proj = list(picked_proj)
    write("picked.xyz", [strucs[i] for i in picked_proj], format='extxyz', write_results=False)
    retained_proj = [i for i in range(len(strucs)) if i not in picked_proj]
    write('retained.xyz', [strucs[i] for i in retained_proj], format='extxyz', write_results=False)
    plt.scatter(proj[picked_proj, 0], proj[picked_proj, 1], alpha=0.5, color='C1', label="Picked")
    plt.scatter(proj[retained_proj, 0], proj[retained_proj, 1], alpha=0.5, color='C0', label="Retained")
    plt.legend()
    plt.savefig("retain-pick.png", dpi=150, bbox_inches='tight')

