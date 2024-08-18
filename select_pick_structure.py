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
        print("Usage: python sclect_pick_structure.py select min_distance")
        print("Usage: python sclect_pick_structure.py pick x1 x2 y1 y2")
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
    plt.savefig("all-points.png")

elif sys.argv[1] == "select":
    min_distance = float(sys.argv[2])
    selected_strucs = FarthestPointSample(des, min_distance=min_distance)  
    write("selected.xyz", [strucs[i] for i in selected_strucs], format='extxyz', write_results=False)
    abandoned_strucs = [i for i in range(len(strucs)) if i not in selected_strucs]
    write('abandoned.xyz', [strucs[i] for i in abandoned_strucs], format='extxyz', write_results=False)
    plt.scatter(proj[:, 0], proj[:, 1], alpha=0.5, c="C0", label="All")
    selected_proj = reducer.transform(np.array([des[i] for i in selected_strucs]))
    plt.scatter(selected_proj[:, 0], selected_proj[:, 1], s=8, color='C1', label="Selected")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.savefig("select.png")

elif sys.argv[1] == "pick":
    range_x = (float(sys.argv[2]), float(sys.argv[3]))
    range_y = (float(sys.argv[4]), float(sys.argv[5]))
    pick_proj = pick_points(proj, range_x, range_y)
    picked_strucs = [strucs[i] for i in pick_proj]
    write("picked.xyz", picked_strucs, format='extxyz', write_results=False)
    retained_proj = [i for i in range(len(strucs)) if i not in pick_proj]
    write('retained.xyz', [strucs[i] for i in retained_proj], format='extxyz', write_results=False)
    plt.scatter(proj[pick_proj, 0], proj[pick_proj, 1], alpha=0.5, color='C1', label="Picked")
    plt.scatter(proj[retained_proj, 0], proj[retained_proj, 1], alpha=0.5, color='C0', label="Retainted")
    plt.legend()
    plt.savefig("retain-pick.png")
