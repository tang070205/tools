import numpy as np
from ase.io import read, write
import matplotlib.pyplot as plt
from calorine.nep import get_descriptors
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

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

strucs = read("dump.xyz", ":")
des = np.array([np.mean(get_descriptors(i, model_filename='nep.txt'), axis=0) for i in strucs])
selected_i = FarthestPointSample(des, min_distance=0.05)  
write("selected.traj", [strucs[i] for i in selected_i])
abandoned_i = [i for i in range(len(strucs)) if i not in selected_i]
write('test.xyz', [strucs[i] for i in abandoned_i])
reducer = PCA(n_components=2)
reducer.fit(des)
proj = reducer.transform(des)
plt.scatter(proj[:, 0], proj[:, 1], alpha=0.5, c="C0", label="All")
selected_proj = reducer.transform(np.array([des[i] for i in selected_i]))
plt.scatter(selected_proj[:, 0], selected_proj[:, 1], s=8, color='C1', label="Selected")
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.savefig("select.png")
