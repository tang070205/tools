import numpy as np
import dpdata
from dpdata import System
from dpdata.System import get_cell_perturb_matrix, get_atom_perturb_vector

def elem_perturb(
    self,
    pert_num: int,
    cell_pert_fraction: float,
    atom_pert_distance: float,
    atom_pert_style: str = "normal",
    atom_pert_prob: float = 1.0,
    elem_pert_list: list = None,
):
    """
    Add Parameters
    ----------
    elem_pert_list: list
        Determine to perturb the atoms corresponding to the specified element.
    """
    if type(self) is not dpdata.System:
        raise RuntimeError(
            f"Using method perturb() of an instance of {type(self)}. "
            f"Must use method perturb() of the instance of class dpdata.System."
        )
    perturbed_system = System()
    nframes = self.get_nframes()
    for ii in range(nframes):
        for jj in range(pert_num):
            tmp_system = self[ii].copy()
            cell_perturb_matrix = get_cell_perturb_matrix(cell_pert_fraction)
            tmp_system.data["cells"][0] = np.matmul(
                tmp_system.data["cells"][0], cell_perturb_matrix
            )
            tmp_system.data["coords"][0] = np.matmul(
                tmp_system.data["coords"][0], cell_perturb_matrix
            )
            if elem_pert_list is not None:
                if all(isinstance(el, str) for el in elem_pert_list):
                    elements = [tmp_system.data["atom_names"][types] for types in tmp_system.data["atom_types"]]
                    perturb_indices = [idx for idx, el in enumerate(elements) if el in elem_pert_list]
                elif all(isinstance(el, int) for el in elem_pert_list):
                    perturb_indices = elem_pert_list
                else:
                    perturb_indices = list(range(len(tmp_system.data["coords"][0])))
            pert_natoms = int(atom_pert_prob * len(perturb_indices))
            pert_atom_id = sorted(
                np.random.choice(
                    perturb_indices,
                    pert_natoms,
                    replace=False,
                ).tolist()
            )
            for kk in pert_atom_id:
                atom_perturb_vector = get_atom_perturb_vector(
                    atom_pert_distance, atom_pert_style
                )
                tmp_system.data["coords"][0][kk] += atom_perturb_vector
            tmp_system.rot_lower_triangular()
            perturbed_system.append(tmp_system)
    return perturbed_system
