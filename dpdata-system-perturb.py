    def perturb(
        self,
        pert_num: int,
        cell_pert_fraction: float,
        atom_pert_distance: float,
        atom_pert_style: str = "normal",
        atom_pert_prob: float = 1.0,
        elem_pert_list: list = None,
    ):
        """Perturb each frame in the system randomly.
        The cell will be deformed randomly, and atoms will be displaced by a random distance in random direction.

        Parameters
        ----------
        pert_num : int
            Each frame in the system will make `pert_num` copies,
            and all the copies will be perturbed.
            That means the system to be returned will contain `pert_num` * frame_num of the input system.
        cell_pert_fraction : float
            A fraction determines how much (relatively) will cell deform.
            The cell of each frame is deformed by a symmetric matrix perturbed from identity.
            The perturbation to the diagonal part is subject to a uniform distribution in [-cell_pert_fraction, cell_pert_fraction),
            and the perturbation to the off-diagonal part is subject to a uniform distribution in [-0.5*cell_pert_fraction, 0.5*cell_pert_fraction).
        atom_pert_distance : float
            unit: Angstrom. A distance determines how far atoms will move.
            Atoms will move about `atom_pert_distance` in random direction.
            The distribution of the distance atoms move is determined by atom_pert_style
        atom_pert_style : str
            Determines the distribution of the distance atoms move is subject to.
            Avaliable options are
                - `'normal'`: the `distance` will be object to `chi-square distribution with 3 degrees of freedom` after normalization.
                    The mean value of the distance is `atom_pert_fraction*side_length`
                - `'uniform'`: will generate uniformly random points in a 3D-balls with radius as `atom_pert_distance`.
                    These points are treated as vector used by atoms to move.
                    Obviously, the max length of the distance atoms move is `atom_pert_distance`.
                - `'const'`: The distance atoms move will be a constant `atom_pert_distance`.
        atom_pert_prob : float
            Determine the proportion of the total number of atoms in a frame that are perturbed.
        elem_pert_list: list
            Determine to perturb the atoms corresponding to the specified element.

        Returns
        -------
        perturbed_system : System
            The perturbed_system. It contains `pert_num` * frame_num of the input system frames.
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
                        pert_indices = list(range(len(tmp_system.data["coords"][0])))
                pert_natoms = int(atom_pert_prob * len(pert_indices)
                pert_atom_id = sorted(
                    np.random.choice(
                        pert_indices,
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
