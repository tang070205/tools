大家对脚本有什么建议都可以提出来, 大胆提issue

aimd_OUTCAR_xyz2POSCAR.py is used to extract the structure of aimd (vasp), then separate it, and use an ase package to export POSCAR for each single point energy to be calculated.

create-perturb.py only uses dpdata, but it is usually sufficient.

create_phonon_compare.py is used to compare the phonon dispersion between GPUMD and phonopy exported (VASP/abacus/QE).If you have phonon spectrum data calculated by other DFT or MD software, I can add corresponding drawing code to enter the script.
![phonon](https://github.com/user-attachments/assets/e6d52e29-2090-44cb-b1ea-e40fee8151db)![phonon_with_group_velocity](https://github.com/user-attachments/assets/3766d76b-d8d8-4201-bace-21d178dab5fd)

create_strain_deform_rattle_perturb.py uses hiphive and dpdata to generate the training set.

dumpxyz2POSCAR.py is a scaled down version of aimd, which directly uses the dump. xyz file output from GPUMD and exports it to POSCAR using ASE to calculate single point energy.

get_nep_points2strucs.py is used to filter points in the graph based on the nep diagonal graph and output the corresponding structure.

inorexyz2exyz_ratio.py is a conversion from the old training set format (. in) to the new exyz format, similar in functionality to the shuf_xyz.Py script of pynep.

messxyz2POSCAR.py is designed to calculate the single point energy of a trajectory, but the number of atoms in each structure of the trajectory is not the same. It is similar to the dummy xyz2POSCAR. py script.

plot_compute_shc.py is a drawing script for calculating the nemd result in a single run.
![compute](https://github.com/user-attachments/assets/84f189b3-25b1-4aca-ab7c-3827fd52f7e7)![shc](https://github.com/user-attachments/assets/8f1609d2-3357-4e2c-a301-16204e66fd35)

plot_hac_multiple.py is a graph that calculates the hac method multiple times using the GPUMD software package (EMD method).
![emd-multiple](https://github.com/user-attachments/assets/e4e5b71d-6557-49e6-9b43-f0198a41baea)

plot_kappa_multiple.py is a graph that calculates the hnemd method multiple times using the GPUMD software package (hnemd method).
![hnemd](https://github.com/user-attachments/assets/179af5b4-c685-441e-9030-b020d3957f07)

plot_kappa_shc.py is a graphical script for calculating the hnemd result in a single run.
![hnemd](https://github.com/user-attachments/assets/9bc541ed-b7a0-485f-aa07-85180ab5a07b)![shc](https://github.com/user-attachments/assets/473b0b46-1b33-4875-a2a0-b750f699cea6)

plot_nep_results.py is used to draw the nep output file for training and prediction.
![nep-loss](https://github.com/user-attachments/assets/a7cf843b-dd83-4ae5-8a3d-24d57076fb3b)![nep-efvs-diagonals](https://github.com/user-attachments/assets/2c356661-91fb-4a4c-bbb5-ca97231adce7)![nep-descriptor](https://github.com/user-attachments/assets/796ea239-ca51-4894-9b03-00333af0f48a)![nep-descriptor-peratom](https://github.com/user-attachments/assets/095992b6-567c-4b8b-a6af-a33ad57f2f65)

plot_shc_multiple.py is a graph that calculates the shc method multiple times using the GPUMD software package.

select_pick_structure.py is a drawing script that performs farthest point sampling based on descriptors and reduces dimensionality using PCA.
![select_200](https://github.com/user-attachments/assets/79a50d55-a471-4e6e-8528-f7e6078901f6)

split_group.py is used to generate the model. xyz file and group it (only NEMD method).
split_heterojunction_group.py can be used to group grain boundaries or heterojunctions.

The multiFrame-abacus2nep-exyz.sh and singleFrame-abacus2nep-exyz.sh are shell scripts for extracting the aimd and single point energy of abacus, respectively.
