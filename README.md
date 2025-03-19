大家对脚本有什么建议都可以提出来, 大胆提issue

aimd_OUTCAR_xyz2POSCAR.py is used to extract the structure of aimd (vasp), then separate it, and use an ase package to export POSCAR for each single point energy to be calculated.

create-perturb.py only uses dpdata, but it is usually sufficient.

create_phonon_compare.py is used to compare the phonon dispersion between GPUMD and phonopy exported (or QE).

create_strain_deform_rattle_perturb.py uses hiphive and dpdata to generate the training set.

dumpxyz2POSCAR.py is a scaled down version of aimd, which directly uses the dump. xyz file output from GPUMD and exports it to POSCAR using ASE to calculate single point energy.

get_nep_points2strucs.py is used to filter points in the graph based on the nep diagonal graph and output the corresponding structure.

inorexyz2exyz_ratio.py is a conversion from the old training set format (. in) to the new exyz format, similar in functionality to the shuf_xyz.Py script of pynep.

messxyz2POSCAR.py is designed to calculate the single point energy of a trajectory, but the number of atoms in each structure of the trajectory is not the same. It is similar to the dummy xyz2POSCAR. py script.

plot_compute_shc.py is a drawing script for calculating the nemd result in a single run.

plot_hac_multiple.py is a graph that calculates the hac method multiple times using the GPUMD software package (EMD method).

plot_kappa_multiple.py is a graph that calculates the hnemd method multiple times using the GPUMD software package (hnemd method).

plot_kappa_shc.py is a graphical script for calculating the hnemd result in a single run.

plot_nep_results.py is used to draw the nep output file for training and prediction.

plot_shc_multiple.py is a graph that calculates the shc method multiple times using the GPUMD software package.

select_pick_structure.py is a drawing script that performs farthest point sampling based on descriptors and reduces dimensionality using PCA.

split_group.py is used to generate the model. xyz file and group it (NEMD method).
split_heterojunction_group.py can be used to group grain boundaries or heterojunctions.

The multiFrame-abacus2nep-exyz.sh and singleFrame-abacus2nep-exyz.sh are shell scripts for extracting the aimd and single point energy of abacus, respectively.
