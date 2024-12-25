大家对脚本有什么建议都可以提出来，大胆提issue

aimd-OUTCARs-xyz2POSCAR.py is used to extract the structure of aimd and then separate it before using the ase package to derive POSCARs for each single point energy to be calculated.
dumpxyz2POSCAR.py It's a scaled-down version of aimd, a POSCAR exported directly to the GPUMD output dump.xyz with ase.

create-strain_deform-rattle-perturb.py is to generate a training set using hiphive and dpdata.
create-perturb.py is only useing dpdata but usually is enough.

create_phonon_compare.py is to compare the phonon dispersion of GPUMD and VASP (or QE).

plot_hnemd_multiple.py is a drawing of the hnemd method computed many times using the GPUMD software package(HNEMD method).
plot_shc_multiple.py is a drawing of the shc method computed many times using the GPUMD software package.

plt_nep_results.py is to graph the nep output file, both for training and prediction.

split_group.py is used to generate model.xyz files and group them (NEMD method).
split_heterojunction_group.py can group grain boundaries or heterojunctions.

plot_hac_multiple.py Unlike the manual does not require gpyumd (EMD method).