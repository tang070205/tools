aimd-OUTCARs-xyz2POSCAR.py is to extract the structure for aimd, then separate it, and then use ovito to derive a POSCAR for each single point energy to be calculated.

create-strain_deform-rattle-perturb.py is to generate a training set using hiphive and dpdata.
create-perturb.py is only useing dpdata but usually is enough.

create_phonon_compare.py is to compare the phonon dispersion of GPUMD and VASP (or QE).

database2xyz.py is to extract .db format files, which may not be universal for the time being.

dumpxyz2POSCAR.py It's a scaled-down version of aimd, a POSCAR exported directly to the GPUMD output dump.xyz with ovito

plot_hnemd_multiple.py is a drawing of the hnemd method computed many times using the GPUMD software package.

plt_nep_results.py is to graph the nep output file, both for training and prediction.

