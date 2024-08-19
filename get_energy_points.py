import sys
import numpy as np
from ase.io import read, write

def main():
    if len(sys.argv) < 2:
        print("Usage: python get_energy_points.py <energy.out> <*.xyz> <dif_value+> <dif_value->")
        sys.exit(1)
if __name__ == "__main__":
    main()

energt_train = np.loadtxt(sys.argv[1])
value = energt_train[:,1] - energt_train[:,0]
dif_value_positive = float(sys.argv[3])
dif_value_negative = float(sys.argv[3])
dif_id = np.where((value > dif_value_positive) | (value < -dif_value_negative))
struc_id = dif_id[0].tolist() 

strucs = read(sys.argv[2], ":")
write('deviate.xyz', [strucs[i] for i in struc_id], format='extxyz', write_results=False)
retain_id = [i for i in range(len(strucs)) if i not in struc_id]
write('reserve.xyz', [strucs[i] for i in retain_id], format='extxyz', write_results=False)