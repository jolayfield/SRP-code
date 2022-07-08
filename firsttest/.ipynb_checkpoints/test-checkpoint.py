from parm-opt import *


method_num, n_molec, n_atoms, charge, structures, energy_files, structure_files, at_num, coords, n_geoms, geoms, n_weights, weights = read_input('main.inp')
abinitio_energies = read_abinito(energy_files)
parm_labels, parm_vals = read_parms(sys.argv[1])
for mol in range(n_molec):
    write_input(structure_files[mol],
                structures[mol],
                n_atoms[mol],at_num[mol],
                method_num, mol, charge[mol])

x, flag = leastsq(big_loop, parm_vals,epsfcn=float(sys.argv[2]))
