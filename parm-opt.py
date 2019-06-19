import scipy
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

method_dict = {'PM3':-7,'AM1':-2, 'OM1':-5, 'OM2':-6, 'OM3':-8,'ODM2':-22, 'ODM3':-23}
            
at_num_dict = {'h':1, 'he':2, 'li':3, 'be':4, 'b':5,'c':6, 'n':7, 'o':8, 'f':9}

def write_input (file, n_struc, n_atoms, at_nums, method_num, mol_num, charge):
    read_file = open(file[0],'r')
    if file[1] == 1:
        scale = 0.529
    else:
        scale = 1
        
    lines = read_file.readlines()
    read_file.close()
    write_file = open(f'mol{mol_num}.inp','w')
    
    n=0
    for structure in range(n_struc):
        if(structure == 0):
                    opt_file = open(f'opt{mol_num}.inp','w')
                    opt_file.write(f'iparok=1 nprint=-4 kharge={charge} iform=1 iop={method_num} jop=0 igeom=1\n')
                    opt_file.write(f'Molecule {mol_num} Optimization\n\n')
        n +=1
        write_file.write(f'iparok=1 nprint=-4 kharge={charge} iform=1 iop={method_num} jop=-1 igeom=1\n')
        write_file.write("Molecule #"+str(structure)+"\n\n")
        for atom in range(n_atoms):
            if(structure == 0):
                opt_file.write(f'{at_nums[atom]} ')
                [opt_file.write(f'{x*scale} 1 ') for x in np.array(lines[n].split(), dtype=float)]
                opt_file.write('\n')

            write_file.write(f'{at_nums[atom]} ')
            [write_file.write(f'{x*scale} 0 ') for x in np.array(lines[n].split(), dtype=float)]
            write_file.write('\n')
            n += 1
        write_file.write('0\n')
        if(structure == 0):
            opt_file.write('0\n')
            opt_file.close()
    write_file.close()

def read_input(file):
    with open(file) as f_in:
        lines = list(filter(None, (line.rstrip() for line in f_in))
    n_parms= int(lines[0].split()[0])
    method = lines[1].split()[0]
    method_num = method_dict[method.upper()]
    n_molec = int(lines[2].split()[0])
    n_atoms=[]
    charge =[]
    structures=[]
    energy_files=[]
    structure_files = []
    at_num=[[] for x in range(n_molec)]
    coords=[[] for x in range(n_molec)]
    geoms =[[] for x in range(n_molec)]
    
    
    n = 3
    for mol in range(n_molec):
        n_atoms.append(int(lines[n].split()[0]))   # set the number of atoms in a molecule
        charge.append(int(lines[n+1].split()[0]))  # set the charge on the molecule
        structures.append(int(lines[n+2].split()[0])) # set the number of structures
        energy_files.append(lines[n+3].split()[0])  # set the file to find the energies
        structure_files.append([lines[n+4].split()[0],int(lines[n+4].split()[1])]) # set the filename for the structures (the second flag is for atomic units = 1 or angstroms =0)
        if (structure_files[mol][1] == 1):
            scale = 0.529
        else:
            scale = 1.
        for atom in range(n_atoms[mol]):
            line = lines[n+5+atom].split() 
            check = np.array(line[1:],dtype=float)
            at_num[mol].append(at_num_dict[line[0].lower()])
            coords[mol].append(np.array([x*scale for x in check]))
        n += n_atoms[mol]+5
        
    return method_num, n_molec, n_atoms, charge, structures, energy_files, structure_files, at_num, coords

def run_mndo(mol_num):
    os.system(f'mndo99 < mol{mol_num}.inp > mol{mol_num}.out')
    os.system(f'mndo99 < opt{mol_num}.inp > opt{mol_num}.out')
    
def read_energies(n_molec):  
    energies=[]
    for mol in range(n_molec):
        energy = []
        with open(f'mol{mol}.out','r') as f:
            data = f.readlines()
        for line in data:
            if "TOTAL ENERGY" in line:
                energy.append(float(line.split()[3]))
        energies = np.hstack((energies,((np.array(energy)-np.min(energy))/27.2114)))
    return np.array(energies)

def read_abinito(energy_files):  
    energies=[]
    for file in energy_files:
        energy = []
        with open(file) as f:
            data = f.readlines()
        for line in data:
            energy.append(float(line.split()[0]))
        energies = np.hstack((energies,((np.array(energy)-np.min(energy)))))
    return np.array(energies)

def calc_fvec():
    for mol in range(n_molec):
        run_mndo(mol)
        
    energies = read_energies(n_molec)
    fvec = (energies-abinitio_energies)
    return fvec, energies

def read_parms(file):
    with open(file) as f_in:
            data = list(filter(None, (line.rstrip() for line in f_in))
    parm_labels = []
    parm_vals   = []
    for line in data:
        if len(line.split()) == 0:
            break
        parm_labels.append(line.split()[0:2])
        parm_vals.append(float(line.split()[2]))
    return parm_labels, parm_vals

def write_parms(X):
    with open('fort.14','w') as f:
        for i,line in enumerate(parm_labels):
            f.write(line[0]+'   '+line[1]+ ' ' +str(X[i])+'\n')

def big_loop(X):
    write_parms(X)  # Write the current set of parameters to fort.14
    fvec, energies = calc_fvec()
    print  ('rmsd  ' + str(627.51*349.75*np.sqrt(np.mean(np.square(fvec)))))
#      (f'RMSD {627.51*349.75*np.sqrt(np.mean(np.square(fvec)))}')
    return fvec


def clear_files():
    os.system('rm mol* fort* opt*')   


#def main():
method_num, n_molec, n_atoms, charge, structures, energy_files, structure_files, at_num, coords = read_input('main.inp')
abinitio_energies = read_abinito(energy_files)
parm_labels, parm_vals = read_parms('INITIALPARMS/pm3.txt')
for mol in range(n_molec):
    write_input(structure_files[mol],
                structures[mol],
                n_atoms[mol],at_num[mol],
                method_num, mol, charge[mol])
                                                 
x, flag = leastsq(big_loop, parm_vals,epsfcn=1e-4)
big_loop(x)
fvec, energies = calc_fvec()
print(f'FINAL RMSD= {349.75*627.51*np.sqrt(np.mean(np.square(fvec)))}')
print(

plt.plot(energies)
plt.plot(abinitio_energies)
plt.savefig('test.png')
#plt.show ()



#main()
