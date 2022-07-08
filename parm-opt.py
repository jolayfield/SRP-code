import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import math
from scipy.optimize import leastsq

method_dict = {'PM3':-7,'AM1':-2, 'RM1':-2, 'OM1':-5, 'OM2':-6, 'OM3':-8,'ODM2':-22, 'ODM3':-23}
            
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
                    opt_file.write(f'iparok=1 iscf=7 iplscf=7 nsav15=9 kharge={charge} iform=1 iop={method_num} jop=0 igeom=1\n')
                    opt_file.write(f'Molecule {mol_num} Optimization\n\n')
        n +=1
        write_file.write(f'iparok=1 iscf=7 iplscf=7 nsav15=9 kharge={charge} iform=1 iop={method_num} jop=-1 igeom=1\n')
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
        lines = f_in.readlines() 
    n_parms= int(lines[0].split()[0])
    method = lines[1].split()[0]
    method_num = method_dict[method.upper().replace("ORTHO", "")]
    n_molec = int(lines[2].split()[0])
    n_atoms=[]
    charge =[]
    structures=[]
    energy_files=[]
    structure_files = []
    n_geoms=[]
    n_weights=[]
    at_num=[[] for x in range(n_molec)]
    coords=[[] for x in range(n_molec)]
    geoms=[[] for x in range(n_molec)]
    weights=[[] for x in range(n_molec)]
    
    
    
    n = 3
    for mol in range(n_molec):
        n_atoms.append(int(lines[n].split()[0]))   # set the number of atoms in a molecule
        charge.append(int(lines[n+1].split()[0]))  # set the charge on the molecule
        structures.append(int(lines[n+2].split()[0])) # set the number of structures
        energy_files.append(lines[n+3].split()[0])  # set the file to find the energies
        structure_files.append([lines[n+4].split()[0],int(lines[n+4].split()[1])])# set the filename for the structures (the second flag is for atomic units = 1 or angstroms =0)
        if (structure_files[mol][1] == 1):
            scale = 0.529
        else:
            scale = 1.
        for atom in range(n_atoms[mol]):
            line = lines[n+5+atom].split() 
            check = np.array(line[1:],dtype=float)
            at_num[mol].append(at_num_dict[line[0].lower()])
            coords[mol].append(np.array([x*scale for x in check]))
        n_geoms.append(int(lines[n+5+n_atoms[mol]].split()[0])) #set the number of geometry calculations
        for geom in range(n_geoms[mol]):
            geoms[mol].append(lines[n+6+n_atoms[mol]+geom].split())
        n += n_atoms[mol]+n_geoms[mol]+6
        n_weights.append(int(lines[n].split()[0]))
        for weight in range(n_weights[mol]):
            weights[mol].append(lines[n+1+weight].split())
        n += n_weights[mol]+1
        
    return method_num, n_molec, n_atoms, charge, structures, energy_files, structure_files, at_num, coords, n_geoms, geoms, n_weights, weights

def run_mndo(mol_num):
    os.system(f'mndo99 < mol{mol_num}.inp > mol{mol_num}.out')
    os.system(f'mv fort.15 mol{mol_num}.aux')
    os.system(f'mndo99 < opt{mol_num}.inp > opt{mol_num}.out')
    os.system(f'mv fort.15 opt{mol_num}.aux')
    
def read_opt(mol_num):
    intgeom = []
    optgeom = []
    with open(f'opt{mol_num}.out','r') as outfile:
        optlines = outfile.readlines()
    for n,line in enumerate(optlines):
        if "INPUT GEOMETRY" in line:
            for atoms in range(n_atoms[mol_num]):
                m = optlines[n+atoms+6].strip()
                o = m.split()[2::2]
                intgeom.append(o)   
        if "FINAL CARTESIAN GRADIENT NORM" in line:
            for atoms in range(n_atoms[mol_num]):
                z = optlines[n+atoms+8].strip()
                y = z.split()[2::2]
                optgeom.append(y)
    intgeom = np.array(intgeom).astype(float)
    optgeom = np.array(optgeom).astype(float)
    outfile.close()
    return intgeom, optgeom
    
def comp_geoms(n_molec):
    return_geoms = [] 
    for mol in range(n_molec):
        intgeom, optgeom = read_opt(mol)
        for geom in geoms[mol]:
            if geom[0] == 'bond':
                at_bond = ' '.join(map(str, geom))
                dist1 = np.linalg.norm(intgeom[int(geom[1])-1]-intgeom[int(geom[2])-1])
                dist4 = np.linalg.norm(optgeom[int(geom[1])-1]-optgeom[int(geom[2])-1])
                diff1 = dist1-dist4
                return_geoms.append(diff1)
                print(f'{at_bond} {dist1:.4} {dist4:.4} {diff1:.4}') #how many decimal places?
            if geom[0] == 'angle':    
                at_ang = ' '.join(map(str, geom))
                dot1 = np.dot((intgeom[int(geom[1])-1]-intgeom[int(geom[2])-1]), (intgeom[int(geom[3])-1]-intgeom[int(geom[2])-1]))
                dist2 = np.linalg.norm(intgeom[int(geom[1])-1]-intgeom[int(geom[2])-1])
                dist3 = np.linalg.norm(intgeom[int(geom[3])-1]-intgeom[int(geom[2])-1])
                cos1 = dot1/(dist2*dist3)
                angle1 = (math.acos(cos1))*57.295779513
                dot2 = np.dot((optgeom[int(geom[1])-1]-optgeom[int(geom[2])-1]), (optgeom[int(geom[3])-1]-optgeom[int(geom[2])-1]))
                dist5 = np.linalg.norm(optgeom[int(geom[1])-1]-optgeom[int(geom[2])-1])
                dist6 = np.linalg.norm(optgeom[int(geom[3])-1]-optgeom[int(geom[2])-1])
                cos2 = dot2/(dist5*dist6)
                angle2 = (math.acos(cos2))*57.295779513
                diff2 = angle1-angle2
                return_geoms.append(diff2)
                print(f'{at_ang} {angle1:.4} {angle2:.4} {diff2:.4}')
    return np.array(return_geoms)
                

def read_energies(n_molec):  
    energies=[]
    for mol in range(n_molec):
        energy = []
        with open(f'mol{mol}.aux','r') as f:
            data = f.readlines()
        for n,line in enumerate(data):
            if "ENERGY" in line:
                energy.append(float(data[n+1].split()[0]))
        energies = np.hstack((energies,((np.array(energy)-np.min(energy))/627.50956)))
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

def calc_fvec(structures, weights, n_geoms, geoms):
    for mol in range(n_molec):
        run_mndo(mol)
    w = np.ones(np.sum(structures+n_geoms)) 
    s = 0
    for mol in range(n_molec):
        for weight in weights[mol]:
            if weight[0] == '1':
                w[int(weight[3])-1+s:int(weight[4])-1+s] = w[int(weight[3])-1+s:int(weight[4])-1+s]*int(weight[1])
            if weight[0] == '2':
                for i in range(structures[mol]):
                    if str(i) in weight[2:]:
                        w[i]+=float(weight[1])
        s += structures[mol]
    for mol in range(n_molec):
        for geom in geoms[mol]:
            w[s] = w[s]*int(geom[-1])
            s += 1      
    energies = read_energies(n_molec)
    fvec = (energies-abinitio_energies)*627.51*349.75
    fvec = np.hstack((fvec,comp_geoms(n_molec)))
    print  ('rmsd  ' + str(np.sqrt(np.mean(np.square(fvec)))))
    fvec = fvec*w
    return fvec, energies

def read_parms(file):
    with open(file) as f_in:
            data = f_in.readlines() 
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
    fvec, energies = calc_fvec(structures, weights, n_geoms, geoms)
    return fvec


def clear_files():
    os.system('rm mol* fort* opt*')   



method_num, n_molec, n_atoms, charge, structures, energy_files, structure_files, at_num, coords, n_geoms, geoms, n_weights, weights = read_input('main.inp')
abinitio_energies = read_abinito(energy_files)
parm_labels, parm_vals = read_parms(sys.argv[1])
for mol in range(n_molec):
    write_input(structure_files[mol],
                structures[mol],
                n_atoms[mol],at_num[mol],
                method_num, mol, charge[mol])
                                                 
x, flag = leastsq(big_loop, parm_vals, epsfcn=float(sys.argv[2]))
big_loop(x)
fvec, energies = calc_fvec(structures, weights, n_geoms, geoms)
if np.sum(n_geoms) > 0:
    print  ('FINAL RMSD ' + str(np.sqrt(np.mean(np.square(fvec[0:-np.sum(n_geoms)])))))
else:
    print  ('FINAL RMSD  ' + str(np.sqrt(np.mean(np.square(fvec)))))
plt.plot(energies)
plt.plot(abinitio_energies)
plt.savefig('test.png', dpi=300)
