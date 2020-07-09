import scipy
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import math
from scipy.optimize import leastsq

method_dict = {'PM3':-7,'AM1':-2, 'RM1':-2, 'OM1':-5, 'OM2':-6, 'OM3':-8,'ODM2':-22, 'ODM3':-23,'XTB':-14}
            
at_num_dict = {'h':1, 'he':2, 'li':3, 'be':4, 'b':5,'c':6, 'n':7, 'o':8, 'f':9}
at_sym_dict = {1:"H", 2:"He", 3:"Li", 4:"Be", 5:"B", 6:"C", 7:"N",8:"O", 9:"F"}

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
                    opt_file.write(f'iparok=1 nsav15=9 nprint=-4 kharge={charge} iform=1 iop={method_num} jop=0 igeom=1\n')
                    opt_file.write(f'Molecule {mol_num} Optimization\n\n')
        n +=1
        write_file.write(f'iparok=1 nsav15=9 nprint=-4 kharge={charge} iform=1 iop={method_num} jop=-1 igeom=1\n')
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
            check = np.array(line[1:4],dtype=float)
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
        with open(f'mol{mol}.out','r') as f:
            data = f.readlines()
        for n,line in enumerate(data):
            if "SCF TOTAL ENERGY" in line:
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

def calc_fvec(structures, weights, n_geoms, geoms,method):
    if method == -14:
        #xtb
    else:
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

def big_loop(X,method):
    write_parms(X)  # Write the current set of parameters to fort.14
    fvec, energies = calc_fvec(structures, weights, n_geoms, geoms,method)
    # if np.sum(n_geoms) > 0:
        # print  ('rmsd  ' + str(np.sqrt(np.mean(np.square(fvec[0:-np.sum(n_geoms)])))))
    #else:
       # print  ('rmsd  ' + str(np.sqrt(np.mean(np.square(fvec)))))
#      (f'RMSD {627.51*349.75*np.sqrt(np.mean(np.square(fvec)))}')   
    return fvec

def ev_to_hartree(energies):
    new_energies = [energy/27.2114 for energy in energies]
    return new_energies

def zero_energy(endData):
    energies = np.array(endData) 
    energies -= np.min(energies)
    return energies

def anp_int_spec(energies,n_atoms,at_num,input_file):
    
    energies = zero_energy(energies)
    iterData = iter(energies) 
    
    anpass_header    = "../anpass.header"
    anpass_footer    = "../anpass1.footer"
    intder_header1   = "../new-geom.in"
    intder_header2   = "../intder.in"
    spectro_template = "../spectro.in"
    dispDat          = "../disp.dat"
    fileName = "freq_files"
   
    os.mkdir(fileName)
    os.chdir(fileName)
    
    anpassInput = open("xtb-Anpass","w")
    with open(anpass_header,"r") as readHeader: #read header
        header = readHeader.readlines()
    for line in header:
        anpassInput.write(line) #write header
    with open(dispDat) as displacement:
        disp = displacement.readlines()
    for line in disp:
        anpassInput.write(f"{line.rstrip()}{next(iterData):20.12f}\n")
    with open(anpass_footer,"r") as foot: #read footer
        footer = foot.readlines()
    for line in footer:
        anpassInput.write(line) 
    anpassInput.close()
    os.system("/home/freu9584/bin/anpass-fit.e <xtb-Anpass> Anpass1.out")
    
    secondInput = open("AnpassSecond","w") 
    with open('xtb-Anpass',"r") as copyFile:
        anpassLines = copyFile.readlines()
    for line in anpassLines[:-4]:
        secondInput.write(line)
    secondInput.write(f"STATIONARY POINT\n")
    with open("fort.983","r") as statPoint:
        statData = statPoint.readline()
    secondInput.write(statData)
    secondInput.write(f"END OF DATA\n!FIT\n!END\n")
    secondInput.close()
    os.system(f"/home/freu9584/bin/anpass-fit.e <AnpassSecond> Anpass2.out")
    
    outputFile = open('IntderFile',"w")
    with open(intder_header1,"r") as headerFile:
        header = headerFile.readlines()
    data = list(map(lambda u:float(u),statData.split()[:-1]))
    disps = []
    for l in data:
        if l != 0.0:
            disps.append(l)
    for line in header[:-3]:
        outputFile.write(line)
        if "DISP" in line: break
    for t in range(len(disps)) :
        outputFile.write(f"{t+1: 5}{disps[t]:22.12f}\n")
    outputFile.write(f"    0\n")
    outputFile.close()
    os.system("/home/freu9584/bin/Intder2005.e <IntderFile> Intder.out")
    
    end_file = open("IntderFile2","w")
    with open(intder_header2,"r") as headerFile:
        header = headerFile.readlines()
        
    deriv = int(header[1].split()[3]) 

    w = 0
    for line in header:
        w+=1
        end_file.write(line)
        if line.strip() == "0": break
            
    with open('Intder.out',"r") as geomFile:
        geometry = geomFile.readlines()
    for line in geometry[-n_atoms:]:
        numbers = line.split()
        end_file.write(f"{float(numbers[0]):18.10f}{float(numbers[1]):19.10f}{float(numbers[2]):19.10f}\n")
    end_file.write(header[w+n_atoms])

    os.system("/home/freu9584/c-c4/sort_fort.sh")

    with open("sorted_fort.9903","r") as symmetryFile:
        symmetry = symmetryFile.readlines()
    columnCounter = 2 
    begin = False
    for line in symmetry[:]: 
        if begin == True:
            tempLine = line.split()
            if not(columnCounter == 4) and not(tempLine[columnCounter]=="0"):
                columnCounter+=1
                end_file.write(f"    0\n")
            end_file.write(line)
        elif len(line.split()) ==1:
            begin = True
    end_file.write(f"    0\n")
    end_file.close()
    os.system("/home/freu9584/bin/Intder2005.e <IntderFile2> Intder2.out")
    
    spectroFile = open("SpectroFile","w")
    atomicNumIter = iter(at_num)
    with open(spectro_template,"r") as templateFile:
        template = templateFile.readlines()
    for line in template[:5]:
        spectroFile.write(line)
    with open("Intder2.out","r") as geomFile:
        geom = geomFile.readlines()
    for line in geom[16:16+n_atoms]:
        numbers = line.split()
        spectroFile.write(f"{next(atomicNumIter):5.2f}{float(numbers[0]):19.10f}{float(numbers[1]):19.10f}{float(numbers[2]):19.10f}\n")
    for line in template[5+n_atoms:]:
        spectroFile.write(line)
    spectroFile.close()
    
    num = iter([[15,15],[20,30],[24,40]])
    for p in range(3):
        files = next(num)
        with open(f"file{files[0]}","r") as originalFile:
            original = originalFile.readlines()
        copy = open(f"fort.{files[1]}","w")
        for line in original:
            copy.write(line)
        copy.close()
    os.system("/home/freu9584/bin/spectro.e <SpectroFile> Spectro.out")
    os.chdir("..")

def xtb_method(n_atoms, charge, structures, structure_file, at_nums, template):
    
    input_path = "xtb_files"
    os.system(f"mkdir {input_path}") #making inputs folder
    moleculeName = os.getcwd().split("/")[-2]
    
    with open(os.path.join(moleculeName,structure_file),"r") as geomfile:
        lines = geomfile.readlines()
    padding_zeros = len(str(structures)) # calculate the number of padding zeroes needed from the number of molecules
    first =1
    last =n_atoms+1
    
    for mol in range(structures):
        current_file = open(os.path.join(input_path,f"{moleculeName}-coord{mol:0{padding_zeros}}.dat"),"w") 
        # formatted strings allow for added variables in the string. 
        atom_count = -1
        current_file.write("$coord\n")
        for line in lines[first:last]:
            if line[0] !="#":
                atom_count += 1
                current_file.write(line.strip()+" "+ at_sym_dict[at_nums[atom_count]]+"\n")
        current_file.write("$end\n")
        first = last+1
        last = first+n_atoms
        current_file.close()
    
    endData = []
    os.system(f"echo {charge} > .CHRG")# specifying the charge of the molecule before running xtb
    for mol in range(structures):
        in_file = os.path.join(input_path,f"{moleculeName}-coord{mol:0{padding_zeros}}.dat")
        output_name = os.path.join(input_path,'output')
        os.system(f"xtb {in_file} > {output_name}")
        with open(output_name,"r") as output:
            out_lines = output.readlines()
        for line in out_lines:
            if ("TOTAL ENERGY" in line):
                endData.append(float(line.split()[3]))
    return endData

def clear_files():
    os.system('rm mol* fort* opt*')   

energies = []

if method_num == -14:
    for mol in range(n_molec):
        energies.append(xtb_method(n_atoms[mol], charge[mol], structures[mol], structure_files[mol][0],at_num[mol]))
else:
    method_num, n_molec, n_atoms, charge, structures, energy_files, structure_files, at_num, coords, n_geoms, geoms, n_weights, weights = read_input('main.inp')
    sturcture_files = np.array(structure_files)
    abinitio_energies = read_abinito(energy_files)
    parm_labels, parm_vals = read_parms(sys.argv[1])
    for mol in range(n_molec):
        write_input(structure_files[mol],
                    structures[mol],
                    n_atoms[mol],at_num[mol],
                    method_num, mol, charge[mol])
                                                 
    x, flag = leastsq(big_loop, parm_vals,epsfcn=1e-4)
    big_loop(x)
    fvec, energies = calc_fvec(structures, weights, n_geoms, geoms)
    if np.sum(n_geoms) > 0:
        print  ('FINAL RMSD ' + str(np.sqrt(np.mean(np.square(fvec[0:-np.sum(n_geoms)])))))
    else:
        print  ('FINAL RMSD  ' + str(np.sqrt(np.mean(np.square(fvec)))))
    plt.plot(energies)
    plt.plot(abinitio_energies)
    plt.savefig('test.png', dpi=300)

for n, energy in enumerate(energies):
    anp_int_spec(energy,n_atoms[n],at_num[n])

#main()
