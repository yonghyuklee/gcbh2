##########################################################################################
# this is an example script to run the Grand Canonical Basin Hopping
# this self-contained script generates the executables to run basin hopping
# you still need to add the chemical potentials, input.traj, and the bh_options.json
##########################################################################################
# generalizations to the code such as general lammmps input file, etc. to come or whatever

import glob, os, sys, json, argparse
import itertools
import numpy as np
from scipy import sparse
from ase.io import read, write
from ase.neighborlist import NeighborList, natural_cutoffs
from gcbh2.scripts.gcbh2 import GrandCanonicalBasinHopping
from pygcga2 import (randomize_all,
                     remove_H, 
                     remove_O, 
                     add_multiple_H, 
                     add_H, 
                     add_O, 
                     add_OH, 
                     add_cluster, 
                     add_OH_cluster,
                     add_H_cluster,
                     add_O_cluster,
                     cluster_random_perturbation, 
                     cluster_random_displacement,
                     add_molc_on_cluster,
                     molc_random_displacement,
                     examine_unconnected_components)

atom_elem_to_num = {"H": 1, "O": 8, "Zr": 40, "Cu": 29, "Pd": 46, "C": 6,}
elements = {
            1 : 40,
            2 : 8,
            3 : 1,
            4 : 29,
            5 : 46,
            6 : 6,
            }
bond_range = {
              ("Zr", "Zr"): [1.0, 10],
              ("Zr", "O"): [1.0, 10],
              ("Zr", "H"): [1.0, 10],
              ("O", "O"): [0.8, 10],
              ("O", "H"): [0.8, 10],
              ("H", "H"): [0.5, 10],
              ("Zr", "Cu"): [2.0, 10],
              ("Zr", "Pd"): [2.0, 10],
              ("O", "Cu"): [1.4, 10],
              ("O", "Pd"): [1.4, 10],
              ("H", "Cu"): [1.0, 10],
              ("H", "Pd"): [1.0, 10],
              ("Cu", "Cu"): [2.0, 10],
              ("Cu", "Pd"): [2.0, 10],
              ("Pd", "Pd"): [2.0, 10],
              ("C", "Cu"): [1.8, 10],
              ("C", "Pd"): [1.8, 10],
              ("C", "Zr"): [1.8, 10],
              ("C", "O"): [1.2, 10],
              ("C", "H"): [1.0, 10],
              ("C", "C"): [1.0, 10],
             }

def write_molc(molc_type):
    if molc_type == 'Cyclohexene':
        with open("Cyclohexene.in", "w") as f:
            f.write("""lattice_vector 19.2995739999999998 0.0000000000000000 0.0000000000000000
lattice_vector 0.0000000000000000 19.9647199999999998 0.0000000000000000
lattice_vector 0.0000000000000000 0.0000000000000000 15.0000000000000000
atom 9.8074937939011217 11.4469232395023468 7.3952685003005261 C
atom 11.0305955423200324 10.6407355124439675 7.8483849799441101 C
atom 11.0178443521410081 9.2397495829935625 7.2297531540738103 C
atom 9.7723810901485955 8.4614476067691626 7.6705453398276031 C
atom 8.5263318699080806 9.3022767238146571 7.5992605750695974 C
atom 8.5421279549259701 10.6344536319917964 7.4539974259784580 C
atom 9.9578847593489410 11.8185140073176562 6.3646036580590994 H
atom 9.6943996749883006 12.3513186562151489 8.0154622230303065 H
atom 11.9572320070369251 11.1738658786280531 7.5888878207810375 H
atom 11.0145118053491373 10.5479135544400684 8.9473188055743904 H
atom 11.0147522858749252 9.3328755813810016 6.1307115421049323 H
atom 11.9295172313321505 8.6857618608552674 7.4984580237121579 H
atom 9.6451181628011220 7.5597147997373257 7.0492343481303363 H
atom 9.9040166048156220 8.0866908557791319 8.7026185491075676 H
atom 7.5668041242266320 8.7817241813181752 7.6634347400064815 H
atom 7.5953227481274048 11.1766154478630568 7.3805171241036058 H
                    """)
    elif molc_type == 'Benzene':
        with open ("Benzene.in", "w") as f:
            f.write("""lattice_vector 19.2995739999999998 0.0000000000000000 0.0000000000000000
lattice_vector 0.0000000000000000 19.9647199999999998 0.0000000000000000
lattice_vector 0.0000000000000000 0.0000000000000000 15.0000000000000000
atom 9.6497899999999994 11.3798899999999996 7.5000000000000000 C
atom 10.8600800000000000 10.6811199999999999 7.5000000000000000 C
atom 10.8600800000000000 9.2835999999999999 7.5000000000000000 C
atom 9.6497899999999994 8.5848300000000002 7.5000000000000000 C
atom 8.4395000000000007 9.2835999999999999 7.5000000000000000 C
atom 8.4395000000000007 10.6811199999999999 7.5000000000000000 C
atom 9.6497899999999994 12.4707299999999996 7.5000000000000000 H
atom 11.8047799999999992 11.2265300000000003 7.5000000000000000 H
atom 11.8047799999999992 8.7381899999999995 7.5000000000000000 H
atom 9.6497899999999994 7.4939900000000002 7.5000000000000000 H
atom 7.4947999999999997 8.7381899999999995 7.5000000000000000 H
atom 7.4947999999999997 11.2265300000000003 7.5000000000000000 H
                    """)
    else:
        raise RuntimeError("Molecule type unknown")

def write_opt_file(atom_order, lammps_loc, model_label=None, model_path=None, multiple=False, molc=None):
    # opt.py file
    if multiple:
        with open("opt.py", "w") as f:
            f.write("""import re
import os
import glob

from ase.io import *
from ase.io.trajectory import TrajectoryWriter
from ase import Atoms
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.constraints import FixAtoms
from xyz2data import *
from mpi4py import MPI
from lammps import PyLammps
from scipy import sparse
from ase.neighborlist import NeighborList, natural_cutoffs
from pygcga2 import examine_unconnected_components
# from pymatgen.io.lammps.data import LammpsData
# from pymatgen.io.ase import AseAtomsAdaptor

re_energies = re.compile(\"\"\"^\s*Step \"\"\")

atom_elem_to_num = {"H": 1, "O": 8, "Zr": 40, "Cu": 29, "Pd": 46, "C": 6,}
atom_order = ["Zr", "O", "H", "Cu", "Pd", "C",]
""")
            if molc:
                f.write(f"molc = {molc}\n")
            f.write("""
def lammps_energy(
                  logfile,
                  ):
    lf = open(logfile,'r')
    line = lf.readline()
    while line:
        if re_energies.match(line):
            energies = []
            w = line.split()
            n = np.where(np.array(w) == 'PotEng')[0][0]
            line = lf.readline()
            while line: 
                if line.startswith("WARNING"):
                    line = lf.readline()
                else:
                    try:
                        w = line.split()
                        energies.append([float(a) for a in line.split()][n])
                    except:
                        break
                    line = lf.readline() 
        line = lf.readline()
    return energies

def examine_water_molecule_presents(newatoms):
    nat_cut = natural_cutoffs(newatoms, mult=1.25)
    nl = NeighborList(nat_cut, skin=0, self_interaction=False, bothways=True)
    nl.update(newatoms)
    matrix = nl.get_connectivity_matrix()
    surf_ind = []
    for a in newatoms:
        surf_ind.append(a.index)
    water = []
    for i in surf_ind:
        if newatoms[i].symbol == 'O':
            indices, offsets = nl.get_neighbors(i)
            near_H = []
            for a in indices:
                if newatoms[a].symbol == 'H':
                    near_H.append(a)
            # print(near_H)
            if len(near_H) >= 2:
                water.append(i)
                for n in near_H:
                    water.append(n)
    print("There are {} number of water molecules in the system".format(len(water)/3))
    if len(water) > 0:
        return True
    else:
        return False
                                  
def main():
    L = PyLammps()
    L.command("units metal")
    L.command("atom_style charge")

    atoms = read("./input.traj", ":")
    final_atoms = []
    #n = len(atoms, ":")
    for n, atom in enumerate(atoms):
        os.makedirs("%02d" % n, exist_ok=True)
        os.chdir("%02d" % n)""")
        #     if molc:
        #         f.write("""
        # molc = []
        # for a in atom:
        #     if a.tag == 2:
        #         molc.append(a.index)
        #             """)
            f.write("""
        xyz2data(
                 atom,
                 vacuum_layer = 10,
                 filename = 'slab.data',
                 slab = True,
                 qO = -0.82,
                 )

        atom_order_str = []
        atom_order_copy = atom_order.copy()
#        for s in atom_order_copy:
#            if all(a.symbol != s for a in atom):
#                atom_order_copy.remove(s)
        for a in atom_order_copy:
            atom_order_str.append(atom_elem_to_num[a])
        atom_order_str = ' '.join(map(str, atom_order_str))
                    
        pos = atom.get_positions()
        posz = pos[:, 2]
        posz_min = np.min(posz)
        # posz_mid = np.average(posz)
        posz_mid = posz_min + 5
                    
        tag_list = atom.get_tags()
                    
        if n == 0:
            # print('yes')
            L.command("read_data slab.data")
            L.command("pair_style quip")
            """)
            f.write(f"L.command(\"pair_coeff * * {model_path} '{model_label}' {{}}\".format(atom_order_str))\n")
            f.write("""
        else:
            L.command("reset_timestep 0")
            L.command("read_data slab.data add append")
        """)
            f.write("L.command(\"region slab block EDGE EDGE EDGE EDGE 0 {}\".format(posz_mid))")
            f.write("""
        L.command("group fixed_slab region slab")
        L.command("fix freeze fixed_slab setforce 0.0 0.0 0.0")
        L.command("thermo_style custom step press cpu ke pe etotal temp")
        L.command("dump dump_minimization all custom 1 md.lammpstrj id type x y z vx vy vz fx fy fz q")
        L.command("thermo 1")
        L.command("min_style cg")
        L.command("minimize 0.0 1.0e-4 200 1000000")
        L.command("undump dump_minimization")
        L.command("unfix freeze")
        L.command("group fixed_slab delete")
        L.command("region slab delete")
        L.command("delete_atoms group all")
        
        os.system("sleep 1")
        
        images = read("md.lammpstrj", ":")
        traj = TrajectoryWriter("opt.traj", "a")
    
        file_name = glob.glob("../log.lammps")[0]
        e_pot = lammps_energy(file_name)
    
        print(len(e_pot))
    
        f_all = []
        for a in images:
            f = a.get_forces()
            f_all.append(f)
    
        for i, a in enumerate(images):
            an = a.get_atomic_numbers()
""")
    
            for ind, atom in enumerate(atom_order):
                f.write(
                    f"            an = [{atom_elem_to_num[atom]} if x == {ind+1} else x for x in an]\n"
                )
            f.write("""
            a.set_atomic_numbers(an)
            traj.write(a, energy=e_pot[i], forces=f_all[i])
    
        os.system("sleep 1")
        a = read("opt.traj@-1")
        e = a.get_potential_energy()
        f = a.get_forces()
        pos = a.get_positions()
        posz = pos[:, 2]
        # posz_mid = np.average(posz)
        
        ndx = np.where(posz < posz_mid)[0]
        c = FixAtoms(ndx)
        a.set_constraint(c)
        a.set_calculator(SPC(a, energy=e, forces=f))
        a.set_tags(tag_list)
                    
        final_atoms.append(a)
        # a.write("optimized.traj")

        os.chdir("..")

    ffinal_atoms = []
    for a in final_atoms:
        connected, n_components = examine_unconnected_components(a)
        if not examine_water_molecule_presents(a) and connected:
            ffinal_atoms.append(a)
        elif not examine_water_molecule_presents(a) and n_components <= 2:
            if molc:
                nat_cut = natural_cutoffs(a, mult=1.2)
                nl = NeighborList(nat_cut, skin=0, self_interaction=False, bothways=True)
                nl.update(a)
                matrix = nl.get_connectivity_matrix()
                _, component_list = sparse.csgraph.connected_components(matrix)
                _, counts = np.unique(component_list, return_counts=True)
                if np.min(counts) == len(molc):
                    ffinal_atoms.append(a)
            else:
                ffinal_atoms.append(a)

    final_atom = None
    for a in ffinal_atoms:
        if not final_atom:
            if n_components > 1:
                print(f"WARNING: The final structure has {n_components} components")
            final_atom = a
        elif a.get_potential_energy() < final_atom.get_potential_energy():
            if n_components > 1:
                print(f"WARNING: The final structure has {n_components} components")
            final_atom = a
        else:
            pass
    
    # if all samples have water molecule, save the last one temporary. This will not be accepted in GCBH run.
    if not final_atom:
        a.set_calculator(SPC(a, energy=0, forces=f))
        final_atom = a

    final_atom.write("optimized.traj")
                    
main()""")


    else: # Not multiple
        with open("opt.py", "w") as f:
            f.write("""import re
import os
import glob

from ase.io import *
from ase.io.trajectory import TrajectoryWriter
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.constraints import FixAtoms
from xyz2data import *
# from pymatgen.io.lammps.data import LammpsData
# from pymatgen.io.ase import AseAtomsAdaptor

re_energies = re.compile(\"\"\"^\s*Step \"\"\")

atom_elem_to_num = {"H": 1, "O": 8, "Zr": 40, "Cu": 29, "Pd": 46, "C": 6,}
atom_order = ["Zr", "O", "H", "Cu", "Pd", "C",]

                 
def lammps_energy(
                  logfile,
                  ):
    lf = open(logfile,'r')
    line = lf.readline()
    energies = []
    while line:
        if re_energies.match(line):
            w = line.split()
            n = np.where(np.array(w) == 'PotEng')[0][0]
            line = lf.readline()
            while line: 
                try:
                    w = line.split()
                    energies.append([float(a) for a in line.split()][n])
                except:
                    break
                line = lf.readline() 
        line = lf.readline()
    return energies
                           
def main():
    atoms = read("./input.traj")
    n = len(atoms)
    tag_list = atom.get_tags()
    xyz2data(
             atoms,
             vacuum_layer = 10,
             filename = 'slab.data',
             slab = True,
             qO = -0.82,
             )
                
    atom_order_str = []
    for s in atom_order:
        if all(atom.symbol != s for atom in atoms):
            atom_order.remove(s)
    for atom in atom_order:
        atom_order_str.append(atom_elem_to_num[atom])
    atom_order_str = ' '.join(map(str, atom_order_str))
    with open("in.opt", 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith('pair_coeff'):
            parts = line.split()
            kept_part = ' '.join(parts[:6])
            new_line = f"{kept_part} {atom_order_str} \\n"
            lines[i] = new_line
    with open("in.opt", 'w') as file:
        file.writelines(lines)
    
""")
            f.write('    os.system("srun {} -in in.opt > out")\n'.format(lammps_loc))
            f.write("""
    images = read("md.lammpstrj", ":")
    traj = TrajectoryWriter("opt.traj", "a")

    file_name = glob.glob("log.lammps")[0]
    e_pot = lammps_energy(file_name)

    print(len(e_pot))

    f_all = []
    for atoms in images:
        f = atoms.get_forces()
        f_all.append(f)

    for i, atoms in enumerate(images):
        an = atoms.get_atomic_numbers()

""")
            for ind, atom in enumerate(atom_order):
                f.write(
                    f"        an = [{atom_elem_to_num[atom]} if x == {ind+1} else x for x in an]\n"
                )
            f.write("""
        atoms.set_atomic_numbers(an)
        atoms.set_tags(tag_list)
        traj.write(atoms, energy=e_pot[i], forces=f_all[i])

    atoms = read("opt.traj@-1")
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    pos = atoms.get_positions()
    posz = pos[:, 2]
    pos_min = np.min(posz)
    posz_mid = np.average(posz)
    
    ndx = np.where(posz < posz_mid)[0]
    c = FixAtoms(ndx)
    atoms.set_constraint(c)
    atoms.set_calculator(SPC(atoms, energy=e, forces=f))
    atoms.write("optimized.traj")

main()
""")


def write_lammps_input_file(model_path, model_label, atom_order):
    """
    Write the lammps input file
    """
    with open("in.opt", "w") as f:
        f.write("""#input 
units             metal
dimension         3
processors        * * *
boundary          p p p

#real data
atom_style        charge
read_data         slab.data

#potential
pair_style        quip
""")
        atom_order_str = []
        # name = glob.glob("input.traj")
        slab = read("input.traj")
        atom_order_copy = atom_order.copy()
        for s in atom_order_copy:
            if all(atom.symbol != s for atom in slab):
                atom_order_copy.remove(s)
        for atom in atom_order_copy:
            atom_order_str.append(atom_elem_to_num[atom])
        atom_order_str = ' '.join(map(str, atom_order_str))
        f.write(f"pair_coeff * * {model_path} \"{model_label}\" {atom_order_str}")
        f.write("""
timestep 0.0001

region slab block EDGE EDGE EDGE EDGE 0 ZZ
group fixed_slab region slab
fix freeze fixed_slab setforce 0.0 0.0 0.0
dump 1 all custom 1 md.lammpstrj id type x y z vx vy vz fx fy fz
thermo 1
thermo_style custom step fmax press cpu ke pe etotal temp

#minimize
min_style cg
minimize 0.0 1.0e-4 200 1000000

""")


# def run_ase(options): 
#     """
#     from ase import units
#     from ase.md.langevin import Langevin
#     from ase.io import read, write
#     import numpy as np
#     import time
#     from mace.calculators import MACECalculator

#     calculator = MACECalculator(model_path='/content/checkpoints/MACE_model_run-123.model', device='cuda')
#     init_conf = read('BOTNet-datasets/dataset_3BPA/test_300K.xyz', '0')
#     init_conf.set_calculator(calculator)

#     dyn = Langevin(init_conf, 0.5*units.fs, temperature_K=310, friction=5e-3)
#     def write_frame():
#             dyn.atoms.write('md_3bpa.xyz', append=True)
#     dyn.attach(write_frame, interval=50)
#     dyn.run(100)
#     """

def write_optimize_sh(model_path, multiple=False):
    pwd = os.getcwd()
    with open("optimize.sh", "w") as f:
        f.write("pwd\n")
        # f.write("cp {} .\n".format(model_path))
        f.write(f"cp {pwd}/opt.py .\n")
        if multiple:
            f.write("srun python opt.py > out\n")
        else:
            f.write(f"cp {pwd}/in.opt .\n")
            f.write("python opt.py\n")


def run_bh(options, multiple=False):
    filescopied = ["opt.py"]
    name = glob.glob("input.traj")
    slab_clean = read(name[0])

    bh_run = GrandCanonicalBasinHopping(
        temperature=options["temperature"],
        t_nve=options["t_nve"],
        atoms=slab_clean,
        bash_script="optimize.sh",
        files_to_copied=filescopied,
        restart=True,
        chemical_potential="chemical_potentials.dat",
        stop_steps=20,
        # model_file=options["model_file"],
        # model_label=options["model_label"],
        # elements=atom_elem_to_num,
    )

    # cell = slab_clean.get_cell()
    # a = cell[0, 0]
    # b = cell[1, 0]
    # c = cell[1, 1]
    # tol = 1.5
    # boundary = np.array(
    #     [[-tol, -tol], [a + tol, -tol], [a + b + tol, c + tol], [b - tol, c + tol]]
    # )

    # bond_range = {}
    # for v in itertools.product(["Zr", "O", "H"], repeat=2):
    #     bond_range[frozenset(v)] = [1]

    # bh_run.add_modifier(
    #     randomize_all,
    #     name="randomize",
    #     dr=0.5,
    #     bond_range=bond_range,
    #     max_trial=50,
    #     weight=1,
    # )
    # bh_run.add_modifier(nve_n2p2, name="nve",bond_range=bond_range,  z_fix=6, N=100)
    # bh_run.add_modifier(mirror_mutate, name="mirror", weight=2)
    # bh_run.add_modifier(add_multiple_H, name="add_multiple_H", bond_range=bond_range, max_trial=100, weight=1.5)

    ## Hydroxylate surface
    # bh_run.add_modifier(add_H, name="add_H", bond_range=bond_range, max_trial=50, weight=1.5)
    # bh_run.add_modifier(add_O, name="add_O", bond_range=bond_range, max_trial=50, weight=1.5)
    # bh_run.add_modifier(add_OH, name="add_OH", bond_range=bond_range, max_trial=50, weight=1.5)
    # bh_run.add_modifier(remove_H, name="remove_H", weight=0.5)
    # bh_run.add_modifier(remove_O, name="remove_O", weight=0.5)

    ## Cluster configuration
    # bh_run.add_modifier(cluster_random_perturbation, name="cluster_random_perturbation", elements=['Cu', 'Pd'], max_trial=500, weight=1.5)
    # bh_run.add_modifier(cluster_random_displacement, name="cluster_random_displacement", elements=['Cu', 'Pd'], max_trial=500, weight=1.0)

    ## Cluster/substrate hydroxylate
    # bh_run.add_modifier(add_H, name="add_H", bond_range=bond_range, max_trial=50, weight=1.5)
    # bh_run.add_modifier(add_O, name="add_O", bond_range=bond_range, max_trial=50, weight=1.5)
    # bh_run.add_modifier(add_OH, name="add_OH", bond_range=bond_range, max_trial=50, weight=1.5)
    # bh_run.add_modifier(add_H_cluster, name="add_H_cluster", bond_range=bond_range, max_trial=50, weight=1.5)
    # bh_run.add_modifier(add_O_cluster, name="add_O_cluster", bond_range=bond_range, max_trial=50, weight=1.5)
    # bh_run.add_modifier(add_OH_cluster, name="add_OH_cluster", bond_range=bond_range, max_trial=50, weight=1.5)
    # bh_run.add_modifier(cluster_random_displacement, name="cluster_random_displacement", elements=['Cu', 'Pd'], tags=[1], max_trial=500, weight=1.0)
    # bh_run.add_modifier(remove_H, name="remove_H", weight=0.5)
    # bh_run.add_modifier(remove_O, name="remove_O", weight=0.5)

    bh_run.add_modifier(molc_random_displacement, name="molc_random_displacement", molc=options['molc'], bond_range=bond_range, elements=['C'], max_trial=50, weight=1.0)

    n_steps = 4000

    bh_run.run(n_steps, multiple=multiple, n_multiple=10)


def main(multiple=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--options", type=str, default="./bh_options.json")
    parser.add_argument("--cluster", action='store_true')
    parser.add_argument("--n_Cu", type=int, default=5)
    parser.add_argument("--n_Pd", type=int, default=0)
    parser.add_argument("--molc", action='store_true')
    parser.add_argument("--molc_type", type=str, default="Cyclohexene")
    args = parser.parse_args()

    with open(args.options) as f:
        options = json.load(f)
    model_file = options["model_file"]
    model_label = options["model_label"]
    atom_order = options["atom_order"]
    lammps_loc = options["lammps_loc"]

    name = glob.glob("input.traj")
    slab_clean = read(name[0])

    # Map atom species
    if any(atom.symbol == 'He' for atom in slab_clean):
        slab_clean.set_atomic_numbers([elements[n] for n in slab_clean.get_atomic_numbers()])
        slab_clean.set_pbc((True,True,True))

    # align slab to the bottom of the cell
    pos = slab_clean.get_positions()
    posz = pos[:, 2]
    posz_min = np.min(posz)
    dz = - posz_min + 1
    pos[:, 2] += dz
    slab_clean.set_positions(pos)

    # Check if there is unconnected species around the slab
    connected, n_components = examine_unconnected_components(slab_clean)
    if not connected:
        nat_cut = natural_cutoffs(slab_clean, mult=1.2)
        nl = NeighborList(nat_cut, skin=0, self_interaction=False, bothways=True)
        nl.update(slab_clean)
        matrix = nl.get_connectivity_matrix()
        n_components, component_list = sparse.csgraph.connected_components(matrix)
        unique, counts = np.unique(component_list, return_counts=True)
        disconnected_atom = []
        for n, c in enumerate(component_list):
            if c != unique[np.argmax(counts)]:
                disconnected_atom.append(n)
        del slab_clean[disconnected_atom]

    # If no Cu-Pd cluster on support, add a pentamer or tetramer
    if args.cluster:
        symbols = slab_clean.get_chemical_symbols()
        at = np.unique(symbols)
        if 'Cu' not in at and 'Pd' not in at:
            slab_clean = add_cluster(slab_clean, element={'Cu': args.n_Cu, 'Pd': args.n_Pd}, bond_range=bond_range, max_trial=500)
            write("input.traj", slab_clean)
        pos = slab_clean.get_positions()
        posz = pos[:, 2] # gets z positions of atoms in surface
        posz_mid = np.average(posz)

    if args.molc:
        symbols = slab_clean.get_chemical_symbols()
        at = np.unique(symbols)
        if 'C' not in at:
            write_molc(args.molc_type)
            molc = read("{}.in".format(args.molc_type))
            options['molc'] = molc
            slab_clean = add_molc_on_cluster(slab_clean, molc=molc, bond_range=bond_range, max_trial=500)
            write("input.traj", slab_clean)
        else:
            molc = read("{}.in".format(args.molc_type))
            options['molc'] = molc

    if multiple:
        if args.molc:
            write_opt_file(atom_order=atom_order, lammps_loc=lammps_loc, model_path=model_file, model_label=model_label, multiple=True, molc=options['molc'])
        else:
            write_opt_file(atom_order=atom_order, lammps_loc=lammps_loc, model_path=model_file, model_label=model_label, multiple=True)
        write_optimize_sh(model_path=model_file, multiple=multiple)
        run_bh(options, multiple=True)
    else:
        write_opt_file(atom_order=atom_order, lammps_loc=lammps_loc)
        write_lammps_input_file(model_path=model_file, model_label=model_label, atom_order=atom_order)
        with open("in.opt", 'r') as f:
            content = f.read()
        new_content = content.replace('ZZ', "{}".format(posz_mid))
        with open("in.opt", 'w') as f:
            f.write(new_content)
        write_optimize_sh(model_path=model_file, multiple=multiple)
        run_bh(options)


main(multiple=True)
