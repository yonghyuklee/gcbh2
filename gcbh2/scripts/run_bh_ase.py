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
from pygcga2 import randomize_all, remove_H, remove_O, add_multiple_H, add_H, add_O#, rand_clustering, mirror_mutate

atom_elem_to_num = {"H": 1, "O": 8, "Zr": 40}
elements = {
            1 : 40,
            2 : 8,
            3 : 1,
            }


def write_opt_file(atom_order, lammps_loc):
    # opt.py file

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

atom_elem_to_num = {"H": 1, "O": 8, "Zr": 40}
atom_order = ["Zr", "O", "H"]
                 
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
        traj.write(atoms, energy=e_pot[i], forces=f_all[i])

    atoms = read("opt.traj@-1")
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    pos = atoms.get_positions()
    posz = pos[:, 2]
    pos_min = np.min(posz)
    posz_mid = np.min(posz) + 5
    
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


def examine_unconnected_components(atoms):
    nat_cut = natural_cutoffs(atoms, mult=1.0)
    nl = NeighborList(nat_cut, self_interaction=False, bothways=True)
    nl.update(atoms)
    matrix = nl.get_connectivity_matrix()
    n_components, component_list = sparse.csgraph.connected_components(matrix)
    if n_components == 1:
        return True
    elif n_components > 1:
        return False
    

def write_optimize_sh(model_path):
    pwd = os.getcwd()
    with open("optimize.sh", "w") as f:
        f.write("pwd\n")
        # f.write("cp {} .\n".format(model_path))
        f.write(f"cp {pwd}/in.opt .\n")
        f.write(f"cp {pwd}/opt.py .\n")
        f.write("python opt.py\n")


def run_bh(options):
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
    )

    bond_range = {
        ("Zr", "Zr"): [1.0, 10],
        ("Zr", "O"): [1.0, 10],
        ("Zr", "H"): [1.0, 10],
        ("O", "O"): [0.8, 10],
        ("O", "H"): [0.8, 10],
        ("H", "H"): [0.5, 10],
    }

    cell = slab_clean.get_cell()
    a = cell[0, 0]
    b = cell[1, 0]
    c = cell[1, 1]
    tol = 1.5
    boundary = np.array(
        [[-tol, -tol], [a + tol, -tol], [a + b + tol, c + tol], [b - tol, c + tol]]
    )

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
    bh_run.add_modifier(add_H, name="add_H", bond_range=bond_range, max_trial=50, weight=1.5)
    bh_run.add_modifier(add_O, name="add_O", bond_range=bond_range, max_trial=50, weight=1.5)
    bh_run.add_modifier(remove_H, name="remove_H", weight=0.5)
    bh_run.add_modifier(remove_O, name="remove_O", weight=0.5)

    n_steps = 4000

    bh_run.run(n_steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--options", type=str, default="./bh_options.json")
    args = parser.parse_args()
    with open(args.options) as f:
        options = json.load(f)
    model_file = options["model_file"]
    model_label = options["model_label"]
    atom_order = options["atom_order"]
    lammps_loc = options["lammps_loc"]

    name = glob.glob("input.traj")
    slab_clean = read(name[0])
    if any(atom.symbol == 'He' for atom in slab_clean):
        slab_clean.set_atomic_numbers([elements[n] for n in slab_clean.get_atomic_numbers()])
        slab_clean.set_pbc((True,True,True))
    if not examine_unconnected_components(slab_clean):
        nat_cut = natural_cutoffs(slab_clean, mult=1.0)
        nl = NeighborList(nat_cut, self_interaction=False, bothways=True)
        nl.update(slab_clean)
        matrix = nl.get_connectivity_matrix()
        n_components, component_list = sparse.csgraph.connected_components(matrix)
        unique, counts = np.unique(component_list, return_counts=True)
        disconnected_atom = []
        for n, c in enumerate(component_list):
            if c != unique[np.argmax(counts)]:
                disconnected_atom.append(n)
        del slab_clean[disconnected_atom]
    write("input.traj", slab_clean)
    pos = slab_clean.get_positions()
    posz = pos[:, 2] # gets z positions of atoms in surface
    posz_mid = np.min(posz) + 5 # np.average(posz)

    write_opt_file(atom_order=atom_order, lammps_loc=lammps_loc)
    write_lammps_input_file(model_path=model_file, model_label=model_label, atom_order=atom_order)
    with open("in.opt", 'r') as f:
        content = f.read()
    new_content = content.replace('ZZ', "{}".format(posz_mid))
    with open("in.opt", 'w') as f:
        f.write(new_content)
    write_optimize_sh(model_path=model_file)
    run_bh(options)


main()
