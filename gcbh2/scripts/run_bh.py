##########################################################################################
# this is an example script to run the Grand Canonical Basin Hopping
# this self-contained script generates the executables to run basin hopping
# you still need to add the chemical potentials, input.traj, and the bh_options.json
##########################################################################################
# generalizations to the code such as general lammmps input file, etc. to come or whatever

import glob, os, sys, json, argparse
import itertools
import numpy as np
from ase.io import read
from gcbh2.scripts.gcbh2 import GrandCanonicalBasinHopping
from pygcga2 import randomize_all, mirror_mutate, remove_H, add_H#, add_h_gas

atom_elem_to_num = {"H": 1, "C": 6, "O": 8, "Al": 13, "Pt": 78}


def write_opt_file(atom_order, lammps_loc):
    # opt.py file

    with open("opt.py", "w") as f:
        f.write("import re\n")
        f.write("import os\n")
        f.write("import glob\n")
        f.write("\n")
        f.write("from ase.io import *\n")
        f.write("from ase.io.trajectory import TrajectoryWriter\n")
        f.write("import numpy as np\n")
        f.write(
            "from ase.calculators.singlepoint import SinglePointCalculator as SPC\n"
        )
        f.write("from ase.constraints import FixAtoms\n")
        f.write("from pymatgen.io.lammps.data import LammpsData\n")
        f.write("from pymatgen.io.ase import AseAtomsAdaptor\n")
        f.write("def main():\n")
        f.write('    atoms = read("./input.traj")\n')
        f.write("    n = len(atoms)\n")
        f.write("    ase_adap = AseAtomsAdaptor()\n")
        f.write("    atoms_ = ase_adap.get_structure(atoms)\n")
        f.write('    ld = LammpsData.from_structure(atoms_, atom_style="atomic")\n')
        f.write('    ld.write_file("struc.data")\n')
        f.write("\n")
        f.write('    os.system("{} < in.opt")\n'.format(lammps_loc))
        f.write("\n")
        f.write('    images = read("md.lammpstrj", ":")\n')
        f.write('    traj = TrajectoryWriter("opt.traj", "a")\n')
        f.write("\n")
        f.write('    file_name = glob.glob("log.lammps")[0]\n')
        f.write("    f = open(file_name, 'r')\n")
        f.write("    Lines = f.readlines()\n")
        f.write('    patten = r"(\\d+\\s+\\-+\\d*\\.?\\d+)"\n')
        f.write("    e_pot = []\n")
        f.write("    for i, line in enumerate(Lines):\n")
        f.write("        s = line.strip()\n")
        f.write("        match = re.match(patten, s)\n")
        f.write("        if match != None:\n")
        f.write("            D = np.fromstring(s, sep=' ')\n")
        f.write("            e_pot.append(D[1])\n")
        f.write("\n")
        f.write("    print(len(e_pot))\n")
        f.write("\n")
        f.write("    f_all = []\n")
        f.write("    for atoms in images:\n")
        f.write("        f = atoms.get_forces()\n")
        f.write("        f_all.append(f)\n")
        f.write("\n")
        f.write("    for i, atoms in enumerate(images):\n")
        f.write("        an = atoms.get_atomic_numbers()\n")
        for ind, atom in enumerate(atom_order):
            f.write(
                f"        an = [{atom_elem_to_num[atom]} if x == {ind+1} else x for x in an]\n"
            )
        f.write("\n")
        f.write("        atoms.set_atomic_numbers(an)\n")
        f.write("        traj.write(atoms, energy=e_pot[i], forces=f_all[i])\n")
        f.write("\n")
        f.write('    atoms = read("opt.traj@-1")\n')
        f.write("    e = atoms.get_potential_energy()\n")
        f.write("    f = atoms.get_forces()\n")
        f.write("    pos = atoms.get_positions()\n")
        f.write("    posz = pos[:, 2]\n")
        f.write("    ndx = np.where(posz < 5.5)[0]\n")
        f.write("    c = FixAtoms(ndx)\n")
        f.write("    atoms.set_constraint(c)\n")
        f.write("    atoms.set_calculator(SPC(atoms, energy=e, forces=f))\n")
        f.write('    atoms.write("optimized.traj")\n')
        f.write("main()\n")


def write_lammps_input_file(model_path, atom_order):
    """
    Write the lammps input file
    """
    with open("in.opt", "w") as f:
        f.write("#input \n")
        f.write("units                  metal\n")
        f.write("dimension	       3\n")
        f.write("processors	       * * *\n")
        f.write("box tilt 	       large\n")
        f.write("boundary 	       p p f\n")
        f.write("\n")
        f.write("#real data\n")
        f.write("atom_style	       atomic\n")
        f.write("read_data	       struc.data\n")
        f.write("\n")
        f.write("#potential\n")
        f.write("pair_style	allegro\n")
        atom_order_str = " ".join(atom_order)
        f.write(f"pair_coeff	* * {model_path} {atom_order_str}\n")
        f.write("\n")
        f.write("timestep 0.0001\n")
        f.write("\n")
        f.write("region slab block EDGE EDGE EDGE EDGE 0 5.5\n")
        f.write("group fixed_slab region slab\n")
        f.write("fix freeze fixed_slab setforce 0.0 0.0 0.0\n")
        f.write("dump 1 all custom 1 md.lammpstrj id type x y z fx fy fz\n")
        f.write("thermo 1\n")
        f.write("thermo_style custom step pe fmax\n")
        f.write("\n")
        f.write("#minimize\n")
        f.write("min_modify norm max\n")
        f.write("minimize 0.0 0.3 200 100000\n")


def write_optimize_sh(model_path):
    with open("optimize.sh", "w") as f:
        f.write("pwd\n")
        # f.write("cp {} .\n".format(model_path))
        f.write("cp ../../in.opt .\n")
        f.write("cp ../../opt.py .\n")
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

    cell = slab_clean.get_cell()
    a = cell[0, 0]
    b = cell[1, 0]
    c = cell[1, 1]
    tol = 1.5
    boundary = np.array(
        [[-tol, -tol], [a + tol, -tol], [a + b + tol, c + tol], [b - tol, c + tol]]
    )

    # bond_range = {}
    # for v in itertools.product(["Al", "O", "Pt", "C", "H"], repeat=2):
    #    bond_range[frozenset(v)] = [1]
    bond_range = {
        frozenset(("C", "Pt")): [1.5, 10],
        frozenset(("Pt", "Pt")): [2.0, 10.0],
        frozenset(("C", "C")): [0.8, 10],
        frozenset(("C", "O")): [0.6, 10],
        frozenset(("Pt", "O")): [1.5, 10],
        frozenset(("O", "O")): [1.0, 2.0],
        frozenset(("Al", "Al")): [2.0, 3.5],
        frozenset(("H", "H")): [0.7, 1.5],
        frozenset(("H", "Pt")): [1.4, 2.5],
        frozenset(("H", "C")): [0.7, 1.4],
        frozenset(("Al", "O")): [1.5, 3.0],
        frozenset(("Al", "Pt")): [2.1, 3.5],
        frozenset(("Al", "H")): [1.1, 2.4],
        frozenset(("Al", "C")): [1.7, 2.5],
        frozenset(("O", "H")): [0.7, 1.5],
    }

    bh_run.add_modifier(
        randomize_all,
        name="randomize",
        dr=1,
        bond_range=bond_range,
        max_trial=50,
        weight=0.5,
    )
    # bh_run.add_modifier(nve_n2p2, name="nve",bond_range=bond_range,  z_fix=6, N=100)
    bh_run.add_modifier(mirror_mutate, name="mirror", weight=2)
    bh_run.add_modifier(add_H, bond_range=bond_range, max_trial=50, weight=0.5)
    # bh_run.add_modifier(add_h_gas, bond_range=bond_range, name="add_h2")
    bh_run.add_modifier(remove_H, name="remove_h", weight=0.5)
    n_steps = 4000

    bh_run.run(n_steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--options", type=str, default="./bh_options.json")
    args = parser.parse_args()
    with open(args.options) as f:
        options = json.load(f)
    model_file = options["model_file"]
    atom_order = options["atom_order"]
    lammps_loc = options["lammps_loc"]

    write_opt_file(atom_order=atom_order, lammps_loc=lammps_loc)
    write_lammps_input_file(model_path=model_file, atom_order=atom_order)
    write_optimize_sh(model_path=model_file)
    run_bh(options)


main()
