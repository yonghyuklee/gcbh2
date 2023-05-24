import re
import os
import glob

from ase.io import *
from ase.io.trajectory import TrajectoryWriter
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.constraints import FixAtoms
from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.ase import AseAtomsAdaptor


def main():
    atoms = read("./input.traj")
    n = len(atoms)
    ase_adap = AseAtomsAdaptor()
    atoms_ = ase_adap.get_structure(atoms)
    ld = LammpsData.from_structure(atoms_, atom_style="atomic")
    ld.write_file("struc.data")

    lammps_loc = "/ocean/projects/che160019p/santi92/lammps/build/lmp"
    os.system("{lammps_loc} < in.opt")

    images = read("md.lammpstrj", ":")
    traj = TrajectoryWriter("opt.traj", "a")

    file_name = glob.glob("log.lammps")[0]
    f = open(file_name, "r")
    Lines = f.readlines()
    patten = r"(\d+\s+\-+\d*\.?\d+)"
    e_pot = []
    for i, line in enumerate(Lines):
        s = line.strip()
        match = re.match(patten, s)
        if match != None:
            D = np.fromstring(s, sep=" ")
            e_pot.append(D[1])

    print(len(e_pot))

    f_all = []
    for atoms in images:
        f = atoms.get_forces()
        f_all.append(f)

    for i, atoms in enumerate(images):
        an = atoms.get_atomic_numbers()
        an = [78 if x == 1 else x for x in an]
        an = [1 if x == 2 else x for x in an]
        an = [13 if x == 3 else x for x in an]
        an = [6 if x == 4 else x for x in an]
        an = [8 if x == 5 else x for x in an]

        atoms.set_atomic_numbers(an)
        traj.write(atoms, energy=e_pot[i], forces=f_all[i])

    atoms = read("opt.traj@-1")
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    pos = atoms.get_positions()
    posz = pos[:, 2]
    ndx = np.where(posz < 5.5)[0]

    c = FixAtoms(ndx)
    atoms.set_constraint(c)
    atoms.set_calculator(SPC(atoms, energy=e, forces=f))
    atoms.write("optimized.traj")


main()
