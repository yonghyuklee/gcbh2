import glob, os, sys, json, argparse
import itertools
import numpy as np
from ase.io import read
from gcbh2 import GrandCanonicalBasinHopping
from pygcga2 import randomize_all, mirror_mutate, remove_H, add_H, rand_clustering


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--options", type=str, default="./viz_options.json")
    args = parser.parse_args()
    with open(args.options) as f:
        options = json.load(f)

    filescopied = ["opt.py"]  # files required to complete an optimization
    name = glob.glob("input.traj")
    print(name)
    slab_clean = read(name[0])

    bh_run = GrandCanonicalBasinHopping(
        temperature=2000.0,
        t_nve=1000,
        atoms=slab_clean,
        bash_script="optimize.sh",
        files_to_copied=filescopied,
        restart=True,
        chemical_potential="chemical_potentials.dat",
    )
    bond_range = {
        ("C", "Pt"): [1.2, 10],
        ("Pt", "Pt"): [1, 10.0],
        ("C", "C"): [1.9, 10],
        ("C", "O"): [0.6, 10],
        ("Pt", "O"): [1.5, 10],
        ("O", "O"): [1.9, 10],
        ("Al", "Al"): [1, 10],
    }

    cell = slab_clean.get_cell()
    a = cell[0, 0]
    b = cell[1, 0]
    c = cell[1, 1]
    tol = 1.5
    boundary = np.array(
        [[-tol, -tol], [a + tol, -tol], [a + b + tol, c + tol], [b - tol, c + tol]]
    )

    bond_range = {}
    for v in itertools.product(["Al", "O", "Pt", "C", "H"], repeat=2):
        bond_range[frozenset(v)] = [1]

    bh_run.add_modifier(
        randomize_all,
        name="randomize",
        dr=1,
        bond_range=bond_range,
        max_trial=50,
        weight=1,
    )
    # bh_run.add_modifier(nve_n2p2, name="nve",bond_range=bond_range,  z_fix=6, N=100)
    bh_run.add_modifier(mirror_mutate, name="mirror", weight=2)
    # bh_run.add_modifier(add_molecule_on_cluster, name="add_H", weight=3)
    bh_run.add_modifier(remove_H, name="remove_H", weight=0.5)
    bh_run.add_modifier(add_H, bond_range=bond_range, max_trial=50, weight=2)

    bh_run.run(4000)


main()
