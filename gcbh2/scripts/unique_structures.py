from ase.io import *
from ase.io.trajectory import TrajectoryWriter


def main():
    traj = read("local_minima.db", ":")
    trajw = TrajectoryWriter("unq.traj", "a")
    collect = []

    for atoms in traj:
        if atoms not in collect:
            collect.append(atoms)
            trajw.write(atoms)


main()
