import argparse
import os


def write_sh_gpu(folder, name="job_gpu.sh"):
    file = open("{}/{}".format(folder, name), "w")
    file.write("#!/bin/bash\n")
    file.write("#SBATCH --partition=GPU-shared\n")
    file.write("#SBATCH --gres=gpu:v100-32:1\n")
    file.write("#SBATCH --mem=20G\n")
    file.write("#SBATCH -t 03:00:00\n")
    file.write("#SBATCH --mail-user=santiagovargas921@gmail\n")
    file.write("#SBATCH --mail-type=ALL\n")
    file.write("\n")
    file.write("# bridges2 \n")
    file.write("\n")
    file.write("module load intel/2021.3.0\n")
    file.write("module load intelmpi\n")
    file.write("module load AI/anaconda3-tf2.2020.11\n")
    file.write("conda activate deepmd\n")
    file.write("export OMP_NUM_THREADS=16\n")
    file.write("python run_bh.py\n")
    file.close()


def main():
    # argparse to options json
    parser = argparse.ArgumentParser()
    # add arguement for number of folders
    parser.add_argument("--num_folders", type=int, default=5)

    args = parser.parse_args()
    num_folders = args.num_folders
    print("number of folders: {}".format(num_folders))

    # find chemical_potential.json in current directory
    chemical_potential_file = [
        f for f in os.listdir(".") if f.endswith("chemical_potentials.dat")
    ][0]
    print("chemical potential file: {}".format(chemical_potential_file))

    # find run_bh.py in current directory
    run_bh_file = [f for f in os.listdir(".") if f.endswith("run_bh.py")][0]
    print("run_bh file: {}".format(run_bh_file))

    # find *.traj
    traj_file = [f for f in os.listdir(".") if f.endswith(".traj")][0]
    print("traj file: {}".format(traj_file))

    # find  bh_options.json in current directory
    bh_options_file = [f for f in os.listdir(".") if f.endswith("bh_options.json")][0]
    print("bh_options file: {}".format(bh_options_file))

    # create a new folder for each run
    for i in range(num_folders):
        os.mkdir("run_{}".format(i))
        os.system("cp {} run_{}/run_bh.py".format(run_bh_file, i))
        os.system("cp {} run_{}/bh_options.json".format(bh_options_file, i))
        os.system("cp {} run_{}/{}".format(traj_file, i, traj_file))
        os.system(
            "cp {} run_{}/{}".format(
                chemical_potential_file, i, chemical_potential_file
            )
        )
        write_sh_gpu("run_{}".format(i), name="job_gpu.sh")


main()
