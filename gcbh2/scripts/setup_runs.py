import argparse
import os 

def main():
    # argparse to options json
    parser = argparse.ArgumentParser()
    # add arguement for number of folders
    parser.add_argument("--num_folders", type=int, default=5)

    args = parser.parse_args()
    num_folders = args.num_folders
    print("number of folders: {}".format(num_folders))


    # find chemical_potential.json in current directory
    chemical_potential_file = [f for f in os.listdir('.') if f.endswith('chemical_potential.json')][0]
    print("chemical potential file: {}".format(chemical_potential_file))

    # find run_bh.py in current directory
    run_bh_file = [f for f in os.listdir('.') if f.endswith('run_bh.py')][0]
    print("run_bh file: {}".format(run_bh_file))

    # find  bh_options.json in current directory
    bh_options_file = [f for f in os.listdir('.') if f.endswith('bh_options.json')][0]
    print("bh_options file: {}".format(bh_options_file))

    # create a new folder for each run
    for i in range(num_folders):
        os.mkdir("run_{}".format(i))
        os.system("cp {} run_{}/run_bh.py".format(run_bh_file, i))
        os.system("cp {} run_{}/bh_options.json".format(bh_options_file, i))
        os.system("cp {} run_{}/{}".format(chemical_potential_file, i, chemical_potential_file))
        
main()
