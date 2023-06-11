import os


def main():
    # go through subfolders of depth 0 and if the file job_gpu.sh exists, run it
    for folder in os.listdir("."):
        if os.path.isdir(folder):
            if "job_gpu.sh" in os.listdir("./{}".format(folder)):
                os.chdir("./{}".format(folder))
                os.system("sbatch job_gpu.sh")
                os.chdir("../")


main()
