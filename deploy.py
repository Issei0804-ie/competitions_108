import argparse
from fabric import Connection
import os
import conf
HOSTNAME = "ie-gpu"
host = Connection(HOSTNAME)

p = argparse.ArgumentParser(description="deploy to ie-gpu")
p.add_argument("branch_name")
args = p.parse_args()
BRANCH_NAME = args.branch_name
WORK_DIR = "workdir"
REPOSITORY_NAME = "competitions_108"
GIT_LINK = "https://github.com/Issei0804-ie/competitions_108.git"
RSYNC_FILES = conf.RSYNC_FILES
IMAGE_SOURCE = os.path.join("~", WORK_DIR, "torch.sif")

host.run(f"mkdir -p {os.path.join(WORK_DIR, REPOSITORY_NAME)}")
with host.cd(os.path.join(WORK_DIR, REPOSITORY_NAME)):
    result = host.run("ls")
    dirs = result.stdout.split("\n")
    print(dirs)
    if not BRANCH_NAME in dirs:
        host.run(f"git clone {GIT_LINK} -b {BRANCH_NAME} {BRANCH_NAME}")
        host.run(f"cp {IMAGE_SOURCE} {BRANCH_NAME}")
    with host.cd(BRANCH_NAME):
        result = host.run(f"git pull")
        print(result)
        for file in RSYNC_FILES:
            os.system(f"rsync -avhz {file} {HOSTNAME}:{os.path.join('~', WORK_DIR, REPOSITORY_NAME, BRANCH_NAME)}")
        host.run(f"make slurm-run")




