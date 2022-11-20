import os.path

from fabric import Connection
import os
HOSTNAME = "ie-gpu"
host = Connection(HOSTNAME)

WORK_DIR = "workdir"
REPOSITORY_NAME = "competitions_108"
GIT_LINK = "https://github.com/Issei0804-ie/competitions_108.git"
BRUNCH_NAME = "main"
RSYNC_FILES = ["train", "train_master.tsv"]
IMAGE_SOURCE = os.path.join("~", WORK_DIR, "torch.sif")

host.run(f"mkdir -p {os.path.join(WORK_DIR, REPOSITORY_NAME)}")
with host.cd(os.path.join(WORK_DIR, REPOSITORY_NAME)):
    result = host.run("ls")
    dirs = result.stdout.split("\n")
    print(dirs)
    if not BRUNCH_NAME in dirs:
        host.run(f"git clone {GIT_LINK} -b {BRUNCH_NAME} {BRUNCH_NAME}")
        host.run(f"cp {IMAGE_SOURCE} {BRUNCH_NAME}")
    with host.cd(BRUNCH_NAME):
        result = host.run(f"git pull")
        print(result)
        for file in RSYNC_FILES:
            os.system(f"rsync -avhz {file} {HOSTNAME}:{os.path.join('~', WORK_DIR, REPOSITORY_NAME, BRUNCH_NAME)}")
        host.run(f"make slurm-run")




