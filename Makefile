build-sif:
	singularity build --fakeroot output/torch.sif torch.def

run:
	singularity run --nv torch.sif python3 main.py vgg

pip:
	singularity run output/torch.sif pip install -r requirements.txt

tensorboard:
	singularity exec output/torch.sif tensorboard --bind_all --logdir ./

slurm-run:
	sbatch run.sbatch

local-run:
	python main.py
