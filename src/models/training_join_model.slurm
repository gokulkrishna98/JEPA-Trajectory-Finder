#SBATCH --job-name=job_wgpu
#SBATCH --account=csci_ga_2572-2024fa
#SBATCH --partition=n1s8-v100-1
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --requeue

singularity exec --bind /scratch --nv --overlay /scratch/gm3314/environments/overlay-25GB-500K.ext3:rw  /scratch/gm3314/environments/ubuntu-20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate my_env
cd /scratch/gm3314/JEPA-Trajectory-Finder/src/models/
python3 ./joint_model_train.py
"