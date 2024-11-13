#!/bin/bash

# models=("iYL1228" "iRC1080" "iMM904" "iAF692")
models=("iSSON_1240" "iSFxv_1172" "iSF_1195" "iSFV_1184" "iS_1188" "iSBO_1134" "iJO1366" "STM_v1_0" "iSDY_1059" "iAF1260b" "iYL1228" "iRC1080" "iMM904" "iAF692")
splits=("0.2 80%" "0.4 60%" "0.6 40%" "0.8 20%")
# splits=("0.6 40%" "0.8 20%")

for model in "${models[@]}"
do
    for split_remove in "${splits[@]}"
    do
        train_split=$(echo $split_remove | awk '{print $1}')
        remove=$(echo $split_remove | awk '{print $2}')
        job_name="${model}_hgnn_recover_${train_split}_${remove}"
        echo "Submitting job: $job_name"

        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=/nas/longleaf/home/cyixiao/Project/Cheshire/workplace/slurm/%x/%j/out.txt
#SBATCH --error=/nas/longleaf/home/cyixiao/Project/Cheshire/workplace/slurm/%x/%j/error.txt
#SBATCH --partition=volta-gpu,l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00

cd /nas/longleaf/home/cyixiao
source test_env/bin/activate
cd Project/Cheshire

python main.py --name "$model" --repeat 5 --model "HGNN" --lr 0.001 --emb_dim 256 --max_epoch 100 --train_split $train_split --remove "$remove" --max_epoch 100 --recover

EOT
    done
done