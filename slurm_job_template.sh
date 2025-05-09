#!/bin/bash
#SBATCH -A IscrC_SAIL
#SBATCH -p boost_usr_prod
#SBATCH --qos boost_qos_lprod
#SBATCH --time 0-03:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=123000

source venv/bin/activate

echo "Running job for config: $CONFIG in mode: $MODE"
bash run.sh "$MODE" "$CONFIG"
