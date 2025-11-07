#!/bin/bash

#SBATCH --job-name=lgpn_job          # Nome do strabalho
#SBATCH --output=lgpn_saida_%j.log   # Arquivo para onde a saída padrão vai
#SBATCH --error=lgpn_erro_%j.log     # Arquivo para onde os erros vão
#SBATCH --time=08:00:00              # Tempo máximo de execução (8 horas)
#SBATCH --partition=h100n3           # Partição 
#SBATCH --gres=gpu:h100:1            # Pede UMA GPU

# module load cuda12.6/toolkit/12.6.2
source /home/user_beatrizalmeida/workspace/lgpn_venv/bin/activate

echo "=========================================================="
echo "Data de início: $(date)"
echo "Nó de execução: $(hostname)"
echo "GPUs alocadas: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

cd /home/user_beatrizalmeida/lgpn/scripts/

bash run_br.sh
