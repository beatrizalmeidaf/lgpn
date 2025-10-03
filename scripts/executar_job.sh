#!/bin/bash

#SBATCH --job-name=laqda_job         # Nome do strabalho
#SBATCH --output=laqda_saida_%j.log  # Arquivo para onde a saída padrão vai
#SBATCH --error=laqda_erro_%j.log    # Arquivo para onde os erros vão
#SBATCH --time=08:00:00              # Tempo máximo de execução (8 horas)
#SBATCH --partition=h100n3           # Partição OBRIGATÓRIA que foi informada
#SBATCH --gres=gpu:1                 # Pede UMA GPU


echo "=========================================================="
echo "Data de início: $(date)"
echo "Nó de execução: $(hostname)"
echo "GPUs alocadas: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

cd /home/user_beatrizalmeida/lgpn/scripts/

sh ours_all.sh
