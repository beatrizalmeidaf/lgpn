#!/bin/bash

#SBATCH --job-name=laqda_job         # Nome do seu trabalho
#SBATCH --output=laqda_saida_%j.log  # Arquivo para onde a saída padrão vai
#SBATCH --error=laqda_erro_%j.log   # Arquivo para onde os erros vão
#SBATCH --time=08:00:00              # Tempo máximo de execução (8 horas, ajuste se precisar de mais)
#SBATCH --partition=h100n3           # Partição OBRIGATÓRIA que foi informada
#SBATCH --gres=gpu:1                 # Pede UMA GPU

# Imprime informações sobre o ambiente do job para depuração
echo "=========================================================="
echo "Data de início: $(date)"
echo "Nó de execução: $(hostname)"
echo "GPUs alocadas: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

# Navega para o diretório onde está o script
cd /home/user_beatrizalmeida/lgpn/scripts/

# Executa o seu script principal que chama o main.py várias vezes
sh ours_all.sh
