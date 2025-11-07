#!/bin/bash
set -e 

RAID_BASE_PATH="/raid/user_beatrizalmeida/lgpn_results_br_binary"
echo "Salvando resultados em: $RAID_BASE_PATH"

cuda=0
lr=1e-6
seed=2023
temp=5
query=25

way=2 
TEMPLATE='Esta é uma avaliação [MASK]: [sentence]' 

DATASET_BASE_PATH="../reviews_remapped_sentiment_binary"
DATASETS_TO_RUN=("B2WCorpus" "BrandsCorpus" "ReProCorpus" "BuscapeCorpus" "KaggleTweetsCorpus" "OlistCorpus" "UTLCorpus")
FOLDS=(01 02 03 04 05)
SHOTS=(1 5)


for dataset_name in "${DATASETS_TO_RUN[@]}"; do
    echo "============================================================"
    echo "PROCESSANDO DATASET (LGPN): $dataset_name"
    echo "============================================================"

    for fold in "${FOLDS[@]}"; do
        echo "Iniciando processamento para o fold: $fold"

        SOURCE_DATA_DIR="${DATASET_BASE_PATH}/${dataset_name}/few_shot/${fold}"
        
        CURRENT_OUTPUT_DIR="${RAID_BASE_PATH}/${dataset_name}/${fold}"
        mkdir -p "$CURRENT_OUTPUT_DIR"
        for shot in "${SHOTS[@]}"; do

            RESULT_FILE="${CURRENT_OUTPUT_DIR}/result/${way}-way_${shot}-shot_${dataset_name}_result.csv"

            if [ -f "$RESULT_FILE" ]; then
                echo "Arquivo de resultado encontrado. Pulando: ${dataset_name} (Fold ${fold}) ${way}-way ${shot}-shot"
            else
                echo "Executando: ${dataset_name} (Fold ${fold}) ${way}-way ${shot}-shot"

                python ../src/main.py \
                    --output_dir "$CURRENT_OUTPUT_DIR" \
                    --cuda $cuda \
                    --way $way \
                    --shot $shot \
                    --query $query \
                    --mode train \
                    --classifier mbc \
                    --dataset="$dataset_name" \
                    --data_path="$SOURCE_DATA_DIR" \
                    --pool prompt \
                    --template "$TEMPLATE" \
                    --add_prol \
                    --add_prosq \
                    --protype single \
                    --seed=$seed \
                    --T $temp \
                    --SG mean \
                    --lr=$lr

            fi
        done 
    done 
done 

echo "Processamento (LGPN) de todos os datasets brasileiros concluído."