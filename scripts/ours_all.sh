#!/bin/bash

RESULT_DIR="/raid/user_beatrizalmeida/lgpn_results"

mkdir -p $RESULT_DIR

#
# Experimentos para Huffpost
#
dataset=huffpost
data_path="../data/huffpost.json"
n_train_class=20
n_val_class=5
n_test_class=16
temp=5
lr=1e-6
seed=2023
cuda=0

# --- Huffpost 1-shot ---
way=5
shot=1
RESULT_FILE="${RESULT_DIR}/result/${way}-way_${shot}-shot_${dataset}_result.csv"
if [ -f "$RESULT_FILE" ]; then
    echo "Arquivo de resultado encontrado. Pulando: ${dataset} ${way}-way ${shot}-shot"
else
    echo "Executando: ${dataset} ${way}-way ${shot}-shot"
    python ../src/main.py \
        --output_dir /raid/user_beatrizalmeida/lgpn_results \
        --cuda $cuda --way $way --shot $shot --query 25 --mode train --classifier mbc \
        --dataset=$dataset --data_path=$data_path --n_train_class=$n_train_class \
        --n_val_class=$n_val_class --n_test_class=$n_test_class \
        --pool prompt --template 'This is a [MASK] news: [sentence]' \
        --add_prol --add_prosq --protype single --seed=$seed --T $temp --SG mean --lr=$lr
fi

# --- Huffpost 5-shot ---
way=5
shot=5
RESULT_FILE="${RESULT_DIR}/result/${way}-way_${shot}-shot_${dataset}_result.csv"
if [ -f "$RESULT_FILE" ]; then
    echo "Arquivo de resultado encontrado. Pulando: ${dataset} ${way}-way ${shot}-shot"
else
    echo "Executando: ${dataset} ${way}-way ${shot}-shot"
    python ../src/main.py \
        --output_dir /raid/user_beatrizalmeida/lgpn_results \
        --cuda $cuda --way $way --shot $shot --query 25 --mode train --classifier mbc \
        --dataset=$dataset --data_path=$data_path --n_train_class=$n_train_class \
        --n_val_class=$n_val_class --n_test_class=$n_test_class \
        --pool prompt --template 'This is a [MASK] news: [sentence]' \
        --add_prol --add_prosq --protype single --seed=$seed --T $temp --SG mean --lr=$lr
fi


#
# Experimentos para Reuters
#
dataset=reuters
data_path="../data/reuters.json"
n_train_class=15
n_val_class=5
n_test_class=11

# --- Reuters 1-shot ---
way=5
shot=1
RESULT_FILE="${RESULT_DIR}/result/${way}-way_${shot}-shot_${dataset}_result.csv"
if [ -f "$RESULT_FILE" ]; then
    echo "Arquivo de resultado encontrado. Pulando: ${dataset} ${way}-way ${shot}-shot"
else
    echo "Executando: ${dataset} ${way}-way ${shot}-shot"
    python ../src/main.py \
        --output_dir /raid/user_beatrizalmeida/lgpn_results \
        --cuda $cuda --way $way --shot $shot --query 15 --mode train --classifier mbc \
        --dataset=$dataset --data_path=$data_path --n_train_class=$n_train_class \
        --n_val_class=$n_val_class --n_test_class=$n_test_class \
        --pool prompt --template 'This is a [MASK] news: [sentence]' \
        --add_prol --add_prosq --protype single --patience 20 --T 1 --SG mean --seed=$seed --lr=$lr
fi

# --- Reuters 5-shot ---
way=5
shot=5
RESULT_FILE="${RESULT_DIR}/result/${way}-way_${shot}-shot_${dataset}_result.csv"
if [ -f "$RESULT_FILE" ]; then
    echo "Arquivo de resultado encontrado. Pulando: ${dataset} ${way}-way ${shot}-shot"
else
    echo "Executando: ${dataset} ${way}-way ${shot}-shot"
    python ../src/main.py \
        --output_dir /raid/user_beatrizalmeida/lgpn_results \
        --cuda $cuda --way $way --shot $shot --query 15 --mode train --classifier mbc \
        --dataset=$dataset --data_path=$data_path --n_train_class=$n_train_class \
        --n_val_class=$n_val_class --n_test_class=$n_test_class \
        --pool prompt --template 'This is a [MASK] news: [sentence]' \
        --add_prol --add_prosq --protype single --seed=$seed --T $temp --SG mean --lr=$lr
fi


#
# Experimentos para 20newsgroup
#
dataset=20newsgroup
data_path="../data/20news.json"
n_train_class=8
n_val_class=5
n_test_class=7

# --- 20newsgroup 1-shot ---
way=5
shot=1
RESULT_FILE="${RESULT_DIR}/result/${way}-way_${shot}-shot_${dataset}_result.csv"
if [ -f "$RESULT_FILE" ]; then
    echo "Arquivo de resultado encontrado. Pulando: ${dataset} ${way}-way ${shot}-shot"
else
    echo "Executando: ${dataset} ${way}-way ${shot}-shot"
    python ../src/main.py \
        --output_dir /raid/user_beatrizalmeida/lgpn_results \
        --cuda $cuda --way $way --shot $shot --query 25 --mode train --classifier mbc \
        --dataset=$dataset --data_path=$data_path --n_train_class=$n_train_class \
        --n_val_class=$n_val_class --n_test_class=$n_test_class \
        --pool prompt --template 'This is a [MASK] news: [sentence]' \
        --add_prol --add_prosq --protype single --seed=$seed --T $temp --SG mean --lr=$lr
fi

# --- 20newsgroup 5-shot ---
way=5
shot=5
RESULT_FILE="${RESULT_DIR}/result/${way}-way_${shot}-shot_${dataset}_result.csv"
if [ -f "$RESULT_FILE" ]; then
    echo "Arquivo de resultado encontrado. Pulando: ${dataset} ${way}-way ${shot}-shot"
else
    echo "Executando: ${dataset} ${way}-way ${shot}-shot"
    python ../src/main.py \
        --output_dir /raid/user_beatrizalmeida/lgpn_results \
        --cuda $cuda --way $way --shot $shot --query 25 --mode train --classifier mbc \
        --dataset=$dataset --data_path=$data_path --n_train_class=$n_train_class \
        --n_val_class=$n_val_class --n_test_class=$n_test_class \
        --pool prompt --template 'This is a [MASK] news: [sentence]' \
        --add_prol --add_prosq --protype single --seed=$seed --T 1 --SG mean --lr=$lr
fi


#
# Experimentos para Amazon
#
dataset=amazon
data_path="../data/amazon.json"
n_train_class=10
n_val_class=5
n_test_class=9

# --- Amazon 1-shot ---
way=5
shot=1
RESULT_FILE="${RESULT_DIR}/result/${way}-way_${shot}-shot_${dataset}_result.csv"
if [ -f "$RESULT_FILE" ]; then
    echo "Arquivo de resultado encontrado. Pulando: ${dataset} ${way}-way ${shot}-shot"
else
    echo "Executando: ${dataset} ${way}-way ${shot}-shot"
    python ../src/main.py \
        --output_dir /raid/user_beatrizalmeida/lgpn_results \
        --cuda $cuda --way $way --shot $shot --query 25 --mode train --classifier mbc \
        --dataset=$dataset --data_path=$data_path --n_train_class=$n_train_class \
        --n_val_class=$n_val_class --n_test_class=$n_test_class \
        --pool prompt --template 'This is a [MASK] review: [sentence]' \
        --add_prol --add_prosq --protype single --seed=$seed --T 0.5 --SG mean --lr=$lr
fi

# --- Amazon 5-shot ---
way=5
shot=5
RESULT_FILE="${RESULT_DIR}/result/${way}-way_${shot}-shot_${dataset}_result.csv"
if [ -f "$RESULT_FILE" ]; then
    echo "Arquivo de resultado encontrado. Pulando: ${dataset} ${way}-way ${shot}-shot"
else
    echo "Executando: ${dataset} ${way}-way ${shot}-shot"
    python ../src/main.py \
        --output_dir /raid/user_beatrizalmeida/lgpn_results \
        --cuda $cuda --way $way --shot $shot --query 25 --mode train --classifier mbc \
        --dataset=$dataset --data_path=$data_path --n_train_class=$n_train_class \
        --n_val_class=$n_val_class --n_test_class=$n_test_class \
        --pool prompt --template 'This is a [MASK] review: [sentence]' \
        --add_prol --add_prosq --protype single --seed=$seed --T 1 --SG mean --lr=$lr
fi


#
# Experimentos para clinc150
#
dataset=clinc150
data_path="../data/clinc150.json"
n_train_class=60
n_val_class=15
n_test_class=75
n_train_domain=4
n_val_domain=1
n_test_domain=5

# --- clinc150 10-way 1-shot ---
way=10
shot=1
RESULT_FILE="${RESULT_DIR}/result/${way}-way_${shot}-shot_${dataset}_result.csv"
if [ -f "$RESULT_FILE" ]; then
    echo "Arquivo de resultado encontrado. Pulando: ${dataset} ${way}-way ${shot}-shot"
else
    echo "Executando: ${dataset} ${way}-way ${shot}-shot"
    python ../src/main.py \
        --output_dir /raid/user_beatrizalmeida/lgpn_results \
        --cuda $cuda --way $way --shot $shot --query 25 --mode train --classifier mbc \
        --dataset=$dataset --data_path=$data_path --n_train_class=$n_train_class \
        --n_val_class=$n_val_class --n_test_class=$n_test_class --n_train_domain=$n_train_domain \
        --n_val_domain=$n_val_domain --n_test_domain=$n_test_domain \
        --pool prompt --template 'This is a [MASK] intent: [sentence]' \
        --add_prol --add_prosq --cross_domain --protype single --seed=$seed --T $temp --SG mean --lr=$lr
fi

# --- clinc150 10-way 5-shot ---
way=10
shot=5
RESULT_FILE="${RESULT_DIR}/result/${way}-way_${shot}-shot_${dataset}_result.csv"
if [ -f "$RESULT_FILE" ]; then
    echo "Arquivo de resultado encontrado. Pulando: ${dataset} ${way}-way ${shot}-shot"
else
    echo "Executando: ${dataset} ${way}-way ${shot}-shot"
    python ../src/main.py \
        --output_dir /raid/user_beatrizalmeida/lgpn_results \
        --cuda $cuda --way $way --shot $shot --query 25 --mode train --classifier mbc \
        --dataset=$dataset --data_path=$data_path --n_train_class=$n_train_class \
        --n_val_class=$n_val_class --n_test_class=$n_test_class --n_train_domain=$n_train_domain \
        --n_val_domain=$n_val_domain --n_test_domain=$n_test_domain \
        --pool prompt --template 'This is a [MASK] intent: [sentence]' \
        --add_prol --add_prosq --protype single --seed=$seed --cross_domain --T $temp --SG mean --lr=$lr
fi

# --- clinc150 15-way 1-shot ---
way=15
shot=1
RESULT_FILE="${RESULT_DIR}/result/${way}-way_${shot}-shot_${dataset}_result.csv"
if [ -f "$RESULT_FILE" ]; then
    echo "Arquivo de resultado encontrado. Pulando: ${dataset} ${way}-way ${shot}-shot"
else
    echo "Executando: ${dataset} ${way}-way ${shot}-shot"
    python ../src/main.py \
        --output_dir /raid/user_beatrizalmeida/lgpn_results \
        --cuda $cuda --way $way --shot $shot --query 25 --mode train --classifier mbc \
        --dataset=$dataset --data_path=$data_path --n_train_class=$n_train_class \
        --n_val_class=$n_val_class --n_test_class=$n_test_class --n_train_domain=$n_train_domain \
        --n_val_domain=$n_val_domain --n_test_domain=$n_test_domain \
        --pool prompt --template 'This is a [MASK] intent: [sentence]' \
        --add_prol --add_prosq --protype single --seed=$seed --cross_domain --T $temp --SG mean --lr=$lr
fi

# --- clinc150 15-way 5-shot ---
way=15
shot=5
RESULT_FILE="${RESULT_DIR}/result/${way}-way_${shot}-shot_${dataset}_result.csv"
if [ -f "$RESULT_FILE" ]; then
    echo "Arquivo de resultado encontrado. Pulando: ${dataset} ${way}-way ${shot}-shot"
else
    echo "Executando: ${dataset} ${way}-way ${shot}-shot"
    python ../src/main.py \
        --output_dir /raid/user_beatrizalmeida/lgpn_results \
        --cuda $cuda --way $way --shot $shot --query 25 --mode train --classifier mbc \
        --dataset=$dataset --data_path=$data_path --n_train_class=$n_train_class \
        --n_val_class=$n_val_class --n_test_class=$n_test_class --n_train_domain=$n_train_domain \
        --n_val_domain=$n_val_domain --n_test_domain=$n_test_domain \
        --pool prompt --template 'This is a [MASK] intent: [sentence]' \
        --add_prol --add_prosq --protype single --seed=$seed --cross_domain --T $temp --SG mean --lr=$lr
fi


#
# Experimentos para banking77
#
dataset=banking77
data_path="../data/banking_data/"
n_train_class=30
n_val_class=15
n_test_class=32

# --- banking77 10-way 1-shot ---
way=10
shot=1
RESULT_FILE="${RESULT_DIR}/result/${way}-way_${shot}-shot_${dataset}_result.csv"
if [ -f "$RESULT_FILE" ]; then
    echo "Arquivo de resultado encontrado. Pulando: ${dataset} ${way}-way ${shot}-shot"
else
    echo "Executando: ${dataset} ${way}-way ${shot}-shot"
    python ../src/main.py \
        --output_dir /raid/user_beatrizalmeida/lgpn_results \
        --cuda $cuda --way $way --shot $shot --query 25 --mode train --classifier mbc \
        --dataset=$dataset --data_path=$data_path --n_train_class=$n_train_class \
        --n_val_class=$n_val_class --n_test_class=$n_test_class \
        --pool prompt --template 'This is a [MASK] intent: [sentence]' \
        --add_prol --add_prosq --protype single --seed=$seed --T $temp --SG mean --lr=$lr
fi

# --- banking77 10-way 5-shot ---
way=10
shot=5
RESULT_FILE="${RESULT_DIR}/result/${way}-way_${shot}-shot_${dataset}_result.csv"
if [ -f "$RESULT_FILE" ]; then
    echo "Arquivo de resultado encontrado. Pulando: ${dataset} ${way}-way ${shot}-shot"
else
    echo "Executando: ${dataset} ${way}-way ${shot}-shot"
    python ../src/main.py \
        --output_dir /raid/user_beatrizalmeida/lgpn_results \
        --cuda $cuda --way $way --shot $shot --query 25 --mode train --classifier mbc \
        --dataset=$dataset --data_path=$data_path --n_train_class=$n_train_class \
        --n_val_class=$n_val_class --n_test_class=$n_test_class \
        --pool prompt --template 'This is a [MASK] intent: [sentence]' \
        --add_prol --add_prosq --protype single --seed=$seed --T $temp --SG mean --lr=$lr
fi

# --- banking77 15-way 1-shot ---
way=15
shot=1
RESULT_FILE="${RESULT_DIR}/result/${way}-way_${shot}-shot_${dataset}_result.csv"
if [ -f "$RESULT_FILE" ]; then
    echo "Arquivo de resultado encontrado. Pulando: ${dataset} ${way}-way ${shot}-shot"
else
    echo "Executando: ${dataset} ${way}-way ${shot}-shot"
    python ../src/main.py \
        --output_dir /raid/user_beatrizalmeida/lgpn_results \
        --cuda $cuda --way $way --shot $shot --query 25 --mode train --classifier mbc \
        --dataset=$dataset --data_path=$data_path --n_train_class=$n_train_class \
        --n_val_class=$n_val_class --n_test_class=$n_test_class \
        --pool prompt --template 'This is a [MASK] intent: [sentence]' \
        --add_prol --add_prosq --protype single --seed=$seed --T $temp --SG mean --lr=$lr
fi

# --- banking77 15-way 5-shot ---
way=15
shot=5
RESULT_FILE="${RESULT_DIR}/result/${way}-way_${shot}-shot_${dataset}_result.csv"
if [ -f "$RESULT_FILE" ]; then
    echo "Arquivo de resultado encontrado. Pulando: ${dataset} ${way}-way ${shot}-shot"
else
    echo "Executando: ${dataset} ${way}-way ${shot}-shot"
    python ../src/main.py \
        --output_dir /raid/user_beatrizalmeida/lgpn_results \
        --cuda $cuda --way $way --shot $shot --query 25 --mode train --classifier mbc \
        --dataset=$dataset --data_path=$data_path --n_train_class=$n_train_class \
        --n_val_class=$n_val_class --n_test_class=$n_test_class \
        --pool prompt --template 'This is a [MASK] intent: [sentence]' \
        --add_prol --add_prosq --protype single --seed=$seed --T $temp --SG mean --lr=$lr
fi

echo "Todos os experimentos foram concluídos ou já existiam."