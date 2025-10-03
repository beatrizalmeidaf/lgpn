# LGPN

## Criação do ambiente Python

```bash
conda create -n LDS 
source activate LDS
pip install -r requirements.txt
```

## Ambiente na DGX

```bash
export PATH="/home/user_beatrizalmeida/.local/bin:$PATH"
python3 -m venv ~/workspace/lgpn_venv
source ~/workspace/lgpn_venv/bin/activate

pip install --upgrade pip
pip install --upgrade setuptools wheel
pip install -r /workspace/requirements.txt 
# ou 
python -m pip install --no-build-isolation -r /workspace/requirements.txt
```

## Execução rápida

```bash
cd scripts
sbatch executar_job.sh

# ou 
sh ours_all.sh
```
Ver job
```bash 
squeue -u user_beatrizalmeida # verificar andamento

tail -f laqda_saida_1113.log # ver saida em tempo real
scancel 1113 # cancelar job
```
Os parâmetros utilizados no artigo são consistentes com os definidos em `ours_all.sh`.

**Observação:** antes de iniciar, é necessário fazer o download do modelo [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) e atualizar o caminho no arquivo `ours_all.sh` para o local correspondente no seu ambiente.
