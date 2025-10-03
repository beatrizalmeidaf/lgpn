import torch
import datetime
from typing import Union
from typing import *

def tprint(s):
    """
    Imprime a data e hora atuais seguidas por uma string.
    @params:
        s (str): A string a ser impressa.
    """
    print('{}: {}'.format(
        datetime.datetime.now(), s),
        flush=True)

def to_tensor(data, cuda, exclude_keys=[]):
    """
    Converte todos os valores numpy em um dicionário para tensores do PyTorch.
    """
    # itera sobre todas as chaves no dicionário de dados
    for key in data.keys():
        # ignora as chaves que estão na lista de exclusão
        if key in exclude_keys:
            continue

        # converte o array numpy para um tensor torch
        data[key] = torch.from_numpy(data[key])
        # se um dispositivo cuda for especificado, move o tensor para a gpu
        if cuda != -1:
            data[key] = data[key].cuda(cuda)

    return data

def select_subset(old_data, new_data, keys, idx, max_len=None):
    '''
    Seleciona um subconjunto de dados de um dicionário e o armazena em outro.
    Modifica o dicionário 'new_data' diretamente.

    @param old_data: Dicionário de origem dos dados.
    @param new_data: Dicionário de destino para armazenar o subconjunto.
    @param keys: Lista de chaves a serem transferidas.
    @param idx: Lista de índices a serem selecionados.
    @param max_len: (Opcional) Comprimento máximo para truncar as sequências.
    '''
    for k in keys:
        # seleciona os dados usando os índices fornecidos
        new_data[k] = old_data[k][idx]
        # se um comprimento máximo for definido e o tensor tiver mais de uma dimensão, trunca a segunda dimensão
        if max_len is not None and len(new_data[k].shape) > 1:
            new_data[k] = new_data[k][:,:max_len]

    return new_data

class InputExample(object):
    """
    Uma classe para representar um único exemplo de entrada de dados.
    Consiste em segmentos de texto, um rótulo para tarefas de classificação
    ou uma sequência alvo para tarefas de geração.
    Outras informações desejadas podem ser passadas através do dicionário 'meta'.

    Args:
        guid (:obj:`str`, opcional): Um identificador único para o exemplo.
        text_a (:obj:`str`, opcional): O texto principal da sequência.
        text_b (:obj:`str`, opcional): Uma segunda sequência de texto, nem sempre necessária.
        label (:obj:`int`, opcional): O ID do rótulo para tarefas de classificação.
        meta (:obj:`Dict`, opcional): Um dicionário para armazenar informações extras arbitrárias.
        tgt_text (:obj:`Union[str,List[str]]`, opcional): A sequência de texto alvo para tarefas de geração.
    """
    def __init__(self,
                 guid = None,
                 text_a = "",
                 text_b = "",
                 label = None,
                 meta: Optional[Dict] = None,
                 tgt_text: Optional[Union[str, List[str]]] = None
                 ):
        """Inicializa uma instância de InputExample."""
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.meta = meta if meta else {}
        self.tgt_text = tgt_text