import copy
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class EMBED_BERT(nn.Module):
    """
    Modelo de incorporação baseado em BERT.
    Usa um modelo BERT pré-treinado para gerar representações de texto.
    Pode usar diferentes estratégias de pooling para agregar as representações.
    """

    def __init__(self, args):
        """
        Inicializa o modelo de incorporação BERT com base nos argumentos fornecidos.
        Carrega o modelo e o tokenizador BERT pré-treinados.
        Args:
            args: Argumentos contendo configurações como o modelo BERT pré-treinado,
                  diretório de cache, estratégia de pooling e template.
        """

        super(EMBED_BERT, self).__init__()
        self.args = args
        print("{}, Loading pretrained bert".format(
            datetime.datetime.now()), flush=True)

        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert)
        
        self.model = BertModel.from_pretrained(self.args.pretrained_bert,
                                               cache_dir=self.args.bert_cache_dir)
        
        self.modelfix = BertModel.from_pretrained(self.args.pretrained_bert,
                                                  cache_dir=self.args.bert_cache_dir)

        self.embedding_dim = self.model.config.hidden_size
        self.ebd_dim = self.model.config.hidden_size
        self.template = args.template
        args.embedding_dim = self.embedding_dim

    def forward(self, input_example, query=False):
        """
        Gera representações de texto para uma lista de exemplos de entrada.
        Args:
            input_example: Lista de exemplos de entrada, onde cada exemplo é uma instância de InputExample.
            query: Booleano indicando se o exemplo é uma consulta (não usado aqui).
        Returns:
            predictions: Tensor contendo as representações de texto.
            label_outputs: Tensor contendo as representações dos rótulos.
        """
        
        sentence = [self.args.template.replace(
            "[sentence]", x.text_a)for x in input_example]
        
        if self.args.dataset == '20newsgroup2' or self.args.dataset == '20newsgroup' or self.args.dataset == 'reuters':
            inputs = self.tokenizer.batch_encode_plus(sentence, return_tensors='pt',padding='max_length', max_length=256, truncation=True)
        
        elif self.args.dataset == 'reuters':
            inputs = self.tokenizer.batch_encode_plus(sentence, return_tensors='pt',padding='max_length', max_length=64, truncation=True)
        
        elif self.args.dataset == 'amazon2' or self.args.dataset == 'amazon':
            inputs = self.tokenizer.batch_encode_plus(sentence, return_tensors='pt',padding='max_length', max_length=128, truncation=True)
        
        else:
            inputs = self.tokenizer.batch_encode_plus(
                sentence, return_tensors='pt', padding=True)
        
        inputs.to(self.args.device)
        outputs = self.model(**inputs)
        
        labels = [x.text_b for x in input_example]
        
        label_inputs = self.tokenizer.batch_encode_plus(
            labels, return_tensors='pt', padding=True).to(self.args.device)
        
        label_outputs = self.model(
            **label_inputs).last_hidden_state.mean(dim=1)

        predictions = torch.zeros(
            [inputs['input_ids'].shape[0], self.embedding_dim]).to(self.args.device)
        mask_token_index = torch.where(
            inputs['input_ids'] == self.tokenizer.mask_token_id)[1]

        if self.args.pool == 'prompt':
            for i in range(len(predictions)):
                predictions[i] = outputs.last_hidden_state[i,
                                                           mask_token_index[i], :]
        elif self.args.pool == 'cls':
            for i in range(len(predictions)):
                predictions[i] = outputs.last_hidden_state[i, 0, :]
        
        elif self.args.pool == 'avg':
            predictions = outputs.last_hidden_state.mean(dim=1)
        
        return predictions, label_outputs
