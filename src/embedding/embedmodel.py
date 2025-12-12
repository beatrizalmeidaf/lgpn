import copy
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class EMBED_BERT(nn.Module):
    """
    Modelo de incorporação baseado em BERT.
    """

    def __init__(self, args):
        super(EMBED_BERT, self).__init__()
        self.args = args
        print("{}, Loading pretrained bert".format(datetime.datetime.now()), flush=True)

        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert)
        self.model = BertModel.from_pretrained(args.pretrained_bert,
                                               cache_dir=args.bert_cache_dir)
        self.modelfix = BertModel.from_pretrained(args.pretrained_bert,
                                                  cache_dir=args.bert_cache_dir)

        self.embedding_dim = self.model.config.hidden_size
        self.ebd_dim = self.model.config.hidden_size
        self.template = args.template
        args.embedding_dim = self.embedding_dim

    def forward(self, input_example, query=False):
        """
        Gera representações de texto para uma lista de exemplos de entrada.
        """

        # Proteção contra lista vazia
        if not input_example:
            device = self.args.device
            return (
                torch.zeros(0, self.embedding_dim).to(device),
                torch.zeros(0).to(device)
            )

        # Aplica template
        sentence = [self.args.template.replace("[sentence]", x.text_a)
                    for x in input_example]

        # Define max_len
        max_len = 512
        if self.args.dataset in ['20newsgroup2', '20newsgroup', 'reuters']:
            max_len = 256
        elif self.args.dataset in ['amazon2', 'amazon']:
            max_len = 128

        # Tokenização
        inputs = self.tokenizer.batch_encode_plus(
            sentence,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_len
        ).to(self.args.device)

        outputs = self.model(**inputs)

        # Codifica labels
        labels = [x.text_b for x in input_example]
        label_inputs = self.tokenizer.batch_encode_plus(
            labels, return_tensors='pt', padding=True
        ).to(self.args.device)

        label_outputs = self.model(**label_inputs).last_hidden_state.mean(dim=1)

        # Pooling
        predictions = torch.zeros(
            [inputs['input_ids'].shape[0], self.embedding_dim],
            device=self.args.device
        )

        mask_token_index = torch.where(
            inputs['input_ids'] == self.tokenizer.mask_token_id
        )[1]

        if self.args.pool == 'prompt':
            for i in range(len(predictions)):
                predictions[i] = outputs.last_hidden_state[i, mask_token_index[i], :]

        elif self.args.pool == 'cls':
            for i in range(len(predictions)):
                predictions[i] = outputs.last_hidden_state[i, 0, :]

        elif self.args.pool == 'avg':
            predictions = outputs.last_hidden_state.mean(dim=1)

        return predictions, label_outputs
