import time
import numpy as np

class FewshotSampler():
    def __init__(self, data, args, num_episodes=None, state='train'):
        """
        Inicializa o amostrador para aprendizado few-shot.

        @param data: O conjunto de dados completo a partir do qual amostrar.
        @param args: Argumentos de configuração (contém way, shot, query, etc.).
        @param num_episodes: O número de episódios a serem gerados em uma época.
        @param state: O estado atual ('train', 'test' ou 'val'), para selecionar o conjunto correto de classes.
        """
        self.data = data
        self.args = args
        self.state = state
        self.num_episodes = num_episodes
        # seleciona o número total de classes com base no estado (treino, teste, validação)
        if self.state == 'train':
            self.num_classes = args.train_classes
        elif self.state == 'test':
            self.num_classes = args.test_classes
        elif self.state == 'val':
            self.num_classes = args.val_classes

    def get_sample(self, classes, data):
        """
        Cria um conjunto de suporte e um conjunto de consulta a partir de uma lista de classes.

        @param classes: Uma lista de rótulos de classe a serem incluídos no episódio.
        @param data: O conjunto de dados de onde as amostras serão retiradas.
        @return: Uma tupla contendo a lista de exemplos de suporte e a lista de exemplos de consulta.
        """
        # dicionário para agrupar exemplos por classe
        examples = {}
        for c in classes:
            examples[c] = []
        # listas para os conjuntos de suporte e consulta
        support_examples = []
        query_examples = []
 
        # itera sobre os dados para agrupar exemplos por classe
        for d in data:
            if d.label in classes:
                examples[d.label].append(d)
                
        # para cada classe selecionada, amostra 'shot' exemplos para o suporte e 'query' para a consulta
        for c in classes:
            support_examples.extend(examples[c][:self.args.shot])
            query_examples.extend(examples[c][self.args.shot: self.args.shot + self.args.query])
       
        return support_examples, query_examples

    def get_epoch(self):
        """
        Gerador que produz episódios para uma época inteira.
        Cada episódio consiste em um conjunto de suporte e um de consulta.
        """
        # gera um número definido de episódios
        for _ in range(self.num_episodes):
            # seleciona aleatoriamente 'way' classes do total de classes disponíveis
            sampled_classes = np.random.permutation(self.num_classes)[:self.args.way]
            # obtém os conjuntos de suporte e consulta para as classes selecionadas
            support, query = self.get_sample(sampled_classes, self.data)

            # retorna o episódio como um par (suporte, consulta)
            yield support, query