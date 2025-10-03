import torch
import torch.nn as nn
import numpy as np

class Contrastive_Loss_base(nn.Module):
    """
    Implementa a perda contrastiva de forma genérica, usando uma 
    temperatura fixa (tau) para controlar a escala das similaridades.
    Essa versão base serve como modelo simples para calcular similaridades
    e a perda associada entre pares de amostras (positivos e negativos).
    """

    def __init__(self, tau=5.0):
        """
        Inicializa a classe Contrastive_Loss_base.
        Define o parâmetro de temperatura (tau), que controla a escala
        das similaridades.
        """
        super(Contrastive_Loss_base, self).__init__()
        self.tau = tau

    def similarity(self, x1, x2):
        """
        Calcula a similaridade entre dois tensores.
        - Pode ser feita com base no produto escalar (similaridade de cosseno escalada).
        - Aplica uma normalização pela temperatura e a função exponencial
          para obter scores de similaridade estáveis numericamente.
        Retorna uma matriz de similaridade.
        """
        # gaussian kernel (método comentado)

        # M = euclidean_dist(x1, x2)
        # s = torch.exp(-M/self.tau)

        # produto escalar
        # calcula a similaridade de cosseno e divide pela temperatura
        M = dot_similarity(x1, x2)/self.tau
        # aplica a função exponencial para obter os scores, subtraindo o máximo para estabilidade numérica
        s = torch.exp(M - torch.max(M, dim=1, keepdim=True)[0])
        return s

    def forward(self, batch_label, *x):
        """
        Define o passo de forward da perda contrastiva.
        - Concatena os tensores de entrada.
        - Cria máscaras para identificar pares positivos (mesmo rótulo)
          e pares negativos (rótulos diferentes).
        - Calcula as similaridades entre todos os pares.
        - Aplica a fórmula da perda contrastiva baseada em log-likelihood
          (NLL) para incentivar alta similaridade entre pares positivos
          e baixa entre negativos.
        Retorna o valor da perda média.
        """

        # concatena todas as entradas em um único tensor
        X = torch.cat(x, 0)
        # cria um tensor de rótulos correspondente ao tensor concatenado
        batch_labels = torch.cat([batch_label for i in range(len(x))], 0)
        # obtém o tamanho do lote
        len_ = batch_labels.size()[0]

        # calculando similaridades para cada par positivo e negativo
        s = self.similarity(X, X)

        # calculando máscaras para a perda contrastiva
        if len(x) == 1:
            # se houver apenas uma entrada, a máscara considera todos os pares
            mask_i = torch.from_numpy(
                np.ones((len_, len_))).to(batch_labels.device)
        else:
            # soma sobre os itens no numerador (exclui a diagonal principal)
            mask_i = 1. - \
                torch.from_numpy(np.identity(len_)).to(batch_labels.device)
        # cria uma matriz de rótulos para comparação par a par
        label_matrix = batch_labels.unsqueeze(0).repeat(len_, 1)
        # cria uma máscara para os pares positivos (mesmo rótulo)
        mask_j = (batch_labels.unsqueeze(1) - label_matrix ==
                  0).float()*mask_i  # soma sobre os itens no denominador
        # conta o número de pares positivos para cada amostra
        pos_num = torch.sum(mask_j, 1)

        # perda nll ponderada (negative log likelihood)
        # soma das similaridades para o denominador (todos os pares, exceto o próprio)
        s_i = torch.clamp(torch.sum(s*mask_i, 1), min=1e-10)
        # similaridades apenas dos pares positivos para o numerador
        s_j = torch.clamp(s*mask_j, min=1e-10)
        # calcula o log da probabilidade para os pares positivos e normaliza
        log_p = torch.sum(-torch.log(s_j/s_i)*mask_j, 1)/pos_num
        # calcula a média da perda sobre o lote
        loss = torch.mean(log_p)

        return loss

def dot_similarity(XS, XQ):
    """
    Calcula a similaridade pelo produto escalar entre dois conjuntos de vetores.
    Retorna uma matriz onde cada entrada é o produto escalar entre 
    uma amostra de XS e uma de XQ.
    """
    return torch.matmul(XS, XQ.t())

def euclidean_dist(x, y):
    """
    Calcula a distância euclidiana ao quadrado entre todos os pares
    de vetores de dois tensores.
    - Expande as dimensões dos tensores para permitir comparação par a par.
    - Retorna uma matriz com as distâncias ao quadrado entre cada par
      de vetores.
    """

    # x: N x D
    # y: M x D
    # obtém as dimensões dos tensores de entrada
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    # verifica se a dimensão do embedding é a mesma
    assert d == y.size(1)

    # expande as dimensões para permitir o cálculo par a par
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    # retorna a soma dos quadrados das diferenças
    return torch.pow(x - y, 2).sum(2)

class Contrastive_Loss(nn.Module):
    """
    Classe Contrastive_Loss:
    Implementa a versão principal da perda contrastiva, com suporte 
    a diferentes tipos de cálculo de similaridade:
      - 'l2': distância euclidiana.
      - Caso contrário: produto escalar (similaridade de cosseno escalada).
    Usa uma temperatura (tau) fornecida nos argumentos para ajustar
    a escala das similaridades.
    """

    def __init__(self, args):
        """
        Inicializa a classe Contrastive_Loss.
        - Recebe os argumentos do modelo.
        - Define a temperatura (tau) a partir dos argumentos.
        """
        super(Contrastive_Loss, self).__init__()
        self.tau = args.T
        self.args = args

    def similarity(self, x1, x2):
        """
        Calcula a similaridade entre dois tensores com base no método
        especificado em self.args.sim:
        - Se for 'l2', usa distância euclidiana.
        - Caso contrário, usa produto escalar (similaridade de cosseno escalada).
        Retorna uma matriz de similaridades após aplicar a exponencial
        e normalização numérica.
        """

        # gaussian kernel
        if self.args.sim == 'l2':
            # usa a distância euclidiana
            M = euclidean_dist(x1, x2)
            s = torch.exp(-M/self.tau)
        else:
            # produto escalar
            # usa a similaridade de cosseno
            M = dot_similarity(x1, x2)/self.tau
            s = torch.exp(M - torch.max(M, dim=1, keepdim=True)[0])
        return s

    def forward(self, batch_labels, *x):
        """
        Define o passo de forward da perda contrastiva.
        - Concatena as entradas no tensor X.
        - Constrói máscaras para pares positivos (mesmo rótulo) e negativos.
        - Calcula as similaridades entre todos os pares de amostras.
        - Aplica a fórmula da perda contrastiva NLL ponderada:
          log_p = -log(similaridade_positiva / soma_das_similaridades)
        - Normaliza pela quantidade de pares positivos.
        - Retorna a média da perda para o lote.
        """

        # concatena todas as entradas em um único tensor
        X = torch.cat(x, 0)
        # batch_labels = torch.cat([batch_label for i in range(len(x))], 0)
        # obtém o tamanho do lote
        len_ = batch_labels.size()[0]
        # import pdb
        # pdb.set_trace()
        # calculando similaridades para cada par positivo e negativo
        s = self.similarity(X, X)

        # calculando máscaras para a perda contrastiva
        if len(x) == 1:
            # se houver apenas uma entrada, a máscara considera todos os pares
            mask_i = torch.from_numpy(
                np.ones((len_, len_))).to(batch_labels.device)
        else:
            # soma sobre os itens no numerador (exclui a si mesmo)
            mask_i = 1. - \
                torch.from_numpy(np.identity(len_)).to(batch_labels.device)
        # cria uma matriz de rótulos para comparação
        label_matrix = batch_labels.unsqueeze(0).repeat(len_, 1)
        # cria uma máscara para os pares positivos
        mask_j = (batch_labels.unsqueeze(1) - label_matrix ==
                  0).float() * mask_i  # soma sobre os itens no denominador
        # conta o número de amostras positivas por âncora
        pos_num = torch.sum(mask_j, 1)

        # perda nll ponderada
        # calcula o denominador da perda (soma das similaridades com todos os outros)
        s_i = torch.clamp(torch.sum(s * mask_i, 1), min=1e-10)
        # calcula o numerador (similaridades com os positivos)
        s_j = torch.clamp(s * mask_j, min=1e-10)
        # calcula a perda de log, normalizada pelo número de positivos
        log_p = torch.sum(-torch.log(s_j/s_i)*mask_j, 1)/pos_num
        # calcula a média da perda no lote
        loss = torch.mean(log_p)

        return loss