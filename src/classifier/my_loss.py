import torch
import torch.nn as nn
import numpy as np

def dot_similarity(XS, XQ):
    """Calcula a similaridade de produto escalar entre dois tensores."""
    return torch.matmul(XS, XQ.t())

def euclidean_dist(x, y):
    """Calcula a distância euclidiana ao quadrado par a par entre dois tensores."""
    # x: N x D
    # y: M x D
    # obtém as dimensões dos tensores de entrada
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    # garante que a dimensão do embedding seja a mesma para ambos os tensores
    assert d == y.size(1)

    # expande as dimensões para permitir o cálculo par a par
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    # retorna a soma dos quadrados das diferenças (distância euclidiana ao quadrado)
    return torch.pow(x - y, 2).sum(2)

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        """
        Inicializa a camada de Triplet Loss (Perda Triplete).
        A margem é um hiperparâmetro que força a distância entre pares negativos
        a ser maior do que a distância entre pares positivos por pelo menos esse valor.
        """
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Calcula a Triplet Loss.
        O objetivo é minimizar a distância entre a âncora e o positivo,
        e maximizar a distância entre a âncora e o negativo.
       """
        distance_positive = torch.pairwise_distance(anchor, positive)
        distance_negative = torch.pairwise_distance(anchor, negative)
        # a perda é zero se a distância negativa for maior que a positiva pela margem
        loss = torch.relu(distance_positive - distance_negative + self.margin)
        return loss.mean()


class LG_loss(nn.Module):

    def __init__(self, args):
        """
        Inicializa a camada de perda LG (Local-Global).
        Utiliza um parâmetro de temperatura 'tau' para escalar as similaridades.
        """
        super(LG_loss, self).__init__()
        self.tau = args.T
        self.args = args

    def similarity(self, x1, x2):
        """
        Calcula a matriz de similaridade entre dois conjuntos de vetores.
        Pode usar a distância euclidiana (l2) ou o produto escalar como base.
        """
        # kernel gaussiano
        if self.args.sim == 'l2':
            M = euclidean_dist(x1, x2)
            s = torch.exp(-M/self.tau)
        else:
            # produto escalar
            M = dot_similarity(x1, x2)/self.tau
            # subtrai o máximo para estabilidade numérica antes da exponencial (log-sum-exp trick)
            s = torch.exp(M - torch.max(M, dim=1, keepdim=True)[0])
        return s

    def forward(self, batch_labels, X, L):
        """
        Calcula a perda contrastiva entre as características (X) e os embeddings dos rótulos (L).
        A perda incentiva a similaridade entre uma característica e seu rótulo correspondente
        a ser maior do que com outros rótulos.
        """
        # obtém o tamanho do lote
        len_ = batch_labels.size()[0]

        # calculando similaridades para cada par positivo e negativo
        s = self.similarity(X, L)

        # máscara para incluir todos os pares na soma do denominador
        mask_i = torch.from_numpy(
            np.ones((len_, len_))).to(batch_labels.device)

        # cria uma matriz de rótulos para identificar pares positivos
        label_matrix = batch_labels.unsqueeze(0).repeat(len_, 1)

        # máscara para identificar os pares positivos (mesmo rótulo)
        mask_j = (batch_labels.unsqueeze(1) - label_matrix ==
                  0).float() * mask_i  # soma sobre os itens no denominador

        # conta o número de pares positivos para normalização
        pos_num = torch.sum(mask_j, 1)

        # perda nll (negative log likelihood) ponderada
        # denominador: soma das similaridades com todos os rótulos
        s_i = torch.clamp(torch.sum(s*mask_i, 1), min=1e-10)
        # numerador: similaridade apenas com o rótulo correto (positivo)
        s_j = torch.clamp(s*mask_j, min=1e-10)
        # calcula a perda de log, normalizada pelo número de positivos
        log_p = torch.sum(-torch.log(s_j/s_i) * mask_j, 1)/pos_num
        # calcula a média da perda para o lote
        loss = torch.mean(log_p)

        return loss