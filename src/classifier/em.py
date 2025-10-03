import argparse
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class EM_fusion(nn.Module):
    def __init__(self, args, embedding_dim=768):
        """
        Inicializa o módulo de fusão em.
        - Armazena os argumentos de configuração.
        - Define um epsilon para estabilidade numérica.
        - Registra uma matriz identidade como um buffer, que não é um parâmetro do modelo.
        """
        super().__init__()
        self.args = args
        # um valor pequeno para evitar divisão por zero ou instabilidade numérica
        self.eps = 1e-6
        # registra a matriz identidade como um buffer para que ela seja movida para a gpu/cpu junto com o modelo
        self.register_buffer('eye', torch.eye(embedding_dim))
        
    def em_fusion(self, sample, label):
        """
        Realiza uma única iteração do algoritmo expectation-maximization (em) para calcular o peso de fusão.
        - Combina a amostra e o rótulo em um único tensor.
        - Inicializa a média como a média aritmética da amostra e do rótulo.
        - Usa uma matriz identidade fixa como matriz de covariância.
        - Executa o passo e (expectation): calcula as responsabilidades (pesos) da amostra e do rótulo.
        - Retorna o peso calculado para a amostra original.
        """
        # combina a amostra atual e o rótulo correspondente
        combined = torch.stack([sample, label])

        # inicializa a média (média da amostra e do rótulo)
        mean = 0.5 * (sample + label)
        
        # covariância fixa (não é possível estimar a covariância de uma única amostra, usa-se a matriz identidade)
        cov = torch.eye(sample.size(0), device=sample.device)
        
        # iteração única de em (amostra única não requer múltiplas iterações)
        diff = combined - mean
        # calcula a log-responsabilidade (semelhante a uma log-probabilidade não normalizada)
        log_resp = -0.5 * diff.pow(2).sum(dim=1, keepdim=True)  # [2,1]
        # normaliza as log-responsabilidades para obter probabilidades (pesos)
        resp = (log_resp - torch.logsumexp(log_resp, dim=0)).exp()
        
        # retorna o peso da amostra (resp[0] é o peso da amostra)
        return resp[0].item()  # retorna um valor escalar

    def forward(self, feature_1, feature_2):
        """
        Define o passo de forward para fundir dois conjuntos de características.
        - Inicializa tensores para armazenar os pesos e os resultados da fusão.
        - Itera sobre cada amostra no lote.
        - Para cada par de características (amostra e rótulo), calcula um peso de fusão usando o método em_fusion.
        - Realiza a fusão como uma soma ponderada: fused = weight * feature_1 + (1 - weight) * feature_2.
        - Retorna o tensor com as características fundidas.
        """
        # inicializa tensores para os pesos e o resultado da fusão
        weights = torch.zeros(feature_1.size(0), device=feature_1.device)
        fused = torch.zeros_like(feature_1, device=feature_1.device)
        
        # processa amostra por amostra
        for i in range(feature_1.size(0)):
            # obtém a amostra atual e o rótulo correspondente
            sample = feature_1[i]  # [dim]
            label = feature_2[i]   # [dim]
            
            # calcula o peso de fusão para a amostra atual
            weight = self.em_fusion(sample, label)
            # armazena o peso
            weights[i] = weight
            
            # realiza a fusão
            fused[i] = weight * sample + (1 - weight) * label
        
        # retorna o tensor de características fundidas
        return fused