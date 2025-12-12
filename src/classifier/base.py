import torch
import torch.nn as nn
import torch.nn.functional as F

class BASE(nn.Module):
    '''Classe BASE:
    Implementa a estrutura base de um modelo de few-shot learning.
    - Herda de nn.Module para usar a infraestrutura de redes neurais do PyTorch.
    - Armazena parâmetros de configuração (args).
    - Cria uma matriz identidade usada como referência para codificação one-hot.
    - Fornece funções auxiliares para cálculo de distâncias (L2 e cosseno),
      reorganização de rótulos, inicialização de MLPs, conversão de rótulos
      para one-hot e cálculo de acurácia.

    Serve como bloco fundamental para experimentos em tarefas
    de classificação com poucos exemplos.
    '''
    
    def __init__(self, args):
        super(BASE, self).__init__()
        # armazena os argumentos passados
        self.args = args

        # tensor em cache para velocidade
        # cria uma matriz identidade que será usada para a codificação one-hot
        self.I_way = nn.Parameter(torch.eye(self.args.way, dtype=torch.float),
                                  requires_grad=False)

    def _compute_l2(self, XS, XQ):
        """
        Calcula a distância L2 (euclidiana ao quadrado) entre todos os pares 
        de vetores do conjunto de suporte (XS) e do conjunto de consulta (XQ).
        Retorna uma matriz onde cada posição representa a distância entre 
        uma amostra de consulta e uma de suporte.
        """
        # obtém as dimensões dos tensores de entrada
        n = XS.size(0)
        m = XQ.size(0)
        d = XS.size(1)
    
        # Supondo que a intenção era comparar com XQ.
        assert d == XQ.size(1)

        # expande as dimensões dos tensores para permitir o cálculo pairwise
        x = XS.unsqueeze(1).expand(n, m, d)
        y = XQ.unsqueeze(0).expand(n, m, d)

        # calcula a soma dos quadrados das diferenças (distância L2 ao quadrado)
        return torch.pow(x - y, 2).sum(2)
        # diff = XS.unsqueeze(0) - XQ.unsqueeze(1)
        # dist = torch.norm(diff, dim=2)
        # return dist

    def _compute_cos(self, XS, XQ):
        """
        Calcula a distância de cosseno entre todos os pares de vetores 
        de suporte (XS) e de consulta (XQ).
        A distância é definida como 1 - similaridade do cosseno, 
        de modo que valores menores indicam maior similaridade.
        """
        # import pdb; pdb.set_trace()
        # calcula o produto escalar entre todos os pares de vetores de suporte (XS) e de consulta (XQ)
        dot = torch.matmul(
            XS.unsqueeze(0).unsqueeze(-2),
            XQ.unsqueeze(1).unsqueeze(-1)
        )
        # remove dimensões extras do resultado
        dot = dot.squeeze(-1).squeeze(-1)

        # calcula o produto das normas dos vetores para normalização
        scale = (torch.norm(XS, dim=1).unsqueeze(0) *
                 torch.norm(XQ, dim=1).unsqueeze(1))

        # garante que a escala não seja zero para evitar divisão por zero
        scale = torch.max(scale,
                          torch.ones_like(scale) * 1e-8)

        # calcula a distância de cosseno como 1 - similaridade de cosseno
        dist = 1 - dot/scale
        # dist = dot/scale

        return dist

    def reidx_y(self, YS, YQ):
        """
        Reorganiza os rótulos dos conjuntos de suporte (YS) e de consulta (YQ) 
        para que fiquem no intervalo [0, way-1].
        Verifica se os conjuntos de classes são consistentes entre suporte 
        e consulta e se correspondem ao número esperado de classes (way).
        Retorna os rótulos remapeados.
        """
        # encontra os rótulos únicos e os índices para mapeá-los para 0, 1, 2...
        # Identifica as classes únicas presentes no Suporte e ordena
        # return_inverse=True nos dá os índices transformados para YS automaticamente
        unique_classes, inv_S = torch.unique(YS, sorted=True, return_inverse=True)
        
        # Para o Query (YQ), nós NÃO calculamos o unique separadamente (isso causava o erro).
        # Em vez disso, procuramos onde os valores de YQ se encaixam na referência (unique_classes).
        # Isso funciona mesmo que YQ não tenha todas as classes de YS.
        
        # torch.searchsorted encontra os índices dos elementos de YQ dentro de unique_classes
        inv_Q = torch.searchsorted(unique_classes, YQ)

        return inv_S, inv_Q

    def _init_mlp(self, in_d, hidden_ds, drop_rate):
        """
        Inicializa uma rede MLP (perceptron multicamadas) a partir de uma 
        dimensão de entrada, uma lista de dimensões ocultas e uma taxa de dropout.
        A rede alterna entre camadas lineares, dropout e ReLU, 
        e finaliza com uma camada de saída linear.
        Retorna o modelo sequencial pronto para uso.
        """
        # inicializa uma lista para armazenar as camadas da rede
        modules = []

        # itera sobre as dimensões das camadas ocultas para construir a rede
        for d in hidden_ds[:-1]:
            modules.extend([
                nn.Dropout(drop_rate),      # camada de dropout para regularização
                nn.Linear(in_d, d),         # camada linear
                nn.ReLU()])                 # função de ativação ReLU
            # atualiza a dimensão de entrada para a próxima camada
            in_d = d

        # adiciona a camada de saída final
        modules.extend([
            nn.Dropout(drop_rate),
            nn.Linear(in_d, hidden_ds[-1])])

        # retorna um container sequencial com todas as camadas
        return nn.Sequential(*modules)

    def _label2onehot(self, Y):
        """
        Converte rótulos inteiros para representação one-hot.
        Cada rótulo é transformado em um vetor binário de dimensão "way", 
        com 1 na posição correspondente à classe.
        Retorna um tensor no formato one-hot.
        """
        # converte rótulos de inteiros para o formato one-hot usando a matriz identidade
        Y_onehot = F.embedding(Y, self.I_way)

        return Y_onehot

    @staticmethod
    def compute_acc(pred, true):
        """
        Calcula a acurácia das previsões.
        Compara os rótulos previstos (pred) com os verdadeiros (true) 
        e retorna a proporção de acertos como um número entre 0 e 1.
        """
        # calcula a acurácia como a média de previsões corretas
        return torch.mean((pred == true).float()).item()