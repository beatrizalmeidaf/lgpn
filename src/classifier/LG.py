import torch
import torch.nn as nn
from classifier.em import EM_fusion

class SingleFusion(nn.Module):
    def __init__(self, feature_dim):
        """Inicializa o módulo de fusão 'Single'."""
        super(SingleFusion, self).__init__()

    def forward(self, feature_1, feature_2):
        """Retorna apenas a primeira característica, ignorando a segunda."""
        return feature_1


class MeanFusion(nn.Module):
    def __init__(self, feature_dim):
        """Inicializa o módulo de fusão por média."""
        super(MeanFusion, self).__init__()

    def forward(self, feature_1, feature_2):
        """Calcula a média elemento a elemento das duas características."""
        return (feature_1 + feature_2)/2


class FMeanFusion(nn.Module):
    def __init__(self, feature_dim, alpha):
        """Inicializa o módulo de fusão por média ponderada fixa."""
        super(FMeanFusion, self).__init__()
        self.alpha = alpha

    def forward(self, feature_1, feature_2):
        """Calcula uma média ponderada fixa entre as duas características usando alpha."""
        return self.alpha * feature_1 + (1 - self.alpha) * feature_2


class ConnectFusion(nn.Module):
    def __init__(self, feature_dim):
        """Inicializa o módulo de fusão por concatenação."""
        super(ConnectFusion, self).__init__()
        # self.mlp = nn.Linear(feature_dim * 2, feature_dim)
        self.mlp = nn.Sequential(nn.Linear(feature_dim * 2, feature_dim))

    def forward(self, feature_1, feature_2):
        """Concatena as duas características e as projeta de volta à dimensão original."""
        feature = torch.cat([feature_1, feature_2], dim=1)
        # import pdb
        # pdb.set_trace()
        return self.mlp(feature)


class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        """Inicializa o módulo de fusão por atenção simples."""
        super(AttentionFusion, self).__init__()
        self.attention_weights_model = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Tanh(),
            nn.Flatten()
        )

    def forward(self, feature_1, feature_2):
        """
        Realiza a fusão através de uma média ponderada, onde os pesos são
        aprendidos por um mecanismo de atenção.
        """
        attention_weights_1 = self.attention_weights_model(feature_1)
        attention_weights_2 = self.attention_weights_model(feature_2)
        attention_weights = torch.nn.functional.softmax(
            torch.cat([attention_weights_1, attention_weights_2], dim=1), dim=1)

        fused_feature = attention_weights[:, 0:1] * \
            feature_1 + attention_weights[:, 1:2] * feature_2

        return fused_feature


class AttentionFusionTransformer(nn.Module):
    def __init__(self, input_size, num_heads=8, hidden_size=128):
        """Inicializa o módulo de fusão baseado em Transformer."""
        super(AttentionFusionTransformer, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # camadas lineares para query, key e value
        self.query_linear = nn.Linear(input_size, hidden_size)
        self.key_linear = nn.Linear(input_size, hidden_size)
        self.value_linear = nn.Linear(input_size, hidden_size)

        # camada de atenção
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)

        # camada feedforward
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # normalização de camada
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x1, x2):
        """
        Realiza a fusão usando uma arquitetura de bloco de encoder de Transformer.
        x1 atua como 'query' e x2 como 'key' e 'value'.
        """
        # import pdb
        # pdb.set_trace()
        x1 = torch.unsqueeze(x1, dim=1)
        x2 = torch.unsqueeze(x2, dim=1)
        # x1 e x2 são os vetores de características de entrada
        # query = self.query_linear(x1)  # [batch_size, seq_len, hidden_size]
        # key = self.key_linear(x2)      # [batch_size, seq_len, hidden_size]
        # value = self.value_linear(x2)  # [batch_size, seq_len, hidden_size]
        query = x1
        key = x2
        value = x2
        # calcula a auto-atenção
        attention_output, _ = self.attention(query, key, value)

        # adiciona a conexão residual e normaliza
        x = self.norm1(x1 + attention_output)

        # passa pela camada feedforward
        feedforward_output = self.feedforward(x)

        # adiciona a conexão residual e normaliza
        x = self.norm2(x + feedforward_output)
        x = x.squeeze(dim=1)
        return x


class AttentionFusion3(nn.Module):
    def __init__(self, feature_dim):
        """Inicializa o módulo de fusão por atenção v3."""
        super(AttentionFusion3, self).__init__()

        self.attention_weights_model = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Tanh(),
            nn.Flatten()
        )

    def forward(self, A, B):
        """
        Calcula os pesos de atenção a partir da diagonal da matriz de similaridade
        e os utiliza para uma fusão ponderada.
        """
        similarity_matrix = torch.matmul(A, B.t())

        # attention_weights = torch.softmax(similarity_matrix, dim=1)
        attention_weights = self.attention_weights_model(
            torch.diag(similarity_matrix))
        merged_vector = A * attention_weights + B * (1 - attention_weights)
        # merged_vector = torch.mean(merged_vector, dim=1)
        return merged_vector


class AutoFusion(nn.Module):
    def __init__(self, feature_dim):
        """Inicializa o módulo de fusão com peso alpha aprendível."""
        super(AutoFusion, self).__init__()

        self.alpha = nn.Parameter(torch.tensor(1, dtype=torch.float))

    def forward(self, feature_1, feature_2):
        """Realiza uma média ponderada onde o peso (alpha) é um parâmetro aprendível."""
        return self.alpha * feature_1 + (1 - self.alpha) * feature_2


class AttentionFusion4(nn.Module):
    def __init__(self, feature_dim):
        """Inicializa o módulo de fusão por atenção v4."""
        super(AttentionFusion4, self).__init__()

        self.attention_weights_model = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
            nn.Flatten()
        )

    def forward(self, A, B):
        """
        Calcula a distância de cosseno como pesos de atenção e realiza uma fusão ponderada.
        """
        # similarity_matrix = torch.matmul(A, B.t())
        dot = torch.matmul(
            A.unsqueeze(0).unsqueeze(-2),
            B.unsqueeze(1).unsqueeze(-1)
        )
        dot = dot.squeeze(-1).squeeze(-1)

        scale = (torch.norm(A, dim=1).unsqueeze(0) *
                 torch.norm(B, dim=1).unsqueeze(1))

        scale = torch.max(scale,
                          torch.ones_like(scale) * 1e-8)

        similarity_matrix = 1 - dot/scale

        # sig = torch.unsqueeze(torch.diag(similarity_matrix), dim=0)

        # attention_weights = self.attention_weights_model(sig)
        attention_weights = torch.unsqueeze(
            torch.diag(similarity_matrix), dim=0)
        # merged_vector = (attention_weights.t() * A +
        #                  (2-attention_weights.t()) * B)/2
        # import pdb
        # pdb.set_trace()
        merged_vector = (attention_weights.t() * A +
                         (2-attention_weights.t()) * B)/2
        # merged_vector = attention_weights.t() * A +  B
    
        # merged_vector = torch.mean(merged_vector, dim=1)
        return merged_vector


class AttentionFusion5(nn.Module):
    def __init__(self, feature_dim):
        """
        Inicializa o módulo de fusão por atenção v5, que aprende um único peso
        a partir das características concatenadas.
        """
        super(AttentionFusion5, self).__init__()
        # self.mlp = nn.Linear(feature_dim * 2, feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, 1),
            nn.Sigmoid(),
            nn.Flatten()
        )

    def forward(self, feature_1, feature_2):
        """
        Concatena as características, passa por uma MLP para gerar um peso,
        e realiza uma fusão ponderada.
        """
        feature = torch.cat([feature_1, feature_2], dim=1)
        weight = self.mlp(feature)
        fused_feature = weight * feature_1 + (1 - weight) * feature_2
        return fused_feature


class AttentionFusionTransformer2(nn.Module):
    def __init__(self, input_size, num_heads=8, hidden_size=128):
        """Inicializa o módulo de fusão com atenção cruzada dupla."""
        super(AttentionFusionTransformer2, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # camada de atenção
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
            nn.Flatten()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
            nn.Flatten()
        )

    def forward(self, feature_1, feature_2):
        """
        Aplica atenção cruzada em ambas as direções (f1->f2 e f2->f1),
        gera pesos para cada uma, normaliza-os com softmax e realiza a fusão ponderada.
        """
        feature_1 = torch.unsqueeze(feature_1, dim=1)
        feature_2 = torch.unsqueeze(feature_2, dim=1)
        attention_output, _ = self.attention(feature_1, feature_2, feature_2)
        weight1 = self.mlp(attention_output.squeeze(dim=1))
        attention_output2, _ = self.attention(feature_2, feature_1, feature_1)
        weight2 = self.mlp2(attention_output2.squeeze(dim=1))
        attention_weights = torch.nn.functional.softmax(torch.cat([weight1, weight2], dim=1),
                                                        dim=1)
        fuse_feature = attention_weights[:, 0:1] * \
            feature_1.squeeze(
                dim=1) + attention_weights[:, 1:2] * feature_2.squeeze(dim=1)

        # fuse_feature = weight1 * \
        #     feature_1.squeeze(dim=1) + weight2 * feature_2.squeeze(dim=1)
        return fuse_feature


class AttentionFusionTransformer3(nn.Module):
    def __init__(self, input_size, num_heads=8, hidden_size=128):
        """Inicializa o módulo de fusão com atenção cruzada única."""
        super(AttentionFusionTransformer3, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # camada de atenção
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
            nn.Flatten()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
            nn.Flatten()
        )

    def forward(self, feature_1, feature_2):
        """
        Aplica atenção cruzada em uma direção, gera um peso e realiza a fusão.
        """
        feature_1 = torch.unsqueeze(feature_1, dim=1)
        feature_2 = torch.unsqueeze(feature_2, dim=1)
        attention_output, _ = self.attention(feature_1, feature_2, feature_2)
        weight1 = self.mlp(attention_output.squeeze(dim=1))
        fuse_feature = (weight1 *
                        feature_1.squeeze(dim=1) + (2-weight1) *
                        feature_2.squeeze(dim=1))/2
        return fuse_feature


class SG(nn.Module):
    def __init__(self, args):
        """
        Módulo seletor que instancia um dos vários métodos de fusão
        com base no argumento 'args.SG'.
        """
        super(SG, self).__init__()
        # seleciona o método de fusão com base no argumento
        if args.SG == 'att':
            self.fusion = AttentionFusion(args.embedding_dim)
        if args.SG == 'mean':
            self.fusion = MeanFusion(args.embedding_dim)
        if args.SG == 'connect':
            self.fusion = ConnectFusion(args.embedding_dim)
        if args.SG == 'single':
            self.fusion = SingleFusion(args.embedding_dim)
        if args.SG == 'fmean':
            self.fusion = FMeanFusion(args.embedding_dim, args.falpha)
        if args.SG == 'att2':
            self.fusion = AttentionFusionTransformer(
                args.embedding_dim, num_heads=4, hidden_size=args.embedding_dim)
        if args.SG == 'att3':
            self.fusion = AttentionFusion3(args.embedding_dim)
        if args.SG == 'auto':
            self.fusion = AutoFusion(args.embedding_dim)
        if args.SG == 'att4':
            self.fusion = AttentionFusion4(args.way*args.shot)
        if args.SG == 'att5':
            self.fusion = AttentionFusion5(args.embedding_dim)
        if args.SG == 'att6':
            self.fusion = AttentionFusionTransformer2(
                args.embedding_dim, num_heads=4, hidden_size=args.embedding_dim)
        if args.SG == 'att7':
            self.fusion = AttentionFusionTransformer3(
                args.embedding_dim, num_heads=4, hidden_size=args.embedding_dim)
        if args.SG == 'EM':
            self.fusion = EM_fusion(
                args, args.embedding_dim)

    def forward(self, feature_1, feature_2):
        """Encaminha as entradas para o método de fusão selecionado."""
        return self.fusion(feature_1, feature_2)