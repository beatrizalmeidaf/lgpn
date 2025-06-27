import argparse
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class EM_fusion(nn.Module):
    def __init__(self, args, embedding_dim=768):
        super().__init__()
        self.args = args
        self.eps = 1e-6
        self.register_buffer('eye', torch.eye(embedding_dim))
        
    def em_fusion(self, sample, label):
        # 合并当前样本和对应标签 [2, dim]
        combined = torch.stack([sample, label])  

        # 初始化均值 (样本和标签的均值)
        mean = 0.5 * (sample + label)
        
        # 固定协方差（单样本无法估计协方差，使用单位矩阵）
        cov = torch.eye(sample.size(0), device=sample.device)
        
        # 单次EM迭代（单样本无需多次迭代）
        diff = combined - mean
        log_resp = -0.5 * diff.pow(2).sum(dim=1, keepdim=True)  # [2,1]
        resp = (log_resp - torch.logsumexp(log_resp, dim=0)).exp()
        
        # 返回样本权重（resp[0]是样本的权重）
        return resp[0].item()  # 返回标量值

    def forward(self, feature_1, feature_2):

        weights = torch.zeros(feature_1.size(0), device=feature_1.device)
        fused = torch.zeros_like(feature_1, device=feature_1.device)
        
        # 逐个样本处理
        for i in range(feature_1.size(0)):
            # 获取当前样本和对应标签
            sample = feature_1[i]  # [dim]
            label = feature_2[i]   # [dim]
            
            # 计算当前样本的融合权重
            weight = self.em_fusion(sample, label)
            weights[i] = weight
            
            # 执行融合
            fused[i] = weight * sample + (1 - weight) * label
        
        return fused