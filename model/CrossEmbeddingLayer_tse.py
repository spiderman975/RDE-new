import torch
import torch.nn as nn
import torch.nn.functional as F
 
def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    if (norm < eps).any():
        print("Warning: norm has very small values:", norm.min().item())
    X = torch.div(X, norm)
    return X


def maxk_pool1d_var(x, dim, k, lengths):
    """https://github.com/woodfrog/vse_infty, thanks!"""
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        k = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results

def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)

def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN) from https://github.com/woodfrog/vse_infty, thanks!"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):
        B, N, D = x.size()
        x = x.reshape(B * N, D)
        for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        x = x.view(B, N, self.output_dim)
        return x
 
class TexualEmbeddingLayer(nn.Module):
    def __init__(self, input_dim=512, embed_dim=1024, ratio=0.3):
        super(TexualEmbeddingLayer, self).__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(input_dim, embed_dim)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
        self.ratio = ratio

    def forward(self, features, text, atten):
        """
        features: (bs, seq_len, input_dim)
        text:     (bs, seq_len)
        atten:    (bs, seq_len, seq_len)
        """
        device = features.device
        bs, seq_len, _ = atten.size()

        # 1) 构造非 padding mask，并计算 token 总长度（含 SOS/EOS）
        mask = (text != 0).float()           # (bs, seq_len)
        token_lens = mask.sum(dim=1)         # (bs,)

        # 2) 计算 top-K 参数，至少为 1
        k = max(1, int((seq_len - 2) * self.ratio))

        # 3) 定位 EOS token
        eos_pos = (token_lens - 1).long().clamp(min=0, max=seq_len - 1)  # (bs,)

        # 4) 替换 attention 中的 NaN
        if torch.isnan(atten).any():
            print("Warning: NaN detected in atten; replacing with -1")
            atten = torch.nan_to_num(atten, nan=-1.0)

        batch_idx = torch.arange(bs, device=device)

        # 5) 屏蔽 SOS（pos=0）和 EOS 位置
        atten[batch_idx, :, 0] = -1.0
        atten[batch_idx, :, eos_pos] = -1.0
        atten[batch_idx, eos_pos, :] = -1.0

        # 6) 取出每条样本 EOS 行，并打 mask
        atten_sel = atten[batch_idx, eos_pos, :]  # (bs, seq_len)
        atten_sel = atten_sel * mask

        # 7) Top-K 选择
        topk_vals, topk_idx = atten_sel.topk(k, dim=-1)  # (bs, k)

        # 8) 聚集对应的特征
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, features.size(2))  # (bs, k, input_dim)
        feats_k = features.gather(1, idx_exp)                              # (bs, k, input_dim)

        # 9) L2 归一化，添加 ε 避免除零
        norm = feats_k.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)     # (bs, k, 1)
        feats_k = feats_k / norm                                           # (bs, k, input_dim)

        # 10) 计算池化长度，至少为 1 且不超过 k
        pool_lens = (token_lens - 2).clamp(min=1, max=k).long()            # (bs,)

        # 11) 线性映射与 MLP 融合
        cap_emb = self.linear(feats_k)       # (bs, k, embed_dim)
        feats_mlp = self.mlp(feats_k)        # (bs, k, embed_dim)
        fused = feats_mlp + cap_emb          # (bs, k, embed_dim)

        # 12) 可变长度 max-pooling
        out = maxk_pool1d_var(fused, 1, 1, pool_lens.to(device))  # (bs, embed_dim)

        # 13) 最终 NaN 检查
        if torch.isnan(out).any():
            print("Warning: NaN detected in TexualEmbeddingLayer output!")

        return out.float()



class VisualEmbeddingLayer(nn.Module):
    def __init__(self, input_dim=512, embed_dim=1024, ratio=0.3):
        super(VisualEmbeddingLayer, self).__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(input_dim, embed_dim)
        self.ratio = ratio
        self.fc = nn.Linear(input_dim, embed_dim)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)

    def forward(self, base_features, atten):
        # 检查 base_features 是否为 None
        if base_features is None:
            raise ValueError("base_features is None in VisualEmbeddingLayer.forward")
        #print("VisualEmbeddingLayer.forward - base_features.shape:", base_features.shape)
        #print("VisualEmbeddingLayer.forward - base_features.dtype:", base_features.dtype)

        # 如果 base_features 为2D，则扩展为3D：[B, D] -> [B, 1, D]
        if base_features.dim() == 2:
            base_features = base_features.unsqueeze(1)
            #print("After unsqueeze, base_features.shape:", base_features.shape)
        # base_features: [B, T, D], atten: [B, T, *] 或 None
        bs, T, D = base_features.size()

        # 如果token数不足，直接用线性变换
        if T <= 1:
            # 转换为 FP32 后再传入线性层
            features = self.linear(base_features.squeeze(1).half()).float()
            return features.float()

        # 如果没有提供注意力信息，则采用简单平均池化
        if atten is None:
            pooled = base_features.mean(dim=1)  # [B, D]

            pooled = l2norm(pooled, dim=-1)
            cap_emb = self.linear(pooled.half())
            out = self.mlp(pooled) + cap_emb
            #print("Output after average pooling - dtype:", out.dtype)
            return out.float()

        # 检查 atten 中是否含有 NaN，若有则替换
        if torch.isnan(atten).any():
            print("Warning: NaN detected in atten in VisualEmbeddingLayer; replacing with -1")
            atten = torch.nan_to_num(atten, nan=-1)

        # 原方法：排除 CLS token，对后续 token 做 top-k 选择
        #k = int((atten.size(1) - 1) * self.ratio)  # 注意这里减去 CLS token（索引0）
        # 计算 top-k 值，确保 k 至少为 1
        k = max(1, int((atten.size(1) - 1) * self.ratio))
        #print("Computed k:", k)
        # 检查 atten 的数据类型
        #print("atten dtype:", atten.dtype)

        # 将 CLS token 对应的注意力置为 -1
        atten[torch.arange(bs), :, 0] = -1
        # 选择第一个 token 位置的注意力排序结果
        atten_topK = atten[:, 0].topk(dim=-1, k=k)[1]  # 形状 [B, k]
        #print("atten_topK.shape:", atten_topK.shape)
        atten_topK = atten_topK.unsqueeze(-1).expand(bs, k, D)  # 形状 [B, k, D]
        features = torch.gather(input=base_features, dim=1, index=atten_topK)  # [B, k, D]
        features = l2norm(features, dim=-1)

        # 构造一个长度向量（这里每个样本的 token 数固定为 k）
        feat_lengths = torch.full((bs,), k, dtype=base_features.dtype, device=base_features.device)

        cap_emb = self.linear(features.half())
        features = self.mlp(features) + cap_emb
        features = maxk_pool1d_var(features, 1, 1, feat_lengths)  # max pooling

        # 检查最终输出是否含 NaN
        if torch.isnan(features).any():
            print("Warning: NaN detected in VisualEmbeddingLayer output!")
        return features.float()

