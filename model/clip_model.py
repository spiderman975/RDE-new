""" CLIP Model
Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
from collections import OrderedDict
import logging
import math
import os
from typing import List, Tuple, Union
import hashlib
import urllib
from tqdm import tqdm
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


logger = logging.getLogger("IRRA.model")

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}

def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())

def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        # self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.positional_embedding = nn.Parameter(torch.randn((spacial_dim[0] * spacial_dim[1]) + 1, embed_dim)/ embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        spacial_dim = (
            input_resolution[0] // 32,
            input_resolution[1] // 32,
        )
        self.attnpool = AttentionPool2d(spacial_dim, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)

    def forward(self, inputs):
        # x = x + self.attention(self.ln_1(x))
        x = inputs[0]
        atten, atten_weight = self.attention(self.ln_1(x))
        x = x + atten
        x = x + self.mlp(self.ln_2(x))
        return [x, atten_weight]


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], patch_size: int, stride_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution # (384, 128)
        self.num_x = (input_resolution[1] - patch_size) // stride_size + 1
        self.num_y = (input_resolution[0] - patch_size) // stride_size + 1
        num_patches = self.num_x * self.num_y

        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=stride_size, bias=False)

        scale = width ** -0.5 # 1/sqrt(768)
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_patches + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))


    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        outputs = self.transformer([x])
        x = outputs[0]
        atten = outputs[1]
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x = self.ln_post(x[:, 0, :])
        x = self.ln_post(x)

        if self.proj is not None:
            x = x @ self.proj
    
        return x, atten

class Transformer_3D(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, max_frames=-1):
        return self.resblocks((x, max_frames))[0]


class VisualTransformer(nn.Module):
    def __init__(self,
                 input_resolution: Tuple[int, int],
                 patch_size: int,
                 stride_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 output_dim: int,
                 linear_patch: str = '3d'):
        super().__init__()
        self.input_resolution = input_resolution  # (H, W)
        self.output_dim = output_dim
        self.linear_patch = linear_patch

        # 根据 patch 类型构建卷积层
        if self.linear_patch == '2d':
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
                                   kernel_size=patch_size, stride=stride_size, bias=False)
            # 根据 2d patch 计算网格数
            self.num_x = (input_resolution[1] - patch_size) // stride_size + 1
            self.num_y = (input_resolution[0] - patch_size) // stride_size + 1
            num_patches = self.num_x * self.num_y
        elif self.linear_patch == '3d':
            # 保留 conv1 以便于 2d 情况下使用，同时定义 conv2 用于 3d patch 提取
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
                                   kernel_size=patch_size, stride=stride_size, bias=False)
            self.conv2 = nn.Conv3d(in_channels=3, out_channels=width,
                                   kernel_size=(3, patch_size, patch_size),
                                   stride=(1, patch_size, patch_size),
                                   padding=(1, 0, 0), bias=False)
            # 计算 spatial grid 数量
            self.num_y = (input_resolution[0] - patch_size) // patch_size + 1
            self.num_x = (input_resolution[1] - patch_size) // patch_size + 1
            num_patches = self.num_y * self.num_x
        else:
            raise ValueError("linear_patch 参数仅支持 '2d' 或 '3d'")

        # 初始化 class token 与 positional embedding（patch 数+1）
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_patches + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, max_frames: int = -1):
        # 如果选择 3d patch，则对输入进行 reshape 和 conv2 操作
        if self.linear_patch == '3d':
            assert max_frames != -1, "当 linear_patch 为 '3d' 时，必须提供 max_frames 参数"
            # 假设输入 x 原本 shape 为 [B * max_frames, C, H, W]，先 reshape 成 [B, max_frames, C, H, W]
            x_3d = x.reshape(-1, max_frames, x.shape[-3], x.shape[-2], x.shape[-1])
            # 将通道移到第二维：[B, C, max_frames, H, W]
            x_3d = x_3d.permute(0, 2, 1, 3, 4)
            # 为确保数据类型匹配，将输入转换为 float，再调用 conv2
            x_3d = x_3d.float()
            x_3d = self.conv2(x_3d)
            x_3d = x_3d.to(x.dtype)
            # 调整维度为 [B, max_frames, width, grid, grid]
            x_3d = x_3d.permute(0, 2, 1, 3, 4)
            # 再 reshape 回 2d 格式：[B * max_frames, width, grid, grid]
            x = x_3d.reshape(-1, x_3d.shape[-3], x_3d.shape[-2], x_3d.shape[-1]).contiguous()
        else:
            # 2d 情况直接通过 conv1 提取 patch 特征
            x = self.conv1(x)  # shape = [B, width, grid, grid]

        # 将卷积输出 reshape 为 patch 序列：[B, width, num_patches] -> [B, num_patches, width]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        # 生成 class token，并拼接到 patch 序列前面
        class_token = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                        dtype=x.dtype, device=x.device)
        x = torch.cat([class_token, x], dim=1)  # shape = [B, num_patches + 1, width]
        # 加上 positional embedding
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # 将序列维度移到第一维供 transformer 处理：[B, L, width] -> [L, B, width]
        x = x.permute(1, 0, 2)
        # 调用 transformer
        out = self.transformer(x)
        if isinstance(out, (tuple, list)):
            x, atten = out[0], out[1]
        else:
            x, atten = out, None

        # 如果 transformer 返回的是稀疏张量，则先转换为 dense
        if x.is_sparse:
            x = x.to_dense()

        # 根据 transformer 输出的维度进行后续处理
        if x.dim() == 3:
            # 输出形状为 [L, B, width]，转换为 [B, L, width]后取 CLS token（第 0 个 token）
            x = x.permute(1, 0, 2)
            x = self.ln_post(x[:, 0, :])
        elif x.dim() == 2:
            # 输出形状为 [B, width]，直接处理
            x = self.ln_post(x)
        else:
            raise ValueError("Unexpected tensor dimension after transformer")

        if self.proj is not None:
            x = x @ self.proj

        return x, atten



class VisualTransformer_0(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 linear_patch: str = '3d'):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        #num_patches = (input_resolution // patch_size) ** 2
        if isinstance(input_resolution, tuple):
            h, w = input_resolution
        else:
            h = w = input_resolution
        self.num_y = h // patch_size  # 新增
        self.num_x = w // patch_size  # 新增
        num_patches = self.num_y * self.num_x  # 修改
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_patches + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # For 3D
        assert linear_patch in ['2d', '3d']
        self.linear_patch = linear_patch
        if self.linear_patch == '3d':
            self.conv2 = nn.Conv3d(in_channels=3, out_channels=width, kernel_size=(3, patch_size, patch_size),
                                   stride=(1, patch_size, patch_size), padding=(1, 0, 0), bias=False)

    def forward(self, x: torch.Tensor, max_frames: int = -1):
        if self.linear_patch == '3d':
            assert max_frames != -1, "For 3d linear_patch, please provide video_frame number."
            x_3d = x.reshape(-1, max_frames, x.shape[-3], x.shape[-2], x.shape[-1])
            x_3d = x_3d.permute(0, 2, 1, 3, 4)
            # 修改：将 x_3d 转换为 half 精度，确保与 conv2 的权重匹配
            x_3d = x_3d.half()
            # 确保 conv2 权重也为 half（如果还没有转换）
            self.conv2.weight.data = self.conv2.weight.data.half()  # 修改
            x_3d = self.conv2(x_3d)  # shape = [*, width, frame, grid, grid]
            x_3d = x_3d.permute(0, 2, 1, 3, 4)  # shape = [*, frame, width, grid, grid]
            x = x_3d.reshape(-1, x_3d.shape[-3], x_3d.shape[-2], x_3d.shape[-1]).contiguous()  # shape = [*, width, grid, grid]
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]

        # 将卷积输出 reshape 成 patch 序列
        x = x.reshape(x.shape[0], x.shape[1], -1)   # [*, width, grid**2]
        x = x.permute(0, 2, 1)                        # [*, grid**2, width]
        # 将 class token 加入序列
        class_token = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([class_token, x], dim=1)         # [*, grid**2 + 1, width]
        # 加入位置编码
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # 交换维度以符合 transformer 输入要求：LND
        x = x.permute(1, 0, 2)
        # 获取 transformer 输出（假设返回值为 (x, atten)）
        outputs = self.transformer([x])
        x = outputs[0]
        atten = outputs[1]
        # 还原回原始维度顺序：NLD
        x = x.permute(1, 0, 2)

        # 后处理层归一化和投影
        x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj

        return x, atten



class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: Union[int, Tuple[int, int]],
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 stride_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            '''
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                stride_size=stride_size,  # 添加 stride_size 参数
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                linear_patch='3d'  # 默认为 '2d'，可根据需要设置为 '3d'
            )
            '''

            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                stride_size=stride_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        outputs = self.transformer([x])
        x = outputs[0]
        atten = outputs[1]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        x = x @ self.text_projection

        return x,atten

    def forward(self, image, text):
        image_features,atten_i = self.encode_image(image)
        text_features,atten_t = self.encode_text(text)

        # # normalized features
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_image = logit_scale * image_features @ text_features.t()
        # logits_per_text = logits_per_image.t()

        # # shape = [global_batch_size, global_batch_size]
        # return logits_per_image, logits_per_text

        return image_features,atten_i, text_features,atten_t
    
    
    def load_param(self, state_dict):
        # 将pretrained_dict里不属于model_dict的键剔除掉
        param_dict =  {k: v for k, v in state_dict.items() if k in self.state_dict()}

        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if k == 'visual.positional_embedding' and v.shape != self.visual.positional_embedding.shape:
                v = resize_pos_embed(v, self.visual.positional_embedding, self.visual.num_y, self.visual.num_x)
            elif k == 'positional_embedding' and v.shape != self.positional_embedding.shape:
                v = resize_text_pos_embed(v, self.context_length)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print(f'===========================ERROR occur in copy {k}, {v.shape}=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
    
class VideoCLIP_0(CLIP):
    """
    VideoCLIP 在原 CLIP 基础上增加时序模块，
    用于处理视频帧序列输入（形状 [B, T, C, H, W]）。
    具体步骤：
      1. 将视频帧展开为图像批次 [B*T, C, H, W]，逐帧通过视觉编码器得到特征；
      2. 重塑为 [B, T, d_model]；
      3. 在帧序列前加上一个可学习的 CLS token，形成 [B, T+1, d_model]；
      4. 转置为 [T+1, B, d_model]，送入时序 Transformer；
      5. 取输出序列中第一个位置（CLS token 对应的输出）作为视频全局特征。
    """
    def __init__(self, *args, temporal_layers=2, **kwargs):
        super().__init__(*args, **kwargs)
        # d_model 取自视觉编码器投影层输出维度（若不存在则取 visual.output_dim）
        d_model = self.visual.proj.shape[1] if self.visual.proj is not None else self.visual.output_dim
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8),
            num_layers=temporal_layers
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
    def encode_video(self, video: torch.Tensor):
        """
        video: [B, T, C, H, W]
        """
        B, T, C, H, W = video.shape
        video_reshaped = video.view(B * T, C, H, W)
        features, _ = self.visual(video_reshaped.type(self.dtype))
        d_model = features.shape[-1]
        features = features.view(B, T, d_model)  # [B, T, d_model]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        features = torch.cat([cls_tokens, features], dim=1)  # [B, T+1, d_model]
        temporal_input = features.permute(1, 0, 2)  # [T+1, B, d_model]
        temporal_output = self.temporal_transformer(temporal_input)
        video_feat = temporal_output[0]  # CLS token输出, shape [B, d_model]
        return video_feat, temporal_output
    def forward(self, video, text):
        video_features, atten_video = self.encode_video(video)
        text_features, atten_text = self.encode_text(text)
        return video_features, atten_video, text_features, atten_text




def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb = posemb.unsqueeze(0)
    posemb_new = posemb_new.unsqueeze(0)

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb.squeeze(0)


# 自定义一个强制使用 FP32 的 TransformerEncoderLayer
class FP32TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, **kwargs):
        super().__init__(d_model, nhead, **kwargs)
        # 强制将本层所有参数转换为 FP32
        for p in self.parameters():
            p.data = p.data.float()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 将输入转换为 FP32
        src = src.float()
        return super().forward(src, src_mask, src_key_padding_mask)


# 在 VideoCLIP 中使用自定义 FP32TransformerEncoderLayer
class VideoCLIP(CLIP):
    """
    VideoCLIP 在原 CLIP 基础上增加时序模块，
    用于处理视频帧序列输入（形状 [B, T, C, H, W]）。
    具体步骤：
      1. 将视频帧展开为图像批次 [B*T, C, H, W]，逐帧通过视觉编码器得到特征；
      2. 如果视觉编码器返回多 token 特征，则取第 0 个 token 作为每帧全局特征；
      3. 重塑为 [B, T, d_model]；
      4. 在帧序列前加上一个可学习的 CLS token，形成 [B, T+1, d_model]；
      5. 转置为 [T+1, B, d_model]，并显式转换为 FP32；
      6. 送入时序 Transformer，输出的 CLS token 即为视频全局特征。
    """
    def __init__(self, *args, temporal_layers=2, **kwargs):
        super().__init__(*args, **kwargs)
        # d_model 取自视觉编码器投影层输出维度（若不存在则取 visual.output_dim）
        d_model = self.visual.proj.shape[1] if self.visual.proj is not None else self.visual.output_dim
        # 使用自定义 FP32TransformerEncoderLayer
        encoder_layer = FP32TransformerEncoderLayer(d_model=d_model, nhead=8)
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=temporal_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

    def encode_video(self, video: torch.Tensor):
        """
        video: [B, T, C, H, W]
        """
        B, T, C, H, W = video.shape
        video_reshaped = video.view(B * T, C, H, W)
        # 视觉编码器使用 half 输入，因为其权重可能为 half
        features, _ = self.visual(video_reshaped.half())
        features = features.float()  # 转换为 FP32
        # 如果返回的是 [B*T, L, d_model]，取第 0 个 token
        if features.dim() == 3:
            features = features[:, 0, :]
        d_model = features.shape[-1]
        features = features.view(B, T, d_model)  # [B, T, d_model]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        features = torch.cat([cls_tokens, features], dim=1)  # [B, T+1, d_model]
        temporal_input = features.permute(1, 0, 2).float()  # [T+1, B, d_model]
        temporal_output = self.temporal_transformer(temporal_input)
        video_feat = temporal_output[0].float()  # CLS token输出, [B, d_model]
        return video_feat, temporal_output

    def forward(self, video, text):
        video_features, atten_video = self.encode_video(video)
        text_features, atten_text = self.encode_text(text)
        return video_features, atten_video, text_features, atten_text


class VideoCLIP_3D(CLIP):
    """
    VideoCLIP_Avg_3D 在原 CLIP 基础上使用 VisualTransformer 的 3D 分支，
    对视频进行时空特征提取，然后沿时间维度进行平均池化得到全局视频特征。

    具体步骤：
      1. 将视频直接传入视觉编码器（VisualTransformer）处理，启用 3D 分支（video_frame=T）；
      2. 如果视觉编码器返回多 token 特征，则取第 0 个 token（CLS token）作为视频整体表示；
      3. （如果输出仍包含时序维度，则沿时间维度平均池化）得到全局视频特征 [B, d_model]；
      4. 返回全局视频特征和 None 作为视觉注意力占位符。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 从模型配置中获取 max_frames，若不存在则使用默认值12
        self.max_frames = kwargs.get('max_frames', 12)

    def encode_video(self, video: torch.Tensor):
        # video: [B, T, C, H, W]
        B, T, C, H, W = video.shape
        # 直接调用视觉编码器的 3D 分支，需要传入 video_frame 参数
        features, _ = self.visual(video.half(), max_frames=self.max_frames)
        features = features.float()
        # 如果视觉编码器返回的是 [B, L, d_model]，且 L > 1，假设第 0 个 token 表示全局特征
        if features.dim() == 3:
            features = features[:, 0, :]  # 取 CLS token 作为全局表示
        # 此时 features 的形状应为 [B, d_model]，若仍有时序维度，则可以做平均池化
        # 例如（可选）：video_feat = features.mean(dim=1)
        # 但通常 3D 分支已经在卷积层中融合了时间信息，因此直接使用 CLS token 即可
        video_feat = features
        return video_feat, None

    def forward(self, video, text):
        video_features, _ = self.encode_video(video)
        text_features, atten_text = self.encode_text(text)
        return video_features, None, text_features, atten_text


class VideoCLIP_Avg(CLIP):
    """
    VideoCLIP_Avg 在原 CLIP 基础上不使用时序 Transformer，
    而采用简单的平均池化对视频帧特征进行聚合。

    具体步骤：
      1. 将视频帧展开为图像批次 [B*T, C, H, W]，逐帧通过视觉编码器得到特征；
      2. 如果视觉编码器返回多 token 特征，则取第 0 个 token 作为每帧全局特征；
      3. 重塑为 [B, T, d_model]；
      4. 沿时间维度对每个视频的帧特征做平均池化，得到视频全局特征 [B, d_model]；
      5. 返回时，额外返回一个 None 作为 atten_v。
    """

    def encode_video(self, video: torch.Tensor):
        # video: [B, T, C, H, W]
        B, T, C, H, W = video.shape
        video_reshaped = video.view(B * T, C, H, W)
        # 将视频帧转换为 half 送入视觉编码器，然后转换为 FP32
        features, _ = self.visual(video_reshaped.half())
        features = features.float()
        # 如果视觉编码器返回 [B*T, L, d_model]，取第 0 个 token
        if features.dim() == 3:
            features = features[:, 0, :]  # [B*T, d_model]
        d_model = features.shape[-1]
        # 重塑为 [B, T, d_model]
        features = features.view(B, T, d_model)
        # 采用平均池化，得到全局视频特征 [B, d_model]
        video_feat = features.mean(dim=1)
        # 返回 video_feat 及一个占位的 None 用于 atten_v
        return video_feat, None

    def forward(self, video, text):
        video_features, _ = self.encode_video(video)
        text_features, atten_text = self.encode_text(text)
        return video_features, None, text_features, atten_text


class VideoCLIP_Avg_mask(CLIP):
    """
    VideoCLIP_Avg 在原 CLIP 基础上不使用时序 Transformer，
    而采用简单的平均池化对视频帧特征进行聚合。

    具体步骤：
      1. 将视频帧展开为图像批次 [B*T, C, H, W]，逐帧通过视觉编码器得到特征；
      2. 如果视觉编码器返回多 token 特征，则取第 0 个 token 作为每帧全局特征；
      3. 重塑为 [B, T, d_model]；
      4. 如果提供了有效性掩码，则按真实帧进行加权平均池化，否则直接对所有帧做平均池化，
         得到视频全局特征 [B, d_model]；
      5. 返回时，额外返回一个 None 作为 atten_v。
    """

    def encode_video(self, video: torch.Tensor, video_mask: torch.Tensor = None):
        # video: [B, T, C, H, W]
        B, T, C, H, W = video.shape
        video_reshaped = video.view(B * T, C, H, W)
        # 将视频帧转换为 half 后送入视觉编码器，然后转换为 FP32
        features, _ = self.visual(video_reshaped.half())
        features = features.float()
        # 如果视觉编码器返回 [B*T, L, d_model]，取第 0 个 token 作为每帧特征
        if features.dim() == 3:
            features = features[:, 0, :]  # [B*T, d_model]
        d_model = features.shape[-1]
        # 重塑为 [B, T, d_model]
        features = features.view(B, T, d_model)

        if video_mask is not None:
            # video_mask: [B, T]，1 表示真实帧，0 表示补齐帧
            mask = video_mask.unsqueeze(-1).float()  # [B, T, 1]
            # 加权求和后除以真实帧数
            features_sum = (features * mask).sum(dim=1)
            valid_counts = mask.sum(dim=1).clamp(min=1.0)
            video_feat = features_sum / valid_counts
        else:
            video_feat = features.mean(dim=1)

        return video_feat, None

    def forward(self, video, text, video_mask=None):
        # video_mask 为可选参数，如果上游（例如 collate 函数）返回了有效性掩码，则传入
        video_features, _ = self.encode_video(video, video_mask)
        text_features, atten_text = self.encode_text(text)
        return video_features, None, text_features, atten_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16, but skip transformer modules."""
    def _convert_weights_to_fp16(l):
        # 如果是 TransformerEncoderLayer 则跳过
        if isinstance(l, nn.TransformerEncoderLayer):
            return
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                         "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj", "mcq_proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()
    model.apply(_convert_weights_to_fp16)


def build_CLIP_from_openai_pretrained(name: str, image_size: Union[int, Tuple[int, int]], stride_size: int, jit: bool = False, download_root: str = None):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    
    image_size: Union[int, Tuple[int, int]]
        Input image size, in Re-ID task, image size commonly set to 384x128, instead of 224x224

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu")
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    state_dict = state_dict or model.state_dict()

    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model_cfg = {
        'embed_dim': embed_dim,
        'image_resolution': image_resolution,
        'vision_layers': vision_layers, 
        'vision_width': vision_width, 
        'vision_patch_size': vision_patch_size,
        'context_length': context_length, 
        'vocab_size': vocab_size, 
        'transformer_width': transformer_width, 
        'transformer_heads': transformer_heads, 
        'transformer_layers': transformer_layers
    }


    # modify image resolution to adapt Re-ID task
    model_cfg['image_resolution'] = image_size
    model_cfg['stride_size'] = stride_size
    logger.info(f"Load pretrained {name} CLIP model with model config: {model_cfg}")
    #model = VideoCLIP_Avg(**model_cfg)
    model = VideoCLIP_Avg_mask(**model_cfg)

    # covert model to fp16
    # convert_weights(model)

    # resize modified pos embedding
    model.load_param(state_dict)
    return model, model_cfg


