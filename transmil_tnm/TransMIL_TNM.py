import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from math import ceil
from torch import einsum
from nystrom_attention import NystromAttention


def _exists(value):
    return value is not None


def _moore_penrose_iter_pinv(x, iters=6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, "... i j -> ... j i") / (torch.max(col) * torch.max(row))

    eye = torch.eye(x.shape[-1], device=device)
    eye = rearrange(eye, "i j -> () i j")

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * eye - (xz @ (15 * eye - (xz @ (7 * eye - xz)))))

    return z


def _nystrom_cls_attention(attn_module, x, cls_index=0, mask=None):
    """Return CLS-row attention that is explicitly used to compute CLS output."""
    b, n, _, h = *x.shape, attn_module.heads
    m = attn_module.num_landmarks
    iters = attn_module.pinv_iterations
    eps = attn_module.eps

    remainder = n % m
    padding = 0
    if remainder > 0:
        padding = m - remainder
        x = F.pad(x, (0, 0, padding, 0), value=0)
        if _exists(mask):
            mask = F.pad(mask, (padding, 0), value=False)

    q, k, v = attn_module.to_qkv(x).chunk(3, dim=-1)
    q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

    if _exists(mask):
        mask = rearrange(mask, "b n -> b () n")
        q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

    q = q * attn_module.scale

    landmark_group_size = ceil(n / m)
    landmark_einops_eq = "... (n l) d -> ... n d"
    q_landmarks = reduce(q, landmark_einops_eq, "sum", l=landmark_group_size)
    k_landmarks = reduce(k, landmark_einops_eq, "sum", l=landmark_group_size)

    divisor = landmark_group_size
    if _exists(mask):
        mask_landmarks_sum = reduce(mask, "... (n l) -> ... n", "sum", l=landmark_group_size)
        divisor = mask_landmarks_sum[..., None] + eps
        mask_landmarks = mask_landmarks_sum > 0

    q_landmarks = q_landmarks / divisor
    k_landmarks = k_landmarks / divisor

    einsum_eq = "... i d, ... j d -> ... i j"
    sim1 = einsum(einsum_eq, q, k_landmarks)
    sim2 = einsum(einsum_eq, q_landmarks, k_landmarks)
    sim3 = einsum(einsum_eq, q_landmarks, k)

    if _exists(mask):
        mask_value = -torch.finfo(q.dtype).max
        sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
        sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
        sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

    attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
    attn2_inv = _moore_penrose_iter_pinv(attn2, iters)

    padded_cls_index = padding + cls_index
    cls_attn = (attn1[:, :, padded_cls_index:padded_cls_index + 1, :] @ attn2_inv) @ attn3
    cls_out = cls_attn @ v

    if attn_module.residual:
        cls_out = cls_out + attn_module.res_conv(v)[:, :, padded_cls_index:padded_cls_index + 1, :]

    cls_out = rearrange(cls_out, "b h n d -> b n (h d)", h=h)
    cls_out = attn_module.to_out(cls_out)
    return cls_out, cls_attn, padding


class TransLayer(nn.Module):
    """Transformer层 - 使用NystromAttention"""
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,
            pinv_iterations = 6,
            residual = True,
            dropout=0.1
        )
        self.last_attn = None
        self.last_internal_padding = 0

    def forward(self, x, return_attn=False):
        if return_attn:
            cls_out, cls_attn, internal_padding = _nystrom_cls_attention(self.attn, self.norm(x), cls_index=0)
            self.last_attn = cls_attn
            self.last_internal_padding = internal_padding
            x = torch.cat((x[:, :1, :] + cls_out, x[:, 1:, :]), dim=1)
            return x

        self.last_attn = None
        self.last_internal_padding = 0
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    """位置编码增强模块"""
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL_TNM(nn.Module):
    """
    TransMIL-TNM: 仅用于TNM分期的TransMIL
    
    4个分类头：T、N、M、Stage
    总损失 = t_loss + n_loss + m_loss + stage_loss
    """
    def __init__(self, t_classes=4, n_classes=4, m_classes=2, stage_classes=4):
        super(TransMIL_TNM, self).__init__()
        
        # ===== 1. 共享特征提取器 (与原版TransMIL相同) =====
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        
        # ===== TNM分期类别数 =====
        self.t_classes = t_classes      # T1-T4: 4类
        self.n_classes = n_classes      # N0-N3: 4类
        self.m_classes = m_classes     # M0-M1: 2类
        self.stage_classes = stage_classes  # Stage I-IV: 4类
        
        # ===== 2. 4个独立的分类头 =====
        # 每个分类头: 512 -> 128 -> num_classes
        
        # T分期分类头
        self.t_classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, self.t_classes)
        )
        
        # N分期分类头
        self.n_classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, self.n_classes)
        )
        
        # M分期分类头
        self.m_classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, self.m_classes)
        )
        
        # Stage分期分类头
        self.stage_classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, self.stage_classes)
        )

    def forward(self, **kwargs):
        """
        前向传播
        
        Returns:
            dict: 包含4个分类头的预测结果
        """
        return_attn = bool(kwargs.get("return_attn", False) or kwargs.get("return_attention", False))
        h = kwargs['data'].float() #[B, n, 1024]
        original_patch_count = int(h.shape[1])
        
        # ===== 特征提取 (与原版相同) =====
        h = self._fc1(h) #[B, n, 512]
        
        # pad to square
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1)
        square_patch_count = int(h.shape[1])

        # cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        # Translayer x1
        h = self.layer1(h)

        # PPEG
        h = self.pos_layer(h, _H, _W)
        
        # Translayer x2
        h = self.layer2(h, return_attn=return_attn)

        # 提取[CLS]token作为全局特征
        h = self.norm(h)[:,0]  # [B, 512]

        # ===== 4个TNM分类头 =====
        
        # T分期
        t_logits = self.t_classifier(h)
        t_hat = torch.argmax(t_logits, dim=1)
        t_prob = F.softmax(t_logits, dim=1)
        
        # N分期
        n_logits = self.n_classifier(h)
        n_hat = torch.argmax(n_logits, dim=1)
        n_prob = F.softmax(n_logits, dim=1)
        
        # M分期
        m_logits = self.m_classifier(h)
        m_hat = torch.argmax(m_logits, dim=1)
        m_prob = F.softmax(m_logits, dim=1)
        
        # Stage分期
        stage_logits = self.stage_classifier(h)
        stage_hat = torch.argmax(stage_logits, dim=1)
        stage_prob = F.softmax(stage_logits, dim=1)

        results_dict = {
            # T分期
            't_logits': t_logits,
            't_prob': t_prob,
            't_hat': t_hat,
            # N分期
            'n_logits': n_logits,
            'n_prob': n_prob,
            'n_hat': n_hat,
            # M分期
            'm_logits': m_logits,
            'm_prob': m_prob,
            'm_hat': m_hat,
            # Stage分期
            'stage_logits': stage_logits,
            'stage_prob': stage_prob,
            'stage_hat': stage_hat,
        }

        if return_attn:
            results_dict.update({
                'attn2_cls': self.layer2.last_attn,
                'attn2_internal_padding': self.layer2.last_internal_padding,
                'orig_n_patches': original_patch_count,
                'square_n_patches': square_patch_count,
                'square_pad_length': int(add_length),
            })
        
        return results_dict


if __name__ == "__main__":
    # 测试
    data = torch.randn((1, 6000, 1024)).cuda()
    
    model = TransMIL_TNM(
        t_classes=4,     # T分期
        n_classes=4,     # N分期
        m_classes=2,     # M分期
        stage_classes=4  # Stage分期
    ).cuda()
    
    print("TransMIL-TNM 模型结构:")
    print(model)
    
    results = model(data=data)
    print("\n预测结果:")
    print(f"  T: {results['t_hat'].item()} (prob: {results['t_prob'].max().item():.3f})")
    print(f"  N: {results['n_hat'].item()} (prob: {results['n_prob'].max().item():.3f})")
    print(f"  M: {results['m_hat'].item()} (prob: {results['m_prob'].max().item():.3f})")
    print(f"  Stage: {results['stage_hat'].item()} (prob: {results['stage_prob'].max().item():.3f})")
