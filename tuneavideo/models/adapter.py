from collections import OrderedDict

from diffusers import ModelMixin
import torch
import torch.nn as nn


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
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", nn.Linear(d_model, d_model * 4)), ("gelu", QuickGELU()),
                         ("c_proj", nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class StyleAdapter(ModelMixin):

    def __init__(self, width=1024, context_dim=768, num_head=8, n_layes=3, num_token=4):
        super().__init__()

        scale = width ** -0.5
        self.transformer_layes = nn.Sequential(*[ResidualAttentionBlock(width, num_head) for _ in range(n_layes)])
        self.num_token = num_token
        self.style_embedding = nn.Parameter(torch.randn(1, num_token, width) * scale)
        self.ln_post = LayerNorm(width)
        self.ln_pre = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, context_dim))

    def forward(self, x):
        # x shape [N, HW+1, C]
        style_embedding = self.style_embedding + torch.zeros(
            (x.shape[0], self.num_token, self.style_embedding.shape[-1]), device=x.device)
        x = torch.cat([x, style_embedding], dim=1)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer_layes(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, -self.num_token:, :])
        x = x @ self.proj

        return x