import torch
import torch.nn as nn


class ConvTransformerNet(nn.Module):
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        embed_dim: int = 128,  # token embedding size
        depth: int = 5,  # number of Transformer layers
        n_heads: int = 8,  # attention heads
        ff_dim: int = 128,  # feed-forward hidden dim
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_positions = num_rows * num_cols

        # 1) Conv-stem for local 3x3 features + residual 1x1
        self.stem = nn.Sequential(
            nn.Conv2d(2, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )
        self.stem_residual = nn.Conv2d(2, embed_dim, kernel_size=1)

        # learnable [CLS] token for value head
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # positional embeddings for each board cell + [CLS]
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_positions + 1, embed_dim))

        # relative-position bias table
        coords = torch.stack(
            torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij"), dim=-1
        )  # H×W×2
        coords_flat = coords.view(-1, 2)
        rel = coords_flat[:, None, :] - coords_flat[None, :, :]  # P×P×2
        idx = (rel[..., 0] + num_rows - 1) * (2 * num_cols - 1) + (rel[..., 1] + num_cols - 1)
        self.rel_pos_bias = nn.Parameter(torch.zeros((2 * num_rows - 1) * (2 * num_cols - 1), n_heads))
        self.register_buffer("rel_pos_index", idx)

        # Transformer encoder (PreNorm)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # use Pre-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Head dropouts
        self.policy_dropout = nn.Dropout(0.1)
        self.value_dropout = nn.Dropout(0.1)

        # Policy head: one logit per column
        self.policy_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
        )

        # Value head: direct scalar regression in [-1,1]
        self.value_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x, legal_mask=None):
        batch_size = x.size(0)

        # 1) Local feature projection via conv-stem + residual
        h0 = self.stem_residual(x)
        h1 = self.stem(x)
        h = h0 + h1  # (B, embed_dim, H, W)

        # 2) Flatten to tokens: (B, P, D)
        h = h.flatten(2).transpose(1, 2)

        # 3) Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, D)
        h = torch.cat([cls, h], dim=1)  # (B, P+1, D)

        # 4) Add positional embeddings
        h = h + self.pos_embed

        # 5) Inject relative-position bias into each layer's self-attn
        #    initial bias: (P, P, heads) → permute → (heads, P, P)
        bias = self.rel_pos_bias[self.rel_pos_index]  # (P, P, H)
        bias = bias.permute(2, 0, 1)  # (H, P, P)
        # pad columns (prepend a zero-col for CLS)
        cls_pad_col = torch.zeros((bias.size(0), self.num_positions, 1), device=bias.device)
        bias = torch.cat([cls_pad_col, bias], dim=2)  # (H, P, P+1)
        # pad rows (prepend a zero-row for CLS)
        cls_pad_row = torch.zeros((bias.size(0), 1, bias.size(2)), device=bias.device)
        bias = torch.cat([cls_pad_row, bias], dim=1)  # (H, P+1, P+1)
        # assign to each layer
        for layer in self.transformer.layers:
            layer.self_attn.bias = bias

        # 6) Transformer encoder
        h = self.transformer(h)  # (B, P+1, D)

        # 7) Policy: average tokens per column + dropout + head
        tokens = h[:, 1:, :].view(batch_size, self.num_rows, self.num_cols, -1)
        policy_tokens = tokens.mean(dim=1)  # (B, num_cols, D)
        logits = self.policy_head(self.policy_dropout(policy_tokens)).squeeze(-1)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float("-inf"))
        policy = torch.nn.functional.softmax(logits, dim=-1)

        # 8) Value: use CLS, dropout, head
        cls_out = h[:, 0, :]
        value = self.value_head(self.value_dropout(cls_out)).squeeze(-1)

        return policy, value
