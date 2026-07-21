# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------



import numpy as np

import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, n_cls_token=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [n_cls_token+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if n_cls_token>0:
        pos_embed = np.concatenate([np.zeros([n_cls_token, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


#----------------------------------------------------------
# RoPE2D: RoPE implementation in 2D
#----------------------------------------------------------

# try:
#     from models.curope import cuRoPE2D
#     RoPE2D = cuRoPE2D
# except ImportError:
print('Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead')
class RoPE2D(torch.nn.Module):
        
        def __init__(self, freq=100.0, F0=1.0):
            super().__init__()
            self.base = freq 
            self.F0 = F0
            self.cache = {}

        def get_cos_sin(self, D, seq_len, device, dtype):
            seq_len = int(seq_len)
            D = int(D)

            if D <= 0 or D % 2 != 0:
                raise RuntimeError(f"Invalid RoPE D={D}, seq_len={seq_len}, device={device}, dtype={dtype}")

            if seq_len <= 0 or seq_len > 4097:
                raise RuntimeError(f"Invalid RoPE seq_len={seq_len}, D={D}, device={device}, dtype={dtype}")
            
            key = (D, seq_len, str(device), str(dtype))
            if key not in self.cache:
                inv_freq = 1.0 / (
                    self.base ** (torch.arange(0, D, 2, device=device).float() / D)
                )
                t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
                freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
                freqs = torch.cat((freqs, freqs), dim=-1)
                cos = freqs.cos()
                sin = freqs.sin()
                self.cache[key] = (cos, sin)

            return self.cache[key]

            # if (D,seq_len,device,dtype) not in self.cache:
            #     inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            #     t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            #     freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            #     freqs = torch.cat((freqs, freqs), dim=-1)
            #     cos = freqs.cos() # (Seq, Dim)
            #     sin = freqs.sin()
            #     self.cache[D,seq_len,device,dtype] = (cos,sin)
            # return self.cache[D,seq_len,device,dtype]
            
        @staticmethod
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
            
        def apply_rope1d(self, tokens, pos1d, cos, sin):
            assert pos1d.ndim==2
            cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
            sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
            return (tokens * cos) + (self.rotate_half(tokens) * sin)
            
        def forward(self, tokens, positions):
            """
            input:
                * tokens: batch_size x nheads x ntokens x dim
                * positions: batch_size x ntokens x 2 (y and x position of each token)
            output:
                * tokens after applying RoPE2D
            """
            if tokens.size(3) % 2 != 0:
                raise RuntimeError(
                    f"Invalid RoPE token dim: tokens_shape={tuple(tokens.shape)}"
                )

            D = tokens.size(3) // 2

            if positions.ndim != 3 or positions.shape[-1] != 2:
                raise RuntimeError(
                    f"Invalid RoPE positions shape before cast: "
                    f"positions_shape={tuple(positions.shape)}, "
                    f"tokens_shape={tuple(tokens.shape)}"
                )

            # 只在 positions 是浮点时检查 NaN/Inf
            if torch.is_floating_point(positions):
                if not bool(torch.isfinite(positions).all().item()):
                    raise RuntimeError(
                        f"Non-finite RoPE positions before cast: "
                        f"positions_shape={tuple(positions.shape)}, "
                        f"tokens_shape={tuple(tokens.shape)}, "
                        f"positions_dtype={positions.dtype}"
                    )

            positions = positions.to(device=tokens.device, dtype=torch.long).contiguous()

            if positions.shape[0] != tokens.shape[0] or positions.shape[1] != tokens.shape[2]:
                raise RuntimeError(
                    f"RoPE shape mismatch: "
                    f"positions_shape={tuple(positions.shape)}, "
                    f"tokens_shape={tuple(tokens.shape)}"
                )

            if positions.numel() == 0:
                raise RuntimeError(
                    f"Empty RoPE positions: "
                    f"positions_shape={tuple(positions.shape)}, "
                    f"tokens_shape={tuple(tokens.shape)}"
                )

            min_pos = int(positions.amin().item())
            max_pos = int(positions.amax().item())
            seq_len = max_pos + 1

            if min_pos < 0 or max_pos > 4096 or seq_len <= 0 or seq_len > 4097:
                raise RuntimeError(
                    f"Invalid RoPE positions: min={min_pos}, max={max_pos}, seq_len={seq_len}, "
                    f"positions_shape={tuple(positions.shape)}, dtype={positions.dtype}, "
                    f"tokens_shape={tuple(tokens.shape)}, tokens_dtype={tokens.dtype}, "
                    f"device={tokens.device}"
                )

            cos, sin = self.get_cos_sin(D, seq_len, tokens.device, tokens.dtype)

            y, x = tokens.chunk(2, dim=-1)
            y = self.apply_rope1d(y, positions[:, :, 0], cos, sin)
            x = self.apply_rope1d(x, positions[:, :, 1], cos, sin)
            tokens = torch.cat((y, x), dim=-1)
            return tokens


# # patch embedding
# class PositionGetter(object):
#     """ return positions of patches """

#     def __init__(self):
#         self.cache_positions = {}
        
#     def __call__(self, b, h, w, device):
#         if not (h,w) in self.cache_positions:
#             x = torch.arange(w, device=device)
#             y = torch.arange(h, device=device)
#             self.cache_positions[h,w] = torch.cartesian_prod(y, x) # (h, w, 2)
#         pos = self.cache_positions[h,w].view(1, h*w, 2).expand(b, -1, 2).clone()
#         return pos

class PositionGetter(object):
    """return positions of patches"""

    def __init__(self):
        self.cache_positions = {}

    def __call__(self, b, h, w, device):
        b = int(b)
        h = int(h)
        w = int(w)
        key = (h, w, str(device))

        if h <= 0 or w <= 0:
            raise RuntimeError(f"Invalid PositionGetter h/w: b={b}, h={h}, w={w}, device={device}")

        if key not in self.cache_positions:
            x = torch.arange(w, device=device, dtype=torch.long)
            y = torch.arange(h, device=device, dtype=torch.long)
            self.cache_positions[key] = torch.cartesian_prod(y, x).contiguous()

        pos = self.cache_positions[key].view(1, h * w, 2).expand(b, -1, 2).clone()

        if pos.shape != (b, h * w, 2):
            raise RuntimeError(
                f"Invalid PositionGetter output: "
                f"pos_shape={tuple(pos.shape)}, expected={(b, h*w, 2)}, "
                f"h={h}, w={w}, device={device}"
            )

        return pos