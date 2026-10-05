"""Diffusion transformer implemented manually, as an educational exercise.


Conceptual view of DiT:
    DiT
    ├── PatchEmbed
    ├── timestep/class conditioning
    ├── DiTBlock × depth
    └── FinalLayer + unpatchify

Conceptual view of DiT Block
  x
   └─ LayerNorm
       └─ MHA
           ├─ QKV projections
           ├─ attention within each head
           └─ output projection  ← part of MHA
   └─ residual add
   └─ LayerNorm
       └─ MLP / FFN             ← separate sublayer
   └─ residual add

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


from .factory import register_model


# implementing positional embedding...
def pos_embedding(T, D):
    assert D%2==0, f"dim must be even"
    pos = torch.arange(T, dtype=torch.float32)[:, None]
    freq_values = 10000**(-torch.arange(D//2)*2/D)[None,:]
    angles = pos*freq_values

    sines = torch.sin(angles)
    cosines = torch.cos(angles)

    pos_embedding = torch.concat([sines, cosines], dim=-1)
    return pos_embedding[None, ...]

def pos_embedding_2d(H_p, W_p, D):

    assert D%4==0, f"dim must be divisible by 4"
    row_embeddings = pos_embedding(H_p, D//2).squeeze(dim=0)
    col_embeddings = pos_embedding(W_p, D//2).squeeze(dim=0)

    row_embeddings = row_embeddings[:, None].expand(H_p, W_p, D//2)
    col_embeddings = col_embeddings[None,:].expand(H_p, W_p, D//2)

    pos_emb = torch.concat([row_embeddings, col_embeddings], dim=-1)
    return pos_emb.reshape(H_p*W_p, D).unsqueeze(dim=0)

    
# implementing positional embedding...

def time_embedding(t, emb_dim, emb_scale=1000):
    """Sinsoidal embedding for conditioning imput...
    For conditioning on 0.0 < t < 1.0, use emb_scale=1000, to get meaningful variation between different values
    For conditioning on 0.0 < sigma < 150.0, or 0.0 < t < 999.0, use emb_scale=1

    Args:
        t: (B, )
        emb_dim: int
        emb_scale: float (1 or 1000) 
    """
    # (B, )
    assert emb_dim%2==0, f"dim must be even"
    freq_values = 10000**(-torch.arange(emb_dim//2, device=t.device)*2/emb_dim)[None,:]          # (1, D//2)
    # (B, 1)* (1, D//2) --> (B, D//2)
    angles = t[:,None]*freq_values

    sines = torch.sin(emb_scale*angles)
    cosines = torch.cos(emb_scale*angles)
    return torch.concat([sines, cosines], dim=-1)
    


class MHA(nn.Module):
    def __init__(self, dim, n_heads, attn_dropout_p=0.0, bias=False):
        super().__init__()
        assert dim % n_heads ==0, "d_model must be integer multiple of n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = self.dim//self.n_heads
        self.attn_dropout_p = attn_dropout_p

        self.qkv = nn.Linear(self.dim, 3*self.dim, bias=bias)
        self.output_proj = nn.Linear(self.dim, self.dim, bias=bias)

    def forward(self, x):
        B, T, D = x.shape

        # (B, T, D)  --> (B, T, 3*D)
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # (B, T, D) --> # (B, T, H, dim) --> # (B, H, T, dim)
        (q, k, v) = (z.reshape(B, T, self.n_heads, self.head_dim).transpose(1,2) for z in (q, k, v))

        attn = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.attn_dropout_p if self.training else 0.0
            )
        attn = attn.transpose(1,2).reshape(B, T, self.dim)
        out = self.output_proj(attn)
        return out



class MLP(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()

        self.linear1 = nn.Linear(dim, 4*dim, bias=bias)
        self.linear2 = nn.Linear(4*dim, dim, bias=bias)
        self.gelu = nn.GELU(approximate='tanh')

    def forward(self, x):
        # (B,T,D) --> (B,T,4*D) --> (B,T,D) 
        x = self.gelu(self.linear1(x))
        return self.linear2(x)

class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim, dim, emb_scale=1000.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.emb_scale = emb_scale

        self.emb_proj = nn.Sequential(
            nn.Linear(emb_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            
        )

    def forward(self, t):
        # (B, )
        t_emb = time_embedding(t, self.emb_dim, self.emb_scale)
        t_emb = self.emb_proj(t_emb)
        return t_emb


class DiTBlock(nn.Module):
    def __init__(self, dim, n_heads, attn_dropout_p=0.0, bias=False):
        super().__init__()

        self.dim = dim
        self.attn = MHA(dim, n_heads, attn_dropout_p, bias=bias)
        self.mlp = MLP(dim, bias=bias)

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6*dim, bias=bias)
        )

        nn.init.zeros_(self.cond_proj[1].weight)
        if self.cond_proj[1].bias is not None:
            nn.init.zeros_(self.cond_proj[1].bias)

    def conditioning(self, cond):
        # (B, dim) --> (B, 6*dim)
        scale1, shift1, gate1, scale2, shift2, gate2 = self.cond_proj(cond).chunk(6, dim=-1)
        return (scale1.unsqueeze(dim=1), shift1.unsqueeze(dim=1), gate1.unsqueeze(dim=1), 
            scale2.unsqueeze(dim=1), shift2.unsqueeze(dim=1), gate2.unsqueeze(dim=1))

    def forward(self, x, cond):

        # x - (B, T, D)
        # c - (B, D)
        scale1, shift1, gate1, scale2, shift2, gate2 = self.conditioning(cond)   

        res1 = x
        norm1 = self.norm1(x)
        ada_norm1 = norm1*(1 + scale1) + shift1
        x1 = res1 + gate1 * self.attn(ada_norm1)

        res2 = x1
        norm2 = self.norm2(x1)
        ada_norm2 = norm2*(1 + scale2) + shift2
        out = res2 + gate2 * self.mlp(ada_norm2)
        return out

@register_model("dit")
class DiT(nn.Module):
    """
        DiT
        ├── PatchEmbed
        ├── timestep/class conditioning
        ├── DiTBlock × depth
        └── FinalLayer + unpatchify
    """
    def __init__(
        self, n_blocks, res, n_channels, patch_size, dim, n_heads,
        time_emb_dim=256, time_emb_scale=1000.0,
        n_classes=None, attn_dropout_p=0.0, bias=False
        ):
        super().__init__()
        self.dim = dim
        self.n_classes = n_classes
        self.res = res
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.H_p = self.res // patch_size
        self.W_p = self.res // patch_size    

        assert res % patch_size ==0, "image/latent shape must be multiple of patch size"


        self.register_buffer("pos_embed",pos_embedding_2d(self.H_p, self.W_p, self.dim))

        self.patch_layer = nn.Conv2d(n_channels, dim, patch_size, patch_size, bias=bias)
        # DiT matched initialization..
        w = self.patch_layer.weight
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        if self.patch_layer.bias is not None:
            nn.init.zeros_(self.patch_layer.bias)

        self.time_emb_layer = TimeEmbedding(time_emb_dim, dim, time_emb_scale)

        self.blocks = nn.ModuleList(DiTBlock(dim, n_heads, attn_dropout_p, bias) for _ in range(n_blocks))
        if self.n_classes is not None:
            self.class_embedding = nn.Embedding(n_classes, dim)

        # final normalization layer
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cond_modulation_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2*dim, bias=bias)
        )
        nn.init.zeros_(self.cond_modulation_layer[1].weight)
        if self.cond_modulation_layer[1].bias is not None:
            nn.init.zeros_(self.cond_modulation_layer[1].bias)

        # final project step
        self.final_proj = nn.Linear(dim, patch_size*patch_size*n_channels, bias=bias)
        nn.init.zeros_(self.final_proj.weight)
        if self.final_proj.bias is not None:
            nn.init.zeros_(self.final_proj.bias)
        
        


    def patchify(self, x):
        """Converts input image-like objects to tokens. Number of tokens T is
        determined by the patch size p and resolution.
        Args:
            x: (B, n_channels, res, res)

        Returns:
            tensor: (B, T, dim)
        """
        B = x.shape[0]
        x = self.patch_layer(x)             # (B, dim, H, W)
        x = x.reshape(B, self.dim, -1)      # (B, dim, T), where T=H/p*W/p
        return x.transpose(1, 2)          # (B, T, dim)

    def de_patchify(self, x):
        """Converts sequence of tokens back to image-like objects. 
        Args:
            x: (B, T, p*p*cout)

        Returns:
            tensor: (B, n_channels, res, res)
        """
        B, T, D = x.shape  
        assert T == self.H_p * self.W_p, "Number of tokens must be consistent with patch heigt and widths"                 
        #   T = H/p*W/p --> T*p*p = H*W
        # (B, T, D) --> (B, T, p*p*cout) --> (B, T*p*p, cout) --> (B, H*W, cout) 
        x = x.reshape(B, self.H_p, self.W_p, self.patch_size, self.patch_size, self.n_channels)    #(B, H_p, W_p, p, p, cout)
        x = x.permute(0, 5, 1, 3, 2, 4)                                             #(B, cout, H_p, p,  W_p, p)    
        x = x.reshape(B, self.n_channels, self.res, self.res)         #(B, H, W, cout)
        return x


    def forward(self, x, t, y=None):
        """
            x: (B, T, dim)
            t: (B, )
            y: (B, )
        """

        t_emb = self.time_emb_layer(t)                                   #(B, ) --> (B, dim)

        if y is not None:
            assert self.n_classes is not None, f"Number of classes required in the constructor to create class embeddings!"
            y_emb = self.class_embedding(y)         #(B, ) --> (B, dim)
            cond = t_emb + y_emb
        else:
            cond = t_emb

        x = self.patchify(x) + self.pos_embed                        # patchify
        for block in self.blocks:
            x = block(x, cond)

        # tokens:           (B, T, dim)
        # required output:  (B, n_channels, res, res)
        
        # final projection + cond-modulated normalization
        scale, shift = self.cond_modulation_layer(cond).unsqueeze(1).chunk(2, dim=-1)
        norm = self.final_norm(x)
        ada_norm = norm*(1 + scale) + shift
        x = self.final_proj(ada_norm)                       # (B, T, D) --> (B, T, p*p*cout)
        
        out = self.de_patchify(x)                    # de-patchify
        return out
