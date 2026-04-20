import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """LayerNorm but with optional bias. Pytorch doesn't support simply bias=False"""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class ECoGEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_mix = nn.Linear(config.in_channels, config.emb_dim, bias=config.bias)
        self.layer_norm1 = LayerNorm(config.emb_dim, bias=config.bias)
        self.conv = nn.Sequential(
            # stride 2 gets you 800 → 400, close to target
            nn.Conv1d(config.emb_dim, config.emb_dim, 
                      kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            # stride 1: refine features without further downsampling
            nn.Conv1d(config.emb_dim, config.out_channels, 
                      kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )

        self.config = config
        # init all weights
        self.apply(self._init_weights)

        print(f"number of parameters: %.2fM" % (self.get_num_params()/1e6))

    def forward(self, x):
        x = self.linear_mix(x.permute(0, 2, 1))  # (B, n_channels, T) → (B, T, emb_dim)
        x = self.layer_norm1(x)                   # (B, T, emb_dim)
        x = x.permute(0, 2, 1).contiguous()                    # (B, emb_dim, T)
        x = self.conv(x)                    # (B, emb_dim, 400)
        x = F.interpolate(x, size=int(x.shape[-1]*2*self.config.fs_spec/self.config.fs_eeg),      # (B, emb_dim, 344)
                mode='linear',
                align_corners=False)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # PyTorch default (Kaiming uniform) is good for ReLU
        # For GELU, Kaiming is still fine — but you can be explicit:
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def get_num_params(self):
        """Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params