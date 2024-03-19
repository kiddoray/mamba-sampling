from mamba_ssm.modules.mamba_simple import Mamba
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from functools import partial
from .block import Block
import math

import torch, math
from apes.models.utils import ops
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            with torch.no_grad():
                p /= math.sqrt(n_residuals_per_layer * n_layer)



def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        hidden_states = hidden_states
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states = self.drop_out_in_block(hidden_states)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.K = 32
        self.group_type = 'center_diff'
        self.conv1 = nn.Sequential(nn.Conv2d(6, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))

    def forward(self, x):
        x_list = []
        x = ops.group(x, self.K, self.group_type)  # (B, C=3, N) -> (B, C=6, N, K)
        x = self.conv1(x)  # (B, C=6, N, K) -> (B, C=128, N, K)
        x = self.conv2(x)  # (B, C=128, N, K) -> (B, C=64, N, K)
        x = x.max(dim=-1, keepdim=False)[0]  # (B, C=64, N, K) -> (B, C=64, N)
        x_list.append(x)
        x = ops.group(x, self.K, self.group_type)  # (B, C=64, N) -> (B, C=128, N, K)
        x = self.conv3(x)  # (B, C=128, N, K) -> (B, C=128, N, K)
        x = self.conv4(x)  # (B, C=128, N, K) -> (B, C=64, N, K)
        x = x.max(dim=-1, keepdim=False)[0]  # (B, C=64, N, K) -> (B, C=64, N)
        x_list.append(x)
        x = torch.cat(x_list, dim=1)  # (B, C=128, N)
        return x


class N2PAttention(nn.Module):
    def __init__(self):
        super(N2PAttention, self).__init__()
        self.heads = 4
        self.K = 32
        self.group_type = 'diff'
        self.q_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.k_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.v_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Conv1d(128, 512, 1, bias=False), nn.LeakyReLU(0.2), nn.Conv1d(512, 128, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.mamba = MixerModel(d_model=128, 
                                n_layer=1,
                                rms_norm=False,
                                drop_out_in_block=0.,
                                drop_path=0.)

    def forward(self, x):
        neighbors = ops.group(x, self.K, self.group_type)  # (B, C, N) -> (B, C, N, K)
        q = self.q_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()  # (B, C, N) -> (B, C, N, 1)
        q = self.split_heads(q, self.heads)  # (B, C, N, 1) -> (B, H, N, 1, D)
        k = self.k_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        k = self.split_heads(k, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        v = self.v_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        v = self.split_heads(v, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        energy = q @ rearrange(k, 'B H N K D -> B H N D K').contiguous()  # (B, H, N, 1, D) @ (B, H, N, D, K) -> (B, H, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # (B, H, N, 1, K) -> (B, H, N, 1, K)
        tmp = rearrange(attention@v, 'B H N 1 D -> B (H D) N').contiguous()  # (B, H, N, 1, K) @ (B, H, N, K, D) -> (B, H, N, 1, D) -> (B, C=H*D, N)
        
        x = self.bn1(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        tmp = self.ff(x)  # (B, C, N) -> (B, C, N)
        x = self.bn2(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        # print('shape x:', x.shape)

        # mamba 
        x = rearrange(x, 'B C N-> B N C').contiguous()
        x = self.mamba(x)
        x = rearrange(x, 'B N C -> B C N').contiguous()

        return x

    @staticmethod
    def split_heads(x, heads):
        x = rearrange(x, 'B (H D) N K -> B H N K D', H=heads).contiguous()  # (B, C, N, K) -> (B, H, N, K, D)
        return x


class GlobalDownSample(nn.Module):
    def __init__(self, npts_ds):
        super(GlobalDownSample, self).__init__()
        self.npts_ds = npts_ds
        self.q_conv = nn.Conv1d(128, 128, 1, bias=False)
        self.k_conv = nn.Conv1d(128, 128, 1, bias=False)
        self.v_conv = nn.Conv1d(128, 128, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.mamba = MixerModel(d_model=128, 
                                n_layer=1,
                                rms_norm=False,
                                drop_out_in_block=0.,
                                drop_path=0.)

    def forward(self, x):
        q = self.q_conv(x)  # (B, C, N) -> (B, C, N)
        k = self.k_conv(x)  # (B, C, N) -> (B, C, N)
        v = self.v_conv(x)  # (B, C, N) -> (B, C, N)
        energy = rearrange(q, 'B C N -> B N C').contiguous() @ k  # (B, N, C) @ (B, C, N) -> (B, N, N)
        scale_factor = math.sqrt(q.shape[-2])
        attention = self.softmax(energy / scale_factor)  # (B, N, N) -> (B, N, N)
        selection = torch.sum(attention, dim=-2)  # (B, N, N) -> (B, N)
        self.idx = selection.topk(self.npts_ds, dim=-1)[1]  # (B, N) -> (B, M)
        scores = torch.gather(attention, dim=1, index=repeat(self.idx, 'B M -> B M N', N=attention.shape[-1]))  # (B, N, N) -> (B, M, N)
        v = scores @ rearrange(v, 'B C N -> B N C').contiguous()  # (B, M, N) @ (B, N, C) -> (B, M, C)
        
        # mamba
        v = self.mamba(v)

        out = rearrange(v, 'B M C -> B C M').contiguous()  # (B, M, C) -> (B, C, M)
        return out


class LocalDownSample(nn.Module):
    def __init__(self, npts_ds):
        super(LocalDownSample, self).__init__()
        self.npts_ds = npts_ds  # number of downsampled points
        self.K = 32  # number of neighbors
        self.group_type = 'diff'
        self.q_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.k_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.v_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.mamba = MixerModel(d_model=128, 
                                n_layer=1,
                                rms_norm=False,
                                drop_out_in_block=0.,
                                drop_path=0.)

    def forward(self, x):
        neighbors = ops.group(x, self.K, self.group_type)  # (B, C, N) -> (B, C, N, K)
        q = self.q_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()  # (B, C, N) -> (B, C, N, 1)
        q = rearrange(q, 'B C N 1 -> B N 1 C').contiguous()  # (B, C, N, 1) -> (B, N, 1, C)
        k = self.k_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        k = rearrange(k, 'B C N K -> B N C K').contiguous()  # (B, C, N, K) -> (B, N, C, K)
        v = self.v_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        v = rearrange(v, 'B C N K -> B N K C').contiguous()  # (B, C, N, K) -> (B, N, K, C)
        energy = q @ k  # (B, N, 1, C) @ (B, N, C, K) -> (B, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # (B, N, 1, K) -> (B, N, 1, K)
        selection = rearrange(torch.std(attention, dim=-1, unbiased=False), 'B N 1 -> B N').contiguous()  # (B, N, 1, K) -> (B, N, 1) -> (B, N)
        self.idx = selection.topk(self.npts_ds, dim=-1)[1]  # (B, N) -> (B, M)
        scores = torch.gather(attention, dim=1, index=repeat(self.idx, 'B M -> B M 1 K', K=attention.shape[-1]))  # (B, N, 1, K) -> (B, M, 1, K)
        v = torch.gather(v, dim=1, index=repeat(self.idx, 'B M -> B M K C', K=v.shape[-2], C=v.shape[-1]))  # (B, N, K, C) -> (B, M, K, C)
        out = rearrange(scores@v, 'B M 1 C -> B C M').contiguous()  # (B, M, 1, K) @ (B, M, K, C) -> (B, M, 1, C) -> (B, C, M)

        out = rearrange(out, 'B C M-> B M C').contiguous()
        out = self.mamba(out)
        out = rearrange(out, 'B M C -> B C M').contiguous()
        
        return out


class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()
        self.q_conv = nn.Conv1d(128, 128, 1, bias=False)
        self.k_conv = nn.Conv1d(128, 128, 1, bias=False)
        self.v_conv = nn.Conv1d(128, 128, 1, bias=False)
        self.skip_link = nn.Conv1d(128, 128, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, pcd_up, pcd_down):
        q = self.q_conv(pcd_up)  # (B, C, N) -> (B, C, N)
        k = self.k_conv(pcd_down)  # (B, C, M) -> (B, C, M)
        v = self.v_conv(pcd_down)  # (B, C, M) -> (B, C, M)
        energy = rearrange(q, 'B C N -> B N C').contiguous() @ k  # (B, N, C) @ (B, C, M) -> (B, N, M)
        scale_factor = math.sqrt(q.shape[-2])
        attention = self.softmax(energy / scale_factor)  # (B, N, M) -> (B, N, M)
        x = attention @ rearrange(v, 'B C M -> B M C').contiguous()  # (B, N, M) @ (B, M, C) -> (B, N, C)
        x = rearrange(x, 'B N C -> B C N').contiguous()  # (B, N, C) -> (B, C, N)
        x = self.skip_link(pcd_up) + x  # (B, C, N) + (B, C, N) -> (B, C, N)
        return x
