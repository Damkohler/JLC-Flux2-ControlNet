"""Compact Flux.2 Fun ControlNet side model.

The parameter tree deliberately matches
FLUX.2-dev-Fun-Controlnet-Union.safetensors exactly:

    control_img_in
    control_transformer_blocks.0 ... .3

This module is loaded and owned by ComfyUI through CoreModelPatcher.  The forward
path is present for later parity work, but the foundation nodes do not call it or
inject its outputs yet.
"""

from __future__ import annotations

from typing import Optional, Sequence

import logging
import torch
from torch import Tensor, nn
from comfy.ldm.flux.math import attention as comfy_flux_attention

from .constants import CONTROL_LAYERS, PROJECT_LOG_PREFIX

MODEL_ADAPTER_REVISION = "wave2-hotfix2-modulationout-native-attention"
_REVISION_LOGGED = False


class JLCFlux2SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2


class JLCFlux2FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float, *, operations, dtype, device):
        super().__init__()
        inner_dim = int(dim * mult)
        self.linear_in = operations.Linear(
            dim, inner_dim * 2, bias=False, dtype=dtype, device=device
        )
        self.act_fn = JLCFlux2SwiGLU()
        self.linear_out = operations.Linear(
            inner_dim, dim, bias=False, dtype=dtype, device=device
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_out(self.act_fn(self.linear_in(x)))


class JLCFlux2Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        eps: float,
        *,
        operations,
        dtype,
        device,
    ):
        super().__init__()
        self.heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        self.to_q = operations.Linear(dim, self.inner_dim, bias=False, dtype=dtype, device=device)
        self.to_k = operations.Linear(dim, self.inner_dim, bias=False, dtype=dtype, device=device)
        self.to_v = operations.Linear(dim, self.inner_dim, bias=False, dtype=dtype, device=device)
        self.norm_q = operations.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)

        self.to_out = nn.ModuleList(
            [
                operations.Linear(
                    self.inner_dim, dim, bias=False, dtype=dtype, device=device
                ),
                nn.Dropout(0.0),
            ]
        )

        self.add_q_proj = operations.Linear(dim, self.inner_dim, bias=False, dtype=dtype, device=device)
        self.add_k_proj = operations.Linear(dim, self.inner_dim, bias=False, dtype=dtype, device=device)
        self.add_v_proj = operations.Linear(dim, self.inner_dim, bias=False, dtype=dtype, device=device)
        self.norm_added_q = operations.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.norm_added_k = operations.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.to_add_out = operations.Linear(
            self.inner_dim, dim, bias=False, dtype=dtype, device=device
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        image_rotary_emb=None,
        attention_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        query = self.to_q(hidden_states).unflatten(-1, (self.heads, self.head_dim))
        key = self.to_k(hidden_states).unflatten(-1, (self.heads, self.head_dim))
        value = self.to_v(hidden_states).unflatten(-1, (self.heads, self.head_dim))
        query = self.norm_q(query)
        key = self.norm_k(key)

        enc_query = self.add_q_proj(encoder_hidden_states).unflatten(
            -1, (self.heads, self.head_dim)
        )
        enc_key = self.add_k_proj(encoder_hidden_states).unflatten(
            -1, (self.heads, self.head_dim)
        )
        enc_value = self.add_v_proj(encoder_hidden_states).unflatten(
            -1, (self.heads, self.head_dim)
        )
        enc_query = self.norm_added_q(enc_query)
        enc_key = self.norm_added_k(enc_key)

        query = torch.cat((enc_query, query), dim=1)
        key = torch.cat((enc_key, key), dim=1)
        value = torch.cat((enc_value, value), dim=1)

        # ComfyUI's Flux attention helper owns the native RoPE tensor format
        # and selected attention backend. Inputs are [B, H, S, D]; output is
        # [B, S, H*D].
        output = comfy_flux_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            pe=image_rotary_emb,
            mask=attention_mask,
        ).to(query.dtype)

        text_len = encoder_hidden_states.shape[1]
        text_output = self.to_add_out(output[:, :text_len])
        image_output = self.to_out[1](self.to_out[0](output[:, text_len:]))
        return image_output, text_output


class JLCFlux2ControlTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float,
        eps: float,
        block_id: int,
        *,
        operations,
        dtype,
        device,
    ):
        super().__init__()
        self.block_id = block_id

        self.norm1 = operations.LayerNorm(
            dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device
        )
        self.norm1_context = operations.LayerNorm(
            dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device
        )
        self.attn = JLCFlux2Attention(
            dim,
            num_heads,
            head_dim,
            eps,
            operations=operations,
            dtype=dtype,
            device=device,
        )
        self.norm2 = operations.LayerNorm(
            dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device
        )
        self.ff = JLCFlux2FeedForward(
            dim, mlp_ratio, operations=operations, dtype=dtype, device=device
        )
        self.norm2_context = operations.LayerNorm(
            dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device
        )
        self.ff_context = JLCFlux2FeedForward(
            dim, mlp_ratio, operations=operations, dtype=dtype, device=device
        )

        if block_id == 0:
            self.before_proj = operations.Linear(
                dim, dim, bias=True, dtype=dtype, device=device
            )
        self.after_proj = operations.Linear(
            dim, dim, bias=True, dtype=dtype, device=device
        )

    def forward(
        self,
        c: Tensor,
        x: Tensor,
        *,
        encoder_hidden_states: Tensor,
        temb_mod_params_img,
        temb_mod_params_txt,
        image_rotary_emb=None,
        attention_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        if self.block_id == 0:
            c = self.before_proj(c) + x
            accumulated = []
        else:
            accumulated = list(torch.unbind(c))
            c = accumulated.pop()

        img_mod_msa, img_mod_mlp = temb_mod_params_img
        txt_mod_msa, txt_mod_mlp = temb_mod_params_txt

        shift_msa = img_mod_msa.shift
        scale_msa = img_mod_msa.scale
        gate_msa = img_mod_msa.gate
        shift_mlp = img_mod_mlp.shift
        scale_mlp = img_mod_mlp.scale
        gate_mlp = img_mod_mlp.gate

        txt_shift_msa = txt_mod_msa.shift
        txt_scale_msa = txt_mod_msa.scale
        txt_gate_msa = txt_mod_msa.gate
        txt_shift_mlp = txt_mod_mlp.shift
        txt_scale_mlp = txt_mod_mlp.scale
        txt_gate_mlp = txt_mod_mlp.gate

        image_norm = (1 + scale_msa) * self.norm1(c) + shift_msa
        text_norm = (
            (1 + txt_scale_msa) * self.norm1_context(encoder_hidden_states)
            + txt_shift_msa
        )
        image_attn, text_attn = self.attn(
            image_norm,
            text_norm,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
        )

        c = c + gate_msa * image_attn
        c = c + gate_mlp * self.ff((1 + scale_mlp) * self.norm2(c) + shift_mlp)

        encoder_hidden_states = encoder_hidden_states + txt_gate_msa * text_attn
        encoder_hidden_states = encoder_hidden_states + txt_gate_mlp * self.ff_context(
            (1 + txt_scale_mlp) * self.norm2_context(encoder_hidden_states)
            + txt_shift_mlp
        )

        c_skip = self.after_proj(c)
        accumulated.extend((c_skip, c))
        return encoder_hidden_states, torch.stack(accumulated)


class JLCFlux2ControlModel(nn.Module):
    """The 4.116B-parameter compact Flux.2 control side branch."""

    def __init__(
        self,
        *,
        hidden_size: int,
        control_in_dim: int,
        num_blocks: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        operations,
        dtype,
        device,
    ):
        super().__init__()
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.control_in_dim = control_in_dim
        self.num_blocks = num_blocks
        self.control_layers: Sequence[int] = CONTROL_LAYERS[:num_blocks]
        self.control_layers_mapping = {
            layer: index for index, layer in enumerate(self.control_layers)
        }

        # The checkpoint contains a bias for this projection.
        self.control_img_in = operations.Linear(
            control_in_dim, hidden_size, bias=True, dtype=dtype, device=device
        )
        self.control_transformer_blocks = nn.ModuleList(
            [
                JLCFlux2ControlTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    attention_head_dim,
                    mlp_ratio,
                    eps,
                    block_id=index,
                    operations=operations,
                    dtype=dtype,
                    device=device,
                )
                for index in range(num_blocks)
            ]
        )

    def forward_control(
        self,
        *,
        x: Tensor,
        control_context: Tensor,
        encoder_hidden_states: Tensor,
        temb_mod_params_img,
        temb_mod_params_txt,
        image_rotary_emb=None,
        attention_mask: Optional[Tensor] = None,
    ) -> list[Tensor]:
        global _REVISION_LOGGED
        if not _REVISION_LOGGED:
            logging.info(
                "%s Model adapter revision: %s.",
                PROJECT_LOG_PREFIX,
                MODEL_ADAPTER_REVISION,
            )
            _REVISION_LOGGED = True
        """Generate residual hints.

        Present for the later tensor-parity wave.  Foundation integration does
        not invoke this method.
        """
        c = self.control_img_in(control_context)
        kwargs = {
            "x": x,
            "encoder_hidden_states": encoder_hidden_states,
            "temb_mod_params_img": temb_mod_params_img,
            "temb_mod_params_txt": temb_mod_params_txt,
            "image_rotary_emb": image_rotary_emb,
            "attention_mask": attention_mask,
        }
        for block in self.control_transformer_blocks:
            encoder_hidden_states, c = block(c, **kwargs)
            kwargs["encoder_hidden_states"] = encoder_hidden_states
        return list(torch.unbind(c))[:-1]
