# Copyright 2025 Yilin Zhuang
# Based on work by MIT, Tsinghua University, NVIDIA CORPORATION and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 1x1 이 아닌 conv2d 는 구면(spherical) 경계를 위해 SphereConv2d 로 교체했다.

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin
from diffusers.models.activations import get_activation
from diffusers.models.autoencoders.vae import DecoderOutput, EncoderOutput
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm, get_normalization
from diffusers.utils.accelerate_utils import apply_forward_hook

from .sphere_conv import SphereConv2d


class SanaMultiscaleAttentionProjection(nn.Module):
    # qkv depthwise conv로 qkv 표현력 강화
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        kernel_size: int,
    ) -> None:
        super().__init__()

        channels = 3 * in_channels
        self.proj_in = SphereConv2d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
            padding_mode="circular",
        )
        self.proj_out = nn.Conv2d(
            channels, channels, 1, 1, 0, groups=3 * num_attention_heads, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states


class SanaMultiscaleLinearAttention(nn.Module):
    r"""경량 multi-scale linear attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_attention_heads: Optional[int] = None,
        attention_head_dim: int = 8,
        mult: float = 1.0,
        norm_type: str = "batch_norm",
        kernel_sizes: Tuple[int, ...] = (5,),
        eps: float = 1e-15,
        residual_connection: bool = False,
        temb_channels: Optional[int] = None,
    ):
        super().__init__()

        # 순환 import 방지
        from diffusers.models.normalization import get_normalization

        self.eps = eps
        self.attention_head_dim = attention_head_dim
        self.norm_type = norm_type
        self.residual_connection = residual_connection

        num_attention_heads = (
            int(in_channels // attention_head_dim * mult)
            if num_attention_heads is None
            else num_attention_heads
        )
        inner_dim = num_attention_heads * attention_head_dim

        self.to_q = nn.Linear(in_channels, inner_dim, bias=False)
        self.to_k = nn.Linear(in_channels, inner_dim, bias=False)
        self.to_v = nn.Linear(in_channels, inner_dim, bias=False)

        self.to_qkv_multiscale = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.to_qkv_multiscale.append(
                SanaMultiscaleAttentionProjection(
                    inner_dim, num_attention_heads, kernel_size
                )
            )

        self.nonlinearity = nn.ReLU()
        self.to_out = nn.Linear(
            inner_dim * (1 + len(kernel_sizes)), out_channels, bias=False
        )
        self.norm_out = get_normalization(norm_type, num_features=out_channels)

        # temb 미사용(temb_channels=None) — AdaLayerNormZeroSingle4Sana 경로는
        # models/unused_utils_from_DCAE.py 로 분리. 항상 None.
        self.time_emb_porj = None
        self.norm_in = None

        self.processor = SanaMultiscaleAttnProcessor2_0()

    def apply_linear_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor: 
        # query, key ,value는 열벡터 표기를 따름: (B, num_heads, head_dim, H*W)
        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1)  # Adds padding
        with torch.autocast(query.device.type, torch.float32):
            # overflow 방지
            scores = torch.matmul(
                value.to(torch.float32), key.transpose(-1, -2).to(torch.float32)
            )
            hidden_states = torch.matmul(
                scores.to(torch.float32), query.to(torch.float32)
            )

            hidden_states = hidden_states.to(dtype=torch.float32)
            hidden_states = hidden_states[:, :, :-1] / (
                hidden_states[:, :, -1:] + self.eps
            )
        return hidden_states

    def apply_quadratic_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        with torch.autocast(query.device.type, torch.float32):
            # overflow 방지
            scores = torch.matmul(key.transpose(-1, -2), query)
            scores = scores.to(dtype=torch.float32)
            scores = scores / (torch.sum(scores, dim=2, keepdim=True) + self.eps)
            hidden_states = torch.matmul(value, scores) # 열벡터 Notation이기 때문에 V가 앞에 온다
        return hidden_states

    def forward(
        self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.norm_in is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_porj(temb)
            hidden_states, gate_msa = self.norm_in(hidden_states, temb)
        else:
            gate_msa = None
        return self.processor(self, hidden_states, gate=gate_msa)


class SanaMultiscaleAttnProcessor2_0:
    r"""multiscale quadratic attention 프로세서."""

    def __call__(
        self,
        attn: SanaMultiscaleLinearAttention,
        hidden_states: torch.Tensor,
        gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        height, width = hidden_states.shape[-2:]
        if height * width > attn.attention_head_dim:
            use_linear_attention = True
        else:
            use_linear_attention = False

        residual = hidden_states

        batch_size, _, height, width = list(hidden_states.size())
        original_dtype = hidden_states.dtype

        hidden_states = hidden_states.movedim(1, -1) # (B, C, H, W) -> (B, H, W, C)
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        hidden_states = torch.cat([query, key, value], dim=3)
        hidden_states = hidden_states.movedim(-1, 1) # (B, H, W, 3*inner_dim) -> (B, 3*inner_dim, H, W)

        multi_scale_qkv = [hidden_states]
        for block in attn.to_qkv_multiscale:
            multi_scale_qkv.append(block(hidden_states))

        hidden_states = torch.cat(multi_scale_qkv, dim=1)

        if use_linear_attention:
            # linear attention 을 위해 hidden_states 를 float32 로 upcast
            hidden_states = hidden_states.to(dtype=torch.float32)

        hidden_states = hidden_states.reshape(
            batch_size, -1, 3 * attn.attention_head_dim, height * width
        ) # (B, num_heads, 3*head_dim, H*W) 형태로 reshape -> 열벡터 표기 유지

        query, key, value = hidden_states.chunk(3, dim=2)
        query = attn.nonlinearity(query)
        key = attn.nonlinearity(key)

        if use_linear_attention:
            hidden_states = attn.apply_linear_attention(query, key, value)
            hidden_states = hidden_states.to(dtype=original_dtype)
        else:
            hidden_states = attn.apply_quadratic_attention(query, key, value)

        hidden_states = torch.reshape(hidden_states, (batch_size, -1, height, width))
        hidden_states = attn.to_out(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if gate is not None:
            hidden_states = hidden_states * gate

        if attn.norm_type == "rms_norm":
            hidden_states = attn.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
        else:
            hidden_states = attn.norm_out(hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states


class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 4,
        norm_type: Optional[str] = None,
        residual_connection: bool = True,
    ) -> None:
        super().__init__()

        hidden_channels = int(expand_ratio * in_channels)
        self.norm_type = norm_type
        self.residual_connection = residual_connection

        self.nonlinearity = nn.SiLU()
        self.conv_inverted = nn.Conv2d(in_channels, hidden_channels * 2, 1, 1, 0)
        self.conv_depth = SphereConv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            3,
            1,
            1,
            groups=hidden_channels * 2,
            padding_mode="circular",
        )
        self.conv_point = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False)

        self.norm = None
        if norm_type == "rms_norm":
            self.norm = RMSNorm(
                out_channels, eps=1e-7, elementwise_affine=True, bias=True
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.residual_connection:
            residual = hidden_states

        hidden_states = self.conv_inverted(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv_depth(hidden_states)
        hidden_states, gate = torch.chunk(hidden_states, 2, dim=1)
        hidden_states = hidden_states * self.nonlinearity(gate)

        hidden_states = self.conv_point(hidden_states)

        if self.norm_type == "rms_norm":
            # 채널 축에 RMSNorm 을 적용하기 위해 채널을 마지막 차원으로 이동
            hidden_states = self.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "batch_norm",
        act_fn: str = "relu6",
        temb_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.norm_type = norm_type

        self.nonlinearity = (
            get_activation(act_fn) if act_fn is not None else nn.Identity()
        )
        self.conv1 = SphereConv2d(
            in_channels, in_channels, 3, 1, 1, padding_mode="circular"
        )
        self.conv2 = SphereConv2d(
            in_channels, out_channels, 3, 1, 1, bias=False, padding_mode="circular"
        )
        self.norm = get_normalization(norm_type, out_channels)

        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, 2 * out_channels)
        else:
            self.time_emb_proj = None

    def forward(
        self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.time_emb_proj is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * time_scale + time_shift

        hidden_states = self.conv2(hidden_states)

        if self.norm_type == "rms_norm":
            # 채널 축에 RMSNorm 을 적용하기 위해 채널을 마지막 차원으로 이동
            hidden_states = self.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)
        else:
            hidden_states = self.norm(hidden_states)

        return hidden_states + residual


class EfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mult: float = 1.0,
        attention_head_dim: int = 32,
        qkv_multiscales: Tuple[int, ...] = (5,),
        norm_type: str = "batch_norm",
        temb_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.attn = SanaMultiscaleLinearAttention(
            in_channels=in_channels,
            out_channels=in_channels,
            mult=mult,
            attention_head_dim=attention_head_dim,
            norm_type=norm_type,
            kernel_sizes=qkv_multiscales,
            residual_connection=True,
            temb_channels=temb_channels,
        )

        self.conv_out = GLUMBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            norm_type="rms_norm",
        )

    def forward(
        self, x: torch.Tensor, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.attn(x, temb)
        x = self.conv_out(x)
        return x


def get_block(
    block_type: str,
    in_channels: int,
    out_channels: int,
    attention_head_dim: int,
    norm_type: str,
    act_fn: str,
    qkv_mutliscales: Tuple[int] = (),
    temb_channels: Optional[int] = None,
):
    if block_type == "ResBlock":
        block = ResBlock(
            in_channels, out_channels, norm_type, act_fn, temb_channels=temb_channels
        )

    elif block_type == "EfficientViTBlock":
        block = EfficientViTBlock(
            in_channels,
            attention_head_dim=attention_head_dim,
            norm_type=norm_type,
            qkv_multiscales=qkv_mutliscales,
            temb_channels=temb_channels,
        )

    else:
        raise ValueError(f"Block with {block_type=} is not supported.")

    return block


class DCDownBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = False,
        shortcut: bool = True,
        factor: int = 2,          # 다운샘플 배율(pixel_unshuffle). 2 외 3 등 가능 → ×18=2·3·3.
    ) -> None:
        super().__init__()

        self.downsample = downsample
        self.factor = factor
        self.stride = 1 if downsample else factor
        self.group_size = in_channels * self.factor**2 // out_channels
        self.shortcut = shortcut

        out_ratio = self.factor**2
        if downsample:
            assert out_channels % out_ratio == 0
            out_channels = out_channels // out_ratio

        self.conv = SphereConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            padding_mode="circular",
        )

    def forward(self, hidden_states: torch.Tensor, temb=None) -> torch.Tensor:
        x = self.conv(hidden_states)
        if self.downsample:
            x = F.pixel_unshuffle(x, self.factor)

        if self.shortcut:
            y = F.pixel_unshuffle(hidden_states, self.factor)
            y = y.unflatten(1, (-1, self.group_size))
            y = y.mean(dim=2)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states


class DCUpBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        interpolate: bool = False,
        shortcut: bool = True,
        interpolation_mode: str = "nearest",
        factor: int = 2,          # 업샘플 배율(pixel_shuffle). 인코더 다운샘플과 대칭.
    ) -> None:
        super().__init__()

        self.interpolate = interpolate
        self.interpolation_mode = interpolation_mode
        self.shortcut = shortcut
        self.factor = factor
        self.repeats = out_channels * self.factor**2 // in_channels

        out_ratio = self.factor**2

        if not interpolate:
            out_channels = out_channels * out_ratio

        self.conv = SphereConv2d(
            in_channels, out_channels, 3, 1, 1, padding_mode="circular"
        )

    def forward(self, hidden_states: torch.Tensor, temb=None) -> torch.Tensor:
        if self.interpolate:
            x = F.interpolate(
                hidden_states, scale_factor=self.factor, mode=self.interpolation_mode
            )
            x = self.conv(x)
        else:
            x = self.conv(hidden_states)
            x = F.pixel_shuffle(x, self.factor)

        if self.shortcut:
            y = hidden_states.repeat_interleave(self.repeats, dim=1)
            y = F.pixel_shuffle(y, self.factor)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        temb_channels: Optional[int] = None,
        attention_head_dim: int = 32,
        block_type: Union[str, Tuple[str]] = "ResBlock",
        block_out_channels: Tuple[int] = (128, 256, 512, 512, 1024, 1024),
        layers_per_block: Tuple[int] = (2, 2, 2, 2, 2, 2),
        qkv_multiscales: Tuple[Tuple[int, ...], ...] = ((), (), (), (5,), (5,), (5,)),
        downsample_block_type: str = "pixel_unshuffle",
        downsample_factors: Optional[Tuple[int, ...]] = None,  # 단계별 배율, len=num_blocks-1
        out_shortcut: bool = True,
    ):
        super().__init__()

        num_blocks = len(block_out_channels)
        if downsample_factors is None:
            downsample_factors = (2,) * (num_blocks - 1)

        if isinstance(block_type, str):
            block_type = (block_type,) * num_blocks

        if layers_per_block[0] > 0:
            self.conv_in = SphereConv2d(
                in_channels,
                block_out_channels[0]
                if layers_per_block[0] > 0
                else block_out_channels[1],
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
            )
        else:
            self.conv_in = DCDownBlock2d(
                in_channels=in_channels,
                out_channels=block_out_channels[0]
                if layers_per_block[0] > 0
                else block_out_channels[1],
                downsample=downsample_block_type == "pixel_unshuffle",
                shortcut=False,
            )

        self.down_blocks = nn.ModuleList()
        for i, (out_channel, num_layers) in enumerate(
            zip(block_out_channels, layers_per_block)
        ):
            for _ in range(num_layers):
                block = get_block(
                    block_type[i],
                    out_channel,
                    out_channel,
                    attention_head_dim=attention_head_dim,
                    norm_type="rms_norm",
                    act_fn="silu",
                    qkv_mutliscales=qkv_multiscales[i],
                    temb_channels=temb_channels,
                )
                self.down_blocks.append(block)

            if i < num_blocks - 1 and num_layers > 0:
                downsample_block = DCDownBlock2d(
                    in_channels=out_channel,
                    out_channels=block_out_channels[i + 1],
                    downsample=downsample_block_type == "pixel_unshuffle",
                    shortcut=True,
                    factor=downsample_factors[i],
                )
                self.down_blocks.append(downsample_block)

        self.conv_out = SphereConv2d(
            block_out_channels[-1], latent_channels, 3, 1, 1, padding_mode="circular"
        )

        self.out_shortcut = out_shortcut
        if out_shortcut:
            self.out_shortcut_average_group_size = (
                block_out_channels[-1] // latent_channels
            )

    def forward(
        self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        for idx, down_block in enumerate(self.down_blocks):
            hidden_states = down_block(hidden_states, temb)

        if self.out_shortcut:
            x = hidden_states.unflatten(1, (-1, self.out_shortcut_average_group_size))
            x = x.mean(dim=2)
            hidden_states = self.conv_out(hidden_states) + x
        else:
            hidden_states = self.conv_out(hidden_states)

        return hidden_states


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        temb_channels: Optional[int] = None,
        attention_head_dim: int = 32,
        block_type: Union[str, Tuple[str]] = "ResBlock",
        block_out_channels: Tuple[int] = (128, 256, 512, 512, 1024, 1024),
        layers_per_block: Tuple[int] = (2, 2, 2, 2, 2, 2),
        qkv_multiscales: Tuple[Tuple[int, ...], ...] = ((), (), (), (5,), (5,), (5,)),
        norm_type: Union[str, Tuple[str]] = "rms_norm",
        act_fn: Union[str, Tuple[str]] = "silu",
        upsample_block_type: str = "pixel_shuffle",
        upsample_factors: Optional[Tuple[int, ...]] = None,  # 인코더 다운샘플과 대칭, len=num_blocks-1
        in_shortcut: bool = True,
    ):
        super().__init__()

        num_blocks = len(block_out_channels)
        if upsample_factors is None:
            upsample_factors = (2,) * (num_blocks - 1)

        if isinstance(block_type, str):
            block_type = (block_type,) * num_blocks
        if isinstance(norm_type, str):
            norm_type = (norm_type,) * num_blocks
        if isinstance(act_fn, str):
            act_fn = (act_fn,) * num_blocks

        self.conv_in = SphereConv2d(
            latent_channels, block_out_channels[-1], 3, 1, 1, padding_mode="circular"
        )

        self.in_shortcut = in_shortcut
        if in_shortcut:
            self.in_shortcut_repeats = block_out_channels[-1] // latent_channels

        self.up_blocks = nn.ModuleList()
        for i, (out_channel, num_layers) in reversed(
            list(enumerate(zip(block_out_channels, layers_per_block)))
        ):
            if i < num_blocks - 1 and num_layers > 0:
                upsample_block = DCUpBlock2d(
                    block_out_channels[i + 1],
                    out_channel,
                    interpolate=upsample_block_type == "interpolate",
                    shortcut=True,
                    factor=upsample_factors[i],
                )
                self.up_blocks.append(upsample_block)

            for _ in range(num_layers):
                block = get_block(
                    block_type[i],
                    out_channel,
                    out_channel,
                    attention_head_dim=attention_head_dim,
                    norm_type=norm_type[i],
                    act_fn=act_fn[i],
                    qkv_mutliscales=qkv_multiscales[i],
                    temb_channels=temb_channels,
                )
                self.up_blocks.append(block)


        channels = (
            block_out_channels[0] if layers_per_block[0] > 0 else block_out_channels[1]
        )

        self.norm_out = RMSNorm(channels, 1e-7, elementwise_affine=True, bias=True)
        self.conv_act = nn.ReLU()
        self.conv_out = None

        if layers_per_block[0] > 0:
            self.conv_out = SphereConv2d(
                channels, out_channels, 3, 1, 1, padding_mode="circular"
            )
        else:
            self.conv_out = DCUpBlock2d(
                channels,
                out_channels,
                interpolate=upsample_block_type == "interpolate",
                shortcut=False,
            )

    def forward(
        self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.in_shortcut:
            x = hidden_states.repeat_interleave(self.in_shortcut_repeats, dim=1)
            hidden_states = self.conv_in(hidden_states) + x
        else:
            hidden_states = self.conv_in(hidden_states)

        for up_block in self.up_blocks:
            hidden_states = up_block(hidden_states, temb)

        hidden_states = self.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


class AutoencoderDC(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""DCAE(https://arxiv.org/abs/2410.10733) / SANA 에서 쓰인 AutoencoderDC.

    물리장(C,H,W)을 latent 으로 압축/복원하는 공간 autoencoder. 인자는 diffusers
    register_to_config 로 저장되어 from_pretrained 시 그대로 복원된다. 주요 인자:
    in/out_channels, latent_channels, encoder/decoder_block_types,
    encoder/decoder_block_out_channels(블록 수=공간 압축비 2^(n-1)), qkv_multiscales,
    up/downsample_block_type, static_channels(없으면 None), scaling_factor.
    [`ModelMixin`] 상속 — 다운로드/저장 등 공통 메서드는 상위 클래스 문서 참고.
    """

    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        # 기본값 = LaDcast2 configs/dcae_local.yaml 설정 (9ch ERA5, ×4 압축, latent 45×90).
        in_channels: int = 9,
        out_channels: Optional[int] = 9,
        temb_channels: Optional[int] = None,          # 시점 임베딩 미사용
        latent_channels: int = 32,
        attention_head_dim: int = 32,
        encoder_block_types: Union[str, Tuple[str]] = ("ResBlock", "ResBlock", "EfficientViTBlock"),
        decoder_block_types: Union[str, Tuple[str]] = ("ResBlock", "ResBlock", "EfficientViTBlock"),
        encoder_block_out_channels: Tuple[int, ...] = (128, 256, 512),   # 3 stage → ×4 다운샘플
        decoder_block_out_channels: Tuple[int, ...] = (128, 256, 512),
        encoder_layers_per_block: Tuple[int] = (2, 2, 2),
        decoder_layers_per_block: Tuple[int] = (2, 2, 2),
        encoder_qkv_multiscales: Tuple[Tuple[int, ...], ...] = ((), (), (5,)),
        decoder_qkv_multiscales: Tuple[Tuple[int, ...], ...] = ((), (), (5,)),
        upsample_block_type: str = "pixel_shuffle",
        downsample_block_type: str = "pixel_unshuffle",
        # 단계별 다운샘플 배율(len=stage-1). None 이면 모두 2(=2의 거듭제곱).
        # 예: 180×360 → (10,20) 은 [2,3,3](=×18). pixel_unshuffle 은 임의 정수 배율 지원.
        downsample_factors: Optional[Tuple[int, ...]] = None,
        decoder_norm_types: Union[str, Tuple[str]] = "rms_norm",   # dcae_local.yaml 미지정 → 기본 유지
        decoder_act_fns: Union[str, Tuple[str]] = "silu",          # dcae_local.yaml 미지정 → 기본 유지
        scaling_factor: float = 1.0,
        static_channels: Optional[int] = None,         # 로컬 데이터엔 static 없음 (0 아님)
    ) -> None:
        super().__init__()

        n_down = len(encoder_block_out_channels) - 1
        if downsample_factors is None:
            downsample_factors = tuple([2] * n_down)
        downsample_factors = tuple(downsample_factors)
        assert len(downsample_factors) == n_down, (
            f"downsample_factors 길이 {len(downsample_factors)} 는 stage-1={n_down} 이어야 함")

        self.encoder = Encoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            temb_channels=temb_channels,
            attention_head_dim=attention_head_dim,
            block_type=encoder_block_types,
            block_out_channels=encoder_block_out_channels,
            layers_per_block=encoder_layers_per_block,
            qkv_multiscales=encoder_qkv_multiscales,
            downsample_block_type=downsample_block_type,
            downsample_factors=downsample_factors,
        )
        self.decoder = Decoder(
            out_channels=out_channels if out_channels is not None else in_channels,
            latent_channels=latent_channels,
            temb_channels=temb_channels,
            attention_head_dim=attention_head_dim,
            block_type=decoder_block_types,
            block_out_channels=decoder_block_out_channels,
            layers_per_block=decoder_layers_per_block,
            qkv_multiscales=decoder_qkv_multiscales,
            norm_type=decoder_norm_types,
            act_fn=decoder_act_fns,
            upsample_block_type=upsample_block_type,
            upsample_factors=downsample_factors,   # 대칭
        )

        if temb_channels is not None:
            self.time_proj = Timesteps(
                num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
            )
            self.timestep_embedder = TimestepEmbedding(
                in_channels=256, time_embed_dim=temb_channels
            )
        else:
            self.time_proj = None
            self.timestep_embedder = None

        import math as _math
        self.spatial_compression_ratio = _math.prod(downsample_factors)  # ×2 가정 제거
        self.temporal_compression_ratio = 1

        self.static_channels = static_channels

    def _encode(
        self, x: torch.Tensor, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.encoder(x, temb)

    @apply_forward_hook
    def encode(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
        temb: Optional[torch.Tensor] = None,
        embedded_t: bool = False,
        static_conditioning_tensor: Optional[torch.Tensor] = None,
    ) -> Union[EncoderOutput, Tuple[torch.Tensor]]:
        r"""이미지 배치를 latent 으로 인코딩.

        Args:
            x: 입력 이미지 배치.
            return_dict: True 면 EncoderOutput, 아니면 tuple 반환.
        Returns:
            인코딩된 latent (return_dict 에 따라 EncoderOutput 또는 tuple).
        """
        if not embedded_t and temb is not None:
            temb = self.time_proj(temb)
            temb = self.timestep_embedder(temb)

        if static_conditioning_tensor is not None:
            x = torch.cat((x, static_conditioning_tensor), dim=1)

        encoded = self._encode(x, temb)

        if not return_dict:
            return (encoded,)
        return EncoderOutput(latent=encoded)

    def _decode(
        self, z: torch.Tensor, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.decoder(z, temb)

    @apply_forward_hook
    def decode(
        self,
        z: torch.Tensor,
        return_dict: bool = True,
        temb: Optional[torch.Tensor] = None,
        embedded_t: bool = False,
        return_static=False,
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        r"""latent 배치를 이미지로 디코딩.

        Args:
            z: 입력 latent 벡터 배치.
            return_dict: True 면 DecoderOutput, 아니면 tuple 반환.
        Returns:
            디코딩된 이미지 (return_dict 에 따라 DecoderOutput 또는 tuple).
        """
        if not embedded_t and temb is not None:
            temb = self.time_proj(temb)
            temb = self.timestep_embedder(temb)

        decoded = self._decode(z, temb)

        if not return_static:
            if self.static_channels is not None:
                decoded = decoded[:, : -self.static_channels, :, :]

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        return_dict: bool = True,
        time_elapsed: Optional[torch.Tensor] = None,
        static_conditioning_tensor: Optional[torch.Tensor] = None,
        return_static: bool = False,
    ) -> torch.Tensor:
        if time_elapsed is not None:
            temb = self.time_proj(time_elapsed)
            temb = self.timestep_embedder(temb)
        else:
            temb = None
        encoded = self.encode(
            sample,
            return_dict=False,
            temb=temb,
            embedded_t=True,
            static_conditioning_tensor=static_conditioning_tensor,
        )[0]
        decoded = self.decode(
            encoded,
            return_dict=False,
            temb=temb,
            embedded_t=True,
            return_static=return_static,
        )[0]
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)
