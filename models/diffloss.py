"""
per-token diffusion head (s2s_instruction.md §A1, §52).

MAR(`models/diffloss.py`)의 구조를 이식한다:
  - 작은 MLP, residual block, AdaLN 으로 조건 z_i + diffusion step 주입.
  - **모든 토큰에 완전 공유**되는 단일 head.
마스크된 토큰에만 적용되며(§3-주의3), 마스킹+점진 언마스킹 프레임워크 안에서만 의미가 있다.

노이즈 스케줄은 EDM(Karras 2022, §D1) 전용이다. (구 DDPM 경로는 제거됨.)
"""

import math
from typing import Optional

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────
#  네트워크 (SimpleMLPAdaLN) — MAR 와 동일
# ──────────────────────────────────────────────────────────────────────────
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """diffusion step(scalar) -> 벡터 임베딩."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class ResBlock(nn.Module):
    """AdaLN-modulated MLP residual block (width 고정)."""

    def __init__(self, channels: int):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift, scale)
        h = self.mlp(h)
        return x + gate * h


class FinalLayer(nn.Module):
    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class SimpleMLPAdaLN(nn.Module):
    """token 노이즈 epsilon 을 예측하는 공유 MLP (z_i + step 조건, AdaLN)."""

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        z_channels: int,
        num_res_blocks: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.res_blocks = nn.ModuleList(
            [ResBlock(model_channels) for _ in range(num_res_blocks)]
        )
        self.final_layer = FinalLayer(model_channels, out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_basic)
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """x: (N, in), t: (N,), c: (N, z) -> (N, out)."""
        x = self.input_proj(x)
        cond = self.time_embed(t) + self.cond_embed(c)
        for block in self.res_blocks:
            x = block(x, cond)
        return self.final_layer(x, cond)


# ──────────────────────────────────────────────────────────────────────────
#  EDM (Karras 2022) — per-token diffusion (§D1, §8-3: 마스킹과 세트).
# ──────────────────────────────────────────────────────────────────────────
class EDMDiffusion:
    """per-token EDM. preconditioning(c_skip/c_out/c_in/c_noise), lognormal σ 샘플링,
    손실 가중 (σ²+σ_data²)/(σ·σ_data)². LaDCast2 EDM 루프와 동일한 정식.

    ⚠️ σ_data 정렬(§8-5): latent 표준화 std 와 σ_data 가 불일치하면 안 된다.
       dataset.target_std 로 latent std 를 σ_data 에 맞추거나, 여기 sigma_data 를 조정.
    """

    def __init__(
        self,
        sigma_data: float = 0.5,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        num_steps: int = 32,
    ):
        self.sigma_data = sigma_data
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.num_steps = num_steps

    def _precond(self, sigma):
        sd = self.sigma_data
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / torch.sqrt(sigma ** 2 + sd ** 2)
        c_in = 1.0 / torch.sqrt(sigma ** 2 + sd ** 2)
        c_noise = sigma.log() / 4.0
        return c_skip, c_out, c_in, c_noise

    def _denoise(self, net, x, sigma, c):
        """D(x;σ,c) = c_skip·x + c_out·F(c_in·x, c_noise, c)."""
        c_skip, c_out, c_in, c_noise = self._precond(sigma)
        F = net(c_in * x, c_noise.reshape(-1), c)
        return c_skip * x + c_out * F

    def training_losses(self, net, x_start, c):
        """x_start:(N,D), c:(N,z). 반환 (N,) per-token 가중 손실."""
        rnd = torch.randn(x_start.shape[0], 1, device=x_start.device)
        sigma = (rnd * self.P_std + self.P_mean).exp()       # lognormal σ
        n = torch.randn_like(x_start) * sigma
        D = self._denoise(net, x_start + n, sigma, c)
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        return (weight * (D - x_start) ** 2).mean(dim=-1)

    def _sigma_schedule(self, device, num_steps=None):
        ns = num_steps or self.num_steps
        i = torch.arange(ns, device=device)
        smin, smax, rho = self.sigma_min, self.sigma_max, self.rho
        sig = (smax ** (1 / rho) + i / (ns - 1) *
               (smin ** (1 / rho) - smax ** (1 / rho))) ** rho
        return torch.cat([sig, torch.zeros(1, device=device)])  # σ_N=0

    @torch.no_grad()
    def p_sample_loop(self, net, shape, c, temperature: float = 1.0, device="cuda",
                      c_uncond=None, cfg_scale: float = 1.0, num_steps=None):
        """2차 Heun 샘플러. temperature 는 초기 노이즈 스케일(§54 τ). num_steps 로 스텝 수 override.
        c_uncond+cfg_scale≠1 이면 BC guidance 를 denoised D 에 적용:
            D = D_unc + w·(D_cnd − D_unc)."""
        use_cfg = c_uncond is not None and cfg_scale != 1.0
        ns = num_steps or self.num_steps

        def denoise(x, sig):
            D = self._denoise(net, x, sig, c)
            if use_cfg:
                D_u = self._denoise(net, x, sig, c_uncond)
                D = D_u + cfg_scale * (D - D_u)
            return D

        sigmas = self._sigma_schedule(device, num_steps=ns)
        x = torch.randn(shape, device=device) * sigmas[0] * temperature
        for i in range(ns):
            s_cur, s_next = sigmas[i], sigmas[i + 1]
            sig = torch.full((shape[0], 1), float(s_cur), device=device)
            d_cur = (x - denoise(x, sig)) / s_cur
            x_next = x + (s_next - s_cur) * d_cur
            if s_next > 0:                                    # Heun 2차 보정
                sig2 = torch.full((shape[0], 1), float(s_next), device=device)
                d_next = (x_next - denoise(x_next, sig2)) / s_next
                x_next = x + (s_next - s_cur) * 0.5 * (d_cur + d_next)
            x = x_next
        return x


# ──────────────────────────────────────────────────────────────────────────
#  DiffLoss wrapper (MAR 와 동일 인터페이스)
# ──────────────────────────────────────────────────────────────────────────
class DiffLoss(nn.Module):
    def __init__(
        self,
        target_channels: int,
        z_channels: int,
        width: int = 2048,
        depth: int = 6,
        num_sampling_timesteps: Optional[int] = None,
        sigma_data: float = 0.5,
        diffusion_batch_mul: int = 4,     # MAR: 토큰당 noise/step 샘플 수 ↑ → diffusion loss 분산 ↓
    ):
        super().__init__()
        self.in_channels = target_channels
        self.diffusion_batch_mul = diffusion_batch_mul
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,
            z_channels=z_channels,
            num_res_blocks=depth,
        )
        steps = num_sampling_timesteps or 32
        self.train_diffusion = EDMDiffusion(sigma_data=sigma_data, num_steps=steps)
        self.gen_diffusion = self.train_diffusion

    def forward(self, target, z):
        """target/z: (N, D)/(N, z). 반환: scalar (mean per-token EDM loss)."""
        # MAR diffusion_batch_mul: 각 토큰을 mul 배 복제 → 서로 다른 noise/σ 로 학습신호 ↑.
        #   복제본마다 training_losses 안에서 독립 noise/σ 가 샘플링됨(분산 감소).
        mul = self.diffusion_batch_mul
        if mul > 1:
            target = target.repeat(mul, 1)
            z = z.repeat(mul, 1)
        loss = self.train_diffusion.training_losses(self.net, target, z)
        return loss.mean()

    @torch.no_grad()
    def sample(self, z, temperature: float = 1.0, z_uncond=None, cfg_scale: float = 1.0, num_steps=None):
        """z: (N, z) -> (N, D) 샘플 토큰. num_steps 로 샘플 스텝 수 override(학습 맥락 생성 비용↓).
        z_uncond(uncond 조건)+cfg_scale≠1 이면 BC classifier-free guidance 적용."""
        shape = (z.shape[0], self.in_channels)
        return self.gen_diffusion.p_sample_loop(
            self.net, shape, z, temperature=temperature, device=z.device,
            c_uncond=z_uncond, cfg_scale=cfg_scale, num_steps=num_steps,
        )
