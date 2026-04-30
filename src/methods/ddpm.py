"""
Denoising Diffusion Probabilistic Models (DDPM)
"""

import math
from typing import Dict, Tuple, Optional, Literal, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class DDPM(BaseMethod):
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor
    posterior_variance: torch.Tensor

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
    ):
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Fixed DDPM noise schedule. These tensors are buffers: they are not
        # optimized, but they are saved in state_dict and should move with the
        # method. Index t always means the coefficient for timestep t.
        betas = torch.linspace(
            beta_start,
            beta_end,
            self.num_timesteps,
            device=device,
            dtype=torch.float32,
        )

        # beta_t: noise variance added in the forward process at timestep t.
        # alpha_t = 1 - beta_t: signal retained by one forward noising step.
        # alpha_bar_t: cumulative signal retained from x_0 to x_t.
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        # alpha_bars_prev[t] represents alpha_bar_{t-1}. For t = 0, the
        # previous cumulative product is defined as 1 because no noising step
        # has happened before x_0.
        alpha_bars_prev = torch.cat(
            [
                torch.ones(1, device=device, dtype=torch.float32),
                alpha_bars[:-1],
            ]
        )

        # Posterior variance beta_tilde_t for q(x_{t-1} | x_t, x_0).
        # This is the variance term used by the stochastic reverse step.
        posterior_variance = (
            betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        # Forward noising coefficients:
        # x_t = sqrt(alpha_bar_t) * x_0
        #       + sqrt(1 - alpha_bar_t) * epsilon.
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer(
            "sqrt_one_minus_alpha_bars",
            torch.sqrt(1.0 - alpha_bars),
        )
        self.register_buffer("posterior_variance", posterior_variance)

    # =========================================================================
    # You can add, delete or modify as many functions as you would like
    # =========================================================================
    
    # Pro tips: If you have a lot of pseudo parameters that you will specify for each
    # model run but will be fixed once you specified them (say in your config),
    # then you can use super().register_buffer(...) for these parameters

    # Pro tips 2: If you need a specific broadcasting for your tensors,
    # it's a good idea to write a general helper function for that

    def _extract(
        self,
        values: torch.Tensor,
        t: torch.Tensor,
        x_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Select timestep-specific coefficients and reshape them for broadcasting.

        Args:
            values: Full schedule tensor with shape (num_timesteps,).
            t: Batch timesteps with shape (batch_size,).
            x_shape: Shape of the target tensor, usually (B, C, H, W).

        Returns:
            Coefficients with shape (B, 1, 1, 1) for image tensors, or the
            equivalent broadcastable shape for other x-like tensors.
        """
        if t.ndim != 1:
            raise ValueError(f"Expected t to have shape (batch_size,), got {t.shape}")

        t = t.to(device=values.device, dtype=torch.long)
        out = values.gather(0, t)

        # Keep the batch dimension and add singleton dimensions so the selected
        # schedule values broadcast across channels, height, and width.
        broadcast_shape = (t.shape[0],) + (1,) * (len(x_shape) - 1)
        return out.reshape(broadcast_shape)
    
    # =========================================================================
    # Forward process
    # =========================================================================

    def forward_process(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x_0.shape[0] != t.shape[0]:
            raise ValueError(
                "Batch size of x_0 and t must match, "
                f"got {x_0.shape[0]} and {t.shape[0]}"
            )
        
        noise = torch.randn_like(x_0)
        sqrt_alpha_bar_t = self._extract(
            self.sqrt_alpha_bars,
            t,
            x_0.shape,
        )
        sqrt_one_minus_alpha_bar_t = self._extract(
            self.sqrt_one_minus_alpha_bars,
            t,
            x_0.shape,
        )

        x_t = (
            sqrt_alpha_bar_t * x_0
            + sqrt_one_minus_alpha_bar_t * noise
        )
        return x_t, noise


    # =========================================================================
    # Training loss
    # =========================================================================

    def compute_loss(self, x_0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        TODO: Implement your DDPM loss function here

        Args:
            x_0: Clean data samples of shape (batch_size, channels, height, width)
            **kwargs: Additional method-specific arguments
        
        Returns:
            loss: Scalar loss tensor for backpropagation
            metrics: Dictionary of metrics for logging (e.g., {'mse': 0.1})
        """

        raise NotImplementedError

    # =========================================================================
    # Reverse process (sampling)
    # =========================================================================
    
    @torch.no_grad()
    def reverse_process(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        TODO: Implement one step of the DDPM reverse process

        Args:
            x_t: Noisy samples at time t (batch_size, channels, height, width)
            t: the time
            **kwargs: Additional method-specific arguments
        
        Returns:
            x_prev: Noisy samples at time t-1 (batch_size, channels, height, width)
        """
        raise NotImplementedError

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        # TODO: add your arguments here
        **kwargs
    ) -> torch.Tensor:
        """
        TODO: Implement DDPM sampling loop: start from pure noise, iterate through all the time steps using reverse_process()

        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of each image (channels, height, width)
            **kwargs: Additional method-specific arguments (e.g., num_steps)
        
        Returns:
            samples: Generated samples of shape (batch_size, *image_shape)
        """
        self.eval_mode()
        raise NotImplementedError

    # =========================================================================
    # Device / state
    # =========================================================================

    def to(self, device: torch.device) -> "DDPM":
        super().to(device)
        self.device = device
        return self

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["num_timesteps"] = self.num_timesteps
        # TODO: add other things you want to save
        return state

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "DDPM":
        ddpm_config = config.get("ddpm", config)
        return cls(
            model=model,
            device=device,
            num_timesteps=ddpm_config["num_timesteps"],
            beta_start=ddpm_config["beta_start"],
            beta_end=ddpm_config["beta_end"],
            # TODO: add your parameters here
        )
