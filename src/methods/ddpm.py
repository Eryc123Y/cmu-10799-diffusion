"""
Denoising Diffusion Probabilistic Models (DDPM)
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class DDPM(BaseMethod):
    VALID_PREDICTION_TYPES = {"epsilon", "x0", "v", "score"}

    # These attributes are created by register_buffer in __init__. The type
    # annotations make them visible to static checkers such as Pylance/Pyright.
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
        prediction_type: str = "epsilon",
    ):
        """
        Build the fixed DDPM schedules used by training and sampling.

        Args:
            model: Noise prediction network epsilon_theta(x_t, t).
            device: Device where schedule buffers should be initialized.
            num_timesteps: Number of discrete diffusion steps T.
            beta_start: Initial forward-process noise variance.
            beta_end: Final forward-process noise variance.
            prediction_type: What the model is trained to predict. Supported
                values are "epsilon", "x0", "v", and "score".
        """
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.prediction_type = prediction_type

        if self.prediction_type not in self.VALID_PREDICTION_TYPES:
            valid_types = ", ".join(sorted(self.VALID_PREDICTION_TYPES))
            raise ValueError(
                f"Unknown DDPM prediction_type={prediction_type!r}. "
                f"Expected one of: {valid_types}."
            )

        # Fixed DDPM noise schedule. Each tensor has shape (T,), where index t
        # stores the scalar coefficient for timestep t. Buffers are not
        # optimized, but they are part of the module state.
        betas = torch.linspace(
            beta_start,
            beta_end,
            self.num_timesteps,
            device=device,
            dtype=torch.float32,
        )

        # beta_t: variance of the Gaussian noise added at step t.
        # alpha_t = 1 - beta_t: signal retained by one forward step.
        # alpha_bar_t = product_{s<=t} alpha_s: signal retained from x_0 to x_t.
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
        # Forward-process coefficients used by q(x_t | x_0):
        # x_t = sqrt(alpha_bar_t) * x_0
        #       + sqrt(1 - alpha_bar_t) * epsilon.
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer(
            "sqrt_one_minus_alpha_bars",
            torch.sqrt(1.0 - alpha_bars),
        )
        self.register_buffer("posterior_variance", posterior_variance)

    # =========================================================================
    # Schedule helpers
    # =========================================================================

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
            raise ValueError(
                f"Expected t to have shape (batch_size,), got {t.shape}"
            )

        t = t.to(device=values.device, dtype=torch.long)
        out = values.gather(0, t)

        # Keep the batch dimension and add singleton dimensions so the selected
        # schedule values broadcast across channels, height, and width.
        broadcast_shape = (t.shape[0],) + (1,) * (len(x_shape) - 1)
        return out.reshape(broadcast_shape)

    def _alpha_bar_terms(
        self,
        t: torch.Tensor,
        x_shape: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return sqrt(alpha_bar_t) and sqrt(1 - alpha_bar_t) for x-shaped tensors.

        These two coefficients are used by all parameterizations:
        x_t = sqrt(alpha_bar_t) * x_0
              + sqrt(1 - alpha_bar_t) * epsilon.
        """
        sqrt_alpha_bar_t = self._extract(self.sqrt_alpha_bars, t, x_shape)
        sqrt_one_minus_alpha_bar_t = self._extract(
            self.sqrt_one_minus_alpha_bars,
            t,
            x_shape,
        )
        return sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t

    def _get_training_target(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the supervised target for the configured parameterization.

        Args:
            x_0: Clean images with shape (B, C, H, W).
            x_t: Noisy images with shape (B, C, H, W).
            noise: The Gaussian epsilon used to construct x_t.
            t: One timestep per image with shape (B,).

        Returns:
            The target tensor with the same shape as x_0.
        """
        del x_t  # The target formulas use x_0, epsilon, and schedule terms.

        sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t = self._alpha_bar_terms(
            t,
            x_0.shape,
        )

        if self.prediction_type == "epsilon":
            return noise
        if self.prediction_type == "x0":
            return x_0
        if self.prediction_type == "v":
            # Velocity parameterization:
            # v_t = sqrt(alpha_bar_t) * epsilon
            #       - sqrt(1 - alpha_bar_t) * x_0.
            return sqrt_alpha_bar_t * noise - sqrt_one_minus_alpha_bar_t * x_0
        if self.prediction_type == "score":
            # Score of q(x_t | x_0) with respect to x_t:
            # grad log q(x_t | x_0) = -epsilon / sqrt(1 - alpha_bar_t).
            return -noise / sqrt_one_minus_alpha_bar_t

        raise RuntimeError(f"Unhandled prediction_type={self.prediction_type!r}")

    def _prediction_to_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert the raw model output into an epsilon prediction.

        Sampling equations are easiest to keep correct if every
        parameterization is first translated back to epsilon.
        """
        sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t = self._alpha_bar_terms(
            t,
            x_t.shape,
        )

        if self.prediction_type == "epsilon":
            return model_output
        if self.prediction_type == "x0":
            return (x_t - sqrt_alpha_bar_t * model_output) / sqrt_one_minus_alpha_bar_t
        if self.prediction_type == "v":
            return sqrt_one_minus_alpha_bar_t * x_t + sqrt_alpha_bar_t * model_output
        if self.prediction_type == "score":
            return -sqrt_one_minus_alpha_bar_t * model_output

        raise RuntimeError(f"Unhandled prediction_type={self.prediction_type!r}")

    def _prediction_to_x0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert the raw model output into a clean-image x_0 prediction.
        """
        sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t = self._alpha_bar_terms(
            t,
            x_t.shape,
        )

        if self.prediction_type == "x0":
            return model_output

        pred_noise = self._prediction_to_noise(x_t, t, model_output)
        return (x_t - sqrt_one_minus_alpha_bar_t * pred_noise) / sqrt_alpha_bar_t
    
    # =========================================================================
    # Forward process
    # =========================================================================

    def forward_process(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t directly from q(x_t | x_0) for a batch of timesteps.

        Args:
            x_0: Clean images with shape (B, C, H, W).
            t: One timestep per image with shape (B,).

        Returns:
            x_t: Noisy images with the same shape as x_0.
            noise: The sampled epsilon target used to create x_t.
        """
        if x_0.shape[0] != t.shape[0]:
            raise ValueError(
                "Batch size of x_0 and t must match, "
                f"got {x_0.shape[0]} and {t.shape[0]}"
            )

        # epsilon ~ N(0, I), sampled with the same shape, dtype, and device as x_0.
        noise = torch.randn_like(x_0)

        # Extract per-sample coefficients and reshape them to (B, 1, 1, 1), so
        # each image uses its own timestep while broadcasting across C, H, and W.
        sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t = self._alpha_bar_terms(
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

    def compute_loss(
        self,
        x_0: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Train the configured parameterization target from noisy images.

        Args:
            x_0: Clean image batch with shape (B, C, H, W).
            **kwargs: Reserved for future method-specific arguments.
        
        Returns:
            loss: Scalar MSE loss tensor for backpropagation.
            metrics: Float metrics for logging.
        """
        batch_size = x_0.shape[0]

        # Sample one timestep per image, so the model learns every noise level.
        t = torch.randint(
            low=0,
            high=self.num_timesteps,
            size=(batch_size,),
            device=x_0.device,
            dtype=torch.long,
        )

        # Build the supervised denoising pair. The model always sees x_t and t,
        # while the target can be epsilon, x_0, velocity, or score.
        x_t, noise = self.forward_process(x_0, t)
        target = self._get_training_target(x_0, x_t, noise, t)
        model_output = self.model(x_t, t)

        if model_output.shape != target.shape:
            raise ValueError(
                f"Expected model output shape {target.shape}, got {model_output.shape}"
            )

        # DDPM-style objective for the selected parameterization.
        loss = F.mse_loss(model_output, target)

        # Convert back to common quantities so different parameterizations can
        # be compared on the same epsilon and x_0 scales.
        pred_noise = self._prediction_to_noise(x_t, t, model_output)
        pred_x0 = self._prediction_to_x0(x_t, t, model_output)

        noise_mse = F.mse_loss(pred_noise, noise)
        x0_mse = F.mse_loss(pred_x0, x_0)
        noise_cosine = F.cosine_similarity(
            pred_noise.flatten(start_dim=1),
            noise.flatten(start_dim=1),
            dim=1,
        ).mean()
        pred_norm = model_output.flatten(start_dim=1).norm(dim=1).mean()
        target_norm = target.flatten(start_dim=1).norm(dim=1).mean()

        loss_value = loss.detach().item()
        metrics = {
            "loss": loss_value,
            "mse": loss_value,
            "noise_mse": noise_mse.detach().item(),
            "x0_mse": x0_mse.detach().item(),
            "noise_cosine": noise_cosine.detach().item(),
            "pred_norm": pred_norm.detach().item(),
            "target_norm": target_norm.detach().item(),
        }

        per_sample_noise_mse = (pred_noise - noise).square().flatten(start_dim=1).mean(dim=1)
        bin_width = self.num_timesteps // 10
        for bin_index in range(10):
            bin_start = bin_index * bin_width
            bin_end = self.num_timesteps - 1 if bin_index == 9 else (bin_start + bin_width - 1)
            in_bin = (t >= bin_start) & (t <= bin_end)
            metric_name = f"bin_{bin_start:03d}_{bin_end:03d}_noise_mse"
            if in_bin.any():
                metrics[metric_name] = per_sample_noise_mse[in_bin].mean().detach().item()
            else:
                metrics[metric_name] = float("nan")

        return loss, metrics


    # =========================================================================
    # Reverse process (sampling)
    # =========================================================================
    
    @torch.no_grad()
    def reverse_process(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample one reverse DDPM step from p_theta(x_{t-1} | x_t).

        Args:
            x_t: Noisy images at timestep t with shape (B, C, H, W).
            t: One timestep per image with shape (B,).
        
        Returns:
            x_prev: Images after one denoising step, with the same shape as x_t.
        """
        if x_t.shape[0] != t.shape[0]:
            raise ValueError(
                "Batch size of x_t and t must match, "
                f"got {x_t.shape[0]} and {t.shape[0]}"
            )

        beta_t = self._extract(self.betas, t, x_t.shape)
        alpha_t = self._extract(self.alphas, t, x_t.shape)
        alpha_bar_t = self._extract(self.alpha_bars, t, x_t.shape)
        posterior_variance_t = self._extract(
            self.posterior_variance,
            t,
            x_t.shape,
        )

        model_output = self.model(x_t, t)
        if model_output.shape != x_t.shape:
            raise ValueError(
                f"Expected model output shape {x_t.shape}, got {model_output.shape}"
            )
        pred_noise = self._prediction_to_noise(x_t, t, model_output)

        # Reverse mean:
        # 1 / sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * pred_noise)
        noise_scale = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = (
            1.0 / torch.sqrt(alpha_t)
        ) * (
            x_t - noise_scale * pred_noise
        )

        # Add stochastic reverse noise for t > 0. At t = 0, the sample is the
        # final image estimate, so the extra Gaussian noise is masked out.
        reverse_noise = torch.randn_like(x_t)
        variance_noise = torch.sqrt(posterior_variance_t) * reverse_noise
        nonzero_mask = (t != 0).float()
        broadcast_shape = (t.shape[0],) + (1,) * (len(x_t.shape) - 1)
        nonzero_mask = nonzero_mask.reshape(broadcast_shape)

        x_prev = mean + nonzero_mask * variance_noise
        return x_prev

    def _build_sampling_timesteps(self, num_steps: int) -> torch.Tensor:
        """
        Build descending timestep indices for a strided sampler.

        For example, num_steps=100 selects 100 indices from 999 down to 0.
        The returned tensor has shape (num_steps,).
        """
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}.")
        if num_steps > self.num_timesteps:
            raise ValueError(
                f"num_steps cannot exceed {self.num_timesteps}, got {num_steps}."
            )

        if num_steps == self.num_timesteps:
            return torch.arange(
                self.num_timesteps - 1,
                -1,
                -1,
                device=self.device,
                dtype=torch.long,
            )

        timesteps = torch.linspace(
            self.num_timesteps - 1,
            0,
            steps=num_steps,
            device=self.device,
        ).round().long()
        return timesteps

    @torch.no_grad()
    def _ddim_strided_step(
        self,
        x_t: torch.Tensor,
        current_step: int,
        next_step: int,
    ) -> torch.Tensor:
        """
        Deterministic strided DDIM-style jump from timestep t to timestep s.

        This is used only when num_steps is smaller than the training diffusion
        horizon. It reuses the trained DDPM predictor by converting its output
        into epsilon, then reconstructing x_s from the predicted x_0.
        """
        batch_size = x_t.shape[0]
        t = torch.full(
            (batch_size,),
            current_step,
            device=self.device,
            dtype=torch.long,
        )
        model_output = self.model(x_t, t)
        if model_output.shape != x_t.shape:
            raise ValueError(
                f"Expected model output shape {x_t.shape}, got {model_output.shape}"
            )

        pred_noise = self._prediction_to_noise(x_t, t, model_output)
        pred_x0 = self._prediction_to_x0(x_t, t, model_output)

        if next_step < 0:
            return pred_x0

        alpha_bar_next = self.alpha_bars[next_step]
        sqrt_alpha_bar_next = torch.sqrt(alpha_bar_next)
        sqrt_one_minus_alpha_bar_next = torch.sqrt(1.0 - alpha_bar_next)
        return sqrt_alpha_bar_next * pred_x0 + sqrt_one_minus_alpha_bar_next * pred_noise


    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate images by iteratively denoising from pure Gaussian noise.

        Args:
            batch_size: Number of samples to generate.
            image_shape: Shape of each image as (channels, height, width).
            num_steps: Optional number of reverse steps. Full-length sampling
                uses the stochastic DDPM chain; fewer steps use a deterministic
                strided DDIM-style sampler.
            **kwargs: Reserved for future method-specific sampling arguments.
        
        Returns:
            samples: Generated samples with shape (batch_size, *image_shape).
        """
        if num_steps is None:
            num_steps = self.num_timesteps
        timesteps = self._build_sampling_timesteps(num_steps)

        self.eval_mode()

        # Start from x_T ~ N(0, I), one Gaussian noise image per requested sample.
        sample_shape = (batch_size, *image_shape)
        x_t = torch.randn(sample_shape, device=self.device)

        if num_steps == self.num_timesteps:
            for step in timesteps.tolist():
                # All images in this sampling batch are denoised at the same current
                # step, but reverse_process expects one timestep per image: shape (B,).
                t = torch.full(
                    (batch_size,),
                    step,
                    device=self.device,
                    dtype=torch.long,
                )
                x_t = self.reverse_process(x_t, t)
            return x_t

        next_timesteps = torch.cat(
            [
                timesteps[1:],
                torch.tensor([-1], device=self.device, dtype=torch.long),
            ]
        )
        for current_step, next_step in zip(timesteps.tolist(), next_timesteps.tolist()):
            # All images in this sampling batch are denoised at the same current
            # step, then jumped to the next selected timestep.
            x_t = self._ddim_strided_step(x_t, current_step, next_step)

        return x_t

    # =========================================================================
    # Device / state
    # =========================================================================

    def to(self, device: torch.device) -> "DDPM":
        nn.Module.to(self, device)
        self.device = device
        return self

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["num_timesteps"] = self.num_timesteps
        state["beta_start"] = self.beta_start
        state["beta_end"] = self.beta_end
        state["prediction_type"] = self.prediction_type
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
            prediction_type=ddpm_config.get("prediction_type", "epsilon"),
        )
