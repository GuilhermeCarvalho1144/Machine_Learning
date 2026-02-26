import numpy as np
import torch


class DDPMSampler:
    """
    Another name for this is the scheduler
    """

    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
    ) -> None:
        self.__num_training_steps = num_training_steps
        self.__generator = generator

        # defining the betas and alphas for the foward process
        self.__betas = (
            torch.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_training_steps,
                dtype=torch.float32,
            )
            ** 2
        )
        self.__alphas = 1.0 - self.__betas
        self.__alphas_cumprod = torch.cumprod(self.__alphas)
        self.__one = torch.tensor(1.0)
        self.__timesteps = torch.from_numpy(
            np.arange(0, self.__num_training_steps)[::-1].copy()
        )

    def set_inference_time_timesteps(
        self, num_inference_steps: int = 50
    ) -> None:
        self.__num_inference_steps = num_inference_steps
        self.__ratio = self.__num_training_steps // self.__num_inference_steps
        timesteps = (
            (np.arange(0, self.__num_training_steps) * self.__ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.__timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - self.__ratio
        return prev_t

    def _get_variance(self, timestep: int) -> torch.Tensor:
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.__alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.__alphas_cumprod[prev_t] if prev_t >= 0 else self.__one
        )
        current_beta_t = alpha_prod_t / alpha_prod_t_prev

        # compute variance using the formula 7 from the DDPM paper
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)

        return variance

    def set_strength(self, strength: float = 1):
        start_step = self.__num_inference_steps - int(
            self.__num_inference_steps * strength
        )
        self.__timesteps = self.__timesteps[start_step:]
        self.__start_step = start_step

    def step(
        self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Doing the reverse step
        remove noise from a model_output in the timestep t
        """
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.__alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.__alphas_cumprod[prev_t] if prev_t >= 0 else self.__one
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = -alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = -current_alpha_t

        # compute the predicted original sample using formula (15) of the DDPM paper
        pred_original_sample = (
            latents - beta_prod_t**0.5 * model_output
        ) / alpha_prod_t**0.5

        # compute the coeffs for pred_original_sample and current_sample x_t
        pred_original_sample_coeff = (
            alpha_prod_t_prev**0.5 * current_beta_t
        ) / beta_prod_t
        current_sample_coeff = (
            current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t
        )

        # compute the prediction previous sample mean
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * latents
        )

        stdev = 0.0
        noise = 0.0
        if t > 0:
            noise = torch.randn(
                model_output.shape,
                generator=self.__generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            stdev = self._get_variance(t) ** 0.5
        # mean +stdev*noise
        pred_prev_sample = pred_prev_sample + stdev * noise
        return pred_prev_sample

    def add_noise(
        self, original_sample: torch.FloatTensor, timestep: torch.IntTensor
    ) -> torch.FloatTensor:
        """
        Forward process add noise
        """
        device = original_sample.device
        dtype = original_sample.dtype

        # move tensor to device
        alpha_cumprod = self.__alphas_cumprod.to(device=device, dtype=dtype)
        timestep = timestep.to(device=device)

        # get the mean as decribe in the paper
        sqrt_alpha_prod = alpha_cumprod[timestep] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # fix dims
        while len(sqrt_alpha_prod.shape) < len(original_sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # get the std as describe in the paper
        sqrt_one_minus_alpha_prod = (1 - alpha_cumprod[timestep]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # fix dims
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # sample the noise
        noise = torch, torch.randn(
            original_sample.shape,
            generator=self.__generator,
            device=device,
            dtype=dtype,
        )
        # apply the noise
        # noisy = mean+std*noise
        noisy_sample = (
            sqrt_alpha_prod * original_sample
        ) + sqrt_one_minus_alpha_prod * noise
        return noisy_sample
