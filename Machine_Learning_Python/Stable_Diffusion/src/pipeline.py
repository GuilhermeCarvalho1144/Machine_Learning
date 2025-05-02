import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

# CONSTANTS
WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = WIDTH // 8
LATENT_HEIGTH = HEIGHT // 8
MAX_LEN = 77


def rescale(x, old_range, new_range, clamp=False) -> torch.Tensor:
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def get_time_embedding(timestep) -> torch.Tensor:
    # define the freqs acooding to the formula on the paper
    freqs = torch.pow(
        10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160
    )
    # the None add a dim
    # (1,160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # (1, 160) concat with (1, 160) -> (1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


def generate(
    prompt: str,
    uncund_prompt: str,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_step=50,
    models={},
    seed=42,
    device=None,
    idel_device=None,
    tokenizer=None,
) -> None:

    with torch.no_grad():
        assert 0 < strength <= 1, "strength must between 0 and 1"
        to_idel = None
        if idel_device:
            to_idel: lambda x: x.to(idel_device)
        else:
            to_idel: lambda x: x

        num_generator = torch.Generator(device=device)
        num_generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        # Do Classifier Free Guidance

        # Convert the prompt ussing the tokenizer
        cond_tokens = tokenizer.bath_encode_plus(
            [prompt], padding="max_length", max_length=MAX_LEN
        ).input_ids
        # (batch_size, seq_len)
        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
        # (batch_size, seq_len) -> (batch_size, seq_len, Dims = 768)
        cond_context = clip(cond_tokens)
        if do_cfg:
            # doing the same for the uncund_prompt
            uncond_tokens = tokenizer.bath_encode_plus(
                [uncund_prompt], padding="max_length", max_length=MAX_LEN
            ).input_ids
            uncond_tokens = torch.tensor(
                uncund_prompt, dtype=torch.long, device=device
            )
            # (batch_size, seq_len) -> (batch_size, seq_len, Dims = 768)
            uncond_context = clip(uncond_tokens)
            # (2, batch_size, seq_len, Dims)
            context = torch.cat([cond_context, uncond_context])
        else:
            # (1, batch_size, seq_len, Dims)
            context = cond_context

        to_idel(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(num_generator)
            sample.set_inference_steps(n_inference_step)
        else:
            raise ValueError(f"No sampler named {sampler_name}")

        latent_shape = (1, 4, LATENT_HEIGTH, LATENT_WIDTH)

        if input_image:
            # we will do image-to-image generation
            encoder = models["encoder"]
            encoder.to(device)

            # read and resize image
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channel)
            input_image_tensor = torch.tensor(
                input_image_tensor, dtype=torch.float32, device=device
            )
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (Height, Width, Channel) -> (batch_size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (batch_size, Height, Width, Channel) -> (batch_size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # Generate Noise
            encoder_noise = torch.randn(
                latent_shape, generator=num_generator, device=device
            )
            # run the encoder
            latents = encoder(input_image_tensor, encoder_noise)

            # the strength tells the model how musch the output should resamble the input
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            to_idel(encoder)
        else:
            # we will do text-to-image
            latents = torch.randn(
                latent_shape, generator=num_generator, device=device
            )

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1,320)
            time_embedding = get_time_embedding(timestep).to(device)
            # (batch_size, 4, latents_heigth, latents_width)
            model_input = latents
            if do_cfg:
                # reapet batch_size beacause of the cond and uncond context
                # (batch_size, 4, latents_heigth, latents_width) -> (2*batch_size, 4, latents_heigth, latents_width)
                model_input = model_input.repeat(2, 1, 1, 1)
            model_output = diffusion(model_input, context, time_embedding)
            if do_cfg:
                cond_output, uncond_output = model_output.chunck(2)
                model_output = (
                    cfg_scale * (cond_output - uncond_output) + uncond_output
                )
            # remove the noise predict by the UNET
            latents = sampler.step(timestep, latents, model_output)
        to_idel(diffusion)

        decoder = models["decoder"]
        decoder.to(device)
        # convert from latents to images
        images = decoder(latents)
        to_idel(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (batch_size, Channel, Height, Width) -> (batch_size, Height, Width, Channel)
        images = images.permute(0, 3, 1, 2)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
