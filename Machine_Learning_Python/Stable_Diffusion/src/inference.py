from torch.cuda import is_available
import model_loader
import pipeline
from PIL import Image
from transformers import CLIPTokenizer
import torch


ALLOW_CUDA = False
if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"DEVICE: {DEVICE}")

tokenizer = CLIPTokenizer(
    "../data/vocab.json", merges_file="../data/merges.txt"
)
model_file = "../data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standart_weights(model_file, DEVICE)


prompt = "Generate the best image you can"
uncond_prompt = ""
do_cfg = True
cfg_scale = 8

input_image = "../images/dog.jpeg"
strength = 0.9

sampler = "ddpm"
seed = 42
n_inferences = 50


output_image = pipeline.generate(
    prompt=prompt,
    uncund_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_step=n_inferences,
    seed=seed,
    models=models,
    device=DEVICE,
    idel_device="cpu",
    tokenizer=tokenizer,
)

img = Image.fromarray(output_image)
img.save("../images/converted.png")
