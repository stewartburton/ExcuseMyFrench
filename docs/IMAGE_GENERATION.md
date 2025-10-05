# Image Generation Guide

This document explains how to generate consistent images of **Butcher** (French bulldog) and **Nutsy** (squirrel) for the *Excuse My French* project. When existing photos are unavailable or insufficient, you can create new images using DreamBooth, LoRA and Stable Diffusion.

## DreamBooth Training for Butcher

1. **Collect training images** – Gather 20–30 high‑quality photos of Butcher showing his face from different angles, lighting conditions and expressions. Crop them to square (512 × 512 px) and save them in `data/butcher_images/`.
2. **Create captions** – For each image, create a caption containing a unique identifier token (e.g., `sks dog1`). Example: `photo of sks dog1 sitting on a couch`. These captions help the model associate the token with Butcher’s appearance.
3. **Train the model** – Use the DreamBooth training script under `scripts/train_dreambooth.py` (based on the [Kohya‑ss DreamBooth library](https://github.com/kohya-ss/sd-dreambooth-library)). Specify the base model (e.g., `stable-diffusion-1.5`) and set hyperparameters such as `--resolution 512`, `--learning_rate 5e-6` and `--max_train_steps 1200`. Training will produce a new checkpoint in `models/dreambooth_butcher/`.
4. **Apply a LoRA fine‑tuning (optional)** – LoRA allows rapid style adaptation without retraining the entire model. Use `scripts/train_lora.py` to train a LoRA adapter that captures Butcher’s distinctive features and attach it to the base model during inference.

## Generating Nutsy

Nutsy is an imaginary squirrel, so you can generate images directly using a base Stable Diffusion model or a specialized LoRA for animals. Example prompt:

    a realistic brown squirrel with big eyes, wearing a tiny red scarf, cinematic lighting

Run the prompt through your preferred diffusion pipeline (ComfyUI or the Python `diffusers` API) and save the best results in `data/nutsy_generated/`. If you plan to reuse Nutsy in many scenes, consider training a mini DreamBooth model on 5‑10 of your generated squirrel images to ensure consistency.

## Inference

Once you have trained models, use them to generate new images with descriptive prompts. Example Python code using the Hugging Face `diffusers` library:

    from diffusers import StableDiffusionPipeline
    import torch

    model_path = "models/dreambooth_butcher"
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    prompt = "photo of sks dog1 wearing sunglasses and a chef hat, sitting next to a realistic brown squirrel, cinematic lighting"
    image = pipe(prompt).images[0]
    image.save("data/generated/butcher_and_nutsy.png")

## Storing and Tagging Images

Organise generated images in the `data/images/` directory with descriptive filenames. Maintain a metadata file (e.g., `data/images/metadata.csv`) containing columns such as:

| file_name | character | emotion  | orientation | notes                |
|-----------|-----------|----------|-------------|----------------------|
| butcher_smiling_left.png | butcher   | happy     | left        | sunny outdoor shot   |
| nutsy_surprised.png     | nutsy     | surprised | front       | wearing scarf        |

This metadata helps the image selection script choose appropriate frames during episode generation.
