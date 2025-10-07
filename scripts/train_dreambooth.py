#!/usr/bin/env python3
"""
DreamBooth Training Script for Custom Character Models

This script trains custom Stable Diffusion models using DreamBooth for
consistent character generation. Optimized for training Butcher the French Bulldog
and other characters.

Key Features:
- Support for SD 1.5 and SDXL base models
- Prior preservation to maintain model capabilities
- Automatic checkpoint saving and validation
- Progress tracking and logging
- Memory-efficient training with gradient accumulation
- Mixed precision training (FP16/BF16)

Usage:
    python scripts/train_dreambooth.py --config training/config/butcher_config.yaml
    python scripts/train_dreambooth.py --character butcher --steps 1000
"""

import argparse
import json
import logging
import math
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from PIL import Image
from tqdm.auto import tqdm
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / "config" / ".env"
load_dotenv(env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for DreamBooth training."""

    # Model settings
    base_model: str = "runwayml/stable-diffusion-v1-5"
    revision: Optional[str] = None

    # Training data
    instance_data_dir: str = "training/butcher/images"
    instance_prompt: str = "a photo of sks dog"
    class_data_dir: Optional[str] = "training/butcher/class_images"
    class_prompt: Optional[str] = "a photo of a dog"

    # Prior preservation
    with_prior_preservation: bool = True
    prior_loss_weight: float = 1.0
    num_class_images: int = 200

    # Output settings
    output_dir: str = "models/dreambooth_butcher"
    logging_dir: str = "logs/dreambooth"

    # Training hyperparameters
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-6
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 800

    # Model training settings
    train_text_encoder: bool = False
    gradient_checkpointing: bool = True
    use_8bit_adam: bool = False
    mixed_precision: str = "fp16"  # fp16, bf16, or no

    # Data settings
    resolution: int = 512
    center_crop: bool = True

    # Checkpointing
    checkpointing_steps: int = 100
    validation_steps: int = 100
    validation_prompts: List[str] = field(default_factory=list)

    # Misc
    seed: Optional[int] = None
    max_grad_norm: float = 1.0

    @classmethod
    def from_yaml(cls, config_path: str) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, output_path: str):
        """Save configuration to YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)


class DreamBoothDataset(Dataset):
    """Dataset for DreamBooth training."""

    def __init__(
        self,
        instance_data_dir: str,
        instance_prompt: str,
        tokenizer: CLIPTokenizer,
        class_data_dir: Optional[str] = None,
        class_prompt: Optional[str] = None,
        resolution: int = 512,
        center_crop: bool = True,
    ):
        self.instance_data_dir = Path(instance_data_dir)
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.center_crop = center_crop

        # Get instance images
        self.instance_images = self._get_images(self.instance_data_dir)
        if len(self.instance_images) == 0:
            raise ValueError(f"No images found in {instance_data_dir}")

        logger.info(f"Found {len(self.instance_images)} instance images")

        # Get class images for prior preservation
        self.class_images = []
        self.class_prompt = class_prompt
        if class_data_dir is not None and class_prompt is not None:
            self.class_data_dir = Path(class_data_dir)
            self.class_images = self._get_images(self.class_data_dir)
            logger.info(f"Found {len(self.class_images)} class images")

    def _get_images(self, directory: Path) -> List[Path]:
        """Get all image files from directory."""
        if not directory.exists():
            return []

        extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        images = []
        for ext in extensions:
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"*{ext.upper()}"))

        return sorted(images)

    def __len__(self) -> int:
        """Return dataset size (number of instance images)."""
        return len(self.instance_images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training example."""
        example = {}

        # Load and process instance image
        instance_image = Image.open(self.instance_images[idx])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        instance_image = self._process_image(instance_image)
        example["instance_images"] = instance_image

        # Tokenize instance prompt
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # Add class image if using prior preservation
        if len(self.class_images) > 0:
            class_idx = idx % len(self.class_images)
            class_image = Image.open(self.class_images[class_idx])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")

            class_image = self._process_image(class_image)
            example["class_images"] = class_image

            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]

        return example

    def _process_image(self, image: Image.Image) -> torch.Tensor:
        """Process image: resize, crop, normalize."""
        # Resize
        if self.center_crop:
            # Center crop to square
            crop_size = min(image.size)
            left = (image.width - crop_size) // 2
            top = (image.height - crop_size) // 2
            right = left + crop_size
            bottom = top + crop_size
            image = image.crop((left, top, right, bottom))

        # Resize to target resolution
        image = image.resize((self.resolution, self.resolution), Image.LANCZOS)

        # Convert to tensor and normalize to [-1, 1]
        image = torch.from_numpy(
            torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
            .reshape(self.resolution, self.resolution, 3)
            .numpy()
        ).float() / 255.0

        image = image.permute(2, 0, 1)  # HWC to CHW
        image = (image - 0.5) / 0.5  # Normalize to [-1, 1]

        return image


class DreamBoothTrainer:
    """DreamBooth trainer for custom character models."""

    def __init__(self, config: TrainingConfig):
        """
        Initialize the DreamBooth trainer.

        Args:
            config: Training configuration
        """
        self.config = config

        # Setup accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            log_with="tensorboard",
            project_dir=config.logging_dir,
        )

        # Set seed for reproducibility
        if config.seed is not None:
            set_seed(config.seed)

        # Create output directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.accelerator.is_main_process:
            Path(config.logging_dir).mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.unet = None
        self.noise_scheduler = None

        # Training state
        self.global_step = 0

        logger.info("DreamBooth trainer initialized")

    def load_models(self):
        """Load pretrained models."""
        logger.info(f"Loading models from {self.config.base_model}")

        # Load tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.base_model,
            subfolder="tokenizer",
            revision=self.config.revision,
        )

        # Load text encoder
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.base_model,
            subfolder="text_encoder",
            revision=self.config.revision,
        )

        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.config.base_model,
            subfolder="vae",
            revision=self.config.revision,
        )

        # Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.base_model,
            subfolder="unet",
            revision=self.config.revision,
        )

        # Load noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.config.base_model,
            subfolder="scheduler",
        )

        # Freeze VAE
        self.vae.requires_grad_(False)

        # Optionally freeze text encoder
        if not self.config.train_text_encoder:
            self.text_encoder.requires_grad_(False)

        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.config.train_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()

        logger.info("Models loaded successfully")

    def generate_class_images(self):
        """Generate class images for prior preservation."""
        if not self.config.with_prior_preservation:
            return

        class_dir = Path(self.config.class_data_dir)
        class_dir.mkdir(parents=True, exist_ok=True)

        # Check how many images already exist
        existing_images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        num_existing = len(existing_images)

        if num_existing >= self.config.num_class_images:
            logger.info(f"Class images already exist: {num_existing}")
            return

        num_to_generate = self.config.num_class_images - num_existing
        logger.info(f"Generating {num_to_generate} class images...")

        # Load pipeline for generation
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.base_model,
            revision=self.config.revision,
            torch_dtype=torch.float16 if self.accelerator.mixed_precision == "fp16" else torch.float32,
        )
        pipeline.set_progress_bar_config(disable=True)
        pipeline.to(self.accelerator.device)

        # Generate images
        batch_size = 4
        for i in tqdm(range(0, num_to_generate, batch_size), desc="Generating class images"):
            batch_size_actual = min(batch_size, num_to_generate - i)

            images = pipeline(
                [self.config.class_prompt] * batch_size_actual,
                num_inference_steps=50,
                guidance_scale=7.5,
            ).images

            for j, image in enumerate(images):
                image_path = class_dir / f"class_{num_existing + i + j:05d}.jpg"
                image.save(image_path)

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Class images generated successfully")

    def create_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        dataset = DreamBoothDataset(
            instance_data_dir=self.config.instance_data_dir,
            instance_prompt=self.config.instance_prompt,
            tokenizer=self.tokenizer,
            class_data_dir=self.config.class_data_dir if self.config.with_prior_preservation else None,
            class_prompt=self.config.class_prompt if self.config.with_prior_preservation else None,
            resolution=self.config.resolution,
            center_crop=self.config.center_crop,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
        )

        return dataloader

    def setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer."""
        # Collect parameters to optimize
        params_to_optimize = list(self.unet.parameters())
        if self.config.train_text_encoder:
            params_to_optimize += list(self.text_encoder.parameters())

        # Create optimizer
        if self.config.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer_class = bnb.optim.AdamW8bit
            except ImportError:
                logger.warning("bitsandbytes not available, using regular AdamW")
                optimizer_class = torch.optim.AdamW
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8,
        )

        return optimizer

    def train(self):
        """Run training loop."""
        logger.info("Starting training")

        # Load models
        self.load_models()

        # Generate class images if needed
        if self.accelerator.is_main_process:
            self.generate_class_images()

        self.accelerator.wait_for_everyone()

        # Create dataloader
        train_dataloader = self.create_dataloader()

        # Setup optimizer
        optimizer = self.setup_optimizer()

        # Setup learning rate scheduler
        lr_scheduler = get_scheduler(
            self.config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.config.lr_warmup_steps * self.config.gradient_accumulation_steps,
            num_training_steps=self.config.max_train_steps * self.config.gradient_accumulation_steps,
        )

        # Prepare for training with accelerator
        if self.config.train_text_encoder:
            self.unet, self.text_encoder, optimizer, train_dataloader, lr_scheduler = (
                self.accelerator.prepare(
                    self.unet, self.text_encoder, optimizer, train_dataloader, lr_scheduler
                )
            )
        else:
            self.unet, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
                self.unet, optimizer, train_dataloader, lr_scheduler
            )

        # Move VAE to device
        self.vae.to(self.accelerator.device)

        # Calculate number of epochs
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.config.gradient_accumulation_steps)
        num_train_epochs = math.ceil(self.config.max_train_steps / num_update_steps_per_epoch)

        logger.info("***** Training Configuration *****")
        logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
        logger.info(f"  Num epochs = {num_train_epochs}")
        logger.info(f"  Batch size per device = {self.config.train_batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.config.max_train_steps}")

        # Training loop
        progress_bar = tqdm(
            range(self.config.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
            desc="Training",
        )

        for epoch in range(num_train_epochs):
            self.unet.train()
            if self.config.train_text_encoder:
                self.text_encoder.train()

            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    # Convert images to latent space
                    latents = self.vae.encode(batch["instance_images"].to(dtype=self.vae.dtype)).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor

                    # Sample noise
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]

                    # Sample random timesteps
                    timesteps = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                    )
                    timesteps = timesteps.long()

                    # Add noise to latents
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get text embeddings
                    encoder_hidden_states = self.text_encoder(batch["instance_prompt_ids"])[0]

                    # Predict noise
                    model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # Calculate loss
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                    # Add prior preservation loss
                    if self.config.with_prior_preservation and "class_images" in batch:
                        # Convert class images to latent space
                        class_latents = self.vae.encode(batch["class_images"].to(dtype=self.vae.dtype)).latent_dist.sample()
                        class_latents = class_latents * self.vae.config.scaling_factor

                        # Sample noise for class images
                        class_noise = torch.randn_like(class_latents)

                        # Add noise
                        noisy_class_latents = self.noise_scheduler.add_noise(class_latents, class_noise, timesteps)

                        # Get class text embeddings
                        class_encoder_hidden_states = self.text_encoder(batch["class_prompt_ids"])[0]

                        # Predict noise
                        class_model_pred = self.unet(noisy_class_latents, timesteps, class_encoder_hidden_states).sample

                        # Calculate prior preservation loss
                        prior_loss = F.mse_loss(class_model_pred.float(), class_noise.float(), reduction="mean")

                        # Add to total loss
                        loss = loss + self.config.prior_loss_weight * prior_loss

                    # Backward pass
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        params_to_clip = (
                            list(self.unet.parameters())
                            if not self.config.train_text_encoder
                            else list(self.unet.parameters()) + list(self.text_encoder.parameters())
                        )
                        self.accelerator.clip_grad_norm_(params_to_clip, self.config.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Update progress
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    self.global_step += 1

                    # Save checkpoint
                    if self.global_step % self.config.checkpointing_steps == 0:
                        if self.accelerator.is_main_process:
                            self.save_checkpoint()

                    # Run validation
                    if self.global_step % self.config.validation_steps == 0:
                        if self.accelerator.is_main_process and len(self.config.validation_prompts) > 0:
                            self.validate()

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if self.global_step >= self.config.max_train_steps:
                    break

            if self.global_step >= self.config.max_train_steps:
                break

        # Save final model
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.save_model()

        self.accelerator.end_training()
        logger.info("Training complete!")

    def save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving checkpoint to {checkpoint_dir}")

        # Save model state
        self.accelerator.save_state(str(checkpoint_dir))

        # Save config
        self.config.to_yaml(str(checkpoint_dir / "config.yaml"))

    def save_model(self):
        """Save final trained model."""
        logger.info(f"Saving final model to {self.output_dir}")

        # Create pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.base_model,
            unet=self.accelerator.unwrap_model(self.unet),
            text_encoder=self.accelerator.unwrap_model(self.text_encoder) if self.config.train_text_encoder else None,
            revision=self.config.revision,
        )

        # Save pipeline
        pipeline.save_pretrained(str(self.output_dir))

        # Save training config
        self.config.to_yaml(str(self.output_dir / "training_config.yaml"))

        # Save training info
        info = {
            "character": Path(self.config.instance_data_dir).parent.name,
            "base_model": self.config.base_model,
            "training_steps": self.global_step,
            "instance_prompt": self.config.instance_prompt,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.output_dir / "model_info.json", 'w') as f:
            json.dump(info, f, indent=2)

        logger.info("Model saved successfully")

    def validate(self):
        """Run validation with sample prompts."""
        logger.info("Running validation...")

        # Create pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.base_model,
            unet=self.accelerator.unwrap_model(self.unet),
            text_encoder=self.accelerator.unwrap_model(self.text_encoder) if self.config.train_text_encoder else None,
            revision=self.config.revision,
            torch_dtype=torch.float16 if self.accelerator.mixed_precision == "fp16" else torch.float32,
        )
        pipeline.to(self.accelerator.device)

        # Generate validation images
        validation_dir = self.output_dir / f"validation-{self.global_step}"
        validation_dir.mkdir(parents=True, exist_ok=True)

        for i, prompt in enumerate(self.config.validation_prompts):
            image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
            image.save(validation_dir / f"image_{i:02d}.png")

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Validation images saved to {validation_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train DreamBooth model for custom characters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train using config file
  python scripts/train_dreambooth.py --config training/config/butcher_config.yaml

  # Train with custom settings
  python scripts/train_dreambooth.py --character butcher --steps 1000 --lr 5e-6

  # Resume from checkpoint
  python scripts/train_dreambooth.py --config training/config/butcher_config.yaml --resume
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--character",
        type=str,
        help="Character name (e.g., butcher, nutsy)"
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default=os.getenv("STABLE_DIFFUSION_MODEL", "runwayml/stable-diffusion-v1-5"),
        help="Base Stable Diffusion model"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=800,
        help="Number of training steps"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-6,
        help="Learning rate"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size"
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Training resolution"
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--no-prior-preservation",
        action="store_true",
        help="Disable prior preservation"
    )

    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = TrainingConfig.from_yaml(args.config)
    elif args.character:
        # Create config from command-line args
        instance_dir = f"training/{args.character}/images"
        class_dir = f"training/{args.character}/class_images"
        output_dir = os.getenv("DREAMBOOTH_OUTPUT_DIR", f"models/dreambooth_{args.character}")

        config = TrainingConfig(
            base_model=args.base_model,
            instance_data_dir=instance_dir,
            instance_prompt=f"a photo of sks {args.character}",
            class_data_dir=class_dir if not args.no_prior_preservation else None,
            class_prompt=f"a photo of a {args.character}" if not args.no_prior_preservation else None,
            output_dir=output_dir,
            with_prior_preservation=not args.no_prior_preservation,
            max_train_steps=args.steps,
            learning_rate=args.lr,
            train_batch_size=args.batch_size,
            resolution=args.resolution,
            seed=args.seed,
        )
    else:
        parser.error("Either --config or --character must be specified")

    # Create trainer and run
    trainer = DreamBoothTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
