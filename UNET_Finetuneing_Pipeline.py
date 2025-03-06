import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import math
import numpy as np
import random
import gc
from PIL import Image, ImageStat
from itertools import repeat
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.checkpoint import checkpoint


# Assume hook_unet is defined elsewhere in your project.
# from your_module import hook_unet

def hook_unet(unet):
    """
    Registers forward hooks on selected layers of the UNet model and returns a list of those layers.
    
    The hook saves the module's output (converted to float) into an attribute called 'output'.
    This output can later be used to extract intermediate features for guidance.

    Args:
        unet: The UNet model instance.

    Returns:
        A list of the hooked modules.
    """
    # Specify which indices (of down, mid, and up blocks) you want to hook.
    blocks_idx = [0, 1, 2]
    feature_blocks = []

    def hook(module, input, output):
        # If the output is a tuple, take the first element.
        if isinstance(output, tuple):
            output = output[0]
        # Convert the output to float and attach it as an attribute.
        setattr(module, "output", output.float())

    # Register hooks on the UNet's down blocks.
    for idx, block in enumerate(unet.down_blocks):
        if idx in blocks_idx:
            block.register_forward_hook(hook)
            feature_blocks.append(block)

    # Register hooks on the mid block's attention and resnet modules.
    for block in unet.mid_block.attentions + unet.mid_block.resnets:
        block.register_forward_hook(hook)
        feature_blocks.append(block)

    # Register hooks on the UNet's up blocks.
    for idx, block in enumerate(unet.up_blocks):
        if idx in blocks_idx:
            block.register_forward_hook(hook)
            feature_blocks.append(block)

    return feature_blocks

def checkpointed_unet_forward(model, *inputs):
    # This function calls your model forward pass (or a subset of it)
    return model(*inputs)

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import os

# Assume hook_unet is defined and imported from your modules.
# For example: from your_module import hook_unet

class UNetDiffusionTrainer:
    def __init__(self, stable_diffusion_pipeline, unet, vae, text_encoder, lep_unet,
                 scheduler, tokenizer, sketch_simplifier, device):
        """
        Args:
            stable_diffusion_pipeline: Pretrained diffusion pipeline (for text embeddings, etc.).
            unet: The diffusion UNet model to be fine-tuned.
            vae: The frozen VAE for encoding/decoding images.
            text_encoder: Text encoder (frozen) used by the pipeline.
            lep_unet: The frozen ULEP module (used only during evaluation).
            scheduler: Noise scheduler (e.g., DDIM).
            tokenizer: Tokenizer for text prompts.
            sketch_simplifier: (Optional) sketch simplification model.
            device: torch.device.
        """
        self.stable_diffusion_pipeline = stable_diffusion_pipeline
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.lep_unet = lep_unet
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.sketch_simplifier = sketch_simplifier
        self.device = device

        # Set models to appropriate modes:
        self.vae.eval()
        self.text_encoder.eval()
        self.lep_unet.eval()  # ULEP remains frozen.
        self.unet.train()
        self.unet.requires_grad_(True)
        self.lep_unet.requires_grad_(False)

        # Hook the UNet layers (e.g., down, mid, up blocks) to extract intermediate features for ULEP guidance.
        self.feature_blocks = hook_unet(self.unet)

    def train_step(self, images, sketches, text_prompts, optimizer, num_inference_timesteps=50):
        # 1. Encode ground-truth images and sketches.
        with torch.no_grad():
            gt_latents = self.encode_to_latents(images)
            sketch_latents = self.image_to_latents_from_batch(sketches)
        
        # 2. Compute text embeddings.
        final_text_embeddings = self.stable_diffusion_pipeline.encode_prompt(
            prompt=text_prompts,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True
        )[0]  # Extract tensor if a tuple is returned

        # 3. Generate noise and initialize latents.
        batch_size = images.shape[0]
        noise = torch.randn(batch_size,
                            self.unet.config.in_channels,
                            self.unet.config.sample_size,
                            self.unet.config.sample_size).to(self.device)
        latents = sketch_latents + noise.half() * self.scheduler.init_noise_sigma

        # 4. Set scheduler timesteps.
        self.scheduler.set_timesteps(num_inference_timesteps)

        # 5. Iterative denoising.
        for timestep in self.scheduler.timesteps:
            unet_input = self.scheduler.scale_model_input(latents, timestep).to(self.device)
            with torch.cuda.amp.autocast():
                output = self.unet(unet_input, timestep, encoder_hidden_states=final_text_embeddings)
                pred = output.sample
            latents = self.scheduler.step(pred, timestep, latents).prev_sample

        # 6. Compute loss.
        with torch.cuda.amp.autocast():
            loss = F.mse_loss(latents, gt_latents)

        optimizer.zero_grad()
        loss.backward()
        # Apply gradient clipping to stabilize training.
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()


    def evaluate_with_ulep_guidance(self, images, sketches, text_prompts,
                                    num_inference_timesteps=50,
                                    classifier_guidance_strength=8,
                                    sketch_guidance_strength=1.8,
                                    guidance_steps_perc=0.5):
        """
        Evaluates the fine-tuned UNet by generating images with ULEP guidance.
        In this evaluation, the UNet latent is initialized using the sketch latent plus noise.
        ULEP guidance is applied for the first fraction of timesteps, and the final MSE
        (between guided latents and ground-truth latents) is computed.
        
        Args:
            images: Ground-truth images (for computing gt latents).
            sketches: Input sketches (PIL images) used for generating sketch latents.
            text_prompts: Corresponding text prompts.
            num_inference_timesteps: Total diffusion steps.
            classifier_guidance_strength: Scale factor for classifier-free guidance.
            sketch_guidance_strength: Weight for the ULEP guidance gradient.
            guidance_steps_perc: Fraction of timesteps to apply ULEP guidance.
            
        Returns:
            generated_images: List of PIL images decoded from final latents.
            final_loss: MSE loss between guided latents and gt latents.
            guidance_loss_accum: Accumulated ULEP guidance loss.
        """
        with torch.no_grad():
            gt_latents = self.encode_to_latents(images)
            sketch_latents = self.image_to_latents_from_batch(sketches)

        final_text_embeddings = self.stable_diffusion_pipeline._encode_prompt(
            prompt=text_prompts,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True
        )

        batch_size = images.shape[0]
        noise = torch.randn(batch_size,
                            self.unet.config.in_channels,
                            self.unet.config.sample_size,
                            self.unet.config.sample_size).to(self.device)
        # Initialize latents as sketch latent plus noise.
        latents = sketch_latents + noise.half() * self.scheduler.init_noise_sigma
        noise_clone = latents.detach().clone()

        # Prepare target edge maps from sketches.
        encoded_edge_maps = self.prepare_edge_maps(
            prompt=text_prompts,
            num_images_per_prompt=1,
            edge_maps=sketches
        )

        self.scheduler.set_timesteps(num_inference_timesteps)
        guidance_loss_accum = 0.0

        for i, timestep in enumerate(tqdm(self.scheduler.timesteps, desc="Evaluation")):
            grad_mode = torch.enable_grad() if i <= int(guidance_steps_perc * num_inference_timesteps) else torch.no_grad()
            unet_input = self.scheduler.scale_model_input(torch.cat([latents] * 2), timestep).to(self.device)
            unet_input = unet_input.requires_grad_(True)
            with grad_mode, autocast():
                u, t = self.unet(unet_input, timestep, encoder_hidden_states=final_text_embeddings).sample.chunk(2)
            pred = u + classifier_guidance_strength * (t - u)
            latents_old = unet_input.chunk(2)[1]
            latents = self.scheduler.step(pred, timestep, latents).prev_sample

            if i <= int(guidance_steps_perc * num_inference_timesteps):
                with grad_mode, autocast():
                    intermediate_features = []
                    fixed_size = latents.shape[2]
                    for block in self.feature_blocks:
                        feat = block.output
                        resized_feat = F.interpolate(feat, size=fixed_size, mode="bilinear")
                        intermediate_features.append(resized_feat)
                    intermediate_features = torch.cat(intermediate_features, dim=1).to(self.device)
                    noise_level = self.get_noise_level(noise_clone, timestep)
                    lep_out = self.lep_unet(intermediate_features, torch.cat([noise_level] * 2))
                    _, lep_out = lep_out.chunk(2)
                    guidance_loss = torch.linalg.norm(lep_out - encoded_edge_maps) ** 2
                    guidance_loss_accum += guidance_loss.item()
                    grad_full = torch.autograd.grad(guidance_loss, unet_input, retain_graph=True)[0]
                    _, grad = grad_full.chunk(2)
                    alpha = (torch.linalg.norm(latents_old - latents) / (torch.linalg.norm(grad) + 1e-8)) * sketch_guidance_strength
                    latents = latents - alpha * grad
                    latents = latents.detach()

        generated_images = self.latents_to_image(latents)
        with autocast():
            final_loss = F.mse_loss(latents, gt_latents)
        return generated_images, final_loss.item(), guidance_loss_accum

    def save_model(self, save_path):
        """
        Saves the current UNet model weights to the specified file path.
        
        Args:
            save_path: Full file path (including filename) where the model will be saved.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.unet.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    # ----- Helper Methods -----
    def encode_to_latents(self, images):
        """
        Encodes images (with pixel values in [0,1]) into latent representations using the frozen VAE.
        """
        images = images.to(self.vae.dtype)
        images = images * 2 - 1  # Normalize to [-1, 1]
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215
        return latents

    def image_to_latents_from_batch(self, imgs):
        """
        Expects imgs as a list of PIL images (or a tensor of shape compatible with image_to_latents)
        and returns a batch of latents.
        """
        latents_list = []
        for img in imgs:
            # Ensure the image is in RGB (if needed).
            if isinstance(img, Image.Image):
                img = img.convert("RGB")
            latents = self.image_to_latents(img)
            latents_list.append(latents)
        return torch.cat(latents_list, dim=0)

    def get_noise_level(self, noise, timestep):
        """
        Computes a noise level tensor based on the scheduler's alphas_cumprod.
        """
        sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod[timestep]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(noise.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        noise_level = sqrt_one_minus_alpha_prod.to(self.device) * noise
        return noise_level

    def prepare_edge_maps(self, prompt, num_images_per_prompt, edge_maps):
        """
        Prepares edge maps from input sketches:
         - Inverts and resizes sketches.
         - Encodes them into latent representations (using the VAE, for simplicity).
        """
        processed_edge_maps = []
        size = 512  # Example resolution.
        for edge_map in edge_maps:
            arr = np.array(edge_map)
            if arr.ndim == 2:
                inverted = 255 - arr
            else:
                inverted = 255 - arr[:, :, :3]
            processed_img = Image.fromarray(inverted).resize((size, size))
            processed_edge_maps.append(processed_img)
        encoded_edge_maps = [self.image_to_latents(edge.convert("RGB")) for edge in processed_edge_maps]
        encoded_edge_maps_final = [edge for edge in encoded_edge_maps for _ in range(num_images_per_prompt)]
        encoded_edge_maps_tensor = torch.cat(encoded_edge_maps_final, dim=0)
        return encoded_edge_maps_tensor

    def image_to_latents(self, img):
        """
        Converts a PIL image into a latent representation using the frozen VAE.
        """
        img = img.resize((512, 512))
        np_img = np.array(img).astype(np.float32) / 255.0
        np_img = np_img * 2.0 - 1.0
        np_img = np_img[None].transpose(0, 3, 1, 2)
        torch_img = torch.from_numpy(np_img)
        generator = torch.Generator(self.device).manual_seed(0)
        latents = self.vae.encode(torch_img.to(self.vae.dtype).to(self.device)).latent_dist.sample(generator=generator)
        latents = latents * 0.18215
        return latents

    def latents_to_image(self, latents):
        """
        Decodes latent representations into PIL images using the frozen VAE decoder.
        """
        latents = (1 / 0.18215) * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(img) for img in images]
        return pil_images
