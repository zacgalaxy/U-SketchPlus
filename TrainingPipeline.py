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

scaler = GradScaler()

"torch.no_grad is in the other file due to it being used in infrance and evalution"
"""

def hook_unet(unet):
    

    #Hooked to be used during infrneces and are queried for there activation during inference
    #does not chnage when reversing the model 
    blocks_idx = [0, 1, 2]
    feature_blocks = []
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        print(f"Layer {module}: output shape {output.shape}")
        setattr(module, "output", output.float())
        
        if isinstance(output, tuple):
            output = output[0]# output is a tuple, get the first element
        
        if isinstance(output, torch.TensorType): # output is a tensor convert to float data type and stores it as new attribute output in current module
            feature = output.float()
            setattr(module, "output", feature) # dynamiclly attached output as attribute to the module makingis accepble for debugging or querying
        elif isinstance(output, dict): 
            feature = output.sample.float() # sample filed converted to float data type
            setattr(module, "output", feature) # due to outputs usally being dictonaris
        else: 
            feature = output.float() # output is a tensor convert to float data type in case prior fuctions didnt work 
            setattr(module, "output", feature)
    
    
    # these blocks are fundemental to U net architecture
    
    # down block, extract featuers and progressivly reduce spatial dimensions (down smapling) 
    # goes thriugh convalutional block, activation layer and max pooling in the  pipeline
    #outputs a feature map with lower dimention but more featuers
    
    # 0, 1, 2 -> (ldm-down) 2, 4, 8 [ translates to for example ldm down 0 downsmapled by factor of 2]
    for idx, block in enumerate(unet.down_blocks):
        if idx in blocks_idx:
            block.register_forward_hook(hook)
            feature_blocks.append(block) 
    
    # mid block, Operates on the smallest spatial resolution in the network, capturing the most abstract and high-level features.
    # seris of convolutional layers without down sampling 
    # has attention mechanism to capture long-range dependencies
    # ldm-mid 0, 1, 2 [ no factors here as dimensions stay constant]
    for block in unet.mid_block.attentions + unet.mid_block.resnets: #mid block, 
        block.register_forward_hook(hook)
        feature_blocks.append(block) 
    
    #up block,  Reconstruct the spatial resolution by progressively increasing the dimensions (up-sampling).
    # up sampling of layers using convtranspose, skip connections and convolutional layers
    # 0, 1, 2 -> (ldm-up) 2, 4, 8 [ translates to for example ldm up 0 upsmapled by factor of 2]
    for idx, block in enumerate(unet.up_blocks):
        if idx in blocks_idx:
            block.register_forward_hook(hook)
            feature_blocks.append(block)  
            
    return feature_blocks
"""

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

# ----------------------------
# Trainer Class
class SketchGuidedText2ImageTrainer():
    def __init__(self, stable_diffusion_pipeline, unet, vae, text_encoder, lep_unet, scheduler, tokenizer, sketch_simplifier, device):
        self.stable_diffusion_pipeline = stable_diffusion_pipeline
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.device = device
        self.sketch_simplifier = sketch_simplifier
        self.lep_unet = lep_unet

        # Set models to appropriate modes:
        self.vae.eval()            # VAE is frozen.
        self.text_encoder.eval()   # Text encoder is frozen.
        self.lep_unet.eval()       # ULPE remains frozen.
        self.unet.train()          # Only UNet is updated.
        self.unet.requires_grad_(True)
        self.lep_unet.requires_grad_(False)

        # Hook UNet layers to extract intermediate features.
        self.feature_blocks = hook_unet(self.unet)
    def train_step(self, images, sketches, text_prompts, optimizer, noise_scheduler, 
                  num_inference_timesteps, classifier_guidance_strength=8, 
                  sketch_guidance_strength=1.8, guidance_steps_perc=1):
        """
        This training step follows the original inference guidance:
          1. Encode ground-truth images into latents via the frozen VAE.
          2. Compute text embeddings.
          3. Add noise to the image latents.
          4. Iteratively denoise the latents using the UNet.
            At each timestep, intermediate UNet features (collected via hooks) are used
            to compute a guidance edge map via the LEP network.
            For early timesteps, the gradient of the similarity between the LEP output and 
            the target edge map is used to adjust the latents.
          5. The final latent is compared (via MSE loss) with the ground-truth latents.
        """
        # 1. Encode ground-truth images into latents.
        with torch.no_grad():
            gt_latents = self.encode_to_latents(images)
        
        # 2. Compute text embeddings.
        final_text_embeddings = self.stable_diffusion_pipeline._encode_prompt(
            prompt=text_prompts,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True
        )
        
        # 3. Generate random noise and initialize latents.
        batch_size = images.shape[0]
        noise = torch.randn(batch_size,
                            self.unet.config.in_channels,
                            self.unet.config.sample_size,
                            self.unet.config.sample_size).to(self.device)
        latents = noise.half() * self.scheduler.init_noise_sigma
        # Save a clone for guidance computations.
        noise_clone = latents.detach().clone()
        
        # Set the scheduler timesteps (e.g., 12).
        self.scheduler.set_timesteps(num_inference_timesteps)
        
        # Prepare edge maps from sketches (assumed to be working as before).
        encoded_edge_maps = self.prepare_edge_maps(
            prompt=text_prompts,
            num_images_per_prompt=1,
            edge_maps=sketches
        )
        print("Encoded edge maps norm:", torch.linalg.norm(encoded_edge_maps.float()).item())
        
        # 4. Iterative denoising loop.
        for i, timestep in enumerate(tqdm(self.scheduler.timesteps, desc="Denoising")):
            # Determine gradient state based on timestep.
            if i > int(guidance_steps_perc * num_inference_timesteps):
                current_gradient_state = torch.no_grad()
                gradient = False
            else:
                current_gradient_state = torch.enable_grad()
                gradient = True
            
            # Duplicate latents for classifier-free guidance.
            unet_input = self.scheduler.scale_model_input(torch.cat([latents]*2), timestep).to(self.device)
            unet_input = unet_input.requires_grad_(True)
            
            # Forward pass through UNet.
            with current_gradient_state:
                output = self.unet(unet_input, timestep, encoder_hidden_states=final_text_embeddings).sample
                u, t = output.chunk(2)
            pred = u + classifier_guidance_strength * (t - u)
            latents_old = unet_input.chunk(2)[1]
            latents = self.scheduler.step(pred, timestep, latents).prev_sample
            
            print("unet_input norm:", torch.linalg.norm(unet_input).item())
            print("latents_old norm:", torch.linalg.norm(latents_old).item())
            print("latents norm:", torch.linalg.norm(latents).item())
            print("Encoded edge maps norm:", torch.linalg.norm(encoded_edge_maps.float()).item())
            
            # Guidance branch.
            with current_gradient_state:
                intermediate_result = []
                fixed_size = latents.shape[2]
                for block in self.feature_blocks:
                    outputs = block.output  # Use the hook outputs as stored
                    resized = F.interpolate(outputs, size=fixed_size, mode="bilinear")
                    intermediate_result.append(resized)
                    # Do not clear the outputs so that the UNet's internal structure is not affected.
                intermediate_result = torch.cat(intermediate_result, dim=1).to(self.device)
                
                # Obtain noise level.
                noise_level = self.get_noise_level(noise_clone, timestep)
                # In the original pipeline, they concatenate two copies of noise_level.
                # (Assuming noise_level already has the expected shape; if not, adjust accordingly.)
                lep_input = torch.cat([noise_level, noise_level]).to(self.device)
                
                # Run the LEP network.
                result = self.lep_unet(intermediate_result, lep_input)
                _, result = result.chunk(2)
                
                if gradient:
                    similarity = torch.linalg.norm(result - encoded_edge_maps)**2
                    # Compute gradients from the similarity.
                    _, grad = torch.autograd.grad(similarity, unet_input)[0].chunk(2)
                    alpha = (torch.linalg.norm(latents_old - latents) / torch.linalg.norm(grad)) * sketch_guidance_strength
                    print(f"Step {i} | timestep: {timestep} | alpha: {alpha.item():.4f}")
                    latents = latents - alpha * grad
            
            gc.collect()
            torch.cuda.empty_cache()
        
        # 5. Compute final MSE loss.
        with torch.no_grad():
            loss = F.mse_loss(latents, gt_latents)
        return loss
                                    
    
    @torch.no_grad()
    def encode_to_latents(self, images):
        """
        Encodes image tensors (with pixel values in [0,1]) into latent representations using the frozen VAE.
        """
        images = images.to(self.vae.dtype)
        images = images * 2 - 1  # Normalize to [-1,1]
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215
        return latents
    
    @torch.no_grad()
    def text_to_embeddings(self, text):
        """
        Converts text prompts to embeddings.
        """
        tokenized = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        embeddings = self.text_encoder(tokenized.input_ids)[0]
        return embeddings.to(self.unet.dtype)
    
    def get_noise_level(self, noise, timesteps):
        """
        Computes a noise level tensor based on the scheduler's alphas_cumprod.
        """
        sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(noise.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        noise_level = sqrt_one_minus_alpha_prod.to(self.device) * noise
        return noise_level
    
    @torch.no_grad()
    def image_to_latents(self, img, img_type):
        """
        Converts a PIL image to a latent representation using the frozen VAE.
        If img_type is "edge_map", the image may be binarized.
        """

        np_img = np.array(img).astype(np.float16) / 255.0
        if img_type == "edge_map":
            np_img[np_img < 0.5] = 0.
            np_img[np_img >= 0.5] = 1.
        np_img = np_img * 2.0 - 1.0
        # If the image is grayscale (2D), add a channel dimension.
        if np_img.ndim == 2:
            np_img = np.expand_dims(np_img, axis=-1)  # shape becomes (H, W, 1)
        # Now add the batch dimension and transpose to NCHW format.
        np_img = np_img[None].transpose(0, 3, 1, 2)  # shape becomes (1, C, H, W)
        torch_img = torch.from_numpy(np_img)
        generator = torch.Generator(self.device).manual_seed(0)
        latents = self.vae.encode(torch_img.to(self.vae.dtype).to(self.device)).latent_dist.sample(generator=generator)
        latents = latents * 0.18215
        return latents

    
    @torch.no_grad()
    def prepare_edge_maps(self, prompt, num_images_per_prompt, edge_maps):
        batch_size = len(prompt)
        size=512
        #size=256
        if batch_size != len(edge_maps):
            raise ValueError("Wrong number of edge maps")
            
        processed_edge_maps = []
        for edge_map in edge_maps:
            arr = np.array(edge_map)
            # If the image is grayscale (2D), use it as is (after inversion).
            if arr.ndim == 2:
                inverted = 255 - arr
            else:
                inverted = 255 - arr[:,:,:3]
            processed_img = Image.fromarray(inverted).resize((size, size))
            processed_edge_maps.append(processed_img)
            
        encoded_edge_maps = [self.image_to_latents(edge.resize((size,size)), img_type="edge_map") 
                            for edge in processed_edge_maps]
        # Repeat each edge map if necessary.
        encoded_edge_maps_final = [edge for edge in encoded_edge_maps for _ in range(num_images_per_prompt)]
        encoded_edge_maps_tensor = torch.cat(encoded_edge_maps_final, dim=0)
        return encoded_edge_maps_tensor
    
    @torch.no_grad()
    def latents_to_image(self, latents):
        """
        Decodes image latents into PIL images using the frozen VAE decoder.
        """
        latents = (1/0.18215) * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image/2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(img) for img in images]
        return pil_images
    @torch.no_grad()
    def evaluate_step(self, images, sketches, text_prompts, noise_scheduler, num_inference_timesteps=50):
        # 1. Encode ground-truth images into latents.
        gt_latents = self.encode_to_latents(images)
        
        # 2. Compute text embeddings.
        final_text_embeddings = self.stable_diffusion_pipeline.encode_prompt(
            prompt=text_prompts,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True
        )
        
        # 3. Generate random noise and add it to the image latents.
        batch_size = images.shape[0]
        noise = torch.randn(batch_size,
                            self.unet.config.in_channels,
                            self.unet.config.sample_size,
                            self.unet.config.sample_size).to(self.device)
        latents = noise.half() * self.scheduler.init_noise_sigma
        noise_clone = latents.detach().clone()  # For guidance calculations
        
        # Set the scheduler timesteps.
        self.scheduler.set_timesteps(num_inference_timesteps)
        
        # Prepare edge maps from sketches.
        encoded_edge_maps = self.prepare_edge_maps(
            prompt=text_prompts,
            num_images_per_prompt=1,
            edge_maps=sketches
        )
        
        # Iterative denoising loop (no gradient tracking here).
        for i, timestep in enumerate(self.scheduler.timesteps):
            unet_input = self.scheduler.scale_model_input(torch.cat([latents]*2), timestep).to(self.device)
            unet_input = unet_input.requires_grad_(False)  # No gradient tracking
            u, t = self.unet(unet_input, timestep, encoder_hidden_states=final_text_embeddings).sample.chunk(2)
            pred = u + 8 * (t - u)  # assuming classifier_guidance_strength is 8
            latents_old = unet_input.chunk(2)[1]
            latents = self.scheduler.step(pred, timestep, latents).prev_sample
            
            # Guidance branch.
            intermediate_result = []
            fixed_size = latents.shape[2]
            for block in self.feature_blocks:
                feat = block.output
                resized = F.interpolate(feat, size=fixed_size, mode="bilinear")
                intermediate_result.append(resized)
                del block.output
            intermediate_result = torch.cat(intermediate_result, dim=1).to(self.device)
            noise_level = self.get_noise_level(noise_clone, timestep)
            lep_out = self.lep_unet(intermediate_result, torch.cat([noise_level]*2))
            _, lep_out = lep_out.chunk(2)
            # Calculate guidance loss for monitoring (but don't backpropagate).
            similarity = torch.linalg.norm(lep_out - encoded_edge_maps) ** 2
            # Optionally, print intermediate guidance loss every few steps:
            if i % 5 == 0:
                print(f"Step {i}: Guidance loss = {similarity.item():.4f}")
        
        # Compute final loss.
        loss = F.mse_loss(latents, gt_latents)
        return loss
