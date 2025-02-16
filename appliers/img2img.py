import torch
from diffusers.utils import (
    PIL_INTERPOLATION,
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    replace_example_docstring,
)
try:
    from diffusers.utils import randn_tensor
except ImportError:
    from diffusers.utils.torch_utils import randn_tensor
import PIL
import numpy as np
from PIL import Image
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

def find_index(functions, name):
    target_function_index = None
    for index, func in enumerate(functions):
        if (hasattr(func, "__name__") and func.__name__ == name) or \
           (hasattr(func, "func") and hasattr(func.func, "__name__") and func.func.__name__ == name):
            target_function_index = index
            break
    return target_function_index

def apply_img2img(pipe):
    # Ensure that the helper function is attached to the pipeline
    pipe.img2img_prepare_latents = partial(img2img_prepare_latents, pipe)

    # Insert the img2img default function at the beginning
    pipe.pipeline_functions.insert(0, partial(img2img_default, pipe))
    
    # Replace check_inputs_fn with an img2img-specific version
    checker_index = find_index(pipe.pipeline_functions, "check_inputs_fn")
    pipe.img2img_stored_check_inputs = pipe.check_inputs_fn
    pipe.check_inputs_fn = partial(img2img_check_inputs, pipe)
    pipe.pipeline_functions[checker_index] = pipe.check_inputs_fn
    
    # Replace prepare_latents_fn with our custom img2img version
    latent_index = find_index(pipe.pipeline_functions, "prepare_latents_fn")
    pipe.img2img_stored_prepare_latents_fn = pipe.prepare_latents_fn
    pipe.prepare_latents_fn = partial(prepare_latent_var, pipe)
    pipe.pipeline_functions[latent_index] = pipe.prepare_latents_fn
    # Insert the img2img preprocessing step immediately before that
    pipe.pipeline_functions.insert(latent_index, partial(img2img_preprocess_img, pipe))
    
    # Insert a new function for img2img_get_timesteps
    pipe.img2img_get_timesteps = partial(img2img_get_timesteps, pipe)
    
    # Replace prepare_timesteps_fn with an img2img version
    timesteps_index = find_index(pipe.pipeline_functions, "prepare_timesteps_fn")
    pipe.img2img_stored_prepare_timesteps_fn = pipe.prepare_timesteps_fn
    pipe.prepare_timesteps_fn = partial(prepare_timesteps, pipe)
    pipe.pipeline_functions[timesteps_index] = pipe.prepare_timesteps_fn

    def remover_img2img():
        # Revert prepare_timesteps_fn replacement
        pipe.prepare_timesteps_fn = pipe.img2img_stored_prepare_timesteps_fn
        pipe.pipeline_functions[timesteps_index] = pipe.prepare_timesteps_fn
        delattr(pipe, "img2img_stored_prepare_timesteps_fn")
        
        # Remove img2img_get_timesteps
        delattr(pipe, "img2img_get_timesteps")
        
        # Remove the inserted img2img_preprocess_img
        pipe.pipeline_functions.pop(latent_index)
        
        # Revert prepare_latents_fn replacement
        pipe.prepare_latents_fn = pipe.img2img_stored_prepare_latents_fn
        pipe.pipeline_functions[latent_index] = pipe.prepare_latents_fn
        delattr(pipe, "img2img_stored_prepare_latents_fn")
        
        # Revert the check_inputs_fn replacement
        pipe.check_inputs_fn = pipe.img2img_stored_check_inputs
        pipe.pipeline_functions[checker_index] = pipe.check_inputs_fn
        delattr(pipe, "img2img_stored_check_inputs")
        
        # Remove the defaults function inserted at the beginning
        pipe.pipeline_functions.pop(0)
        
        delattr(pipe, "img2img_prepare_latents")
    
    pipe.revert_functions.insert(0, remover_img2img)

def img2img_prepare_latents(self, image, latent_timestep, batch_size, num_images_per_prompt, dtype, device, generator=None, skip_noise=False):
    if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
        raise ValueError(
            f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
        )

    # Handle PIL images
    if isinstance(image, PIL.Image.Image):
        image = self.image_processor.preprocess(image)
    
    # Move image to correct device and dtype
    image = image.to(device=device, dtype=self.vae.dtype)
    
    # Calculate batch size
    effective_batch_size = batch_size * num_images_per_prompt
    
    # Encode the image to get initial latents
    if image.shape[1] == 4:  # Image is already in latent space
        init_latents = image
    else:
        # Encode image using VAE - SANA returns EncoderOutput
        encoder_output = self.vae.encode(image)
        # Get the actual tensor from EncoderOutput (using the retrieve_latents helper)
        init_latents = retrieve_latents(encoder_output, generator=generator)
        # Apply scaling factor
        init_latents = self.vae.config.scaling_factor * init_latents
    
    # Handle batch size requirements
    if effective_batch_size > init_latents.shape[0]:
        if effective_batch_size % init_latents.shape[0] == 0:
            repeat_factor = effective_batch_size // init_latents.shape[0]
            init_latents = init_latents.repeat(repeat_factor, 1, 1, 1)
        else:
            raise ValueError(
                f"Cannot duplicate `image` batch of size {init_latents.shape[0]} to {effective_batch_size} text prompts."
            )
    
    # Add noise to latents unless skipped
    if not skip_noise:
        noise = randn_tensor(init_latents.shape, generator=generator, device=device, dtype=dtype)
        init_latents = self.scheduler.add_noise(init_latents, noise, latent_timestep)
    
    return init_latents

def prepare_latent_var(self, **kwargs):
    # Get the number of channels from the transformer config
    num_channels_latents = self.transformer.config.in_channels
    
    # Get required parameters
    height = kwargs.get('height')
    width = kwargs.get('width')
    device = kwargs.get('device')
    dtype = kwargs.get('dtype')
    
    if 'image' in kwargs and kwargs['image'] is not None:
        latents = self.img2img_prepare_latents(
            kwargs.get('image'),
            kwargs.get('latent_timestep'),
            kwargs.get('batch_size'),
            kwargs.get('num_images_per_prompt'),
            dtype,
            device,
            kwargs.get('generator'),
            kwargs.get('skip_noise', False)
        )
    else:
        # Default txt2img latent preparation
        latents = self.prepare_latents(
            kwargs.get('batch_size') * kwargs.get('num_images_per_prompt', 1),
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            kwargs.get('generator'),
            None,
        )
    
    kwargs['latents'] = latents
    return kwargs

def img2img_get_timesteps(self, num_inference_steps, strength, device):
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
    return timesteps, num_inference_steps - t_start

def prepare_timesteps(self, **kwargs):
    self.scheduler.set_timesteps(kwargs.get('num_inference_steps'), device=kwargs.get('device'))
    num_inference_steps = kwargs.get('num_inference_steps')
    strength = kwargs.get('strength')
    device = kwargs.get('device')
    timesteps, num_inference_steps = self.img2img_get_timesteps(num_inference_steps, strength, device)
    latent_timestep = timesteps[:1].repeat(kwargs.get('batch_size') * kwargs.get('num_images_per_prompt'))
    kwargs['timesteps'] = timesteps
    kwargs['num_inference_steps'] = num_inference_steps
    kwargs['latent_timestep'] = latent_timestep
    return kwargs

def img2img_default(self, **kwargs):
    if kwargs.get('strength') is None:
        kwargs['strength'] = 0.75
    if kwargs.get('skip_noise') is None:
        kwargs['skip_noise'] = False
    return kwargs

def img2img_check_inputs(self, **kwargs):
    strength = kwargs.get('strength')
    image = kwargs.get('image')
    
    if strength < 0 or strength > 1:
        raise ValueError(f"Strength must be in [0.0, 1.0] but is {strength}")
    
    if isinstance(image, list):
        for img in image:
            check_single_image_dimensions(img)
    else:
        check_single_image_dimensions(image)
        
    return kwargs

def img2img_preprocess_img(self, **kwargs):
    image = self.image_processor.preprocess(kwargs.get('image'))
    kwargs['image'] = image
    return kwargs

def check_single_image_dimensions(image):
    if isinstance(image, PIL.Image.Image):
        return  # PIL images don't need dimension checking
    if not isinstance(image, torch.Tensor):
        raise ValueError(f"Image must be a PIL image or torch tensor, got {type(image)}")
    if image.ndim != 4:
        raise ValueError(f"Image batch must have 4 dimensions, got {image.ndim}")

def retrieve_latents(encoder_output: Any, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"):
    # 1. If the encoder output is already a tensor, return it
    if isinstance(encoder_output, torch.Tensor):
        return encoder_output

    # 2. If it is a tuple or list, return the first element that is a tensor
    if isinstance(encoder_output, (tuple, list)):
        for elem in encoder_output:
            if isinstance(elem, torch.Tensor):
                return elem

    # 3. If it has a 'latents' attribute, return that
    latent = getattr(encoder_output, "latents", None)
    if latent is not None:
        return latent

    # 4. If it has a 'latent_dist' attribute, sample or take argmax
    if hasattr(encoder_output, "latent_dist"):
        if sample_mode == "sample":
            return encoder_output.latent_dist.sample(generator)
        elif sample_mode == "argmax":
            return encoder_output.latent_dist.mode()

    # 5. If the encoder output is dict-like, try to get 'latents' or the first tensor value
    if isinstance(encoder_output, dict):
        if "latents" in encoder_output:
            return encoder_output["latents"]
        else:
            for v in encoder_output.values():
                if isinstance(v, torch.Tensor):
                    return v

    # 6. As a last resort, iterate over attributes to find a tensor
    for attr in dir(encoder_output):
        try:
            value = getattr(encoder_output, attr)
            if isinstance(value, torch.Tensor):
                return value
        except Exception:
            continue

    raise AttributeError("Could not access latents of provided encoder_output; available attributes: " + str(dir(encoder_output)))