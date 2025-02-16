import torch
import torch.nn.functional as F
import numpy as np
import PIL
from PIL import Image
from functools import partial
from diffusers.utils import (
    PIL_INTERPOLATION,
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    replace_example_docstring,
)
try:
    from diffusers.utils import is_compiled_module
except ImportError:
    from diffusers.utils.torch_utils import is_compiled_module

# Helper to locate a function by name in a list.
def find_index(functions, name):
    for index, func in enumerate(functions):
        if (hasattr(func, "__name__") and func.__name__ == name) or \
           (hasattr(func, "func") and hasattr(func.func, "__name__") and func.func.__name__ == name):
            return index
    return None

### MULTIDIFFUSION APPLIER FUNCTIONS ###

def apply_multiDiffusion(pipe):
    # Replace determine_batch_size_fn with the multidiffusion version.
    idx00 = find_index(pipe.pipeline_functions, "determine_batch_size_fn")
    pipe.inner_determine_batch_size_multiDiffusion = pipe.determine_batch_size_fn
    pipe.determine_batch_size_fn = partial(determine_batch_size, pipe)
    pipe.pipeline_functions[idx00] = pipe.determine_batch_size_fn

    # Replace encode_prompt_fn with the multidiffusion version.
    idx0 = find_index(pipe.pipeline_functions, "encode_prompt_fn")
    pipe.inner_encode_prompt_multiDiffusion = pipe.encode_prompt_fn
    pipe.encode_prompt_fn = partial(encode_prompt, pipe)
    pipe.pipeline_functions[idx0] = pipe.encode_prompt_fn

    # Insert mask preparation before the denoising loop.
    idx = find_index(pipe.pipeline_functions, "denoising_loop_fn")
    pipe.pipeline_functions.insert(idx, partial(mask_prepare_multiDiffusion, pipe))
    
    # For the noise-prediction step, remove original functions and replace with our multidiffusion version.
    si = find_index(pipe.step_functions, "get_noise_pred")
    ei = find_index(pipe.step_functions, "step_scheduler")
    pipe.multiDiffusionFunctions = [pipe.step_functions.pop(si) for _ in range(ei - si)]
    pipe.inner_get_noise_pred_multiDiffusion = partial(get_noise_pred_multidiffusion, pipe)
    pipe.step_functions.insert(si, pipe.inner_get_noise_pred_multiDiffusion)

    # Define a remover function to restore the original behavior.
    def remover_multiDiffusion():
        pipe.step_functions.pop(si)
        delattr(pipe, f"inner_get_noise_pred_multiDiffusion")
        
        for f in reversed(pipe.multiDiffusionFunctions):
            pipe.step_functions.insert(si, f)
        delattr(pipe, f"multiDiffusionFunctions")
        
        # 3. Remove mask preparation
        pipe.pipeline_functions.pop(idx)
        
        # 2. Restore encode_prompt
        pipe.encode_prompt_fn = pipe.inner_encode_prompt_multiDiffusion
        pipe.pipeline_functions[idx0] = pipe.encode_prompt_fn
        del pipe.inner_encode_prompt_multiDiffusion
        
        # 1. Restore batch size determination
        pipe.determine_batch_size_fn = pipe.inner_determine_batch_size_multiDiffusion
        pipe.pipeline_functions[idx00] = pipe.determine_batch_size_fn
        del pipe.inner_determine_batch_size_multiDiffusion

    pipe.revert_functions.insert(0, remover_multiDiffusion)

def determine_batch_size(self, **kwargs):
    prompt = kwargs.get('prompt')
    prompt_embeds = kwargs.get('prompt_embeds')
    orig_prompt, orig_embeds = prompt, prompt_embeds
    kwargs['prompt'] = prompt[0] if (prompt is not None and isinstance(prompt, list)) else prompt
    kwargs['prompt_embeds'] = prompt_embeds[0] if (prompt_embeds is not None and isinstance(prompt_embeds, list)) else prompt_embeds
    kwargs = self.inner_determine_batch_size_multiDiffusion(**kwargs)
    kwargs['prompt'] = orig_prompt
    kwargs['prompt_embeds'] = orig_embeds
    return kwargs

def encode_prompt(self, **kwargs):
    """
    For each prompt (and negative prompt if provided), call the inner encoder and collect the prompt embeddings
    and attention masks. Set the default outputs (used in the regular run) to the first prompt's values, and
    store the full lists for multidiffusion use.
    """
    prompt_list = kwargs.get('prompt')
    prompt_embeds = kwargs.get('prompt_embeds')
    neg_prompt_list = kwargs.get('negative_prompt')
    neg_prompt_embeds = kwargs.get('negative_prompt_embeds')
    
    multi_embeds = []
    multi_masks = []
    multi_neg_embeds = []
    multi_neg_masks = []
    
    num_prompts = len(prompt_list) if prompt_list is not None else len(prompt_embeds)
    
    for i in range(num_prompts):
        local_kwargs = kwargs.copy()
        if prompt_list is not None:
            local_kwargs['prompt'] = prompt_list[i]
        if prompt_embeds is not None and isinstance(prompt_embeds, list):
            local_kwargs['prompt_embeds'] = prompt_embeds[i]
        if neg_prompt_list is not None:
            local_kwargs['negative_prompt'] = neg_prompt_list[i] if isinstance(neg_prompt_list, list) else neg_prompt_list
        if neg_prompt_embeds is not None and isinstance(neg_prompt_embeds, list):
            local_kwargs['negative_prompt_embeds'] = neg_prompt_embeds[i]
        
        processed = self.inner_encode_prompt_multiDiffusion(**local_kwargs)
        multi_embeds.append(processed['prompt_embeds'])
        if 'prompt_attention_mask' in processed:
            multi_masks.append(processed['prompt_attention_mask'])
        if 'negative_prompt_embeds' in processed:
            multi_neg_embeds.append(processed['negative_prompt_embeds'])
        if 'negative_attention_mask' in processed:
            multi_neg_masks.append(processed['negative_attention_mask'])
    
    kwargs['prompt_embeds'] = multi_embeds[0]
    if multi_masks:
        kwargs['prompt_attention_mask'] = multi_masks[0]
    if multi_neg_embeds:
        kwargs['negative_prompt_embeds'] = multi_neg_embeds[0]
    if multi_neg_masks:
        kwargs['negative_attention_mask'] = multi_neg_masks[0]
    
    kwargs['multi_diffusion_prompt_embeds'] = multi_embeds
    if multi_masks:
        kwargs['multi_diffusion_attention_masks'] = multi_masks
    if multi_neg_embeds:
        kwargs['multi_diffusion_negative_prompt_embeds'] = multi_neg_embeds
    if multi_neg_masks:
        kwargs['multi_diffusion_negative_attention_masks'] = multi_neg_masks
    
    return kwargs

def get_noise_pred_multidiffusion(self, i: int, t: torch.Tensor, **kwargs):
    """
    For each region mask:
      1. Reset the generator state so noise is consistent.
      2. Set region-specific prompt embeddings and attention masks.
      3. Call the inner noise-prediction function and then apply guidance.
      4. Resize the region's 2D mask to match the latent spatial dimensions.
      5. Expand the mask to match batch and channel dimensions.
      6. Multiply the noise prediction by the expanded mask.
      7. Sum over all regions.
    """
    mask_list = kwargs.get('multi_diffusion_mask_list', [])
    multi_embeds = kwargs.get('multi_diffusion_prompt_embeds', [])
    multi_attn = kwargs.get('multi_diffusion_attention_masks', None)
    orig_embeds = kwargs['prompt_embeds']
    orig_latents = kwargs['latent_model_input']
    
    generator = kwargs.get('generator')
    init_state = generator.get_state() if generator is not None else None
    noise_preds = []
    
    for count, mask_2d in enumerate(mask_list):
        if init_state is not None:
            generator.set_state(init_state)
        
        # Use region-specific embeddings if available.
        if multi_embeds and count < len(multi_embeds):
            kwargs['prompt_embeds'] = multi_embeds[count]
            bs = kwargs['prompt_embeds'].shape[0]
            if orig_latents.shape[0] != bs:
                kwargs['latent_model_input'] = orig_latents.repeat(bs, 1, 1, 1)
            if multi_attn and count < len(multi_attn):
                attn = multi_attn[count]
                if isinstance(attn, list):
                    attn = torch.tensor(attn)
                if attn.shape[0] != bs:
                    kwargs['prompt_attention_mask'] = attn.repeat(bs, 1)
                else:
                    kwargs['prompt_attention_mask'] = attn
        else:
            kwargs['prompt_embeds'] = orig_embeds
        for i in self.multiDiffusionFunctions:
            kwargs = i(i, t, **kwargs)
        current_pred = kwargs["noise_pred"]
        
        # IMPORTANT FIX: Use the latent input spatial dimensions as target.
        target_dims = kwargs["latent_model_input"].shape[-2:]
        resized_mask = F.interpolate(
            mask_2d.unsqueeze(0).unsqueeze(0),
            size=target_dims,
            mode="bilinear",
            align_corners=False,
        )
        expanded_mask = resized_mask.expand(current_pred.shape[0], current_pred.shape[1], -1, -1)
        expanded_mask = expanded_mask.to(device=current_pred.device, dtype=current_pred.dtype)
        
        noise_preds.append(current_pred * expanded_mask)
    
    total_noise_pred = sum(noise_preds)
    kwargs["prompt_embeds"] = orig_embeds
    kwargs["latent_model_input"] = orig_latents
    kwargs["noise_pred"] = total_noise_pred
    return kwargs

def resize_mask(mask: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    if mask.ndim > 2:
        mask = mask.squeeze()
    resized = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    return resized

def mask_prepare_multiDiffusion(self, **kwargs):
    
    def create_rectangular_mask(height, width, y_start, x_start, block_height, block_width, strength, device='cpu'):
        mask = torch.zeros(height, width, device=device)
        mask[y_start:y_start+block_height, x_start:x_start+block_width] = strength
        cov = (mask > 0).float().mean().item() * 100
        return mask

    mask_list = []
    dtype = kwargs.get("dtype")
    pos = kwargs["multi_diffusion_pos"]
    height = kwargs["height"]
    width = kwargs["width"]
    strengths = kwargs.get("multi_diffusion_mask_strengths", [])
    plen = len(pos) if pos is not None else 0
    latent_height = height // 32
    latent_width = width // 32

    for i in range(plen):
        if isinstance(pos[i], str):
            pos_base = pos[i].split("-")
            pos_start = pos_base[0].split(":")
            pos_end = pos_base[1].split(":")
            x_start = int(pos_start[0]) // 32
            y_start = int(pos_start[1]) // 32
            x_end = int(pos_end[0]) // 32
            y_end = int(pos_end[1]) // 32
            block_height = y_end - y_start
            block_width = x_end - x_start
            one_filter = create_rectangular_mask(
                latent_height, latent_width,
                y_start, x_start,
                block_height, block_width,
                strengths[i],
                device=kwargs["device"]
            )
        else:
            img = pos[i].convert("L").resize((latent_width, latent_height))
            np_data = np.array(img, dtype=np.float32) / 255.0
            np_data = (np_data > 0.5).astype(np.float32)
            one_filter = torch.from_numpy(np_data).to(kwargs["device"])
            one_filter *= strengths[i]
        if dtype:
            one_filter = one_filter.to(dtype)
        cov = (one_filter > 0).float().mean().item() * 100
        mask_list.append(one_filter)
    
    base_mask = torch.ones(latent_height, latent_width, device=kwargs["device"])
    if dtype:
        base_mask = base_mask.to(dtype)
    combined = torch.zeros_like(base_mask)
    for mask in mask_list:
        combined = torch.max(combined, mask)
    base_mask = 1.0 - combined
    base_cov = (base_mask > 0).float().mean().item() * 100
    
    mask_list.insert(0, base_mask)
    kwargs["multi_diffusion_mask_list"] = mask_list
    
    total = torch.zeros_like(base_mask)
    for i, mask in enumerate(mask_list):
        m_cov = (mask > 0).float().mean().item() * 100
        total += mask
    overall = (total > 0).float().mean().item() * 100
    min_val = total.min().item()
    max_val = total.max().item()
    return kwargs
