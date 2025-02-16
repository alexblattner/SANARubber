import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import torch
from packaging import version
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import SanaPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.configuration_utils import FrozenDict
from diffusers.utils import (
    BACKENDS_MAPPING,
    USE_PEFT_BACKEND,
    is_bs4_available,
    is_ftfy_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.models import AutoencoderDC, SanaTransformer2DModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescales noise prediction tensor based on guidance_rescale.
    See: https://arxiv.org/pdf/2305.08891.pdf
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Sets timesteps (or sigmas) on the scheduler and returns
    the timestep schedule and effective number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The scheduler {scheduler.__class__} does not support custom timesteps."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accepts_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_sigmas:
            raise ValueError(
                f"The scheduler {scheduler.__class__} does not support custom sigmas."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class SanaRubberPipeline(SanaPipeline):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: AutoModelForCausalLM,
        vae: AutoencoderDC,
        transformer: SanaTransformer2DModel,
        scheduler: DPMSolverMultistepScheduler,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        
        # Define a chain of functions that process the pipeline sequentially.
        self.pipeline_functions = [
            self.set_dimensions,
            self.check_inputs_fn, 
            self.determine_batch_size_fn,
            self.set_guidance_scale,
            self.encode_prompt_fn,
            self.prepare_timesteps_fn,
            self.prepare_latents_fn,
            self.prepare_extra_step_kwargs_fn,
            self.guidance_scale_embedding_fn,
            self.denoising_loop_fn,
            self.decode_and_postprocess_fn,
            self.return_output_fn
        ]

        # Functions for each denoising step.
        self.step_functions = [
            self.prepare_latent_model_input,
            self.get_noise_pred,
            self.apply_guidance,
            self.step_scheduler,
            self.handle_callbacks
        ]
        self.revert_functions = []

    def set_dimensions(self, **kwargs):
        height = kwargs.get("height")
        width = kwargs.get("width")
        if height is None or width is None:
            sample_size = self.transformer.config.sample_size
            if isinstance(sample_size, int):
                height = width = sample_size * self.vae_scale_factor
            else:
                height, width = sample_size[0] * self.vae_scale_factor, sample_size[1] * self.vae_scale_factor
        kwargs["height"] = height
        kwargs["width"] = width
        return kwargs

    def check_inputs_fn(self, **kwargs):
        self.check_inputs(
            prompt=kwargs.get("prompt"),
            height=kwargs.get("height"),
            width=kwargs.get("width"),
            callback_on_step_end_tensor_inputs=kwargs.get("callback_on_step_end_tensor_inputs"),
            negative_prompt=kwargs.get("negative_prompt"),
            prompt_embeds=kwargs.get("prompt_embeds"),
            negative_prompt_embeds=kwargs.get("negative_prompt_embeds"),
            prompt_attention_mask=kwargs.get("prompt_attention_mask"),
            negative_prompt_attention_mask=kwargs.get("negative_prompt_attention_mask"),
        )
        return kwargs

    def determine_batch_size_fn(self, **kwargs):
        prompt = kwargs.get("prompt")
        if prompt is not None:
            batch_size = len(prompt) if isinstance(prompt, list) else 1
        else:
            batch_size = kwargs.get("prompt_embeds").shape[0]
        kwargs["batch_size"] = batch_size
        return kwargs
    def set_guidance_scale(self, **kwargs):
        do_classifier_free_guidance = kwargs.get("do_classifier_free_guidance", kwargs.get("guidance_scale", 7.5) > 1.0)
        kwargs["do_classifier_free_guidance"] = do_classifier_free_guidance
        return kwargs
    def encode_prompt_fn(self, **kwargs):
        prompt = kwargs.get("prompt")
        negative_prompt = kwargs.get("negative_prompt")
        if negative_prompt is None:
            negative_prompt = ""
        num_images_per_prompt = kwargs.get("num_images_per_prompt", 1)
        device = kwargs.get("device")
        prompt_embeds = kwargs.get("prompt_embeds")
        negative_prompt_embeds = kwargs.get("negative_prompt_embeds")
        prompt_attention_mask = kwargs.get("prompt_attention_mask")
        negative_prompt_attention_mask = kwargs.get("negative_prompt_attention_mask")
        clean_caption = kwargs.get("clean_caption", False)
        max_sequence_length = kwargs.get("max_sequence_length", 300)
        complex_human_instruction = kwargs.get("complex_human_instruction", None)
        lora_scale = kwargs.get("lora_scale", None)
        do_classifier_free_guidance = kwargs.get("guidance_scale")
        
        (prompt_embeds, prompt_attention_mask,
         negative_prompt_embeds, negative_prompt_attention_mask) = self.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
            complex_human_instruction=complex_human_instruction,
            lora_scale=lora_scale,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
        kwargs["prompt_embeds"] = prompt_embeds
        kwargs["prompt_attention_mask"] = prompt_attention_mask
        kwargs["negative_prompt_embeds"] = negative_prompt_embeds
        kwargs["negative_prompt_attention_mask"] = negative_prompt_attention_mask
        kwargs["do_classifier_free_guidance"] = do_classifier_free_guidance
        return kwargs

    def prepare_timesteps_fn(self, **kwargs):
        device = kwargs.get("device")
        num_inference_steps = kwargs.get("num_inference_steps")
        timesteps = kwargs.get("timesteps")
        sigmas = kwargs.get("sigmas")
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, sigmas)
        kwargs["timesteps"] = timesteps
        kwargs["num_inference_steps"] = num_inference_steps
        return kwargs

    def prepare_latents_fn(self, **kwargs):
        batch_size = kwargs.get("batch_size")
        num_channels_latents = self.transformer.config.in_channels
        height = kwargs.get("height")
        width = kwargs.get("width")
        dtype = kwargs.get("dtype", torch.float32)
        device = kwargs.get("device")
        generator = kwargs.get("generator")
        latents = kwargs.get("latents")
        latents = self.prepare_latents(batch_size * kwargs.get("num_images_per_prompt", 1), num_channels_latents, height, width, dtype, device, generator, latents)
        kwargs["latents"] = latents
        return kwargs

    def prepare_extra_step_kwargs_fn(self, **kwargs):
        generator = kwargs.get("generator")
        eta = kwargs.get("eta", 0.0)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        kwargs["extra_step_kwargs"] = extra_step_kwargs
        return kwargs

    def guidance_scale_embedding_fn(self, **kwargs):
        # For Sana, guidance scale embedding might not be needed.
        return kwargs

    def denoising_loop_fn(self, **kwargs):
        timesteps = kwargs["timesteps"]
        num_inference_steps = kwargs["num_inference_steps"]
        num_warmup_steps = kwargs.get("num_warmup_steps", max(len(timesteps) - num_inference_steps * self.scheduler.order, 0))
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                for step_func in self.step_functions:
                    kwargs = step_func(i, t, **kwargs)
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        return kwargs

    def decode_and_postprocess_fn(self, **kwargs):
        output_type = kwargs.get("output_type", "pil")
        device = kwargs.get("device")
        latents = kwargs.get("latents")
        if output_type == "latent":
            image = latents
        else:
            latents = latents.to(self.vae.dtype)
            try:
                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            except torch.cuda.OutOfMemoryError as e:
                import warnings
                warnings.warn(f"{e}. Consider using VAE tiling for large images.")
                raise e
            if kwargs.get("use_resolution_binning", False):
                orig_width = kwargs.get("orig_width", kwargs.get("width"))
                orig_height = kwargs.get("orig_height", kwargs.get("height"))
                image = self.image_processor.resize_and_crop_tensor(image, orig_width, orig_height)
            image = self.image_processor.postprocess(image, output_type=output_type)
        kwargs["image"] = image
        return kwargs

    def return_output_fn(self, **kwargs):
        image = kwargs.get("image")
        if kwargs.get("return_dict", True):
            from diffusers.pipelines.sana.pipeline_output import SanaPipelineOutput
            return SanaPipelineOutput(images=image)
        return image

    # --- Step functions for the denoising loop ---
    def prepare_latent_model_input(self, i: int, t: torch.Tensor, **kwargs):
        latents = kwargs["latents"]
        prompt_embeds = kwargs["prompt_embeds"]
        do_classifier_free_guidance = kwargs.get("do_classifier_free_guidance", False)

        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = latent_model_input.to(self.transformer.dtype)
        kwargs["latent_model_input"] = latent_model_input
        return kwargs

    def get_noise_pred(self, i: int, t: torch.Tensor, **kwargs):
        latent_model_input = kwargs["latent_model_input"]
        prompt_embeds = kwargs["prompt_embeds"]
        encoder_attention_mask = kwargs.get("prompt_attention_mask")
        extra_attention_kwargs = kwargs.get("attention_kwargs", {})
        batch_size = latent_model_input.shape[0]
        if t.dim() == 0 or t.shape[0] == 1:
            t = t.expand(batch_size).to(latent_model_input.dtype)

        # Ensure the attention mask has the correct batch dimension.
        if encoder_attention_mask is not None:
            # If its first dimension is larger than the current batch size, slice it.
            if encoder_attention_mask.shape[0] > batch_size:
                encoder_attention_mask = encoder_attention_mask[:batch_size, :]
            # If its number of elements doesn't match batch_size*seq_length, try to fix it.
            expected_elements = batch_size * encoder_attention_mask.shape[-1]
            if encoder_attention_mask.numel() != expected_elements:
                # Attempt to reshape it to [batch_size, sequence_length]
                encoder_attention_mask = encoder_attention_mask.view(batch_size, -1)
        noise_pred = self.transformer(
            latent_model_input,
            timestep=t,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
            **extra_attention_kwargs,
        )[0]
        noise_pred = noise_pred.float()
        kwargs["noise_pred"] = noise_pred
        return kwargs

    def apply_guidance(self, i: int, t: torch.Tensor, **kwargs):
        noise_pred = kwargs["noise_pred"]
        do_classifier_free_guidance = kwargs.get("do_classifier_free_guidance", False)
        guidance_scale = kwargs.get("guidance_scale", 7.5)
        guidance_rescale = kwargs.get("guidance_rescale", 0.0)
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            if guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
        latent_channels = self.transformer.config.in_channels
        if self.transformer.config.out_channels // 2 == latent_channels:
            noise_pred = noise_pred.chunk(2, dim=1)[0]
        kwargs["noise_pred"] = noise_pred
        return kwargs

    def step_scheduler(self, i: int, t: torch.Tensor, **kwargs):
        noise_pred = kwargs["noise_pred"]
        latents = kwargs["latents"]
        extra_step_kwargs = kwargs.get("extra_step_kwargs", {})
        extra_step_kwargs['generator']=torch.Generator(device="cuda").manual_seed(1)
        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
        kwargs["latents"] = latents
        return kwargs

    def handle_callbacks(self, i: int, t: torch.Tensor, **kwargs):
        callback_on_step_end = kwargs.get("callback_on_step_end")
        if callback_on_step_end is not None:
            callback_kwargs = {k: kwargs.get(k) for k in kwargs.get("callback_on_step_end_tensor_inputs", [])}
            callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
            kwargs["latents"] = callback_outputs.pop("latents", kwargs["latents"])
            kwargs["prompt_embeds"] = callback_outputs.pop("prompt_embeds", kwargs["prompt_embeds"])
            kwargs["negative_prompt_embeds"] = callback_outputs.pop("negative_prompt_embeds", kwargs["negative_prompt_embeds"])
        return kwargs

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        num_inference_steps: int = 20,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 4.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        clean_caption: bool = False,
        use_resolution_binning: bool = False,
        max_sequence_length: int = 300,
        complex_human_instruction: Optional[List[str]] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ) -> Union[Any, Tuple]:
        pipeline_kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "timesteps": timesteps,
            "sigmas": sigmas,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt,
            "num_images_per_prompt": num_images_per_prompt,
            "eta": eta,
            "generator": generator,
            "latents": latents,
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "output_type": output_type,
            "return_dict": return_dict,
            "callback_on_step_end": callback_on_step_end,
            "callback_on_step_end_tensor_inputs": callback_on_step_end_tensor_inputs,
            "clean_caption": clean_caption,
            "use_resolution_binning": use_resolution_binning,
            "max_sequence_length": max_sequence_length,
            "complex_human_instruction": complex_human_instruction,
            "lora_scale": lora_scale,
            "device": self._execution_device,
            "dtype": self.transformer.dtype,
            **kwargs
        }
        for func in self.pipeline_functions:
            pipeline_kwargs = func(**pipeline_kwargs)
        return pipeline_kwargs
    
    def revert(self):
        print("revert")
        for func in self.revert_functions:
            func()
            del func
        self.revert_functions=[]