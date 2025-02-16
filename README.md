# SANARubber

This project aims to solve the rigidity problem that diffusers has. Instead of creating a pipeline for each variation and combination, you can just implement it for SANARubber and the user will pick the variations he wants to enable or not. This is based on the base txt2img pipeline of diffusers.

There's a special parameter in this pipeline called "stop_step". It's the exact step you want the denoising to stop at.
How to use
1. install diffusers: pip install git+https://github.com/huggingface/diffusers
2. run tester.py
3. choose whatever appliers you want, but warning, some appliers should be applied later if you're stacking them like promptFusion. Also, if you use inpainting, you can't use img2img

or copy this (change whatever you want, it works just like diffusers)

from rubberDiffusers import StableDiffusionRubberPipeline
pipe=StableDiffusionRubberPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32,local_files_only=True,safety_checker=None, requires_safety_checker=False,
)

# vanilla SANARubber
This works exactly the same as the regular SANA pipeline from the diffusers library.
```
import torch
from SANARubber import SanaRubberPipeline

pipe = SanaRubberPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers", torch_dtype=torch.bfloat16
)
pipe.to("cuda")
pipe.text_encoder.to(torch.bfloat16)
pipe.transformer = pipe.transformer.to(torch.bfloat16)
image = pipe(
    width=1024,
    height=1024,
    num_images_per_prompt=3,
    num_inference_steps=8
)[0]
```
# apply_img2img
Changes whatever image you want to change:
```
from appliers.img2img import apply_img2img

apply_img2img(your_pipe)
image = Image.open('mypic.png') #can be an array of images too. it will create many images as a result
image=your_pipe("a handsome alien",image=image).images[0]
```
Default values:

strength=0.75

skip_noise=False #whether to skip the added noise from the strength procedure. Useful to simulate an efficient hires fix implementation

Requirements:

image= an image or list of images

# apply_multiDiffusion
This will let you apply prompts regionally simultaneously.

Usage:
```
from appliers.multidiffusion import apply_multiDiffusion
apply_multiDiffusion(pipe)
image=pipe(prompt=["your prompt","your second prompt],width=512,height=512,pos=["0:0-512:512"],mask_strengths=[.5]).images[0]
```
Requirements:
prompt=an array of prompts to use

OR

prompt_embeds= an array of prompts embeddings to use

pos=where to apply the strength of the prompt (first pos is the position of the second prompt), the format is either a black and white mask or x0:y0-x1-y1 in pixels that are divisible by 8. MUST BE SAME LENGTH AS prompt/prompt_embeds

mask_strengths= value between 0 and 1 that decide how much strength should be applied in the respective pos. if there are spots that have a value below 1, they will be adjusted such that the first prompt will work there. MUST BE SAME LENGTH AS prompt/prompt_embeds

# undo appliers
Assuming you'd like to use the same pipeline with different functionalities, you can do something like this:
```
apply_img2img(pipe)
apply_multiDiffusion(pipe)

# Add debug prints to check the pos parameter
pos = ["0:0-512:512","512:0-1024:512","512:1024-1024:1024","0:512-512:1024"]
image=Image.open('out0.png')
seed = torch.Generator(device="cuda").manual_seed(1)
image = pipe(
    prompt=['bright moon','red','blue','green','black'],
    multi_diffusion_pos=pos,
    multi_diffusion_mask_strengths=[.7,.7,.7,.7],
    width=1024,
    height=1024,
    num_images_per_prompt=3,
    image=image,
    strength=.5,
    generator=seed,
    num_inference_steps=16
)[0]
pipe.rever()
image = pipe(
    "a lion", num_inference_steps=16,width=1024,
    height=1024,num_images_per_prompt=3,
)[0]

```
in the code above, we generated an image with multidiffusion and image2image, then used the base pipeline.
to reset the pipe just do this:
```
pipe.revert()
```
