import torch
from diffusers import SanaPipeline
from SANARubber import SanaRubberPipeline
from appliers.img2img import apply_img2img
from appliers.multidiffusion import apply_multiDiffusion
from PIL import Image

pipe = SanaRubberPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers", torch_dtype=torch.bfloat16
)
pipe.to("cuda")
pipe.text_encoder.to(torch.bfloat16)
pipe.transformer = pipe.transformer.to(torch.bfloat16)
apply_img2img(pipe)
apply_multiDiffusion(pipe)

pos = ["0:0-512:512","512:0-1024:512","512:1024-1024:1024","0:512-512:1024"]
image=Image.open('out000.png')
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
pipe.revert()

# Save each region separately
c = 0
for i in image:
    i.save(f'out{c}.png')
    c += 1
image = pipe(
    "a lion", num_inference_steps=16,width=1024,
    height=1024,num_images_per_prompt=3,image=image,
    strength=.5,
)[0]
# Save each region separately
c = 0
for i in image:
    i.save(f'out1{c}.png')
    c += 1
apply_img2img(pipe)
apply_multiDiffusion(pipe)
pos = ["0:0-512:512","512:0-1024:512","512:1024-1024:1024","0:512-512:1024"]
image=Image.open('out000.png')
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
pipe.revert()
c = 0
for i in image:
    i.save(f'out2{c}.png')
    c += 1