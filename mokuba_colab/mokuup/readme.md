# mokuup
I make the script of [gokayfem/Tile-Upscaler](https://github.com/gokayfem/Tile-Upscaler) to change ckpt, lora, vae and embedding. 
## requirements
Change the runtime type to T4 GPU.  
Next, run next code on Notebook.  
```
!pip install compel
!pip install pyexiv2
!pip install torchsde

!pip uninstall diffusers torch torchvision -y
!pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu126
!pip install diffusers==0.34.0

!pip install py-real-esrgan

import requests,py_real_esrgan,os
url="https://raw.githubusercontent.com/MokubaAttack/scripts/refs/heads/main/mokuba_colab/mokuba_colab.py"
path="mokuba_colab.py"
urlData = requests.get(url).content
with open(path ,mode='wb') as f:
  f.write(urlData)

path=os.path.dirname(py_real_esrgan.__file__)+"/model.py"
url="https://raw.githubusercontent.com/MokubaAttack/scripts/refs/heads/main/mokuba_colab/realersgan/model_mod.py"
urlData = requests.get(url).content
with open(path ,mode='wb') as f:
    f.write(urlData)

import mokuba_colab
```
## explanations
mokuba_colab.mokuup(  
img_path, base_safe, vae_safe, loras, lora_weights, up, gs, step, ss, cs, Interpolation, sample, sgm, seed, pos_emb, neg_emb, pag, out_folder, j_or_p, p, prompt, n_prompt, ccs  
)  
- img_path : str ( default : "" ) It is a path of a image that is upscaled.
- loras : str list ( default : [] ) It is the name list of the lora file excluding extension. If there is not that file in the working folder, you must input the absolute path.
- lora_weights : float list ( default : [] ) It is the lora's weight list.
- up : float ( default : 2 ) It is the upscale.
- pic_number : int ( default : 10 ) It is the number of the output images.
- gs : float ( default : 7 ) It is guidance_scale ( a parameter of StableDiffusion ).
- step : int ( default : 20 ) It is num_inference_steps ( a parameter of StableDiffusion ).
- ss : float ( default : 0.5 ) It is denoising_strength ( a parameter of hires.fix ).
- cs : int ( default : 1 ) It is clip_skip ( a parameter of StableDiffusion ).
- Interpolation : int or str ( default : 3 ) It is the interpolation method of the upscaling. If you input pth file of ESRGAN, images are upscaled by ESRGAN.
  - 1 : NEAREST
  - 2 : BOX
  - 3 : BILINEAR
  - 4 : HAMMING
  - 5 : BICUBIC
  - 6 : LANCZOS
- sample : str ( default : "DDIM" ) It is the scheduler type.
  - Euler a
  - Euler
  - LMS
  - Heun
  - DPM2
  - DPM2 a
  - DPM++
  - DPM++ 2M
  - DPM++ SDE
  - DPM++ 2M SDE
  - DPM++ 3M SDE
  - DDIM
  - PLMS
  - UniPC
  - LCM
- sgm : str (default : "" ) It is the noise schedule and the schedule type.
  - Karras
  - sgm_uniform
  - simple
  - exponential
  - beta
- seed : int or int list ( default : 0 ) It is the seed or the seed list. If you input zero, the random seeds are made.
- out_folder : str ( default : "data" ) It is the output folder path. If the folder doesn't exist, that is made.
- pos_emb : str list ( default : [] ) It is the positive embedding file list.
- neg_emb : str list ( default : [] ) It is the negative embedding file list.
- base_safe : str ( default : "base.safetensors" ) It is the checkpoint file.
- vae_safe : str ( default : "vae.safetensors" ) It is the vae file. If you select the file that doesn't exist, Normal Vae is used.
- pag : float ( default : 3.0 ) It is pag_scale ( a parameter of PAG ).
- j_or_p : str ( default : "j" ) It is the format of output files. "j" is JPG format, and "p" is PNG format.
- p : mokupipe object ( default : None ) If you input the return of this module, you can use same pipeline without making the pipeline.
- prompt : str ( default : "masterpiece,best quality,ultra detailed" ) It is the prompt.
- n_prompt : str (default : "worst quality,low quality,normal quality" ) It is the negative prompt.
- ccs : float ( default : None ) It is controlnet_conditioning_scale ( a parameter of StableDiffusion ). If you input None, controlnet tile are not used.
- return : mokupipe object

Image files are output by naming (index)(the seed).png or (index)(the seed).jpg in the output folder path. 
If safetensors files have CivitAi's Version ID in a item of "id" of metadata (In case of a lora file, lora's weight in a item of "weight" is needed too) , Generation metadata is baked in Output files.  
(Example)  
lora file  
"id" : "111111", "weight" : "1"  
merged lora file  
"id" : "111111,222222", "weight" : "0.5,0.5"  
ckpt file  
"id" : "123456"  
The metadata is read in CivitAi.
## Credits
workflow : [gokayfem/Tile-Upscaler](https://github.com/gokayfem/Tile-Upscaler)  
controlnet sd : [lllyasviel/control_v11f1e_sd15_tile](https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile)  
controlnet sdxl : [OzzyGT/SDXL_Controlnet_Tile_Realistic](https://huggingface.co/OzzyGT/SDXL_Controlnet_Tile_Realistic)  
