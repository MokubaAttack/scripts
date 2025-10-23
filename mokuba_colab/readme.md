# mokuba_calob
I make My Own Modules from the workflow that I use whenever I make images. My workflow is based on Hires.fix. That works on Google Colab and Kaggle by torch.float16. And that doesn't work for Flux.<br>
## requirements
Change the runtime type to T4 GPU.<br>
Next, run next code on Notebook. ( folder_path is the folder path that you save mokuba_colab2.py in. )<br>
```
!pip install compel==2.2.1
#In Kaggle
!pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu124
#In Google
!pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu126

import sys
import os

module_path = os.path.abspath( folder_path )
sys.path.append( module_path )

import mokuba_colab2
```
## explanations
mokuba_colab2.text2image(<br>
&nbsp;&nbsp;&nbsp;&nbsp;loras, lora_weights, prompt, n_prompt, t, prog_ver, pic_number, gs, f_step, step, ss, cs, Interpolation,<br>&nbsp;&nbsp;&nbsp;&nbsp;sample, seed, out_folder, pos_emb, neg_emb, base_safe, vae_safe<br>
)
- loras : str list ( default : [] ) It is the name list of the lora file excluding extension. If there is not that file in the working folder, you must input the absolute path.
- lora_weights : float list ( default : [] ) It is the lora's weight list.
- prompt : str ( default : "" ) It is the prompt.
- n_prompt : str (default : "" ) It is the negative prompt.
- t : str ( default : "v" ) It is the output size of images. Normal mode needs about 3 minutes to make 1 image. Large mode needs about 5 minutes.
  - v : vertically long ( 960-1280 )
  - h : horizontally long ( 1280-960 )
  - s : square ( 1280-1280 )
  - vl : large mode ( 1200-1600 )
  - hl : large mode ( 1600-1200 )
  - sl : large mode ( 1600-1600 )
  - initial width, output width, initial height, output height : You may directly input it.
- prog_ver : int ( default : 2 ) It is the working program version.
  - 0 : normal diffusers
  - 1 : hires.fix
  - 2 : my own workflow
- pic_number : int ( default : 10 ) It is the number of the output images.
- gs : float ( default : 7 ) It is guidance_scale ( a parameter of StableDiffusion ).
- f_step : int ( default : 30 ) It is num_inference_steps ( a parameter of StableDiffusion ). 
- step : int ( default : 20 ) It is Hires steps ( a parameter of hires.fix ).
- ss : float ( default : 0.5 ) It is denoising_strength ( a parameter of hires.fix ).
- cs : int ( default : 1 ) It is clip_skip ( a parameter of StableDiffusion ).
- Interpolation : int ( default : 3 ) It is the interpolation method of the upscaling.
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
  - DPM++ 2M
  - DPM++ SDE
  - DPM++ 2M SDE
  - DPM++ 3M SDE
  - LMS Karras
  - DPM2 Karras
  - DPM2 a Karras
  - DPM++ 2M Karras
  - DPM++ SDE Karras
  - DPM++ 2M SDE Karras
  - DPM++ 3M SDE Karras
  - DDIM
  - PLMS
  - UniPC
  - LCM
- seed : int or int list ( default : 0 ) It is the seed or the seed list. If you input zero, the random seeds are made.
- out_folder : str ( default : "data" ) It is the output folder path. If the folder doesn't exist, that is made.
- pos_emb : str list ( default : [] ) It is the positive embedding file list.
- neg_emb : str list ( default : [] ) It is the negative embedding file list.
- base_safe : str ( default : "base.safetensors" ) It is the checkpoint file.
- vae_safe : str ( default : "vae.safetensors" ) It is the vae file. If you select the file that doesn't exist, Normal Vae is used.
- return : int list ( error : [] ) It is the seed list.
  
Output files are images( name : (index)_(the seed).png ) ~~and the logfile. The logfile is writen input parameters~~.  
<mark>If safetensors files have CivitAi's Version ID in a item of "id" of metadata, Generation metadata is baked in Output files.  
The metadata is read in CivitAi.</mark>
## my own workflow
![flow image](https://github.com/MokubaAttack/scripts/blob/main/mokuba_colab/flow_image.jpg)
## mokuba_diffusers
mokuba_diffusers.ipynb is a file of Google Colab. It is rewritten from code that I use usually.  
You need to input CivitAi's token in "civitai" of the secret key.
Please use it.
