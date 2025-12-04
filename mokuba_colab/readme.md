# mokuba_calob
I make My Own Modules from the workflow that I use whenever I make images. My workflow is based on Hires.fix. That works on Google Colab and Kaggle by torch.float16. And that doesn't work for Flux.<br>
## requirements
Change the runtime type to T4 GPU.<br>
Next, run next code on Notebook.<br>
```
!pip install compel
!pip install pyexiv2
!pip install torchsde
#In Kaggle
!pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu124
#In Google
!pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu126

url="https://raw.githubusercontent.com/MokubaAttack/scripts/refs/heads/main/mokuba_colab/mokuba_colab.py"
path="mokuba_colab.py"
import requests
urlData = requests.get(url).content

with open(path ,mode='wb') as f:
  f.write(urlData)

import mokuba_colab
```
## explanations
mokuba_colab.text2image(<br>
loras, lora_weights, prompt, n_prompt, t, prog_ver, pic_number, gs, f_step, step, ss, cs, Interpolation,sample, sgm, seed, out_folder, pos_emb, neg_emb, base_safe, vae_safe, pas. j_or_p<br>
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
- return : int list ( error : [] ) It is the seed list.
  
Output files are images( name : (index)_(the seed).png or (index)_(the seed).jpg ).  
If safetensors files have CivitAi's Version ID in a item of "id" of metadata (In case of a lora file, lora's weight in a item of "weight" is needed too) , Generation metadata is baked in Output files.  
(Example)  
lora file  
"id" : "111111", "weight" : "1"  
merged lora file  
"id" : "111111,222222", "weight" : "0.5,0.5"  
ckpt file  
"id" : "123456"  
The metadata is read in CivitAi.
## my own workflow
![flow image](https://github.com/MokubaAttack/scripts/blob/main/mokuba_colab/flow_image.jpg)
## mokuba_diffusers
mokuba_diffusers.ipynb is a file of Google Colab. It is rewritten from code that I use usually.  
You need to input CivitAi's token in "civitai" of the secret key.  
Please use it.  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MokubaAttack/scripts/blob/main/mokuba_colab/mokuba_diffusers.ipynb)
