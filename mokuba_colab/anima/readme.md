# About the problem that lora cannot be loaded in hdae/diffusers-anima
When you make images by using [hdae/diffusers-anima](https://github.com/hdae/diffusers-anima) module, you have a error to load lora into the pipeline.  
There was a small mistake in their code. Then I modified [hdae/diffusers-anima/src/diffusers_anima/loaders/lora_pipeline.py](https://github.com/hdae/diffusers-anima/blob/main/src/diffusers_anima/loaders/lora_pipeline.py).  
When you overwrite their code in [modyfied code](https://github.com/MokubaAttack/scripts/blob/main/mokuba_colab/anima/lora_pipeline.py), you can load lora into the pipeline.  

And I introduced hdae/diffusers-anima module to mokucola.   
But you need an abundance of time to make images in Google Colab (you need ~6 minutes to make a 1024-1024 image.), because T4 GPU doesn't support bfloat16.
## requirements
Change the runtime type to T4 GPU.  
Next, run next code on Notebook.  
```
!pip install https://raw.githubusercontent.com/MokubaAttack/scripts/refs/heads/main/mokuba_colab/mokucola-42.10.128.tar.gz

import mokucola
```
## explanations
mokucola.mokuani(  
loras, lora_weights, prompt, n_prompt, pic_number, gs, step, sample, sgm, seed, out_folder, base_safe, j_or_p, url, p, dtype, dev, ser, del_pipe, si, mode, up, Interpolation, step2, ss  
)  
- base_safe : str  
  It is the checkpoint file.
- loras : str list  
  It is the name list of the lora file excluding extension. If there is not that file in the working folder, you must input the absolute path.
- lora_weights : float list  
  It is the lora's weight list.
- prompt : str  
  It is the prompt.
- n_prompt : str  
  It is the negative prompt.
- step : int  
  It is num_inference_steps.
- gs : float  
  It is guidance_scale.
- sample : str  
  It is the scheduler type.
  - flowmatch_euler
  - euler
  - euler_a_rf
  - euler_ancestral_rf
- sgm : str  
  It is the noise schedule and the schedule type.
  - uniform
  - beta
  - simple
  - normal
- pic_number : int  
  It is the number of the output images.
- seed : int or int list  
  It is the seed or the seed list. If you input zero, the random seeds are made.
- out_folder : str  
  It is the output folder path. If the folder doesn't exist, that is made.
- si : bool  
  If you choice True, output images are shown in the output window.
- j_or_p : str  
  It is the format of output files. "j" is JPG format, and "p" is PNG format.
- url : str  
  If you input the webhook url of discord, images are sent to discord.
- del_pipe : bool  
  If you choice True, the mokupipe object is deleted and None is returned.
- x : int  
  It is width of output image.
- y : int  
  It is height of output image.
- ser : str  
  In google colab, please input "colab". In kaggle, please input "kaggle".
- dev : str  
  It is the device that calculates. Choices are cuda and cpu.
- p : mokuanipipe object  
  If you input the return of this module, you can use same pipeline without making the pipeline.
- dtype : str  
  It is the calculation accuracy. Choices are f32 and bf16.
- ss : float  
  It is denoising_strength ( a parameter of hires.fix ).
- mode : int  
  It is the working mode.
  - 0 : normal
  - 1 : hires.fix
- up : float  
  It is the upscale ( a parameter of hires.fix ).
- Interpolation : int or str  
  It is the interpolation method of the upscaling. If you input pth file of ESRGAN, images are upscaled by ESRGAN.
  - 1 : NEAREST
  - 2 : BOX
  - 3 : BILINEAR
  - 4 : HAMMING
  - 5 : BICUBIC
  - 6 : LANCZOS
- step2 : int  
  It is Hires steps ( a parameter of hires.fix ).
- return : mokuanipipe object
## Credits
[hdae/diffusers-anima](https://github.com/hdae/diffusers-anima)
