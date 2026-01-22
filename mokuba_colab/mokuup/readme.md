# mokuup
I make the script of [gokayfem/Tile-Upscaler](https://github.com/gokayfem/Tile-Upscaler) to change ckpt, lora, vae and embedding. But it is not compatible with sd and controlnet tile.  
## requirements
Change the runtime type to T4 GPU.  
Next, run next code on Notebook.  
```
!pip install compel
!pip install pyexiv2
!pip install torchsde

#In Kaggle
!pip uninstall diffusers torch torchvision -y
!pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu124
!pip install diffusers==0.34.0
#In Google
!pip uninstall diffusers torch torchvision -y
!pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu126
!pip install diffusers==0.34.0

!pip install py-real-esrgan

import requests,py_real_esrgan,os
url="https://raw.githubusercontent.com/MokubaAttack/scripts/refs/heads/main/mokuba_colab/mokuup/mokuup.py"
path="mokuup.py"
urlData = requests.get(url).content
with open(path ,mode='wb') as f:
  f.write(urlData)

path=os.path.dirname(py_real_esrgan.__file__)+"/model.py"
url="https://raw.githubusercontent.com/MokubaAttack/scripts/refs/heads/main/mokuba_colab/realersgan/model_mod.py"
urlData = requests.get(url).content
with open(path ,mode='wb') as f:
    f.write(urlData)

import mokuup
```
## explanations

## Credits
[gokayfem/Tile-Upscaler](https://github.com/gokayfem/Tile-Upscaler)
