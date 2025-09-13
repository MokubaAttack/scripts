# make_safetensors
It is a script that burns a vae and loras in a checkpoint. This script is based on [convert_diffusers_to_original_sdxl.py](https://github.com/huggingface/diffusers/blob/main/scripts/convert_diffusers_to_original_sdxl.py) of huggingface/diffusers.
## requirements
python modules
```
pip install diffusers
pip install torch
pip install FreeSimpleGUI
pip install transformers
pip install accelerate
pip install PEFT
pip install plyer
pip install pyperclip
```
## How to use
1. Run this script.
2. Select the checkpoint file ( .safetensors file ).
3. Input a vae and loras that you want to burn in the checkpoint.  
4. Input the output path ( .safetensors file ).
5. Click run button.
6. After a while, the output file is generated.
