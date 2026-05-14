# merge_ckpt
It is a script that merge SDXL checkpoints. This script supports block merge and DARE merge, tensor merge. 
## requirements
python modules
```
pip install safetensors torch FreeSimpleGUI packaging numpy plyer pyperclip
```
data files  
You need to save data.txt in the directory that you save merge_ckpt.py in.
## How to use
1. Run this script.
2. Select the checkpoint files ( .safetensors file ).
3. Input weights of second checkpoint file. If you input 0.3, the weight of first model is 0.7 and the weight of second model is 0.3.
4. If you want to use a vae of a checkpoint, check the radio button of that checkpoint row. If you don't check, vae parameters also are merged.
5. Input the output path ( .safetensors file ).
6. Click run button.
7. After a while, the output file is generated.
## Credits
+ block merge : [hako-mikan/sd-webui-supermerger](https://github.com/hako-mikan/sd-webui-supermerger)
+ tensor merge : [hako-mikan/sd-webui-supermerger](https://github.com/hako-mikan/sd-webui-supermerger)
+ DARE merge : [martyn/safetensors-merge-supermario](https://github.com/martyn/safetensors-merge-supermario)
