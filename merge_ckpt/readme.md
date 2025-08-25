# merge_ckpt
It is a script that merge checkpoints.
## requirements
python modules
```
pip install safetensors==0.6.2
pip install torch==2.8.0
pip install FreeSimpleGUI==5.2.0.post1
pip install packaging==25.0
pip install numpy==2.3.2
```
data files
You need to save data.txt in the directory that you save merge_ckpt.py in.
## How to use
1. Run this script.
2. Select the checkpoint files ( .safetensors file ).
3. Input weights of the checkpoint files. Weights is modified so that the sum becomes one.
4. If you want to use a vae of a checkpoint, check the radio button of that checkpoint row. If you don't check, vae parameters also are merged.
5. Input the output path ( .safetensors file ).
6. Click run button.
7. After a while, the output file is generated.
   
