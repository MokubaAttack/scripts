# get_vae
It is a script that extracts a vae safetensors from a checkpoint safetensors. I'm successful in sdxl model.
## requirements
python modules
```
pip install torch numpy safetensors
```
## how to use
1. input next command.
   python get_vae.py ckpt.safetensors
   ckpt.safetensors is the checkpoint file.
2. a file that ".safetensor" of the checkpoint file turn into "_vae.safetensors" is made.
