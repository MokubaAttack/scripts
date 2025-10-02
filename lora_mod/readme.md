# lora_mod
When the message "ValueError: Checkpoint not supported because layer lora_unet_label_emb_0_0.alpha not supported." appears,try to run this program. I'm successful in lora of Illustrious model.
## requirements
python modules
```
pip install safetensors torch numpy FreeSimpleGUI
```
## how to use
1. input next command.
   python lora_mod.py lora.safetensors
   lora.safetensors is the lora file.   
2. a file that ".safetensor" of the lora file turn into "_mod.safetensors" is made.
