from safetensors.torch import load_file,save_file
import sys
import os
import FreeSimpleGUI as sg

sg.theme('GrayGrayGray')
ok=[
    "lora_unet_out_2.alpha",
    "lora_unet_out_2.lora_down.weight",
    "lora_unet_out_2.lora_up.weight",
    "lora_unet_input_blocks_0_0.alpha",
    "lora_unet_input_blocks_0_0.lora_down.weight",
    "lora_unet_input_blocks_0_0.lora_up.weight",
    "lora_unet_label_emb_0_0.alpha",
    "lora_unet_label_emb_0_0.lora_down.weight",
    "lora_unet_label_emb_0_0.lora_up.weight",
    "lora_unet_label_emb_0_2.alpha",
    "lora_unet_label_emb_0_2.lora_down.weight",
    "lora_unet_label_emb_0_2.lora_up.weight",
    "lora_unet_time_embed_0.alpha",
    "lora_unet_time_embed_0.lora_down.weight",
    "lora_unet_time_embed_0.lora_up.weight",
    "lora_unet_time_embed_2.alpha",
    "lora_unet_time_embed_2.lora_down.weight",
    "lora_unet_time_embed_2.lora_up.weight"
]

try:
    path = sys.argv[1]
except:
    path = sg.popup_get_file('lora file',title="modify lora",file_types=(('lora file', '.safetensors'),))
    
if path!=None:
    if path.endswith(".safetensors"):
        try:
            state_dict=load_file(path)
            for k in ok:
                if k in state_dict:
                    out=state_dict.pop(k)
            out_path=path.replace(".safetensors","_mod.safetensors")
            save_file(state_dict,out_path)
            del state_dict
        except:
            print(path+" is failed.")
            
