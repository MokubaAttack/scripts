import os,sys
import torch
from safetensors.torch import load_file,save_file
import FreeSimpleGUI as sg

sg.theme('GrayGrayGray')
try:
    path = sys.argv[1]
except:
    path = sg.popup_get_file('ckpt file',title="get vae")

if path!=None:
    if path.endswith(".safetensors"):
        try:
            key="first_stage_model."
            base_safe=path
            out=base_safe.replace(".safetensors","_vae.safetensors")
            state_dict=load_file(base_safe)
            out_dict={}
            for k,w in state_dict.items():
                if k.startswith(key):
                    k_out=k.replace(key,"")
                    out_dict[k_out]=w.to(torch.float16)
            save_file(out_dict,out)
        except:
            sg.popup(path+" is failed.",title="error")

