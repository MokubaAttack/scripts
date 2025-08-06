from diffusers import StableDiffusionXLPipeline
import torch
import os
from safetensors.torch import load_file,save_file
import FreeSimpleGUI as sg

def mkemb(in_path,prompt,st):
    try:
        st=float(st)
    except:
        st=1.0
    if not(os.path.exists(in_path)):
        sg.popup("the selected file doesn't exist.",title="error")
        return
    if not(in_path.endswith(".safetensors")):
        sg.popup("you need to select the safetensors file.",title="error")
        return

    try:
        out=in_path.replace(".safetensors","_emb.safetensors")
        pipe=StableDiffusionXLPipeline.from_single_file(in_path,torch_dtype=torch.float16)
        
        prompt_list=prompt.split(",")
        prompt_tens=[]
        prompt_tens2=[]

        for prompt_elem in prompt_list:
            prompt_elem_ids = pipe.tokenizer(prompt_elem,return_tensors="pt",truncation=False).input_ids
            prompt_elem_ten = pipe.text_encoder(prompt_elem_ids)[0]
    
            prompt_elem_ids2 = pipe.tokenizer_2(prompt_elem,return_tensors="pt",truncation=False).input_ids
            prompt_elem_ten2 = pipe.text_encoder_2(prompt_elem_ids2)[0]
    
            if prompt_elem_ten.dim()>2:
                prompt_elem_ten=torch.zeros(1, 768).to(torch.float16)
        
            if prompt_elem_ten2.dim()>2:
                prompt_elem_ten2=torch.zeros(1, 1280).to(torch.float16)
        
            prompt_tens.append(prompt_elem_ten)
            prompt_tens2.append(prompt_elem_ten2)
    
        embs=torch.cat(prompt_tens, dim=0)
        embs2=torch.cat(prompt_tens2, dim=0)
        out_dict={}
        out_dict["clip_l"]=embs
        out_dict["clip_g"]=embs2
        save_file(out_dict,out)
        sg.popup(out,title="fin")
    except:
        sg.popup("fail in the output.",title="error")

layout =[
    [sg.Text("prompt")],
    [sg.InputText(key='prompt_input')],
    [sg.Text("checkpoint safetensors")],
    [sg.InputText(key='ckpt_input'),sg.FileBrowse('select ckpt', key="ckpt_filename", file_types=(('ckpt file', '.safetensors'),))],
    [sg.Text("prompt strength"),sg.InputText(key='prompt_strength')],
    [sg.Button('Exit'),sg.Button('Run')]
        ]

window = sg.Window('Make Embedding', layout)

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    elif event=="Run":
        if values["prompt_input"]!="" and values["ckpt_input"]!="":
            mkemb(values["ckpt_input"],values["prompt_input"],values["prompt_strength"])
window.close()

