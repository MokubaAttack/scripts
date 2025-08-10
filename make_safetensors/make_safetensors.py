from diffusers import StableDiffusionXLPipeline,AutoencoderKL,DDIMScheduler
import torch
import subprocess
import shutil
import os
from safetensors.torch import load_file
import FreeSimpleGUI as sg

os.environ['HF_HOME']=os.getcwd()+"/.cache/huggingface"

def run(base_safe,vae_safe,out_safe,lora1,lora2,lora3,lora1w,lora2w,lora3w):
    dtype=torch.float16
    if not(os.path.exists(base_safe)):
        sg.popup("the ckpt file doesn't exist.",title="error")
        return
    pipe = StableDiffusionXLPipeline.from_single_file(base_safe, torch_dtype=dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if vae_safe!="":
        if os.path.exists(vae_safe):
            pipe.vae=AutoencoderKL.from_single_file(vae_safe,torch_dtype=dtype)

    ls=[]
    ws=[]
    if lora1!="":
        if os.path.exists(lora1):
            pipe.load_lora_weights(".",weight_name=lora1,torch_dtype=dtype, adapter_name="lora1")
            ls.append("lora1")
            if lora1w=="":
                ws.append(1.0)
            else:
                try:
                    ws.append(float(lora1w))
                except:
                    ws.append(1.0)
    if lora2!="":
        if os.path.exists(lora2):
            pipe.load_lora_weights(".",weight_name=lora2,torch_dtype=dtype, adapter_name="lora2")
            ls.append("lora2")
            if lora2w=="":
                ws.append(1.0)
            else:
                try:
                    ws.append(float(lora2w))
                except:
                    ws.append(1.0)
    if lora3!="":
        if lora3!="" and os.path.exists(lora3):
            pipe.load_lora_weights(".",weight_name=lora3,torch_dtype=dtype, adapter_name="lora3")
            ls.append("lora3")
            if lora3w=="":
                ws.append(1.0)
            else:
                try:
                    ws.append(float(lora3w))
                except:
                    ws.append(1.0)

    if ls!=[]:
        pipe.set_adapters(ls, adapter_weights=ws)
        pipe.fuse_lora()
        pipe.unload_lora_weights()

    pipe.save_pretrained(os.getcwd()+"/dummy", safe_serialization=True)

    cmd="python convert_diffusers_to_original_sdxl.py --model_path "+os.getcwd()+"/dummy --checkpoint_path "+out_safe+" --use_safetensors"            
    returncode = subprocess.call(cmd)

    shutil.rmtree(os.getcwd()+"/dummy")
    
layout =[
    [sg.Text("checkpoint file")],
    [sg.InputText(key='ckpt'),sg.FileBrowse('select ckpt', key="selckpt", file_types=(('ckpt file', '.safetensors'),))],
    [sg.Text("vae file")],
    [sg.InputText(key='vae'),sg.FileBrowse('select vae', key="selvae", file_types=(('vae file', '.safetensors'),))],
    [sg.Text("lora1 file")],
    [sg.InputText(key='lora1'),sg.FileBrowse('select lora', key="sellora1", file_types=(('lora file', '.safetensors'),))],
    [sg.Text("weight"),sg.InputText("1.0",key='w1')],
    [sg.Text("lora2 file")],
    [sg.InputText(key='lora2'),sg.FileBrowse('select lora', key="sellora2", file_types=(('lora file', '.safetensors'),))],
    [sg.Text("weight"),sg.InputText("1.0",key='w2')],
    [sg.Text("lora3 file")],
    [sg.InputText(key='lora3'),sg.FileBrowse('select lora', key="sellora3", file_types=(('lora file', '.safetensors'),))],
    [sg.Text("weight"),sg.InputText("1.0",key='w3')],
    [sg.Text("out file")],
    [sg.InputText(key='out'),sg.Button('select out', key="selout")],
    [sg.Button('Exit'),sg.Button('Run')]
        ]

window = sg.Window('make safetensors', layout)

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    elif event=="selout":
        file_path = sg.popup_get_file(
            "select output file",
            multiple_files = False,
            save_as=True,
            file_types=(("ckpt file", "*.safetensors"),),
        )
        window['out'].update(file_path)

    elif event=="RUN":
        base_safe=values["ckpt"]
        vae_safe=values["vae"]
        out_safe=values["out"]
        lora1=values["lora1"]
        lora2=values["lora2"]
        lora3=values["lora3"]
        lora1w=values["w1"]
        lora2w=values["w2"]
        lora3w=values["w3"]
        if base_safe!="" and out_safe!="":
            try:
                run(base_safe,vae_safe,out_safe,lora1,lora2,lora3,lora1w,lora2w,lora3w)
            except:
                sg.popup("fail in the output.",title="error")
            
window.close()
