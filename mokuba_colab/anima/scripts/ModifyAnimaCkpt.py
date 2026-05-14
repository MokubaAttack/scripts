from safetensors.torch import (
    load_file,
    save_file
)
import json
import os
import shutil

check="final_layer.linear.weight"
pass_keys=[
    "model.diffusion_model.pos_embedder.dim_spatial_range",
    "model.diffusion_model.pos_embedder.dim_temporal_range",
    "model.diffusion_model.pos_embedder.seq"
]

keys1={
    "conv1":"quant_conv",
    "conv2":"post_quant_conv",
}
keys2={
    "conv1":"conv_in",
    "head.0":"norm_out",
    "head.2":"conv_out",
    "downsamples":"down_blocks",
    "residual.2":"conv1",
    "residual.6":"conv2",
    "residual.0":"norm1",
    "residual.3":"norm2",
    "shortcut":"conv_shortcut",
    "middle.1":"mid_block.attentions.0",
    "middle.0":"mid_block.resnets.0",
    "middle.2":"mid_block.resnets.1",
}
keys3={
    "conv1":"conv_in",
    "head.0":"norm_out",
    "head.2":"conv_out",
    "residual.2":"conv1",
    "residual.6":"conv2",
    "residual.0":"norm1",
    "residual.3":"norm2",
    "middle.1":"mid_block.attentions.0",
    "middle.0":"mid_block.resnets.0",
    "middle.2":"mid_block.resnets.1",
    "upsamples.3":"up_blocks.0.upsamplers.0",
    "upsamples.7":"up_blocks.1.upsamplers.0",
    "upsamples.11":"up_blocks.2.upsamplers.0",
    "upsamples.0":"up_blocks.0.resnets.0",
    "upsamples.10":"up_blocks.2.resnets.2",
    "upsamples.12":"up_blocks.3.resnets.0",
    "upsamples.13":"up_blocks.3.resnets.1",
    "upsamples.14":"up_blocks.3.resnets.2",
    "upsamples.1":"up_blocks.0.resnets.1",
    "upsamples.2":"up_blocks.0.resnets.2",
    "upsamples.4":"up_blocks.1.resnets.0",
    "shortcut":"conv_shortcut",
    "upsamples.5":"up_blocks.1.resnets.1",
    "upsamples.6":"up_blocks.1.resnets.2",
    "upsamples.8":"up_blocks.2.resnets.0",
    "upsamples.9":"up_blocks.2.resnets.1",
}

conf_json={
    "_class_name": "AutoencoderKLQwenImage",
    "_diffusers_version": "0.36.0",
    "attn_scales": [],
    "base_dim": 96,
    "dim_mult": [
        1,
        2,
        4,
        4
    ],
    "dropout": 0.0,
    "latents_mean": [
        -0.7571,
        -0.7089,
        -0.9113,
        0.1075,
        -0.1745,
        0.9653,
        -0.1517,
        1.5508,
        0.4134,
        -0.0715,
        0.5517,
        -0.3632,
        -0.1922,
        -0.9497,
        0.2503,
        -0.2921
    ],
    "latents_std": [
        2.8184,
        1.4541,
        2.3275,
        2.6558,
        1.2196,
        1.7708,
        2.6052,
        2.0743,
        3.2687,
        2.1526,
        2.8652,
        1.5579,
        1.6382,
        1.1253,
        2.8251,
        1.916
    ],
    "num_res_blocks": 2,
    "temperal_downsample": [
        False,
        True,
        True
    ],
    "z_dim": 16
}

def get_metadata(path):
    f=open(path,"rb")
    data=f.read(8)
    n=int.from_bytes(data,byteorder="little")
    data=f.read(n)
    head=data.decode()
    head_dict=json.loads(head)
    if "__metadata__" in head_dict:
        meta=head_dict["__metadata__"]
    else:
        meta={"format":"pt"}
    f.close()
    return meta

def modsafe(path,out_path,win):
    win["RUN"].Update(disabled=True)
    win["info"].update("modifying")

    meta=get_metadata(path)
    sd=load_file(path)

    head=None
    keys=[]
    for k in sd:
        keys.append(k)
        if k.endswith(check):
            head=k.replace(check,"")

    if head==None:
        win["RUN"].Update(disabled=False)
        win["info"].update("error")
        return
    
    vae_dict={}
    for k in keys:
        if k.startswith("first_stage_model."):
            mk=k.replace("first_stage_model.","")
            vae_dict[mk]=sd[k]
        elif k.startswith(head):
            mk=k.replace(head,"")
            mk="model.diffusion_model."+k
            if not(mk in pass_keys):
                sd[mk]=sd[k]
        del sd[k]

    if vae_dict!={}:
        keys=[]
        for k in vae_dict:
            keys.append(k)
        for k in keys:
            mk=k
            if k.startswith("encoder."):
                for k2 in keys2:
                    if k2 in mk:
                        mk=mk.replace(k2,keys2[k2])
                if mk!=k:
                    vae_dict[mk]=vae_dict[k]

            elif k.startswith("decoder."):
                for k2 in keys3:
                    if k2 in mk:
                        mk=mk.replace(k2,keys3[k2])
                if mk!=k:
                    vae_dict[mk]=vae_dict[k]

            else:
                for k2 in keys1:
                    if k2 in mk:
                        mk=mk.replace(k2,keys1[k2])
                if mk!=k:
                    vae_dict[mk]=vae_dict[k]

            del vae_dict[k]

        vae_folder=out_path.replace(".safetnsors","_vae")
        if os.path.isdir(vae_folder):
            shutil.rmtree(vae_folder)
        os.mkdir(vae_folder)
        save_file(vae_dict,vae_folder+"/diffusion_pytorch_model.safetensors")
        with open(vae_folder+"/config.json", 'w') as f:
            json.dump(conf_json, f, indent=2)
            
    save_file(sd,out_path,metadata=meta)
    win["RUN"].Update(disabled=False)
    win["info"].update("fin")

if __name__ == '__main__':
    import FreeSimpleGUI as sg
    import tkinter as tk
    import pyperclip
    from plyer import notification
    import threading

    keys=['output','input']
    grp_rclick_menu={}
    for k in keys:
        grp_rclick_menu[k]=[
            "",
            [
                "-copy-::"+k,"-cut-::"+k,"-paste-::"+k
            ]
        ]

    layout=[
        [sg.Text("input"),sg.Input(key="input",right_click_menu=grp_rclick_menu["input"]),sg.FilesBrowse(file_types=(('model file', '.safetensors'),))],
        [sg.Text("output"), sg.Input(key="output",right_click_menu=grp_rclick_menu["output"]),sg.FileSaveAs(file_types=(('model file', '.safetensors'),))],
        [sg.Text("infomation",key="info")],
        [sg.Button('RUN', key='RUN'),sg.Button('EXIT', key='EXIT')]
    ]

    window = sg.Window('Modify Anima Ckpt', layout)

    while True:
        event, values = window.read()
            
        if event == sg.WINDOW_CLOSED:
            break
        elif event=="EXIT":
            break
        elif event=="RUN":
            if values["output"]!="" and values["input"]!="":
                outpath=values["output"]
                inpath=values["input"]
                thread1 = threading.Thread(target=modsafe,args=(inpath,outpath,window))
                thread1.start()

        elif "-copy-" in event:
            try:
                key=event.replace("-copy-::","")
                selected = window[key].widget.selection_get()
                pyperclip.copy(selected)
            except:
                pass
        elif "-cut-" in event:
            try:
                key=event.replace("-cut-::","")
                selected = window[key].widget.selection_get()
                pyperclip.copy(selected)
                window[key].widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
            except:
                pass
        elif "-paste-" in event:
            try:
                key=event.replace("-paste-::","")
                selected = pyperclip.paste()
                insert_pos = window[key].widget.index("insert")
                window[key].Widget.insert(insert_pos, selected)
                window[key].widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
            except:
                pass

    window.close()
