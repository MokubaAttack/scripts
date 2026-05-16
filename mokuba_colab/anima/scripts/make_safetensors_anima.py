import torch
from diffusers_anima import AnimaPipeline
from safetensors.torch import (
    save_file,
    load_file
)
import os
import math

if not(os.path.exists(os.getcwd()+"/pipecache")):
    os.mkdir(os.getcwd()+"/pipecache")

def zip_ckpt(ckpt1,ckpt2):
    keys=[]
    for k in ckpt1:
        if k in ckpt2:
            sum1=torch.sum(torch.abs(ckpt1[k])).item()
            sum2=torch.sum(torch.abs(ckpt2[k])).item()
            n=not(math.isnan(sum1) or math.isnan(sum2))
            if n and sum1!=sum2:
                ckpt1[k]=ckpt1[k]*sum2/sum1
            ckpt1[k]=ckpt1[k].to(torch.bfloat16)
        del ckpt2[k]
    return ckpt1

trans_key1={
    "ff.net.0.proj.weight":"mlp.layer1.weight",
    "ff.net.2.weight":"mlp.layer2.weight",
    "norm1.linear_1.weight":"adaln_modulation_self_attn.1.weight",
    "norm1.linear_2.weight":"adaln_modulation_self_attn.2.weight",
    "attn1.norm_q.weight":"self_attn.q_norm.weight",
    "attn1.norm_k.weight":"self_attn.k_norm.weight",
    "attn1.to_q.weight":"self_attn.q_proj.weight",
    "attn1.to_k.weight":"self_attn.k_proj.weight",
    "attn1.to_v.weight":"self_attn.v_proj.weight",
    "attn1.to_out.0.weight":"self_attn.output_proj.weight",
    "norm2.linear_1.weight":"adaln_modulation_cross_attn.1.weight",
    "norm2.linear_2.weight":"adaln_modulation_cross_attn.2.weight",
    "attn2.norm_q.weight":"cross_attn.q_norm.weight",
    "attn2.norm_k.weight":"cross_attn.k_norm.weight",
    "attn2.to_q.weight":"cross_attn.q_proj.weight",
    "attn2.to_k.weight":"cross_attn.k_proj.weight",
    "attn2.to_v.weight":"cross_attn.v_proj.weight",
    "attn2.to_out.0.weight":"cross_attn.output_proj.weight",
    "norm3.linear_1.weight":"adaln_modulation_mlp.1.weight",
    "norm3.linear_2.weight":"adaln_modulation_mlp.2.weight"
}
trans_key2={
    "patch_embed.proj.weight":"x_embedder.proj.1.weight",
    "time_embed.t_embedder.linear_1.weight":"t_embedder.1.linear_1.weight",
    "time_embed.t_embedder.linear_2.weight":"t_embedder.1.linear_2.weight",
    "time_embed.norm.weight":"t_embedding_norm.weight",
    "norm_out.linear_1.weight":"final_layer.adaln_modulation.1.weight",
    "norm_out.linear_2.weight":"final_layer.adaln_modulation.2.weight",
    "proj_out.weight":"final_layer.linear.weight"
}

vae_key1={
    "conv1":"quant_conv",
    "conv2":"post_quant_conv",
}
vae_key2={
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
vae_key3={
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

check="final_layer.linear.weight"
pass_keys=[
    "model.diffusion_model.pos_embedder.dim_spatial_range",
    "model.diffusion_model.pos_embedder.dim_temporal_range",
    "model.diffusion_model.pos_embedder.seq"
]

def checksafe(path):
    sd=load_file(path)

    head=None
    keys=[]
    for k in sd:
        keys.append(k)
        if k.endswith(check):
            head=k.replace(check,"")
    
    if head==None:
        return None
    elif head=="":
        for k in keys:
            if k.startswith(head):
                mk="model.diffusion_model."+k
                if not(mk in pass_keys):
                    sd[mk]=sd[k]
            del sd[k]
    else:
        for k in keys:
            if k.startswith(head):
                mk="model.diffusion_model."+k.removeprefix(head)
                if not(mk in pass_keys):
                    sd[mk]=sd[k]
                else:
                    del sd[k]
            elif k.startswith("first_stage_model.") or k.startswith("cond_stage_model.qwen3_06b.transformer.model."):
                pass
            else:
                del sd[k]

    save_file(sd,path.replace(".safetensors","_dummy.safetensors"))
    return path.replace(".safetensors","_dummy.safetensors")

def mksafe(base_path,loras,ws,out_path,win):
    win["RUN"].Update(disabled=True)
    win["info"].update("making")
    base_path=checksafe(base_path)
    if base_path==None:
        win["RUN"].Update(disabled=False)
        win["info"].update("error")
        return
    pipe = AnimaPipeline.from_single_file(base_path,cache_dir=os.getcwd()+"/pipecache")

    osd={}
    for k,p in getattr(pipe, "transformer").named_parameters():
        if k.startswith("core."):
            k=k.replace("core.","")
            if k.startswith("transformer_"):
                k=k.replace("transformer_","")
                for key in trans_key1:
                    if k.endswith(key):
                        k=k.replace(key,trans_key1[key])
            else:
                k=trans_key2[k]
        k="model.diffusion_model."+k
        osd[k]=p.data.to(torch.float32)
        
    for k,p in getattr(pipe, "vae").named_parameters():
        if k.startswith("encoder."):
            for key in vae_key2:
                if vae_key2[key] in k:
                    k.replace(vae_key2[key],key)
        elif k.startswith("decoder."):
            for key in vae_key3:
                if vae_key3[key] in k:
                    k.replace(vae_key3[key],key)
        else:
            for key in vae_key1:
                if vae_key1[key] in k:
                    k.replace(vae_key1[key],key)
        k="first_stage_model."+k
        osd[k]=p.data.to(torch.float32)
    for k,p in getattr(pipe, "text_encoder").named_parameters():
        k="cond_stage_model.qwen3_06b.transformer.model."+k
        osd[k]=p.data.to(torch.float32)

    for i in range(len(loras)):
        pipe.load_lora_weights(loras[i], adapter_name="style"+str(i))
        pipe.set_adapters("style"+str(i), adapter_weights=[ws[i]])
        pipe.fuse_lora()
        pipe.unload_lora_weights()

    sd={}
    for k,p in getattr(pipe, "transformer").named_parameters():
        if k.startswith("core."):
            k=k.replace("core.","")
            if k.startswith("transformer_"):
                k=k.replace("transformer_","")
                for key in trans_key1:
                    if k.endswith(key):
                        k=k.replace(key,trans_key1[key])
            else:
                k=trans_key2[k]
        k="model.diffusion_model."+k
        sd[k]=p.data.to(torch.float32)
    for k,p in getattr(pipe, "vae").named_parameters():
        if k.startswith("encoder."):
            for key in vae_key2:
                if vae_key2[key] in k:
                    k.replace(vae_key2[key],key)
        elif k.startswith("decoder."):
            for key in vae_key3:
                if vae_key3[key] in k:
                    k.replace(vae_key3[key],key)
        else:
            for key in vae_key1:
                if vae_key1[key] in k:
                    k.replace(vae_key1[key],key)
        k="first_stage_model."+k
        sd[k]=p.data.to(torch.float32)
    for k,p in getattr(pipe, "text_encoder").named_parameters():
        k="cond_stage_model.qwen3_06b.transformer.model."+k
        sd[k]=p.data.to(torch.float32)
        
    output=zip_ckpt(sd,osd)
    save_file(output,out_path)
    os.remove(base_path)

    win["RUN"].Update(disabled=False)
    win["info"].update("fin")

if __name__=="__main__":
    import FreeSimpleGUI as sg
    from plyer import notification
    import tkinter as tk
    import threading,pyperclip

    keys=[
        'ckpt','lora1','lora2','lora3',"out",'w1','w2','w3'
    ]
    grp_rclick_menu={}
    for key in keys:
        grp_rclick_menu[key]=[
            "",
            [
                "-copy-::"+key,"-cut-::"+key,"-paste-::"+key
            ]
        ]
    layout =[
        [sg.Text("checkpoint file")],
        [sg.InputText(key='ckpt',right_click_menu=grp_rclick_menu["ckpt"]),sg.FileBrowse('select ckpt', file_types=(('ckpt file', '.safetensors'),))],
        [sg.Text("lora1 file")],
        [sg.InputText(key='lora1',right_click_menu=grp_rclick_menu["lora1"]),sg.FileBrowse('select lora', file_types=(('lora file', '.safetensors'),))],
        [sg.Text("weight"),sg.InputText("1.0",key='w1',right_click_menu=grp_rclick_menu["w1"])],
        [sg.Text("lora2 file")],
        [sg.InputText(key='lora2',right_click_menu=grp_rclick_menu["lora2"]),sg.FileBrowse('select lora', file_types=(('lora file', '.safetensors'),))],
        [sg.Text("weight"),sg.InputText("1.0",key='w2',right_click_menu=grp_rclick_menu["w2"])],
        [sg.Text("lora3 file")],
        [sg.InputText(key='lora3',right_click_menu=grp_rclick_menu["lora3"]),sg.FileBrowse('select lora', file_types=(('lora file', '.safetensors'),))],
        [sg.Text("weight"),sg.InputText("1.0",key='w3',right_click_menu=grp_rclick_menu["w3"])],
        [sg.Text("out file")],
        [sg.Input(key="out",right_click_menu=grp_rclick_menu["out"]),sg.FileSaveAs(file_types=(('ckpt file', '.safetensors'),))],
        [sg.Text("infomation",key="info")],
        [sg.Button('RUN'),sg.Button('EXIT')]
    ]

    window = sg.Window('make safetensors anima', layout)

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'EXIT'):
            break

        elif event=="RUN":
            base_safe=values["ckpt"]
            loras=[]
            weights=[]
            out_safe=values["out"]
            if values["lora1"]!="":
                loras.append(values["lora1"])
                try:
                    weights.append(float(values["w1"]))
                    window["w1"].update(str(float(values["w1"])))
                except:
                    weights.append(1.0)
                    window["w1"].update("1.0")
                    
            if values["lora2"]!="":
                loras.append(values["lora2"])
                try:
                    weights.append(float(values["w2"]))
                    window["w2"].update(str(float(values["w2"])))
                except:
                    weights.append(1.0)
                    window["w2"].update("1.0")
                    
            if values["lora3"]!="":
                loras.append(values["lora3"])
                try:
                    weights.append(float(values["w3"]))
                    window["w3"].update(str(float(values["w3"])))
                except:
                    weights.append(1.0)
                    window["w3"].update("1.0")

            if base_safe!="" and out_safe!="":
                thread1 = threading.Thread(target=mksafe,args=(base_safe,loras,weights,out_safe,window))
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
