import requests
import os
import json
from safetensors.torch import (
    save_file,
    load_file
)

def dlc(ver_id,path,token,lora=False):
    url = "https://civitai.com/api/download/models/"+str(ver_id)+"?type=Model&format=SafeTensor&token="+str(token)
    data = requests.get(url,stream=True)
    meta_dict={}
    meta_dict["id"]=str(ver_id)
    meta_dict["weight"]=str(1)
    if path.endswith(".safetensors"):
        with open(path, "wb") as fh:
            limit=1024*1024*1024*1024
            dummy_data=b""
            step1=True
            step2=True
            for chunk in data.iter_content(chunk_size=1024*1024):
                if step2:
                    dummy_data+=chunk
                else:
                    fh.write(chunk)
                if len(dummy_data)>8 and step1:
                    limit=int.from_bytes(dummy_data[:8],byteorder="little")
                    dummy_data=dummy_data[8:]
                    step1=False
                if len(dummy_data)>limit and step2:
                    head=dummy_data[:limit].decode()
                    head_dict=json.loads(head)
                    head_dict["__metadata__"]=meta_dict
                    head=str(head_dict)
                    head=head.replace("'",'"')
                    b_data=head.encode()
                    b_len=len(b_data).to_bytes(8,byteorder="little")
                    fh.write(b_len)
                    fh.write(b_data)
                    dummy_data=dummy_data[limit:]
                    fh.write(dummy_data)
                    step2=False
    else:
        with open(path, "wb") as fh:
            for chunk in data.iter_content(chunk_size=1024*1024):
                fh.write(chunk)

        model = torch.load(path)
        while True:
            i=0
            for k in model:
                if isinstance(model[k],torch.Tensor):
                    i+=1
            if i==0:
                model=model[list(model.keys())[0]]
            else:
                break
            if not(isinstance(model,dict)):
                model={}
                break
        for k in model:
            if not(isinstance(model[k],torch.Tensor)):
                del model[k]
        k="."+path.split(".")[-1]
        path=path.replace(k,".safetensors")
        save_file(model,path,metadata=meta_dict)
    if lora:
        unnecessary=[
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
        model=load_file(path)
        model2={}
        for k in model:
            if not(k in unnecessary):
                model2[k]=model[k]
        del model
        save_file(model2,path,metadata=meta_dict)
        