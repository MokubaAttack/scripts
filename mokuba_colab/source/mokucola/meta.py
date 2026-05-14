import ast
import pyexiv2
from PIL import PngImagePlugin

def plus_meta(vs,img):
    try:
        if "pr" in vs:
            if vs["pr"]=="":
                metadata="None\n\n"
            else:
                metadata=vs["pr"]+"\n\n"
        if "ne" in vs:
            if vs["ne"]=="":
                metadata=metadata+"Negative prompt: None\n\n"
            else:
                metadata=metadata+"Negative prompt: "+vs["ne"]+"\n\n"
        if "st" in vs:
            if vs["st"]!="":
                metadata=metadata+"Steps: "+vs["st"]+", " 
        if "sa" in vs:
            if vs["sa"]!="":
                metadata=metadata+"Sampler: "+vs["sa"]+", "
        if "cf" in vs:
            if vs["cf"]!="":
                metadata=metadata+"CFG scale: "+vs["cf"]+", "
        if "se" in vs:
            if vs["se"]!="":
                metadata=metadata+"Seed: "+vs["se"]+", "
        if "cl" in vs:
            if vs["cl"]!="":
                metadata=metadata+"Clip skip: "+vs["cl"]+", "
        if "pag" in vs:
            if vs["pag"]!="":
                metadata=metadata+"PAG scale: "+vs["pag"]+", "
        if "ds" in vs:        
            if vs["ds"]!="":
                metadata=metadata+"Denoising strength: "+vs["ds"]+", "
        if "hu" in vs:
            if vs["hu"]!="":
                metadata=metadata+"Hires upscale: "+vs["hu"]+", "
        if "hs" in vs:
            if vs["hs"]!="":
                metadata=metadata+"Hires steps: "+vs["hs"]+", "
        if "hum" in vs:
            if vs["hum"]!="":
                metadata=metadata+"Hires upscaler: "+vs["hum"]+", "
        if "tu" in vs:
            if vs["tu"]!="":
                metadata=metadata+"Tile upscale: "+vs["tu"]+", "
        if "tum" in vs:
            if vs["tum"]!="":
                metadata=metadata+"Tile upscaler: "+vs["tum"]+", "
        if "ccs" in vs:
            if vs["ccs"]!="":
                metadata=metadata+"controlnet_conditioning_scale: "+vs["ccs"]+", "

        metadata=metadata+'Civitai resources: ['
        if "ckpt" in vs:
            if vs["ckpt"]!="":
                metadata=metadata+'{"type":"checkpoint","modelVersionId":'+vs["ckpt"]+"}"

        if "lora" in vs:
            if vs["lora"]!="[]":
                lora_list= ast.literal_eval(vs["lora"])
                w_list=ast.literal_eval(vs["w"])
                for i in range(len(lora_list)):
                    metadata=metadata+',{"type":"lora","weight":'+str(w_list[i])+',"modelVersionId":'+str(lora_list[i])+"}"

        if "embed" in vs:
            if vs["embed"]!="[]":
                embed_list=ast.literal_eval(vs["embed"])
                for i in range(len(embed_list)):
                    metadata=metadata+',{"type":"embed","modelVersionId":'+str(embed_list[i])+"}"
        if "vae" in vs:
            if vs["vae"]!="":
                metadata=metadata+',{"type":"ae","modelVersionId":'+vs["vae"]+"}"
        if "cont" in vs:
            if vs["cont"]!="":
                metadata=metadata+',{"type":"controlnet","modelVersionId":'+vs["cont"]+"}"
        if "up" in vs:
            if vs["up"]!="":
                metadata=metadata+',{"type":"upscaler","modelVersionId":'+vs["up"]+"}"
                
        metadata=metadata+'], Civitai metadata: {}'

        if "[," in metadata:
            metadata=metadata.replace("[,","[")
    
        image_path=vs["input"]
        if image_path.endswith(".jpg"):
            img.save(image_path, 'JPEG' ,quality=85)
            with pyexiv2.Image(image_path) as img:
                img.modify_exif({'Exif.Photo.UserComment':metadata})
        else:
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("parameters", metadata)
            img.save(image_path, "PNG", pnginfo=pnginfo)
    except:
        image_path=vs["input"]
        if image_path.endswith(".jpg"):
            img.save(image_path, 'JPEG' ,quality=85)
        else:
            img.save(image_path, "PNG")