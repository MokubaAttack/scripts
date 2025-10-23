from diffusers import StableDiffusionXLPipeline,StableDiffusionXLImg2ImgPipeline,AutoencoderKL
from diffusers import StableDiffusionPipeline,StableDiffusionImg2ImgPipeline
from safetensors.torch import load_file
import safetensors
import torch
import random
import os
import shutil
import datetime
import ast
from PIL import Image
from PIL import PngImagePlugin
from IPython.display import display
from IPython.display import clear_output
from compel import CompelForSD,CompelForSDXL
from diffusers import EulerDiscreteScheduler
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import LMSDiscreteScheduler
from diffusers import HeunDiscreteScheduler
from diffusers import KDPM2DiscreteScheduler
from diffusers import KDPM2AncestralDiscreteScheduler
from diffusers import DPMSolverMultistepScheduler
from diffusers import DPMSolverSinglestepScheduler
from diffusers import PNDMScheduler
from diffusers import UniPCMultistepScheduler
from diffusers import LCMScheduler
from diffusers import DDIMScheduler
from diffusers import DPMSolverSDEScheduler

def plus_meta(vs,img):
    try:
        metadata=vs["pr"]+"\n\n"
        metadata=metadata+"Negative prompt: "+vs["ne"]+"\n\n"
        metadata=metadata+"Steps: "+vs["st"]+", " 
        metadata=metadata+"Sampler: "+vs["sa"]+", "
        metadata=metadata+"CFG scale: "+vs["cf"]+", "
        metadata=metadata+"Seed: "+vs["se"]+", "
        metadata=metadata+"Clip skip: "+vs["cl"]+", "
        if vs["ds"]!="":
            metadata=metadata+"Denoising strength: "+vs["ds"]+", "
        if vs["hu"]!="":
            metadata=metadata+"Hires upscale: "+vs["hu"]+", "
        if vs["hs"]!="":
            metadata=metadata+"Hires steps: "+vs["hs"]+", "
        if vs["hum"]!="":
            metadata=metadata+"Hires upscaler: "+vs["hum"]+", "
        if vs["ckpt"]!="":
            metadata=metadata+'Civitai resources: [{"type":"checkpoint","modelVersionId":'+vs["ckpt"]+"}"
        if vs["lora"]!="[]":
            lora_list= ast.literal_eval(vs["lora"])
            w_list=ast.literal_eval(vs["w"])
            for i in range(len(lora_list)):
                metadata=metadata+',{"type":"lora","weight":'+str(w_list[i])+',"modelVersionId":'+str(lora_list[i])+"}"

        if vs["embed"]!="[]":
            embed_list=ast.literal_eval(vs["embed"])
            for i in range(len(embed_list)):
                metadata=metadata+',{"type":"embed","modelVersionId":'+str(embed_list[i])+"}"

        if vs["vae"]!="":
            metadata=metadata+',{"modelVersionId":'+vs["vae"]+"}"
        metadata=metadata+'], Civitai metadata: {}'
    
        image_path=vs["input"]
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", metadata)
        img.save(image_path, "PNG", pnginfo=pnginfo)
    except:
        image_path=vs["input"]
        img.save(image_path)

def text2image(loras=[], lora_weights=[], prompt = "", n_prompt = "", t="v", prog_ver=2, pic_number=10, gs=7,f_step=10, step=30, ss=0.6, cs=1, Interpolation=3, sample="DDIM",seed=0,out_folder="data",pos_emb=[],neg_emb=[],base_safe="base.safetensors",vae_safe="vae.safetensors"):
    meta_dict={}
    memo="seed\n"
    if isinstance(seed, list):
        pic_number=len(seed)
        for i in range(pic_number):
            try:
                if int(seed[i])==0:
                    seed[i]=random.randint(1, 1000000000)
                else:
                    seed[i]=int(seed[i])
            except:
                seed[i]=random.randint(1, 1000000000)
            memo=memo+str(seed[i])+"\n"
    else:
        try:
            if int(seed)==0:
                seed=[]
                for i in range(pic_number):
                    seed.append(random.randint(1, 1000000000))
            else:
                seed=[int(seed)]
                pic_number=1
        except:
            seed=[]
            for i in range(pic_number):
                seed.append(random.randint(1, 1000000000))
        for i in range(pic_number):
            memo=memo+str(seed[i])+"\n"
    print(memo)
            
    if prog_ver!=1:
        if prog_ver!=2:
            prog_ver=0
        
    if not(os.path.isfile(base_safe)):
        print("the checkpoint file does not exist.")
        return []
    memo=memo+"checkpoint : "+base_safe+"\n"
    clear_output(True)
    print(memo)
    try:
        f=safetensors.safe_open(base_safe, framework="pt", device="cpu")
        meta_dict["ckpt"]=f.metadata()["id"]
        del f
    except:
        meta_dict["ckpt"]=""
    
    if os.path.isfile(vae_safe):
        memo=memo+"vae : "+vae_safe+"\n"
        try:
            f=safetensors.safe_open(vae_safe, framework="pt", device="cpu")
            meta_dict["vae"]=f.metadata()["id"]
            del f
        except:
            meta_dict["vae"]=""
    else:
        memo=memo+"vae : original vae"+"\n"
        meta_dict["vae"]=""
    clear_output(True)
    print(memo)
    if loras!=[]:    
        if len(loras)!=len(lora_weights):
            print("the number of lora does not equal the number of lora weight.")
            return []

        i=0
        memo=memo+"lora : weight\n"
        meta_id_list=[]
        meta_weight_list=[]
        for line in loras:
            if line.endswith(".safetensors"):
                line=line.replace(".safetensors","")
            if os.path.isfile(line+".safetensors"):
                memo=memo+line+".safetensors : "+str(lora_weights[i])+" ok\n"
                clear_output(True)
                print(memo)
                list1=meta_id_list
                list2=meta_weight_list
                try:
                    f=safetensors.safe_open(line+".safetensors", framework="pt", device="cpu")
                    meta_id=f.metadata()["id"]
                    meta_id = ast.literal_eval(meta_id)
                    for j in meta_id:
                        meta_id_list.append(j)
                    del meta_id
                    meta_weight=f.metadata()["weight"]
                    meta_weight = ast.literal_eval(meta_weight)
                    for j in meta_weight:
                        meta_weight_list.append(float(j)*lora_weights[i])
                    del meta_weight
                    del f
                except:
                    meta_id_list=list1
                    meta_weight_list=list2
                
            else:
                print(line+".safetensors : "+str(lora_weights[i])+" ng")
                return []
            i=i+1

        if len(meta_weight_list)!=len(meta_id_list):
            meta_id_list=[]
            meta_weight_list=[]
        meta_dict["lora"]=str(meta_id_list)
        meta_dict["w"]=str(meta_weight_list)
    
    meta_embed_list=[]
    memo=memo+"Positive Embedding\n"
    clear_output(True)
    print(memo)
    if pos_emb==[]:
        memo=memo+"nothing\n"
        clear_output(True)
        print(memo)
    else:
        for line in pos_emb:
            if os.path.isfile(line):
                memo=memo+line+" ok\n"
                clear_output(True)
                print(memo)
                list1=meta_embed_list
                try:
                    f=safetensors.safe_open(line, framework="pt", device="cpu")
                    meta_embed_list.append(f.metadata()["id"])
                    del f
                except:
                    meta_embed_list=list1
                
            else:
                print(line+" ng")
                return []
                
    memo=memo+"Negative Embedding\n"
    clear_output(True)
    print(memo)
    if neg_emb==[]:
        memo=memo+"nothing\n"
        clear_output(True)
        print(memo)
    else:
        for line in neg_emb:
            if os.path.isfile(line):
                memo=memo+line+" ok\n"
                clear_output(True)
                print(memo)
                list1=meta_embed_list
                try:
                    f=safetensors.safe_open(line, framework="pt", device="cpu")
                    meta_embed_list.append(f.metadata()["id"])
                    del f
                except:
                    meta_embed_list=list1
            else:
                print(line+" ng")
                return []
    meta_dict["embed"]=str(meta_embed_list)

    if t=="v":
        tate=[800,1280]
        yoko=[600,900]
    elif t=="s":
        tate=[800,1280]
        yoko=[800,1280]
    elif t=="h":
        yoko=[800,1280]
        tate=[600,960]
    elif t=="vl":
        tate=[800,1600]
        yoko=[600,1200]
    elif t=="sl":
        tate=[800,1600]
        yoko=[800,1600]
    elif t=="hl":
        yoko=[800,1600]
        tate=[600,1200]
    else:
        t_list=t.split(",")
        if len(t_list)==4:
            iw=round(float(t_list[0])/8)*8
            ow=round(float(t_list[1])/8)*8
            ih=round(float(t_list[2])/8)*8
            oh=round(float(t_list[3])/8)*8

            yoko=[iw,ow]
            tate=[ih,oh]
        else:
            print("t setting is error.")
            print(" initial width, output width, initial height, output height")
            return []

    if prog_ver!=0:
        if Interpolation==1:
            p=Image.NEAREST
            memo=memo+"Interpolation : NEAREST\n"
            meta_dict["hum"]="NEAREST"
        elif Interpolation==2:
            p=Image.BOX
            memo=memo+"Interpolation : BOX\n"
            meta_dict["hum"]="BOX"
        elif Interpolation==3:
            p=Image.BILINEAR
            memo=memo+"Interpolation : BILINEAR\n"
            meta_dict["hum"]="BILINEAR"
        elif Interpolation==4:
            p=Image.HAMMING
            memo=memo+"Interpolation : HAMMING\n"
            meta_dict["hum"]="HAMMING"
        elif Interpolation==5:
            p=Image.BICUBIC
            memo=memo+"Interpolation : BICUBIC\n"
            meta_dict["hum"]="BICUBIC"
        else:
            p=Image.LANCZOS
            memo=memo+"Interpolation : LANCZOS\n"
            meta_dict["hum"]="LANCZOS"
    else:
        meta_dict["hum"]=""

    del t,Interpolation
    clear_output(True)
    print(memo)

    dtype=torch.float16
    if os.path.isfile(vae_safe):
        pipe = StableDiffusionXLPipeline.from_single_file(base_safe, torch_dtype=dtype)
        pipe.vae=AutoencoderKL.from_single_file(vae_safe,torch_dtype=dtype)
        pipe.to("cuda:0")
    else:
        pipe = StableDiffusionXLPipeline.from_single_file(base_safe, torch_dtype=dtype).to("cuda:0")
        
    if sample=="Euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : Euler\n"
        meta_dict["sa"]=sample
    elif sample=="Euler a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : Euler a\n"
        meta_dict["sa"]=sample
    elif sample=="LMS":
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : LMS\n"
        meta_dict["sa"]=sample
    elif sample=="Heun":
        pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : Heun\n"
        meta_dict["sa"]=sample
    elif sample=="DPM2":
        pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : DPM2\n"
        meta_dict["sa"]=sample
    elif sample=="DPM2 a":
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : DPM2 a\n"
        meta_dict["sa"]=sample
    elif sample=="DPM++ 2M":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : DPM++ 2M\n"
        meta_dict["sa"]=sample
    elif sample=="DPM++ SDE":
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : DPM++ SDE\n"
        meta_dict["sa"]=sample
    elif sample=="DPM++ 2M SDE":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,algorithm_type="sde-dpmsolver++")
        memo=memo+"scheduler : DPM++ 2M SDE\n"
        meta_dict["sa"]=sample
    elif sample=="LMS Karras":
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        memo=memo+"scheduler : LMS Karras\n"
        meta_dict["sa"]=sample
    elif sample=="DPM2 Karras":
        pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        memo=memo+"scheduler : DPM2 Karras\n"
        meta_dict["sa"]=sample
    elif sample=="DPM2 a Karras":
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        memo=memo+"scheduler : DPM2 a Karras\n"
        meta_dict["sa"]=sample
    elif sample=="DPM++ 2M Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        memo=memo+"scheduler : DPM++ 2M Karras\n"
        meta_dict["sa"]=sample
    elif sample=="DPM++ SDE Karras":
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        memo=memo+"scheduler : DPM++ SDE Karras\n"
        meta_dict["sa"]=sample
    elif sample=="DPM++ 2M SDE Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,algorithm_type="sde-dpmsolver++",use_karras_sigmas=True)
        memo=memo+"scheduler : DPM++ 2M SDE Karras\n"
        meta_dict["sa"]=sample
    elif sample=="PLMS":
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : PLMS\n"
        meta_dict["sa"]=sample
    elif sample=="UniPC":
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : UniPC\n"
        meta_dict["sa"]=sample
    elif sample=="LCM":
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : LCM\n"
        meta_dict["sa"]=sample
    elif sample=="DPM++ 3M SDE":
        pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : DPM++ 3M SDE\n"
        meta_dict["sa"]=sample
    elif sample=="DPM++ 3M SDE Karras":
        pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        memo=memo+"scheduler : DPM++ 3M SDE Karras\n"
        meta_dict["sa"]=sample
    elif sample=="DPM++ 3M SDE Exponential":
        pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config,use_exponential_sigmas=True)
        memo=memo+"scheduler : DPM++ 3M SDE Exponential\n"
        meta_dict["sa"]=sample
    else:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : DDIM\n"
        meta_dict["sa"]="DDIM"
    
    memo=memo+"num_inference_steps : "+str(f_step)+"\n"
    meta_dict["st"]=str(f_step)
    memo=memo+"guidance_scale : "+str(gs)+"\n"
    meta_dict["cf"]=str(gs)
    memo=memo+"clip_skip : "+str(cs)+"\n"
    meta_dict["cl"]=str(cs)

    if prog_ver!=0:
        memo=memo+"Hires steps : "+str(step)+"\n"
        meta_dict["hs"]=str(step)
        memo=memo+"Denoising strength : "+str(ss)+"\n"
        meta_dict["ds"]=str(ss)
        memo=memo+"Hires upscale : "+str(yoko[1]/yoko[0])+"\n"
        meta_dict["hu"]=str(yoko[1]/yoko[0])
    else:
        meta_dict["hs"]=""
        meta_dict["ds"]=""
        meta_dict["hu"]=""
    
    clear_output(True)
    print(memo)
    
    if loras!=[]:
        i=0
        for line in loras:
            if line.endswith(".safetensors"):
                line=line.replace(".safetensors","")
            pipe.load_lora_weights(".",weight_name=line+".safetensors",torch_dtype=dtype)
            memo=memo+line+".safetensors is loaded.\n"
            clear_output(True)
            print(memo)
            pipe.fuse_lora(lora_scale= lora_weights[i])
            pipe.unload_lora_weights()
            i=i+1
        
    if pos_emb!=[]:
        for line in pos_emb:
            key=os.path.basename(line).replace(".safetensors","")
            state_dict = load_file(line)
            pipe.load_textual_inversion(state_dict["clip_g"],token=key,text_encoder=pipe.text_encoder_2,tokenizer=pipe.tokenizer_2,torch_dtype=dtype)
            pipe.load_textual_inversion(state_dict["clip_l"],token=key,text_encoder=pipe.text_encoder,tokenizer=pipe.tokenizer,torch_dtype=dtype)
            del state_dict
            prompt = prompt+","+key
            memo=memo+line+" is loaded.\n"
            clear_output(True)
            print(memo)

    if neg_emb!=[]:
        for line in neg_emb:
            key=os.path.basename(line).replace(".safetensors","")
            state_dict = load_file(line)
            pipe.load_textual_inversion(state_dict["clip_g"],token=key,text_encoder=pipe.text_encoder_2,tokenizer=pipe.tokenizer_2,torch_dtype=dtype)
            pipe.load_textual_inversion(state_dict["clip_l"],token=key,text_encoder=pipe.text_encoder,tokenizer=pipe.tokenizer,torch_dtype=dtype)
            del state_dict
            n_prompt=n_prompt+","+key
            memo=memo+line+" is loaded.\n"
            clear_output(True)
            print(memo)
    
    memo=memo+"prompt\n"+prompt+"\n"
    memo=memo+"negative_prompt\n"+n_prompt+"\n"
    meta_dict["pr"]=prompt
    meta_dict["ne"]=n_prompt
    clear_output(True)
    print(memo)

    pipe.enable_vae_tiling()
    pipe.enable_xformers_memory_efficient_attention()
    
    comple = CompelForSDXL(pipe)
    conditioning = comple(prompt, negative_prompt=n_prompt)
    del comple

    if not(os.path.isdir(out_folder)):
        os.mkdir(out_folder)

    images=[]
    for i in range(pic_number):
            
        if prog_ver==2 or prog_ver==1:
            image = pipe(
                eta=1.0,
                prompt_embeds=conditioning.embeds,
                pooled_prompt_embeds=conditioning.pooled_embeds,
                negative_prompt_embeds=conditioning.negative_embeds,
                negative_pooled_prompt_embeds=conditioning.negative_pooled_embeds,
                height=tate[0],
                width=yoko[0],
                guidance_scale=gs,
                num_inference_steps=f_step,
                clip_skip=cs,generator=torch.manual_seed(seed[i])
            ).images[0]
            if prog_ver==2:
                image0=image.resize((int(sum(yoko)/2), int(sum(tate)/2)), resample=p)
                images.append(image0)
                del image,image0
                torch.cuda.empty_cache()
            else:
                image0=image.resize((yoko[1], tate[1]), resample=p)
                images.append(image0)
                del image,image0
                torch.cuda.empty_cache()
        else:
            image = pipe(
                eta=1.0,
                prompt_embeds=conditioning.embeds,
                pooled_prompt_embeds=conditioning.pooled_embeds,
                negative_prompt_embeds=conditioning.negative_embeds,
                negative_pooled_prompt_embeds=conditioning.negative_pooled_embeds,
                height=tate[1],
                width=yoko[1],
                guidance_scale=gs,
                num_inference_steps=step,
                clip_skip=cs,
                generator=torch.manual_seed(seed[i])
            ).images[0]
            meta_dict["se"]=str(seed[i])
            meta_dict["input"]=out_folder+"/"+str(i)+"_"+str(seed[i])+".png"
            plus_meta(meta_dict,image)
            print(str(i)+"_"+str(seed[i])+".png")
            display(image)
            del image
            torch.cuda.empty_cache()

    del pipe,conditioning

    if prog_ver==2 or prog_ver==1:
        if torch.cuda.device_count()==1:
            d="cuda:0"
        else:
            d="cuda:1"
        if os.path.isfile(vae_safe):
            pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(base_safe, torch_dtype=dtype)
            pipe.vae=AutoencoderKL.from_single_file(vae_safe,torch_dtype=dtype)
            pipe.to(d)
        else:
            pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(base_safe, torch_dtype=dtype).to(d)
            
        if sample=="Euler":
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        elif sample=="Euler a":
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        elif sample=="LMS":
            pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        elif sample=="Heun":
            pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
        elif sample=="DPM2":
            pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
        elif sample=="DPM2 a":
            pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        elif sample=="DPM++ 2M":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        elif sample=="DPM++ SDE":
            pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
        elif sample=="DPM++ 2M SDE":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,algorithm_type="sde-dpmsolver++")
        elif sample=="LMS Karras":
            pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        elif sample=="DPM2 Karras":
            pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        elif sample=="DPM2 a Karras":
            pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        elif sample=="DPM++ 2M Karras":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        elif sample=="DPM++ SDE Karras":
            pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        elif sample=="DPM++ 2M SDE Karras":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,algorithm_type="sde-dpmsolver++",use_karras_sigmas=True)
        elif sample=="PLMS":
            pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        elif sample=="UniPC":
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        elif sample=="LCM":
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        elif sample=="DPM++ 3M SDE":
            pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config)
        elif sample=="DPM++ 3M SDE Karras":
            pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        elif sample=="DPM++ 3M SDE Exponential":
            pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config,use_exponential_sigmas=True)
        else:
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            
        if loras!=[]:
            i=0
            for line in loras:
                if line.endswith(".safetensors"):
                    line=line.replace(".safetensors","")
                pipe.load_lora_weights(".",weight_name=line+".safetensors",torch_dtype=dtype)
                pipe.fuse_lora(lora_scale= lora_weights[i])
                pipe.unload_lora_weights()
                i=i+1

        if pos_emb!=[]:
            for line in pos_emb:
                key=os.path.basename(line).replace(".safetensors","")
                state_dict = load_file(line)
                pipe.load_textual_inversion(state_dict["clip_g"],token=key,text_encoder=pipe.text_encoder_2,tokenizer=pipe.tokenizer_2,torch_dtype=dtype)
                pipe.load_textual_inversion(state_dict["clip_l"],token=key,text_encoder=pipe.text_encoder,tokenizer=pipe.tokenizer,torch_dtype=dtype)
                del state_dict

        if neg_emb!=[]:
            for line in neg_emb:
                key=os.path.basename(line).replace(".safetensors","")
                state_dict = load_file(line)
                pipe.load_textual_inversion(state_dict["clip_g"],token=key,text_encoder=pipe.text_encoder_2,tokenizer=pipe.tokenizer_2,torch_dtype=dtype)
                pipe.load_textual_inversion(state_dict["clip_l"],token=key,text_encoder=pipe.text_encoder,tokenizer=pipe.tokenizer,torch_dtype=dtype)
                del state_dict

        clear_output(True)
        print(memo)

        pipe.enable_vae_tiling()
        pipe.enable_xformers_memory_efficient_attention()
                
        comple = comple = CompelForSDXL(pipe)
        conditioning = comple(prompt, negative_prompt=n_prompt)
        del comple

        for i in range(pic_number):
            if prog_ver==2:
                image = pipe(
                    eta=1.0,
                    prompt_embeds=conditioning.embeds,
                    pooled_prompt_embeds=conditioning.pooled_embeds,
                    negative_prompt_embeds=conditioning.negative_embeds,
                    negative_pooled_prompt_embeds=conditioning.negative_pooled_embeds,
                    image=images[i],
                    guidance_scale=gs,
                    generator=torch.manual_seed(seed[i]),
                    num_inference_steps=int((step+f_step)/2/ss)+1,
                    clip_skip=cs,
                    strength=ss
                ).images[0]
                image0=image.resize((yoko[1], tate[1]), resample=p)
                images[i]=image0
                del image,image0
                torch.cuda.empty_cache()
            image = pipe(
                eta=1.0,
                prompt_embeds=conditioning.embeds,
                pooled_prompt_embeds=conditioning.pooled_embeds,
                negative_prompt_embeds=conditioning.negative_embeds,
                negative_pooled_prompt_embeds=conditioning.negative_pooled_embeds,
                image=images[i],
                guidance_scale=gs,
                generator=torch.manual_seed(seed[i]),
                num_inference_steps=int(step/ss)+1,
                clip_skip=cs,
                strength=ss
            ).images[0]
            meta_dict["se"]=str(seed[i])
            meta_dict["input"]=out_folder+"/"+str(i)+"_"+str(seed[i])+".png"
            plus_meta(meta_dict,image)
            print(str(i)+"_"+str(seed[i])+".png")
            display(image)
            del image
            torch.cuda.empty_cache()
        del pipe,conditioning
    del images
    return seed

def text2image15(loras=[], lora_weights=[], prompt = "", n_prompt = "", t="v", prog_ver=2, pic_number=10, gs=7, f_step=10,step=30, ss=0.6, cs=2, Interpolation=3, sample=1,seed=0,out_folder="data",pos_emb=[],neg_emb=[],base_safe="base.safetensors",vae_safe="vae.safetensors"):
    meta_dict={}
    memo="seed\n"
    if isinstance(seed, list):
        pic_number=len(seed)
        for i in range(pic_number):
            try:
                if int(seed[i])==0:
                    seed[i]=random.randint(1, 1000000000)
                else:
                    seed[i]=int(seed[i])
            except:
                seed[i]=random.randint(1, 1000000000)
            memo=memo+str(seed[i])+"\n"
    else:
        try:
            if int(seed)==0:
                seed=[]
                for i in range(pic_number):
                    seed.append(random.randint(1, 1000000000))
            else:
                seed=[int(seed)]
                pic_number=1
        except:
            seed=[]
            for i in range(pic_number):
                seed.append(random.randint(1, 1000000000))
        for i in range(pic_number):
            memo=memo+str(seed[i])+"\n"
    print(memo)
            
    if prog_ver!=1:
        if prog_ver!=2:
            prog_ver=0
        
    if not(os.path.isfile(base_safe)):
        print("the checkpoint file does not exist.")
        return []
    memo=memo+"checkpoint : "+base_safe+"\n"
    clear_output(True)
    print(memo)
    try:
        f=safetensors.safe_open(base_safe, framework="pt", device="cpu")
        meta_dict["ckpt"]=f.metadata()["id"]
        del f
    except:
        meta_dict["ckpt"]=""
    
    if os.path.isfile(vae_safe):
        memo=memo+"vae : "+vae_safe+"\n"
        try:
            f=safetensors.safe_open(vae_safe, framework="pt", device="cpu")
            meta_dict["vae"]=f.metadata()["id"]
            del f
        except:
            meta_dict["vae"]=""
    else:
        memo=memo+"vae : original vae"+"\n"
        meta_dict["vae"]=""
    clear_output(True)
    print(memo)
    if loras!=[]:    
        if len(loras)!=len(lora_weights):
            print("the number of lora does not equal the number of lora weight.")
            return []

        i=0
        memo=memo+"lora : weight\n"
        meta_id_list=[]
        meta_weight_list=[]
        for line in loras:
            if line.endswith(".safetensors"):
                line=line.replace(".safetensors","")
            if os.path.isfile(line+".safetensors"):
                memo=memo+line+".safetensors : "+str(lora_weights[i])+" ok\n"
                clear_output(True)
                print(memo)
                list1=meta_id_list
                list2=meta_weight_list
                try:
                    f=safetensors.safe_open(line+".safetensors", framework="pt", device="cpu")
                    meta_id=f.metadata()["id"]
                    meta_id = ast.literal_eval(meta_id)
                    for j in meta_id:
                        meta_id_list.append(j)
                    del meta_id
                    meta_weight=f.metadata()["weight"]
                    meta_weight = ast.literal_eval(meta_weight)
                    for j in meta_weight:
                        meta_weight_list.append(float(j)*lora_weights[i])
                    del meta_weight
                    del f
                except:
                    meta_id_list=list1
                    meta_weight_list=list2
                
            else:
                print(line+".safetensors : "+str(lora_weights[i])+" ng")
                return []
            i=i+1

        if len(meta_weight_list)!=len(meta_id_list):
            meta_id_list=[]
            meta_weight_list=[]
        meta_dict["lora"]=str(meta_id_list)
        meta_dict["w"]=str(meta_weight_list)
    
    meta_embed_list=[]
    memo=memo+"Positive Embedding\n"
    clear_output(True)
    print(memo)
    if pos_emb==[]:
        memo=memo+"nothing\n"
        clear_output(True)
        print(memo)
    else:
        for line in pos_emb:
            if os.path.isfile(line):
                memo=memo+line+" ok\n"
                clear_output(True)
                print(memo)
                list1=meta_embed_list
                try:
                    f=safetensors.safe_open(line, framework="pt", device="cpu")
                    meta_embed_list.append(f.metadata()["id"])
                    del f
                except:
                    meta_embed_list=list1
                
            else:
                print(line+" ng")
                return []
                
    memo=memo+"Negative Embedding\n"
    clear_output(True)
    print(memo)
    if neg_emb==[]:
        memo=memo+"nothing\n"
        clear_output(True)
        print(memo)
    else:
        for line in neg_emb:
            if os.path.isfile(line):
                memo=memo+line+" ok\n"
                clear_output(True)
                print(memo)
                list1=meta_embed_list
                try:
                    f=safetensors.safe_open(line, framework="pt", device="cpu")
                    meta_embed_list.append(f.metadata()["id"])
                    del f
                except:
                    meta_embed_list=list1
            else:
                print(line+" ng")
                return []
    meta_dict["embed"]=str(meta_embed_list)

    if t=="v":
        tate=[800,1280]
        yoko=[600,900]
    elif t=="s":
        tate=[800,1280]
        yoko=[800,1280]
    elif t=="h":
        yoko=[800,1280]
        tate=[600,960]
    elif t=="vl":
        tate=[800,1600]
        yoko=[600,1200]
    elif t=="sl":
        tate=[800,1600]
        yoko=[800,1600]
    elif t=="hl":
        yoko=[800,1600]
        tate=[600,1200]
    else:
        t_list=t.split(",")
        if len(t_list)==4:
            iw=round(float(t_list[0])/8)*8
            ow=round(float(t_list[1])/8)*8
            ih=round(float(t_list[2])/8)*8
            oh=round(float(t_list[3])/8)*8

            yoko=[iw,ow]
            tate=[ih,oh]
        else:
            print("t setting is error.")
            print(" initial width, output width, initial height, output height")
            return []

    if prog_ver!=0:
        if Interpolation==1:
            p=Image.NEAREST
            memo=memo+"Interpolation : NEAREST\n"
            meta_dict["hum"]="NEAREST"
        elif Interpolation==2:
            p=Image.BOX
            memo=memo+"Interpolation : BOX\n"
            meta_dict["hum"]="BOX"
        elif Interpolation==3:
            p=Image.BILINEAR
            memo=memo+"Interpolation : BILINEAR\n"
            meta_dict["hum"]="BILINEAR"
        elif Interpolation==4:
            p=Image.HAMMING
            memo=memo+"Interpolation : HAMMING\n"
            meta_dict["hum"]="HAMMING"
        elif Interpolation==5:
            p=Image.BICUBIC
            memo=memo+"Interpolation : BICUBIC\n"
            meta_dict["hum"]="BICUBIC"
        else:
            p=Image.LANCZOS
            memo=memo+"Interpolation : LANCZOS\n"
            meta_dict["hum"]="LANCZOS"
    else:
        meta_dict["hum"]=""

    del t,Interpolation
    clear_output(True)
    print(memo)

    dtype=torch.float16
    if os.path.isfile(vae_safe):
        pipe = StableDiffusionPipeline.from_single_file(base_safe, torch_dtype=dtype)
        pipe.vae=AutoencoderKL.from_single_file(vae_safe,torch_dtype=dtype)
        pipe.to("cuda:0")
    else:
        pipe = StableDiffusionPipeline.from_single_file(base_safe, torch_dtype=dtype).to("cuda:0")
        
    if sample=="Euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : Euler\n"
        meta_dict["sa"]=sample
    elif sample=="Euler a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : Euler a\n"
        meta_dict["sa"]=sample
    elif sample=="LMS":
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : LMS\n"
        meta_dict["sa"]=sample
    elif sample=="Heun":
        pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : Heun\n"
        meta_dict["sa"]=sample
    elif sample=="DPM2":
        pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : DPM2\n"
        meta_dict["sa"]=sample
    elif sample=="DPM2 a":
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : DPM2 a\n"
        meta_dict["sa"]=sample
    elif sample=="DPM++ 2M":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : DPM++ 2M\n"
        meta_dict["sa"]=sample
    elif sample=="DPM++ SDE":
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : DPM++ SDE\n"
        meta_dict["sa"]=sample
    elif sample=="DPM++ 2M SDE":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,algorithm_type="sde-dpmsolver++")
        memo=memo+"scheduler : DPM++ 2M SDE\n"
        meta_dict["sa"]=sample
    elif sample=="LMS Karras":
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        memo=memo+"scheduler : LMS Karras\n"
        meta_dict["sa"]=sample
    elif sample=="DPM2 Karras":
        pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        memo=memo+"scheduler : DPM2 Karras\n"
        meta_dict["sa"]=sample
    elif sample=="DPM2 a Karras":
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        memo=memo+"scheduler : DPM2 a Karras\n"
        meta_dict["sa"]=sample
    elif sample=="DPM++ 2M Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        memo=memo+"scheduler : DPM++ 2M Karras\n"
        meta_dict["sa"]=sample
    elif sample=="DPM++ SDE Karras":
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        memo=memo+"scheduler : DPM++ SDE Karras\n"
        meta_dict["sa"]=sample
    elif sample=="DPM++ 2M SDE Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,algorithm_type="sde-dpmsolver++",use_karras_sigmas=True)
        memo=memo+"scheduler : DPM++ 2M SDE Karras\n"
        meta_dict["sa"]=sample
    elif sample=="PLMS":
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : PLMS\n"
        meta_dict["sa"]=sample
    elif sample=="UniPC":
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : UniPC\n"
        meta_dict["sa"]=sample
    elif sample=="LCM":
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : LCM\n"
        meta_dict["sa"]=sample
    elif sample=="DPM++ 3M SDE":
        pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : DPM++ 3M SDE\n"
        meta_dict["sa"]=sample
    elif sample=="DPM++ 3M SDE Karras":
        pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        memo=memo+"scheduler : DPM++ 3M SDE Karras\n"
        meta_dict["sa"]=sample
    elif sample=="DPM++ 3M SDE Exponential":
        pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config,use_exponential_sigmas=True)
        memo=memo+"scheduler : DPM++ 3M SDE Exponential\n"
        meta_dict["sa"]=sample
    else:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : DDIM\n"
        meta_dict["sa"]="DDIM"
    
    memo=memo+"num_inference_steps : "+str(f_step)+"\n"
    meta_dict["st"]=str(f_step)
    memo=memo+"guidance_scale : "+str(gs)+"\n"
    meta_dict["cf"]=str(gs)
    memo=memo+"clip_skip : "+str(cs)+"\n"
    meta_dict["cl"]=str(cs)

    if prog_ver!=0:
        memo=memo+"Hires steps : "+str(step)+"\n"
        meta_dict["hs"]=str(step)
        memo=memo+"Denoising strength : "+str(ss)+"\n"
        meta_dict["ds"]=str(ss)
        memo=memo+"Hires upscale : "+str(yoko[1]/yoko[0])+"\n"
        meta_dict["hu"]=str(yoko[1]/yoko[0])
    else:
        meta_dict["hs"]=""
        meta_dict["ds"]=""
        meta_dict["hu"]=""

    clear_output(True)
    print(memo)
        
    if loras!=[]:
        i=0
        for line in loras:
            if line.endswith(".safetensors"):
                line=line.replace(".safetensors","")
            pipe.load_lora_weights(".",weight_name=line+".safetensors",torch_dtype=dtype)
            memo=memo+line+".safetensors is loaded.\n"
            clear_output(True)
            print(memo)
            pipe.fuse_lora(lora_scale= lora_weights[i])
            pipe.unload_lora_weights()
            i=i+1

    if pos_emb!=[]:
        for line in pos_emb:
            key=os.path.basename(line).replace(".safetensors","")
            pipe.load_textual_inversion(".", weight_name=line, token=key)
            prompt = prompt+","+key
            memo=memo+line+" is loaded.\n"
            clear_output(True)
            print(memo)

    if neg_emb!=[]:
        for line in neg_emb:
            key=os.path.basename(line).replace(".safetensors","")
            pipe.load_textual_inversion(".", weight_name=line, token=key)
            n_prompt=n_prompt+","+key
            memo=memo+line+" is loaded.\n"
            clear_output(True)
            print(memo)

    memo=memo+"prompt\n"+prompt+"\n"
    memo=memo+"negative_prompt\n"+n_prompt+"\n"
    meta_dict["pr"]=prompt
    meta_dict["ne"]=n_prompt
    clear_output(True)
    print(memo)
        
    pipe.enable_vae_tiling()
    pipe.enable_xformers_memory_efficient_attention()
    
    comple = CompelForSD(pipe)
    conditioning = comple(prompt, negative_prompt=n_prompt)
    del comple
    
    if not(os.path.isdir(out_folder)):
        os.mkdir(out_folder)

    images=[]
    for i in range(pic_number):
        if prog_ver==2 or prog_ver==1:
            image = pipe(
                eta=1.0,
                prompt_embeds=conditioning.embeds,
                negative_prompt_embeds=conditioning.negative_embeds,
                height=tate[0],
                width=yoko[0],
                guidance_scale=gs,
                num_inference_steps=f_step,
                clip_skip=cs,
                generator=torch.manual_seed(seed[i])
            ).images[0]
            if prog_ver==2:
                image0=image.resize((int(sum(yoko)/2), int(sum(tate)/2)), resample=p)
                images.append(image0)
                del image,image0
                torch.cuda.empty_cache()
            else:
                image0=image.resize((yoko[1], tate[1]), resample=p)
                images.append(image0)
                del image,image0
                torch.cuda.empty_cache()
        else:
            image = pipe(
                eta=1.0,
                prompt_embeds=conditioning.embeds,
                negative_prompt_embeds=conditioning.negative_embeds,
                height=tate[1],
                width=yoko[1],
                guidance_scale=gs,
                num_inference_steps=step,
                clip_skip=cs,
                generator=torch.manual_seed(seed[i])
            ).images[0]
            meta_dict["se"]=str(seed[i])
            meta_dict["input"]=out_folder+"/"+str(i)+"_"+str(seed[i])+".png"
            plus_meta(meta_dict,image)
            print(str(i)+"_"+str(seed[i])+".png")
            display(image)
            del image
            torch.cuda.empty_cache()

    del pipe,conditioning

    if prog_ver==2 or prog_ver==1:
        if torch.cuda.device_count()==1:
            d="cuda:0"
        else:
            d="cuda:1"
        if os.path.isfile(vae_safe):
            pipe = StableDiffusionImg2ImgPipeline.from_single_file(base_safe, torch_dtype=dtype)
            pipe.vae=AutoencoderKL.from_single_file(vae_safe,torch_dtype=dtype)
            pipe.to(d)
        else:
            pipe = StableDiffusionImg2ImgPipeline.from_single_file(base_safe, torch_dtype=dtype).to(d)
            
        if sample=="Euler":
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        elif sample=="Euler a":
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        elif sample=="LMS":
            pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        elif sample=="Heun":
            pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
        elif sample=="DPM2":
            pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
        elif sample=="DPM2 a":
            pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        elif sample=="DPM++ 2M":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        elif sample=="DPM++ SDE":
            pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
        elif sample=="DPM++ 2M SDE":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,algorithm_type="sde-dpmsolver++")
        elif sample=="LMS Karras":
            pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        elif sample=="DPM2 Karras":
            pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        elif sample=="DPM2 a Karras":
            pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        elif sample=="DPM++ 2M Karras":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        elif sample=="DPM++ SDE Karras":
            pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        elif sample=="DPM++ 2M SDE Karras":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,algorithm_type="sde-dpmsolver++",use_karras_sigmas=True)
        elif sample=="PLMS":
            pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        elif sample=="UniPC":
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        elif sample=="LCM":
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        elif sample=="DPM++ 3M SDE":
            pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config)
        elif sample=="DPM++ 3M SDE Karras":
            pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        elif sample=="DPM++ 3M SDE Exponential":
            pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config,use_exponential_sigmas=True)
        else:
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            
        if loras!=[]:
            i=0
            for line in loras:
                if line.endswith(".safetensors"):
                    line=line.replace(".safetensors","")
                pipe.load_lora_weights(".",weight_name=line+".safetensors",torch_dtype=dtype)
                pipe.fuse_lora(lora_scale= lora_weights[i])
                pipe.unload_lora_weights()
                i=i+1
            
        if pos_emb!=[]:
            for line in pos_emb:
                key=os.path.basename(line).replace(".safetensors","")
                pipe.load_textual_inversion(".", weight_name=line, token=key)
                prompt = prompt+","+key

        if neg_emb!=[]:
            for line in neg_emb:
                key=os.path.basename(line).replace(".safetensors","")
                pipe.load_textual_inversion(".", weight_name=line, token=key)
                n_prompt=n_prompt+","+key

        clear_output(True)
        print(memo)
            
        pipe.enable_vae_tiling()
        pipe.enable_xformers_memory_efficient_attention()
        
        comple = CompelForSD(pipe)
        conditioning = comple(prompt, negative_prompt=n_prompt)
        del comple

        for i in range(pic_number):
            if prog_ver==2:
                image = pipe(
                    eta=1.0,
                    prompt_embeds=conditioning.embeds,
                    negative_prompt_embeds=conditioning.negative_embeds,
                    image=images[i],
                    guidance_scale=gs,
                    generator=torch.manual_seed(seed[i]),
                    num_inference_steps=int((step+f_step)/2/ss)+1,
                    clip_skip=cs,
                    strength=ss
                ).images[0]
                image0=image.resize((yoko[1], tate[1]), resample=p)
                images[i]=image0
                del image,image0
                torch.cuda.empty_cache()
            image = pipe(
                eta=1.0,
                prompt_embeds=conditioning.embeds,
                negative_prompt_embeds=conditioning.negative_embeds,
                image=images[i],
                guidance_scale=gs,
                generator=torch.manual_seed(seed[i]),
                num_inference_steps=int(step/ss)+1,
                clip_skip=cs,
                strength=ss
            ).images[0]
            meta_dict["se"]=str(seed[i])
            meta_dict["input"]=out_folder+"/"+str(i)+"_"+str(seed[i])+".png"
            plus_meta(meta_dict,image)
            print(str(i)+"_"+str(seed[i])+".png")
            display(image)
            del image
            torch.cuda.empty_cache()
        del pipe,conditioning
    del images
    return seed
