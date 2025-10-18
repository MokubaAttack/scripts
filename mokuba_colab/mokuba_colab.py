from diffusers import StableDiffusionXLPipeline,StableDiffusionXLImg2ImgPipeline,AutoencoderKL
from diffusers import StableDiffusionPipeline,StableDiffusionImg2ImgPipeline
from safetensors.torch import load_file
import torch
import random
import os
import shutil
import datetime
from PIL import Image
from IPython.display import display
from compel import Compel, ReturnedEmbeddingsType
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

def text2image(loras=[], lora_weights=[], prompt = "", n_prompt = "", t="v", prog_ver=2, pic_number=10, gs=7,f_step=10, step=30, ss=0.6, cs=1, Interpolation=3, sample="DDIM",seed=0,out_folder="data",pos_emb=[],neg_emb=[],base_safe="base.safetensors",vae_safe="vae.safetensors"):
    
    memo="seed\n"
    print("seed")
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
            print(seed[i])
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
            print(seed[i])
            memo=memo+str(seed[i])+"\n"
            
    if prog_ver!=1:
        if prog_ver!=2:
            prog_ver=0
        
    if not(os.path.isfile(base_safe)):
        print("the checkpoint file does not exist.")
        return []
    print("checkpoint : "+base_safe)
    memo=memo+"checkpoint : "+base_safe+"\n"
    
    if os.path.isfile(vae_safe):
        print("vae : "+vae_safe)
        memo=memo+"vae : "+vae_safe+"\n"
    else:
        print("vae : normal vae")
        memo=memo+"vae : normal vae"+"\n"
    if loras!=[]:    
        if len(loras)!=len(lora_weights):
            print("the number of lora does not equal the number of lora weight.")
            return []

        i=0
        print("lora : weight")
        memo=memo+"lora : weight\n"
        for line in loras:
            if line.endswith(".safetensors"):
                line=line.replace(".safetensors","")
            if os.path.isfile(line+".safetensors"):
                print(line+".safetensors : "+str(lora_weights[i])+" ok")
                memo=memo+line+".safetensors : "+str(lora_weights[i])+"\n"
            else:
                print(line+".safetensors : "+str(lora_weights[i])+" ng")
                return []
            i=i+1
    
    print("Positive Embedding")
    memo=memo+"Positive Embedding\n"
    if pos_emb==[]:
        print("nothing")
        memo=memo+"nothing\n"
    else:
        for line in pos_emb:
            if os.path.isfile(line):
                print(line+" ok")
                memo=memo+line+"\n"
            else:
                print(line+" ng")
                return []
                
    print("Negative Embedding")
    memo=memo+"Negative Embedding\n"
    if neg_emb==[]:
        print("nothing")
        memo=memo+"nothing\n"
    else:
        for line in neg_emb:
            if os.path.isfile(line):
                print(line+" ok")
                memo=memo+line+"\n"
            else:
                print(line+" ng")
                return []

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

    if Interpolation==1:
        p=Image.NEAREST
    elif Interpolation==2:
        p=Image.BOX
    elif Interpolation==3:
        p=Image.BILINEAR
    elif Interpolation==4:
        p=Image.HAMMING
    elif Interpolation==5:
        p=Image.BICUBIC
    else:
        p=Image.LANCZOS
    del t,Interpolation

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
    elif sample=="Euler a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : Euler a\n"
    elif sample=="LMS":
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : LMS\n"
    elif sample=="Heun":
        pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : Heun\n"
    elif sample=="DPM2":
        pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : DPM2\n"
    elif sample=="DPM2 a":
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : DPM2 a\n"
    elif sample=="DPM++ 2M":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : DPM++ 2M\n"
    elif sample=="DPM++ SDE":
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : DPM++ SDE\n"
    elif sample=="DPM++ 2M SDE":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,algorithm_type="sde-dpmsolver++")
        memo=memo+"scheduler : DPM++ 2M SDE\n"
    elif sample=="LMS Karras":
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        memo=memo+"scheduler : LMS Karras\n"
    elif sample=="DPM2 Karras":
        pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        memo=memo+"scheduler : DPM2 Karras\n"
    elif sample=="DPM2 a Karras":
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        memo=memo+"scheduler : DPM2 a Karras\n"
    elif sample=="DPM++ 2M Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        memo=memo+"scheduler : DPM++ 2M Karras\n"
    elif sample=="DPM++ SDE Karras":
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        memo=memo+"scheduler : DPM++ SDE Karras\n"
    elif sample=="DPM++ 2M SDE Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,algorithm_type="sde-dpmsolver++",use_karras_sigmas=True)
        memo=memo+"scheduler : DPM++ 2M SDE Karras\n"
    elif sample=="PLMS":
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : PLMS\n"
    elif sample=="UniPC":
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : UniPC\n"
    elif sample=="LCM":
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : LCM\n"
    elif sample=="DPM++ 3M SDE":
        pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : DPM++ 3M SDE\n"
    elif sample=="DPM++ 3M SDE Karras":
        pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        memo=memo+"scheduler : DPM++ 3M SDE Karras\n"
    elif sample=="DPM++ 3M SDE Exponential":
        pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config,use_exponential_sigmas=True)
        memo=memo+"scheduler : DPM++ 3M SDE Exponential\n"
    else:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : DDIM\n"
    
    memo=memo+"prompt\n"+prompt+"\n"
    memo=memo+"negative_prompt\n"+n_prompt+"\n"
    memo=memo+"num_inference_steps : "+str(f_step)+"\n"
    memo=memo+"Hires steps : "+str(step)+"\n"
    memo=memo+"guidance_scale : "+str(gs)+"\n"
    memo=memo+"clip_skip : "+str(cs)+"\n"
    memo=memo+"Denoising strength : "+str(ss)
    
    dt_now = datetime.datetime.now()
    dt_now=dt_now.strftime('%Y-%m-%d-%H-%M-%S')+".txt"
    logfile=open(out_folder+"/"+dt_now,"w")
    logfile.write(memo)
    logfile.close()
    del dt_now,logfile
    
    if loras!=[]:
        i=0
        for line in loras:
            if line.endswith(".safetensors"):
                line=line.replace(".safetensors","")
            pipe.load_lora_weights(".",weight_name=line+".safetensors",torch_dtype=dtype)
            print(line+".safetensors is loaded.")
            pipe.fuse_lora(lora_scale= lora_weights[i])
            pipe.unload_lora_weights()
            i=i+1
        
    if pos_emb!=[]:
        i=1
        for line in pos_emb:
            key="mokupos"+str(i)
            state_dict = load_file(line)
            pipe.load_textual_inversion(state_dict["clip_g"],token=key,text_encoder=pipe.text_encoder_2,tokenizer=pipe.tokenizer_2,torch_dtype=dtype)
            pipe.load_textual_inversion(state_dict["clip_l"],token=key,text_encoder=pipe.text_encoder,tokenizer=pipe.tokenizer,torch_dtype=dtype)
            del state_dict
            prompt = prompt+","+key
            print(line+" is loaded.")
            i=i+1

    if neg_emb!=[]:
        i=1
        for line in neg_emb:
            key="mokuneg"+str(i)
            state_dict = load_file(line)
            pipe.load_textual_inversion(state_dict["clip_g"],token=key,text_encoder=pipe.text_encoder_2,tokenizer=pipe.tokenizer_2,torch_dtype=dtype)
            pipe.load_textual_inversion(state_dict["clip_l"],token=key,text_encoder=pipe.text_encoder,tokenizer=pipe.tokenizer,torch_dtype=dtype)
            del state_dict
            n_prompt=n_prompt+","+key
            print(line+" is loaded.")
            i=i+1
    
    pipe.enable_vae_tiling()
    pipe.enable_xformers_memory_efficient_attention()
    
    comple = Compel(
        truncate_long_prompts=False,
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2] ,
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True]
    )
    prompt_embed, prompt_pooled = comple(prompt)
    negative_embed, negative_pooled = comple(n_prompt)
    [prompt_embed, negative_embed] = comple.pad_conditioning_tensors_to_same_length([prompt_embed, negative_embed])
    del comple

    if not(os.path.isdir(out_folder)):
        os.mkdir(out_folder)

    images=[]
    for i in range(pic_number):
            
        if prog_ver==2 or prog_ver==1:
            image = pipe(
                eta=1.0,
                prompt_embeds=prompt_embed,
                pooled_prompt_embeds=prompt_pooled,
                negative_prompt_embeds=negative_embed,
                negative_pooled_prompt_embeds=negative_pooled,
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
                prompt_embeds=prompt_embed,
                pooled_prompt_embeds=prompt_pooled,
                negative_prompt_embeds=negative_embed,
                negative_pooled_prompt_embeds=negative_pooled,
                height=tate[1],
                width=yoko[1],
                guidance_scale=gs,
                num_inference_steps=step,
                clip_skip=cs,
                generator=torch.manual_seed(seed[i])
            ).images[0]
            image.save(out_folder+"/"+str(i)+"_"+str(seed[i])+".png")
            print(str(i)+"_"+str(seed[i])+".png")
            display(image)
            del image
            torch.cuda.empty_cache()

    del pipe,prompt_embed, negative_embed,prompt_pooled,negative_pooled

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
            i=1
            for line in pos_emb:
                key="mokupos"+str(i)
                state_dict = load_file(line)
                pipe.load_textual_inversion(state_dict["clip_g"],token=key,text_encoder=pipe.text_encoder_2,tokenizer=pipe.tokenizer_2,torch_dtype=dtype)
                pipe.load_textual_inversion(state_dict["clip_l"],token=key,text_encoder=pipe.text_encoder,tokenizer=pipe.tokenizer,torch_dtype=dtype)
                del state_dict
                i=i+1

        if neg_emb!=[]:
            i=1
            for line in neg_emb:
                key="mokuneg"+str(i)
                state_dict = load_file(line)
                pipe.load_textual_inversion(state_dict["clip_g"],token=key,text_encoder=pipe.text_encoder_2,tokenizer=pipe.tokenizer_2,torch_dtype=dtype)
                pipe.load_textual_inversion(state_dict["clip_l"],token=key,text_encoder=pipe.text_encoder,tokenizer=pipe.tokenizer,torch_dtype=dtype)
                del state_dict
                i=i+1

        pipe.enable_vae_tiling()
        pipe.enable_xformers_memory_efficient_attention()
                
        comple = Compel(
            truncate_long_prompts=False,
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2] ,
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )
        prompt_embed, prompt_pooled = comple(prompt)
        negative_embed, negative_pooled = comple(n_prompt)
        [prompt_embed, negative_embed] = comple.pad_conditioning_tensors_to_same_length([prompt_embed, negative_embed])
        del comple

        for i in range(pic_number):
            if prog_ver==2:
                image = pipe(
                    eta=1.0,
                    prompt_embeds=prompt_embed,
                    pooled_prompt_embeds=prompt_pooled,
                    negative_prompt_embeds=negative_embed,
                    negative_pooled_prompt_embeds=negative_pooled,
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
            image = pipe(eta=1.0,prompt_embeds=prompt_embed,pooled_prompt_embeds=prompt_pooled,negative_prompt_embeds=negative_embed,negative_pooled_prompt_embeds=negative_pooled,image=images[i],guidance_scale=gs,generator=torch.manual_seed(seed[i]), num_inference_steps=int(step/ss)+1,clip_skip=cs,strength=ss).images[0]
            image.save(out_folder+"/"+str(i)+"_"+str(seed[i])+".png")
            print(str(i)+"_"+str(seed[i])+".png")
            display(image)
            del image
            torch.cuda.empty_cache()
        del pipe,prompt_embed, negative_embed,prompt_pooled,negative_pooled
    del images
    return seed

def text2image15(loras=[], lora_weights=[], prompt = "", n_prompt = "", t="v", prog_ver=2, pic_number=10, gs=7, f_step=10,step=30, ss=0.6, cs=2, Interpolation=3, sample=1,seed=0,out_folder="data",pos_emb=[],neg_emb=[],base_safe="base.safetensors",vae_safe="vae.safetensors"):
    
    memo="seed\n"
    print("seed")
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
            print(seed[i])
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
            print(seed[i])
            memo=memo+str(seed[i])+"\n"
            
    if prog_ver!=1:
        if prog_ver!=2:
            prog_ver=0
        
    if not(os.path.isfile(base_safe)):
        print("the checkpoint file does not exist.")
        return []
    print("checkpoint : "+base_safe)
    memo=memo+"checkpoint : "+base_safe+"\n"
    
    if os.path.isfile(vae_safe):
        print("vae : "+vae_safe)
        memo=memo+"vae : "+vae_safe+"\n"
    else:
        print("vae : normal vae")
        memo=memo+"vae : normal vae"+"\n"
    if loras!=[]:    
        if len(loras)!=len(lora_weights):
            print("the number of lora does not equal the number of lora weight.")
            return []

        i=0
        print("lora : weight")
        memo=memo+"lora : weight\n"
        for line in loras:
            if line.endswith(".safetensors"):
                line=line.replace(".safetensors","")
            if os.path.isfile(line+".safetensors"):
                print(line+".safetensors : "+str(lora_weights[i])+" ok")
                memo=memo+line+".safetensors : "+str(lora_weights[i])+"\n"
            else:
                print(line+".safetensors : "+str(lora_weights[i])+" ng")
                return []
            i=i+1
    
    print("Positive Embedding")
    memo=memo+"Positive Embedding\n"
    if pos_emb==[]:
        print("nothing")
        memo=memo+"nothing\n"
    else:
        for line in pos_emb:
            if os.path.isfile(line):
                print(line+" ok")
                memo=memo+line+"\n"
            else:
                print(line+" ng")
                return []
                
    print("Negative Embedding")
    memo=memo+"Negative Embedding\n"
    if neg_emb==[]:
        print("nothing")
        memo=memo+"nothing\n"
    else:
        for line in neg_emb:
            if os.path.isfile(line):
                print(line+" ok")
                memo=memo+line+"\n"
            else:
                print(line+" ng")
                return []

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

    if Interpolation==1:
        p=Image.NEAREST
    elif Interpolation==2:
        p=Image.BOX
    elif Interpolation==3:
        p=Image.BILINEAR
    elif Interpolation==4:
        p=Image.HAMMING
    elif Interpolation==5:
        p=Image.BICUBIC
    else:
        p=Image.LANCZOS
    del t,Interpolation

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
    elif sample=="Euler a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : Euler a\n"
    elif sample=="LMS":
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : LMS\n"
    elif sample=="Heun":
        pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : Heun\n"
    elif sample=="DPM2":
        pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : DPM2\n"
    elif sample=="DPM2 a":
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : DPM2 a\n"
    elif sample=="DPM++ 2M":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : DPM++ 2M\n"
    elif sample=="DPM++ SDE":
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : DPM++ SDE\n"
    elif sample=="DPM++ 2M SDE":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,algorithm_type="sde-dpmsolver++")
        memo=memo+"scheduler : DPM++ 2M SDE\n"
    elif sample=="LMS Karras":
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        memo=memo+"scheduler : LMS Karras\n"
    elif sample=="DPM2 Karras":
        pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        memo=memo+"scheduler : DPM2 Karras\n"
    elif sample=="DPM2 a Karras":
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        memo=memo+"scheduler : DPM2 a Karras\n"
    elif sample=="DPM++ 2M Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        memo=memo+"scheduler : DPM++ 2M Karras\n"
    elif sample=="DPM++ SDE Karras":
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        memo=memo+"scheduler : DPM++ SDE Karras\n"
    elif sample=="DPM++ 2M SDE Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,algorithm_type="sde-dpmsolver++",use_karras_sigmas=True)
        memo=memo+"scheduler : DPM++ 2M SDE Karras\n"
    elif sample=="PLMS":
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : PLMS\n"
    elif sample=="UniPC":
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : UniPC\n"
    elif sample=="LCM":
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : LCM\n"
    elif sample=="DPM++ 3M SDE":
        pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : DPM++ 3M SDE\n"
    elif sample=="DPM++ 3M SDE Karras":
        pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        memo=memo+"scheduler : DPM++ 3M SDE Karras\n"
    elif sample=="DPM++ 3M SDE Exponential":
        pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config,use_exponential_sigmas=True)
        memo=memo+"scheduler : DPM++ 3M SDE Exponential\n"
    else:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        memo=memo+"scheduler : DDIM\n"
    
    memo=memo+"prompt\n"+prompt+"\n"
    memo=memo+"negative_prompt\n"+n_prompt+"\n"
    memo=memo+"num_inference_steps : "+str(f_step)+"\n"
    memo=memo+"Hires steps : "+str(step)+"\n"
    memo=memo+"guidance_scale : "+str(gs)+"\n"
    memo=memo+"clip_skip : "+str(cs)+"\n"
    memo=memo+"Denoising strength : "+str(ss)
    
    dt_now = datetime.datetime.now()
    dt_now=dt_now.strftime('%Y-%m-%d-%H-%M-%S')+".txt"
    logfile=open(out_folder+"/"+dt_now,"w")
    logfile.write(memo)
    logfile.close()
    del dt_now,logfile
        
    if loras!=[]:
        i=0
        for line in loras:
            if line.endswith(".safetensors"):
                line=line.replace(".safetensors","")
            pipe.load_lora_weights(".",weight_name=line+".safetensors",torch_dtype=dtype)
            print(line+".safetensors is loaded.")
            pipe.fuse_lora(lora_scale= lora_weights[i])
            pipe.unload_lora_weights()
            i=i+1

    if pos_emb!=[]:
        i=1
        for line in pos_emb:
            key="mokupos"+str(i)
            pipe.load_textual_inversion(".", weight_name=line, token=key)
            prompt = prompt+","+key
            print(line+" is loaded.")
            i=i+1

    if neg_emb!=[]:
        i=1
        for line in neg_emb:
            key="mokuneg"+str(i)
            pipe.load_textual_inversion(".", weight_name=line, token=key)
            n_prompt=n_prompt+","+key
            print(line+" is loaded.")
            i=i+1
        
    pipe.enable_vae_tiling()
    pipe.enable_xformers_memory_efficient_attention()
    
    comple = Compel(
        truncate_long_prompts=False,
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder
    )
    prompt_comple=comple(prompt)
    n_prompt_comple=comple(n_prompt)
    del comple
    
    if not(os.path.isdir(out_folder)):
        os.mkdir(out_folder)

    images=[]
    for i in range(pic_number):
        if prog_ver==2 or prog_ver==1:
            image = pipe(
                eta=1.0,
                prompt_embeds=prompt_comple,
                negative_prompt_embeds=n_prompt_comple,
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
                prompt_embeds=prompt_comple,
                negative_prompt_embeds=n_prompt_comple,
                height=tate[1],
                width=yoko[1],
                guidance_scale=gs,
                num_inference_steps=step,
                clip_skip=cs,
                generator=torch.manual_seed(seed[i])
            ).images[0]
            image.save(out_folder+"/"+str(i)+"_"+str(seed[i])+".png")
            print(str(i)+"_"+str(seed[i])+".png")
            display(image)
            del image
            torch.cuda.empty_cache()

    del pipe,prompt_comple,n_prompt_comple

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
            i=1
            for line in pos_emb:
                key="mokupos"+str(i)
                pipe.load_textual_inversion(".", weight_name=line, token=key)
                prompt = prompt+","+key
                i=i+1

        if neg_emb!=[]:
            i=1
            for line in neg_emb:
                key="mokuneg"+str(i)
                pipe.load_textual_inversion(".", weight_name=line, token=key)
                n_prompt=n_prompt+","+key
                i=i+1
            
        pipe.enable_vae_tiling()
        pipe.enable_xformers_memory_efficient_attention()
        
        comple = Compel(
            truncate_long_prompts=False,
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder
        )
        prompt_comple=comple(prompt)
        n_prompt_comple=comple(n_prompt)
        del comple

        for i in range(pic_number):
            if prog_ver==2:
                image = pipe(
                    eta=1.0,
                    prompt_embeds=prompt_comple,
                    negative_prompt_embeds=n_prompt_comple,
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
                prompt_embeds=prompt_comple,
                negative_prompt_embeds=n_prompt_comple,
                image=images[i],
                guidance_scale=gs,
                generator=torch.manual_seed(seed[i]),
                num_inference_steps=int(step/ss)+1,
                clip_skip=cs,
                strength=ss
            ).images[0]
            image.save(out_folder+"/"+str(i)+"_"+str(seed[i])+".png")
            print(str(i)+"_"+str(seed[i])+".png")
            display(image)
            del image
            torch.cuda.empty_cache()
        del pipe,prompt_comple,n_prompt_comple
    del images
    return seed
