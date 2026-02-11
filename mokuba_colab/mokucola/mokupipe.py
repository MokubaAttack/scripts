from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLPAGPipeline,
    StableDiffusionXLPAGImg2ImgPipeline,
    StableDiffusionPAGPipeline,
    StableDiffusionPAGImg2ImgPipeline,
    StableDiffusionXLControlNetPAGImg2ImgPipeline,
    StableDiffusionControlNetPAGInpaintPipeline,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
    LCMScheduler,
    DDIMScheduler,
    DPMSolverSDEScheduler
)
import safetensors
import torch
import os
import gc
import numpy
import cv2
from PIL import Image
from safetensors.torch import load_file
from IPython.display import clear_output
from compel import (
    CompelForSD,
    CompelForSDXL
)

from .meta import plus_meta
from .imgup import imgup
from .discord import to_discord
from .imgshow import imgshow

sgm_use=[
    "Euler","Euler a","DPM++ 2M","DPM++ 2M SDE","DPM++ SDE","DPM++","DPM2","DPM2 a","Heun","LMS","UniPC","DPM++ 3M SDE"
]

def create_gaussian_weight(w,h, sigma=0.3):
    x = numpy.linspace(-1, 1, w)
    y = numpy.linspace(-1, 1, h)
    xx, yy = numpy.meshgrid(x, y)
    gaussian_weight = numpy.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return gaussian_weight

class mokupipe:
    def __init__(self):
        self.meta_dict={}
        self.pipe=None
        self.upscaler=None
    def mkpipe(
        self,
        pos_emb=[],
        neg_emb=[],
        base_safe="base.safetensors",
        vae_safe="",
        loras=[],
        lora_weights=[],
        sample="DDIM",
        sgm=""
        ):
        if not(os.path.exists(base_safe)):
            print("the checkpoint file does not exist.")
            return -1
        try:
            f=safetensors.safe_open(base_safe, framework="pt", device="cpu")
            self.meta_dict["ckpt"]=f.metadata()["id"]
            del f
        except:
            self.meta_dict["ckpt"]=""
        self.meta_dict["ckpt_name"]=base_safe

        if os.path.exists(vae_safe):
            try:
                f=safetensors.safe_open(vae_safe, framework="pt", device="cpu")
                self.meta_dict["vae"]=f.metadata()["id"]
                del f
            except:
                self.meta_dict["vae"]=""
            self.meta_dict["vae_name"]=vae_safe
        else:
            self.meta_dict["vae"]=""
            self.meta_dict["vae_name"]=""

        sd=load_file(base_safe)
        self.is_sdxl="conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in sd

        if self.is_sdxl:
            self.pipe=StableDiffusionXLPAGPipeline.from_single_file(base_safe,torch_dtype=torch.float16)
            print(self.meta_dict["ckpt_name"]+" is loaded.")
            if os.path.isfile(vae_safe):
                self.pipe.vae=AutoencoderKL.from_single_file(vae_safe,torch_dtype=torch.float16)
                print(self.meta_dict["vae_name"]+" is loaded.")
        else:
            self.pipe=StableDiffusionPAGPipeline.from_single_file(base_safe,torch_dtype=torch.float16)
            print(self.meta_dict["ckpt_name"]+" is loaded.")
            if os.path.isfile(vae_safe):
                self.pipe.vae=AutoencoderKL.from_single_file(vae_safe,torch_dtype=torch.float16)
                print(self.meta_dict["vae_name"]+" is loaded.")
        self.pipe.to("cuda")

        self.meta_dict["sa"]=""
        sgm_dict={}
        sgm_dict["use_karras_sigmas"]=False
        sgm_dict["use_exponential_sigmas"]=False
        sgm_dict["use_beta_sigmas"]=False
        if sample in sgm_use:
            if sgm=="Karras":
                sgm_dict["timestep_spacing"]="linspace"
                sgm_dict["use_karras_sigmas"]=True
                self.meta_dict["sa"]=" "+sgm
            elif sgm=="exponential":
                sgm_dict["timestep_spacing"]="linspace"
                sgm_dict["use_exponential_sigmas"]=True
                self.meta_dict["sa"]=" "+sgm
            elif sgm=="beta":
                sgm_dict["timestep_spacing"]="linspace"
                sgm_dict["use_beta_sigmas"]=True
                self.meta_dict["sa"]=" "+sgm
            elif sgm=="sgm_uniform" or sgm=="simple":
                sgm_dict["timestep_spacing"]="trailing"
                self.meta_dict["sa"]=" "+sgm
            else:
                sgm_dict["timestep_spacing"]="linspace"
                self.meta_dict["sa"]=" "+sgm
        else:
            if sgm=="sgm_uniform" or sgm=="simple":
                sgm_dict["timestep_spacing"]="trailing"
                self.meta_dict["sa"]=" "+sgm
            else:
                sgm_dict["timestep_spacing"]="leading"
                self.meta_dict["sa"]=" "+sgm

        if sample=="Euler":
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
            self.meta_dict["sa"]=sample+self.meta_dict["sa"]
        elif sample=="Euler a":
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
            self.meta_dict["sa"]=sample+self.meta_dict["sa"]
        elif sample=="LMS":
            self.pipe.scheduler = LMSDiscreteScheduler.from_config(self.pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
            self.meta_dict["sa"]=sample+self.meta_dict["sa"]
        elif sample=="Heun":
            self.pipe.scheduler = HeunDiscreteScheduler.from_config(self.pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
            self.meta_dict["sa"]=sample+self.meta_dict["sa"]
        elif sample=="DPM2":
            self.pipe.scheduler = KDPM2DiscreteScheduler.from_config(self.pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
            self.meta_dict["sa"]=sample+self.meta_dict["sa"]
        elif sample=="DPM2 a":
            self.pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(self.pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
            self.meta_dict["sa"]=sample+self.meta_dict["sa"]
        elif sample=="DPM++ 2M":
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
            self.meta_dict["sa"]=sample+self.meta_dict["sa"]
        elif sample=="DPM++ SDE":
            self.pipe.scheduler = DPMSolverSinglestepScheduler.from_config(self.pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
            self.meta_dict["sa"]=sample+self.meta_dict["sa"]
        elif sample=="DPM++":
            self.pipe.scheduler = DPMSolverSinglestepScheduler.from_config(self.pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
            self.meta_dict["sa"]=sample+self.meta_dict["sa"]
        elif sample=="DPM++ 2M SDE":
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
            self.meta_dict["sa"]=sample+self.meta_dict["sa"]
        elif sample=="PLMS":
            self.pipe.scheduler = PNDMScheduler.from_config(self.pipe.scheduler.config,timestep_spacing=sgm_dict["timestep_spacing"])
            self.meta_dict["sa"]=sample+self.meta_dict["sa"]
        elif sample=="UniPC":
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
            self.meta_dict["sa"]=sample+self.meta_dict["sa"]
        elif sample=="LCM":
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config,timestep_spacing=sgm_dict["timestep_spacing"])
            self.meta_dict["sa"]=sample+self.meta_dict["sa"]
        elif sample=="DPM++ 3M SDE":
            self.pipe.scheduler = DPMSolverSDEScheduler.from_config(self.pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
            self.meta_dict["sa"]=sample+self.meta_dict["sa"]
        else:
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config,timestep_spacing=sgm_dict["timestep_spacing"])
            self.meta_dict["sa"]=sample+self.meta_dict["sa"]

        if loras!=[]:
            if len(loras)!=len(lora_weights):
                print("the number of lora does not equal the number of lora weight.")
                return -1
            i=0
            meta_id_list=[]
            meta_weight_list=[]
            for line in loras:
                if line.endswith(".safetensors"):
                    line=line.replace(".safetensors","")
                if os.path.exists(line+".safetensors"):
                    self.pipe.load_lora_weights(".",weight_name=line+".safetensors",torch_dtype=torch.float16)
                    print(line+".safetensors is loaded.")
                    self.pipe.fuse_lora(lora_scale= lora_weights[i])
                    self.pipe.unload_lora_weights()

                    list1=meta_id_list
                    list2=meta_weight_list
                    try:
                        f=safetensors.safe_open(line+".safetensors", framework="pt", device="cpu")
                        meta_id=f.metadata()["id"]
                        if "," in meta_id:
                            meta_id = meta_id.split(",")
                            for j in meta_id:
                                meta_id_list.append(int(j))
                        else:
                            meta_id_list.append(int(meta_id))
                        meta_weight=f.metadata()["weight"]
                        if "," in meta_weight:
                            meta_weight = meta_weight.split(",")
                            for j in meta_weight:
                                meta_weight_list.append(float(j)*lora_weights[i])
                        else:
                            meta_weight_list.append(float(meta_weight)*lora_weights[i])
                        del f,meta_id,meta_weight
                    except:
                        meta_id_list=list1
                        meta_weight_list=list2
                    del list1,list2
                else:
                    print(line+".safetensors does not exist.")
                    return -1
                i=i+1
            self.meta_dict["lora"]=str(meta_id_list)
            self.meta_dict["w"]=str(meta_weight_list)
            del meta_id_list,meta_weight_list
        else:
            self.meta_dict["lora"]="[]"
        self.meta_dict["loras"]=loras
        self.meta_dict["lora_weights"]=lora_weights
        
        meta_embed_list=[]
        self.prompt_a=""
        if pos_emb!=[]:
            for line in pos_emb:
                if os.path.exists(line):
                    key=os.path.basename(line).replace(".safetensors","")
                    if self.is_sdxl:
                        state_dict = load_file(line)
                        self.pipe.load_textual_inversion(state_dict["clip_g"],token=key,text_encoder=self.pipe.text_encoder_2,tokenizer=self.pipe.tokenizer_2,torch_dtype=torch.float16)
                        self.pipe.load_textual_inversion(state_dict["clip_l"],token=key,text_encoder=self.pipe.text_encoder,tokenizer=self.pipe.tokenizer,torch_dtype=torch.float16)
                        del state_dict
                    else:
                        self.pipe.load_textual_inversion(".", weight_name=line, token=key)
                    self.prompt_a = self.prompt_a+","+key
                    print(line+".safetensors is loaded.")
                    del key
                    list1=meta_embed_list
                    try:
                        f=safetensors.safe_open(line, framework="pt", device="cpu")
                        meta_embed_list.append(f.metadata()["id"])
                        del f
                    except:
                        meta_embed_list=list1
                    del list1
                else:
                    print(line+" does not exist.")
                    return -1

        self.n_prompt_a=""
        if neg_emb!=[]:
            for line in neg_emb:
                if os.path.exists(line):
                    key=os.path.basename(line).replace(".safetensors","")
                    if self.is_sdxl:
                        state_dict = load_file(line)
                        self.pipe.load_textual_inversion(state_dict["clip_g"],token=key,text_encoder=self.pipe.text_encoder_2,tokenizer=self.pipe.tokenizer_2,torch_dtype=torch.float16)
                        self.pipe.load_textual_inversion(state_dict["clip_l"],token=key,text_encoder=self.pipe.text_encoder,tokenizer=self.pipe.tokenizer,torch_dtype=torch.float16)
                        del state_dict
                    else:
                        self.pipe.load_textual_inversion(".", weight_name=line, token=key)
                    self.n_prompt_a=self.n_prompt_a+","+key
                    print(line+".safetensors is loaded.")
                    del key
                    list1=meta_embed_list
                    try:
                        f=safetensors.safe_open(line, framework="pt", device="cpu")
                        meta_embed_list.append(f.metadata()["id"])
                        del f
                    except:
                        meta_embed_list=list1
                    del list1
                else:
                    print(line+" does not exist.")
                    return -1
        self.meta_dict["embed"]=str(meta_embed_list)
        del meta_embed_list
        self.meta_dict["pos"]=pos_emb
        self.meta_dict["neg"]=neg_emb
        self.prompts=None
        gc.collect()
        return 1

    def mkpipe_upscale(self,path):
        self.upscaler=imgup(path)

    def mkprompt(self,prompt,n_prompt):
        if self.pipe==None:
            print("You must make a pipeline.")
            return -1
        if self.is_sdxl:
            comple = CompelForSDXL(self.pipe)
        else:
            comple = CompelForSD(self.pipe)
        conditioning = comple(prompt, negative_prompt=n_prompt)
        if self.is_sdxl:
            self.prompts=[conditioning.embeds,conditioning.pooled_embeds,conditioning.negative_embeds,conditioning.negative_pooled_embeds]
        else:
            self.prompts=[conditioning.embeds,conditioning.negative_embeds]
        del comple,conditioning
        self.prompt=prompt
        self.n_prompt=n_prompt
        gc.collect()
        return 1

    def text2image(self,prompt,n_prompt,gs,step,cs,seed,pag,x,y,out_folder="",j_or_p="",url="",xf=False):
        if self.pipe==None:
            print("You must make a pipeline.")
            return []
        if self.is_sdxl:
            self.pipe=StableDiffusionXLPAGPipeline.from_pipe(self.pipe,torch_dtype=torch.float16)
        else:
            self.pipe=StableDiffusionPAGPipeline.from_pipe(self.pipe,torch_dtype=torch.float16)
        self.pipe.to("cuda")
        prompt=prompt+self.prompt_a
        n_prompt=n_prompt+self.n_prompt_a
        self.meta_dict["pr"]=prompt
        self.meta_dict["ne"]=n_prompt
        memo="seed\n"
        for i in seed:
            memo=memo+str(i)+"\n"
        memo=memo+"ckpt : "+self.meta_dict["ckpt_name"]+"\n"
        if self.meta_dict["vae_name"]!="":
            memo=memo+"vae : "+self.meta_dict["vae_name"]+"\n"
        memo=memo+"scheduler : "+self.meta_dict["sa"]+"\n"
        if self.meta_dict["loras"]!=[]:
            memo=memo+"lora : weight\n"
            for i in range(len(self.meta_dict["loras"])):
                memo=memo+self.meta_dict["loras"][i]+" : "+str(self.meta_dict["lora_weights"][i])+"\n"
        if self.meta_dict["pos"]!=[]:
            memo=memo+"Positive Embedding\n"
            for i in range(len(self.meta_dict["pos"])):
                memo=memo+self.meta_dict["pos"][i]+"\n"
        if self.meta_dict["neg"]!=[]:
            memo=memo+"Negative Embedding\n"
            for i in range(len(self.meta_dict["neg"])):
                memo=memo+self.meta_dict["neg"][i]+"\n"
        memo=memo+"num_inference_steps : "+str(step)+"\n"
        self.meta_dict["st"]=str(step)
        memo=memo+"guidance_scale : "+str(gs)+"\n"
        self.meta_dict["cf"]=str(gs)
        memo=memo+"clip_skip : "+str(cs)+"\n"
        self.meta_dict["cl"]=str(cs)
        memo=memo+"pag_scale : "+str(pag)+"\n"
        self.meta_dict["pag"]=str(pag)
        memo=memo+"prompt\n"+prompt+"\nnegative prompt\n"+n_prompt+"\n"
        for k in ["hs","ds","hu","hum","up","ccs","cont","tu","tum"]:
            if k in self.meta_dict:
                del self.meta_dict[k]

        self.pipe.vae.enable_tiling()
        if xf:
            self.pipe.enable_xformers_memory_efficient_attention()
        
        if self.prompts==None:
            self.mkprompt(prompt=prompt,n_prompt=n_prompt)
        if self.prompt!=prompt or self.n_prompt!=n_prompt:
            self.mkprompt(prompt=prompt,n_prompt=n_prompt)

        images=[]

        if out_folder!="":
            if not(os.path.exists(out_folder)):
                os.makedirs(out_folder)
        j=0
        for i in seed:
            j=j+1
            clear_output(True)
            print(memo)
            if j>1:
                imgshow(imgs=images)

            if self.is_sdxl:
                image = self.pipe(
                    eta=1.0,
                    prompt_embeds=self.prompts[0],
                    pooled_prompt_embeds=self.prompts[1],
                    negative_prompt_embeds=self.prompts[2],
                    negative_pooled_prompt_embeds=self.prompts[3],
                    height=y,
                    width=x,
                    guidance_scale=gs,
                    num_inference_steps=step,
                    clip_skip=cs,
                    generator=torch.manual_seed(i),
                    pag_scale=pag
                ).images[0]
            else:
                image = self.pipe(
                    eta=1.0,
                    prompt_embeds=self.prompts[0],
                    negative_prompt_embeds=self.prompts[1],
                    height=y,
                    width=x,
                    guidance_scale=gs,
                    num_inference_steps=step,
                    clip_skip=cs,
                    generator=torch.manual_seed(i),
                    pag_scale=pag
                ).images[0]
            if out_folder!="":
                self.meta_dict["se"]=str(i)
                if j_or_p=="j":
                    self.meta_dict["input"]=out_folder+"/"+str(j)+"_"+str(i)+".jpg"
                else:
                    self.meta_dict["input"]=out_folder+"/"+str(j)+"_"+str(i)+".png"
                plus_meta(self.meta_dict,image)
                if url!="":
                    to_discord(self.meta_dict["input"],url)
            images.append(image)
            del image
            torch.cuda.empty_cache()
        clear_output(True)
        print(memo)
        imgshow(imgs=images)
        gc.collect()
        return images

    def image2imageup(self,prompt,n_prompt,gs,step,cs,seed,pag,x,y,ss,images,out_folder="",j_or_p="",url="",xf=False):
        if self.pipe==None:
            print("You must make a pipeline.")
            return []
        if self.is_sdxl:
            self.pipe=StableDiffusionXLPAGImg2ImgPipeline.from_pipe(self.pipe,torch_dtype=torch.float16)
        else:
            self.pipe=StableDiffusionPAGImg2ImgPipeline.from_pipe(self.pipe,torch_dtype=torch.float16)
        self.pipe.to("cuda")
        prompt=prompt+self.prompt_a
        n_prompt=n_prompt+self.n_prompt_a
        self.meta_dict["pr"]=prompt
        self.meta_dict["ne"]=n_prompt
        memo="seed\n"
        for i in seed:
            memo=memo+str(i)+"\n"
        memo=memo+"ckpt : "+self.meta_dict["ckpt_name"]+"\n"
        if self.meta_dict["vae_name"]!="":
            memo=memo+"vae : "+self.meta_dict["vae_name"]+"\n"
        memo=memo+"scheduler : "+self.meta_dict["sa"]+"\n"
        if self.meta_dict["loras"]!=[]:
            memo=memo+"lora : weight\n"
            for i in range(len(self.meta_dict["loras"])):
                memo=memo+self.meta_dict["loras"][i]+" : "+str(self.meta_dict["lora_weights"][i])+"\n"
        if self.meta_dict["pos"]!=[]:
            memo=memo+"Positive Embedding\n"
            for i in range(len(self.meta_dict["pos"])):
                memo=memo+self.meta_dict["pos"][i]+"\n"
        if self.meta_dict["neg"]!=[]:
            memo=memo+"Negative Embedding\n"
            for i in range(len(self.meta_dict["neg"])):
                memo=memo+self.meta_dict["neg"][i]+"\n"
        if "st" in self.meta_dict:
            memo=memo+"num_inference_steps : "+self.meta_dict["st"]+"\n"
        else:
            self.meta_dict["st"]=str(step)
            memo=memo+"num_inference_steps : "+self.meta_dict["st"]+"\n"
        memo=memo+"guidance_scale : "+str(gs)+"\n"
        self.meta_dict["cf"]=str(gs)
        memo=memo+"clip_skip : "+str(cs)+"\n"
        self.meta_dict["cl"]=str(cs)
        memo=memo+"pag_scale : "+str(pag)+"\n"
        self.meta_dict["pag"]=str(pag)
        memo=memo+"prompt\n"+prompt+"\nnegative prompt\n"+n_prompt+"\n"
        for k in ["hs","ds","hu","hum","up","ccs","cont","tu","tum"]:
            if k in self.meta_dict:
                del self.meta_dict[k]

        self.pipe.vae.enable_tiling()
        if xf:
            self.pipe.enable_xformers_memory_efficient_attention()
        
        if self.prompts==None:
            self.mkprompt(prompt=prompt,n_prompt=n_prompt)
        if self.prompt!=prompt or self.n_prompt!=n_prompt:
            self.mkprompt(prompt=prompt,n_prompt=n_prompt)

        if out_folder!="":
            if not(os.path.exists(out_folder)):
                os.makedirs(out_folder)
        j=0
        for i in seed:
            j=j+1
            if x!=images[j-1].width:
                if self.upscaler==None:
                    print("You must make a upscaler.")
                    return []
                memo1=memo+"Hires steps : "+str(step)+"\n"
                self.meta_dict["hs"]=str(step)
                memo1=memo1+"Denoising strength : "+str(ss)+"\n"
                self.meta_dict["ds"]=str(ss)
                memo1=memo1+"Hires upscale : "+str(x/images[j-1].width)+"\n"
                self.meta_dict["hu"]=str(x/images[j-1].width)
                self.meta_dict["hum"],self.meta_dict["up"]=self.upscaler.get_method()
                memo1=memo1+"Hires upscaler : "+self.meta_dict["hum"]+"\n"
                images[j-1]=self.upscaler.run(images[j-1],x,y)
            else:
                memo1=memo+"Denoising strength : "+str(ss)+"\n"
                self.meta_dict["ds"]=str(ss)
            clear_output(True)
            print(memo1)
            if j>1:
                imgshow(imgs=images)

            if self.is_sdxl:
                image = self.pipe(
                    eta=1.0,
                    prompt_embeds=self.prompts[0],
                    pooled_prompt_embeds=self.prompts[1],
                    negative_prompt_embeds=self.prompts[2],
                    negative_pooled_prompt_embeds=self.prompts[3],
                    image=images[j-1],
                    guidance_scale=gs,
                    num_inference_steps=int(step/ss)+1,
                    clip_skip=cs,
                    generator=torch.manual_seed(i),
                    strength=ss,
                    pag_scale=pag
                ).images[0]
            else:
                image = self.pipe(
                    eta=1.0,
                    prompt_embeds=self.prompts[0],
                    negative_prompt_embeds=self.prompts[1],
                    image=images[j-1],
                    guidance_scale=gs,
                    num_inference_steps=int(step/ss)+1,
                    clip_skip=cs,
                    generator=torch.manual_seed(i),
                    strength=ss,
                    pag_scale=pag
                ).images[0]
            if out_folder!="":
                self.meta_dict["se"]=str(i)
                if j_or_p=="j":
                    self.meta_dict["input"]=out_folder+"/"+str(j)+"_"+str(i)+".jpg"
                else:
                    self.meta_dict["input"]=out_folder+"/"+str(j)+"_"+str(i)+".png"
                plus_meta(self.meta_dict,image)
                if url!="":
                    to_discord(self.meta_dict["input"],url)
            images[j-1]=image
            del image
            torch.cuda.empty_cache()
        clear_output(True)
        print(memo1)
        imgshow(imgs=images)
        gc.collect()
        return images

    def tileup(self,prompt,n_prompt,gs,step,cs,seed,pag,x,y,ss,images,out_folder="",j_or_p="",url="",ccs=None,tile_size=(0,0),ol=0,xf=False):
        if self.pipe==None:
            print("You must make a pipeline.")
            return []
        try:
            if isinstance(tile_size,tuple):
                tile_size=(int(tile_size[0]),int(tile_size[1]))
                limit=512
            else:
                limit=round(tile_size/8)*8
                tile_size=(0,0)
                if limit<8:
                    limit=8
        except:
            tile_size=(0,0)
            limit=512

        try:
            ol=int(ol)
        except:
            ol=0

        if ccs==None:
            if self.is_sdxl:
                self.pipe=StableDiffusionXLPAGImg2ImgPipeline.from_pipe(self.pipe,torch_dtype=torch.float16)
            else:
                self.pipe=StableDiffusionPAGImg2ImgPipeline.from_pipe(self.pipe,torch_dtype=torch.float16)
            for k in ["cont","ccs"]:
                if k in self.meta_dict:
                    del self.meta_dict[k]
        else:
            if self.is_sdxl:
                controlnet = ControlNetModel.from_pretrained("OzzyGT/SDXL_Controlnet_Tile_Realistic",torch_dtype=torch.float16,variant="fp16")
                self.pipe=StableDiffusionXLControlNetPAGImg2ImgPipeline.from_pipe(self.pipe,torch_dtype=torch.float16,controlnet=controlnet)
                self.meta_dict["cont"]=str(370104)
            else:
                controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile',torch_dtype=torch.float16)
                self.pipe=StableDiffusionControlNetPAGInpaintPipeline.from_pipe(self.pipe,torch_dtype=torch.float16,controlnet=controlnet)
                self.meta_dict["cont"]=str(67566)
            self.meta_dict["ccs"]=str(ccs)
        self.pipe.to("cuda")
        prompt=prompt+self.prompt_a
        n_prompt=n_prompt+self.n_prompt_a
        self.meta_dict["pr"]=prompt
        self.meta_dict["ne"]=n_prompt
        memo="seed\n"
        for i in seed:
            memo=memo+str(i)+"\n"
        memo=memo+"ckpt : "+self.meta_dict["ckpt_name"]+"\n"
        if self.meta_dict["vae_name"]!="":
            memo=memo+"vae : "+self.meta_dict["vae_name"]+"\n"
        memo=memo+"scheduler : "+self.meta_dict["sa"]+"\n"
        if self.meta_dict["loras"]!=[]:
            memo=memo+"lora : weight\n"
            for i in range(len(self.meta_dict["loras"])):
                memo=memo+self.meta_dict["loras"][i]+" : "+str(self.meta_dict["lora_weights"][i])+"\n"
        if self.meta_dict["pos"]!=[]:
            memo=memo+"Positive Embedding\n"
            for i in range(len(self.meta_dict["pos"])):
                memo=memo+self.meta_dict["pos"][i]+"\n"
        if self.meta_dict["neg"]!=[]:
            memo=memo+"Negative Embedding\n"
            for i in range(len(self.meta_dict["neg"])):
                memo=memo+self.meta_dict["neg"][i]+"\n"

        self.meta_dict["st"]=str(step)
        memo=memo+"num_inference_steps : "+self.meta_dict["st"]+"\n"
        memo=memo+"guidance_scale : "+str(gs)+"\n"
        self.meta_dict["cf"]=str(gs)
        memo=memo+"clip_skip : "+str(cs)+"\n"
        self.meta_dict["cl"]=str(cs)
        memo=memo+"pag_scale : "+str(pag)+"\n"
        self.meta_dict["pag"]=str(pag)
        memo=memo+"prompt\n"+prompt+"\nnegative prompt\n"+n_prompt+"\n"
        for k in ["hs","ds","hu","hum","up","tu","tum"]:
            if k in self.meta_dict:
                del self.meta_dict[k]

        self.pipe.vae.enable_tiling()
        if xf:
            self.pipe.enable_xformers_memory_efficient_attention()
        
        if self.prompts==None:
            self.mkprompt(prompt=prompt,n_prompt=n_prompt)
        if self.prompt!=prompt or self.n_prompt!=n_prompt:
            self.mkprompt(prompt=prompt,n_prompt=n_prompt)

        if out_folder!="":
            if not(os.path.exists(out_folder)):
                os.makedirs(out_folder)

        if ccs==None:
            x=round(x/8)*8
            y=round(y/8)*8
        else:
            x=round(x/64)*64
            y=round(y/64)*64

        j=0
        for i in seed:
            j=j+1
            if self.upscaler==None:
                print("You must make a upscaler.")
                return []
            memo1=memo+"Denoising strength : "+str(ss)+"\n"
            self.meta_dict["ds"]=str(ss)
            memo1=memo1+"Tile upscale : "+str(x/images[j-1].width)+"\n"
            self.meta_dict["tu"]=str(x/images[j-1].width)
            self.meta_dict["tum"],self.meta_dict["up"]=self.upscaler.get_method()
            memo1=memo1+"Tile upscaler : "+self.meta_dict["tum"]+"\n"
            if ccs!=None:
                memo1=memo1+"controlnet_conditioning_scale : "+self.meta_dict["ccs"]+"\n"
            images[j-1]=self.upscaler.run(images[j-1],x,y)
            clear_output(True)
            print(memo1)
            if j>1:
                imgshow(imgs=images)

            if tile_size[0]>=8 and tile_size[1]>=8:
                tile_w=round(tile_size[0]/8)*8
                tile_h=round(tile_size[1]/8)*8
            else:
                aspect_ratio = x/y
                if aspect_ratio>1:
                    tile_w = min(x, 2*limit)
                    tile_h = min(round(tile_w /aspect_ratio/8)*8, 2*limit)
                else:
                    tile_h = min(y, 2*limit)
                    tile_w = min(round(tile_h*aspect_ratio/8)*8, 2*limit)
                tile_w = max(limit,tile_w)
                tile_h = max(limit,tile_h)

            if 1<=ol<tile_w and 1<=ol<tile_h:
                overlap=ol
            else:
                overlap = min( tile_w // 4, tile_h // 4)

            result = numpy.zeros((y, x, 3), dtype=numpy.float32)
            weight_sum = numpy.zeros((y, x, 1), dtype=numpy.float32)
            gaussian_weight = create_gaussian_weight(tile_w,tile_h,0.3)

            bottom=overlap
            while bottom<y:
                right=overlap
                top=bottom-overlap
                bottom=min(top+tile_h,y)
                while right<x:
                    left=right-overlap
                    right=min(left+tile_w,x)
                    current_tile_size = (right - left,bottom - top)

                    tile = images[j-1].crop((left, top, right, bottom))
                    if ccs==None:
                        if self.is_sdxl:
                            result_tile = self.pipe(
                                eta=1.0,
                                prompt_embeds=self.prompts[0],
                                pooled_prompt_embeds=self.prompts[1],
                                negative_prompt_embeds=self.prompts[2],
                                negative_pooled_prompt_embeds=self.prompts[3],
                                image=tile,
                                guidance_scale=gs,
                                generator=torch.manual_seed(i),
                                num_inference_steps=int(step/ss)+1,
                                clip_skip=cs,
                                strength=ss,
                                pag_scale=pag
                            ).images[0]
                        else:
                            result_tile = self.pipe(
                                eta=1.0,
                                prompt_embeds=self.prompts[0],
                                negative_prompt_embeds=self.prompts[1],
                                image=tile,
                                guidance_scale=gs,
                                generator=torch.manual_seed(i),
                                num_inference_steps=int(step/ss)+1,
                                clip_skip=cs,
                                strength=ss,
                                pag_scale=pag
                            ).images[0]
                    else:
                        if self.is_sdxl:
                            result_tile = self.pipe(
                                eta=1.0,
                                prompt_embeds=self.prompts[0],
                                pooled_prompt_embeds=self.prompts[1],
                                negative_prompt_embeds=self.prompts[2],
                                negative_pooled_prompt_embeds=self.prompts[3],
                                image=tile,
                                control_image=tile,
                                guidance_scale=gs,
                                generator=torch.manual_seed(i),
                                num_inference_steps=int(step/ss)+1,
                                clip_skip=cs,
                                strength=ss,
                                controlnet_conditioning_scale=ccs,
                                pag_scale=pag
                            ).images[0]
                        else:
                            result_tile = self.pipe(
                                eta=1.0,
                                prompt_embeds=self.prompts[0],
                                negative_prompt_embeds=self.prompts[1],
                                image=tile,
                                control_image=tile,
                                guidance_scale=gs,
                                generator=torch.manual_seed(i),
                                num_inference_steps=int(step/ss)+1,
                                clip_skip=cs,
                                strength=ss,
                                controlnet_conditioning_scale=ccs,
                                pag_scale=pag,
                                mask_image=Image.new("RGB", current_tile_size, (255,255,255))
                            ).images[0]

                    if current_tile_size!=(result_tile.width,result_tile.height):
                        result_tile = result_tile.resize( current_tile_size)

                    if current_tile_size != (tile_w, tile_h):
                        tile_weight = cv2.resize(gaussian_weight,current_tile_size)
                    else:
                        tile_weight = gaussian_weight[:current_tile_size[1], :current_tile_size[0]]

                    numpy_result_tile = numpy.array(result_tile)
                    result[top:bottom,left:right]=result[top:bottom,left:right]+numpy_result_tile*tile_weight[:,:,numpy.newaxis]
                    weight_sum[top:bottom,left:right]=weight_sum[top:bottom,left:right]+tile_weight[:,:,numpy.newaxis]
                    del tile_weight,result_tile,tile,numpy_result_tile
                    torch.cuda.empty_cache()
            final_result = (result / weight_sum).astype(numpy.uint8)
            image = Image.fromarray(final_result)
            images[j-1]=image

            if out_folder!="":
                self.meta_dict["se"]=str(i)
                if j_or_p=="j":
                    self.meta_dict["input"]=out_folder+"/"+str(j)+"_"+str(i)+".jpg"
                else:
                    self.meta_dict["input"]=out_folder+"/"+str(j)+"_"+str(i)+".png"
                plus_meta(self.meta_dict,image)
                if url!="":
                    to_discord(self.meta_dict["input"],url)
            del image,final_result,result,weight_sum
        clear_output(True)
        print(memo1)
        imgshow(imgs=images)    
        gc.collect()
        return images

    def deldiffusionparams(self):
        self.prompts=None
        del_keys=["tum","se","input","ds","tu","css","pr","ne","st","cf","cl","pag","hs","hu","cont","hum","up"]
        for k in del_keys:
            if k in self.meta_dict:
                del self.meta_dict[k]
        gc.collect()
