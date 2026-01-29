from diffusers import AutoencoderKL,ControlNetModel
from diffusers import StableDiffusionXLPAGPipeline,StableDiffusionXLPAGImg2ImgPipeline,AutoencoderKL
from diffusers import StableDiffusionPAGPipeline,StableDiffusionPAGImg2ImgPipeline
from diffusers import StableDiffusionXLControlNetPAGImg2ImgPipeline,StableDiffusionControlNetPAGInpaintPipeline
from safetensors.torch import load_file
import safetensors,torch,random,os,shutil,json,pyexiv2,math,ast,requests,gc,sys,numpy,io,cv2
from PIL import Image,PngImagePlugin
from IPython.display import clear_output,display
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
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer

sgm_use=[
    "Euler","Euler a","DPM++ 2M","DPM++ 2M SDE","DPM++ SDE","DPM++","DPM2","DPM2 a","Heun","LMS","UniPC","DPM++ 3M SDE"
]

class imgup:
    def __init__(self,path,scale=4):
        if not(isinstance(path,int)):
            if not(os.path.exists(path)):
                path=os.getcwd()+'/upscaler/RealESRGAN_x4plus.pth'
                if not(os.path.dirname(path)):
                    os.mkdir(os.path.dirname(path))
                url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
                res = requests.get(url,stream=True)
                f=open(path, 'wb')
                for chunk in res.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
                f.close()
                del res

            sd=torch.load(path)
            if "id" in sd:
                self.id=str(sd["id"].item())
            else:
                self.id=str(164898)
            if "params" in sd:
                sd=sd["params"]
            elif "params_ema" in sd:
                sd=sd["params_ema"]
            else:
                self.model,self.path=self.interpolation(6)
                return None
            if "conv_first.weight" in sd:
                nf=sd["conv_first.weight"].size()[0]
                for i in range(1000):
                    if not("body."+str(i)+".rdb1.conv1.weight" in sd):
                        break
                nb=i
                net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=nf, num_block=nb, num_grow_ch=32, scale=scale)
            elif "body.0.weight" in sd:
                nf=sd["body.0.weight"].size()[0]
                for i in range(1000):
                    if not("body."+str(i*2+4)+".weight" in sd):
                        break
                nc=i
                net=SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=nf, num_conv=nc, upscale=scale, act_type='prelu')
            else:
                self.model,self.path=self.interpolation(6)
                return None

            self.model = RealESRGANer(
                scale=scale,
                model_path=path,
                dni_weight=None,
                model=net,
                tile=256,
                tile_pad=10,
                pre_pad=0,
                half=True,
                device="cuda"
            )
            self.path=os.path.basename(path)
        else:
            self.model,self.path=self.interpolation(path)
    
    def interpolation(self,path):
        if path==1:
            model=Image.NEAREST
            path="NEAREST"
        elif path==2:
            model=Image.BOX
            path="BOX"
        elif path==6:
            model=Image.LANCZOS
            path="LANCZOS"
        elif path==4:
            model=Image.HAMMING
            path="HAMMING"
        elif path==5:
            model=Image.BICUBIC
            path="BICUBIC"
        else:
            model=Image.BILINEAR
            path="BILINEAR"
        self.id=""
        return model,path

    def get_method(self):
        return self.path,self.id

    def run(self,img,x,y):
        if not(self.path in ["NEAREST","BOX","LANCZOS","HAMMING","BICUBIC","BILINEAR"]):
            input_image = img.convert('RGB')
            input_image = numpy.array(input_image)
            while Image.fromarray(input_image).width<x or Image.fromarray(input_image).height<y:
                input_image,dummy = self.model.enhance(input_image)
                del dummy
            input_image=Image.fromarray(input_image)
            if input_image.width==x and input_image.height==y:
                image0=input_image
            else:
                image0=input_image.resize((x,y))
        else:
            image0=image.resize((x,y), resample=self.model)
        return image0
        
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
        self.meta_dict["hum"],self.meta_dict["id"]=self.upscaler.get_method()

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

    def text2image(self,prompt,n_prompt,gs,step,cs,seed,pag,x,y,out_folder="",j_or_p=""):
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

        self.pipe.vae.enable_tiling()
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
            for j2 in range(j-1):
                print(str(j2+1))
                display(Images[j2])
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
                for k in ["hs","ds","hu","hum","up"]:
                    if k in self.meta_dict:
                        del self.meta_dict[k]
                self.meta_dict["se"]=str(i)
                if j_or_p=="j":
                    self.meta_dict["input"]=out_folder+"/"+str(j)+"_"+str(i)+".jpg"
                else:
                    self.meta_dict["input"]=out_folder+"/"+str(j)+"_"+str(i)+".png"
                plus_meta(self.meta_dict,image)
            images.append(image)
            del image
            torch.cuda.empty_cache()
        clear_output(True)
        print(memo)
        for j2 in range(len(images)):
            print(str(j2+1))
            display(Images[j2])
        gc.collect()
        return images

    def image2image(self,prompt,n_prompt,gs,step,cs,seed,pag,x,y,ss,images,out_folder="",j_or_p=""):
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

        self.pipe.vae.enable_tiling()
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
            for j2 in range(j-1):
                print(str(j2+1))
                display(images[j2])
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
            images[j-1]=image
            del image
            torch.cuda.empty_cache()
        clear_output(True)
        print(memo1)
        for j2 in range(len(images)):
            print(str(j2+1))
            display(images[j2])
        gc.collect()
        return images

    def tile_up(self,prompt,n_prompt,gs,step,cs,seed,pag,x,y,ss,images,out_folder="",j_or_p="",ccs=None):
        if self.pipe==None:
            print("You must make a pipeline.")
            return []
        if ccs==None:
            if self.is_sdxl:
                self.pipe=StableDiffusionXLPAGImg2ImgPipeline.from_pipe(self.pipe,torch_dtype=torch.float16)
            else:
                self.pipe=StableDiffusionPAGImg2ImgPipeline.from_pipe(self.pipe,torch_dtype=torch.float16)
        else:
            if self.is_sdxl:
                controlnet = ControlNetModel.from_pretrained("OzzyGT/SDXL_Controlnet_Tile_Realistic",torch_dtype=torch.float16,variant="fp16")
                self.pipe=StableDiffusionXLControlNetPAGImg2ImgPipeline.from_pipe(self.pipe,torch_dtype=torch.float16,controlnet=controlnet)
                self.meta_dict["cont"]=str(370104)
            else:
                controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile',torch_dtype=torch.float16)
                self.pipe=StableDiffusionControlNetPAGInpaintPipeline.from_pipe(self.pipe,torch_dtype=torch.float16,controlnet=controlnet)
                self.meta_dict["cont"]=str(67566)
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

        self.pipe.vae.enable_tiling()
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
            if "hum" in self.meta_dict):
                del self.meta_dict["hum"]
            self.meta_dict["tum"],self.meta_dict["id"]=self.upscaler.get_method()
            memo1=memo1+"Tile upscaler : "+self.meta_dict["tum"]+"\n"
            if ccs!=None:
                self.meta_dict["ccs"]=str(ccs)
                memo1=memo1+"controlnet_conditioning_scale : "+self.meta_dict["ccs"]+"\n"
            images[j-1]=self.upscaler.run(images[j-1],x,y)
            clear_output(True)
            print(memo1)
            for j2 in range(j-1):
                print(str(j2+1))
                display(images[j2])

            aspect_ratio = x/y
            if aspect_ratio>1:
                tile_w = min(x, 1024)
                tile_h = min(round(tile_w /aspect_ratio/8)*8, 1024)
            else:
                tile_h = min(y, 1024)
                tile_w = min(round(tile_h*aspect_ratio/8)*8, 1024)
            tile_w = max(512,tile_w)
            tile_h = max(512,tile_h)
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
            del image,final_result,result,weight_sum    
        clear_output(True)
        print(memo1)
        for j2 in range(len(images)):
            print(str(j2+1))
            display(images[j2])      
        gc.collect()
        return images

    def deldiffusionparams(self):
        self.prompts=None
        del_keys=["tum","se","input","ds","tu","css","pr","ne","st","cf","cl","pag","hs","hu","cont","up","hum"]
        for k in del_keys:
            if k in self.meta_dict:
                del self.meta_dict[k]
        gc.collect()
    
def text2image(
    loras=[],
    lora_weights=[],
    prompt = "",
    n_prompt = "",
    t="v",
    prog_ver=2,
    pic_number=10,
    gs=7,
    f_step=10,
    step=30,
    ss=0.6,
    cs=1,
    Interpolation=3,
    sample="DDIM",
    sgm="",
    seed=0,
    out_folder="data",
    pos_emb=[],
    neg_emb=[],
    base_safe="base.safetensors",
    vae_safe="",
    pag=3.0,
    j_or_p="j",
    p=None
    ):
    memo="seed\n"
    if isinstance(seed, list):
        pic_number=len(seed)
        for i in range(pic_number):
            try:
                if int(seed[i])==0:
                    seed[i]=random.randint(1, sys.maxsize)
                else:
                    seed[i]=int(seed[i])
            except:
                seed[i]=random.randint(1, sys.maxsize)
            memo=memo+str(seed[i])+"\n"
    else:
        try:
            if int(seed)==0:
                seed=[]
                for i in range(pic_number):
                    seed.append(random.randint(1, sys.maxsize))
            else:
                seed=[int(seed)]
                pic_number=1
        except:
            seed=[]
            for i in range(pic_number):
                seed.append(random.randint(1, sys.maxsize))
        for i in range(pic_number):
            memo=memo+str(seed[i])+"\n"
    clear_output(True)
    print(memo)
            
    if prog_ver!=1:
        if prog_ver!=2:
            prog_ver=0
        
    if t=="v":
        tate=[1024,1600]
        yoko=[768,1200]
    elif t=="s":
        tate=[888,1384]
        yoko=[888,1384]
    elif t=="h":
        yoko=[1024,1600]
        tate=[768,1200]
    elif t=="vl":
        tate=[800,1600]
        yoko=[600,1200]
    elif t=="sl":
        tate=[696,1384]
        yoko=[696,1384]
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
            del iw,ow,ih,oh
        else:
            print("t setting is error.")
            print(" initial width, output width, initial height, output height")
            return []
        del t_list
    del t
    gc.collect()

    if p==None:
        pipe=mokupipe()
        check=pipe.mkpipe(
            pos_emb=pos_emb,
            neg_emb=neg_emb,
            base_safe=base_safe,
            vae_safe=vae_safe,
            loras=loras,
            lora_weights=lora_weights,
            sample=sample,
            sgm=sgm
        )

        if check==-1:
            return None
    else:
        pipe=p
        pipe.deldiffusionparams()

    if prog_ver==0:
        images=pipe.text2image(
            prompt=prompt,
            n_prompt=n_prompt,
            gs=gs,
            step=f_step,
            cs=cs,
            seed=seed,
            pag=pag,
            x=yoko[1],
            y=tate[1],
            out_folder=out_folder,
            j_or_p=j_or_p
        )
    else:
        images=pipe.text2image(
            prompt=prompt,
            n_prompt=n_prompt,
            gs=gs,
            step=f_step,
            cs=cs,
            seed=seed,
            pag=pag,
            x=yoko[0],
            y=tate[0],
            out_folder="",
            j_or_p=""
        )

    if prog_ver!=0:
        pipe.mkpipe_upscale(Interpolation)

    if prog_ver==1:
        images=pipe.image2image(
            prompt=prompt,
            n_prompt=n_prompt,
            gs=gs,
            step=step,
            cs=cs,
            seed=seed,
            pag=pag,
            x=yoko[1],
            y=tate[1],
            ss=ss,
            images=images,
            out_folder=out_folder,
            j_or_p=j_or_p
        )
    elif prog_ver==2:
        images=pipe.image2image(
            prompt=prompt,
            n_prompt=n_prompt,
            gs=gs,
            step=(step+f_step)/2,
            cs=cs,
            seed=seed,
            pag=pag,
            x=round((yoko[0]+yoko[1])/2/8)*8,
            y=round((tate[0]+tate[1])/2/8)*8,
            ss=ss,
            images=images,
            out_folder="",
            j_or_p=""
        )
    
    if prog_ver==2:
        images=pipe.image2image(
            prompt=prompt,
            n_prompt=n_prompt,
            gs=gs,
            step=step,
            cs=cs,
            seed=seed,
            pag=pag,
            x=yoko[1],
            y=tate[1],
            ss=ss,
            images=images,
            out_folder=out_folder,
            j_or_p=j_or_p
        )

    del images,seed
    return pipe

def mokuup(
    img_path="",
    base_safe="ckpt.safetensors",
    vae_safe="vae.safetensors",
    loras=[],
    lora_weights=[],
    up=2,
    gs=7,
    step=20,
    ss=0.5,
    cs=2,
    Interpolation=3,
    sample="DDIM",
    sgm="",
    seed=[],
    pos_emb=[],
    neg_emb=[],
    pag=3.0,
    out_folder="output",
    j_or_p="j",
    p=None,
    prompt="masterpiece,best quality,ultra detailed",
    n_prompt="worst quality,low quality,normal quality",
    ccs=None
    ):

    memo="seed\n"
    if isinstance(seed, list):
        pic_number=len(seed)
        for i in range(pic_number):
            try:
                if int(seed[i])==0:
                    seed[i]=random.randint(1, sys.maxsize)
                else:
                    seed[i]=int(seed[i])
            except:
                seed[i]=random.randint(1, sys.maxsize)
            memo=memo+str(seed[i])+"\n"
    else:
        try:
            if int(seed)==0:
                seed=[]
                for i in range(pic_number):
                    seed.append(random.randint(1, sys.maxsize))
            else:
                seed=[int(seed)]
                pic_number=1
        except:
            seed=[]
            for i in range(pic_number):
                seed.append(random.randint(1, sys.maxsize))
        for i in range(pic_number):
            memo=memo+str(seed[i])+"\n"
    clear_output(True)
    print(memo)

    if img_path=="":
        print("Please select a image file.")
        return None
    else:
        images=[]
        for i in range(len(seed)):
            try:
                if img_path.startswith("https") or img_path.startswith("http"):
                    path=io.BytesIO(requests.get(img_path).content)
                else:
                    path=img_path
                img=Image.open(path)
                images.append(img)
                output_size=(up*img.width,up*img.height)
                del img
            except:
                print("I can't read "+img_path+".")
                return None
    
    if p==None:
        pipe=mokupipe()
        check=pipe.mkpipe(
            pos_emb=pos_emb,
            neg_emb=neg_emb,
            base_safe=base_safe,
            vae_safe=vae_safe,
            loras=loras,
            lora_weights=lora_weights,
            sample=sample,
            sgm=sgm
        )

        if check==-1:
            return None
    else:
        pipe=p
        pipe.deldiffusionparams()

    pipe.mkpipe_upscale(Interpolation)
    images=pipe.tile_up(
        prompt=prompt,
        n_prompt=n_prompt,
        gs=gs,
        step=step,
        cs=cs,
        seed=seed,
        pag=pag,
        x=output_size[0],
        y=output_size[1],
        ss=ss,
        images=images,
        out_folder=out_folder,
        ccs=ccs
        )

    del images,seed
    return pipe
    





