from diffusers import StableDiffusionXLPAGImg2ImgPipeline,AutoencoderKL
from diffusers import StableDiffusionPAGImg2ImgPipeline
from safetensors.torch import load_file
import safetensors,torch,random,os,shutil,json,pyexiv2,ast,gc,sys,numpy,cv2
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
from py_real_esrgan.model import RealESRGAN

sgm_use=[
    "Euler","Euler a","DPM++ 2M","DPM++ 2M SDE","DPM++ SDE","DPM++","DPM2","DPM2 a","Heun","LMS","UniPC","DPM++ 3M SDE"
]

class imgup:
    def __init__(self,path):
        device = torch.device('cuda')
        self.model = RealESRGAN(device, scale=4)
        self.model.load_weights(path,download=False)
    def run(self,img,x,y):
        while True:
            input_image = img.convert('RGB')
            input_image = self.model.predict(input_image)
            if input_image.width>=x and input_image.height>=y:
                if input_image.width==x and input_image.height==y:
                    image0=input_image
                else:
                    image0=input_image.resize((x,y))
                break
        return image0

def create_gaussian_weight(w,h, sigma=0.3):
    x = numpy.linspace(-1, 1, w)
    y = numpy.linspace(-1, 1, h)
    xx, yy = numpy.meshgrid(x, y)
    gaussian_weight = numpy.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return gaussian_weight

def plus_meta(vs,img):
    try:
        metadata=vs["pr"]+"\n\n"
        metadata=metadata+"Negative prompt: "+vs["ne"]+"\n\n"
        metadata=metadata+"Steps: "+vs["st"]+", "
        metadata=metadata+"Sampler: "+vs["sa"]+", "
        metadata=metadata+"CFG scale: "+vs["cf"]+", "
        metadata=metadata+"Seed: "+vs["se"]+", "
        metadata=metadata+"Clip skip: "+vs["cl"]+", "
        metadata=metadata+"PAG scale: "+vs["pag"]+", "
        metadata=metadata+"Denoising strength: "+vs["ds"]+", "
        metadata=metadata+"Tile upscale: "+vs["hu"]+", "
        metadata=metadata+"Tile upscaler: "+vs["hum"]+", "
        metadata=metadata+'Civitai resources: ['
        if vs["ckpt"]!="":
            metadata=metadata+'{"type":"checkpoint","modelVersionId":'+vs["ckpt"]+"}"

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
            metadata=metadata+',{"type":"ae","modelVersionId":'+vs["vae"]+"}"
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

def tile_up(
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
    url="",
    out_folder="output",
    j_or_p="j"
    ):
    prompt="masterpiece,best quality,ultra detailed"
    n_prompt="worst quality,low quality,normal quality"
    memo="seed\n"
    out_sizes=[]
    images=[]
    img_paths=[]
    if not(isinstance(seed, list)) or seed==[]:
        seed=[0]
    if img_path=="":
        print("Please select a image file.")
        return []
    else:
        for i in range(len(seed)):
            try:
                if img_path.startswith("https") or img_path.startswith("http"):
                    p=io.BytesIO(requests.get(img_path).content)
                else:
                    p=img_path
                img=Image.open(p)
                images.append(img)
                out_sizes.append([round(up*img.width/8)*8,round(up*img.height/8)*8])
                try:
                    seed[i]=int(seed[i])
                    if seed[i]==0:
                        seed[i]=random.randint(1, sys.maxsize)
                except:
                    seed[i]=random.randint(1, sys.maxsize)
                if j_or_p=="j":
                    p=out_folder+"/"+str(i)+"_"+str(seed[i])+".jpg"
                else:
                    p=out_folder+"/"+str(i)+"_"+str(seed[i])+".png"
                img_paths.append(p)
                memo=memo+str(seed[i])+"\n"
            except:
                memo=memo+img_path+" : ng\n"
                clear_output(True)
                print(memo)
                return []
    del img
    clear_output(True)
    print(memo)

    meta_dict={}
    meta_dict["hu"]=str(up)
        
    if not(os.path.isfile(base_safe)):
        print("the checkpoint file does not exist.")
        return []
    memo=memo+"checkpoint : "+base_safe+"\n"
    try:
        f=safetensors.safe_open(base_safe, framework="pt", device="cpu")
        meta_dict["ckpt"]=f.metadata()["id"]
        del f
    except:
        meta_dict["ckpt"]=""
    clear_output(True)
    print(memo)
    
    if os.path.exists(vae_safe):
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
                    d=f.metadata()
                    meta_id=d["id"]
                    if "," in meta_id:
                        meta_id = meta_id.split(",")
                        for j in meta_id:
                            meta_id_list.append(int(j))
                    else:
                        meta_id_list.append(int(meta_id))
                    meta_weight=d["weight"]
                    if "," in meta_weight:
                        meta_weight = meta_weight.split(",")
                        for j in meta_weight:
                            meta_weight_list.append(float(j)*lora_weights[i])
                    else:
                        meta_weight_list.append(float(meta_weight)*lora_weights[i])
                    del f,d,meta_id,meta_weight
                except:
                    meta_id_list=list1
                    meta_weight_list=list2
                del list1,list2
            else:
                memo=memo+line+".safetensors : "+str(lora_weights[i])+" ng"
                clear_output(True)
                print(memo)
                return []
            i=i+1

        if len(meta_weight_list)!=len(meta_id_list):
            meta_id_list=[]
            meta_weight_list=[]
        meta_dict["lora"]=str(meta_id_list)
        meta_dict["w"]=str(meta_weight_list)
        del meta_id_list,meta_weight_list
    else:
        meta_dict["lora"]="[]"
    
    meta_embed_list=[]
    memo=memo+"Positive Embedding\n"
    if pos_emb==[]:
        memo=memo+"nothing\n"
    else:
        for line in pos_emb:
            if os.path.isfile(line):
                memo=memo+line+" ok\n"
                list1=meta_embed_list
                try:
                    f=safetensors.safe_open(line, framework="pt", device="cpu")
                    meta_embed_list.append(f.metadata()["id"])
                    del f
                except:
                    meta_embed_list=list1
                del list1
            else:
                memo=memo+line+" ng"
                clear_output(True)
                print(memo)
                return []
    clear_output(True)
    print(memo)
                
    memo=memo+"Negative Embedding\n"
    if neg_emb==[]:
        memo=memo+"nothing\n"
    else:
        for line in neg_emb:
            if os.path.isfile(line):
                memo=memo+line+" ok\n"
                list1=meta_embed_list
                try:
                    f=safetensors.safe_open(line, framework="pt", device="cpu")
                    meta_embed_list.append(f.metadata()["id"])
                    del f
                except:
                    meta_embed_list=list1
                del list1
            else:
                memo=memo+line+" ng"
                clear_output(True)
                print(memo)
                return []
    clear_output(True)
    print(memo)
    meta_dict["embed"]=str(meta_embed_list)
    del meta_embed_list
 
    p="esrgan"
    if isinstance(Interpolation, int):
        if Interpolation==1:
            p=Image.NEAREST
            meta_dict["hum"]="NEAREST"
        elif Interpolation==2:
            p=Image.BOX
            meta_dict["hum"]="BOX"
        elif Interpolation==3:
            p=Image.BILINEAR
            meta_dict["hum"]="BILINEAR"
        elif Interpolation==4:
            p=Image.HAMMING
            meta_dict["hum"]="HAMMING"
        elif Interpolation==5:
            p=Image.BICUBIC
            meta_dict["hum"]="BICUBIC"
        else:
            p=Image.LANCZOS
            meta_dict["hum"]="LANCZOS"

    sd=load_file(base_safe)
    is_sdxl="conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in sd
    del sd
    gc.collect()

    dtype=torch.float16
    if is_sdxl:
        if os.path.isfile(vae_safe):
            pipe = StableDiffusionXLPAGImg2ImgPipeline.from_single_file(base_safe, torch_dtype=dtype)
            pipe.vae=AutoencoderKL.from_single_file(vae_safe,torch_dtype=dtype)
        else:
            pipe = StableDiffusionXLPAGImg2ImgPipeline.from_single_file(base_safe, torch_dtype=dtype)
    else:
        if os.path.isfile(vae_safe):
            pipe = StableDiffusionPAGImg2ImgPipeline.from_single_file(base_safe, torch_dtype=dtype)
            pipe.vae=AutoencoderKL.from_single_file(vae_safe,torch_dtype=dtype)
        else:
            pipe = StableDiffusionPAGImg2ImgPipeline.from_single_file(base_safe, torch_dtype=dtype)
    pipe.to("cuda")

    sgm_dict={}
    sgm_dict["use_karras_sigmas"]=False
    sgm_dict["use_exponential_sigmas"]=False
    sgm_dict["use_beta_sigmas"]=False
    if sample in sgm_use:
        if sgm=="Karras":
            sgm_dict["timestep_spacing"]="linspace"
            sgm_dict["use_karras_sigmas"]=True
        elif sgm=="exponential":
            sgm_dict["timestep_spacing"]="linspace"
            sgm_dict["use_exponential_sigmas"]=True
        elif sgm=="beta":
            sgm_dict["timestep_spacing"]="linspace"
            sgm_dict["use_beta_sigmas"]=True
        elif sgm=="sgm_uniform" or sgm=="simple":
            sgm_dict["timestep_spacing"]="trailing"
        else:
            sgm_dict["timestep_spacing"]="linspace"
    else:
        if sgm=="sgm_uniform" or sgm=="simple":
            sgm_dict["timestep_spacing"]="trailing"
        else:
            sgm_dict["timestep_spacing"]="leading"

    if sample=="Euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config,
        timestep_spacing=sgm_dict["timestep_spacing"],
        use_karras_sigmas=sgm_dict["use_karras_sigmas"],
        use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
        use_beta_sigmas=sgm_dict["use_beta_sigmas"]
        )
        memo=memo+"scheduler : Euler"
        meta_dict["sa"]=sample
    elif sample=="Euler a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config,
        timestep_spacing=sgm_dict["timestep_spacing"],
        use_karras_sigmas=sgm_dict["use_karras_sigmas"],
        use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
        use_beta_sigmas=sgm_dict["use_beta_sigmas"]
        )
        memo=memo+"scheduler : Euler a"
        meta_dict["sa"]=sample
    elif sample=="LMS":
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config,
        timestep_spacing=sgm_dict["timestep_spacing"],
        use_karras_sigmas=sgm_dict["use_karras_sigmas"],
        use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
        use_beta_sigmas=sgm_dict["use_beta_sigmas"]
        )
        memo=memo+"scheduler : LMS"
        meta_dict["sa"]=sample
    elif sample=="Heun":
        pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config,
        timestep_spacing=sgm_dict["timestep_spacing"],
        use_karras_sigmas=sgm_dict["use_karras_sigmas"],
        use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
        use_beta_sigmas=sgm_dict["use_beta_sigmas"]
        )
        memo=memo+"scheduler : Heun"
        meta_dict["sa"]=sample
    elif sample=="DPM2":
        pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config,
        timestep_spacing=sgm_dict["timestep_spacing"],
        use_karras_sigmas=sgm_dict["use_karras_sigmas"],
        use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
        use_beta_sigmas=sgm_dict["use_beta_sigmas"]
        )
        memo=memo+"scheduler : DPM2"
        meta_dict["sa"]=sample
    elif sample=="DPM2 a":
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config,
        timestep_spacing=sgm_dict["timestep_spacing"],
        use_karras_sigmas=sgm_dict["use_karras_sigmas"],
        use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
        use_beta_sigmas=sgm_dict["use_beta_sigmas"]
        )
        memo=memo+"scheduler : DPM2 a"
        meta_dict["sa"]=sample
    elif sample=="DPM++ 2M":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,
        timestep_spacing=sgm_dict["timestep_spacing"],
        use_karras_sigmas=sgm_dict["use_karras_sigmas"],
        use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
        use_beta_sigmas=sgm_dict["use_beta_sigmas"]
        )
        memo=memo+"scheduler : DPM++ 2M"
        meta_dict["sa"]=sample
    elif sample=="DPM++ SDE":
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config,
        algorithm_type="sde-dpmsolver++",
        timestep_spacing=sgm_dict["timestep_spacing"],
        use_karras_sigmas=sgm_dict["use_karras_sigmas"],
        use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
        use_beta_sigmas=sgm_dict["use_beta_sigmas"]
        )
        memo=memo+"scheduler : DPM++ SDE"
        meta_dict["sa"]=sample
    elif sample=="DPM++":
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config,
        timestep_spacing=sgm_dict["timestep_spacing"],
        use_karras_sigmas=sgm_dict["use_karras_sigmas"],
        use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
        use_beta_sigmas=sgm_dict["use_beta_sigmas"]
        )
        memo=memo+"scheduler : DPM++"
        meta_dict["sa"]=sample
    elif sample=="DPM++ 2M SDE":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,
        algorithm_type="sde-dpmsolver++",
        timestep_spacing=sgm_dict["timestep_spacing"],
        use_karras_sigmas=sgm_dict["use_karras_sigmas"],
        use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
        use_beta_sigmas=sgm_dict["use_beta_sigmas"]
        )
        memo=memo+"scheduler : DPM++ 2M SDE"
        meta_dict["sa"]=sample
    elif sample=="PLMS":
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config,timestep_spacing=sgm_dict["timestep_spacing"])
        memo=memo+"scheduler : PLMS"
        meta_dict["sa"]=sample
    elif sample=="UniPC":
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config,
        timestep_spacing=sgm_dict["timestep_spacing"],
        use_karras_sigmas=sgm_dict["use_karras_sigmas"],
        use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
        use_beta_sigmas=sgm_dict["use_beta_sigmas"]
        )
        memo=memo+"scheduler : UniPC"
        meta_dict["sa"]=sample
    elif sample=="LCM":
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config,timestep_spacing=sgm_dict["timestep_spacing"])
        memo=memo+"scheduler : LCM"
        meta_dict["sa"]=sample
    elif sample=="DPM++ 3M SDE":
        pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config,
        timestep_spacing=sgm_dict["timestep_spacing"],
        use_karras_sigmas=sgm_dict["use_karras_sigmas"],
        use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
        use_beta_sigmas=sgm_dict["use_beta_sigmas"]
        )
        memo=memo+"scheduler : DPM++ 3M SDE"
        meta_dict["sa"]=sample
    else:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config,timestep_spacing=sgm_dict["timestep_spacing"])
        memo=memo+"scheduler : DDIM"
        meta_dict["sa"]="DDIM"

    if sample in sgm_use:
        if sgm=="Karras":
            memo=memo+" "+sgm
            meta_dict["sa"]=meta_dict["sa"]+" "+sgm
        elif sgm=="exponential":
            memo=memo+" "+sgm
            meta_dict["sa"]=meta_dict["sa"]+" "+sgm
        elif sgm=="beta":
            memo=memo+" "+sgm
            meta_dict["sa"]=meta_dict["sa"]+" "+sgm

    if sgm=="sgm_uniform" or sgm=="simple":
        memo=memo+" "+sgm
        meta_dict["sa"]=meta_dict["sa"]+" "+sgm

    memo=memo+"\n"
    
    memo=memo+"num_inference_steps : "+str(step)+"\n"
    meta_dict["st"]=str(step)
    memo=memo+"guidance_scale : "+str(gs)+"\n"
    meta_dict["cf"]=str(gs)
    memo=memo+"clip_skip : "+str(cs)+"\n"
    meta_dict["cl"]=str(cs)
    memo=memo+"pag_scale : "+str(pag)+"\n"
    meta_dict["pag"]=str(pag)
    memo=memo+"Denoising strength : "+str(ss)+"\n"
    meta_dict["ds"]=str(ss)
    if not(isinstance(Interpolation,int)):
        if os.path.exists(Interpolation):
            meta_dict["hum"]=os.path.basename(Interpolation)
        else:
            p=Image.BILINEAR
            meta_dict["hum"]="BILINEAR"
    memo=memo+"Tile upscaler : "+meta_dict["hum"]+"\n"
    memo=memo+"Tile upscale : "+meta_dict["hu"]+"\n"

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
            if is_sdxl:
                state_dict = load_file(line)
                pipe.load_textual_inversion(state_dict["clip_g"],token=key,text_encoder=pipe.text_encoder_2,tokenizer=pipe.tokenizer_2,torch_dtype=dtype)
                pipe.load_textual_inversion(state_dict["clip_l"],token=key,text_encoder=pipe.text_encoder,tokenizer=pipe.tokenizer,torch_dtype=dtype)
                del state_dict
            else:
                pipe.load_textual_inversion(".", weight_name=line, token=key)
            prompt = prompt+","+key
            memo=memo+line+" is loaded.\n"
            clear_output(True)
            print(memo)
            del key

    if neg_emb!=[]:
        for line in neg_emb:
            key=os.path.basename(line).replace(".safetensors","")
            if is_sdxl:
                state_dict = load_file(line)
                pipe.load_textual_inversion(state_dict["clip_g"],token=key,text_encoder=pipe.text_encoder_2,tokenizer=pipe.tokenizer_2,torch_dtype=dtype)
                pipe.load_textual_inversion(state_dict["clip_l"],token=key,text_encoder=pipe.text_encoder,tokenizer=pipe.tokenizer,torch_dtype=dtype)
                del state_dict
            else:
                pipe.load_textual_inversion(".", weight_name=line, token=key)
            n_prompt=n_prompt+","+key
            memo=memo+line+" is loaded.\n"
            clear_output(True)
            print(memo)
            del key
    
    memo=memo+"prompt\n"+prompt+"\n"
    memo=memo+"negative_prompt\n"+n_prompt+"\n"
    meta_dict["pr"]=prompt
    meta_dict["ne"]=n_prompt
    clear_output(True)
    print(memo)

    pipe.vae.enable_tiling()
    pipe.enable_xformers_memory_efficient_attention()
    
    if is_sdxl:
        comple = CompelForSDXL(pipe)
    else:
        comple = CompelForSD(pipe)
    conditioning = comple(prompt, negative_prompt=n_prompt)
    if is_sdxl:
        prompts=[conditioning.embeds,conditioning.pooled_embeds,conditioning.negative_embeds,conditioning.negative_pooled_embeds]
    else:
        prompts=[conditioning.embeds,conditioning.negative_embeds]
    del comple,conditioning
    gc.collect()

    pipe.text_encoder=None
    pipe.tokenizer=None
    if is_sdxl:
        pipe.text_encoder_2=None
        pipe.tokenizer_2=None

    if p=="esrgan":
        uppipe=imgup(Interpolation)
    del Interpolation

    if not(os.path.exists(out_folder)):
        os.mkdir(out_folder)

    for i in range(len(seed)):
        if p=="esrgan":
            images[i]=uppipe.run(images[i],out_sizes[i][0], out_sizes[i][1])
        else:
            images[i]=images[i].resize((out_sizes[i][0], out_sizes[i][1]), resample=p)

        aspect_ratio = out_sizes[i][0]/out_sizes[i][1]
        if aspect_ratio>1:
            tile_w = min(out_sizes[i][0], 1024)
            tile_h = min(round(tile_w /aspect_ratio/8)*8, 1024)
        else:
            tile_h = min(out_sizes[i][1], 1024)
            tile_w = min(round(tile_h*aspect_ratio/8)*8, 1024)
        tile_w = max(512,tile_w)
        tile_h = max(512,tile_h)
        overlap = min( tile_w // 4, tile_h // 4)

        result = numpy.zeros((out_sizes[i][1], out_sizes[i][0], 3), dtype=numpy.float32)
        weight_sum = numpy.zeros((out_sizes[i][1], out_sizes[i][0], 1), dtype=numpy.float32)

        gaussian_weight = create_gaussian_weight(tile_w,tile_h,0.3)
        bottom=overlap
        while bottom<out_sizes[i][1]:
            right=overlap
            top=bottom-overlap
            bottom=min(top+tile_h,out_sizes[i][1])
            while right<out_sizes[i][0]:
                left=right-overlap
                right=min(left+tile_w,out_sizes[i][0])
                current_tile_size = (right - left,bottom - top)

                tile = images[i].crop((left, top, right, bottom))

                if is_sdxl:
                    result_tile = pipe(
                        eta=1.0,
                        prompt_embeds=prompts[0],
                        pooled_prompt_embeds=prompts[1],
                        negative_prompt_embeds=prompts[2],
                        negative_pooled_prompt_embeds=prompts[3],
                        image=tile,
                        guidance_scale=gs,
                        generator=torch.manual_seed(seed[i]),
                        num_inference_steps=int(step/ss)+1,
                        clip_skip=cs,
                        strength=ss,
                        pag_scale=pag
                    ).images[0]
                else:
                    result_tile = pipe(
                        eta=1.0,
                        prompt_embeds=prompts[0],
                        negative_prompt_embeds=prompts[1],
                        image=tile,
                        guidance_scale=gs,
                        generator=torch.manual_seed(seed[i]),
                        num_inference_steps=int(step/ss)+1,
                        clip_skip=cs,
                        strength=ss,
                        pag_scale=pag
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
        final_result = (result / weight_sum).astype(numpy.uint8)
        pil_image = Image.fromarray(final_result)

        meta_dict["se"]=str(seed[i])
        meta_dict["input"]=img_paths[i]
        plus_meta(meta_dict,pil_image)
        del pil_image,final_result,result,weight_sum
        gc.collect()
        clear_output(True)
        print(memo)
        display(image)
    del images,pipe
    if p=="esrgan":
        del uppipe
    gc.collect()
    return seed
