from diffusers import StableDiffusionXLPAGPipeline,StableDiffusionXLPAGImg2ImgPipeline,AutoencoderKL
from diffusers import StableDiffusionPAGPipeline,StableDiffusionPAGImg2ImgPipeline
from safetensors.torch import load_file
import safetensors,torch,random,os,shutil,ast,pyexiv2,math
from PIL import Image,PngImagePlugin
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
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

sgm_use=[
    "Euler","Euler a","DPM++ 2M","DPM++ 2M SDE","DPM++ SDE","DPM++","DPM2","DPM2 a","Heun","LMS","UniPC","DPM++ 3M SDE"
]

def show_img(imgs):
    ncols=2
    nrows=math.ceil(len(imgs)/ncols)
    w, h = imgs[0].size
    if h>=w:
        sh=32
        sw=int(w*32/h)
    else:
        sw=32
        sh=int(h*32/w)
    fig = plt.figure(figsize=(sw,sh))
    grid = ImageGrid(fig,111,nrows_ncols=(nrows, ncols),axes_pad=0.5)
    for i in range(len(imgs)):
        grid[i].set_title(str(i),fontsize=15)
        grid[i].imshow(imgs[i])
    for ax in grid.axes_all:
        ax.axis('off')
    plt.show()

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
        if vs["ds"]!="":
            metadata=metadata+"Denoising strength: "+vs["ds"]+", "
        if vs["hu"]!="":
            metadata=metadata+"Hires upscale: "+vs["hu"]+", "
        if vs["hs"]!="":
            metadata=metadata+"Hires steps: "+vs["hs"]+", "
        if vs["hum"]!="":
            metadata=metadata+"Hires upscaler: "+vs["hum"]+", "
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
    vae_safe="vae.safetensors",
    pag=3.0,
    j_or_p="j"
    ):
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
    clear_output(True)
    print(memo)
            
    if prog_ver!=1:
        if prog_ver!=2:
            prog_ver=0
        
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
                    del f,d
                except:
                    meta_id_list=list1
                    meta_weight_list=list2
                
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
            else:
                memo=memo+line+" ng"
                clear_output(True)
                print(memo)
                return []
    clear_output(True)
    print(memo)
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
    clear_output(True)
    print(memo)

    del t,Interpolation

    dtype=torch.float16
    if os.path.isfile(vae_safe):
        pipe = StableDiffusionXLPAGPipeline.from_single_file(base_safe, torch_dtype=dtype)
        pipe.vae=AutoencoderKL.from_single_file(vae_safe,torch_dtype=dtype)
        pipe.to("cuda:0")
    else:
        pipe = StableDiffusionXLPAGPipeline.from_single_file(base_safe, torch_dtype=dtype).to("cuda:0")
        
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
    
    memo=memo+"num_inference_steps : "+str(f_step)+"\n"
    meta_dict["st"]=str(f_step)
    memo=memo+"guidance_scale : "+str(gs)+"\n"
    meta_dict["cf"]=str(gs)
    memo=memo+"clip_skip : "+str(cs)+"\n"
    meta_dict["cl"]=str(cs)
    memo=memo+"pag_scale : "+str(pag)+"\n"
    meta_dict["pag"]=str(pag)

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
                clip_skip=cs,generator=torch.manual_seed(seed[i]),
                pag_scale=pag
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
                generator=torch.manual_seed(seed[i]),
                pag_scale=pag
            ).images[0]
            meta_dict["se"]=str(seed[i])
            if j_or_p=="j":
                meta_dict["input"]=out_folder+"/"+str(i)+"_"+str(seed[i])+".jpg"
            else:
                meta_dict["input"]=out_folder+"/"+str(i)+"_"+str(seed[i])+".png"
            plus_meta(meta_dict,image)
            images.append(image)
            clear_output(True)
            print(memo)
            show_img(images[:i+1])
            del image
            torch.cuda.empty_cache()

    del pipe,conditioning

    if prog_ver==2 or prog_ver==1:
        if torch.cuda.device_count()==1:
            d="cuda:0"
        else:
            d="cuda:1"
        if os.path.isfile(vae_safe):
            pipe = StableDiffusionXLPAGImg2ImgPipeline.from_single_file(base_safe, torch_dtype=dtype)
            pipe.vae=AutoencoderKL.from_single_file(vae_safe,torch_dtype=dtype)
            pipe.to(d)
        else:
            pipe = StableDiffusionXLPAGImg2ImgPipeline.from_single_file(base_safe, torch_dtype=dtype).to(d)
            
        if sample=="Euler":
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
        elif sample=="Euler a":
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
        elif sample=="LMS":
            pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
        elif sample=="Heun":
            pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
        elif sample=="DPM2":
            pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
        elif sample=="DPM2 a":
            pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
        elif sample=="DPM++ 2M":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
        elif sample=="DPM++ SDE":
            pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
        elif sample=="DPM++":
            pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
        elif sample=="DPM++ 2M SDE":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
        elif sample=="PLMS":
            pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config,timestep_spacing=sgm_dict["timestep_spacing"])
        elif sample=="UniPC":
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
        elif sample=="LCM":
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config,timestep_spacing=sgm_dict["timestep_spacing"])
        elif sample=="DPM++ 3M SDE":
            pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
        else:
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config,timestep_spacing=sgm_dict["timestep_spacing"])
            
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
                    strength=ss,
                    pag_scale=pag
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
                strength=ss,
                pag_scale=pag
            ).images[0]
            meta_dict["se"]=str(seed[i])
            if j_or_p=="j":
                meta_dict["input"]=out_folder+"/"+str(i)+"_"+str(seed[i])+".jpg"
            else:
                meta_dict["input"]=out_folder+"/"+str(i)+"_"+str(seed[i])+".png"
            plus_meta(meta_dict,image)
            images[i]=image
            clear_output(True)
            print(memo)
            show_img(images[:i+1])
            del image
            torch.cuda.empty_cache()
        del pipe,conditioning
    del images
    return seed

def text2image15(
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
    cs=2,
    Interpolation=3,
    sample=1,
    sgm="",
    seed=0,
    out_folder="data",
    pos_emb=[],
    neg_emb=[],
    base_safe="base.safetensors",
    vae_safe="vae.safetensors",
    pag=3.0,
    j_or_p="j"
    ):
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
    clear_output(True)
    print(memo)
            
    if prog_ver!=1:
        if prog_ver!=2:
            prog_ver=0
        
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
                    del f,d
                except:
                    meta_id_list=list1
                    meta_weight_list=list2
                
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
            else:
                memo=memo+line+" ng"
                clear_output(True)
                print(memo)
                return []
    clear_output(True)
    print(memo)
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
    clear_output(True)
    print(memo)

    del t,Interpolation

    dtype=torch.float16
    if os.path.isfile(vae_safe):
        pipe = StableDiffusionPAGPipeline.from_single_file(base_safe, torch_dtype=dtype)
        pipe.vae=AutoencoderKL.from_single_file(vae_safe,torch_dtype=dtype)
        pipe.to("cuda:0")
    else:
        pipe = StableDiffusionPAGPipeline.from_single_file(base_safe, torch_dtype=dtype).to("cuda:0")
        
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
    
    memo=memo+"num_inference_steps : "+str(f_step)+"\n"
    meta_dict["st"]=str(f_step)
    memo=memo+"guidance_scale : "+str(gs)+"\n"
    meta_dict["cf"]=str(gs)
    memo=memo+"clip_skip : "+str(cs)+"\n"
    meta_dict["cl"]=str(cs)
    memo=memo+"pag_scale : "+str(pag)+"\n"
    meta_dict["pag"]=str(pag)

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
                generator=torch.manual_seed(seed[i]),
                pag_scale=pag
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
                generator=torch.manual_seed(seed[i]),
                pag_scale=pag
            ).images[0]
            meta_dict["se"]=str(seed[i])
            if j_or_p=="j":
                meta_dict["input"]=out_folder+"/"+str(i)+"_"+str(seed[i])+".jpg"
            else:
                meta_dict["input"]=out_folder+"/"+str(i)+"_"+str(seed[i])+".png"
            plus_meta(meta_dict,image)
            images.append(image)
            clear_output(True)
            print(memo)
            show_img(images[:i+1])
            del image
            torch.cuda.empty_cache()

    del pipe,conditioning

    if prog_ver==2 or prog_ver==1:
        if torch.cuda.device_count()==1:
            d="cuda:0"
        else:
            d="cuda:1"
        if os.path.isfile(vae_safe):
            pipe = StableDiffusionPAGImg2ImgPipeline.from_single_file(base_safe, torch_dtype=dtype)
            pipe.vae=AutoencoderKL.from_single_file(vae_safe,torch_dtype=dtype)
            pipe.to(d)
        else:
            pipe = StableDiffusionPAGImg2ImgPipeline.from_single_file(base_safe, torch_dtype=dtype).to(d)
            
        if sample=="Euler":
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
        elif sample=="Euler a":
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
        elif sample=="LMS":
            pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
        elif sample=="Heun":
            pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
        elif sample=="DPM2":
            pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
        elif sample=="DPM2 a":
            pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
        elif sample=="DPM++ 2M":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
        elif sample=="DPM++ SDE":
            pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
        elif sample=="DPM++":
            pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
        elif sample=="DPM++ 2M SDE":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
        elif sample=="PLMS":
            pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config,timestep_spacing=sgm_dict["timestep_spacing"])
        elif sample=="UniPC":
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
        elif sample=="LCM":
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config,timestep_spacing=sgm_dict["timestep_spacing"])
        elif sample=="DPM++ 3M SDE":
            pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config,
            timestep_spacing=sgm_dict["timestep_spacing"],
            use_karras_sigmas=sgm_dict["use_karras_sigmas"],
            use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
            use_beta_sigmas=sgm_dict["use_beta_sigmas"]
            )
        else:
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config,timestep_spacing=sgm_dict["timestep_spacing"])
            
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
                    strength=ss,
                    pag_scale=pag
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
                strength=ss,
                pag_scale=pag
            ).images[0]
            meta_dict["se"]=str(seed[i])
            if j_or_p=="j":
                meta_dict["input"]=out_folder+"/"+str(i)+"_"+str(seed[i])+".jpg"
            else:
                meta_dict["input"]=out_folder+"/"+str(i)+"_"+str(seed[i])+".png"
            plus_meta(meta_dict,image)
            images[i]=image
            clear_output(True)
            print(memo)
            show_img(images[:i+1])
            del image
            torch.cuda.empty_cache()
        del pipe,conditioning
    del images
    return seed

    


