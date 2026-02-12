import random
import sys
import requests
import io
import gc
from PIL import Image
from IPython.display import clear_output

from .mokupipe import mokupipe
from .reset_func import reset_func
from .discord import zip_to_discord

def mokucola(
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
    url="",
    p=None,
    dtype="f16",
    dev="cuda",
    xf=False,
    ser="colab",
    del_pipe=True,
    si=True
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
            return p
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
            reset_func(f=pipe,s=ser)
            return None
    else:
        pipe=p
        pipe.deldiffusionparams()

    pipe.set_diffparams(
        dtype=dtype,
        xf=xf,
        dev=dev
        )
    pipe.set_outparams(
        out_folder=out_folder,
        j_or_p=j_or_p,
        url=url,
        si=si
        )

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
            out=True
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
            out=False
        )

    if prog_ver!=0:
        pipe.mkpipe_upscale(Interpolation)

    if prog_ver==1:
        images=pipe.image2imageup(
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
            out=True
        )
    elif prog_ver==2:
        images=pipe.image2imageup(
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
            out=False
        )
    
    if prog_ver==2:
        images=pipe.image2imageup(
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
            out=True
        )

    if url!="":
        zip_to_discord(out_folder,url)
    del images,seed
    if del_pipe:
        reset_func(f=pipe,s=ser)
        pipe=None
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
    url="",
    out_folder="output",
    j_or_p="j",
    p=None,
    prompt="masterpiece,best quality,ultra detailed",
    n_prompt="worst quality,low quality,normal quality",
    ccs=None,
    tile_size=(0,0),
    ol=0,
    dtype="f16",
    dev="cuda",
    xf=False,
    ser="colab",
    del_pipe=True,
    si=True
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
        return p
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
                return p
    
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
            reset_func(f=pipe,s=ser)
            return None
    else:
        pipe=p
        pipe.deldiffusionparams()

    pipe.set_diffparams(
        dtype=dtype,
        xf=xf,
        dev=dev
        )
    pipe.set_outparams(
        out_folder=out_folder,
        j_or_p=j_or_p,
        url=url,
        si=si
        )

    pipe.mkpipe_upscale(Interpolation)
    images=pipe.tileup(
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
        ccs=ccs,
        tile_size=tile_size,
        ol=ol,
        out=True
        )

    if url!="":
        zip_to_discord(out_folder,url)
    del images,seed
    if del_pipe:
        reset_func(f=pipe,s=ser)
        pipe=None
    return pipe
