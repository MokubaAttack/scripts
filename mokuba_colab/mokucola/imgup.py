import os
import requests
import torch
import numpy
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer

class imgup:
    def __init__(self,path):
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
                if 3*4==sd["conv_first.weight"].size()[1]:
                    scale=2
                elif 3*16==sd["conv_first.weight"].size()[1]:
                    scale=1
                else:
                    scale=4
                net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=nf, num_block=nb, num_grow_ch=32, scale=scale)
            elif "body.0.weight" in sd:
                nf=sd["body.0.weight"].size()[0]
                for i in range(1000):
                    if not("body."+str(i*2+4)+".weight" in sd):
                        break
                nc=i
                for i in range(1,1000):
                    if int(sd["body."+str(nc*2+2)+".weight"].size()[0])==int(3*(2**i)):
                        break
                scale=i
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