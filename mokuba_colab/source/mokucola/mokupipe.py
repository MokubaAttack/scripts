from diffusers import (
	AutoencoderKL,
	ControlNetModel,
	StableDiffusionXLPipeline,
	StableDiffusionXLImg2ImgPipeline,
	StableDiffusionPipeline,
	StableDiffusionImg2ImgPipeline,
	StableDiffusionXLControlNetImg2ImgPipeline,
	StableDiffusionControlNetImg2ImgPipeline,
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
import torch
import os
import numpy
import cv2
from PIL import Image
from safetensors.torch import (
	load_file,
	save_file
)
from IPython.display import clear_output
from compel import (
	CompelForSD,
	CompelForSDXL
)
from optimum.quanto import (
	freeze,
	qfloat8,
	quantize
)
from lycoris import create_lycoris_from_weights
from lycoris.modules.locon import LoConModule
from lycoris.modules.loha import LohaModule
from lycoris.modules.lokr import LokrModule
from lycoris.modules.full import FullModule
from lycoris.modules.norms import NormModule
from lycoris.modules.diag_oft import DiagOFTModule
from lycoris.modules.boft import ButterflyOFTModule
from lycoris.modules.glora import GLoRAModule
from lycoris.modules.dylora import DyLoraModule
from lycoris.modules.ia3 import IA3Module

MODULE_LIST = [
	LoConModule,
	LohaModule,
	IA3Module,
	LokrModule,
	FullModule,
	NormModule,
	DiagOFTModule,
	ButterflyOFTModule,
	GLoRAModule,
	DyLoraModule,
]

from .meta import plus_meta
from .imgup import imgup
from .discord import to_discord
from .imgshow import imgshow
from .ccc import (
	flush,
	getid
)

sgm_use=[
	"Euler","Euler a","DPM++ 2M","DPM++ 2M SDE","DPM++ SDE","DPM++","DPM2","DPM2 a","Heun","LMS","UniPC","DPM++ 3M SDE"
]

sdxl_unet_keys={
	"in_layers_0": "norm1",
	"in_layers_2": "conv1",
	"out_layers_0": "norm2",
	"out_layers_3": "conv2",
	"emb_layers_1": "time_emb_proj",
}
for i in range(3):
	for j in range(2):
		sdxl_unet_keys["input_blocks_"+str(3 * i + j + 1)+"_0"]="down_blocks_"+str(i)+"_resnets_"+str(j)
		if i > 0:
			sdxl_unet_keys["input_blocks_"+str(3 * i + j + 1)+"_1"]="down_blocks_"+str(i)+"_attentions_"+str(j)

	for j in range(4):
		sdxl_unet_keys["output_blocks_"+str(3 * i + j)+"_0"]="up_blocks_"+str(i)+"_resnets_"+str(j)
		if i < 2:
			sdxl_unet_keys["output_blocks_"+str(3 * i + j)+"_1"]="up_blocks_"+str(i)+"_attentions_"+str(j)

	if i < 3:
		sdxl_unet_keys["input_blocks_"+str(3 * (i + 1))+"_0_op"]="down_blocks_"+str(i)+"_downsamplers_0_conv"

		if i==0:
			sdxl_unet_keys["output_blocks_"+str(3 * i + 2)+"_1"]="up_blocks_"+str(i)+"_upsamplers_0"
		else:
			sdxl_unet_keys["output_blocks_"+str(3 * i + 2)+"_2"]="up_blocks_"+str(i)+"_upsamplers_0"
sdxl_unet_keys["output_blocks_2_1_conv"]="output_blocks_2_2_conv"

sdxl_unet_keys["middle_block_1"]="mid_block_attentions_0"
for j in range(2):
	sdxl_unet_keys["middle_block_"+str(2 * j)]="mid_block_resnets_"+str(j)
	
sd_unet_keys={
	"in_layers_0": "norm1",
	"in_layers_2": "conv1",
	"out_layers_0": "norm2",
	"out_layers_3": "conv2",
	"emb_layers_1": "time_emb_proj",
}
for i in range(4):
	for j in range(2):
		sd_unet_keys["input_blocks_"+str(3 * i + j + 1)+"_0"]="down_blocks_"+str(i)+"_resnets_"+str(j)
		if i < 3:
			sdxl_unet_keys["input_blocks_"+str(3 * i + j + 1)+"_1"]="down_blocks_"+str(i)+"_attentions_"+str(j)

	for j in range(3):
		sd_unet_keys["output_blocks_"+str(3 * i + j)+"_0"]="up_blocks_"+str(i)+"_resnets_"+str(j)
		if i > 0:
			sd_unet_keys["output_blocks_"+str(3 * i + j)+"_1"]="up_blocks_"+str(i)+"_attentions_"+str(j)

	if i < 3:
		sd_unet_keys["input_blocks_"+str(3 * (i + 1))+"_0_op"]="down_blocks_"+str(i)+"_downsamplers_0_conv"

		if i==0:
			sd_unet_keys["output_blocks_"+str(3 * i + 2)+"_1"]="up_blocks_"+str(i)+"_upsamplers_0"
		else:
			sd_unet_keys["output_blocks_"+str(3 * i + 2)+"_2"]="up_blocks_"+str(i)+"_upsamplers_0"

sd_unet_keys["middle_block_1"]="mid_block_attentions_0"
for j in range(2):
	sd_unet_keys["middle_block_"+str(2 * j)]="mid_block_resnets_"+str(j)

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
		self.prompts=None
		self.prompt=""
		self.n_prompt=""
		self.prompt_a=""
		self.n_prompt_a=""

		self.out_folder="output"
		self.url=""
		self.si=True

		self.dtype=torch.float16
		self.dev="cuda"
		self.lowmem=False

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
		self.meta_dict["ckpt"]=getid(base_safe,None)
		self.meta_dict["ckpt_name"]=base_safe

		if os.path.exists(vae_safe):
			self.meta_dict["vae"]=getid(vae_safe,None)
			self.meta_dict["vae_name"]=vae_safe
		else:
			self.meta_dict["vae"]=""
			self.meta_dict["vae_name"]=""

		sd=load_file(base_safe)
		self.is_sdxl="conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in sd

		if self.is_sdxl:
			self.pipe=StableDiffusionXLPipeline.from_single_file(base_safe,torch_dtype=self.dtype)
			print(self.meta_dict["ckpt_name"]+" is loaded.")
			if os.path.isfile(vae_safe):
				self.pipe.vae=AutoencoderKL.from_single_file(vae_safe,torch_dtype=self.dtype)
				print(self.meta_dict["vae_name"]+" is loaded.")
		else:
			self.pipe=StableDiffusionPipeline.from_single_file(base_safe,torch_dtype=self.dtype)
			print(self.meta_dict["ckpt_name"]+" is loaded.")
			if os.path.isfile(vae_safe):
				self.pipe.vae=AutoencoderKL.from_single_file(vae_safe,torch_dtype=self.dtype)
				print(self.meta_dict["vae_name"]+" is loaded.")
		if self.lowmem:
			self.pipe.vae.tile_sample_min_size=256
			sample_size=256
			self.pipe.vae.tile_latent_min_size = int(sample_size / (2 ** (len(self.pipe.vae.config.block_out_channels) - 1)))
		self.pipe.to(self.dev)

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
					sd=load_file(line+".safetensors")
					lora_check=False
					for k in sd:
						if k.endswith(".lora_up.weight") or k.endswith(".lora_B.weight"):
							lora_check=True
							break

					if lora_check:
						if self.is_sdxl:
							ukeys=[]
							for name, module in self.pipe.unet.named_modules():
								ukeys.append(name.replace(".","_"))
							t1keys=[]
							for name, module in self.pipe.text_encoder.named_modules():
								t1keys.append(name.replace(".","_"))
							t2keys=[]
							for name, module in self.pipe.text_encoder_2.named_modules():
								t2keys.append(name.replace(".","_"))
	
							msd={}
							for k in sd:
								if not(k.endswith(".lora_up.weight") or k.endswith(".lora_B.weight")):
									continue
								if k.endswith(".lora_up.weight"):
									m=k.removesuffix(".lora_up.weight")
								else:
									m=k.removesuffix(".lora_B.weight")
								if m.replace(".","_").startswith("lora_unet_"):
									m2=m.replace(".","_").removeprefix("lora_unet_")
									for k2 in sdxl_unet_keys:
										if k2 in m2:
											m2=m2.replace(k2,sdxl_unet_keys[k2])
									if m2 in ukeys:
										for k2 in [".lora_up.weight",".lora_down.weight",".lora_B.weight",".lora_A.weight",".alpha"]:
											if m+k2 in sd:
												msd[m+k2]=sd[m+k2]
								elif m.replace(".","_").startswith("lora_te1_"):
									if m.replace(".","_").removeprefix("lora_te1_") in t1keys:
										for k2 in [".lora_up.weight",".lora_down.weight",".lora_B.weight",".lora_A.weight",".alpha"]:
											if m+k2 in sd:
												msd[m+k2]=sd[m+k2]
								elif m.replace(".","_").startswith("lora_te2_"):
									if m.replace(".","_").removeprefix("lora_te2_") in t2keys:
										for k2 in [".lora_up.weight",".lora_down.weight",".lora_B.weight",".lora_A.weight",".alpha"]:
											if m+k2 in sd:
												msd[m+k2]=sd[m+k2]
						else:
							ukeys=[]
							for name, module in self.pipe.unet.named_modules():
								ukeys.append(name.replace(".","_"))
							t1keys=[]
							for name, module in self.pipe.text_encoder.named_modules():
								t1keys.append(name.replace(".","_"))
								
							msd={}
							for k in sd:
								if not(k.endswith(".lora_up.weight") or k.endswith(".lora_B.weight")):
									continue
								if k.endswith(".lora_up.weight"):
									m=k.removesuffix(".lora_up.weight")
								else:
									m=k.removesuffix(".lora_B.weight")
								if m.replace(".","_").startswith("lora_unet_"):
									m2=m.replace(".","_").removeprefix("lora_unet_")
									for k2 in sd_unet_keys:
										if k2 in m2:
											m2=m2.replace(k2,sd_unet_keys[k2])
									if m2 in ukeys:
										for k2 in [".lora_up.weight",".lora_down.weight",".lora_B.weight",".lora_A.weight",".alpha"]:
											if m+k2 in sd:
												msd[m+k2]=sd[m+k2]
								elif m.replace(".","_").startswith("lora_te_"):
									if m.replace(".","_").removeprefix("lora_te_") in t1keys:
										for k2 in [".lora_up.weight",".lora_down.weight",".lora_B.weight",".lora_A.weight",".alpha"]:
											if m+k2 in sd:
												msd[m+k2]=sd[m+k2]

						if msd=={}:
							raise ValueError('These weights are not supported.')
							
						self.pipe.load_lora_weights(pretrained_model_name_or_path_or_dict=msd,torch_dtype=self.dtype)
						self.pipe.fuse_lora(lora_scale= lora_weights[i])
						self.pipe.unload_lora_weights()
						if self.is_sdxl:
							del msd,ukeys,t1keys,t2keys
						else:
							del msd,ukeys,t1keys
					else:
						MODULE_type=None
						for m in MODULE_LIST:
							for k in m.weight_list_det:
								for k2 in sd:
									if k2.endswith(k):
										MODULE_type=m
										break
								if MODULE_type!=None:
									break
							if MODULE_type!=None:
								break
						if MODULE_type==None:
							raise ValueError('These weights are not supported.')
						key_name=[]
						for k in sd:
							for k2 in MODULE_type.weight_list_det:
								if k.endswith("."+k2):
									key_name.append(k.removesuffix("."+k2))

						usd={}
						t1sd={}
						t2sd={}
						tsd={}

						for k in key_name:
							m=k.replace(".","_")
							if k.startswith("lora_unet_"):
								m=m.removeprefix("lora_unet_")
								if self.is_sdxl:
									keys=sdxl_unet_keys
								else:
									keys=sd_unet_keys
								for k2 in keys:
									if k2 in m:
										m=m.replace(k2,keys[k2])
								for k2 in MODULE_type.weight_list:
									if k+"."+k2 in sd:
										usd["lycoris_"+m+"."+k2]=sd[k+"."+k2]
							elif k.startswith("lora_te1_"):
								m=m.removeprefix("lora_te1_")
								for k2 in MODULE_type.weight_list:
									if k+"."+k2 in sd:
										t1sd["lycoris_"+m+"."+k2]=sd[k+"."+k2]
							elif k.startswith("lora_te2_"):
								m=m.removeprefix("lora_te2_")
								for k2 in MODULE_type.weight_list:
									if k+"."+k2 in sd:
										t2sd["lycoris_"+m+"."+k2]=sd[k+"."+k2]
							elif k.startswith("lora_te_"):
								m=m.removeprefix("lora_te_")
								for k2 in MODULE_type.weight_list:
									if k+"."+k2 in sd:
										tsd["lycoris_"+m+"."+k2]=sd[k+"."+k2]

						if self.is_sdxl and usd=={} and t1sd=={} and t2sd=={}:
							raise ValueError('These weights are not supported.')
						elif self.is_sdxl==False and usd=={} and tsd=={}:
							raise ValueError('These weights are not supported.')

						if usd!={}:
							wrapper, _ = create_lycoris_from_weights(multiplier=lora_weights[i],file="dummy.safetensors",module=self.pipe.unet, weights_sd=usd)
							wrapper.merge_to()
						del usd

						if self.is_sdxl and t1sd!={}:
							wrapper, _ = create_lycoris_from_weights(multiplier=lora_weights[i],file="dummy.safetensors",module=self.pipe.text_encoder, weights_sd=t1sd)
							wrapper.merge_to()
						del t1sd

						if self.is_sdxl and t2sd!={}:
							wrapper, _ = create_lycoris_from_weights(multiplier=lora_weights[i],file="dummy.safetensors",module=self.pipe.text_encoder_2, weights_sd=t2sd)
							wrapper.merge_to()
						del t2sd

						if self.is_sdxl==False and tsd!={}:
							wrapper, _ = create_lycoris_from_weights(multiplier=lora_weights[i],file="dummy.safetensors",module=self.pipe.text_encoder, weights_sd=tsd)
							wrapper.merge_to()
						del tsd

					print(line+".safetensors is loaded.")

					list1,list2=getid(line+".safetensors",lora_weights[i])
					meta_id_list=meta_id_list+list1
					meta_weight_list=meta_weight_list+list2
					del list1,list2,sd
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
						self.pipe.load_textual_inversion(state_dict["clip_g"],token=key,text_encoder=self.pipe.text_encoder_2,tokenizer=self.pipe.tokenizer_2,torch_dtype=self.dtype)
						self.pipe.load_textual_inversion(state_dict["clip_l"],token=key,text_encoder=self.pipe.text_encoder,tokenizer=self.pipe.tokenizer,torch_dtype=self.dtype)
						del state_dict
					else:
						self.pipe.load_textual_inversion(".", weight_name=line, token=key)
					self.prompt_a = self.prompt_a+","+key
					print(line+".safetensors is loaded.")
					del key
					list1,list2=getid(line,1)
					meta_embed_list=meta_embed_list+list1
					del list1,list2
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
						self.pipe.load_textual_inversion(state_dict["clip_g"],token=key,text_encoder=self.pipe.text_encoder_2,tokenizer=self.pipe.tokenizer_2,torch_dtype=self.dtype)
						self.pipe.load_textual_inversion(state_dict["clip_l"],token=key,text_encoder=self.pipe.text_encoder,tokenizer=self.pipe.tokenizer,torch_dtype=self.dtype)
						del state_dict
					else:
						self.pipe.load_textual_inversion(".", weight_name=line, token=key)
					self.n_prompt_a=self.n_prompt_a+","+key
					print(line+".safetensors is loaded.")
					del key
					list1,list2=getid(line,1)
					meta_embed_list=meta_embed_list+list1
					del list1,list2
				else:
					print(line+" does not exist.")
					return -1
		self.meta_dict["embed"]=str(meta_embed_list)
		del meta_embed_list
		self.meta_dict["pos"]=pos_emb
		self.meta_dict["neg"]=neg_emb

		if self.lowmem:
			quantize(self.pipe.unet, weights=qfloat8)
			freeze(self.pipe.unet)
		
		return 1

	def mkpipe_upscale(self,path,dev):
		self.upscaler=imgup(path,dev)

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
		return 1

	def set_outparams(self,out_folder="",url="",si=True):
		self.out_folder=out_folder
		self.url=url
		self.si=si

	def set_diffparams(self,dtype="f16",dev="cuda",lowmem=False):
		if dtype=="f32":
			self.dtype=torch.float32
		elif dtype=="bf16":
			self.dtype=torch.bfloat16
		else:
			self.dtype=torch.float16
		self.dev=dev
		self.lowmem=lowmem

	def text2image(self,prompt,n_prompt,gs,step,cs,seed,x,y,out):
		if self.pipe==None:
			print("You must make a pipeline.")
			return []
		if self.is_sdxl:
			self.pipe=StableDiffusionXLPipeline.from_pipe(self.pipe,torch_dtype=self.dtype)
		else:
			self.pipe=StableDiffusionPipeline.from_pipe(self.pipe,torch_dtype=self.dtype)
		if self.lowmem:
			self.pipe.vae.tile_sample_min_size=256
			sample_size=256
			self.pipe.vae.tile_latent_min_size = int(sample_size / (2 ** (len(self.pipe.vae.config.block_out_channels) - 1)))
		self.pipe.to(self.dev)
			
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

		memo=memo+"prompt\n"+prompt+"\nnegative prompt\n"+n_prompt+"\n"
		for k in ["hs","ds","hu","hum","up","ccs","cont","tu","tum"]:
			if k in self.meta_dict:
				del self.meta_dict[k]

		self.pipe.vae.enable_tiling()
		
		if self.prompts==None:
			self.mkprompt(prompt=prompt,n_prompt=n_prompt)
		if self.prompt!=prompt or self.n_prompt!=n_prompt:
			self.mkprompt(prompt=prompt,n_prompt=n_prompt)

		images=[]

		if out:
			if not(os.path.exists(self.out_folder)):
				os.makedirs(self.out_folder)
		j=0
		for i in seed:
			j=j+1
			clear_output(True)
			print(memo)
			if len(images)>0 and self.si:
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
				).images[0]
			if out:
				self.meta_dict["se"]=str(i)
				self.meta_dict["input"]=self.out_folder+"/"+str(j)+"_"+str(i)+".jpg"
				plus_meta(self.meta_dict,image)
				if self.url!="":
					to_discord(self.meta_dict["input"],self.url)
			images.append(image)
			del image
			flush()

		clear_output(True)
		print(memo)
		if self.si:
			imgshow(imgs=images)

		return images

	def image2imageup(self,prompt,n_prompt,gs,step,cs,seed,x,y,ss,images,out):
		if self.pipe==None:
			print("You must make a pipeline.")
			return []
		if self.is_sdxl:
			self.pipe=StableDiffusionXLImg2ImgPipeline.from_pipe(self.pipe,torch_dtype=self.dtype)
		else:
			self.pipe=StableDiffusionImg2ImgPipeline.from_pipe(self.pipe,torch_dtype=self.dtype)
		if self.lowmem:
			self.pipe.vae.tile_sample_min_size=512
			sample_size=512
			self.pipe.vae.tile_latent_min_size = int(sample_size / (2 ** (len(self.pipe.vae.config.block_out_channels) - 1)))
		if self.lowmem:
			self.pipe.vae.tile_sample_min_size=256
			sample_size=256
			self.pipe.vae.tile_latent_min_size = int(sample_size / (2 ** (len(self.pipe.vae.config.block_out_channels) - 1)))
		self.pipe.to(self.dev)

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

		memo=memo+"prompt\n"+prompt+"\nnegative prompt\n"+n_prompt+"\n"
		for k in ["hs","ds","hu","hum","up","ccs","cont","tu","tum"]:
			if k in self.meta_dict:
				del self.meta_dict[k]

		self.pipe.vae.enable_tiling()
		
		if self.prompts==None:
			self.mkprompt(prompt=prompt,n_prompt=n_prompt)
		if self.prompt!=prompt or self.n_prompt!=n_prompt:
			self.mkprompt(prompt=prompt,n_prompt=n_prompt)

		if out:
			if not(os.path.exists(self.out_folder)):
				os.makedirs(self.out_folder)

		if self.upscaler==None:
			print("You must make a upscaler.")
			return []
		else:
			checklist=[]
			for j in range(len(images)):
				if images[j]==images[0]:
					checklist.append(1)
			if len(checklist)==len(images):
				u_list=[]
				for j in range(len(images)):
					u_list.append(1)
					checklist[j]=False
				if x!=images[0].width or y!=images[0].height:
					u=x/images[0].width
					images[0]=self.upscaler.run(images[0],x,y)
					for j in range(len(images)):
						u_list[j]=u
						images[j]=images[0]
						checklist[j]=True
			else:
				checklist=[]
				u_list=[]
				for j in range(len(images)):
					if x!=images[j].width or y!=images[j].height:
						u_list.append(x/images[j].width)
						images[j]=self.upscaler.run(images[j],x,y)
						checklist.append(True)
					else:
						u_list.append(1)
						checklist.append(False)

		del self.upscaler.model
		j=0
		for i in seed:
			j=j+1
			if checklist[j-1]:
				memo1=memo+"Hires steps : "+str(step)+"\n"
				self.meta_dict["hs"]=str(step)
				memo1=memo1+"Denoising strength : "+str(ss)+"\n"
				self.meta_dict["ds"]=str(ss)
				memo1=memo1+"Hires upscale : "+str(u_list[j-1])+"\n"
				self.meta_dict["hu"]=str(u_list[j-1])
				self.meta_dict["hum"],self.meta_dict["up"]=self.upscaler.get_method()
				memo1=memo1+"Hires upscaler : "+self.meta_dict["hum"]+"\n"
			else:
				memo1=memo+"Denoising strength : "+str(ss)+"\n"
				self.meta_dict["ds"]=str(ss)
			clear_output(True)
			print(memo1)
			if len(images)>0 and self.si:
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
				).images[0]
			if out:
				self.meta_dict["se"]=str(i)
				self.meta_dict["input"]=self.out_folder+"/"+str(j)+"_"+str(i)+".jpg"
				plus_meta(self.meta_dict,image)
				if self.url!="":
					to_discord(self.meta_dict["input"],self.url)
			images[j-1]=image
			del image
			flush()

		clear_output(True)
		print(memo1)
		if self.si:
			imgshow(imgs=images)

		return images

	def tileup(self,prompt,n_prompt,gs,step,cs,seed,x,y,ss,images,ccs=None,tile_size=(0,0),ol=0,out=True):
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
				self.pipe=StableDiffusionXLPAGImg2ImgPipeline.from_pipe(self.pipe,torch_dtype=self.dtype)
			else:
				self.pipe=StableDiffusionPAGImg2ImgPipeline.from_pipe(self.pipe,torch_dtype=self.dtype)
			for k in ["cont","ccs"]:
				if k in self.meta_dict:
					del self.meta_dict[k]
		else:
			if self.is_sdxl:
				controlnet = ControlNetModel.from_pretrained("OzzyGT/SDXL_Controlnet_Tile_Realistic",torch_dtype=self.dtype,variant="fp16")
				self.pipe=StableDiffusionXLControlNetImg2ImgPipeline.from_pipe(self.pipe,torch_dtype=self.dtype,controlnet=controlnet)
				self.meta_dict["cont"]=str(370104)
			else:
				controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile',torch_dtype=self.dtype)
				self.pipe=StableDiffusionControlNetImg2ImgPipeline.from_pipe(self.pipe,torch_dtype=self.dtype,controlnet=controlnet)
				self.meta_dict["cont"]=str(67566)
			self.meta_dict["ccs"]=str(ccs)
		if self.lowmem:
			self.pipe.vae.tile_sample_min_size=256
			sample_size=256
			self.pipe.vae.tile_latent_min_size = int(sample_size / (2 ** (len(self.pipe.vae.config.block_out_channels) - 1)))
		self.pipe.to(self.dev)
			
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

		memo=memo+"prompt\n"+prompt+"\nnegative prompt\n"+n_prompt+"\n"
		for k in ["hs","ds","hu","hum","up","tu","tum"]:
			if k in self.meta_dict:
				del self.meta_dict[k]

		self.pipe.vae.enable_tiling()
		
		if self.prompts==None:
			self.mkprompt(prompt=prompt,n_prompt=n_prompt)
		if self.prompt!=prompt or self.n_prompt!=n_prompt:
			self.mkprompt(prompt=prompt,n_prompt=n_prompt)

		if out:
			if not(os.path.exists(self.out_folder)):
				os.makedirs(self.out_folder)

		if ccs==None:
			x=round(x/8)*8
			y=round(y/8)*8
		else:
			x=round(x/64)*64
			y=round(y/64)*64

		if self.upscaler==None:
			print("You must make a upscaler.")
			return []
		else:
			checklist=[]
			for j in range(len(images)):
				if images[j]==images[0]:
					checklist.append(1)
			if len(checklist)==len(images):
				u_list=[]
				for i in range(len(images)):
					u_list.append(1)
				if x!=images[0].width or y!=images[0].height:
					u=x/images[0].width
					images[0]=self.upscaler.run(images[0],x,y)
					for j in range(len(images)):
						u_list[j]=u
						images[j]=images[0]
			else:
				u_list=[]
				for j in range(len(images)):
					if x!=images[j].width or y!=images[j].height:
						u_list.append(x/images[j].width)
						images[j]=self.upscaler.run(images[j],x,y)
					else:
						u_list.append(1)

		del self.upscaler.model
		j=0
		for i in seed:
			j=j+1
			
			memo1=memo+"Denoising strength : "+str(ss)+"\n"
			self.meta_dict["ds"]=str(ss)
			memo1=memo1+"Tile upscale : "+str(u_list[j-1])+"\n"
			self.meta_dict["tu"]=str(u_list[j-1])
			self.meta_dict["tum"],self.meta_dict["up"]=self.upscaler.get_method()
			memo1=memo1+"Tile upscaler : "+self.meta_dict["tum"]+"\n"
			if ccs!=None:
				memo1=memo1+"controlnet_conditioning_scale : "+self.meta_dict["ccs"]+"\n"
			clear_output(True)
			print(memo1)
			if len(images)>0 and self.si:
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
					flush()
			final_result = (result / weight_sum).astype(numpy.uint8)
			image = Image.fromarray(final_result)
			images[j-1]=image

			if out:
				self.meta_dict["se"]=str(i)
				self.meta_dict["input"]=self.out_folder+"/"+str(j)+"_"+str(i)+".jpg"
				plus_meta(self.meta_dict,image)
				if self.url!="":
					to_discord(self.meta_dict["input"],self.url)
			del image,final_result,result,weight_sum
			flush()

		clear_output(True)
		print(memo1)
		if self.si:
			imgshow(imgs=images)	

		return images

	def deldiffusionparams(self):
		self.prompts=None
		del_keys=["tum","se","input","ds","tu","css","pr","ne","st","cf","cl","pag","hs","hu","cont","hum","up"]
		for k in del_keys:
			if k in self.meta_dict:
				del self.meta_dict[k]
		flush()
