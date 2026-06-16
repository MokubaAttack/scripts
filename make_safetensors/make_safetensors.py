from diffusers import (
	StableDiffusionXLPipeline,
	AutoencoderKL
)
import torch
import shutil
import os
import re
import math
from safetensors.torch import (
	load_file, 
	save_file
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

unet_keys={
	"middle_block_0":"mid_block_resnets_0",
	"middle_block_1":"mid_block_attentions_0",
	"middle_block_2":"mid_block_resnets_1",
	"in_layers_0": "norm1",
	"in_layers_2": "conv1",
	"out_layers_0": "norm2",
	"out_layers_3": "conv2",
	"emb_layers_1": "time_emb_proj",
	"skip_connection": "conv_shortcut",
	"output_blocks_0_0":"up_blocks_0_resnets_0",
	"output_blocks_0_1":"up_blocks_0_attentions_0",
	"output_blocks_1_0":"up_blocks_0_resnets_1",
	"output_blocks_1_1":"up_blocks_0_attentions_1",
	"output_blocks_2_0":"up_blocks_0_resnets_2",
	"output_blocks_2_1":"up_blocks_0_attentions_2",
	"output_blocks_2_2":"up_blocks_0_upsamplers_0",
	"output_blocks_3_0":"up_blocks_1_resnets_0",
	"output_blocks_3_1":"up_blocks_1_attentions_0",
	"output_blocks_4_0":"up_blocks_1_resnets_1",
	"output_blocks_4_1":"up_blocks_1_attentions_1",
	"output_blocks_5_0":"up_blocks_1_resnets_2",
	"output_blocks_5_1":"up_blocks_1_attentions_2",
	"output_blocks_5_2":"up_blocks_1_upsamplers_0",
	"output_blocks_6_0":"up_blocks_2_resnets_0",
	"output_blocks_7_0":"up_blocks_2_resnets_1",
	"output_blocks_8_0":"up_blocks_2_resnets_2",
	"input_blocks_1_0":"down_blocks_0_resnets_0",
	"input_blocks_2_0":"down_blocks_0_resnets_1",
	"input_blocks_3_0_op":"down_blocks_0_downsamplers_0_conv",
	"input_blocks_4_0":"down_blocks_1_resnets_0",
	"input_blocks_4_1":"down_blocks_1_attentions_0",
	"input_blocks_5_0":"down_blocks_1_resnets_1",
	"input_blocks_5_1":"down_blocks_1_attentions_1",
	"input_blocks_6_0_op":"down_blocks_1_downsamplers_0_conv",
	"input_blocks_7_0":"down_blocks_2_resnets_0",
	"input_blocks_7_1":"down_blocks_2_attentions_0",
	"input_blocks_8_0":"down_blocks_2_resnets_1",
	"input_blocks_8_1":"down_blocks_2_attentions_1",
}

if not(os.path.exists(os.getcwd()+"/pipecache")):
	os.mkdir(os.getcwd()+"/pipecache")
	
# =================#
# UNet Conversion #
# =================#

unet_conversion_map = [
	# (stable-diffusion, HF Diffusers)
	("time_embed.0.weight", "time_embedding.linear_1.weight"),
	("time_embed.0.bias", "time_embedding.linear_1.bias"),
	("time_embed.2.weight", "time_embedding.linear_2.weight"),
	("time_embed.2.bias", "time_embedding.linear_2.bias"),
	("input_blocks.0.0.weight", "conv_in.weight"),
	("input_blocks.0.0.bias", "conv_in.bias"),
	("out.0.weight", "conv_norm_out.weight"),
	("out.0.bias", "conv_norm_out.bias"),
	("out.2.weight", "conv_out.weight"),
	("out.2.bias", "conv_out.bias"),
	# the following are for sdxl
	("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
	("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
	("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
	("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
]

unet_conversion_map_resnet = [
	# (stable-diffusion, HF Diffusers)
	("in_layers.0", "norm1"),
	("in_layers.2", "conv1"),
	("out_layers.0", "norm2"),
	("out_layers.3", "conv2"),
	("emb_layers.1", "time_emb_proj"),
	("skip_connection", "conv_shortcut"),
]

unet_conversion_map_layer = []
# hardcoded number of downblocks and resnets/attentions...
# would need smarter logic for other networks.
for i in range(3):
	# loop over downblocks/upblocks

	for j in range(2):
		# loop over resnets/attentions for downblocks
		hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
		sd_down_res_prefix = f"input_blocks.{3 * i + j + 1}.0."
		unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

		if i > 0:
			hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
			sd_down_atn_prefix = f"input_blocks.{3 * i + j + 1}.1."
			unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

	for j in range(4):
		# loop over resnets/attentions for upblocks
		hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
		sd_up_res_prefix = f"output_blocks.{3 * i + j}.0."
		unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

		if i < 2:
			# no attention layers in up_blocks.0
			hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
			sd_up_atn_prefix = f"output_blocks.{3 * i + j}.1."
			unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

	if i < 3:
		# no downsample in down_blocks.3
		hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
		sd_downsample_prefix = f"input_blocks.{3 * (i + 1)}.0.op."
		unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

		# no upsample in up_blocks.3
		hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
		sd_upsample_prefix = f"output_blocks.{3 * i + 2}.{1 if i == 0 else 2}."
		unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))
unet_conversion_map_layer.append(("output_blocks.2.2.conv.", "output_blocks.2.1.conv."))

hf_mid_atn_prefix = "mid_block.attentions.0."
sd_mid_atn_prefix = "middle_block.1."
unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))
for j in range(2):
	hf_mid_res_prefix = f"mid_block.resnets.{j}."
	sd_mid_res_prefix = f"middle_block.{2 * j}."
	unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

# ================#
# VAE Conversion #
# ================#

vae_conversion_map = [
	# (stable-diffusion, HF Diffusers)
	("nin_shortcut", "conv_shortcut"),
	("norm_out", "conv_norm_out"),
	("mid.attn_1.", "mid_block.attentions.0."),
]

for i in range(4):
	# down_blocks have two resnets
	for j in range(2):
		hf_down_prefix = f"encoder.down_blocks.{i}.resnets.{j}."
		sd_down_prefix = f"encoder.down.{i}.block.{j}."
		vae_conversion_map.append((sd_down_prefix, hf_down_prefix))

	if i < 3:
		hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0."
		sd_downsample_prefix = f"down.{i}.downsample."
		vae_conversion_map.append((sd_downsample_prefix, hf_downsample_prefix))

		hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
		sd_upsample_prefix = f"up.{3 - i}.upsample."
		vae_conversion_map.append((sd_upsample_prefix, hf_upsample_prefix))

	# up_blocks have three resnets
	# also, up blocks in hf are numbered in reverse from sd
	for j in range(3):
		hf_up_prefix = f"decoder.up_blocks.{i}.resnets.{j}."
		sd_up_prefix = f"decoder.up.{3 - i}.block.{j}."
		vae_conversion_map.append((sd_up_prefix, hf_up_prefix))

# this part accounts for mid blocks in both the encoder and the decoder
for i in range(2):
	hf_mid_res_prefix = f"mid_block.resnets.{i}."
	sd_mid_res_prefix = f"mid.block_{i + 1}."
	vae_conversion_map.append((sd_mid_res_prefix, hf_mid_res_prefix))

vae_conversion_map_attn = [
	# (stable-diffusion, HF Diffusers)
	("norm.", "group_norm."),
	# the following are for SDXL
	("q.", "to_q."),
	("k.", "to_k."),
	("v.", "to_v."),
	("proj_out.", "to_out.0."),
]

# =========================#
# Text Encoder Conversion #
# =========================#

textenc_conversion_lst = [
	# (stable-diffusion, HF Diffusers)
	("transformer.resblocks.", "text_model.encoder.layers."),
	("ln_1", "layer_norm1"),
	("ln_2", "layer_norm2"),
	(".c_fc.", ".fc1."),
	(".c_proj.", ".fc2."),
	(".attn", ".self_attn"),
	("ln_final.", "text_model.final_layer_norm."),
	("token_embedding.weight", "text_model.embeddings.token_embedding.weight"),
	("positional_embedding", "text_model.embeddings.position_embedding.weight"),
]
protected = {re.escape(x[1]): x[0] for x in textenc_conversion_lst}
textenc_pattern = re.compile("|".join(protected.keys()))

# Ordering is from https://github.com/pytorch/pytorch/blob/master/test/cpp/api/modules.cpp
code2idx = {"q": 0, "k": 1, "v": 2}
	
def convert_unet_state_dict(unet_state_dict):
	# buyer beware: this is a *brittle* function,
	# and correct output requires that all of these pieces interact in
	# the exact order in which I have arranged them.
	mapping = {k: k for k in unet_state_dict.keys()}
	for sd_name, hf_name in unet_conversion_map:
		mapping[hf_name] = sd_name
	for k, v in mapping.items():
		if "resnets" in k:
			for sd_part, hf_part in unet_conversion_map_resnet:
				v = v.replace(hf_part, sd_part)
			mapping[k] = v
	for k, v in mapping.items():
		for sd_part, hf_part in unet_conversion_map_layer:
			v = v.replace(hf_part, sd_part)
		mapping[k] = v
	new_state_dict = {sd_name: unet_state_dict[hf_name] for hf_name, sd_name in mapping.items()}
	return new_state_dict

def reshape_weight_for_sd(w):
	# convert HF linear weights to SD conv2d weights
	if not w.ndim == 1:
		return w.reshape(*w.shape, 1, 1)
	else:
		return w
	
def convert_vae_state_dict(vae_state_dict):
	mapping = {k: k for k in vae_state_dict.keys()}
	for k, v in mapping.items():
		for sd_part, hf_part in vae_conversion_map:
			v = v.replace(hf_part, sd_part)
		mapping[k] = v
	for k, v in mapping.items():
		if "attentions" in k:
			for sd_part, hf_part in vae_conversion_map_attn:
				v = v.replace(hf_part, sd_part)
			mapping[k] = v
	new_state_dict = {v: vae_state_dict[k] for k, v in mapping.items()}
	weights_to_convert = ["q", "k", "v", "proj_out"]
	for k, v in new_state_dict.items():
		for weight_name in weights_to_convert:
			if f"mid.attn_1.{weight_name}.weight" in k:
				#print(f"Reshaping {k} for SD format")
				new_state_dict[k] = reshape_weight_for_sd(v)
	return new_state_dict

def convert_openclip_text_enc_state_dict(text_enc_dict):
	new_state_dict = {}
	capture_qkv_weight = {}
	capture_qkv_bias = {}
	for k, v in text_enc_dict.items():
		if (
			k.endswith(".self_attn.q_proj.weight")
			or k.endswith(".self_attn.k_proj.weight")
			or k.endswith(".self_attn.v_proj.weight")
		):
			k_pre = k[: -len(".q_proj.weight")]
			k_code = k[-len("q_proj.weight")]
			if k_pre not in capture_qkv_weight:
				capture_qkv_weight[k_pre] = [None, None, None]
			capture_qkv_weight[k_pre][code2idx[k_code]] = v
			continue

		if (
			k.endswith(".self_attn.q_proj.bias")
			or k.endswith(".self_attn.k_proj.bias")
			or k.endswith(".self_attn.v_proj.bias")
		):
			k_pre = k[: -len(".q_proj.bias")]
			k_code = k[-len("q_proj.bias")]
			if k_pre not in capture_qkv_bias:
				capture_qkv_bias[k_pre] = [None, None, None]
			capture_qkv_bias[k_pre][code2idx[k_code]] = v
			continue

		relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k)
		new_state_dict[relabelled_key] = v

	for k_pre, tensors in capture_qkv_weight.items():
		if None in tensors:
			raise Exception("CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing")
		relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k_pre)
		new_state_dict[relabelled_key + ".in_proj_weight"] = torch.cat(tensors)

	for k_pre, tensors in capture_qkv_bias.items():
		if None in tensors:
			raise Exception("CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing")
		relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k_pre)
		new_state_dict[relabelled_key + ".in_proj_bias"] = torch.cat(tensors)

	return new_state_dict

def convert_openai_text_enc_state_dict(text_enc_dict):
	return text_enc_dict

def zip_ckpt(ckpt1,ckpt2):
	for k in ckpt1:
		if k in ckpt2 and not("first_stage_model." in k):
			sum1=torch.sum(torch.abs(ckpt1[k].to(torch.float32))).item()
			sum2=torch.sum(torch.abs(ckpt2[k].to(torch.float32))).item()
			n=not(math.isnan(sum1) or math.isnan(sum2))
			if n and sum1!=sum2:
				ckpt1[k]=ckpt1[k]*sum2/sum1
	return ckpt1

def run(base_safe,vae_safe,out_safe,lora1,lora2,lora3,lora1w,lora2w,lora3w,win=None):
	if win!=None:
		win["RUN"].Update(disabled=True)
	try:
		dtype=torch.float16
		if not(os.path.exists(base_safe)):
			if win==None:
				print("error : the ckpt file doesn't exist.")
			else:
				win['RUN'].Update(disabled=False)
				win["info"].update("error : the ckpt file doesn't exist.")
			return
		pipe = StableDiffusionXLPipeline.from_single_file(base_safe, torch_dtype=dtype,cache_dir=os.getcwd()+"/pipecache")
				
		if vae_safe!="":
			if os.path.exists(vae_safe):
				pipe.vae=AutoencoderKL.from_single_file(vae_safe,torch_dtype=dtype)
			else:
				if win==None:
					print("error : the vae file doesn't exist.")
				else:
					win['RUN'].Update(disabled=False)
					win["info"].update("error : the vae file doesn't exist.")
				return

		paths=[lora1,lora2,lora3]
		lora_weights=[lora1w,lora2w,lora3w]
		for n in range(len(paths))
			if paths[n]!="":
				if os.path.exists(paths[n]):
					sd=load_file(paths[n])
					lora_check=False
					for k in sd:
						if k.endswith(".lora_up.weight") or k.endswith(".lora_B.weight"):
							lora_check=True
							break

					if lora_check:
						ukeys=[]
						for name, module in pipe.unet.named_modules():
							ukeys.append(name.replace(".","_"))
						t1keys=[]
						for name, module in pipe.text_encoder.named_modules():
							t1keys.append(name.replace(".","_"))
						t2keys=[]
						for name, module in pipe.text_encoder_2.named_modules():
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
								for k2 in unet_keys:
									if k2 in m2:
										m2=m2.replace(k2,unet_keys[k2])
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

						if msd=={}:
							if win==None:
								print("error : the lora"+str(n+1)+" file isn't supported.")
							else:
								win['RUN'].Update(disabled=False)
								win["info"].update("error : the lora"+str(n+1)+" file isn't supported.")
							return
						pipe.load_lora_weights(pretrained_model_name_or_path_or_dict=msd,torch_dtype=dtype)
						if lora1w=="":
							lora1w=1.0
						else:
							try:
								lora1w=float(lora_weights[n])
							except:
								lora1w=1.0
						pipe.fuse_lora(lora_scale=lora1w)
						pipe.unload_lora_weights()
						del msd,ukeys,t1keys,t2keys,sd
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
							if win==None:
								print("error : the lora"+str(n+1)+" file isn't supported.")
							else:
								win['RUN'].Update(disabled=False)
								win["info"].update("error : the lora"+str(n+1)+" file isn't supported.")
							return
						key_name=[]
						head=None
						for k in sd:
							for k2 in MODULE_type.weight_list_det:
								if k.endswith("."+k2):
									key_name.append(k.removesuffix("."+k2))
							if head==None and len(key_name)>0:
								if ("input_blocks" in key_name[-1]):
									head=key_name[-1].index("input_blocks")
								elif ("down_blocks" in key_name[-1]):
									head=key_name[-1].index("down_blocks")

						msd={}
						if head!=None:
							for k in key_name:
								m=k[head:].replace(".","_")
								for k2 in unet_keys:
									if k2 in m:
										m=m.replace(k2,unet_keys[k2])
								for k2 in MODULE_type.weight_list:
									if k+"."+k2 in sd:
										msd["lycoris_"+m+"."+k2]=sd[k+"."+k2]
						else:
							if win==None:
								print("error : the lora"+str(n+1)+" file isn't supported.")
							else:
								win['RUN'].Update(disabled=False)
								win["info"].update("error : the lora"+str(n+1)+" file isn't supported.")
							return

						if lora1w=="":
							lora1w=1.0
						else:
							try:
								lora1w=float(lora_weights[n])
							except:
								lora1w=1.0

						wrapper, _ = create_lycoris_from_weights(multiplier=lora1w,file="dummy.safetensors",module=pipe.unet, weights_sd=msd)
						wrapper.merge_to()
						del msd,sd
				else:
					if win==None:
						print("error : the lora"+str(n+1)+" file doesn't exist.")
					else:
						win['RUN'].Update(disabled=False)
						win["info"].update("error : the lora"+str(n+1)+" file doesn't exist.")
					return

		pipe.save_pretrained(os.getcwd()+"/dummy", safe_serialization=True)

		model_path=os.getcwd()+"/dummy"

		# Path for safetensors
		unet_path = os.path.join(model_path, "unet", "diffusion_pytorch_model.safetensors")
		vae_path = os.path.join(model_path, "vae", "diffusion_pytorch_model.safetensors")
		text_enc_path = os.path.join(model_path, "text_encoder", "model.safetensors")
		text_enc_2_path = os.path.join(model_path, "text_encoder_2", "model.safetensors")

		# Load models from safetensors if it exists, if it doesn't pytorch
		unet_state_dict = load_file(unet_path, device="cpu")

		vae_state_dict = load_file(vae_path, device="cpu")

		text_enc_dict = load_file(text_enc_path, device="cpu")

		text_enc_2_dict = load_file(text_enc_2_path, device="cpu")

		# Convert the UNet model
		unet_state_dict = convert_unet_state_dict(unet_state_dict)
		unet_state_dict = {"model.diffusion_model." + k: v for k, v in unet_state_dict.items()}

		# Convert the VAE model
		vae_state_dict = convert_vae_state_dict(vae_state_dict)
		vae_state_dict = {"first_stage_model." + k: v for k, v in vae_state_dict.items()}

		# Convert text encoder 1
		text_enc_dict = convert_openai_text_enc_state_dict(text_enc_dict)
		text_enc_dict = {"conditioner.embedders.0.transformer." + k: v for k, v in text_enc_dict.items()}

		# Convert text encoder 2
		text_enc_2_dict = convert_openclip_text_enc_state_dict(text_enc_2_dict)
		text_enc_2_dict = {"conditioner.embedders.1.model." + k: v for k, v in text_enc_2_dict.items()}
		# We call the `.T.contiguous()` to match what's done in
		# https://github.com/huggingface/diffusers/blob/84905ca7287876b925b6bf8e9bb92fec21c78764/src/diffusers/loaders/single_file_utils.py#L1085
		text_enc_2_dict["conditioner.embedders.1.model.text_projection"] = text_enc_2_dict.pop(
			"conditioner.embedders.1.model.text_projection.weight"
		).T.contiguous()

		# Put together new checkpoint
		state_dict = {**unet_state_dict, **vae_state_dict, **text_enc_dict, **text_enc_2_dict}
		state_dict2=load_file(base_safe)
		state_dict=zip_ckpt(state_dict,state_dict2)

		state_dict = {k: v.half() for k, v in state_dict.items()}

		save_file(state_dict, out_safe)
		shutil.rmtree(os.getcwd()+"/dummy")
		if win==None:
			print("fin : "+out_safe)
		else:
			win['RUN'].Update(disabled=False)
			win["info"].update("fin : "+out_safe)
	except:
		if win==None:
			print("error : fail in the output.")
		else:
			win['RUN'].Update(disabled=False)
			win["info"].update("error : fail in the output.")
	
if __name__=="__main__":
	import FreeSimpleGUI as sg
	import tkinter as tk
	import threading,pyperclip

	keys=[
		'ckpt','vae','lora1','lora2','lora3',"out",'w1','w2','w3'
	]
	grp_rclick_menu={}
	for key in keys:
		grp_rclick_menu[key]=[
			"",
			[
				"-copy-::"+key,"-cut-::"+key,"-paste-::"+key
			]
		]
	layout =[
		[sg.Text("checkpoint file")],
		[sg.InputText(key='ckpt',right_click_menu=grp_rclick_menu["ckpt"]),sg.FileBrowse('select ckpt', file_types=(('ckpt file', '.safetensors'),))],
		[sg.Text("vae file")],
		[sg.InputText(key='vae',right_click_menu=grp_rclick_menu["vae"]),sg.FileBrowse('select vae', file_types=(('vae file', '.safetensors'),))],
		[sg.Text("lora1 file")],
		[sg.InputText(key='lora1',right_click_menu=grp_rclick_menu["lora1"]),sg.FileBrowse('select lora', file_types=(('lora file', '.safetensors'),))],
		[sg.Text("weight"),sg.InputText("1.0",key='w1',right_click_menu=grp_rclick_menu["w1"])],
		[sg.Text("lora2 file")],
		[sg.InputText(key='lora2',right_click_menu=grp_rclick_menu["lora2"]),sg.FileBrowse('select lora', file_types=(('lora file', '.safetensors'),))],
		[sg.Text("weight"),sg.InputText("1.0",key='w2',right_click_menu=grp_rclick_menu["w2"])],
		[sg.Text("lora3 file")],
		[sg.InputText(key='lora3',right_click_menu=grp_rclick_menu["lora3"]),sg.FileBrowse('select lora', file_types=(('lora file', '.safetensors'),))],
		[sg.Text("weight"),sg.InputText("1.0",key='w3',right_click_menu=grp_rclick_menu["w3"])],
		[sg.Text("out file")],
		[sg.Input(key="out",right_click_menu=grp_rclick_menu["out"]),sg.FileSaveAs(file_types=(('ckpt file', '.safetensors'),))],
		[sg.Text("infomation",key="info")],
		[sg.Button('RUN'),sg.Button('EXIT')]
	]

	window = sg.Window('make safetensors', layout)

	while True:
		event, values = window.read()
		if event in (sg.WIN_CLOSED, 'EXIT'):
			break

		elif event=="RUN":
			base_safe=values["ckpt"]
			vae_safe=values["vae"]
			out_safe=values["out"]
			lora1=values["lora1"]
			lora2=values["lora2"]
			lora3=values["lora3"]
			lora1w=values["w1"]
			lora2w=values["w2"]
			lora3w=values["w3"]
			if base_safe!="" and out_safe!="":
				thread1 = threading.Thread(target=run,args=(base_safe,vae_safe,out_safe,lora1,lora2,lora3,lora1w,lora2w,lora3w,window))
				thread1.start()
					
		elif "-copy-" in event:
			try:
				key=event.replace("-copy-::","")
				selected = window[key].widget.selection_get()
				pyperclip.copy(selected)
			except:
				pass
		elif "-cut-" in event:
			try:
				key=event.replace("-cut-::","")
				selected = window[key].widget.selection_get()
				pyperclip.copy(selected)
				window[key].widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
			except:
				pass
		elif "-paste-" in event:
			try:
				key=event.replace("-paste-::","")
				selected = pyperclip.paste()
				insert_pos = window[key].widget.index("insert")
				window[key].Widget.insert(insert_pos, selected)
				window[key].widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
			except:
				pass
				
	window.close()
