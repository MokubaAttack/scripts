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
import json
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

unet_keys1={
	"middle_block.0":"mid_block.resnets.0",
	"middle_block.1":"mid_block.attentions.0",
	"middle_block.2":"mid_block.resnets.1",
	"skip_connection": "conv_shortcut",
	"output_blocks.0.0":"up_blocks.0.resnets.0",
	"output_blocks.0.1":"up_blocks.0.attentions.0",
	"output_blocks.1.0":"up_blocks.0.resnets.1",
	"output_blocks.1.1":"up_blocks.0.attentions.1",
	"output_blocks.2.0":"up_blocks.0.resnets.2",
	"output_blocks.2.1":"up_blocks.0.attentions.2",
	"output_blocks.2.2":"up_blocks.0.upsamplers.0",
	"output_blocks.3.0":"up_blocks.1.resnets.0",
	"output_blocks.3.1":"up_blocks.1.attentions.0",
	"output_blocks.4.0":"up_blocks.1.resnets.1",
	"output_blocks.4.1":"up_blocks.1.attentions.1",
	"output_blocks.5.0":"up_blocks.1.resnets.2",
	"output_blocks.5.1":"up_blocks.1.attentions.2",
	"output_blocks.5.2":"up_blocks.1.upsamplers.0",
	"output_blocks.6.0":"up_blocks.2.resnets.0",
	"output_blocks.7.0":"up_blocks.2.resnets.1",
	"output_blocks.8.0":"up_blocks.2.resnets.2",
	"input_blocks.1.0":"down_blocks.0.resnets.0",
	"input_blocks.2.0":"down_blocks.0.resnets.1",
	"input_blocks.3.0.op":"down_blocks.0.downsamplers.0.conv",
	"input_blocks.4.0":"down_blocks.1.resnets.0",
	"input_blocks.4.1":"down_blocks.1.attentions.0",
	"input_blocks.5.0":"down_blocks.1.resnets.1",
	"input_blocks.5.1":"down_blocks.1.attentions.1",
	"input_blocks.6.0.op":"down_blocks.1.downsamplers.0.conv",
	"input_blocks.7.0":"down_blocks.2.resnets.0",
	"input_blocks.7.1":"down_blocks.2.attentions.0",
	"input_blocks.8.0":"down_blocks.2.resnets.1",
	"input_blocks.8.1":"down_blocks.2.attentions.1",
}
unet_keys2={
	"time_embedding.linear_1":"time_embed.0",
	"time_embedding.linear_2":"time_embed.2",
	"conv_in":"input_blocks.0.0",
	"conv_norm_out":"out.0",
	"conv_out":"out.2",
	"add_embedding.linear_1":"label_emb.0.0",
	"add_embedding.linear_2":"label_emb.0.2",
}
unet_keys3={
	"in_layers.0": "norm1",
	"in_layers.2": "conv1",
	"out_layers.0": "norm2",
	"out_layers.3": "conv2",
	"emb_layers.1": "time_emb_proj",
}
change_keys=[
	"input_blocks.1.0",
	"input_blocks.2.0",
	"input_blocks.4.0",
	"input_blocks.5.0",
	"input_blocks.7.0",
	"input_blocks.8.0",
	"middle_block.0",
	"middle_block.2",
	"output_blocks.0.0",
	"output_blocks.1.0",
	"output_blocks.2.0",
	"output_blocks.3.0",
	"output_blocks.4.0",
	"output_blocks.5.0",
	"output_blocks.6.0",
	"output_blocks.7.0",
	"output_blocks.8.0",
]

text2_keys={
	"self_attn":"attn",
	"layer_norm1":"ln_1",
	"layer_norm2":"ln_2",
	".fc1.":".c_fc.",
	".fc2.":".c_proj.",
}

vae_keys={
	"conv_norm_out":"norm_out",
	"mid_block.attentions.0.":"mid.attn_1.",
	"mid_block.resnets.0.":"mid.block_1.",
	"mid_block.resnets.1.":"mid.block_2.",
	"group_norm.":"norm.",
	"to_q.":"q.",
	"to_k.":"k.",
	"to_v.":"v.",
	"to_out.0.":"proj_out.",
}

if not(os.path.exists(os.getcwd()+"/pipecache")):
	os.mkdir(os.getcwd()+"/pipecache")

def save_ckpt(keys,path):
	out_dict={}
	out_dict["__metadata__"]={"format":"pt"}
	n=0
	for k in keys:
		f=open(os.getcwd()+"/safe_temp/"+k+".safetensors","rb")
		l=int.from_bytes(f.read(8),byteorder="little")
		head=f.read(l).decode()
		head=json.loads(head)
		out_dict[k]=head[k]
		offsets=out_dict[k]["data_offsets"][1]
		out_dict[k]["data_offsets"][0]=n
		n=n+offsets
		out_dict[k]["data_offsets"][1]=n
		f.close()

	output=open(path,"wb")
	out_dict=str(out_dict).replace("'",'"')
	out_dict=out_dict.encode()
	l=len(out_dict).to_bytes(8,byteorder="little")
	output.write(l)
	output.write(out_dict)

	for k in keys:
		f=open(os.getcwd()+"/safe_temp/"+k+".safetensors","rb")
		l=int.from_bytes(f.read(8),byteorder="little")
		head=f.read(l)
		output.write(f.read())
		f.close()
		os.remove(os.getcwd()+"/safe_temp/"+k+".safetensors")
	output.close()
	
def zip_ckpt(ckpt1,ckpt2):
	keys=[]
	for k in ckpt1:
		if k in ckpt2 and not("first_stage_model." in k):
			sum1=torch.sum(torch.abs(ckpt1[k].to(torch.float32))).item()
			sum2=torch.sum(torch.abs(ckpt2[k].to(torch.float32))).item()
			n=not(math.isnan(sum1) or math.isnan(sum2))
			if n and sum1!=sum2:
				ckpt1[k]=ckpt1[k]*sum2/sum1
		sd={}
		sd[k]=ckpt1[k].to(torch.float16)
		save_file(sd,os.getcwd()+"/safe_temp/"+k+".safetensors")
		keys.append(k)
	return keys

def run(base_safe,vae_safe,out_safe,lora1,lora2,lora3,lora1w,lora2w,lora3w,win=None):
	if win!=None:
		win["RUN"].Update(disabled=True)
	try:
		if not(os.path.exists(base_safe)):
			if win==None:
				print("error : the ckpt file doesn't exist.")
			else:
				win['RUN'].Update(disabled=False)
				win["info"].update("error : the ckpt file doesn't exist.")
			return
		if win==None:
			print("making pipeline")
		else:
			win["info"].update("making pipeline")
		pipe = StableDiffusionXLPipeline.from_single_file(base_safe, torch_dtype=torch.float32,cache_dir=os.getcwd()+"/pipecache")
				
		if vae_safe!="":
			if os.path.exists(vae_safe):
				pipe.vae=AutoencoderKL.from_single_file(vae_safe)
			else:
				if win==None:
					print("error : the vae file doesn't exist.")
				else:
					win['RUN'].Update(disabled=False)
					win["info"].update("error : the vae file doesn't exist.")
				return

		if win==None:
			print("merging loras")
		else:
			win["info"].update("merging loras")
		paths=[lora1,lora2,lora3]
		lora_weights=[lora1w,lora2w,lora3w]
		for n in range(len(paths)):
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

		if win==None:
			print("making output")
		else:
			win["info"].update("making output")
		sd={}
		for k,p in getattr(pipe, "text_encoder").named_parameters():
			sd["conditioner.embedders.0.transformer."+k]=p.data

		sd2={}
		for k,p in getattr(pipe, "text_encoder_2").named_parameters():
			k=k.removeprefix("text_model.")
			if k.startswith("final_layer_norm"):
				k=k.replace("final_layer_norm","ln_final")
			elif k.startswith("encoder.layers"):
				k=k.replace("encoder.layers.","transformer.resblocks.")
				for k2 in text2_keys:
					if k2 in k:
						k=k.replace(k2,text2_keys[k2])
			if k=="text_projection.weight":
				k="text_projection"
			elif k=="embeddings.position_embedding.weight":
				k="positional_embedding"
			elif k=="embeddings.token_embedding.weight":
				k="token_embedding.weight"
			sd2["conditioner.embedders.1.model."+k]=p.data
		sd2_keys=list(sd2)
		for k in sd2_keys:
			if k.endswith(".out_proj.weight"):
				sd[k]=sd2.pop(k)

				k2=k.removesuffix(".out_proj.weight")

				q_weight=sd2.pop(k2+".q_proj.weight")
				k_weight=sd2.pop(k2+".k_proj.weight")
				v_weight=sd2.pop(k2+".v_proj.weight")
				sd[k2+".in_proj_weight"]=torch.cat((q_weight,k_weight,v_weight)).to(torch.float16)

				q_bias=sd2.pop(k2+".q_proj.bias")
				k_bias=sd2.pop(k2+".k_proj.bias")
				v_bias=sd2.pop(k2+".v_proj.bias")
				sd[k2+".in_proj_bias"]=torch.cat((q_bias,k_bias,v_bias)).to(torch.float16)
			elif k.endswith(".q_proj.weight") or k.endswith(".k_proj.weight") or k.endswith(".v_proj.weight"):
				pass
			elif k.endswith(".q_proj.bias") or k.endswith(".k_proj.bias") or k.endswith(".v_proj.bias"):
				pass
			else:
				sd[k]=sd2.pop(k)

		for k,p in getattr(pipe, "vae").named_parameters():
			if k.startswith("encoder.down_blocks."):
				if "downsamplers" in k:
					m=re.match(r"encoder\.down_blocks\.([0-9]+)\.downsamplers\.([0-9]+)\.(\S+)",k)
					m1=m.group(1)
					m3=m.group(3)
					k="first_stage_model.encoder.down."+m1+".downsample."+m3
				else:
					m=re.match(r"encoder\.down_blocks\.([0-9]+)\.resnets\.([0-9]+)\.(\S+)",k)
					m1=m.group(1)
					m2=m.group(2)
					m3=m.group(3)
					if "conv_shortcut" in m3:
						m3=m3.replace("conv_shortcut","nin_shortcut")
					k="first_stage_model.encoder.down."+m1+".block."+m2+"."+m3
			elif k.startswith("decoder.up_blocks."):
				if "upsamplers" in k:
					m=re.match(r"decoder\.up_blocks\.([0-9]+)\.upsamplers\.([0-9]+)\.(\S+)",k)
					m1=str(3-int(m.group(1)))
					m3=m.group(3)
					k="first_stage_model.decoder.up."+m1+".upsample."+m3
				else:
					m=re.match(r"decoder\.up_blocks\.([0-9]+)\.resnets\.([0-9]+)\.(\S+)",k)
					m1=str(3-int(m.group(1)))
					m2=m.group(2)
					m3=m.group(3)
					if "conv_shortcut" in m3:
						m3=m3.replace("conv_shortcut","nin_shortcut")
					k="first_stage_model.decoder.up."+m1+".block."+m2+"."+m3
			else:
				for k2 in vae_keys:
					if k2 in k:
						k=k.replace(k2,vae_keys[k2])
				k="first_stage_model."+k
			sd[k]=p.data
			
		for k,p in getattr(pipe, "unet").named_parameters():
			for k2 in unet_keys1:
				if unet_keys1[k2] in k:
					k=k.replace(unet_keys1[k2],k2)
					if k2 in change_keys:
						for k3 in unet_keys3:
							if unet_keys3[k3] in k:
								k=k.replace(unet_keys3[k3],k3)
			for k2 in unet_keys2:
				if k2 in k:
					k=k.replace(k2,unet_keys2[k2])
			k="model.diffusion_model."+k
			sd[k]=p.data
	except:
		if win==None:
			print("error : fail in the output.")
		else:
			win['RUN'].Update(disabled=False)
			win["info"].update("error : fail in the output.")

	try:
		if os.path.exists(os.getcwd()+"/safe_temp"):
			shutil.rmtree(os.getcwd()+"/safe_temp")
		os.mkdir(os.getcwd()+"/safe_temp")
		sd2=load_file(base_safe)
		keys=zip_ckpt(sd,sd2)
		save_ckpt(keys,out_safe)
		
		shutil.rmtree(os.getcwd()+"/safe_temp")
		if win==None:
			print("fin : "+out_safe)
		else:
			win['RUN'].Update(disabled=False)
			win["info"].update("fin : "+out_safe)
	except:
		shutil.rmtree(os.getcwd()+"/safe_temp")
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
