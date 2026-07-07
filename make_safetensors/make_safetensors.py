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
import gc
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

#unet_keys
#sd - hf
unet_conversion_map={
	"time_embed.0.": "time_embedding.linear_1.",
	"time_embed.2.": "time_embedding.linear_2.",
	"input_blocks.0.0.": "conv_in.",
	"out.0.": "conv_norm_out.",
	"out.2.": "conv_out.",
	"label_emb.0.0.": "add_embedding.linear_1.",
	"label_emb.0.2.": "add_embedding.linear_2.",
}
unet_conversion_map_resnet={
	"in_layers.0": "norm1",
	"in_layers.2": "conv1",
	"out_layers.0": "norm2",
	"out_layers.3": "conv2",
	"emb_layers.1": "time_emb_proj",
	"skip_connection": "conv_shortcut",
}
unet_conversion_map_layer=[]
for i in range(3):
	for j in range(2):
		unet_conversion_map_layer+=[("input_blocks."+str(3 * i + j + 1)+".0.","down_blocks."+str(i)+".resnets."+str(j)+".")]
		if i > 0:
			unet_conversion_map_layer+=[("input_blocks."+str(3 * i + j + 1)+".1.","down_blocks."+str(i)+".attentions."+str(j)+".")]

	for j in range(4):
		unet_conversion_map_layer+=[("output_blocks."+str(3 * i + j)+".0.","up_blocks."+str(i)+".resnets."+str(j)+".")]
		if i < 2:
			unet_conversion_map_layer+=[("output_blocks."+str(3 * i + j)+".1.","up_blocks."+str(i)+".attentions."+str(j)+".")]

	if i < 3:
		unet_conversion_map_layer+=[("input_blocks."+str(3 * (i + 1))+".0.op.","down_blocks."+str(i)+".downsamplers.0.conv.")]

		if i==0:
			unet_conversion_map_layer+=[("output_blocks."+str(3 * i + 2)+".1.","up_blocks."+str(i)+".upsamplers.0.")]
		else:
			unet_conversion_map_layer+=[("output_blocks."+str(3 * i + 2)+".2.","up_blocks."+str(i)+".upsamplers.0.")]
unet_conversion_map_layer+=[("output_blocks.2.2.conv.","output_blocks.2.1.conv.")]

unet_conversion_map_layer+=[("middle_block.1.","mid_block.attentions.0.")]
for j in range(2):
	unet_conversion_map_layer+=[("middle_block."+str(2 * j)+".","mid_block.resnets."+str(j)+".")]

#vae_keys
#sd - hf
vae_conversion_map={
	"nin_shortcut": "conv_shortcut",
	"norm_out": "conv_norm_out",
	"mid.attn_1.": "mid_block.attentions.0.",
	"mid.block_1.":"mid_block.resnets.0.",
	"mid.block_2.":"mid_block.resnets.1.",
}
for i in range(4):
	for j in range(2):
		vae_conversion_map["encoder.down."+str(i)+".block."+str(j)+"."]="encoder.down_blocks."+str(i)+".resnets."+str(j)+"."

	if i < 3:
		vae_conversion_map["down."+str(i)+".downsample."]="down_blocks."+str(i)+".downsamplers.0."
		vae_conversion_map["up."+str(3 - i)+".upsample."]="up_blocks."+str(i)+".upsamplers.0."

	for j in range(3):
		vae_conversion_map["decoder.up."+str(3 - i)+".block."+str(j)+"."]="decoder.up_blocks."+str(i)+".resnets."+str(j)+"."
vae_conversion_map_attn={
	"norm.": "group_norm.",
	"q.": "to_q.",
	"k.": "to_k.",
	"v.": "to_v.",
	"proj_out.": "to_out.0.",
}

textenc_conversion_lst={
	"attn":"self_attn",
	"ln_1":"layer_norm1",
	"ln_2":"layer_norm2",
	".c_fc.":".fc1.",
	".c_proj.":".fc2.",
}

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

def run(base_safe,vae_safe,out_safe,lora1,lora2,lora3,lora1w,lora2w,lora3w,win=None):
	if win!=None:
		win["RUN"].Update(disabled=True)
	if not(os.path.exists(base_safe)):
		if win==None:
			print("error : the ckpt file doesn't exist.")
		else:
			win['RUN'].Update(disabled=False)
			win["info"].update("error : the ckpt file doesn't exist.")
		return

	if win==None:
		print("checking loras")
	else:
		win["info"].update("checking loras")
	paths=[lora1,lora2,lora3]
	lora_weights=[lora1w,lora2w,lora3w]
	datas=[]
	for n in range(len(paths)):
		if paths[n]!="":
			if os.path.exists(paths[n]):
				sd=load_file(paths[n])
				
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
				key_dict={}
				for k in sd:
					for k2 in MODULE_type.weight_list_det:
						if k.endswith("."+k2):
							k=k.removesuffix("."+k2)
							key_dict[k]=k.replace(".","_")

				usd={}
				t1sd={}
				t2sd={}
				for k in unet_conversion_map:
					m="lora_unet_"+k.removesuffix(".").replace(".","_")
					if m in key_dict.values():
						for k3 in key_dict:
							if key_dict[k3]==m:
								k4=k3
								break
						del key_dict[k4]
						for k2 in MODULE_type.weight_list:
							m2=k4+"."+k2
							if m2 in sd:
								usd["lycoris_"+unet_conversion_map[k].removesuffix(".").replace(".","_")+"."+k2]=sd.pop(m2)

				for k in key_dict:
					m=k.replace(".","_")
					if k.startswith("lora_unet_"):
						m=m.removeprefix("lora_unet_")
						m=m.replace("output_blocks_2_2_conv","up_blocks_0_upsamplers_0_conv")
						for k2 in unet_conversion_map_layer:
							k3=k2[0].removesuffix(".").replace(".","_")
							m=m.replace(k3,k2[1].removesuffix(".").replace(".","_"))
						if "resnets" in m:
							for k2 in unet_conversion_map_resnet:
								m=m.replace(k2.replace(".","_"),unet_conversion_map_resnet[k2])
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
				if usd=={} and t1sd=={} and t2sd=={}:
					if win==None:
						print("error : the lora"+str(n+1)+" file isn't supported.")
					else:
						win['RUN'].Update(disabled=False)
						win["info"].update("error : the lora"+str(n+1)+" file isn't supported.")
					return
				del sd

				if lora1w=="":
					lora1w=1.0
				else:
					try:
						lora1w=float(lora_weights[n])
					except:
						lora1w=1.0

				datas.append((lora1w,usd,t1sd,t2sd))
	
	sd2=load_file(base_safe)
	out_key=[]
	if os.path.exists(os.getcwd()+"/safe_temp"):
		shutil.rmtree(os.getcwd()+"/safe_temp")
	os.mkdir(os.getcwd()+"/safe_temp")

	pipe = StableDiffusionXLPipeline.from_single_file(
		base_safe,
		torch_dtype=torch.float32,
		cache_dir=os.getcwd()+"/pipecache",
		)
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
		print("making text_encoder")
	else:
		win["info"].update("making text_encoder")

	for w,_,sd,_ in datas:
		if sd!={}:
			wrapper, _ = create_lycoris_from_weights(multiplier=w,file="dummy.safetensors",module=pipe.text_encoder, weights_sd=sd)
			wrapper.merge_to()

	sd={}
	mapping={}
	for k,p in pipe.text_encoder.named_parameters():
		sd[k]=p.data
		mapping[k]=k
	del pipe.text_encoder
	gc.collect()

	for k in mapping:
		t1=sd.pop(k).to(torch.float32)
		if "conditioner.embedders.0.transformer." +mapping[k] in sd2:
			t2=sd2.pop("conditioner.embedders.0.transformer." +mapping[k]).to(torch.float32)
			sum1=torch.sum(torch.abs(t1)).item()
			sum2=torch.sum(torch.abs(t2)).item()
			n=not(math.isnan(sum1) or math.isnan(sum2))
			if n and sum1!=sum2:
				t1=t1*sum2/sum1
		out_sd={}
		out_sd["conditioner.embedders.0.transformer." +mapping[k]]=t1.to(torch.float16)
		save_file(out_sd,os.getcwd()+"/safe_temp/conditioner.embedders.0.transformer."+mapping[k]+".safetensors")
		out_key.append("conditioner.embedders.0.transformer." +mapping[k])
	
	if win==None:
		print("making text_encoder_2")
	else:
		win["info"].update("making text_encoder_2")

	for w,_,_,sd in datas:
		if sd!={}:
			wrapper, _ = create_lycoris_from_weights(multiplier=w,file="dummy.safetensors",module=pipe.text_encoder_2, weights_sd=sd)
			wrapper.merge_to()

	sd={}
	mapping={}
	for k,p in pipe.text_encoder_2.named_parameters():
		sd[k]=p.data
		mapping[k]=k
	del pipe.text_encoder_2
	gc.collect()

	for k,v in mapping.items():
		v=v.removeprefix("text_model.")
		v=v.replace("final_layer_norm","ln_final")
		v=v.replace("text_projection.weight","text_projection")
		v=v.replace("embeddings.position_embedding.weight","positional_embedding")
		v=v.replace("embeddings.token_embedding.weight","token_embedding.weight")
		if v.startswith("encoder.layers"):
			v=v.replace("encoder.layers.","transformer.resblocks.")
			for k2 in textenc_conversion_lst:
				v=v.replace(textenc_conversion_lst[k2],k2)
		mapping[k]=v

	for k in mapping:
		if mapping[k].endswith(".out_proj.weight"):
			t1=sd.pop(k).to(torch.float32)
			if "conditioner.embedders.1.model." +mapping[k] in sd2:
				t2=sd2.pop("conditioner.embedders.1.model." +mapping[k]).to(torch.float32)
				sum1=torch.sum(torch.abs(t1)).item()
				sum2=torch.sum(torch.abs(t2)).item()
				n=not(math.isnan(sum1) or math.isnan(sum2))
				if n and sum1!=sum2:
					t1=t1*sum2/sum1
			out_sd={}
			out_sd["conditioner.embedders.1.model." +mapping[k]]=t1.to(torch.float16)
			save_file(out_sd,os.getcwd()+"/safe_temp/conditioner.embedders.1.model."+mapping[k]+".safetensors")
			out_key.append("conditioner.embedders.1.model." +mapping[k])

			k2=mapping[k].removesuffix(".out_proj.weight")
			for k3 in mapping:
				if mapping[k3]==k2+".q_proj.weight":
					kq=k3
				elif mapping[k3]==k2+".k_proj.weight":
					kk=k3
				elif mapping[k3]==k2+".v_proj.weight":
					kv=k3
			q_weight=sd.pop(kq)
			k_weight=sd.pop(kk)
			v_weight=sd.pop(kv)
			k3="conditioner.embedders.1.model." +k2+".in_proj_weight"
			t1=torch.cat((q_weight,k_weight,v_weight)).to(torch.float32)
			if k3 in sd2:
				t2=sd2.pop(k3).to(torch.float32)
				sum1=torch.sum(torch.abs(t1)).item()
				sum2=torch.sum(torch.abs(t2)).item()
				n=not(math.isnan(sum1) or math.isnan(sum2))
				if n and sum1!=sum2:
					t1=t1*sum2/sum1
			out_sd={}
			out_sd[k3]=t1.to(torch.float16)
			save_file(out_sd,os.getcwd()+"/safe_temp/"+k3+".safetensors")
			out_key.append(k3)

			for k3 in mapping:
				if mapping[k3]==k2+".q_proj.bias":
					kq=k3
				elif mapping[k3]==k2+".k_proj.bias":
					kk=k3
				elif mapping[k3]==k2+".v_proj.bias":
					kv=k3
			q_bias=sd.pop(kq)
			k_bias=sd.pop(kk)
			v_bias=sd.pop(kv)
			k3="conditioner.embedders.1.model." +k2+".in_proj_bias"
			t1=torch.cat((q_bias,k_bias,v_bias)).to(torch.float32)
			if k3 in sd2:
				t2=sd2.pop(k3).to(torch.float32)
				sum1=torch.sum(torch.abs(t1)).item()
				sum2=torch.sum(torch.abs(t2)).item()
				n=not(math.isnan(sum1) or math.isnan(sum2))
				if n and sum1!=sum2:
					t1=t1*sum2/sum1
			out_sd={}
			out_sd[k3]=t1.to(torch.float16)
			save_file(out_sd,os.getcwd()+"/safe_temp/"+k3+".safetensors")
			out_key.append(k3)
			
		elif mapping[k].endswith(".q_proj.weight") or mapping[k].endswith(".k_proj.weight") or mapping[k].endswith(".v_proj.weight"):
			pass
		elif mapping[k].endswith(".q_proj.bias") or mapping[k].endswith(".k_proj.bias") or mapping[k].endswith(".v_proj.bias"):
			pass
		else:
			t1=sd.pop(k).to(torch.float32)
			if "conditioner.embedders.1.model." +mapping[k] in sd2:
				t2=sd2.pop("conditioner.embedders.1.model." +mapping[k]).to(torch.float32)
				sum1=torch.sum(torch.abs(t1)).item()
				sum2=torch.sum(torch.abs(t2)).item()
				n=not(math.isnan(sum1) or math.isnan(sum2))
				if n and sum1!=sum2:
					t1=t1*sum2/sum1
			out_sd={}
			out_sd["conditioner.embedders.1.model." +mapping[k]]=t1.to(torch.float16)
			save_file(out_sd,os.getcwd()+"/safe_temp/conditioner.embedders.1.model."+mapping[k]+".safetensors")
			out_key.append("conditioner.embedders.1.model." +mapping[k])
			
	if win==None:
		print("making vae")
	else:
		win["info"].update("making vae")
				
	sd={}
	mapping={}
	for k,p in pipe.vae.named_parameters():
		sd[k]=p.data
		mapping[k]=k
	del pipe.vae
	gc.collect()

	for k, v in mapping.items():
		for k2 in vae_conversion_map:
			v=v.replace(vae_conversion_map[k2],k2)
			mapping[k]=v
	for k,v in mapping.items():
		if "attentions" in k:
			for k2 in vae_conversion_map_attn:
				v = v.replace(vae_conversion_map_attn[k2], k2)
			mapping[k] = v
	for k in mapping:
		t1=sd.pop(k).to(torch.float32)
		for k2 in ["q", "k", "v", "proj_out"]:
			if "mid.attn_1."+k2+".weight" in k2:
				if t1.ndim != 1:
					t1=t1.reshape(*t1.shape,1,1)
		out_sd={}
		out_sd["first_stage_model." +mapping[k]]=t1.to(torch.float16)
		save_file(out_sd,os.getcwd()+"/safe_temp/first_stage_model."+mapping[k]+".safetensors")
		out_key.append("first_stage_model." +mapping[k])
		
	if win==None:
		print("making unet")
	else:
		win["info"].update("making unet")

	for w,sd,_,_ in datas:
		if sd!={}:
			wrapper, _ = create_lycoris_from_weights(multiplier=w,file="dummy.safetensors",module=pipe.unet, weights_sd=sd)
			wrapper.merge_to()

	sd={}
	mapping={}
	for k,p in pipe.unet.named_parameters():
		sd[k]=p.data
		mapping[k]=k
	del pipe.unet
	del pipe
	gc.collect()

	for k in unet_conversion_map:
		mapping[unet_conversion_map[k]+"weight"] = k+"weight"
		mapping[unet_conversion_map[k]+"bias"] = k+"bias"
	for k, v in mapping.items():
		if "resnets" in k:
			for k2 in unet_conversion_map_resnet:
				v = v.replace(unet_conversion_map_resnet[k2], k2)
			mapping[k] = v
	for k, v in mapping.items():
		for k2 in unet_conversion_map_layer:
			v = v.replace(k2[1], k2[0])
		mapping[k] = v
	for k in mapping:
		t1=sd.pop(k).to(torch.float32)
		if "model.diffusion_model."+mapping[k] in sd2:
			t2=sd2.pop("model.diffusion_model."+mapping[k]).to(torch.float32)
			sum1=torch.sum(torch.abs(t1)).item()
			sum2=torch.sum(torch.abs(t2)).item()
			n=not(math.isnan(sum1) or math.isnan(sum2))
			if n and sum1!=sum2:
				t1=t1*sum2/sum1
		out_sd={}
		out_sd["model.diffusion_model."+mapping[k]]=t1.to(torch.float16)
		save_file(out_sd,os.getcwd()+"/safe_temp/model.diffusion_model."+mapping[k]+".safetensors")
		out_key.append("model.diffusion_model."+mapping[k])
			
	if win==None:
		print("making output")
	else:
		win["info"].update("making output")

	save_ckpt(out_key,out_safe)
	
	shutil.rmtree(os.getcwd()+"/safe_temp")
	if win==None:
		print("fin : "+out_safe)
	else:
		win['RUN'].Update(disabled=False)
		win["info"].update("fin : "+out_safe)
	
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
