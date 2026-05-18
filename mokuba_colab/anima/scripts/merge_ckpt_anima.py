import os
import shutil
import json
import numpy
import gc
from safetensors.torch import (
	save_file,
	load_file
)
import torch
from diffusers_anima import AnimaPipeline

trans_key1={
	"ff.net.0.proj.weight":"mlp.layer1.weight",
	"ff.net.2.weight":"mlp.layer2.weight",
	"norm1.linear_1.weight":"adaln_modulation_self_attn.1.weight",
	"norm1.linear_2.weight":"adaln_modulation_self_attn.2.weight",
	"attn1.norm_q.weight":"self_attn.q_norm.weight",
	"attn1.norm_k.weight":"self_attn.k_norm.weight",
	"attn1.to_q.weight":"self_attn.q_proj.weight",
	"attn1.to_k.weight":"self_attn.k_proj.weight",
	"attn1.to_v.weight":"self_attn.v_proj.weight",
	"attn1.to_out.0.weight":"self_attn.output_proj.weight",
	"norm2.linear_1.weight":"adaln_modulation_cross_attn.1.weight",
	"norm2.linear_2.weight":"adaln_modulation_cross_attn.2.weight",
	"attn2.norm_q.weight":"cross_attn.q_norm.weight",
	"attn2.norm_k.weight":"cross_attn.k_norm.weight",
	"attn2.to_q.weight":"cross_attn.q_proj.weight",
	"attn2.to_k.weight":"cross_attn.k_proj.weight",
	"attn2.to_v.weight":"cross_attn.v_proj.weight",
	"attn2.to_out.0.weight":"cross_attn.output_proj.weight",
	"norm3.linear_1.weight":"adaln_modulation_mlp.1.weight",
	"norm3.linear_2.weight":"adaln_modulation_mlp.2.weight"
}
trans_key2={
	"patch_embed.proj.weight":"x_embedder.proj.1.weight",
	"time_embed.t_embedder.linear_1.weight":"t_embedder.1.linear_1.weight",
	"time_embed.t_embedder.linear_2.weight":"t_embedder.1.linear_2.weight",
	"time_embed.norm.weight":"t_embedding_norm.weight",
	"norm_out.linear_1.weight":"final_layer.adaln_modulation.1.weight",
	"norm_out.linear_2.weight":"final_layer.adaln_modulation.2.weight",
	"proj_out.weight":"final_layer.linear.weight"
}

vae_key1={
	"conv1":"quant_conv",
	"conv2":"post_quant_conv",
}
vae_key2={
	"downsamples":"down_blocks",
	"residual.2":"conv1",
	"residual.6":"conv2",
	"residual.0":"norm1",
	"residual.3":"norm2",
	"shortcut":"conv_shortcut",
	"middle.1":"mid_block.attentions.0",
	"middle.0":"mid_block.resnets.0",
	"middle.2":"mid_block.resnets.1",
	"conv1":"conv_in",
	"head.0":"norm_out",
	"head.2":"conv_out",
}
vae_key3={
	"residual.2":"conv1",
	"residual.6":"conv2",
	"residual.0":"norm1",
	"residual.3":"norm2",
	"middle.1":"mid_block.attentions.0",
	"middle.0":"mid_block.resnets.0",
	"middle.2":"mid_block.resnets.1",
	"upsamples.3":"up_blocks.0.upsamplers.0",
	"upsamples.7":"up_blocks.1.upsamplers.0",
	"upsamples.11":"up_blocks.2.upsamplers.0",
	"upsamples.0":"up_blocks.0.resnets.0",
	"upsamples.10":"up_blocks.2.resnets.2",
	"upsamples.12":"up_blocks.3.resnets.0",
	"upsamples.13":"up_blocks.3.resnets.1",
	"upsamples.14":"up_blocks.3.resnets.2",
	"upsamples.1":"up_blocks.0.resnets.1",
	"upsamples.2":"up_blocks.0.resnets.2",
	"upsamples.4":"up_blocks.1.resnets.0",
	"shortcut":"conv_shortcut",
	"upsamples.5":"up_blocks.1.resnets.1",
	"upsamples.6":"up_blocks.1.resnets.2",
	"upsamples.8":"up_blocks.2.resnets.0",
	"upsamples.9":"up_blocks.2.resnets.1",
	"conv1":"conv_in",
	"head.0":"norm_out",
	"head.2":"conv_out",
}

check="final_layer.linear.weight"
pass_keys=[
	"model.diffusion_model.pos_embedder.dim_spatial_range",
	"model.diffusion_model.pos_embedder.dim_temporal_range",
	"model.diffusion_model.pos_embedder.seq"
]

def spckpt(path,i):
	pipe=AnimaPipeline.from_single_file(path,cache_dir=os.getcwd()+"/pipecache")
	keys=[]
	for k,p in getattr(pipe, "transformer").named_parameters():
		osd={}
		if k.startswith("core."):
			k=k.replace("core.","")
			if k.startswith("transformer_"):
				k=k.replace("transformer_","")
				for key in trans_key1:
					if k.endswith(key):
						k=k.replace(key,trans_key1[key])
			else:
				k=trans_key2[k]
		k="model.diffusion_model."+k
		osd[k]=p.data.to(torch.float32)
		save_file(osd,os.getcwd()+"/safe_temp/"+str(i)+"_"+k+".safetensors")
		keys.append(k)

	for k,p in getattr(pipe, "vae").named_parameters():
		osd={}
		if k.startswith("encoder."):
			for key in vae_key2:
				if vae_key2[key] in k:
					k=k.replace(vae_key2[key],key)
		elif k.startswith("decoder."):
			for key in vae_key3:
				if vae_key3[key] in k:
					k=k.replace(vae_key3[key],key)
		else:
			for key in vae_key1:
				if vae_key1[key] in k:
					k=k.replace(vae_key1[key],key)
		k="first_stage_model."+k
		osd[k]=p.data.to(torch.float32)
		save_file(osd,os.getcwd()+"/safe_temp/"+str(i)+"_"+k+".safetensors")
		keys.append(k)
		
	for k,p in getattr(pipe, "text_encoder").named_parameters():
		osd={}
		k="cond_stage_model.qwen3_06b.transformer.model."+k
		osd[k]=p.data.to(torch.float32)
		save_file(osd,os.getcwd()+"/safe_temp/"+str(i)+"_"+k+".safetensors")
		keys.append(k)
	
	return keys

def del_safe(k):
	if os.path.exists(os.getcwd()+"/safe_temp/1_"+k+".safetensors"):
		os.remove(os.getcwd()+"/safe_temp/1_"+k+".safetensors")
	if os.path.exists(os.getcwd()+"/safe_temp/2_"+k+".safetensors"):
		os.remove(os.getcwd()+"/safe_temp/2_"+k+".safetensors")
	gc.collect()

def mergeckpt(ckpts,ws,out_path,mode="normal",win=None):
	if win!=None:
		win["RUN"].Update(disabled=True)

	if not(out_path.endswith(".safetensors")):
		if win==None:
			print("the output path is needed to be a safetensors file.")
		else:
			win["RUN"].Update(disabled=False)
			win["info"].update("error")
			notification.notify(title="error",message="the output path is needed to be a safetensors file.")
		return

	for path in ckpts:
		if not(os.path.exists(path)):
			if win==None:
				print(path+" does not exist.")
			else:
				win["RUN"].Update(disabled=False)
				win["info"].update("error")
				notification.notify(title="error",message=path+" does not exist.")
			return

	if not(mode in ["normal","tensor1","tensor2"]):
		mode="normal"

	try:
		if os.path.exists(os.getcwd()+"/safe_temp"):
			shutil.rmtree(os.getcwd()+"/safe_temp")
		os.mkdir(os.getcwd()+"/safe_temp")
	
		c1=spckpt(ckpts[0],1)
		c2=spckpt(ckpts[1],2)
	
		data_dict=list(set(c1+c2))
		dict_sum=len(data_dict)
		key_count=0
	
		with torch.no_grad():
			for k in data_dict:
				key_count=key_count+1
				if win!=None:
					win["info"].update("merging "+str(key_count)+"/"+str(dict_sum))
				else:
					print("\rmerging "+str(key_count)+"/"+str(dict_sum),end="")
	
				out_dict={}
				if os.path.exists(os.getcwd()+"/safe_temp/1_"+k+".safetensors"):
					t1=load_file(os.getcwd()+"/safe_temp/1_"+k+".safetensors")[k].to(torch.float32)
				else:
					t1=(load_file(os.getcwd()+"/safe_temp/2_"+k+".safetensors")[k]*0).to(torch.float32)
	
				if os.path.exists(os.getcwd()+"/safe_temp/2_"+k+".safetensors"):
					t2=load_file(os.getcwd()+"/safe_temp/2_"+k+".safetensors")[k].to(torch.float32)
				else:
					t2=(load_file(os.getcwd()+"/safe_temp/1_"+k+".safetensors")[k]*0).to(torch.float32)
	
				if k.startswith("model.diffusion_model.blocks."):
					ind=int(k.split(".")[3])
					w=ws[ind+1]
				elif k.startswith("model.diffusion_model.llm_adapter."):
					w=ws[29]
				else:
					w=ws[0]
	
				if mode=="normal":
					out_dict[k]=((1-w)*t1+w*t2).to(torch.float16)
	
				elif "tensor" in mode:
					w1=(1-w)/2
					w2=w
					w1=round(t1.size()[0]*w1)
					w2=round(t1.size()[0]*(w1+w2))
					if w1==0:
						out_dict[k]=t2.to(torch.float16)
						save_file(out_dict,os.getcwd()+"/safe_temp/"+k+".safetensors")
						del w,out_dict,t1,t2,w1,w1
						del_safe(k)
						continue
					elif w2==0:
						out_dict[k]=t1.to(torch.float16)
						save_file(out_dict,os.getcwd()+"/safe_temp/"+k+".safetensors")
						del w,out_dict,t1,t2,w1,w1
						del_safe(k)
						continue
					if mode=="tensor1":
						if t1.dim()==1:
							t1[w1:w2]=t2[w1:w2]
						elif t1.dim()==2:
							t1[w1:w2,:]=t2[w1:w2,:]
						elif t1.dim()==3:
							t1[w1:w2,:,:]=t2[w1:w2,:,:]
						elif t1.dim()==4:
							t1[w1:w2,:,:,:]=t2[w1:w2,:,:,:]
						elif t1.dim()==5:
							t1[w1:w2,:,:,:,:]=t2[w1:w2,:,:,:,:]
					else:
						if t1.dim()==1:
							t1[w1:w2]=t2[w1:w2]
						elif t1.dim()==2:
							t1[:,w1:w2]=t2[:,w1:w2]
						elif t1.dim()==3:
							t1[:,w1:w2,:]=t2[:,w1:w2,:]
						elif t1.dim()==4:
							t1[:,w1:w2,:,:]=t2[:,w1:w2,:,:]
						elif t1.dim()==5:
							t1[:,w1:w2,:,:,:]=t2[:,w1:w2,:,:,:]
					out_dict[k]=t1.to(torch.float16)
					del w1,w2
	
				save_file(out_dict,os.getcwd()+"/safe_temp/"+k+".safetensors")
				del w,out_dict,t1,t2
				del_safe(k)
	
		if win==None:
			print("")
	
		if win==None:
			print("making output")
		else:
			win["info"].update("making output")
		out_dict={}
		out_dict["__metadata__"]={"format":"pt"}
		n=0
		for k in data_dict:
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
	
		output=open(out_path,"wb")
		out_dict=str(out_dict).replace("'",'"')
		out_dict=out_dict.encode()
		l=len(out_dict).to_bytes(8,byteorder="little")
		output.write(l)
		output.write(out_dict)
	
		dict_sum=len(data_dict)
		key_count=0
		for k in data_dict:
			key_count=key_count+1
			if win==None:
				print("\r"+str(key_count)+"/"+str(dict_sum),end="")
			else:
				win["info"].update("making output "+str(key_count)+"/"+str(dict_sum))
			f=open(os.getcwd()+"/safe_temp/"+k+".safetensors","rb")
			l=int.from_bytes(f.read(8),byteorder="little")
			head=f.read(l)
			output.write(f.read())
			f.close()
			os.remove(os.getcwd()+"/safe_temp/"+k+".safetensors")
		output.close()
			
		f=open(out_path.replace(".safetensors",".txt"),"w")
		for i in range(len(ckpts)):
			f.write("ckpt"+str(i+1)+" : "+ckpts[i]+"\n")
		f.write("weight : "+str(ws)+"\n")
		f.close()
		shutil.rmtree(os.getcwd()+"/safe_temp")
		del out_dict,l,head,n,offsets
		gc.collect()
		if win==None:
			print("")
			print(out_path)
		else:
			win["RUN"].Update(disabled=False)
			win["info"].update("fin")
			notification.notify(title="fin",message=out_path)

	except:
		if os.path.exists(os.getcwd()+"/safe_temp"):
			shutil.rmtree(os.getcwd()+"/safe_temp")
		if win==None:
			print("I failed in the output.")
		else:
			win["RUN"].Update(disabled=False)
			win["info"].update("error")
			notification.notify(title="error",message="I failed in the output.")

if __name__=="__main__":
	import FreeSimpleGUI as sg
	import tkinter as tk
	import pyperclip
	from plyer import notification
	import threading

	sg.theme('TealMono')
	  
	keys=["ckpt1","ckpt2","out"]
	for i in range(31):
		keys.append("w"+str(i))

	grp_rclick_menu={}
	for key in keys:
		grp_rclick_menu[key]=[
			"",
			[
				"-copy-::"+key,"-cut-::"+key,"-paste-::"+key
			]
		]

	lay=[
		[sg.Text("ckpt1"), sg.Input(key="ckpt1",right_click_menu=grp_rclick_menu["ckpt1"]),sg.FileBrowse( file_types=(('ckpt file', '.safetensors'),))],
		[sg.Text("ckpt2"), sg.Input(key="ckpt2",right_click_menu=grp_rclick_menu["ckpt2"]),sg.FileBrowse( file_types=(('ckpt file', '.safetensors'),))],
		[sg.Checkbox('BLOCK', key='block',default=False,enable_events=True)],
		[sg.Text("weight of ckpt2",key="w_text")],
		[sg.Input(key="w0",right_click_menu=grp_rclick_menu["w0"])],
		[
			sg.Frame("BASE",[[sg.Input(key="w1",right_click_menu=grp_rclick_menu["w1"], size=(10, 1))]],key="base"),
			sg.Frame("BLOCK00",[[sg.Input(key="w2",right_click_menu=grp_rclick_menu["w2"], size=(10, 1))]],key="b0"),
			sg.Frame("BLOCK01",[[sg.Input(key="w3",right_click_menu=grp_rclick_menu["w3"], size=(10, 1))]],key="b1"),
			sg.Frame("BLOCK02",[[sg.Input(key="w4",right_click_menu=grp_rclick_menu["w4"], size=(10, 1))]],key="b2"),
			sg.Frame("BLOCK03",[[sg.Input(key="w5",right_click_menu=grp_rclick_menu["w5"], size=(10, 1))]],key="b3"),
			sg.Frame("BLOCK04",[[sg.Input(key="w6",right_click_menu=grp_rclick_menu["w6"], size=(10, 1))]],key="b4"),
			sg.Frame("BLOCK05",[[sg.Input(key="w7",right_click_menu=grp_rclick_menu["w7"], size=(10, 1))]],key="b5"),
			sg.Frame("BLOCK06",[[sg.Input(key="w8",right_click_menu=grp_rclick_menu["w8"], size=(10, 1))]],key="b6"),
			sg.Frame("BLOCK07",[[sg.Input(key="w9",right_click_menu=grp_rclick_menu["w9"], size=(10, 1))]],key="b7"),
			sg.Frame("BLOCK08",[[sg.Input(key="w10",right_click_menu=grp_rclick_menu["w10"], size=(10, 1))]],key="b8"),
		],
		[
			sg.Frame("BLOCK09",[[sg.Input(key="w11",right_click_menu=grp_rclick_menu["w11"], size=(10, 1))]],key="b9"),
			sg.Frame("BLOCK10",[[sg.Input(key="w12",right_click_menu=grp_rclick_menu["w12"], size=(10, 1))]],key="b10"),
			sg.Frame("BLOCK11",[[sg.Input(key="w13",right_click_menu=grp_rclick_menu["w13"], size=(10, 1))]],key="b11"),
			sg.Frame("BLOCK12",[[sg.Input(key="w14",right_click_menu=grp_rclick_menu["w14"], size=(10, 1))]],key="b12"),
			sg.Frame("BLOCK13",[[sg.Input(key="w15",right_click_menu=grp_rclick_menu["w15"], size=(10, 1))]],key="b13"),
			sg.Frame("BLOCK14",[[sg.Input(key="w16",right_click_menu=grp_rclick_menu["w16"], size=(10, 1))]],key="b14"),
			sg.Frame("BLOCK15",[[sg.Input(key="w17",right_click_menu=grp_rclick_menu["w17"], size=(10, 1))]],key="b15"),
			sg.Frame("BLOCK16",[[sg.Input(key="w18",right_click_menu=grp_rclick_menu["w18"], size=(10, 1))]],key="b16"),
			sg.Frame("BLOCK17",[[sg.Input(key="w19",right_click_menu=grp_rclick_menu["w19"], size=(10, 1))]],key="b17"),
			sg.Frame("BLOCK18",[[sg.Input(key="w20",right_click_menu=grp_rclick_menu["w20"], size=(10, 1))]],key="b18"),
		],
		[
			sg.Frame("BLOCK19",[[sg.Input(key="w21",right_click_menu=grp_rclick_menu["w21"], size=(10, 1))]],key="b19"),
			sg.Frame("BLOCK20",[[sg.Input(key="w22",right_click_menu=grp_rclick_menu["w22"], size=(10, 1))]],key="b20"),
			sg.Frame("BLOCK21",[[sg.Input(key="w23",right_click_menu=grp_rclick_menu["w23"], size=(10, 1))]],key="b21"),
			sg.Frame("BLOCK22",[[sg.Input(key="w24",right_click_menu=grp_rclick_menu["w24"], size=(10, 1))]],key="b22"),
			sg.Frame("BLOCK23",[[sg.Input(key="w25",right_click_menu=grp_rclick_menu["w25"], size=(10, 1))]],key="b23"),
			sg.Frame("BLOCK24",[[sg.Input(key="w26",right_click_menu=grp_rclick_menu["w26"], size=(10, 1))]],key="b24"),
			sg.Frame("BLOCK25",[[sg.Input(key="w27",right_click_menu=grp_rclick_menu["w27"], size=(10, 1))]],key="b25"),
			sg.Frame("BLOCK26",[[sg.Input(key="w28",right_click_menu=grp_rclick_menu["w28"], size=(10, 1))]],key="b26"),
			sg.Frame("BLOCK27",[[sg.Input(key="w29",right_click_menu=grp_rclick_menu["w29"], size=(10, 1))]],key="b27"),
			sg.Frame("LLM",[[sg.Input(key="w30",right_click_menu=grp_rclick_menu["w30"], size=(10, 1))]],key="llm"),
		],
		[
			sg.Radio('NORMAL', key='normal',default=True,group_id='destination'),
			sg.Radio('TENSOR1', key='tensor1',default=False,group_id='destination'),
			sg.Radio('TENSOR2', key='tensor2',default=False,group_id='destination')
		],
		[sg.Text("output path"), sg.Input(key="out",right_click_menu=grp_rclick_menu["out"]),sg.FileSaveAs(file_types=(('ckpt file', '.safetensors'),))],
		[sg.Text("infomation",key="info")],
		[sg.Button('RUN', key='RUN'),sg.Button('EXIT', key='EXIT')]
	]

	window = sg.Window('Merge Ckpt Anima', lay)

	def lay_che(b,win):
		win["ckpt1"].hide_row()
		win["ckpt2"].hide_row()
		win["block"].hide_row()
		win["w_text"].hide_row()
		win["w0"].hide_row()
		win["base"].hide_row()
		win["b9"].hide_row()
		win["b19"].hide_row()
		win["normal"].hide_row()
		win["out"].hide_row()
		win["info"].hide_row()
		win["RUN"].hide_row()
		win["ckpt1"].unhide_row()
		win["ckpt2"].unhide_row()
		win["block"].unhide_row()
		win["w_text"].unhide_row()
		if b:
			win["base"].unhide_row()
			win["b9"].unhide_row()
			win["b19"].unhide_row()
		else:
			win["w0"].unhide_row()
		win["normal"].unhide_row()
		win["out"].unhide_row()
		win["info"].unhide_row()
		win["RUN"].unhide_row()

	event, values = window.read(timeout=0)
	lay_che(False,window)

	while True:
		event, values = window.read()
			
		if event == sg.WINDOW_CLOSED:
			break
		elif event=="EXIT":
			break
		elif event=="RUN":
			if values["out"]!="" and values["ckpt1"]!="" and values["ckpt2"]!="":
				ckpts=[values["ckpt1"],values["ckpt2"]]
				out_path=values["out"]
				if values["block"]:
					weights=[]
					weight=0.5
					for i in range(30):
						try:
							weights.append(float(values["w"+str(i+1)]))
						except:
							weights.append(weight)
						weight=weights[-1]
						window["w"+str(i+1)].update(str(weight))
				else:
					try:
						weight=float(values["w0"])
					except:
						weight=0.5
					window["w0"].update(str(weight))
					weights=[]
					for i in range(30):
						weights.append(weight)

				if values["normal"]:
					mode="normal"
				elif values["tensor1"]:
					mode="tensor1"
				else:
					mode="tensor2"

				thread1 = threading.Thread(target=mergeckpt,args=(ckpts,weights,out_path,mode,window))
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
		elif event=="block":
			lay_che(values["block"],window)

	window.close()
