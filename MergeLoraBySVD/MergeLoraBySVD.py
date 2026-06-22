from safetensors.torch import (
	load_file,
	save_file
)
import itertools
import json
import os
import shutil
import re
import torch

ok=[
	"lora_unet_out_2",
	"lora_unet_input_blocks_0_0",
	"lora_unet_label_emb_0_0",
	"lora_unet_label_emb_0_2",
	"lora_unet_time_embed_0",
	"lora_unet_time_embed_2"
]

CLAMP_QUANTILE = 0.99

ACCEPTABLE = [12, 17, 20, 26]
SDXL_LAYER_NUM = [12, 20]

LAYER12 = {
	"BASE": True,
	"IN00": False,
	"IN01": False,
	"IN02": False,
	"IN03": False,
	"IN04": True,
	"IN05": True,
	"IN06": False,
	"IN07": True,
	"IN08": True,
	"IN09": False,
	"IN10": False,
	"IN11": False,
	"MID": True,
	"OUT00": True,
	"OUT01": True,
	"OUT02": True,
	"OUT03": True,
	"OUT04": True,
	"OUT05": True,
	"OUT06": False,
	"OUT07": False,
	"OUT08": False,
	"OUT09": False,
	"OUT10": False,
	"OUT11": False,
}

LAYER17 = {
	"BASE": True,
	"IN00": False,
	"IN01": True,
	"IN02": True,
	"IN03": False,
	"IN04": True,
	"IN05": True,
	"IN06": False,
	"IN07": True,
	"IN08": True,
	"IN09": False,
	"IN10": False,
	"IN11": False,
	"MID": True,
	"OUT00": False,
	"OUT01": False,
	"OUT02": False,
	"OUT03": True,
	"OUT04": True,
	"OUT05": True,
	"OUT06": True,
	"OUT07": True,
	"OUT08": True,
	"OUT09": True,
	"OUT10": True,
	"OUT11": True,
}

LAYER20 = {
	"BASE": True,
	"IN00": True,
	"IN01": True,
	"IN02": True,
	"IN03": True,
	"IN04": True,
	"IN05": True,
	"IN06": True,
	"IN07": True,
	"IN08": True,
	"IN09": False,
	"IN10": False,
	"IN11": False,
	"MID": True,
	"OUT00": True,
	"OUT01": True,
	"OUT02": True,
	"OUT03": True,
	"OUT04": True,
	"OUT05": True,
	"OUT06": True,
	"OUT07": True,
	"OUT08": True,
	"OUT09": False,
	"OUT10": False,
	"OUT11": False,
}

LAYER26 = {
	"BASE": True,
	"IN00": True,
	"IN01": True,
	"IN02": True,
	"IN03": True,
	"IN04": True,
	"IN05": True,
	"IN06": True,
	"IN07": True,
	"IN08": True,
	"IN09": True,
	"IN10": True,
	"IN11": True,
	"MID": True,
	"OUT00": True,
	"OUT01": True,
	"OUT02": True,
	"OUT03": True,
	"OUT04": True,
	"OUT05": True,
	"OUT06": True,
	"OUT07": True,
	"OUT08": True,
	"OUT09": True,
	"OUT10": True,
	"OUT11": True,
}

assert len([v for v in LAYER12.values() if v]) == 12
assert len([v for v in LAYER17.values() if v]) == 17
assert len([v for v in LAYER20.values() if v]) == 20
assert len([v for v in LAYER26.values() if v]) == 26

RE_UPDOWN = re.compile(r"(up|down)_blocks_(\d+)_(resnets|upsamplers|downsamplers|attentions)_(\d+)_")

def get_lbw_block_index(lora_name: str, is_sdxl: bool = False) -> int:
	# lbw block index is 0-based, but 0 for text encoder, so we return 0 for text encoder
	if "text_model_encoder_" in lora_name:  # LoRA for text encoder
		return 0

	# lbw block index is 1-based for U-Net, and no "input_blocks.0" in CompVis SD, so "input_blocks.1" have index 2
	block_idx = -1  # invalid lora name
	if not is_sdxl:
		NUM_OF_BLOCKS = 12  # up/down blocks
		m = RE_UPDOWN.search(lora_name)
		if m:
			g = m.groups()
			up_down = g[0]
			i = int(g[1])
			j = int(g[3])
			if up_down == "down":
				if g[2] == "resnets" or g[2] == "attentions":
					idx = 3 * i + j + 1
				elif g[2] == "downsamplers":
					idx = 3 * (i + 1)
				else:
					return block_idx  # invalid lora name
			elif up_down == "up":
				if g[2] == "resnets" or g[2] == "attentions":
					idx = 3 * i + j
				elif g[2] == "upsamplers":
					idx = 3 * i + 2
				else:
					return block_idx  # invalid lora name

			if g[0] == "down":
				block_idx = 1 + idx  # 1-based index, down block index
			elif g[0] == "up":
				block_idx = 1 + NUM_OF_BLOCKS + 1 + idx  # 1-based index, num blocks, mid block, up block index

		elif "mid_block_" in lora_name:
			block_idx = 1 + NUM_OF_BLOCKS  # 1-based index, num blocks, mid block
	else:
		# SDXL: some numbers are skipped
		if lora_name.startswith("lora_unet_"):
			name = lora_name[len("lora_unet_") :]
			if name.startswith("time_embed_") or name.startswith("label_emb_"):  # 1, No LoRA in sd-scripts
				block_idx = 1
			elif name.startswith("input_blocks_"):  # 1-8 to 2-9
				block_idx = 1 + int(name.split("_")[2])
			elif name.startswith("middle_block_"):  # 13
				block_idx = 13
			elif name.startswith("output_blocks_"):  # 0-8 to 14-22
				block_idx = 14 + int(name.split("_")[2])
			elif name.startswith("out_"):  # 23, No LoRA in sd-scripts
				block_idx = 23

	return block_idx

def format_lbws(lbws):
	try:
		# lbwは"[1,1,1,1,1,1,1,1,1,1,1,1]"のような文字列で与えられることを期待している
		lbws = [json.loads(lbw) for lbw in lbws]
	except Exception:
		raise ValueError(f"format of lbws are must be json / 層別適用率はJSON形式で書いてください")
	assert all(isinstance(lbw, list) for lbw in lbws), f"lbws are must be list / 層別適用率はリストにしてください"
	assert len(set(len(lbw) for lbw in lbws)) == 1, "all lbws should have the same length  / 層別適用率は同じ長さにしてください"
	assert all(
		len(lbw) in ACCEPTABLE for lbw in lbws
	), f"length of lbw are must be in {ACCEPTABLE} / 層別適用率の長さは{ACCEPTABLE}のいずれかにしてください"
	assert all(
		all(isinstance(weight, (int, float)) for weight in lbw) for lbw in lbws
	), f"values of lbs are must be numbers / 層別適用率の値はすべて数値にしてください"

	layer_num = len(lbws[0])
	is_sdxl = True if layer_num in SDXL_LAYER_NUM else False
	FLAGS = {
		"12": LAYER12.values(),
		"17": LAYER17.values(),
		"20": LAYER20.values(),
		"26": LAYER26.values(),
	}[str(layer_num)]
	LBW_TARGET_IDX = [i for i, flag in enumerate(FLAGS) if flag]
	return lbws, is_sdxl, LBW_TARGET_IDX

def merge_lora_models(models, ratios, lbws, new_rank, new_conv_rank, device, merge_dtype,win):
	merged_lora_sd = {}

	if lbws:
		lbws, is_sdxl, LBW_TARGET_IDX = format_lbws(lbws)
	else:
		is_sdxl = False
		LBW_TARGET_IDX = []

	sds=[]
	keys=[]
	lbw_weights=[]
	i=0
	for lora in models:
		sd=load_file(lora)
		
		for k in sd:
			if not(k.endswith(".lora_down.weight")):
				if k in ok:
					del sd[k]
				continue
			k=k.removesuffix(".lora_down.weight")
			lora_name = k.replace(".", "_")
			if lora_name.startswith("lora_unet"):
				if "down_blocks_1_attentions_0" in lora_name:
					lora_name=lora_name.replace("down_blocks_1_attentions_0","input_blocks_4_1")
				elif "down_blocks_1_attentions_1" in lora_name:
					lora_name=lora_name.replace("down_blocks_1_attentions_1","input_blocks_5_1")
				elif "down_blocks_2_attentions_0" in lora_name:
					lora_name=lora_name.replace("down_blocks_2_attentions_0","input_blocks_7_1")
				elif "down_blocks_2_attentions_1" in lora_name:
					lora_name=lora_name.replace("down_blocks_2_attentions_1","input_blocks_8_1")
				elif "up_blocks_0_attentions_0" in lora_name:
					lora_name=lora_name.replace("up_blocks_0_attentions_0","output_blocks_0_1")
				elif "up_blocks_0_attentions_1" in lora_name:
					lora_name=lora_name.replace("up_blocks_0_attentions_1","output_blocks_1_1")
				elif "up_blocks_0_attentions_2" in lora_name:
					lora_name=lora_name.replace("up_blocks_0_attentions_2","output_blocks_2_1")
				elif "up_blocks_1_attentions_0" in lora_name:
					lora_name=lora_name.replace("up_blocks_1_attentions_0","output_blocks_3_1")
				elif "up_blocks_1_attentions_1" in lora_name:
					lora_name=lora_name.replace("up_blocks_1_attentions_1","output_blocks_4_1")
				elif "up_blocks_1_attentions_2" in lora_name:
					lora_name=lora_name.replace("up_blocks_1_attentions_2","output_blocks_5_1")
				elif "mid_block_attentions_0" in lora_name:
					lora_name=lora_name.replace("mid_block_attentions_0","middle_block_1")
			for k2 in [".lora_down.weight",".lora_up.weight",".alpha"]:
				if k+k2 in sd:
					sd[lora_name+k2]=sd[k+k2]
					if k!=lora_name:
						del sd[k+k2]
	
		sds.append(sd)
		keys=keys+list(sd)
		if lbws==[]:
			lbw_weights.append(False)
		else:
			lbw=lbws[i]
			weights = [1] * 26
			for index, value in zip(LBW_TARGET_IDX, lbw):
				weights[index] = value
			lbw_weights.append(weights)
		i+=1

	keys=list(set(keys))
	key_sum=len(keys)
	key_count=0

	if win==None:
		print("svd")

	for k in keys:
		key_count=key_count+1
		if win!=None:
			win["info"].update("svd : "+str(key_count)+"/"+str(key_sum))
		else:
			print("\r"+str(key_count)+"/"+str(key_sum),end="")
		if not(k.endswith(".lora_down.weight")):
			continue

		mat=0
		for i in range(len(sds)):
			if not(k in sds[i]):
				continue
			wa=sds[i].pop(k)
			wb=sds[i].pop(k.replace(".lora_down.weight",".lora_up.weight"))

			network_dim = wa.size()[0]
			if k.replace(".lora_down.weight",".alpha") in sds[i]:
				alpha=sds[i].pop(k.replace(".lora_down.weight",".alpha"))
			else:
				alpha=network_dim
			in_dim = wa.size()[1]
			out_dim = wb.size()[0]
			conv2d = len(wa.size()) == 4
			kernel_size = None if not conv2d else wa.size()[2:4]
			scale = alpha / network_dim

			if lbw_weights[i]:
				index = get_lbw_block_index(k, is_sdxl)
				is_lbw_target = index in LBW_TARGET_IDX
				if is_lbw_target:
					scale *= lbw_weights[i][index]

			if type(mat) is int:
				mat = torch.zeros((out_dim, in_dim, *kernel_size) if conv2d else (out_dim, in_dim), dtype=merge_dtype)

			if device:
				mat = mat.to(device)
				wb = wb.to(device)
				wa = wa.to(device)
				scale = scale.to(device)
				
			if not conv2d:
				mat = mat + ratios[i] * (wb @ wa) * scale
			elif kernel_size == (1, 1):
				mat = (
					mat
					+ ratios[i]
					* (wb.squeeze(3).squeeze(2) @ wa.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
					* scale
				)
			else:
				conved = torch.nn.functional.conv2d(wa.permute(1, 0, 2, 3), wb).permute(1, 0, 2, 3)
				mat = mat + ratios[i] * conved * scale

		conv2d = len(mat.size()) == 4
		kernel_size = None if not conv2d else mat.size()[2:4]
		conv2d_3x3 = conv2d and kernel_size != (1, 1)
		out_dim, in_dim = mat.size()[0:2]

		if conv2d:
			if conv2d_3x3:
				mat = mat.flatten(start_dim=1)
			else:
				mat = mat.squeeze()

		module_new_rank = new_conv_rank if conv2d_3x3 else new_rank
		module_new_rank = min(module_new_rank, in_dim, out_dim)

		U, S, Vh = torch.linalg.svd(mat)

		U = U[:, :module_new_rank]
		S = S[:module_new_rank]
		U = U @ torch.diag(S)

		Vh = Vh[:module_new_rank, :]

		dist = torch.cat([U.flatten(), Vh.flatten()])
		hi_val = torch.quantile(dist, CLAMP_QUANTILE)
		low_val = -hi_val

		U = U.clamp(low_val, hi_val)
		Vh = Vh.clamp(low_val, hi_val)

		if conv2d:
			U = U.reshape(out_dim, module_new_rank, 1, 1)
			Vh = Vh.reshape(module_new_rank, in_dim, kernel_size[0], kernel_size[1])

		up_weight = U
		down_weight = Vh

		merged_lora_sd[k.replace(".lora_down.weight",".lora_up.weight")] = up_weight.to("cpu").contiguous()
		merged_lora_sd[k] = down_weight.to("cpu").contiguous()
		merged_lora_sd[k.replace(".lora_down.weight",".alpha")] = torch.tensor(module_new_rank, device="cpu")

	if win==None:
		print("")

	return merged_lora_sd

def merge(
	loras=[],
	weights=[],
	lbws = [],
	precision="float",
	save_precision="fp16",
	new_rank=16,
	new_conv_rank=None,
	device=None,
	save_to=None,
	win=None,
	meta_dict=None
):
	assert len(loras) == len(
		weights
	), f"number of models must be equal to number of ratios / モデルの数と重みの数は合わせてください"
	if lbws:
		assert len(loras) == len(
			lbws
		), f"number of models must be equal to number of ratios / モデルの数と層別適用率の数は合わせてください"
	else:
		lbws = []

	def str_to_dtype(p):
		if p == "float":
			return torch.float
		if p == "fp16":
			return torch.float16
		if p == "bf16":
			return torch.bfloat16
		return None

	merge_dtype = str_to_dtype(precision)
	save_dtype = str_to_dtype(save_precision)
	if save_dtype is None:
		save_dtype = merge_dtype

	new_conv_rank = new_conv_rank if new_conv_rank is not None else new_rank
	state_dict = merge_lora_models(
		loras, weights, lbws, new_rank, new_conv_rank, device, merge_dtype,win
	)

	for key in list(state_dict):
		value = state_dict[key]
		if type(value) == torch.Tensor and value.dtype.is_floating_point and value.dtype != save_dtype:
			state_dict[key] = value.to(save_dtype)

	if isinstance(meta_dict, dict):
		save_file(state_dict,save_to,metadata=meta_dict)
	else:
		save_file(state_dict,save_to)
	
	if win==None:
		print("fin")

def get_function(vs,win):
	win['RUN'].Update(disabled=True)
	loras=[]
	ws=[]
	ids=[]
	for i in range(4):
		if vs["ckpt"+str(i+1)]!="":
			if vs["ckpt"+str(i+1)].endswith(".safetensors"):
				if not(os.path.exists(vs["ckpt"+str(i+1)])):
					win['RUN'].Update(disabled=False)
					win["info"].update("error : lora"+str(i+1)+" file does not exist.")
					return
				loras.append(vs["ckpt"+str(i+1)])
				try:
					if vs["w"+str(i+1)]=="":
						ws.append(1.0)
						win["w"+str(i+1)].update("1.0")
					else:
						ws.append(float(vs["w"+str(i+1)]))
						win["w"+str(i+1)].update(str(float(vs["w"+str(i+1)])))
				except:
					ws.append(1.0)
					win["w"+str(i+1)].update("1.0")
				if vs["id"+str(i+1)]!="":
					try:
						ids.append(int(vs["id"+str(i+1)]))
						win["id"+str(i+1)].update(str(int(vs["id"+str(i+1)])))
					except:
						win["id"+str(i+1)].update("")
			else:
				win['RUN'].Update(disabled=False)
				win["info"].update("You need to select the safetensors file for lora"+str(i+1)+" file.")
				return
	
	if vs["out"].endswith(".safetensors"):
		out_path=vs["out"]
	else:
		win['RUN'].Update(disabled=False)
		win["info"].update("You need to select the safetensors file for output file.")
		return
	
	try:
		if vs["d"]=="":
			dim=16
			win["d"].update("16")
		else:
			dim=int(vs["d"])
			win["d"].update(str(int(vs["d"])))
	except:
		dim=16
		win["d"].update("16")
		
	try:
		meta_dict={}
		meta_dict["id"]=str(ids).replace("[","").replace("]","").replace(" ","")
		meta_dict["weight"]=str(ws).replace("[","").replace("]","").replace(" ","")
		merge(loras=loras,weights=ws,save_to=out_path,new_rank=dim,win=win,meta_dict=meta_dict)
		
		if vs["dof"]:
			for i in range(4):
				if vs["ckpt"+str(i+1)]!="":
					os.remove(vs["ckpt"+str(i+1)])
		win['RUN'].Update(disabled=False)
		win["out"].update(out_path)
		win["info"].update("fin")
	except:
		win['RUN'].Update(disabled=False)
		win["info"].update("I failed in the output.")

if __name__=="__main__":
	import threading
	import tkinter as tk
	import pyperclip
	import FreeSimpleGUI as sg

	sg.theme('GrayGrayGray')
	grp_rclick_menu={}
	keys=["ckpt1","ckpt2","ckpt3","ckpt4","w1","w2","w3","w4","id1","id2","id3","id4","out","d"]
	for key in keys:
		grp_rclick_menu[key]=[
			"",
			[
				"-copy-::"+key,"-cut-::"+key,"-paste-::"+key
			]
		]		
			
	layout=[
		[sg.Text("lora1"), sg.Input(key="ckpt1",right_click_menu=grp_rclick_menu["ckpt1"]),sg.FileBrowse( file_types=(('lora file', '.safetensors'),)),sg.Button('clear', key='clear1')],
		[sg.Text("weight"),sg.Input("0.7",key="w1",right_click_menu=grp_rclick_menu["w1"], size=(10, 1)),sg.Text("id"),sg.Input("",key="id1",right_click_menu=grp_rclick_menu["id1"], size=(20, 1))],
		[sg.Text("lora2"), sg.Input(key="ckpt2",right_click_menu=grp_rclick_menu["ckpt2"]),sg.FileBrowse( file_types=(('lora file', '.safetensors'),)),sg.Button('clear', key='clear2')],
		[sg.Text("weight"),sg.Input("0.7",key="w2",right_click_menu=grp_rclick_menu["w2"], size=(10, 1)),sg.Text("id"),sg.Input("",key="id2",right_click_menu=grp_rclick_menu["id2"], size=(20, 1))],
		[sg.Text("lora3"), sg.Input(key="ckpt3",right_click_menu=grp_rclick_menu["ckpt3"]),sg.FileBrowse( file_types=(('lora file', '.safetensors'),)),sg.Button('clear', key='clear3')],
		[sg.Text("weight"),sg.Input("0.7",key="w3",right_click_menu=grp_rclick_menu["w3"], size=(10, 1)),sg.Text("id"),sg.Input("",key="id3",right_click_menu=grp_rclick_menu["id3"], size=(20, 1))],
		[sg.Text("lora4"), sg.Input(key="ckpt4",right_click_menu=grp_rclick_menu["ckpt4"]),sg.FileBrowse( file_types=(('lora file', '.safetensors'),)),sg.Button('clear', key='clear4')],
		[sg.Text("weight"),sg.Input("0.7",key="w4",right_click_menu=grp_rclick_menu["w4"], size=(10, 1)),sg.Text("id"),sg.Input("",key="id4",right_click_menu=grp_rclick_menu["id4"], size=(20, 1))],
		[sg.Text("dim"), sg.Input("16",key="d",right_click_menu=grp_rclick_menu["d"])],
		[sg.Text("output path"), sg.Input(key="out",right_click_menu=grp_rclick_menu["out"]),sg.FileSaveAs(file_types=(('lora file', '.safetensors'),)),sg.Button('clear', key='clear_out')],
		[sg.Checkbox('del original files', key='dof')],
		[sg.Text("infomation",key="info")],
		[sg.Button('RUN', key='RUN'),sg.Button('EXIT', key='EXIT')]
	]

	window = sg.Window('merge lora', layout,keep_on_top=True)

	while True:
		event, values = window.read()
		if event == sg.WINDOW_CLOSED:
			break
		elif event=="EXIT":
			break
		elif event=="RUN":
			c1=not(values["ckpt1"]=="" and values["ckpt2"]=="" and values["ckpt3"]=="" and values["ckpt4"]=="")
			if c1:
				if values["out"]=="":
					names=[]
					for i in range(4):
						if values["ckpt"+str(i+1)]!="":
							names.append(values["ckpt"+str(i+1)])
					out_path=""
					for line in names:
						if out_path=="":
							out_path=os.path.dirname(line)+"/"+os.path.basename(line).split(".")[0]
						else:
							out_path=out_path+"_"+os.path.basename(line).split(".")[0]
					out_path=out_path+".safetensors"
					values["out"]=out_path
				else:
					out_path=values["out"]
				ok = sg.popup_ok_cancel(out_path,title='output file',keep_on_top=True)
				if ok=="OK":
					thread1 = threading.Thread(target=get_function,args=(values,window))
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
		elif "clear" in event:
			try:
				if event=="clear_out":
					key="out"
				else:
					key="ckpt"+event.replace("clear","")
					key2="id"+event.replace("clear","")
					window[key2].update("")
				window[key].update("")
			except:
				pass

	window.close()
	