from PIL import (
	Image,
	PngImagePlugin
)
import shutil
import piexif
import piexif.helper

keys=[
	"input","pr","ne","st","sa","cf","se","cl","ckpt","lora1","lora2","lora3","lora4","embed1","embed2","embed3","embed4","w1","w2","w3","w4","vae","hu","hs","hum","ds","sc","tu","tum","up","cont","ccs"
]

def plus(vs,win=None):
	path=vs["input"]
	if win!=None:
		win["info"].print("write "+path, end="\n")
	else:
		print(path)
	pr=vs["pr"]
	ne=vs["ne"]
	st=vs["st"]
	sa=vs["sa"]
	sc=vs["sc"]
	cf=vs["cf"]
	se=vs["se"]
	cl=vs["cl"]
	hu=vs["hu"]
	hs=vs["hs"]
	hum=vs["hum"]
	ds=vs["ds"]
	ckpt=vs["ckpt"]
	vae=vs["vae"]
	tu=vs["tu"]
	tum=vs["tum"]
	up=vs["up"]
	ccs=vs["ccs"]
	cont=vs["cont"]
	loras=vs["lora"]
	ws=vs["w"]
	embeds=vs["embed"]
	if len(loras)>len(ws):
		if win==None:
			print("The number of weights is not enough.")
		else:
			win["info"].print("The number of weights is not enough.", end="\n")
		return 0
	for i in range(len(loras)):
		loras[i]=str(loras[i])
		ws[i]=str(ws[i])
	for i in range(len(embeds)):
		embeds[i]=str(embeds[i])
	try:
		metadata=pr+"\n"
		metadata=metadata+"Negative prompt: "+ne+"\n"
		if st!="":
			metadata=metadata+"Steps: "+st+", " 
		if sa!="":
			if sc=="":
				metadata=metadata+"Sampler: "+sa+", "
			else:
				metadata=metadata+"Sampler: "+sa+" "+sc+", "
		else:
			metadata=metadata+"Sampler: Undefined, "
		if cf!="":
			metadata=metadata+"CFG scale: "+cf+", "
		if se!="":
			metadata=metadata+"Seed: "+se+", "
		if cl!="":
			metadata=metadata+"Clip skip: "+cl+", "
		if ds!="":
			metadata=metadata+"Denoising strength: "+ds+", "
		if hu!="":
			metadata=metadata+"Hires upscale: "+hu+", "
		if hs!="":
			metadata=metadata+"Hires steps: "+hs+", "
		if hum!="":
			metadata=metadata+"Hires upscaler: "+hum+", "

		if tu!="":
			metadata=metadata+"Tile scale: "+tu+", "
		if tum!="":
			metadata=metadata+"Tile scaler: "+tum+", "
		if ccs!="":
			metadata=metadata+"controlnet_conditioning_scale: "+ccs+", "

		metadata=metadata+'Civitai resources: ['
		if ckpt!="":
			metadata=metadata+'{"type":"checkpoint","modelVersionId":'+ckpt+"}"
		for i in range(len(loras)):
			metadata=metadata+',{"type":"lora","weight":'+ws[i]+',"modelVersionId":'+loras[i]+"}"
		for i in range(len(embeds)):
			metadata=metadata+',{"type":"embed","modelVersionId":'+embeds[i]+"}"
		if vae!="":
			metadata=metadata+',{"type":"ae","modelVersionId":'+vae+"}"
		if cont!="":
			metadata=metadata+',{"type":"controlnet","modelVersionId":'+cont+"}"
		if up!="":
			metadata=metadata+',{"type":"upscaler","modelVersionId":'+up+"}"
		metadata=metadata+']'

		if "[," in metadata:
			metadata=metadata.replace("[,","[")
			
		if path.endswith(".png") or path.endswith(".PNG"):
			if path.endswith(".png"):
				output_path=path.replace(".png","_meta.png")
			else:
				output_path=path.replace(".PNG","_meta.png")
		elif path.endswith(".jpg") or path.endswith(".JPG"):
			if path.endswith(".jpg"):
				output_path=path.replace(".jpg","_meta.jpg")
			else:
				output_path=path.replace(".JPG","_meta.jpg")
		else:
			if win==None:
				print("You need to select a png file or a jpg file.")
			else:
				win["info"].print("You need to select a png file or a jpg file.", end="\n")
			return 0
		image = Image.open(path)
		if path.endswith(".png") or path.endswith(".PNG"):
			pnginfo = PngImagePlugin.PngInfo()
			pnginfo.add_text("parameters", metadata)
			image.save(output_path, "PNG", pnginfo=pnginfo)
		else:
			exif_data=piexif.helper.UserComment.dump(metadata, encoding="unicode")
			exif_dict={
				'Exif':{
					piexif.ExifIFD.UserComment:exif_data,
				}
			}
			exif_bytes = piexif.dump(exif_dict)
			image.save(output_path,"JPEG",quality = 85, exif=exif_bytes)
		if win==None:
			print("fin")
		else:
			win["info"].print("fin", end="\n")
		return 1
	
	except:
		if win==None:
			print("error")
		else:
			win["info"].print("error", end="\n")
		return 0

def run(vs,win):
	win["RUN"].Update(disabled=True)
	if ";" in vs["input"]:
		paths=vs["input"].split(";")
	else:
		paths=[vs["input"]]
	
	loras=[]
	ws=[]
	for i in range(4):
		if vs["lora"+str(i+1)]!="":
			loras.append(vs["lora"+str(i+1)])
			if vs["w"+str(i+1)]!="":
				ws.append(vs["w"+str(i+1)])
			else:
				ws.append(str(1.0))
				win["w"+str(i+1)].update(str(1.0))
	vs["lora"]=loras
	vs["w"]=ws

	embeds=[]
	for i in range(4):
		if vs["embed"+str(i+1)]!="":
			embeds.append(vs["embed"+str(i+1)])
	vs["embed"]=embeds
		
	for path in paths:
		vs["input"]=path
		c=plus(vs,win)
		
		if vs["dof"] and c==1:
			os.remove(path)
	win["input"].update("")
	win["info"].print("fin", end="\n")
	win['RUN'].Update(disabled=False)

def save_data_as(vs):
	try:
		result = sg.popup_get_file("select pkl file",title="SAVE",file_types=(('pkl file', '.pkl'),),save_as = True)
	except:
		result=None
	if result!=None:
		state_dict={}
		for l in keys:
			if l!="input":
				state_dict[l]=vs[l]
		f=open(result, 'wb')
		pickle.dump(state_dict, f)
		del state_dict
		f.close()
		f=open("setting.ini","w")
		f.write(result)
		f.close()

def save_data(vs):
	if os.path.exists("setting.ini"):
		f=open("setting.ini","r")
		result=f.read()
		f.close()
		if os.path.exists(result):
			state_dict={}
			for l in keys:
				if l!="input":
					state_dict[l]=vs[l]
			f=open(result, 'wb')
			pickle.dump(state_dict, f)
			del state_dict
			f.close()
		else:
			save_data_as(vs)
	else:
		save_data_as(vs)

def load_data(win):
	try:
		result = sg.popup_get_file("select pkl file",title="LOAD",file_types=(('pkl file', '.pkl'),))
	except:
		result=None
	if result!=None:
		if os.path.exists(result):
			f=open(result, 'rb')
			state_dict= pickle.load(f)
			for l in keys:
				if l!="input":
					try:
						win[l].update(state_dict[l])
					except:
						win[l].update("")
			del state_dict
			f.close()
			f=open("setting.ini","w")
			f.write(result)
			f.close()

def read_meta(path,win):
	win["info"].print("read "+path, end="\n")
	for l in keys:
		if l!="input":
			win[l].update("")
	try:
		img=Image.open(path)
		if path.endswith(".jpg"):
			exif_data=img._getexif()
			exif_data=exif_data[37510].decode()
		if path.endswith(".png"):
			exif_data = img.info['parameters']

		exif_data=exif_data.split("\n")
		exif_data2=str(exif_data.pop(-1).encode())
		k=None
		for i in range(len(exif_data)):
			if "\x00" in exif_data[i]:
				exif_data[i]=exif_data[i].replace("\x00","")
			if exif_data[i].startswith("Negative prompt: "):
				k=i
		if k==None:
			pro="".join(exif_data)
			neg=""
		else:
			pro="".join(exif_data[:k])
			neg="".join(exif_data[k:])
			neg=neg.replace("Negative prompt: ","")
		pro=pro.removeprefix("UNICODE")
		win["pr"].update(pro)
		win["ne"].update(neg)
		inds={
			"Steps:":"st",
			"CFG scale:":"cf",
			"Seed:":"se",
			"Clip skip:":"cl",
			"Denoising strength:":"ds",
			"Hires upscale:":"hu",
			"Hires steps:":"hs",
			"Hires upscaler:":"hum",
			"Tile upscale:":"tu",
			"Tile upscaler:":"tum",
			"controlnet_conditioning_scale:":"ccs"
		}
		ss=["Karras","beta","exponential","sgm_uniform","simple","uniform","normal"]
		exif_data=exif_data2.replace(r"\x00","").removesuffix("'").removeprefix("b'").split(", ")
		for line in exif_data:
			for ind in inds:
				if line.startswith(ind):
					line2=line.split(": ")
					win[inds[ind]].update(line2[1])
			if line.startswith("Sampler:"):
				line2=line.split(": ")
				win["sa"].update(line2[1])
				for ind in ss:
					if ind in line2[1]:
						win["sc"].update(ind)
						win["sa"].update(line2[1].replace(" "+ind,""))
			if line.startswith("Civitai resources:"):
				line2=line.replace("Civitai resources: ","")
				line2 = ast.literal_eval(line2)
				k1=1
				k2=1
				for sd in line2:
					if sd["type"]=="checkpoint":
						win["ckpt"].update(str(sd["modelVersionId"]))
					elif sd["type"]=="lora":
						win["lora"+str(k1)].update(str(sd["modelVersionId"]))
						win["w"+str(k1)].update(str(sd["weight"]))
						k1=k1+1
						if k1>4:
							k1=4
							win["info"].print("warning : There is more number of loras than 4.")
					elif sd["type"]=="embed":
						win["embed"+str(k2)].update(str(sd["modelVersionId"]))
						k2=k2+1
						if k2>4:
							k2=4
							win["info"].print("warning : There is more number of embeds than 4.")
					elif sd["type"]=="controlnet":
						win["cont"].update(str(sd["modelVersionId"]))
					elif sd["type"]=="upscaler":
						win["up"].update(str(sd["modelVersionId"]))
					else:
						win["vae"].update(str(sd["modelVersionId"]))
		win["info"].print("fin", end="\n")
	except:
		win["info"].print("error", end="\n")

if __name__=="__main__":
	import FreeSimpleGUI as sg
	import pickle,os,threading,pyperclip,ast
	import tkinter as tk

	sg.theme('GrayGrayGray')

	sa_list=[
		"Euler a",
		"Euler",
		"LMS",
		"Heun",
		"DPM2",
		"DPM2 a",
		"DPM++"
		"DPM++ 2S a", #
		"DPM++ 2M",
		"DPM++ SDE",
		"DPM++ 2M SDE",
		"DPM++ 3M SDE",
		"DPM fast", #
		"DPM adaptive", #
		"DDIM",
		"PLMS",
		"UniPC",
		"LCM",
		"flowmatch_euler",
		"euler",
		"euler_a_rf",
		"euler_ancestral_rf"
	]
	sc_list=[
		"","Karras","beta","exponential","sgm_uniform","simple","uniform","normal"
	]
	hum_list=["NEAREST","BOX","BILINEAR","HAMMING","BICUBIC","LANCZOS",""]
	ivs={}
	ivs["pr"]="prompt"
	ivs["ne"]="negative prompt"
	ivs["st"]="30"
	ivs["sa"]="DDIM"
	ivs["sc"]=""
	ivs["cf"]="7"
	ivs["se"]=""
	ivs["cl"]="2"
	ivs["ckpt"]=""
	ivs["lora1"]=""
	ivs["lora2"]=""
	ivs["lora3"]=""
	ivs["lora4"]=""
	ivs["embed1"]=""
	ivs["embed2"]=""
	ivs["embed3"]=""
	ivs["embed4"]=""
	ivs["vae"]=""
	ivs["w1"]=""
	ivs["w2"]=""
	ivs["w3"]=""
	ivs["w4"]=""
	ivs["hu"]=""
	ivs["hs"]=""
	ivs["hum"]=""
	ivs["ds"]=""
	ivs["tu"]=""
	ivs["tum"]=""
	ivs["ccs"]=""
	ivs["up"]=""
	ivs["cont"]=""

	if os.path.exists("setting.ini"):
		f=open("setting.ini","r")
		result=f.read()
		f.close()
		if os.path.exists(result):
			f=open(result, 'rb')
			state_dict= pickle.load(f)
			for l in keys:
				if l!="input":
					try:
						ivs[l]=state_dict[l]
					except:
						ivs[l]=""
			del state_dict
			f.close()

	grp_rclick_menu={}
	for key in keys:
		if key=="sa" or key=="sc" or key=="hum" or key=="tum":
			continue
		grp_rclick_menu[key]=[
			"",
			[
				"-copy-::"+key,"-cut-::"+key,"-paste-::"+key
			]
		] 
		
	col1=[
		[sg.Text("prompt")],
		[sg.Multiline(ivs["pr"], size=(50, 5),key="pr",right_click_menu=grp_rclick_menu["pr"])]
	]
	col2=[
		[sg.Text("negative prompt")],
		[sg.Multiline(ivs["ne"], size=(50, 5),key="ne",right_click_menu=grp_rclick_menu["ne"])]
	]
	col3=[
		[sg.Text("Steps"), sg.Input(ivs["st"],key="st",right_click_menu=grp_rclick_menu["st"], size=(10, 1))],
		[sg.Text("Sampler"), sg.Combo(default_value=ivs["sa"],values=sa_list,key="sa")],
		[sg.Text("Schedule type"), sg.Combo(default_value=ivs["sc"],values=sc_list,key="sc")],
		[sg.Text("CFG scale"), sg.Input(ivs["cf"],key="cf",right_click_menu=grp_rclick_menu["cf"], size=(10, 1))],
		[sg.Text("Seed"), sg.Input(ivs["se"],key="se",right_click_menu=grp_rclick_menu["se"], size=(20, 1))],
		[sg.Text("Clip skip"), sg.Input(ivs["cl"],key="cl",right_click_menu=grp_rclick_menu["cl"], size=(10, 1))],
	]
	col4=[
		[sg.Text("Denoising strength"), sg.Input(ivs["ds"],key="ds",right_click_menu=grp_rclick_menu["ds"], size=(10, 1))],
		[sg.Text("Hires upscale"), sg.Input(ivs["hu"],key="hu",right_click_menu=grp_rclick_menu["hu"], size=(10, 1))],
		[sg.Text("Hires steps"), sg.Input(ivs["hs"],key="hs",right_click_menu=grp_rclick_menu["hs"], size=(10, 1))],
		[sg.Text("Hires upscaler"), sg.Combo(default_value=ivs["hum"],key="hum",values=hum_list)],
		[sg.Text("Tile upscale"), sg.Input(ivs["tu"],key="tu",right_click_menu=grp_rclick_menu["tu"], size=(10, 1))],
		[sg.Text("Tile upscaler"), sg.Combo(default_value=ivs["tum"],key="tum",values=hum_list)],
		[sg.Text("controlnet_conditioning_scale"), sg.Input(ivs["ccs"],key="ccs",right_click_menu=grp_rclick_menu["ccs"], size=(10, 1))],
	]
		
	layout=[
		[
			sg.Text("input"),
			sg.Input(key="input",right_click_menu=grp_rclick_menu["input"]),sg.FilesBrowse(file_types=(('image file', '.png'),('image file', '.jpg'))),
			sg.Button('READ', key='READ'),
			sg.Checkbox('del original files', key='dof')
		],
		[sg.Column(col1),sg.Column(col2)],
		[sg.Column(col3),sg.Column(col4)],
		[sg.Text("ckpt modelVersionId"), sg.Input(ivs["ckpt"],key="ckpt",right_click_menu=grp_rclick_menu["ckpt"], size=(20, 1)),sg.Text("vae modelVersionId"), sg.Input(ivs["vae"],key="vae",right_click_menu=grp_rclick_menu["vae"], size=(20, 1))],
		[sg.Text("lora1 modelVersionId"), sg.Input(ivs["lora1"],key="lora1",right_click_menu=grp_rclick_menu["lora1"], size=(20, 1)),sg.Text("weight"), sg.Input(ivs["w1"],key="w1",right_click_menu=grp_rclick_menu["w1"], size=(10, 1))],
		[sg.Text("lora2 modelVersionId"), sg.Input(ivs["lora2"],key="lora2",right_click_menu=grp_rclick_menu["lora2"], size=(20, 1)),sg.Text("weight"), sg.Input(ivs["w2"],key="w2",right_click_menu=grp_rclick_menu["w2"], size=(10, 1))],
		[sg.Text("lora3 modelVersionId"), sg.Input(ivs["lora3"],key="lora3",right_click_menu=grp_rclick_menu["lora3"], size=(20, 1)),sg.Text("weight"), sg.Input(ivs["w3"],key="w3",right_click_menu=grp_rclick_menu["w3"], size=(10, 1))],
		[sg.Text("lora4 modelVersionId"), sg.Input(ivs["lora4"],key="lora4",right_click_menu=grp_rclick_menu["lora4"], size=(20, 1)),sg.Text("weight"), sg.Input(ivs["w4"],key="w4",right_click_menu=grp_rclick_menu["w4"], size=(10, 1))],
		[sg.Text("embed1 modelVersionId"), sg.Input(ivs["embed1"],key="embed1",right_click_menu=grp_rclick_menu["embed1"], size=(20, 1)),sg.Text("embed2 modelVersionId"), sg.Input(ivs["embed2"],key="embed2",right_click_menu=grp_rclick_menu["embed2"], size=(20, 1))],
		[sg.Text("embed3 modelVersionId"), sg.Input(ivs["embed3"],key="embed3",right_click_menu=grp_rclick_menu["embed3"], size=(20, 1)),sg.Text("embed4 modelVersionId"), sg.Input(ivs["embed4"],key="embed4",right_click_menu=grp_rclick_menu["embed4"], size=(20, 1))],
		[sg.Text("controlnet modelVersionId"), sg.Input(ivs["cont"],key="cont",right_click_menu=grp_rclick_menu["cont"], size=(20, 1)),sg.Text("upscaler modelVersionId"), sg.Input(ivs["up"],key="up",right_click_menu=grp_rclick_menu["up"], size=(20, 1))],
		[sg.Multiline("infomation\n",key="info", size=(100, 3))],
		[sg.Button('RUN', key='RUN'),sg.Button('EXIT', key='EXIT'),sg.Push(),sg.Button('SAVE', key='SAVE'),sg.Button('SAVE AS', key='SAAS'),sg.Button('LOAD', key='LOAD')]
	]

	window = sg.Window('metadata', layout,location=(0,0),keep_on_top=True)

	while True:
		event, values = window.read()
		if event == sg.WINDOW_CLOSED:
			break
		elif event=="EXIT":
			break
		elif event=="RUN":
			if values["input"]!="" and values["ckpt"]!="":
				thread1 = threading.Thread(target=run,args=(values,window))
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
		elif event=="SAVE":
			save_data(values)
		elif event=="LOAD":
			load_data(window)
		elif event=="SAAS":
			save_data_as(values)
		elif event=="READ":
			read_meta(values["input"],window)

	window.close()
