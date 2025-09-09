from PIL import Image, PngImagePlugin
import FreeSimpleGUI as sg
import pickle
import os

choices=[
    "Euler a","Euler","LMS","Heun","DPM2","DPM2 a","DPM++ 2S a","DPM++ 2M","DPM++ SDE","DPM++ 2M SDE","DPM++ 3M SDE","DPM fast","DPM adaptive","LMS Karras","DPM2 Karras","DPM2 a Karras","DPM++ 2S a Karras","DPM++ 2M Karras","DPM++ SDE Karras","DPM++ 2M SDE Karras","DPM++ 3M SDE Karras","DPM++ 3M SDE Exponential","DDIM","PLMS","UniPC","LCM"
]
ivs=[
    "prompt",
    "negative prompt",
    "30",
    "DDIM",
    "7",
    "",
    "2",
    "",
    ["",""],
    ["",""],
    ["",""],
    ["",""],
    "",
    "",
    "",
    ""
]
keys=[
    "pr","ne","st","sa","cf","se","cl","ckpt","lora1","lora2","lora3","lora4","embed1","embed2","embed3","embed4","w1","w2","w3","w4"
]
def run(vs,w):
    try:
        if ";" in vs["input"]:
            paths=vs["input"].split(";")
        else:
            paths=[vs["input"]]
        
        metadata=vs["pr"]+"\n\n"
        metadata=metadata+"Negative prompt: "+vs["ne"]+"\n\n"
        if vs["st"]!="":
            metadata=metadata+"Steps: "+vs["st"]+", " 
        if vs["sa"]!="":
            metadata=metadata+"Sampler: "+vs["sa"]+", "
        else:
            metadata=metadata+"Sampler: Undefined, "
        if vs["cf"]!="":
            metadata=metadata+"CFG scale: "+vs["cf"]+", "
        if vs["se"]!="":
            metadata=metadata+"Seed: "+vs["se"]+", "
        if vs["cl"]!="":
            metadata=metadata+"Clip skip: "+vs["cl"]+", "
        metadata=metadata+'Civitai resources: [{"type":"checkpoint","modelVersionId":'+vs["ckpt"]+"}"
        for i in range(4):
            if vs["lora"+str(i+1)]!="":
                metadata=metadata+',{"type":"lora","weight":'
                if vs["w"+str(i+1)]!="":
                    metadata=metadata+vs["w"+str(i+1)]+',"modelVersionId":'+vs["lora"+str(i+1)]+"}"
                else:
                    metadata=metadata+'1,"modelVersionId":'+vs["lora"+str(i+1)]+"}"
        for i in range(4):
            if vs["embed"+str(i+1)]!="":
                metadata=metadata+',{"type":"embed","modelVersionId":'+vs["embed"+str(i+1)]+"}"
        metadata=metadata+'], Civitai metadata: {}'
            
        for path in paths:
            image_path=path
            output_path=image_path.replace(".png","_meta.png")

            image = Image.open(image_path)
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("parameters", metadata)
            image.save(output_path, "PNG", pnginfo=pnginfo)
            print(output_path)
    except:
        print("error")

if os.path.exists("metadata.pkl"):
    f=open('metadata.pkl', 'rb')
    state_dict= pickle.load(f)
    ivs[0]=state_dict["pr"]
    ivs[1]=state_dict["ne"]
    ivs[2]=state_dict["st"]
    ivs[3]=state_dict["sa"]
    ivs[4]=state_dict["cf"]
    ivs[5]=state_dict["se"]
    ivs[6]=state_dict["cl"]
    ivs[7]=state_dict["ckpt"]
    ivs[8][0]=state_dict["lora1"]
    ivs[9][0]=state_dict["lora2"]
    ivs[10][0]=state_dict["lora3"]
    ivs[11][0]=state_dict["lora4"]
    ivs[12]=state_dict["embed1"]
    ivs[13]=state_dict["embed2"]
    ivs[14]=state_dict["embed3"]
    ivs[15]=state_dict["embed4"]
    ivs[8][1]=state_dict["w1"]
    ivs[9][1]=state_dict["w2"]
    ivs[10][1]=state_dict["w3"]
    ivs[11][1]=state_dict["w4"]
    del state_dict
    f.close()
    
col1=[
    [sg.Text("prompt")],
    [sg.Multiline(ivs[0], size=(40, 5),key="pr")]
]
col2=[
    [sg.Text("negative prompt")],
    [sg.Multiline(ivs[1], size=(40, 5),key="ne")]
]
    
layout=[
    [sg.Text("input"), sg.Input(key="input"),sg.FilesBrowse(file_types=(('image file', '.png'),))],
    [sg.Column(col1),sg.Column(col2)],
    [sg.Text("Steps"), sg.Input(ivs[2],key="st")],
    [sg.Text("Sampler"), sg.Combo(default_value=ivs[3],values=choices,key="sa")],
    [sg.Text("CFG scale"), sg.Input(ivs[4],key="cf")],
    [sg.Text("Seed"), sg.Input(ivs[5],key="se")],
    [sg.Text("Clip skip"), sg.Input(ivs[6],key="cl")],
    [sg.Text("ckpt modelVersionId"), sg.Input(ivs[7],key="ckpt")],
    [sg.Text("lora1 modelVersionId"), sg.Input(ivs[8][0],key="lora1"),sg.Text("weight"), sg.Input(ivs[8][1],key="w1")],
    [sg.Text("lora2 modelVersionId"), sg.Input(ivs[9][0],key="lora2"),sg.Text("weight"), sg.Input(ivs[9][1],key="w2")],
    [sg.Text("lora3 modelVersionId"), sg.Input(ivs[10][0],key="lora3"),sg.Text("weight"), sg.Input(ivs[10][1],key="w3")],
    [sg.Text("lora4 modelVersionId"), sg.Input(ivs[11][0],key="lora4"),sg.Text("weight"), sg.Input(ivs[11][1],key="w4")],
    [sg.Text("embed1 modelVersionId"), sg.Input(ivs[12],key="embed1")],
    [sg.Text("embed2 modelVersionId"), sg.Input(ivs[13],key="embed2")],
    [sg.Text("embed3 modelVersionId"), sg.Input(ivs[14],key="embed3")],
    [sg.Text("embed4 modelVersionId"), sg.Input(ivs[15],key="embed4")],
    [sg.Button('RUN', key='RUN'),sg.Button('EXIT', key='EXIT')]
]
window = sg.Window('metadata', layout)

while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    elif event=="EXIT":
        break
    elif event=="RUN":
        if values["input"]!="" and values["ckpt"]!="":
            run(values,window)

state_dict={}
for key in keys:
    state_dict[key]=values[key]
with open("metadata.pkl","wb") as f:
    pickle.dump(state_dict, f)
window.close()