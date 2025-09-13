import FreeSimpleGUI as sg
import os
from safetensors.torch import save_file,load_file
import torch
import tkinter as tk
import pyperclip
from plyer import notification
import threading

def mergeckpt(ckpts,weights,v,out_path):
    w.find_element('RUN').Update(disabled=True)
    if not(out_path.endswith(".safetensors")):
        w.find_element('RUN').Update(disabled=False)
        notification.notify(title="error",message=path+" does not exist.",timeout=8)
        return
    for path in ckpts:
        if not(os.path.exists(path)):
            w.find_element('RUN').Update(disabled=False)
            notification.notify(title="error",message="I failed in the output.",timeout=8)
            return
    try:
        weights_sum=sum(weights)
        for i in range(len(weights)):
            weights[i]=weights[i]/weights_sum
        out_dict={}
        safe=open(os.getcwd()+"/data.txt","r")
        for line in safe:
            data=line.split(",")
            s=[]
            for i in range(len(data)):
                if i==0:
                    k=data[i]
                else:
                    s.append(int(data[i]))
            out_dict[k]=torch.zeros(s)
        safe.close()
        for i in range(len(ckpts)):
            state_dict=load_file(ckpts[i])
            for k,w in out_dict.items():
                if k in state_dict:
                    out_dict[k]=out_dict[k].to(torch.float16)+(state_dict[k]*weights[i]).to(torch.float16)
        if v!=-1:
            state_dict=load_file(ckpts[v])
            for k,w in out_dict.items():
                if k.startswith("first_stage_model."):
                    out_dict[k]=state_dict[k].to(torch.float16)
        save_file(out_dict,out_path)
        f=open(out_path.replace(".safetensors",".txt"),"w")
        for i in range(len(ckpts)):
            f.write("ckpt"+str(i+1)+" : "+ckpts[i]+"\nckpt"+str(i+1)+"_weight : "+str(weights[i])+"\n")
        if v!=-1:
            f.write("vae : "+ckpts[v]+"\n")
        else:
            f.write("vae : None\n")
        f.close()
        del out_dict,state_dict
        w.find_element('RUN').Update(disabled=False)
        notification.notify(title="fin",message=out_path,timeout=8)
    except:
        w.find_element('RUN').Update(disabled=False)
        notification.notify(title="error",message="I failed in the output.",timeout=8)
        
keys=[
    "ckpt1","ckpt2","ckpt3","ckpt4","ckpt5","ckpt6","ckpt7",
    "w1","w2","w3","w4","w5","w6","w7","out"
]        
for key in keys:
    grp_rclick_menu[key]=[
        "",
        [
            "-copy-::"+key,"-cut-::"+key,"-paste-::"+key
        ]
    ]

box1=[
    [sg.Text("checkpoint file")],
    [sg.Column([[sg.Text("ckpt1"), sg.Input(key="ckpt1",right_click_menu=grp_rclick_menu["ckpt1"]),sg.FileBrowse( file_types=(('ckpt file', '.safetensors'),))]],size=(435,35))],
    [sg.Column([[sg.Text("ckpt2"), sg.Input(key="ckpt2",right_click_menu=grp_rclick_menu["ckpt2"]),sg.FileBrowse( file_types=(('ckpt file', '.safetensors'),))]],size=(435,35))],
    [sg.Column([[sg.Text("ckpt3"), sg.Input(key="ckpt3",right_click_menu=grp_rclick_menu["ckpt3"]),sg.FileBrowse( file_types=(('ckpt file', '.safetensors'),))]],size=(435,35))],
    [sg.Column([[sg.Text("ckpt4"), sg.Input(key="ckpt4",right_click_menu=grp_rclick_menu["ckpt4"]),sg.FileBrowse( file_types=(('ckpt file', '.safetensors'),))]],size=(435,35))],
    [sg.Column([[sg.Text("ckpt5"), sg.Input(key="ckpt5",right_click_menu=grp_rclick_menu["ckpt5"]),sg.FileBrowse( file_types=(('ckpt file', '.safetensors'),))]],size=(435,35))],
    [sg.Column([[sg.Text("ckpt6"), sg.Input(key="ckpt6",right_click_menu=grp_rclick_menu["ckpt6"]),sg.FileBrowse( file_types=(('ckpt file', '.safetensors'),))]],size=(435,35))],
    [sg.Column([[sg.Text("ckpt7"), sg.Input(key="ckpt7",right_click_menu=grp_rclick_menu["ckpt7"]),sg.FileBrowse( file_types=(('ckpt file', '.safetensors'),))]],size=(435,35))]
]

box2=[
    [sg.Text("weight")],
    [sg.Column([[sg.Input(key="w1",right_click_menu=grp_rclick_menu["w1"])]],size=(80,35))],
    [sg.Column([[sg.Input(key="w2",right_click_menu=grp_rclick_menu["w2"])]],size=(80,35))],
    [sg.Column([[sg.Input(key="w3",right_click_menu=grp_rclick_menu["w3"])]],size=(80,35))],
    [sg.Column([[sg.Input(key="w4",right_click_menu=grp_rclick_menu["w4"])]],size=(80,35))],
    [sg.Column([[sg.Input(key="w5",right_click_menu=grp_rclick_menu["w5"])]],size=(80,35))],
    [sg.Column([[sg.Input(key="w6",right_click_menu=grp_rclick_menu["w6"])]],size=(80,35))],
    [sg.Column([[sg.Input(key="w7",right_click_menu=grp_rclick_menu["w7"])]],size=(80,35))]
]

box3=[
    [sg.Text("vae")],
    [sg.Column([[sg.Radio("",key="v1",group_id='destination')]],size=(35,35))],
    [sg.Column([[sg.Radio("",key="v2",group_id='destination')]],size=(35,35))],
    [sg.Column([[sg.Radio("",key="v3",group_id='destination')]],size=(35,35))],
    [sg.Column([[sg.Radio("",key="v4",group_id='destination')]],size=(35,35))],
    [sg.Column([[sg.Radio("",key="v5",group_id='destination')]],size=(35,35))],
    [sg.Column([[sg.Radio("",key="v6",group_id='destination')]],size=(35,35))],
    [sg.Column([[sg.Radio("",key="v7",group_id='destination')]],size=(35,35))]
]

layout=[
    [sg.Column(box1),sg.Column(box2),sg.Column(box3)],
    [sg.Text("output path"), sg.Input(key="out",right_click_menu=grp_rclick_menu["out"]),sg.FileSaveAs(file_types=(('ckpt file', '.safetensors'),))],
    [sg.Button('RUN', key='RUN'),sg.Button('Cancel Vae', key='cancel'),sg.Button('EXIT', key='EXIT')]
]

window = sg.Window('merge ckpt', layout)

while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    elif event=="EXIT":
        break
    elif event=="RUN":
        c1=not(values["ckpt1"]=="" and values["ckpt2"]=="" and values["ckpt3"]=="" and values["ckpt4"]=="" and values["ckpt5"]=="" and values["ckpt6"]=="" and values["ckpt7"]=="")
        if values["out"]!="" and c1:
            out_path=values["out"]
            v=-1
            ckpts=[]
            weights=[]
            for i in range(7):
                if values["ckpt"+str(i+1)]!="":
                    ckpts.append(values["ckpt"+str(i+1)])
                    if values["w"+str(i+1)]=="":
                        weights.append(1.0)
                        window["w"+str(i+1)].update("1.0")
                    else:
                        try:
                            weights.append(float(values["w"+str(i+1)]))
                        except:
                            weights.append(1.0)
                            window["w"+str(i+1)].update("1.0")
                    if values["v"+str(i+1)]:
                        v=i
            thread1 = threading.Thread(target=mergeckpt,args=(ckpts,weights,v,out_path))
            thread1.start()
    elif event=="cancel":
        for i in range(7):
            window["v"+str(i+1)].update(False)
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
