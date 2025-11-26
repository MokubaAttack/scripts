import torch
from safetensors.torch import load_file,save_file
import json
import FreeSimpleGUI as sg
import os
import tkinter as tk
import threading

fp16_type = torch.float16
bf16_type = torch.bfloat16
e4m3_type = torch.float8_e4m3fn
e5m2_type = torch.float8_e5m2

choices=[
    "fp8_e4m3","fp8_e5m2","fp16","bf16"
]

def conv(path,t,win):
    win.find_element('RUN').Update(disabled=True)
    try:
        f=open(path,"rb")
        l=int.from_bytes(f.read(8),byteorder="little")
        head=f.read(l).decode()
        f.close()
        head=json.loads(head)
        metadata=head["__metadata__"]

        if t=="fp8_e4m3":
            a=e4m3_type
        elif t=="fp8_e5m2":
            a=e5m2_type
        elif t=="bf16":
            a=bf16_type
        else:
            a=fp16_type

        sd=load_file(path)
        for k,w in sd.items():
            sd[k]=w.to(a)
        out_path=path.replace(".safetensors","_"+t+".safetensors")
        save_file(sd,out_path,metadata=metadata)
        win["info"].update("fin")
        win.find_element('RUN').Update(disabled=False)
    except:
        win["info"].update("error")
        win.find_element('RUN').Update(disabled=False)

keys=[
    "ckpt"
]
grp_rclick_menu={}
for key in keys:
    grp_rclick_menu[key]=[
        "",
        [
            "-copy-::"+key,"-cut-::"+key,"-paste-::"+key
        ]
    ]

layout=[
    [sg.Text("ckpt"), sg.Input(key="ckpt",right_click_menu=grp_rclick_menu["ckpt"]),sg.FileBrowse( file_types=(('ckpt file', '.safetensors'),))],
    [sg.Text("accuracy"), sg.Combo(default_value="fp16",values=choices,key="ac")],
    [sg.Button('RUN', key='RUN'),sg.Button('EXIT', key='EXIT'),sg.Text(key="info")]
]

window = sg.Window('accuracy', layout)

while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    elif event=="EXIT":
        break
    elif event=="RUN":
        if os.path.exists(values["ckpt"]):
            thread1 = threading.Thread(target=conv,args=(values["ckpt"],values["ac"],window))
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