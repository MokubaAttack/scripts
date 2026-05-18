from safetensors.torch import (
    load_file,
    save_file
)

check="final_layer.linear.weight"
pass_keys=[
    "model.diffusion_model.pos_embedder.dim_spatial_range",
    "model.diffusion_model.pos_embedder.dim_temporal_range",
    "model.diffusion_model.pos_embedder.seq"
]

def get_metadata(path):
    f=open(path,"rb")
    data=f.read(8)
    n=int.from_bytes(data,byteorder="little")
    data=f.read(n)
    head=data.decode()
    head_dict=json.loads(head)
    if "__metadata__" in head_dict:
        meta=head_dict["__metadata__"]
    else:
        meta={"format":"pt"}
    f.close()
    return meta

def modsafe(path,out_path,win):
    win["RUN"].Update(disabled=True)
    win["info"].update("modifying")

    meta=get_metadata(path)
    sd=load_file(path)

    head=None
    keys=[]
    for k in sd:
        keys.append(k)
        if k.endswith(check):
            head=k.replace(check,"")

    if head==None:
        win["RUN"].Update(disabled=False)
        win["info"].update("error")
        return
    
    for k in keys:
        if k.startswith("first_stage_model.") or k.startswith("cond_stage_model.qwen3_06b.transformer.model."):
            pass
        elif k.startswith(head):
            mk=k.replace(head,"")
            mk="model.diffusion_model."+k
            if not(mk in pass_keys):
                sd[mk]=sd[k]
            if head!="model.diffusion_model.":
                del sd[k]
        else:
            del sd[k]
            
    save_file(sd,out_path,metadata=meta)
    win["RUN"].Update(disabled=False)
    win["info"].update("fin")

if __name__ == '__main__':
    import FreeSimpleGUI as sg
    import tkinter as tk
    import pyperclip
    from plyer import notification
    import threading

    keys=['output','input']
    grp_rclick_menu={}
    for k in keys:
        grp_rclick_menu[k]=[
            "",
            [
                "-copy-::"+k,"-cut-::"+k,"-paste-::"+k
            ]
        ]

    layout=[
        [sg.Text("input"),sg.Input(key="input",right_click_menu=grp_rclick_menu["input"]),sg.FilesBrowse(file_types=(('model file', '.safetensors'),))],
        [sg.Text("output"), sg.Input(key="output",right_click_menu=grp_rclick_menu["output"]),sg.FileSaveAs(file_types=(('model file', '.safetensors'),))],
        [sg.Text("infomation",key="info")],
        [sg.Button('RUN', key='RUN'),sg.Button('EXIT', key='EXIT')]
    ]

    window = sg.Window('Modify Anima Ckpt', layout)

    while True:
        event, values = window.read()
            
        if event == sg.WINDOW_CLOSED:
            break
        elif event=="EXIT":
            break
        elif event=="RUN":
            if values["output"]!="" and values["input"]!="":
                outpath=values["output"]
                inpath=values["input"]
                thread1 = threading.Thread(target=modsafe,args=(inpath,outpath,window))
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
