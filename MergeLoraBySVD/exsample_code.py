import os,sys
# you need to input path of sd-scripts directory.
path="sd-scripts"
sys.path.append(path)
import MergeLoraBySVD
import FreeSimpleGUI as sg

def run_function(vs,w):
    loras=[]
    ws=[]
    for i in range(4):
        if vs["ckpt"+str(i+1)]!="":
            if vs["ckpt"+str(i+1)].endswith(".safetensors"):
                if not(os.path.exists(vs["ckpt"+str(i+1)])):
                    sg.popup("lora"+str(i+1)+" file does not exist.",title = "error")
                    return
                loras.append(vs["ckpt"+str(i+1)])
                try:
                    if vs["w"+str(i+1)]=="":
                        ws.append(1.0)
                        w["w"+str(i+1)].update("1.0")
                    else:
                        ws.append(float(vs["w"+str(i+1)]))
                        w["w"+str(i+1)].update(str(float(vs["w"+str(i+1)])))
                except:
                    ws.append(1.0)
                    w["w"+str(i+1)].update("1.0")
            else:
                sg.popup("You need to select the safetensors file for lora"+str(i+1)+" file.",title = "error")
                return
            
    if vs["out"].endswith(".safetensors"):
        out_path=vs["out"]
    else:
        sg.popup("You need to select the safetensors file for output file.",title = "error")
        return
    
    try:
        if vs["d"]=="":
            dim=16
            w["d"].update("16")
        else:
            dim=int(vs["d"])
            w["d"].update(str(int(vs["d"])))
    except:
        dim=16
        w["d"].update("16")
        
    try:
        MergeLoraBySVD.merge(loras=loras,weights=ws,save_to=out_path,new_rank=dim)
        sg.popup(out_path,title = "fin")
    except:
        sg.popup("I failed in the output.",title = "error")

box1=[
    [sg.Text("lora file")],
    [sg.Column([[sg.Text("lora1"), sg.Input(key="ckpt1"),sg.FileBrowse( file_types=(('lora file', '.safetensors'),))]],size=(435,35))],
    [sg.Column([[sg.Text("lora2"), sg.Input(key="ckpt2"),sg.FileBrowse( file_types=(('lora file', '.safetensors'),))]],size=(435,35))],
    [sg.Column([[sg.Text("lora3"), sg.Input(key="ckpt3"),sg.FileBrowse( file_types=(('lora file', '.safetensors'),))]],size=(435,35))],
    [sg.Column([[sg.Text("lora4"), sg.Input(key="ckpt4"),sg.FileBrowse( file_types=(('lora file', '.safetensors'),))]],size=(435,35))]
]

box2=[
    [sg.Text("weight")],
    [sg.Column([[sg.Input("0.7",key="w1")]],size=(80,35))],
    [sg.Column([[sg.Input("0.7",key="w2")]],size=(80,35))],
    [sg.Column([[sg.Input("0.7",key="w3")]],size=(80,35))],
    [sg.Column([[sg.Input("0.7",key="w4")]],size=(80,35))]
]

layout=[
    [sg.Column(box1),sg.Column(box2)],
    [sg.Text("dim"), sg.Input("16",key="d")],
    [sg.Text("output path"), sg.Input(key="out"),sg.FileSaveAs(file_types=(('lora file', '.safetensors'),))],
    [sg.Button('RUN', key='RUN'),sg.Button('EXIT', key='EXIT')]
]

window = sg.Window('merge lora', layout)

while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    elif event=="EXIT":
        break
    elif event=="RUN":
        c1=not(values["ckpt1"]=="" and values["ckpt2"]=="" and values["ckpt3"]=="" and values["ckpt4"]=="")
        if c1 and values["out"]!="":
            run_function(values,window)

window.close()
