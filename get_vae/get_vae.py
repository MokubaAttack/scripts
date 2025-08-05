import FreeSimpleGUI as sg
import os
import torch
from safetensors.torch import load_file,save_file

layout =[
    [sg.InputText(key='-INPUT-'),sg.FileBrowse('select ckpt', key="-FILENAME-", file_types=(('ckpt file', '.safetensors'),))],
    [sg.Button('Exit'),sg.Button('Run',key="-RUN-")]
        ]

window = sg.Window('Get Vae File', layout)

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    elif event=="-RUN-":
        if values['-INPUT-']!="":
            if os.path.exists(values['-INPUT-']):
                if values['-INPUT-'].endswith(".safetensors"):
                    try:
                        key="first_stage_model."
                        base_safe=values['-INPUT-']
                        out=base_safe.replace(".safetensors","_vae.safetensors")
                        state_dict=load_file(base_safe)
                        out_dict={}
                        for k,w in state_dict.items():
                            if k.startswith(key):
                                k_out=k.replace(key,"")
                                out_dict[k_out]=w.to(torch.float16)
                        save_file(out_dict,out)
                        sg.popup(out,title="fin")
                    except:
                        sg.popup("fail in the output.",title="error")
                else:
                    sg.popup("you need to select the safetensors file.",title="error")
            else:
                sg.popup("the selected file doesn't exist.",title="error")
        else:
            sg.popup("you need to input the file path.",title="error")
            
window.close()