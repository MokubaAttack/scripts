import os
from safetensors.torch import load_file,save_file
import FreeSimpleGUI as sg

layout =[
    [sg.InputText(key='-INPUT-'),sg.FileBrowse('select lora', key="-FILENAME-", file_types=(('lora file', '.safetensors'),))],
    [sg.Button('Exit'),sg.Button('Run',key="-RUN-")]
        ]

window = sg.Window('Modify Lora', layout)

ok=[
    "lora_unet_out_2.alpha",
    "lora_unet_out_2.lora_down.weight",
    "lora_unet_out_2.lora_up.weight",
    "lora_unet_input_blocks_0_0.alpha",
    "lora_unet_input_blocks_0_0.lora_down.weight",
    "lora_unet_input_blocks_0_0.lora_up.weight",
    "lora_unet_label_emb_0_0.alpha",
    "lora_unet_label_emb_0_0.lora_down.weight",
    "lora_unet_label_emb_0_0.lora_up.weight",
    "lora_unet_label_emb_0_2.alpha",
    "lora_unet_label_emb_0_2.lora_down.weight",
    "lora_unet_label_emb_0_2.lora_up.weight",
    "lora_unet_time_embed_0.alpha",
    "lora_unet_time_embed_0.lora_down.weight",
    "lora_unet_time_embed_0.lora_up.weight",
    "lora_unet_time_embed_2.alpha",
    "lora_unet_time_embed_2.lora_down.weight",
    "lora_unet_time_embed_2.lora_up.weight"
]

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    elif event=="-RUN-":
        if values['-INPUT-']!="":
            if os.path.exists(values['-INPUT-']):
                if values['-INPUT-'].endswith(".safetensors"):
                    try:
                        in_path=values['-INPUT-']
                        out=in_path.replace(".safetensors","_mod.safetensors")
                        state_dict=load_file(in_path)
                        out_dict={}
                        for k,w in state_dict.items():
                            if not(k in ok):
                                out_dict[k]=w
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

