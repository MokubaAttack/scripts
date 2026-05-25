from diffusers import StableDiffusionXLPipeline
import torch,gc,os,shutil
from safetensors.torch import load_file, save_file

CLAMP_QUANTILE=0.99

def mk_box(ind,path,out_path,c):
    pipe = StableDiffusionXLPipeline.from_single_file(path, torch_dtype=torch.float16,cache_dir=os.getcwd()+"/pipecache")
    names=[]
    if c[0]:
        for name, module in pipe.unet.named_modules():
            if module.__class__.__name__ in ["Transformer2DModel"]:
                for child_name, child_module in module.named_modules():
                    is_linear = child_module.__class__.__name__ == "Linear"
                    is_conv2d = child_module.__class__.__name__ == "Conv2d"
                    is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                    if is_linear or is_conv2d:
                        lora_name = "lora_unet" + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")

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

                        loras={}
                        loras[lora_name]=child_module.weight.contiguous()
                        save_file(loras,out_path+"/"+str(ind)+"_"+lora_name+".safetensors")
                        names.append(lora_name)
    if c[1]:
        for name, module in pipe.text_encoder.named_modules():
            if module.__class__.__name__ in ["CLIPAttention", "CLIPSdpaAttention", "CLIPMLP"]:
                for child_name, child_module in module.named_modules():
                    is_linear = child_module.__class__.__name__ == "Linear"
                    is_conv2d = child_module.__class__.__name__ == "Conv2d"
                    is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                    if is_linear or is_conv2d:
                        lora_name = "lora_te1" + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")

                        loras={}
                        loras[lora_name]=child_module.weight.contiguous()
                        save_file(loras,out_path+"/"+str(ind)+"_"+lora_name+".safetensors")
                        names.append(lora_name)
    if c[2]:
        for name, module in pipe.text_encoder_2.named_modules():
            if module.__class__.__name__ in ["CLIPAttention", "CLIPSdpaAttention", "CLIPMLP"]:
                for child_name, child_module in module.named_modules():
                    is_linear = child_module.__class__.__name__ == "Linear"
                    is_conv2d = child_module.__class__.__name__ == "Conv2d"
                    is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                    if is_linear or is_conv2d:
                        lora_name = "lora_te2" + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")

                        loras={}
                        loras[lora_name]=child_module.weight.contiguous()
                        save_file(loras,out_path+"/"+str(ind)+"_"+lora_name+".safetensors")
                        names.append(lora_name)

    del pipe,loras
    gc.collect()
    return names

def diff_ckpt(paths,out_path,dim,c,win=None):
    if win!=None:
        win["RUN"].Update(disabled=True)

    if not(out_path.endswith(".safetensors")):
        if win==None:
            print("the output path is needed to be a safetensors file.")
        else:
            win["RUN"].Update(disabled=False)
            win["info"].update("the output path is needed to be a safetensors file.")
        return

    for path in paths:
        if not(os.path.exists(path)):
            if win==None:
                print(path+" does not exist.")
            else:
                win["RUN"].Update(disabled=False)
                win["info"].update(os.path.basename(path)+" does not exist.")
            return

    if c[0]==False and c[1]==False and c[2]==False:
        if win==None:
            print("You choose no contents.")
        else:
            win["RUN"].Update(disabled=False)
            win["info"].update("You choose no contents.")
        return

    temp_path=os.getcwd()+"/safe_temp"
    try:
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        os.mkdir(temp_path)

        if win==None:
            print("load "+paths[0])
        else:
            win["info"].update("load "+paths[0])
        names0=mk_box(0,paths[0],temp_path,c)

        if win==None:
            print("load "+paths[1])
        else:
            win["info"].update("load "+paths[1])
        names1=mk_box(1,paths[1],temp_path,c)

        names=list(set(names0+names1))
        sd={}
        names_sum=len(names)
        name_count=0

        for name in names:
            name_count=name_count+1
            if win!=None:
                win["info"].update("differ "+str(name_count)+"/"+str(names_sum))
            else:
                print("\rdiffer "+str(name_count)+"/"+str(names_sum),end="")
            sd1=load_file(temp_path+"/0_"+name+".safetensors")
            sd2=load_file(temp_path+"/1_"+name+".safetensors")
            mat=(sd2[name]-sd1[name]).to(torch.float)

            conv2d = len(mat.size()) == 4
            kernel_size = None if not conv2d else mat.size()[2:4]
            conv2d_3x3 = conv2d and kernel_size != (1, 1)
            out_dim, in_dim = mat.size()[0:2]

            if conv2d:
                if conv2d_3x3:
                    mat = mat.flatten(start_dim=1)
                else:
                    mat = mat.squeeze()

            module_new_rank = dim
            module_new_rank = min(module_new_rank, in_dim, out_dim)

            try:
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
            except:
                if win==None:
                    print("")
                    print(name)
                else:
                    win["RUN"].Update(disabled=False)
                    win["info"].update("error "+name)
                shutil.rmtree(temp_path)
                return

            up_weight = U
            down_weight = Vh

            sd[name + ".lora_up.weight"] = up_weight.to(torch.float16).contiguous()
            sd[name + ".lora_down.weight"] = down_weight.to(torch.float16).contiguous()
            sd[name + ".alpha"] = torch.tensor(module_new_rank, dtype=torch.float16)
            del sd1,sd2,mat,up_weight,down_weight,U,S,Vh,dist,hi_val,low_val
            gc.collect()
        save_file(sd,out_path)
        shutil.rmtree(temp_path)
        if win==None:
            print("")
            print("fin")
        else:
            win["RUN"].Update(disabled=False)
            win["info"].update("fin")
    except:
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        if win==None:
            print("I failed in the output.")
        else:
            win["RUN"].Update(disabled=False)
            win["info"].update("I failed in the output.")

if __name__=="__main__":
    import tkinter as tk
    import pyperclip
    import threading
    import FreeSimpleGUI as sg

    sg.theme('GrayGrayGray')

    keys=["ckpt1","ckpt2","out","dim"]

    grp_rclick_menu={}
    for key in keys:
        grp_rclick_menu[key]=[
            "",
            [
                "-copy-::"+key,"-cut-::"+key,"-paste-::"+key
            ]
        ]

    layout=[
        [sg.Text("model_org"), sg.Input(key="ckpt1",right_click_menu=grp_rclick_menu["ckpt1"]),sg.FileBrowse( file_types=(('ckpt file', '.safetensors'),))],
        [sg.Text("model_tuned"), sg.Input(key="ckpt2",right_click_menu=grp_rclick_menu["ckpt2"]),sg.FileBrowse( file_types=(('ckpt file', '.safetensors'),))],
        [sg.Text("dim"), sg.Input(key="dim",right_click_menu=grp_rclick_menu["dim"])],
        [sg.Text("output path"), sg.Input(key="out",right_click_menu=grp_rclick_menu["out"]),sg.FileSaveAs(file_types=(('ckpt file', '.safetensors'),))],
        [sg.Checkbox('unet', default=True, key='unet'),sg.Checkbox('text_encoder', default=True, key='text1'),sg.Checkbox('text_encoder_2', default=True, key='text2')],
        [sg.Text("infomation",key="info")],
        [sg.Button('RUN', key='RUN'),sg.Button('EXIT', key='EXIT')]
    ]

    window = sg.Window('extract lora from models', layout)

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break
        elif event=="EXIT":
            break
        elif event=="RUN":
            if values["out"]!="" and values["ckpt1"]!="" and values["ckpt2"]!="":
                if values["dim"]=="":
                    dim=16
                else:
                    try:
                        dim=abs(int(values["dim"]))
                    except:
                        dim=16
                window["dim"].update(str(dim))
                out_path=values["out"]
                paths=[
                    values["ckpt1"],values["ckpt2"]
                ]
                c=[
                    values["unet"],values["text1"],values["text2"]
                ]
                thread1 = threading.Thread(target=diff_ckpt,args=(paths,out_path,dim,c,window))
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
