import os,shutil,json
from safetensors.torch import save_file,load_file
import torch
import gc

def spckpt(path,folder,ind):
    f=open(path,"rb")
    l=int.from_bytes(f.read(8),byteorder="little")
    head=f.read(l).decode()
    head=json.loads(head)
    head.pop("__metadata__")
    for k,w in head.items():
        sd={"__metadata__":{"format":"pt"}}
        sd[k]=w
        n=sd[k]["data_offsets"][1]-sd[k]["data_offsets"][0]
        sd[k]["data_offsets"][0]=0
        sd[k]["data_offsets"][1]=n
        sd=str(sd).replace("'",'"')
        sd=sd.encode()
        l_sd=len(sd).to_bytes(8,byteorder="little")
        out=open(folder+"/"+ind+"_"+k+".safetensors","wb")
        out.write(l_sd)
        out.write(sd)
        out.write(f.read(n))
        out.close()
    f.close()

def mergeckpt(ckpts,weights,v,out_path,win=None):
    if win!=None:
        win["RUN"].Update(disabled=True)
    if not(out_path.endswith(".safetensors")):
        if win==None:
            print("the output path is needed to be a safetensors file.")
        else:
            win["RUN"].Update(disabled=False)
            win["info"].update("error")
            notification.notify(title="error",message="the output path is needed to be a safetensors file.",timeout=8)
        return
    for path in ckpts:
        if not(os.path.exists(path)):
            if win==None:
                print(path+" does not exist.")
            else:
                win["RUN"].Update(disabled=False)
                win["info"].update("error")
                notification.notify(title="error",message=path+" does not exist.",timeout=8)
            return
    try:
        safe=open(os.getcwd()+"/data.txt","r")
        if os.path.exists(os.getcwd()+"/safe_temp"):
            shutil.rmtree(os.getcwd()+"/safe_temp")
        os.mkdir(os.getcwd()+"/safe_temp")
        win["info"].update("making initial ckpt")
        data_dict={}
        for line in safe:
            data=line.split(",")
            s=[]
            out_dict={}
            for i in range(len(data)):
                if i==0:
                    k=data[i]
                else:
                    s.append(int(data[i]))
            data_dict[k]=s
        safe.close()

        spckpt(ckpts[0],os.getcwd()+"/safe_temp","1")
        spckpt(ckpts[1],os.getcwd()+"/safe_temp","2")

        dict_sum=len(list(data_dict))
        key_count=0
        for k in data_dict:
            key_count=key_count+1
            win["info"].update("merging "+str(key_count)+"/"+str(dict_sum))
            out_dict={}
            out_dict[k]=torch.zeros(data_dict[k])
            for i in range(2):
                if os.path.exists(os.getcwd()+"/safe_temp/"+str(i+1)+"_"+k+".safetensors"):
                    state_dict=load_file(os.getcwd()+"/safe_temp/"+str(i+1)+"_"+k+".safetensors")
                    if k.startswith("model.diffusion_model.middle_block."):
                        out_dict[k]=out_dict[k]+(state_dict[k]*weights[10][i])
                    elif k.startswith("model.diffusion_model.input_blocks."):
                        j=int(k.split(".")[3])
                        out_dict[k]=out_dict[k]+(state_dict[k]*weights[1+j][i])
                    elif k.startswith("model.diffusion_model.output_blocks."):
                        j=int(k.split(".")[3])
                        out_dict[k]=out_dict[k]+(state_dict[k]*weights[11+j][i])
                    else:
                        if k.startswith("first_stage_model."):
                            if v!=-1:
                                if v==i:
                                    out_dict[k]=state_dict[k]
                            else:
                                out_dict[k]=out_dict[k]+(state_dict[k]*weights[0][i])
                        else:
                            out_dict[k]=out_dict[k]+(state_dict[k]*weights[0][i])
                    del state_dict
            out_dict[k]=out_dict[k].to(torch.float16)
            save_file(out_dict,os.getcwd()+"/safe_temp/"+k+".safetensors")
            del out_dict
            gc.collect()

        win["info"].update("making output")
        out_dict={}
        out_dict["__metadata__"]={"format":"pt"}
        n=0
        for k in data_dict:
            f=open(os.getcwd()+"/safe_temp/"+k+".safetensors","rb")
            l=int.from_bytes(f.read(8),byteorder="little")
            head=f.read(l).decode()
            head=json.loads(head)
            out_dict[k]=head[k]
            offsets=out_dict[k]["data_offsets"][1]
            out_dict[k]["data_offsets"][0]=n
            n=n+offsets
            out_dict[k]["data_offsets"][1]=n
            f.close()
        output=open(out_path,"wb")
        out_dict=str(out_dict).replace("'",'"')
        out_dict=out_dict.encode()
        l=len(out_dict).to_bytes(8,byteorder="little")
        output.write(l)
        output.write(out_dict)
        for k in data_dict:
            f=open(os.getcwd()+"/safe_temp/"+k+".safetensors","rb")
            l=int.from_bytes(f.read(8),byteorder="little")
            head=f.read(l)
            output.write(f.read())
            f.close()
        output.close()
        f=open(out_path.replace(".safetensors",".txt"),"w")
        for i in range(len(ckpts)):
            f.write("ckpt"+str(i+1)+" : "+ckpts[i]+"\n")
        if v!=-1:
            f.write("vae : "+ckpts[v]+"\n")
        else:
            f.write("vae : None\n")
        for i in range(20):
            weights[i]=weights[i][0]
        f.write("weight : "+str(weights)+"\n")
        f.close()
        shutil.rmtree(os.getcwd()+"/safe_temp")
        del out_dict,l,head,n,offsets
        gc.collect()
        if win==None:
            print(out_path)
        else:
            win["RUN"].Update(disabled=False)
            win["info"].update("fin")
            notification.notify(title="fin",message=out_path,timeout=8)
    except:
        if os.path.exists(os.getcwd()+"/safe_temp"):
            shutil.rmtree(os.getcwd()+"/safe_temp")
        if win==None:
            print("I failed in the output.")
        else:
            win["RUN"].Update(disabled=False)
            win["info"].update("error")
            notification.notify(title="error",message="I failed in the output.",timeout=8)

if __name__=="__main__":
    import FreeSimpleGUI as sg
    import tkinter as tk
    import pyperclip
    from plyer import notification
    import threading

    sg.theme('GrayGrayGray')       
    keys=["ckpt1","ckpt2","out"]
    for i in range(20):
        keys.append("w"+str(i+1))
    grp_rclick_menu={}
    for key in keys:
        grp_rclick_menu[key]=[
            "",
            [
                "-copy-::"+key,"-cut-::"+key,"-paste-::"+key
            ]
        ]

    layout=[
        [sg.Text("ckpt1"), sg.Input(key="ckpt1",right_click_menu=grp_rclick_menu["ckpt1"]),sg.FileBrowse( file_types=(('ckpt file', '.safetensors'),)),sg.Text("vae"),sg.Radio("",key="v1",group_id='destination')],
        [sg.Text("ckpt2"), sg.Input(key="ckpt2",right_click_menu=grp_rclick_menu["ckpt2"]),sg.FileBrowse( file_types=(('ckpt file', '.safetensors'),)),sg.Text("vae"),sg.Radio("",key="v2",group_id='destination')],
        [sg.Text("weight")],
        [
            sg.Frame("BASE",[[sg.Input(key="w1",right_click_menu=grp_rclick_menu["w1"], size=(10, 1))]]),
            sg.Frame("IN00",[[sg.Input(key="w2",right_click_menu=grp_rclick_menu["w2"], size=(10, 1))]]),
            sg.Frame("IN01",[[sg.Input(key="w3",right_click_menu=grp_rclick_menu["w3"], size=(10, 1))]]),
            sg.Frame("IN02",[[sg.Input(key="w4",right_click_menu=grp_rclick_menu["w4"], size=(10, 1))]]),
            sg.Frame("IN03",[[sg.Input(key="w5",right_click_menu=grp_rclick_menu["w5"], size=(10, 1))]]),
            sg.Frame("IN04",[[sg.Input(key="w6",right_click_menu=grp_rclick_menu["w6"], size=(10, 1))]]),
            sg.Frame("IN05",[[sg.Input(key="w7",right_click_menu=grp_rclick_menu["w7"], size=(10, 1))]]),
            sg.Frame("IN06",[[sg.Input(key="w8",right_click_menu=grp_rclick_menu["w8"], size=(10, 1))]]),
            sg.Frame("IN07",[[sg.Input(key="w9",right_click_menu=grp_rclick_menu["w9"], size=(10, 1))]]),
            sg.Frame("IN08",[[sg.Input(key="w10",right_click_menu=grp_rclick_menu["w10"], size=(10, 1))]])
        ],
        [
            sg.Frame("MID",[[sg.Input(key="w11",right_click_menu=grp_rclick_menu["w11"], size=(10, 1))]]),
            sg.Frame("OUT00",[[sg.Input(key="w12",right_click_menu=grp_rclick_menu["w12"], size=(10, 1))]]),
            sg.Frame("OUT01",[[sg.Input(key="w13",right_click_menu=grp_rclick_menu["w13"], size=(10, 1))]]),
            sg.Frame("OUT02",[[sg.Input(key="w14",right_click_menu=grp_rclick_menu["w14"], size=(10, 1))]]),
            sg.Frame("OUT03",[[sg.Input(key="w15",right_click_menu=grp_rclick_menu["w15"], size=(10, 1))]]),
            sg.Frame("OUT04",[[sg.Input(key="w16",right_click_menu=grp_rclick_menu["w16"], size=(10, 1))]]),
            sg.Frame("OUT05",[[sg.Input(key="w17",right_click_menu=grp_rclick_menu["w17"], size=(10, 1))]]),
            sg.Frame("OUT06",[[sg.Input(key="w18",right_click_menu=grp_rclick_menu["w18"], size=(10, 1))]]),
            sg.Frame("OUT07",[[sg.Input(key="w19",right_click_menu=grp_rclick_menu["w19"], size=(10, 1))]]),
            sg.Frame("OUT08",[[sg.Input(key="w20",right_click_menu=grp_rclick_menu["w20"], size=(10, 1))]])
        ],
        [sg.Text("output path"), sg.Input(key="out",right_click_menu=grp_rclick_menu["out"]),sg.FileSaveAs(file_types=(('ckpt file', '.safetensors'),))],
        [sg.Text("infomation",key="info")],
        [sg.Button('RUN', key='RUN'),sg.Button('Cancel Vae', key='cancel'),sg.Button('EXIT', key='EXIT')]
    ]

    icon_path=b"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"
    window = sg.Window('Merge Ckpt (Block)', layout,icon=icon_path)

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break
        elif event=="EXIT":
            break
        elif event=="RUN":
            if values["out"]!="" and values["ckpt1"]!="" and values["ckpt2"]!="":
                out_path=values["out"]
                if values["v1"]:
                    v=0
                elif values["v2"]:
                    v=1
                else:
                    v=-1
                ckpts=[values["ckpt1"],values["ckpt2"]]
                weights=[]
                for i in range(20):
                    if values["w"+str(i+1)]!="":
                        try:
                            r=float(values["w"+str(i+1)])
                            weights.append([1-r,r])
                            window["w"+str(i+1)].update(str(weights[i][1]))
                        except:
                            if i==0:
                                weights.append([0.5,0.5])
                                window["w"+str(i+1)].update("0.5")
                            else:
                                weights.append(weights[i-1])
                                window["w"+str(i+1)].update(str(weights[i][1]))
                    else:
                        if i==0:
                            weights.append([0.5,0.5])
                            window["w"+str(i+1)].update("0.5")
                        else:
                            weights.append(weights[i-1])
                            window["w"+str(i+1)].update(str(weights[i][1]))
                thread1 = threading.Thread(target=mergeckpt,args=(ckpts,weights,v,out_path,window))
                thread1.start()
        elif event=="cancel":
            for i in range(2):
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
