import os,shutil,json,numpy,torch,gc
from safetensors.torch import save_file,load_file

check="final_layer.linear.weight"
pass_keys=[
    "model.diffusion_model.pos_embedder.dim_spatial_range",
    "model.diffusion_model.pos_embedder.dim_temporal_range",
    "model.diffusion_model.pos_embedder.seq"
]

def spckpt(path,folder,ind):
    f=open(path,"rb")
    l=int.from_bytes(f.read(8),byteorder="little")
    head=f.read(l).decode()
    head=json.loads(head)
    if "__metadata__" in head:
        head.pop("__metadata__")
    keys=[]
    h=None
    for k,w in head.items():
        keys.append(k)
        if k.endswith(check):
            h=k.replace(check,"")
    if h==None:
        return []
    elif h=="":
        for k in keys:
            mk="model.diffusion_model."+k
            if not(mk in pass_keys):
                head[mk]=head[k]
            del head[k]
    else:
        for k in keys:
            mk="model.diffusion_model."+k.removeprefix(h)
            if not(mk in pass_keys):
                head[mk]=head[k]
            del head[k]

    keys=[]
    for k,w in head.items():
        keys.append(k)
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
    return keys

def del_safe(k):
    if os.path.exists(os.getcwd()+"/safe_temp/1_"+k+".safetensors"):
        os.remove(os.getcwd()+"/safe_temp/1_"+k+".safetensors")
    if os.path.exists(os.getcwd()+"/safe_temp/2_"+k+".safetensors"):
        os.remove(os.getcwd()+"/safe_temp/2_"+k+".safetensors")
    gc.collect()

def mergeckpt(ckpts,w,out_path,mode="normal",win=None):
    if win!=None:
        win["RUN"].Update(disabled=True)

    if not(out_path.endswith(".safetensors")):
        if win==None:
            print("the output path is needed to be a safetensors file.")
        else:
            win["RUN"].Update(disabled=False)
            win["info"].update("error")
            notification.notify(title="error",message="the output path is needed to be a safetensors file.")
        return

    for path in ckpts:
        if not(os.path.exists(path)):
            if win==None:
                print(path+" does not exist.")
            else:
                win["RUN"].Update(disabled=False)
                win["info"].update("error")
                notification.notify(title="error",message=path+" does not exist.")
            return

    if not(mode in ["normal","tensor1","tensor2"]):
        mode="normal"

    try:
        if os.path.exists(os.getcwd()+"/safe_temp"):
            shutil.rmtree(os.getcwd()+"/safe_temp")
        os.mkdir(os.getcwd()+"/safe_temp")

        c1=spckpt(ckpts[0],os.getcwd()+"/safe_temp","1")
        if c1==[]:
            c1=1/0
        c2=spckpt(ckpts[1],os.getcwd()+"/safe_temp","2")
        if c2==[]:
            c2=1/0

        data_dict=list(set(c1+c2))
        dict_sum=len(data_dict)
        key_count=0

        with torch.no_grad():
            for k in data_dict:
                key_count=key_count+1
                if win!=None:
                    win["info"].update("merging "+str(key_count)+"/"+str(dict_sum))
                else:
                    print("\rmerging "+str(key_count)+"/"+str(dict_sum),end="")

                out_dict={}
                if os.path.exists(os.getcwd()+"/safe_temp/1_"+k+".safetensors"):
                    t1=load_file(os.getcwd()+"/safe_temp/1_"+k+".safetensors")[k].to(torch.float32)
                else:
                    t1=(load_file(os.getcwd()+"/safe_temp/2_"+k+".safetensors")[k]*0).to(torch.float32)

                if os.path.exists(os.getcwd()+"/safe_temp/2_"+k+".safetensors"):
                    t2=load_file(os.getcwd()+"/safe_temp/2_"+k+".safetensors")[k].to(torch.float32)
                else:
                    t2=(load_file(os.getcwd()+"/safe_temp/1_"+k+".safetensors")[k]*0).to(torch.float32)

                if mode=="normal":
                    out_dict[k]=((1-w)*t1+w*t2).to(torch.float16)

                elif "tensor" in mode:
                    w1=(1-w)/2
                    w2=w
                    w1=round(t1.size()[0]*w1)
                    w2=round(t1.size()[0]*(w1+w2))
                    if w1==0:
                        out_dict[k]=t2.to(torch.float16)
                        save_file(out_dict,os.getcwd()+"/safe_temp/"+k+".safetensors")
                        del w,out_dict,t1,t2,w1,w1
                        del_safe(k)
                        continue
                    elif w2==0:
                        out_dict[k]=t1.to(torch.float16)
                        save_file(out_dict,os.getcwd()+"/safe_temp/"+k+".safetensors")
                        del w,out_dict,t1,t2,w1,w1
                        del_safe(k)
                        continue
                    if mode=="tensor1":
                        if t1.dim()==1:
                            t1[w1:w2]=t2[w1:w2]
                        elif t1.dim()==2:
                            t1[w1:w2,:]=t2[w1:w2,:]
                        elif t1.dim()==3:
                            t1[w1:w2,:,:]=t2[w1:w2,:,:]
                        elif t1.dim()==4:
                            t1[w1:w2,:,:,:]=t2[w1:w2,:,:,:]
                    else:
                        if t1.dim()==1:
                            t1[w1:w2]=t2[w1:w2]
                        elif t1.dim()==2:
                            t1[:,w1:w2]=t2[:,w1:w2]
                        elif t1.dim()==3:
                            t1[:,w1:w2,:]=t2[:,w1:w2,:]
                        elif t1.dim()==4:
                            t1[:,w1:w2,:,:]=t2[:,w1:w2,:,:]
                    out_dict[k]=t1.to(torch.float16)
                    del w1,w2

                save_file(out_dict,os.getcwd()+"/safe_temp/"+k+".safetensors")
                del w,out_dict,t1,t2
                del_safe(k)

        if win==None:
            print("")

        if win==None:
            print("making output")
        else:
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

        dict_sum=len(data_dict)
        key_count=0
        for k in data_dict:
            key_count=key_count+1
            if win==None:
                print("\r"+str(key_count)+"/"+str(dict_sum),end="")
            else:
                win["info"].update("making output "+str(key_count)+"/"+str(dict_sum))
            f=open(os.getcwd()+"/safe_temp/"+k+".safetensors","rb")
            l=int.from_bytes(f.read(8),byteorder="little")
            head=f.read(l)
            output.write(f.read())
            f.close()
            os.remove(os.getcwd()+"/safe_temp/"+k+".safetensors")
        output.close()
            
        f=open(out_path.replace(".safetensors",".txt"),"w")
        for i in range(len(ckpts)):
            f.write("ckpt"+str(i+1)+" : "+ckpts[i]+"\n")
        f.write("weight : "+str(w)+"\n")
        f.close()
        shutil.rmtree(os.getcwd()+"/safe_temp")
        del out_dict,l,head,n,offsets
        gc.collect()
        if win==None:
            print("")
            print(out_path)
        else:
            win["RUN"].Update(disabled=False)
            win["info"].update("fin")
            notification.notify(title="fin",message=out_path)
    except:
        if os.path.exists(os.getcwd()+"/safe_temp"):
            shutil.rmtree(os.getcwd()+"/safe_temp")
        if win==None:
            print("I failed in the output.")
        else:
            win["RUN"].Update(disabled=False)
            win["info"].update("error")
            notification.notify(title="error",message="I failed in the output.")

if __name__=="__main__":
    import FreeSimpleGUI as sg
    import tkinter as tk
    import pyperclip
    from plyer import notification
    import threading

    sg.theme('TealMono')
      
    keys=["ckpt1","ckpt2","out","w"]

    grp_rclick_menu={}
    for key in keys:
        grp_rclick_menu[key]=[
            "",
            [
                "-copy-::"+key,"-cut-::"+key,"-paste-::"+key
            ]
        ]

    lay=[
        [sg.Text("ckpt1"), sg.Input(key="ckpt1",right_click_menu=grp_rclick_menu["ckpt1"]),sg.FileBrowse( file_types=(('ckpt file', '.safetensors'),))],
        [sg.Text("ckpt2"), sg.Input(key="ckpt2",right_click_menu=grp_rclick_menu["ckpt2"]),sg.FileBrowse( file_types=(('ckpt file', '.safetensors'),))],
        [sg.Text("weight of ckpt2"),sg.Input(key="w",right_click_menu=grp_rclick_menu["w"])],
        [
            sg.Radio('NORMAL', key='normal',default=True,group_id='destination'),
            sg.Radio('TENSOR1', key='tensor1',default=False,group_id='destination'),
            sg.Radio('TENSOR2', key='tensor2',default=False,group_id='destination')
        ],
        [sg.Text("output path"), sg.Input(key="out",right_click_menu=grp_rclick_menu["out"]),sg.FileSaveAs(file_types=(('ckpt file', '.safetensors'),))],
        [sg.Text("infomation",key="info")],
        [sg.Button('RUN', key='RUN'),sg.Button('EXIT', key='EXIT')]
    ]

    window = sg.Window('Merge Ckpt Anima', lay)

    while True:
        event, values = window.read()
            
        if event == sg.WINDOW_CLOSED:
            break
        elif event=="EXIT":
            break
        elif event=="RUN":
            if values["out"]!="" and values["ckpt1"]!="" and values["ckpt2"]!="":
                out_path=values["out"]
                try:
                    weight=float(values["w"])
                except:
                    weight=0.5

                if values["normal"]:
                    mode="normal"
                elif values["tensor1"]:
                    mode="tensor1"
                else:
                    mode="tensor2"

                thread1 = threading.Thread(target=mergeckpt,args=(ckpts,weight,out_path,mode,window))
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
