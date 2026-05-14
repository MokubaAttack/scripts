import os,shutil,json,numpy,torch,gc,random
from safetensors.torch import save_file,load_file

def spckpt(path,folder,ind):
    f=open(path,"rb")
    l=int.from_bytes(f.read(8),byteorder="little")
    head=f.read(l).decode()
    head=json.loads(head)
    if "__metadata__" in head:
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

def del_safe(k):
    if os.path.exists(os.getcwd()+"/safe_temp/1_"+k+".safetensors"):
        os.remove(os.getcwd()+"/safe_temp/1_"+k+".safetensors")
    if os.path.exists(os.getcwd()+"/safe_temp/2_"+k+".safetensors"):
        os.remove(os.getcwd()+"/safe_temp/2_"+k+".safetensors")
    gc.collect()

def mergeckpt(ckpts,weights,v,out_path,mode="normal",dp=0,seed=0,win=None):
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

    if mode=="dare":
        if dp<=0 or dp>0.5:
            if win==None:
                print("0 < Dropout probability <= 0.5")
            else:
                win["RUN"].Update(disabled=False)
                win["info"].update("error")
                notification.notify(title="error",message="0 < Dropout probability <= 0.5")
            return
    else:
        if not(mode in ["normal","mokuba","tensor1","tensor2"]):
            if win==None:
                print("You must choose one of normal or dare or mokuba or tensor1 or tensor2.")
            else:
                win["RUN"].Update(disabled=False)
                win["info"].update("error")
                notification.notify(title="error",message="You must choose one of normal or dare or mokuba or tensor1 or tensor2.")
            return

    try:
        safe=open(os.getcwd()+"/data.txt","r")
        if os.path.exists(os.getcwd()+"/safe_temp"):
            shutil.rmtree(os.getcwd()+"/safe_temp")
        os.mkdir(os.getcwd()+"/safe_temp")
        if win==None:
            print("making initial ckpt")
        else:
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

        if mode=="dare":
            try:
                seed=int(seed)
            except:
                seed=0
            if seed==0:
                seed=random.randint(1,2**31-1)
            numpy.random.seed(seed)

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
                    t1=torch.zeros(data_dict[k]).to(torch.float32)

                if os.path.exists(os.getcwd()+"/safe_temp/2_"+k+".safetensors"):
                    t2=load_file(os.getcwd()+"/safe_temp/2_"+k+".safetensors")[k].to(torch.float32)
                else:
                    t2=torch.zeros(data_dict[k]).to(torch.float32)

                if k.startswith("model.diffusion_model.middle_block."):
                    w=weights[10][1]
                elif k.startswith("model.diffusion_model.input_blocks."):
                    j=int(k.split(".")[3])
                    w=weights[1+j][1]
                elif k.startswith("model.diffusion_model.output_blocks."):
                    j=int(k.split(".")[3])
                    w=weights[11+j][1]
                elif k.startswith("model.diffusion_model.label_emb.") or k.startswith("model.diffusion_model.time_embed."):
                    w=weights[1][1]
                elif k.startswith("model.diffusion_model.out."):
                    w=weights[19][1]
                else:
                    if k.startswith("first_stage_model."):
                        if v==0:
                            out_dict[k]=t1.to(torch.float16)
                            save_file(out_dict,os.getcwd()+"/safe_temp/"+k+".safetensors")
                            del out_dict
                            del_safe(k)
                            continue
                        elif v==1:
                            out_dict[k]=t2.to(torch.float16)
                            save_file(out_dict,os.getcwd()+"/safe_temp/"+k+".safetensors")
                            del out_dict
                            del_safe(k)
                            continue
                        else:
                            w=weights[0][1]
                    else:
                        w=weights[0][1]

                if t1.dim() in (1, 2):
                    dw = t2.shape[-1] - t1.shape[-1]
                    if dw > 0:
                        t1 = torch.nn.functional.pad(t1, (0, dw, 0, 0))
                    elif dw < 0: 
                        t2 = torch.nn.functional.pad(t2, (0, -dw, 0, 0))
                    dh = t2.shape[0] - t1.shape[0]
                    if dh > 0:
                        t1 = torch.nn.functional.pad(t1, (0, 0, 0, dh))
                    elif dh < 0:
                        t2 = torch.nn.functional.pad(t2, (0, 0, 0, -dh))
                    del dw,dh

                if mode=="dare":
                    dt=t2-t1
                    m = torch.from_numpy(numpy.random.binomial(1, dp, dt.shape)).to(torch.float32)
                    out_dict[k] = (t1 + w * m * dt / (1 - dp)).to(torch.float16)
                    del dt,m

                elif mode=="normal":
                    out_dict[k]=((1-w)*t1+w*t2).to(torch.float16)

                elif mode=="mokuba":
                    dt=t2-t1
                    dt_mean=torch.mean(dt).item()
                    dt_std=torch.std(dt).item()
                    dt[dt>dt_mean+2*w*dt_std]=dt_mean+2*w*dt_std
                    dt[dt<dt_mean-2*w*dt_std]=dt_mean-2*w*dt_std
                    out_dict[k] = (t1 + w * dt).to(torch.float16)
                    del dt,dt_mean,dt_std

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

        if type(v) is str:
            if os.path.exists(v):
                if win==None:
                    print("baking vae")
                else:
                    win["info"].update("baking vae")
                vae_dict=load_file(v)
                for k in vae_dict:
                    out_dict={}
                    out_dict["first_stage_model."+k]=vae_dict[k].to(torch.float16)
                    save_file(out_dict,os.getcwd()+"/safe_temp/first_stage_model."+k+".safetensors")
                    del out_dict
                del vae_dict
                gc.collect()
            else:
                v=-1

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

        dict_sum=len(list(data_dict))
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
        if type(v) is str:
            f.write("vae : "+v+"\n")
        else:
            if v!=-1:
                f.write("vae : "+ckpts[v]+"\n")
            else:
                f.write("vae : None\n")
        for i in range(20):
            weights[i]=weights[i][1]
        f.write("weight : "+str(weights)+"\n")
        if mode=="dare":
            f.write("Dropout probability : "+str(dp)+"\n")
            f.write("seed : "+str(seed)+"\n")
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
      
    keys=["ckpt1","ckpt2","out","dp","seed","vaefile"]
    for i in range(21):
        keys.append("w"+str(i))
    grp_rclick_menu={}
    for key in keys:
        grp_rclick_menu[key]=[
            "",
            [
                "-copy-::"+key,"-cut-::"+key,"-paste-::"+key
            ]
        ]

    def mk_lay():
        lay=[
            [sg.Text("ckpt1"), sg.Input(key="ckpt1",right_click_menu=grp_rclick_menu["ckpt1"]),sg.FileBrowse( file_types=(('ckpt file', '.safetensors'),)),sg.Text("vae"),sg.Radio("",key="v1",group_id='destination')],
            [sg.Text("ckpt2"), sg.Input(key="ckpt2",right_click_menu=grp_rclick_menu["ckpt2"]),sg.FileBrowse( file_types=(('ckpt file', '.safetensors'),)),sg.Text("vae"),sg.Radio("",key="v2",group_id='destination')],
            [sg.Checkbox('BLOCK', key='block',default=False,enable_events=True)],
            [sg.Text("weight of ckpt2",key="www")],
            [sg.Input(key="w0",right_click_menu=grp_rclick_menu["w0"])],
            [
                sg.Frame("BASE",[[sg.Input(key="w1",right_click_menu=grp_rclick_menu["w1"], size=(10, 1))]],key="base"),
                sg.Frame("IN00",[[sg.Input(key="w2",right_click_menu=grp_rclick_menu["w2"], size=(10, 1))]],key="i0"),
                sg.Frame("IN01",[[sg.Input(key="w3",right_click_menu=grp_rclick_menu["w3"], size=(10, 1))]],key="i1"),
                sg.Frame("IN02",[[sg.Input(key="w4",right_click_menu=grp_rclick_menu["w4"], size=(10, 1))]],key="i2"),
                sg.Frame("IN03",[[sg.Input(key="w5",right_click_menu=grp_rclick_menu["w5"], size=(10, 1))]],key="i3"),
                sg.Frame("IN04",[[sg.Input(key="w6",right_click_menu=grp_rclick_menu["w6"], size=(10, 1))]],key="i4"),
                sg.Frame("IN05",[[sg.Input(key="w7",right_click_menu=grp_rclick_menu["w7"], size=(10, 1))]],key="i5"),
                sg.Frame("IN06",[[sg.Input(key="w8",right_click_menu=grp_rclick_menu["w8"], size=(10, 1))]],key="i6"),
                sg.Frame("IN07",[[sg.Input(key="w9",right_click_menu=grp_rclick_menu["w9"], size=(10, 1))]],key="i7"),
                sg.Frame("IN08",[[sg.Input(key="w10",right_click_menu=grp_rclick_menu["w10"], size=(10, 1))]],key="i8")
            ],
            [
                sg.Frame("MID",[[sg.Input(key="w11",right_click_menu=grp_rclick_menu["w11"], size=(10, 1))]],key="mid"),
                sg.Frame("OUT00",[[sg.Input(key="w12",right_click_menu=grp_rclick_menu["w12"], size=(10, 1))]],key="o0"),
                sg.Frame("OUT01",[[sg.Input(key="w13",right_click_menu=grp_rclick_menu["w13"], size=(10, 1))]],key="o1"),
                sg.Frame("OUT02",[[sg.Input(key="w14",right_click_menu=grp_rclick_menu["w14"], size=(10, 1))]],key="o2"),
                sg.Frame("OUT03",[[sg.Input(key="w15",right_click_menu=grp_rclick_menu["w15"], size=(10, 1))]],key="o3"),
                sg.Frame("OUT04",[[sg.Input(key="w16",right_click_menu=grp_rclick_menu["w16"], size=(10, 1))]],key="o4"),
                sg.Frame("OUT05",[[sg.Input(key="w17",right_click_menu=grp_rclick_menu["w17"], size=(10, 1))]],key="o5"),
                sg.Frame("OUT06",[[sg.Input(key="w18",right_click_menu=grp_rclick_menu["w18"], size=(10, 1))]],key="o6"),
                sg.Frame("OUT07",[[sg.Input(key="w19",right_click_menu=grp_rclick_menu["w19"], size=(10, 1))]],key="o7"),
                sg.Frame("OUT08",[[sg.Input(key="w20",right_click_menu=grp_rclick_menu["w20"], size=(10, 1))]],key="o8")
            ],
            [
                sg.Checkbox('DARE', key='dare',default=False,enable_events=True),
                sg.Checkbox('TENSOR1', key='tensor1',default=False,enable_events=True),
                sg.Checkbox('TENSOR2', key='tensor2',default=False,enable_events=True),
                sg.Checkbox('MOKUBA', key='mokuba',default=False,enable_events=True)
            ],
            [
                sg.Text("Dropout probability"), sg.Input(key="dp",size=(10, 1),right_click_menu=grp_rclick_menu["dp"]),
                sg.Text("seed"), sg.Input(key="seed",right_click_menu=grp_rclick_menu["seed"])
            ],
            [sg.Checkbox('bake vae', key='bakevae',default=False,enable_events=True)],
            [sg.Text("vae path"), sg.Input(key="vaefile",right_click_menu=grp_rclick_menu["vaefile"]),sg.FileBrowse( file_types=(('vae file', '.safetensors'),))],
            [sg.Text("output path"), sg.Input(key="out",right_click_menu=grp_rclick_menu["out"]),sg.FileSaveAs(file_types=(('ckpt file', '.safetensors'),))],
            [sg.Text("infomation",key="info")],
            [sg.Button('RUN', key='RUN'),sg.Button('Cancel Vae', key='cancel'),sg.Button('EXIT', key='EXIT')]
        ]
        return lay

    def lay_che(b,d,v,win):
        win["ckpt1"].hide_row()
        win["ckpt2"].hide_row()
        win["block"].hide_row()
        win["www"].hide_row()
        win["w0"].hide_row()
        win["base"].hide_row()
        win["mid"].hide_row()
        win["dare"].hide_row()
        win["dp"].hide_row()
        win["bakevae"].hide_row()
        win["vaefile"].hide_row()
        win["out"].hide_row()
        win["info"].hide_row()
        win["RUN"].hide_row()
        win["ckpt1"].unhide_row()
        win["ckpt2"].unhide_row()
        win["block"].unhide_row()
        win["www"].unhide_row()
        if b:
            win["base"].unhide_row()
            win["mid"].unhide_row()
        else:
            win["w0"].unhide_row()
        win["dare"].unhide_row()
        if d:
            win["dp"].unhide_row()
        win["bakevae"].unhide_row()
        if v:
            win["vaefile"].unhide_row()
        win["out"].unhide_row()
        win["info"].unhide_row()
        win["RUN"].unhide_row()

    layout=mk_lay()
    window = sg.Window('Merge Ckpt', layout)

    event, values = window.read(timeout=0)
    lay_che(False,False,False,window)

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
                if values["bakevae"]:
                    v=values["vaefile"]
                ckpts=[values["ckpt1"],values["ckpt2"]]
                weights=[]
                if values["block"]:
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
                else:
                    if values["w0"]=="":
                        r=0.5
                    else:
                        try:
                            r=float(values["w0"])
                        except:
                            r=0.5
                    window["w0"].update(str(r))
                    for i in range(20):
                        weights.append([1-r,r])

                if values["dare"]:
                    try:
                        dp=float(values["dp"])
                    except:
                        dp=0.5
                    try:
                        seed=int(values["seed"])
                    except:
                        seed=0
                    window["dp"].update(str(dp))
                    window["seed"].update(str(seed))
                    mode="dare"

                elif values["mokuba"]:
                    dp=0
                    seed=0
                    mode="mokuba"

                elif values["tensor1"]:
                    dp=0
                    seed=0
                    mode="tensor1"
                elif values["tensor2"]:
                    dp=0
                    seed=0
                    mode="tensor2"

                else:
                    dp=0
                    seed=0
                    mode="normal"

                thread1 = threading.Thread(target=mergeckpt,args=(ckpts,weights,v,out_path,mode,dp,seed,window))
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
        elif "mokuba"==event or "dare"==event or ("tensor" in event) or event=="block" or event=="bakevae":
            try:
                if event=="dare":
                    if values["mokuba"]:
                        window["mokuba"].update(False)
                        values["mokuba"]=False
                    elif values["tensor1"]:
                        window["tensor1"].update(False)
                        values["tensor1"]=False
                    elif values["tensor2"]:
                        window["tensor2"].update(False)
                        values["tensor2"]=False
                elif event=="mokuba":
                    if values["dare"]:
                        window["dare"].update(False)
                        values["dare"]=False
                    elif values["tensor1"]:
                        window["tensor1"].update(False)
                        values["tensor1"]=False
                    elif values["tensor2"]:
                        window["tensor2"].update(False)
                        values["tensor2"]=False
                elif event=="tensor1":
                    if values["dare"]:
                        window["dare"].update(False)
                        values["dare"]=False
                    elif values["mokuba"]:
                        window["mokuba"].update(False)
                        values["mokuba"]=False
                    elif values["tensor2"]:
                        window["tensor2"].update(False)
                        values["tensor2"]=False
                elif event=="tensor2":
                    if values["dare"]:
                        window["dare"].update(False)
                        values["dare"]=False
                    elif values["tensor1"]:
                        window["tensor1"].update(False)
                        values["tensor1"]=False
                    elif values["mokuba"]:
                        window["mokuba"].update(False)
                        values["mokuba"]=False
                    
                lay_che(values["block"],values["dare"],values["bakevae"],window)
            except:
                pass

    window.close()
