import FreeSimpleGUI as sg
import os
from safetensors.torch import save_file,load_file
import torch
import tkinter as tk
import pyperclip
from plyer import notification
import threading

def mergeckpt(ckpts,weights,v,out_path,win):
    win.find_element('RUN').Update(disabled=True)
    if not(out_path.endswith(".safetensors")):
        win.find_element('RUN').Update(disabled=False)
        notification.notify(title="error",message=path+" does not exist.",timeout=8)
        return
    for path in ckpts:
        if not(os.path.exists(path)):
            win.find_element('RUN').Update(disabled=False)
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
        win.find_element('RUN').Update(disabled=False)
        notification.notify(title="fin",message=out_path,timeout=8)
    except:
        win.find_element('RUN').Update(disabled=False)
        notification.notify(title="error",message="I failed in the output.",timeout=8)
        
keys=[
    "ckpt1","ckpt2","ckpt3","ckpt4","ckpt5","ckpt6","ckpt7",
    "w1","w2","w3","w4","w5","w6","w7","out"
]
grp_rclick_menu={}
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

icon_path=b"iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAACXBIWXMAAAsSAAALEgHS3X78AAATq0lEQVR4XuVbe7BdZXVf5/2+53EfSe7N6+YNCIQkajEEMjqE6eDYlo4yji8E6dApoIBBgtAkpSASLBJx0GJQ7B/O2JaZSgcLpe3oVJBMmyoCSeDGcG+S+z7nntc+77NPf7+197kgWuTu7Dg43WFzz2Pvb3/r9Vu/tb7veDqdjvx/PvxOhfd4PE5v/Y33ffbaqx7umOZP9x/47mOnM/BCDeo9nYc5uffuXbeE33zfLddd0+/z+Tb6/P5/dzLm6dzj2ANO46F7br/xz4cr1eqBrz7ynac5Trvd3ow/r+H9ie64+3bvkmq1GoFi6l/80lfM03jeW956RhSw48LN8r6tW7dg8nE8fRpumTVNM7fn/v1NfBbA+4+02u0P33DNJ/8Bn9+B8z1er/f5L9/5heXNZnM73r+30Wicj2uTuP9PcI78Ximgt6/PC4x4AEJdBGGreF3C37E9n79xFAJuEOAHL8BnH4Zg28xOp+X3eidbrdYX8XGf7RW4zPMyzlNnSniOe0Yw4HtPPGXCej8gIAUCwUgwGB7w+wNb/H7/n3q9nnMAdsLcg2t4LoYylvr9vi1mx+xDOEgXyOAJP7nj3r+pnkkF0AqOxn+rLHDjZz7Vn0mlrggE/F+FgGF4PSzulbbZRlwbUqlUxahWxef1CpQiEBSK8ks6lZZms6EKgPdwXkfx96V8oVhHSLTxTIbSWDLZ83JPIv7Mzr33tt48+YXK4yoGfO7aq4Yx4U9jUh+HNYctVzal1arR63FYqZPK40khOWG+DviDEg5HCYj4zPICfL8e79cjNPQ9Pca+/xko7xlHlnvTTa4o4PpPfzyIyV2FsXe3TXOQYnYtgRBgGMCydSiiabm936dC81DBvD5JpTLSaNTUA6gYXufF5x54TjKZklqtJtVaBTdIuVar7775K1/7Nes7UYgrCsBk3w+BvmlLJMxZXQVQcJ4UmK6ulqXwduipN3g9CIsywsEroVAIimqp54RCEQosjaahn5mmhuv++77+t886EfY33eMWCD4LCe+BcKcoUFd4AJ4+k4ID/efd2GuBnyqF1yK+pWyU573CmmhH6vWqlCslyWZnYX1gIZXW6Rg7tl1oxYILhysKgIB9OF/AfH7BKVLsdqsNoVtqOR50627sU5Bu7CfjUdvlvbaiWq9fh/vC8IL+/kWSiCcIHhzqri2bNj4OzvAHLsgvjrMA4n45EPxSCLYDwgxCAT/DhN4LBWyGdN5IOCzxeGx+jkT5jWuG5JUTM5LNl6RYLsuiTI9ctHGdPPHj/4FwXsmkEtKbjMvUXFnDhJ7jAQ7UqjWEQRMZpKLKScTjEotF6xj8STz7YVz3I6TLhvrNArOaYw/ARPbjeVsxo8fgzn8IRfwSn51HYzNSedJgtDyIjgwvzshZKxZJKh7WSZqI8QvWr5Cl/SlJJ6IQsCWDfUnZfv4qGUjFBExRR2nUa1IqlxAOtW5qxHiKJSFcQJb4NTx3kVNvcKwACP3RB7/12NUPfPPRJ4H8f4wJ7IOwIeZ2PRHjNAaBKxr0y7tWLdYsOJCOSxPCLVuUkeUDSQhYl5VQTh1/B3t7JAhPOW94kQR8nFo3G1hKpOKo2C5ZwvsjeOYnUCvM1xALVYTjLADhlaH91a2fu7hULj8EC/rI8KLRmMYrCY5asFGXFRAwEvCp1TOwNmivvGv1oLo5MWJ4sE+Ojk1JPyxfgbv3pxNQSlpGxnP0IwtD3gCuqgX1MM8SKOHue27//Dj+/gue+QN8jFz59g/HHsBH7N352X5Y4FvI18kusWkhVrO5rMzMTltui/kPD/bCfUl+RFKxkJw9vFhWLukTP6wcDAZU4A9uPVfi4aDGPWOeXkFParctEtTlDeFQGLwioNfBExgGj+P8PsMB1ePfv33RrStPSwHQ+B4oYC0Fo50U6SEoJ62pD8QnGvKr9acBbGOTOXn1xBTioi0Ts3krq+HO4xNZGcV3h0cnZWauJKVKTZLAikQkCPpsWsJq/dBBiARBjNL4G2amCSO9/iUU8QLC4Epcd+tCFeA4C8DtLoRVfgLreIrFksZof9+AunQ2N6uTTiALeABYwHOZK8EzWQ8gDHhEYe2PXfYedfG/++FzUm0gXeq1SH3wikwyIW2SIFxfKBS7KRBKMyUSjkgmzbqB3EIB8f7d+x7cyXF/Z1kAzyrhvB8PPMSQpMC5uazMZmf0taI/ADBXLEuuVIWCKB8AMRRQaw4C/SPhkERh5RWLe/Ed8APvOVYNyjg1k5OZXEE/15rBZpGkxgRBYguFt/nF1Xfe9BdA2YUfjkEQLvciHncrQPDpLs2tY1Ks+jSFIS6YCeKRsGw5a4VMzhbwnuA3pDxg1RDKfju2379lvWaFvlRcDh+fQOi0wAdicugowR3+w7SJM4TYt5TNFGmFHJ7xQ7x8DuHoqKx1HAIU+q9vu3k7Hv502TACRG9OiAxwCNZ99znDALiQBIEy/SA3LXzOg0rwgvPPwrq8Z2hJvwoJGwMaEOUa8x2tCyayBWnbCjj48nEp1zuaYYrFAuqEgI7HjhLc/+6u7X9nIXDl5ZcyBj+DCQRCQbC+GPI7YtLn88jFF6yVVUhtQ8jrfYjlZg01PoiOtABmzbYKWqs3FRPo0hS6xe/g+mYDIiPum/h+AF4wmI7J8JJeuWTTei2YcnM5SSSSKJ0jmhngAR+EIRzXBo6zwOrVq3sxgQ+wZA2B9rKgoQcwX3frfFqjXm/A0tZJ/0UyhNk6wAF0i1AWd6/VcLD/dVqoGDtQjkAuD/gD41/PjhgVQ6ZnJrU8tkFvA753FP8aRguHDeuOeDx+bjAYHGB+LxbzCkp+xHwDrv7cL34peQBfDcJTIWR3YZzqnt2cCQArI92dnJiBkpqv9wcgqK0pZYxVjJErlOXZF45pXZDqSek4TLX2wcbpaqdyOAZBTOLcrgLfWAIHwACPjc/IOPJ8sichm9ctlw0AvBZDAEcVDQ8//vX1pmR2rqhEiGcDwNdCeETwut5oShChNDZdkEMjJ5EGS1IGXoTRKwjD26w2mlVe49nEsQsx9I+dKMGxByDf+xDzLIFn2NGxuH9HM0AIZIXFTbFckSkgvkkWCME8AK4qQsXK5SFZNjggfWCBfnzHAKjgOw/qhg6AkvdMFQwoBvgA16cnVSoVuP+U9g4YTHheCye7zhfde8etjmRxdBM1jeLlYcTvJXh5kuyP3k0l9Pf2gxD147Vf3TpbRCggLLwQivm/BYDzKEcA2kNZTG8Uxk96C8XwfRjcoA4FTiB1dlMlnxkIBnXMhjZOlVCRi3wI89gNRTiSxdFNfPI9+79RpwIwIT+KnDJjMhqxmhtT06C7sCm5PF335PScKsAKf9iak+fJ91rkAPk98xWOhsR4tiQ1KCGCMa22WbfCpK9QZwq27EBdBk5yaNfd+xz1CB0rgHOAFS+H5v8RltOHF0tFLYKY63szfTpxsraXQG6Y86mQWCSEpqhf6tW6FGfmcOakbljtrlg0rPcS+EbGs1YDFeGVSfcCSMEg2R2mau2mBxRwH16/e/ctN2xxEv+8x7EC7vrCTT4IuBxj/BvOsrJBMja4toEYpSJYFwT8ICydpkxlpzWVsUtEAcZfHZPJYydl+vi4jL14DPygadUOMO/YyVFwgqImRXaCOZ4ancWWrQCMwRg4DAWjGvXuRIvMERdwnAW44IGHnwQiv4IJTMIyS5tIhWOjxzuGYXiGhpbK0NByScd9EvEZcvzEawBHH+huL6zqkeHz12vlyFChQZVCQwmjEP7wyIgyPX/YJ1nIXgDzo+W1m2y7P/7k4C3jXDjBOQiFkBpadHMBh2MF4BkGyM91d375gdZtN1z3cyjEnJyYeapcKt4OwXz5fF5WrVwmy/qZEYCU41MylZuRY6NjsnblGkkne8RDGyoGCKjxnLx49CUxsYjCTBFGNogG6mKATrP+J8tEX7CD5zDz8f2rmUxm6tqbbqPQ9y1A5l+51LECbr/nfg6ksY/iZE8sGitOTJy8mNRNBYPNVi6JI2fTkbk05pEy0mKhkJefHzkiZ63ZMB/T1MDhkSPwgIp6BxkfkjvCoi6ZaBBVodU+h/AP4/xvvLkC1/ynLbxT2fU+xwp441Pv+/ojJ/l++5Zz6tUam7Ws933o/7ErDPSGEjKZtNArYsj/WTBHEEPpiaHmR/ZgFVmpliWJblEeSmKvgGb1oGaIhNBUCSLtNrw59An3Yw/BUXz16E3XXeMYv944d1cG6Q6IMtbyCDYv2dL2BZHfI1AAGxhsfLJz7YF7+2RyZlpSKHZ60z0olQmQVhZj3yCObMB2GZfQiHW9PYEGGik328LrdQ9844ArmyZcVQBik6wMc7YUYCAtFlC95efmpAokD0Ig8gKslwPsxpTZsV44MX4KLbCY1MH6gkiRrCm63d8mAKQv4Zv7wKaBfz4tX/8/bnZVAXhGFev8WNqx6v5quSA5dIjGJ05JKZ+TEBWAAsjqFtdk9NS4TMATuGjKyrCCsjkC9Ge8F9BCI522u8IDoNhb3/EKODU9W0ZKapHnE/SYtJTy2u0sCsmzDBLUE43IyOhrcuTYCHAhqJ5BdhgiONhhxMYIuz+gzByKtNv1wxUQ7M7q+w/uvRzpDFkLRQ82QBhGDlZl7657hQdAF5FsyVAFGAC8stGUxZmkFPCaLJHcgP8YBkx30VhUYokESdDmxx/a6b3i+n2uxH53Rq4p4OiTBy4mNU1Eo/5S2cAiR1qmZlAqZrPzVSIFogfQykXsEuGRAOCxQOLJbjCXVqkEAmkC7e/exUOaHs1Wcw32GWRwy6ybbuAaBsBCl2HmIU4+BiuH4NbNttXo0LVAAqM2MtoSCwWxRaauMR7F6wqAkKmPtQJRv7sjhPsK6thSw6oRxwBGIPV29XBRAb5+ghh7fCGgPCu67nKWtaoD4VEWk88HQYnZJSLg8WAbnDhgLZRYnF87ysmMRBMpq1vk8QTgGytdlR6DuaYAuH+RS+CkrDzI64ulklqdp99umlAVDXR+mlBGAi1zKo0lAas+9gO0X8AB8L9iblpKc9O6rMbeI7xj4B2rAAj5uN/nyUYAZLQ+KzgDXVx6BF379X6AR5sdjHcKxozAxRJ2kprgAcoBlEh1VJkVA0pEczQQ1q7zUrcV4BoIrrn0E89+90vX/9Mro7NXt0FhfZ6mDA3ELUJjd4J0dQhk0YDVuSeArk9Lh4EDKjTXDrjUxYCx46EFIlTG0lg41pG52ezrOy5c0oRrCuB8xienx57/2WF0cQ05d92QLFvco90eHixygnBjCs+1A7LCWXR7GfsMfjLH+UII19ITGFK6VIZ9QvWawbbZr220Pl09uKqAeqP1PFwdoN32hsEFsHdOqmiGVGuwIkpbZoIimCCJUr5clZJRh6ew+lOT28tp1toCO8OaFvFfFa/9VIrD5a+3UpKrCjj40msHq7XGBNB6iHkctBhCNuTlkalfaWZY2UGbAfNpUlMlPrN2l1gN1LOxiWL5EpAjWwJ4CdcAXD1cVUCBfLbTqWiXGCAOkDsOSwLPOhu5LG6xPNBjuwliJT0r3vnP2k6LDIISmZmCCul+TzDlpa5Kj8FcVQBYHrJfq8ZS9tiJmYe2XzC8t1ZrrN189rKDEDLKyTMMlBbwf7Y47AlwsYNC6s4wfFdEo7QH/QGSIO4XsnuB+Xe0AtC5NcHuuKmZa3/tj93xaOPRvVc3aOB4TxrCtKQKQKOgmhZtabh3qIv89AamxzAaKlQk0yPDid6BcRe0/+ftKMtVDyDNRXyb5AE4FP7D0RiInceTxIJJA7s9TTQ+aiiUuq1te3lLd4MgOdjr/6JLZFxkiYIskRPU6rXR7Fzuibcj1EKucVkB6r5qWMis1U4ynVKH5+73IAhPTzqpXL9iGOrqdH0udWgXCRf67DBgptiwdh0yRhSVJdYU/L6ZDdu2Pfdnd31nIfL91mtdVQBQ32c2zIBNfhTBENMo5j3tRt1QS3p9ISihDwIjTFDscEmMCqH1iQv8o/sMdLOVWUS2IDtK45pFY/91EPtlZe63SrWAC1yrBfjMVKrHxC8bTO4GQSrTratYI0ST3No2TYuT0we4cwQnl9FodpIeJT7wBmIDCZCuE7SaTyBW/tVeE5iAt7iOAa4qwDCqGM+jXgVrdxsX3P3i6XaFOiZ2hheyYHaoE6gAe31QN1UBQ1gxkifg1yX1yanJR4ARfXZ1+e2hbVdaLWcXD1dDADIT/bB5UdftsbTLba1o+KES9mHJXHsDAEFFegvV7eU0iwfwsLIjrzNrJaOUxbVL8OEubI444KLc80O5qgDOH96uK1ggPZqzG7U6fiZn+iN+5HT4RCk/pztBaW2KXMXuEPYGSIl1mQyK4QZLbqcF8lcRQn8EGBlBseU6CeL8XFWA7g/kJmerCaitq3wux2d4zQ72A2HPACs7/mBCra1rBNY+AnaHKLTHizEIhvys0Wit3vHJ0TNh+e6YrioACx/4YaS3RlqPWDb4EHR2mvxdoFHMMwYkjQWSOFaETk1wMcn+dZg2TOwfVNAqdk/AzqhnUn73OkKcJep6+jWb+W2EgLaG4AvKB2BNMx6J/sd555zfWrtqrSxetGRW9wG3208Nr1j1JDdX8Gd1iHn8nNaDjQ/eGFDf6pmdwcPVLIByGKwWuwCRApDDNQ6wqQFytX+Kjz+66bxNu5AGdXPmmuE13wNz/FS+kP/I6pWrD61fs17WrVpnXvK+7bsApjuguh+hrmDeP6PH/wLttVIc/UoLOQAAAABJRU5ErkJggg=="
window = sg.Window('merge ckpt', layout,icon=icon_path)

while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    elif event=="EXIT":
        break
    elif event=="RUN":
        c=0
        for i in range(7):
            if values["ckpt"+str(i+1)]!="":
                c=c+1
        if values["out"]!="" and c>0:
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
            thread1 = threading.Thread(target=mergeckpt,args=(ckpts,weights,v,out_path,window))
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
