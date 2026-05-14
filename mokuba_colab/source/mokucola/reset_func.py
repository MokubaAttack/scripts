import importlib
import sys
import pickle
import gc
import torch
import os

def reset_func(f,s="colab"):
    try:
        a1=f.__dict__
    except:
        return

    while True:
        if isinstance(a1,list):
            b1=a1[0]
        else:
            keys=list(a1.keys())
            try:
                b1=a1[keys[0]].__dict__
            except:
                b1=a1[keys[0]]
        if not(isinstance(b1,(dict,list))) or b1=={} or b1==[]:
            if isinstance(a1,dict):
                a1.pop(keys[0])
            else:
                a1.pop(0)
            a1=f.__dict__
            if a1=={}:
                break
        else:
            a1=b1
    
    if s=="colab":
        file=os.path.dirname(torch.__file__).replace("/torch","/mokucola/colab_default.txt")
    else:
        file=os.path.dirname(torch.__file__).replace("/torch","/mokucola/kaggle_default.txt")
    k = open(file,"rb")
    keep = pickle.load(k)
    k.close()

    for k in list(sys.modules.keys()):
        if not(k in keep or k.startswith("torch") or k.startswith("tensorflow") or k.startswith("compel")):
            try:
                importlib.reload(sys.modules[k])
            except:
                pass
    gc.collect()
    torch.cuda.empty_cache()
