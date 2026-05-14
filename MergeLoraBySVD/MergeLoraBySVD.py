from safetensors.torch import load_file,save_file
import itertools,json,os,shutil,re,torch

ok=[
    "lora_unet_out_2",
    "lora_unet_input_blocks_0_0",
    "lora_unet_label_emb_0_0",
    "lora_unet_label_emb_0_2",
    "lora_unet_time_embed_0",
    "lora_unet_time_embed_2"
]

CLAMP_QUANTILE = 0.99

ACCEPTABLE = [12, 17, 20, 26]
SDXL_LAYER_NUM = [12, 20]

LAYER12 = {
    "BASE": True,
    "IN00": False,
    "IN01": False,
    "IN02": False,
    "IN03": False,
    "IN04": True,
    "IN05": True,
    "IN06": False,
    "IN07": True,
    "IN08": True,
    "IN09": False,
    "IN10": False,
    "IN11": False,
    "MID": True,
    "OUT00": True,
    "OUT01": True,
    "OUT02": True,
    "OUT03": True,
    "OUT04": True,
    "OUT05": True,
    "OUT06": False,
    "OUT07": False,
    "OUT08": False,
    "OUT09": False,
    "OUT10": False,
    "OUT11": False,
}

LAYER17 = {
    "BASE": True,
    "IN00": False,
    "IN01": True,
    "IN02": True,
    "IN03": False,
    "IN04": True,
    "IN05": True,
    "IN06": False,
    "IN07": True,
    "IN08": True,
    "IN09": False,
    "IN10": False,
    "IN11": False,
    "MID": True,
    "OUT00": False,
    "OUT01": False,
    "OUT02": False,
    "OUT03": True,
    "OUT04": True,
    "OUT05": True,
    "OUT06": True,
    "OUT07": True,
    "OUT08": True,
    "OUT09": True,
    "OUT10": True,
    "OUT11": True,
}

LAYER20 = {
    "BASE": True,
    "IN00": True,
    "IN01": True,
    "IN02": True,
    "IN03": True,
    "IN04": True,
    "IN05": True,
    "IN06": True,
    "IN07": True,
    "IN08": True,
    "IN09": False,
    "IN10": False,
    "IN11": False,
    "MID": True,
    "OUT00": True,
    "OUT01": True,
    "OUT02": True,
    "OUT03": True,
    "OUT04": True,
    "OUT05": True,
    "OUT06": True,
    "OUT07": True,
    "OUT08": True,
    "OUT09": False,
    "OUT10": False,
    "OUT11": False,
}

LAYER26 = {
    "BASE": True,
    "IN00": True,
    "IN01": True,
    "IN02": True,
    "IN03": True,
    "IN04": True,
    "IN05": True,
    "IN06": True,
    "IN07": True,
    "IN08": True,
    "IN09": True,
    "IN10": True,
    "IN11": True,
    "MID": True,
    "OUT00": True,
    "OUT01": True,
    "OUT02": True,
    "OUT03": True,
    "OUT04": True,
    "OUT05": True,
    "OUT06": True,
    "OUT07": True,
    "OUT08": True,
    "OUT09": True,
    "OUT10": True,
    "OUT11": True,
}

assert len([v for v in LAYER12.values() if v]) == 12
assert len([v for v in LAYER17.values() if v]) == 17
assert len([v for v in LAYER20.values() if v]) == 20
assert len([v for v in LAYER26.values() if v]) == 26

RE_UPDOWN = re.compile(r"(up|down)_blocks_(\d+)_(resnets|upsamplers|downsamplers|attentions)_(\d+)_")

def get_lbw_block_index(lora_name: str, is_sdxl: bool = False) -> int:
    # lbw block index is 0-based, but 0 for text encoder, so we return 0 for text encoder
    if "text_model_encoder_" in lora_name:  # LoRA for text encoder
        return 0

    # lbw block index is 1-based for U-Net, and no "input_blocks.0" in CompVis SD, so "input_blocks.1" have index 2
    block_idx = -1  # invalid lora name
    if not is_sdxl:
        NUM_OF_BLOCKS = 12  # up/down blocks
        m = RE_UPDOWN.search(lora_name)
        if m:
            g = m.groups()
            up_down = g[0]
            i = int(g[1])
            j = int(g[3])
            if up_down == "down":
                if g[2] == "resnets" or g[2] == "attentions":
                    idx = 3 * i + j + 1
                elif g[2] == "downsamplers":
                    idx = 3 * (i + 1)
                else:
                    return block_idx  # invalid lora name
            elif up_down == "up":
                if g[2] == "resnets" or g[2] == "attentions":
                    idx = 3 * i + j
                elif g[2] == "upsamplers":
                    idx = 3 * i + 2
                else:
                    return block_idx  # invalid lora name

            if g[0] == "down":
                block_idx = 1 + idx  # 1-based index, down block index
            elif g[0] == "up":
                block_idx = 1 + NUM_OF_BLOCKS + 1 + idx  # 1-based index, num blocks, mid block, up block index

        elif "mid_block_" in lora_name:
            block_idx = 1 + NUM_OF_BLOCKS  # 1-based index, num blocks, mid block
    else:
        # SDXL: some numbers are skipped
        if lora_name.startswith("lora_unet_"):
            name = lora_name[len("lora_unet_") :]
            if name.startswith("time_embed_") or name.startswith("label_emb_"):  # 1, No LoRA in sd-scripts
                block_idx = 1
            elif name.startswith("input_blocks_"):  # 1-8 to 2-9
                block_idx = 1 + int(name.split("_")[2])
            elif name.startswith("middle_block_"):  # 13
                block_idx = 13
            elif name.startswith("output_blocks_"):  # 0-8 to 14-22
                block_idx = 14 + int(name.split("_")[2])
            elif name.startswith("out_"):  # 23, No LoRA in sd-scripts
                block_idx = 23

    return block_idx

def format_lbws(lbws):
    try:
        # lbwは"[1,1,1,1,1,1,1,1,1,1,1,1]"のような文字列で与えられることを期待している
        lbws = [json.loads(lbw) for lbw in lbws]
    except Exception:
        raise ValueError(f"format of lbws are must be json / 層別適用率はJSON形式で書いてください")
    assert all(isinstance(lbw, list) for lbw in lbws), f"lbws are must be list / 層別適用率はリストにしてください"
    assert len(set(len(lbw) for lbw in lbws)) == 1, "all lbws should have the same length  / 層別適用率は同じ長さにしてください"
    assert all(
        len(lbw) in ACCEPTABLE for lbw in lbws
    ), f"length of lbw are must be in {ACCEPTABLE} / 層別適用率の長さは{ACCEPTABLE}のいずれかにしてください"
    assert all(
        all(isinstance(weight, (int, float)) for weight in lbw) for lbw in lbws
    ), f"values of lbs are must be numbers / 層別適用率の値はすべて数値にしてください"

    layer_num = len(lbws[0])
    is_sdxl = True if layer_num in SDXL_LAYER_NUM else False
    FLAGS = {
        "12": LAYER12.values(),
        "17": LAYER17.values(),
        "20": LAYER20.values(),
        "26": LAYER26.values(),
    }[str(layer_num)]
    LBW_TARGET_IDX = [i for i, flag in enumerate(FLAGS) if flag]
    return lbws, is_sdxl, LBW_TARGET_IDX

def merge_lora_models_lowmem(models, ratios, lbws, new_rank, new_conv_rank, device, merge_dtype,win,mem_limit):
    if os.path.exists("temp"):
        shutil.rmtree("temp")
    os.mkdir("temp")

    if lbws:
        lbws, is_sdxl, LBW_TARGET_IDX = format_lbws(lbws)
    else:
        is_sdxl = False
        LBW_TARGET_IDX = []

    keys=[]
    n=0
    keys.append([])
    for model in models:
        lora_sd = load_file(model)
        all_keys=[]
        for m in range(len(keys)):
            all_keys=all_keys+keys[m]
        for key in list(lora_sd.keys()):
            if key.endswith(".lora_down.weight"):
                if not(key.replace(".lora_down.weight","") in all_keys):
                    if len(keys[n])>=mem_limit:
                        n=n+1
                        keys.append([])
                    keys[n].append(key.replace(".lora_down.weight",""))
    del n,all_keys
    key_sum=0
    for m in range(len(keys)):
        key_sum=key_sum+len(keys[m])
    key_count=0
    if win==None:
        print("load lora")
    lora_sds=[]
    for model in models:
        lora_sds.append(load_file(model))
    for m in range(len(keys)):
        merged_sd={}
        for key in keys[m]:
            key_count=key_count+1
            if win!=None:
                win["info"].update("load lora : "+str(key_count)+"/"+str(key_sum))
            else:
                print("\r"+str(key_count)+"/"+str(key_sum),end="")
            for model, ratio, lbw in itertools.zip_longest(lora_sds, ratios, lbws):
                lora_sd = model

                if lbw:
                    lbw_weights = [1] * 26
                    for index, value in zip(LBW_TARGET_IDX, lbw):
                        lbw_weights[index] = value

                if not(key+ ".lora_down.weight" in lora_sd):
                    continue

                lora_module_name = key

                down_weight = lora_sd[lora_module_name+ ".lora_down.weight"]
                network_dim = down_weight.size()[0]

                up_weight = lora_sd[lora_module_name + ".lora_up.weight"]
                alpha = lora_sd.get(lora_module_name + ".alpha", network_dim)

                in_dim = down_weight.size()[1]
                out_dim = up_weight.size()[0]
                conv2d = len(down_weight.size()) == 4
                kernel_size = None if not conv2d else down_weight.size()[2:4]
                # logger.info(lora_module_name, network_dim, alpha, in_dim, out_dim, kernel_size)

                # make original weight if not exist
                if not(lora_module_name in merged_sd):
                    weight = torch.zeros((out_dim, in_dim, *kernel_size) if conv2d else (out_dim, in_dim), dtype=merge_dtype)
                else:
                    weight = merged_sd[lora_module_name]
                if device:
                    weight = weight.to(device)

                # merge to weight
                if device:
                    up_weight = up_weight.to(device)
                    down_weight = down_weight.to(device)

                # W <- W + U * D
                scale = alpha / network_dim

                if lbw:
                    index = get_lbw_block_index(key+".lora_down.weight", is_sdxl)
                    is_lbw_target = index in LBW_TARGET_IDX
                    if is_lbw_target:
                        scale *= lbw_weights[index]  # keyがlbwの対象であれば、lbwの重みを掛ける

                if device:  # and isinstance(scale, torch.Tensor):
                    scale = scale.to(device)

                if not conv2d:  # linear
                    weight = weight + ratio * (up_weight @ down_weight) * scale
                elif kernel_size == (1, 1):
                    weight = (
                        weight
                        + ratio
                        * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                        * scale
                    )
                else:
                    conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                    weight = weight + ratio * conved * scale

                merged_sd[lora_module_name] = weight.to("cpu")
        save_file(merged_sd,"temp/"+str(m)+".safetensors")
        del merged_sd

    if win==None:
        print("")
    del lora_sds

    merged_lora_sd = {}
    with torch.no_grad():
        key_count=0
        if win==None:
            print("svd")
        for m in range(len(keys)):
            merged_sd=load_file("temp/"+str(m)+".safetensors")
            for lora_module_name in keys[m]:
                key_count=key_count+1
                mat=merged_sd[lora_module_name]
                if win!=None:
                    win["info"].update("svd : "+str(key_count)+"/"+str(key_sum))
                else:
                    print("\r"+str(key_count)+"/"+str(key_sum),end="")
                if not(lora_module_name in ok):
                    if device:
                        mat = mat.to(device)

                    conv2d = len(mat.size()) == 4
                    kernel_size = None if not conv2d else mat.size()[2:4]
                    conv2d_3x3 = conv2d and kernel_size != (1, 1)
                    out_dim, in_dim = mat.size()[0:2]

                    if conv2d:
                        if conv2d_3x3:
                            mat = mat.flatten(start_dim=1)
                        else:
                            mat = mat.squeeze()

                    module_new_rank = new_conv_rank if conv2d_3x3 else new_rank
                    module_new_rank = min(module_new_rank, in_dim, out_dim)  # LoRA rank cannot exceed the original dim

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

                    up_weight = U
                    down_weight = Vh

                    merged_lora_sd[lora_module_name + ".lora_up.weight"] = up_weight.to("cpu").contiguous()
                    merged_lora_sd[lora_module_name + ".lora_down.weight"] = down_weight.to("cpu").contiguous()
                    merged_lora_sd[lora_module_name + ".alpha"] = torch.tensor(module_new_rank, device="cpu")
            del merged_sd

    shutil.rmtree("temp")
    if win==None:
        print("")
    return merged_lora_sd

def merge_lora_models(models, ratios, lbws, new_rank, new_conv_rank, device, merge_dtype,win):
    merged_sd = {}

    if lbws:
        lbws, is_sdxl, LBW_TARGET_IDX = format_lbws(lbws)
    else:
        is_sdxl = False
        LBW_TARGET_IDX = []

    for model, ratio, lbw in itertools.zip_longest(models, ratios, lbws):
        lora_sd = load_file(model)

        if lbw:
            lbw_weights = [1] * 26
            for index, value in zip(LBW_TARGET_IDX, lbw):
                lbw_weights[index] = value
 
        # merge
        key_sum=len(list(lora_sd.keys()))
        key_count=0
        if win==None:
            print(os.path.basename(model))
        for key in list(lora_sd.keys()):
            key_count=key_count+1
            if win!=None:
                win["info"].update(os.path.basename(model)+" : "+str(key_count)+"/"+str(key_sum))
            else:
                print("\r"+str(key_count)+"/"+str(key_sum),end="")
            if "lora_down" not in key:
                continue

            lora_module_name = key[: key.rfind(".lora_down")]

            down_weight = lora_sd[key]
            network_dim = down_weight.size()[0]

            up_weight = lora_sd[lora_module_name + ".lora_up.weight"]
            alpha = lora_sd.get(lora_module_name + ".alpha", network_dim)

            in_dim = down_weight.size()[1]
            out_dim = up_weight.size()[0]
            conv2d = len(down_weight.size()) == 4
            kernel_size = None if not conv2d else down_weight.size()[2:4]
            # logger.info(lora_module_name, network_dim, alpha, in_dim, out_dim, kernel_size)

            # make original weight if not exist
            if lora_module_name not in merged_sd:
                weight = torch.zeros((out_dim, in_dim, *kernel_size) if conv2d else (out_dim, in_dim), dtype=merge_dtype)
            else:
                weight = merged_sd[lora_module_name]
            if device:
                weight = weight.to(device)

            # merge to weight
            if device:
                up_weight = up_weight.to(device)
                down_weight = down_weight.to(device)

            # W <- W + U * D
            scale = alpha / network_dim

            if lbw:
                index = get_lbw_block_index(key, is_sdxl)
                is_lbw_target = index in LBW_TARGET_IDX
                if is_lbw_target:
                    scale *= lbw_weights[index]  # keyがlbwの対象であれば、lbwの重みを掛ける

            if device:  # and isinstance(scale, torch.Tensor):
                scale = scale.to(device)

            if not conv2d:  # linear
                weight = weight + ratio * (up_weight @ down_weight) * scale
            elif kernel_size == (1, 1):
                weight = (
                    weight
                    + ratio
                    * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                    * scale
                )
            else:
                conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                weight = weight + ratio * conved * scale

            merged_sd[lora_module_name] = weight.to("cpu")
        if win==None:
            print("")

    del lora_sd
    merged_lora_sd = {}
    with torch.no_grad():
        key_sum=len(list(merged_sd.keys()))
        key_count=0
        if win==None:
            print("svd")
        for lora_module_name, mat in list(merged_sd.items()):
            key_count=key_count+1
            if win!=None:
                win["info"].update("svd : "+str(key_count)+"/"+str(key_sum))
            else:
                print("\r"+str(key_count)+"/"+str(key_sum),end="")
            if not(lora_module_name in ok):
                if device:
                    mat = mat.to(device)

                conv2d = len(mat.size()) == 4
                kernel_size = None if not conv2d else mat.size()[2:4]
                conv2d_3x3 = conv2d and kernel_size != (1, 1)
                out_dim, in_dim = mat.size()[0:2]

                if conv2d:
                    if conv2d_3x3:
                        mat = mat.flatten(start_dim=1)
                    else:
                        mat = mat.squeeze()

                module_new_rank = new_conv_rank if conv2d_3x3 else new_rank
                module_new_rank = min(module_new_rank, in_dim, out_dim)  # LoRA rank cannot exceed the original dim

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

                up_weight = U
                down_weight = Vh

                merged_lora_sd[lora_module_name + ".lora_up.weight"] = up_weight.to("cpu").contiguous()
                merged_lora_sd[lora_module_name + ".lora_down.weight"] = down_weight.to("cpu").contiguous()
                merged_lora_sd[lora_module_name + ".alpha"] = torch.tensor(module_new_rank, device="cpu")
    if win==None:
        print("")
    return merged_lora_sd

def merge(
    loras=[],
    weights=[],
    lbws = [],
    precision="float",
    save_precision="fp16",
    new_rank=16,
    new_conv_rank=None,
    device=None,
    save_to=None,
    win=None,
    mem_limit=None,
    meta_dict=None
):
    assert len(loras) == len(
        weights
    ), f"number of models must be equal to number of ratios / モデルの数と重みの数は合わせてください"
    if lbws:
        assert len(loras) == len(
            lbws
        ), f"number of models must be equal to number of ratios / モデルの数と層別適用率の数は合わせてください"
    else:
        lbws = []  # zip_longestで扱えるようにlbws未使用時には空のリストにしておく

    if win!=None:
        if not(isinstance(win,sg.window.Window)):
            return

    def str_to_dtype(p):
        if p == "float":
            return torch.float
        if p == "fp16":
            return torch.float16
        if p == "bf16":
            return torch.bfloat16
        return None

    merge_dtype = str_to_dtype(precision)
    save_dtype = str_to_dtype(save_precision)
    if save_dtype is None:
        save_dtype = merge_dtype

    new_conv_rank = new_conv_rank if new_conv_rank is not None else new_rank
    if mem_limit==None:
        state_dict = merge_lora_models(
            loras, weights, lbws, new_rank, new_conv_rank, device, merge_dtype,win
        )
    else:
        state_dict = merge_lora_models_lowmem(
            loras, weights, lbws, new_rank, new_conv_rank, device, merge_dtype,win,mem_limit
        )

    # cast to save_dtype before calculating hashes
    for key in list(state_dict.keys()):
        value = state_dict[key]
        if type(value) == torch.Tensor and value.dtype.is_floating_point and value.dtype != save_dtype:
            state_dict[key] = value.to(save_dtype)

    if isinstance(meta_dict, dict):
        save_file(state_dict,save_to,metadata=meta_dict)
    else:
        save_file(state_dict,save_to)
    
    if win==None:
        print("fin")

def get_function(vs,win):
    win['RUN'].Update(disabled=True)
    loras=[]
    ws=[]
    ids=[]
    for i in range(4):
        if vs["ckpt"+str(i+1)]!="":
            if vs["ckpt"+str(i+1)].endswith(".safetensors"):
                if not(os.path.exists(vs["ckpt"+str(i+1)])):
                    win['RUN'].Update(disabled=False)
                    win["info"].update("error")
                    notification.notify(title="error",message="lora"+str(i+1)+" file does not exist.")
                    return
                loras.append(vs["ckpt"+str(i+1)])
                try:
                    if vs["w"+str(i+1)]=="":
                        ws.append(1.0)
                        win["w"+str(i+1)].update("1.0")
                    else:
                        ws.append(float(vs["w"+str(i+1)]))
                        win["w"+str(i+1)].update(str(float(vs["w"+str(i+1)])))
                except:
                    ws.append(1.0)
                    win["w"+str(i+1)].update("1.0")
                if vs["id"+str(i+1)]!="":
                    try:
                        ids.append(int(vs["id"+str(i+1)]))
                        win["id"+str(i+1)].update(str(int(vs["id"+str(i+1)])))
                    except:
                        win["id"+str(i+1)].update("")
            else:
                win['RUN'].Update(disabled=False)
                win["info"].update("error")
                notification.notify(title="error",message="You need to select the safetensors file for lora"+str(i+1)+" file.")
                return
    
    if vs["out"].endswith(".safetensors"):
        out_path=vs["out"]
    else:
        win['RUN'].Update(disabled=False)
        win["info"].update("error")
        notification.notify(title="error",message="You need to select the safetensors file for output file.")
        return
    
    try:
        if vs["d"]=="":
            dim=16
            win["d"].update("16")
        else:
            dim=int(vs["d"])
            win["d"].update(str(int(vs["d"])))
    except:
        dim=16
        win["d"].update("16")
        
    try:
        meta_dict={}
        meta_dict["id"]=str(ids).replace("[","").replace("]","").replace(" ","")
        meta_dict["weight"]=str(ws).replace("[","").replace("]","").replace(" ","")
        merge(loras=loras,weights=ws,save_to=out_path,new_rank=dim,win=win,meta_dict=meta_dict)
        
        if vs["dof"]:
            for i in range(4):
                if vs["ckpt"+str(i+1)]!="":
                    os.remove(vs["ckpt"+str(i+1)])
        win['RUN'].Update(disabled=False)
        win["out"].update(out_path)
        win["info"].update("fin")
        notification.notify(title="fin",message=out_path)
    except:
        win['RUN'].Update(disabled=False)
        win["info"].update("error")
        notification.notify(title="error",message="I failed in the output.")

if __name__=="__main__":
    import threading
    import tkinter as tk
    import pyperclip
    from plyer import notification
    import FreeSimpleGUI as sg

    sg.theme('GrayGrayGray')
    grp_rclick_menu={}
    keys=["ckpt1","ckpt2","ckpt3","ckpt4","w1","w2","w3","w4","id1","id2","id3","id4","out","d"]
    for key in keys:
        grp_rclick_menu[key]=[
            "",
            [
                "-copy-::"+key,"-cut-::"+key,"-paste-::"+key
            ]
        ]        
            
    layout=[
        [sg.Text("lora1"), sg.Input(key="ckpt1",right_click_menu=grp_rclick_menu["ckpt1"]),sg.FileBrowse( file_types=(('lora file', '.safetensors'),)),sg.Button('clear', key='clear1')],
        [sg.Text("weight"),sg.Input("0.7",key="w1",right_click_menu=grp_rclick_menu["w1"], size=(10, 1)),sg.Text("id"),sg.Input("",key="id1",right_click_menu=grp_rclick_menu["id1"], size=(20, 1))],
        [sg.Text("lora2"), sg.Input(key="ckpt2",right_click_menu=grp_rclick_menu["ckpt2"]),sg.FileBrowse( file_types=(('lora file', '.safetensors'),)),sg.Button('clear', key='clear2')],
        [sg.Text("weight"),sg.Input("0.7",key="w2",right_click_menu=grp_rclick_menu["w2"], size=(10, 1)),sg.Text("id"),sg.Input("",key="id2",right_click_menu=grp_rclick_menu["id2"], size=(20, 1))],
        [sg.Text("lora3"), sg.Input(key="ckpt3",right_click_menu=grp_rclick_menu["ckpt3"]),sg.FileBrowse( file_types=(('lora file', '.safetensors'),)),sg.Button('clear', key='clear3')],
        [sg.Text("weight"),sg.Input("0.7",key="w3",right_click_menu=grp_rclick_menu["w3"], size=(10, 1)),sg.Text("id"),sg.Input("",key="id3",right_click_menu=grp_rclick_menu["id3"], size=(20, 1))],
        [sg.Text("lora4"), sg.Input(key="ckpt4",right_click_menu=grp_rclick_menu["ckpt4"]),sg.FileBrowse( file_types=(('lora file', '.safetensors'),)),sg.Button('clear', key='clear4')],
        [sg.Text("weight"),sg.Input("0.7",key="w4",right_click_menu=grp_rclick_menu["w4"], size=(10, 1)),sg.Text("id"),sg.Input("",key="id4",right_click_menu=grp_rclick_menu["id4"], size=(20, 1))],
        [sg.Text("dim"), sg.Input("16",key="d",right_click_menu=grp_rclick_menu["d"])],
        [sg.Text("output path"), sg.Input(key="out",right_click_menu=grp_rclick_menu["out"]),sg.FileSaveAs(file_types=(('lora file', '.safetensors'),)),sg.Button('clear', key='clear_out')],
        [sg.Checkbox('del original files', key='dof')],
        [sg.Text("infomation",key="info")],
        [sg.Button('RUN', key='RUN'),sg.Button('EXIT', key='EXIT')]
    ]

    icon_path=b"iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAACXBIWXMAAAsSAAALEgHS3X78AAAgAElEQVR4Xk17B2Cc5ZXtmV406r3LtuTeLVfciCkhwNJSIIGQLASSl+TlvSxpm+TlJSFLCoGXvE1lSSHLmpLYkAQeyxJjbIyNjSu23NS7RtJII82Mps87937/mIwsjWfm/79yv3vPPbeM7ZavPJZDFoAdyGWzsNnsyOXA5xxysPGHj/wffmCz8QU/l/f0SZ5zvE8G0Gtt1mv+n9fn+Fpvl7/yLz+W9X8738xa48lHDrsZR+bRuWRpfLLLuuRZ1sZr7PzMzg9ynMPhdHDtZpm8zFqLWat8rmNZa5X7PlAfR3FpKe93wCm70K1ydLNxGcCsyFqK2Zi89Xe7tnNC2ZxIz25z6GaNMDiajmM2nJeSLFp2KhvmZPpRVi508B0+c3u8T9YgG5e5RKguMwD/ybWyOfPHkhjHk2tlzBzHlBfWMeSPzghOxlIh8EfHMnvN8n6ntS0j6b+Tvi5ORZE/XaMPdpcTiakJ/P5OF8oWLtMF/+dzf8NTl4qQLCyTUSyF4bVZG+KceFO2A/fsKEf9wjouxAGHy4W3XnkHaze2weZy66JtLhsymSx+98cO7IvMB7w+PVU5Zd2wUYb8k/Ue5+L4cp9cp+OI6EQLVK1Emy2tte7M8F3VGDl0B3d0y5cf4zX5iyyBW4enE3KCLCewcQKby44ftY+gZeVSpJIzHF71TaU53DWMxtZG3Pn7OWQo9bLh43jk4wvg9fvhKS2Eg+9xqYjH05gYDKKksgi+gM8s2dqcLkoOIZOGo7gZf/rzRewJBuCUvVha5bA0S0xFbrM7XUaDeJS1mEY8kcWUrxIOmjP/WWZghCfawHdx67wMCgoKjAkY29C/lu0ae1GVMxKAnWqa5c2/3D6CkqaFyKTmuFC1HrN7Xli3sJ4TZvGpusu4att8uLybKGo3nIFyZCPjyGUy6DrdhYqaElQ1letGDYZYuzdTiV3xZDh2NIibt3lwR/lKfOexp9EdWGmpL21ehMnNOWh58yKn8ND97UhFEnC5AoDTC2dFO/ovvIsfvxnnvDlkZK5sRjdvHg6dX7Ek4PGi0OeDz+fkrxuBgOc9tdHTpXpxUT/eNIzCxoVcHJdNycmR6AnIYniN0+ZG57EObNuxCF63D1MjIZ70FLKzQf18OjQNr9dJ9bQhPBnhwtJ/t3k5eS7Ksm+1z1xKTxf2OL50z3L8/lNtyLgcCn45jpGlGn95SQ8eunctcqkMXD5e6+BvLoPMxDHU1Xvx009vQpZ7unLIlqYrHImGiOm4aXsOIpqftljgdcPjFuAxYGa0gmvoOQqPxw5v9WI9pXBwEkMXejHZM4C3/3YcB44FEerrg6e4gKfg5QJ9qKhrQFVtAWdK6qJSoVkUVZYgUOhDSTnVz1dCWwSGOIbxLAKCFuzKyVB7HDzNbDwEb1kz4qELePr+JuQ8TrX1r6/oRPOKNh5DxtIc1UejkaJb8SlkosP42X3tyHGPim9XPFheuymAicgcQrNxjIcjCE5FMM6FyoVXzIJG8uP7muD0uBEfexfTY5O032I0Lp6PqvlN+Nijg1hSmkRJQz0a25o5dRqzMzP42zN70H22G+l0DslkAt4SH9596xQunBuHo6ACtnRcTB0NC5p4+jnMTM9Y6ql6pcJXZfVUwpaYEGeB6OB5PPVhL4Ze+A3Kq0vVhASHREON55LbjABkr+nEFNI5Jx67i+YjUCPmTNsRbZPxsxS4Xa+37NBMeQWR9FVtz0FugioWKKLtJ1G86FrOYccoT+7aT76IvV9pQCU3oRpDozz68gECnwMb378F9fOrMTsxjsjkNJLpNN/biOXr6rB/90sC01dMQBZSVFqMyExUkdluE2AzWmjzllCjeOr0HBm+2XLNv+PJ71wNf6AE3SfOY/jSRQxe7EZ0OvZ3AjTA6qSE7Zk43P4AbBMjRggqLIN5DvUCXyIRsvYtaiIuxYjAyPf/vm8MKS4+HIpjxc71GDh9Vv1+3/lBtCysReXyFXBm4zpgNDQJb2EBYrNzKOCJ58QVqf8WEOK4Ch3yB+g924l5K5YQXIVHyBmIR+a1BTXIxYOITE/RSxTBVdyEbCzMj6K02wz+4SvDeOYhF2bDcbholpWNLVxzEsHuEaSyaTQumc/xOJayp0LYS5chHe+B09+Cz/3yMAWSw22L3XARQxw8MMtpy9xig2IbeVXi/9J0aaksqpobUFrqQf/J0+odzp/owmQsidJFi+HyF3FhaQJRCt7iQkXwAl6rtE0eivaWENTGdbdoWd6K4U4uTF7yhAWfcwKCcxPIFS1DUUkZ3A4vTyyGRCqO6SA1aWwco0ODGOkahIc2UVJJYQkRs3tQ3daA4opyxOkN8r7ZXtxIAfNljF4oHYMzPKasRvaoqxN8y+9XXgsyGts3F2zCaR2rt6NLtUFeDHaNIcnVXn3rTnhqNiA1EzQb5U7UhxMDlIlYA4vKmWl0p8onRNPsTicaFrZgsH/YyMnQPOMa5waQK99IZeGrWBS+wlKUzduAwuoaXL8shi/+5BT6u0bw6rOvITYdUbsXDRKQnQpOwVbSSGE6uNkAcpFO2GgKucnT+PY9a4z7073IPYIgAsDWclU2qvlGLh/dVa8mUeA3rlFUpr61FlddtwGBuiX0pjxbql9eaMHB0SuCEldpEzcmVNcCILlfxjbji4raUd/SiDmCsMQeQkx0LOH4c93IlS6h/QuiTyITp1Dcbnz/6+/Dmf4EgTaC2oZynD14BrGQmIg5tPq2JoydeAtwV1IIKdiiY1e0OlBRjKTEPHTLqucyp4rfYMIVQSigUTrptB2JeEZtWW7o7R5GKmVDoEwYnBuZmU44C8sNiPK3trVFB7LRHwvlVbjhs11ec/OiDLkENUQkr27JCMNH06H/VLNVLfHUIRfg5sk25xLilu3K7GzOQq7Djt9+aQVue+QIQuEkyuqLaBIDSFJTVAgkXJ6CAOweP6ieGiMY3k+/HzoJjA9bhI+XS2ySd5uyOiEZluUiw9fCnLKZhGU3VKjCADzCNYjSiXGCYVE97AUtyHoCmJuhW0uQvIhzzw8iApDTtpgmcRfTVFnh+DKvRheq8w6kI3GSlnlAcRsXXwOHSMs7H966VQRX3iNiSE1hdjqEa27dgdd/9D786j8OEf1n4fbZMTkwCHJurjWH0rY2xPrepfCpAWqCcjexIuvBx1bL+oxQ5P188KRqIqqRf/iSIbVcP/m6rD8SSWHeIrq8xkp1YariRP9ULEQ0LiLSzvEw8zGphQcaiRkVE/WWgUqrK02AY9m8Umqqtr28GdkZkiJPvYWfJDiZMeSmj6OgoRFpnvC5N08iUFHFQ0nRvzvwuY+t1YMSFpikpoIml3XTbbrnkW2OIz3D+yUgUPOQiDGLHbtWW77fCEEO/cqJmQyAedmMUV20nhZdx6WLg+i82Ac32WI6noTLnkBm6iJcboKNzY/iempDYSvJh3GjCjC8X+1atcCKy13iDqnm1BR5P0PzcBatZ8xQD1ctXdbE61wYtSF0lMDVQTVOIB7sx/jIBJZsYPSZJL+nABe3r8DS9kXY99JRJKP099z87FAfXEULkU2MorypmfLg4elazLEKvqRJzdU1qJXzYBbVV8NLQiCbLy70o31hM8r8Psz3xhGhr5WL5LSufv96NDRUYS46B7dbfLYdMQJRZqYHI0deQ5okKetgOFxUpb6XBmthg2ySIFqwQFUvS2aXyziR8TYB1VvhKCXa82RSs5NITbyLBE0JSNBfM1ZQE7HB4/MQfIkvdnlPDskOf6GHQVEWlTVlGBoYJ4RkEAkOM/bohi3cRbbJwCjGzerejLqLhmdSMS7NBHJpapI9Th/bvmw+ub4b21cvwrm+AexYtwir5hXoBXLTxOQMLpzuZHDnIjkhvycyO5x2FAQIbgTDvS+8hdTcLLLTp2gOxfy8ALZyupyytUB5O5xl63ThGXsxQXMZuk6dhtNXDHuSPl8XRjUO1MJZyijS4Udm/ITij0KvqqAN44ODGO0KKXAF+3rMpvhRRXUJ/EXipags1Mx0uFcJkz02iXhyDklSfcNxjBYI9UnEDGt08JDsvcEwDnJz88Q2edjXrl+Fjq5RVFd7EaD7k8kbWqoxMxWDr8CHYA/tijcPXBpEJpkj8ZhBTX0VT26Gp0WE529mkupLDJEtKMgpGLoxMXBBVe+1vxyjFhDYUhHYJt8EJt9Abvww0lPkG1R/h532bJEyk2HKoYzrq2ksQ3Q2giq6ztH+IVXt4spyrrVCpTGXSNJtG6cuUWeCuJUkQTMe1wC8EKfxnvNqDjKuJgAFCCais5igP/3zoWO4NDJKXsMQ1ysuLMdYO4JFS+owNTGpQYkMFKC5ROm/j+5/G3UNRRgfmtIBZeM6UfgyN0dByDHleAqZMKLj08DEIZSX0EzCnRg7/RYBjW6W4WyWUaMESHYX8YZBkyQ5cmSY+WWL4Jx0pYESaiA/q22sId6kESjwIp1kxMlHQYHhK3mjz1CDfQHGAZYQVTTEYkd29grgX6HCIboaJmtw2/YNuHrtUtqJoLYdvX3j2L/vDHl2CmVVFaS+bpWAr8SDEhIL11wZSou8GGP8L2Ev4mFNfgj6SrydG+cmR9+hC0oSz3gaVE8nQ+Zg7yCqW2p4jzITKo5oD+/jT/Py5crsEvGEMRFxxLzmxIGT6OwYVOIUI1YM9UzCw0MaDzKSVEQXFRdloAfh9RmevoMgbdygLNtQ/dISMV3VS9EAg4giuRPne7DnjZPYd7xDiYqNoWRxeQk2bGTAQ5Sd5Ak6CUiJaJyY4aUL4gk5/bQlGylomBjF0DcaUxsUIciGskRc5RLRSYz2UgM43aI1K3HytR4Ng4c7uykYwyb18LjZTGKOAkxQC8ULUTBcYJreoLd7UPgSsceH+OwU+ntHdO5weAZJEa74d0mUKqOi0iUFwygMS2sNJcihhuZkRCGBmF5rQEIPQ9yVCoQL4U9dNcNgC6hSMS6KgggOjKG/k7+MEeYvW4bEHDdL2xgbHMJYP6mnRTLEUyj4UUVFKNFpniifa1vmobqxlcoSQff5AXoVIdXKFtRFzc7EcOqN49QUl6Gt1KYYbX8ePVTzgnoEh4Y5J3MQVVwbQ/VeYpaYg+whLYLP+34KWA42K9olmzXwwNyGySzJ1skELXgUb6duhzfQ/kXxBCW7L42TfZFSysX86+KiggOTOHHoNM6+zRzfwkXEjgyqGJZK2ttLF6rgp6chi6FWCDrPRenOGL1xQ8V1tVh5w7V4940wtt+yA+cOn2PiiHdpfSGL8toK9FweUveooQFPNMNNuKl9CrLcfGR6DpW1pUykhBGilxJzk0dqjkLWQ5XXwj3t1EqDEcYUhNH6jWD468yno/TjvDDkJMQu+d7EZBjFfiYjCFRO5uSEuc3Fc2hsrkI0UoLX9+wmvZ7Amg0+coQEeb2TE8bhJThJ5iVNFU7F6G85XoGnAEOX+jDcS/LCSC0YCqLpsg2xiQRCvcNwM5dQVF2sSZGSshI9Sc33814HfbeLWiaAmeRp910ewOJVTQgx/ppTCk4tEq0V++f65RDcXuHtWcSpuR5uWgQg1pEia5XDkfhEM84WUTLoLapBqJTwNEWXUl1XxkkmaYNpsj4RTIbgNIe2FQswO+rH/I3rMXbxIrMuAwS2IPOSDoSTDEjm1ShlddhdqupDAzFUltejrq2EJ7wdbgYVWdp47zvvYMOt23Hkj6dQ05ZAIhEn3a5AaysBUsxRgiaRAbc0StOrrGKgwzUuXtNKxseocCqEjC+gjFSprea5JbK0wV9croeYEnRX/i/mzZGoLRnxfnk3aJRfBW3wgFafTZNvU9rlzOnHmfwIU81ErVNkfHPRKPzM8JbQ/4okqxcvpYrG+X4t7XOaMQLzgkTmvAsT7ZTY3CW5eE7iLSphUrSAbtaPBdt3YPDcDArr7Lh0hpEa77VLYlXMQfRfFq3lOuYi+oIYH5tmLqCfQMxECXlIOJIkjWbeUPIRvM5fyHu5izQ/c3M+MYe0atF7RpDmwcpe5G2CfZ4qKgjohWmJdfgfJ6tAkjR0Od0IM80lH6fomsoqShCe4qLLSEA4WI4nPXAhhqbWeahvkjicas3rdTDeH2ZKvGb+MtgY9PR2T1o5AjVuPZHGlRtR3dCE5VdtwuGXLqL/UieP4L34IU00j4ZjmBqbIMZ4UV5TCqfbi8lgCC+EFjAllzL0nibj8gpO5BAalrCXtJqY4WBWWAM9y8QjM+QBVtXIhMNXAMLYg9dFt0TVtZGRSUIk4PGwXlDIpESGAwexYGkbpifC6Dl/Fr/8xW9x64cfQGn9IsRnuogHHhzff1k1aLSPQMYjIASgtKKaJMSOutbN9AZTeGP3HxEm4UpJPEHgKqlYCg/xZcNNO9BzdJZy5YJTXBw3FQ5OEyRTrDncrq/bNt6rY48GI0jVLkZjXUBKHUZ7BXjFYIQBMg0mD8EOk7Qx4D49a/EEjuVUv3klJDTSaK0kgElcT+SVbG4yncTCeXUYvNSL6qZaVDWU4msP/YYhixvDtK9Nt9yK6z79TTQWObDnV4+iucWJzgt9qF1Qjkm6xraljZgZ7uappuB2sYaYGkdhTSXu+eiDmCNmjDLoKqSQM1T/EBH+Xz5zF0pGA4jUx1DAGsJcmNliamoJEzFpAt7xZ/9EAJ3C3tQq5LwZHpghUKoGUiOj1kXF7qW4KibEmoeAoeEaNnhZW0hz8w7WHiwmaBBfsUASI5yM5q9MUFxgjIuKx+bouxupYgyHmecXnKhJRfHd734HW1a3I7DmC9jw8UfxoXu+SqayEEW1C1gvTGAmFGNWmZS0qI7BDtWTap/OzuHRn+/GTZ/9Cmba7sGzB4/Dv/Z+jGSa8K1vfxt/2fsy9u0/gIsnydnFMTLtdvp4FwZJmpqW7MS6m2/EHkbKMW+NJlVrKRg5X+EGspE43V7TsnalxPK+izXNfKFHTJsQrbuVZK6VEpNskAFA8cTnRhM8Jao/pVZJoEtaAPTXPft5AsDpN09h9fJqfPfxR7B0zTpsv2YXAc6PUVaHoys/iid+8gTa1u/EhtvvQmXdFoa5lQjPTKO2ntVgJk19RZX4l1/+Cs+8dBAJltTe94EHMU1itO1zX8P6zZsxj0HX1dduQBOTpmPdQxjo7SceuVHfOA+Xj77FUJeRKj2KaIXYclmRXwEtQ0HLQU4ODJOxCrU2NU4Hcw4m6jS5wEBFsxGI4RgGad9rJCD4OT2K+lJxSdL9+SnBxpZa3HDHVejtHUPjgjp86MO78PjP/w3J7uMoLPaigFUiP93XsjtuxMvh5SbNTfwoZWKiftM61C1o5SZMsOLxeVHFtNXqFQux6c6d2HnfvVizcyEuHTuI1mVL8a1//xVdbVhjBQl0Igkn3PQcDgJcKeORNJcca16pGxJW76VmifYXV5Yp7U0RLzBHbyRrkHym1h7kleyV17OoYvSdrDCPgTUlxVi7uIn2nsDBs/TpI2ECF1Nd1AQHQUSQ9NyJHgSHp1lD5DkSuCpKMqgso5sc68c/rnVjZkktM0XADWXvMyRGQlCd2BArsyCJEbjw6QF8739/EV87FcSqpU3EBhse/OYnYes/pWn1rC2A4cu9ePWvb2Pr9hXoG2KFhyadCZRi9zMvMXIs0bGXtTD3wNyEaEBFfSVCI1OoX7mGcwgAMoagO5QISVmw4ICYIE0gy4hQstTiPVUMW1a14sDxi9i2vIKFiDRRnZlaXTCZHyUqJlJWSsnx+cLZXuy4YROOHO7UcNYzPoi7Wul/J48zJcX7lhfT+3AyulGDLhJZEqjEHcuJUDMccwyfpyfxtakBxHvG4S+vQTkLMPaMAxffOYwlaypw+uhFrF2/BI3LWnCptwMVqzdguG8YL8xy82Z1GDnfAVsdX2mll9UpJk1LW1pZW4jqfEmSJWGlcq3gm4QJKVkbBSH/J7uVuhsTG0T6CgYXFzp7YSfS1jYU06Y4AScM0QfLYC4egdRmS4p9OH+sE1t2LGeE2IMSJjLFRiWEzlJ4DjejRNqdCEdPR5KoXLJEZkoNhGhIepzAWlbG1FURf+kNsly0LetidBhEx4kMSr3NTMFLIqYAbpsHP/zZ8zjmZDHVyt3Khpa4+mnnDYwNogiMMyZIcYVzk0bwZJpJAqODgnGTN2QVFAXkxUOINRBBVixogJs+38coL0yfPI5SNh10cFDTGiNuJ8w8oKzcSbopA3i8XqzYTL8/O4ueIx2Ijo5i+Mw52h1tT4MaE4unKX27jxxcMUaIiECvUFECLN2rbW4O8Qn6eI6fkQoPI7uhcx1YvPp6bLnhg0xcSN1AKjw5PDNWjCN2BlNSUpZVcDMOUvarW4t0nUVlRUgwT9Cy4zoNoEypneuIM3MdNweQ1wKbpMzkIqqAfTwyy6SoAy++eQat9TVoayxHgFlVZjiNG+Gvi0lQURlpaVGTtnLt3kI3jvVeZMmqDg3Ll7IOz0wwKbJexI26AzQLprPzLjafgjANVFJAYYGS4zrJ/+2MH2wjZIyLlqjpSPh8uotZXi2y2BBxkT6L11YhmpTZyiYGSMwQyWlL4XWKhAmhs6bEZ/l9YX35oolBPkanGcEHQ53tfUNBjDHWTnHQ1PQYDp3uxrr5TDvxkUySBPEk69nWItLzMo6WZ9Oexo4SqmZbaxmpKDUjwMSIqDKTq1L3c0rKjBplwguLiwsRsX4148Skiqu8mD0AXqoofTM1S7K48jve2YV9cy4mZMhA2SiRcjMIEsErmBsBnuy4QL5vZ1EmhqG+EJbuYlsOyZnBHIbQ9PMJptFjMabSxfxEUHw/LUka7jdDTeReTLQlj6MDYaodsLp9garxHKs1ctrVDFH1ZtpzWhUS6OkdRSEDJR+FIawK1A47k6g2gWras9zvl9S0WbGOrzlDyc7whG0UlI3g5CTFtrEjjK0q7O1hpphY4uBnP9/9CnY2M+fHjJSzoAFO8SjqVWRvZgO3+PoJqgS+WBrLt6xhRwg5vlSqxQnL6YqwJbOthIBz5umyVaJ38hDsph1OfCUTBxlKjZtUHk51k8BDJqzkwowvyaJ1CckMNzE6OKwL8jE//5Vn/kRKSlvlgLI5ywjNsyVdmUVsVW1PNiL9Pkxw2NjDk2O9IVdXiUwJCQ1NaPTcu+ho3oibdizQHKSDfQAyd57NiTAXNfuxZm2ViRUmZmhK1EzRHgU5OQDmDZgHKCxyYYZJXUmQ6j5Jx6XQI/8XU7FrLMCfhiKiJzfvm2RVVvefY2GERQTaqU+am0zPGSYIeOISncKleWFBgRujDRtpPkRwTaQK+sqvicByGWm6EBMwtQQVhFIDo+qQIgs3mJUaA0F2ZnyCSZPL+rlkpTVbzHK5UV9jTlIzuMD2Gx8J1WDnODbf0M4wWFRfDt4kRuQhgZybmpWaZU6SoCtzO9lzINqT10pagLGrgFds247PrGJ9Xe4mAo/2BzXUzOcEZRF1zeT0XHQhbT5F1zkT5XL4+h6S8/RMgvea/hvzkCZG5gOZQMmyQCHVoyv2Zj42QrAI0/CFCyx3v4Fisr0MtfHMW12avBi6PMZaYD5vSVLGW+6v6KZtZ7CSqp9iFTuj8TCFLgdlmUCaHsDJ9cwR4ySjZcgQr7V0QZZg6gL86PwoJU3JLCDlFauI0aX4eboDUwIUkmAwUpUCaFNrAwGRANXYhGkCEMb7ObIddz97Bg8//jvihDIevV4yRE6pDYrtqx+2os+8jFQ5HHj8q99Cz+mTqGaRRRa6vpY5hzRz/nTDT77KjhArbBHxtlT7sWRtC3MEKQTHRthNQhxRHDPgKNuSc913uF8BVnBB6bGRgJUcNeugSdpQQCYoN0gKK0Uio2kxaSDiz39Nl6ndShFEw0nxrWI7VOeq5vloW9iAW6r7+dqc/Buzxdhw76NIzpoMkp4KQc1Bexe8MdYn6xCP4MbQ+Qs48Iensbl9GSqb6nQMcblt6SNoqPcgwmLMv+2mCVhtbtL/s7jnz+i90IsWFmvmr1hsZfZN5igf2odGx/FmYhEDL9YuqKFJki61e/46WKUyobHkA/h2XSWDH2rAh/xn+HG1LjpFphYlQYk3bmNam2UwzfMbAUtusKqxCn1n6IaYeDxwagLHL79CGy4mGaEvpia13fY4KmyTOP7qv+Z9AIc1pFwEnCYN/s9n9sLPxZWx5a2Qp6juVXw+k6pNOx/Ccs9L5BUUHE6i6xBT8NtbUWabRVOFF1tv2cX09pVonsOaXqS813HzPpevkInWOMrpSaI8kDLZG8d3MgBIS0Ql/xd6eGGIak6w29leTb8v5WsnVYuVHqqtk4GQ1tzEhtSfmzOU9PjZI8ex7QMbcfiiA1sXevD62bBVU6DGTPUhxLaW507Y0V58EQsWsMuUxjF4oQOHXzmAqtIi+Kk1DQsaFRw1PaY+DujyLEYpTyxX6NIqkCRig30MiY978d+uH8aOm64V41INNR48v3W6R2rdJK89eW6MpH+lNne7aK5aJBHjEBOQuMQqoNgrCqjy1ANpIZcssFwUI2A1tbDgyZTUkWdfRedQjO7Ez4O1OLROmWLPXyt++YNn8Id/asDfTvFOApc84pEgAaoPd3zoQVZv/Rj0r8MDP9iD/U/txoVD7zB+D6hGNS5sUlPSKpSYIBf30kARyuavQP+5k7j8zmlUbfwGz4FIno3g2K9bsf3Gqzi5ac83XW0is3xCLMMEKwM0vvPEOJsjJaXGa+W6FAmdzKFLlIqTdb+9kKGtnYu51XnIuDp+4CVwSLLzS3+4TKLgwnP7xlDLpsfwOAGPaGYIhcG5ux+8HTU8RYkdsjYWLdhInUtP4J++/GPcec/HFZCkgbKa3R9pLshDolPTUIN6NlcKbuSTsmN9o+hpvAnLt+7QIGVrxUE2VldiMTM7BejE0P67MTNJt2x1mui4asfEJGF5tH9pmGxctAA9AyGCJoUWnWDxhtGicBzGKfpdBO5VGjMNWHKU3jFKgydwzZZGfUMGHmLSY/kg9rYAAAmdSURBVGxoAgmX6cV95YwpaqalBc7Kr8upCXlKxKPY++xBXN69VtG1wDmDT33hGwxOSql6hjqn6cpWNRVpo0N5XYUWLLVVTttkHDgzQtKy9QH1FjKGO9KPi9Nt6PR/DB++Zh46DnyDfjwBP7PQeuaMDHMiCO0jqqRNO9m4eZntu21ceycevbyE16Tw8J0B9LPCJA8JziQMFzNj+tGAoPAZyRs6SRI0RubmteeGQc2Lx4dpN+T01AZJIk6ToJSTA8TGJ9GwqIUp8XIMsGVGujZufP8GbPr4C/j2Z2/CqHMV1rcvR393l+kEU1XJ4dULU/jMNczfa6+gYYsXxufg33g3FrJxaXKKmsM0tmt0H9K1NzHLFEL/wd/jfzxwHdIFrdxBF2zFTK0zBpEsRprtb9LtiSixZpDfYViykBWnXtz72Dgq2lvwEe9pps35BQ1yFfUsnC9BLPNRWPEII9AcPRNNgYQui7vK3+VFzOdT+mIrtU1l+O0Ph0h4fMSDcfj8PE1OVsvCZNfMFEKsBBeWVbLRsYlkaQyf/PrT2LplC1q2fQItyRhrCGEWMVuMoXDin3zvW2zgqkP3jBPeSBQzdRvYKboO1zGv6GUQpIlP/rx+pAuphhtQFnoOQ64NuPkD7E53E6BnLym1zY6/beBOIj2XXxuzstxgEVNho309WHffEXWtb316GDV18/G51+pwe/KEYeYcX/sI2C8w3nsJgYYVxp0Lq2pfzWZnidN5mZNFkBhrfA4WMSSX5i8opxY48Ys/D+kXJx558gTVO4C01AyYO6xZuRVL1u3CFz7SSJtjMDI3gpGhXuMyqQF7fvsLnH79L3jwgfsYZb0f3i0fx9MvHsa2tc3wMUmRxxMJdZc218A+vB/h0tuxa0kvA8sE6wZse6dQJRur/cbCH2g2M2yLk0RLYi6GL//z8/j6z84iuP+j6NuzC//8+0tY+6mLeO3JF9ERMhVqCZrSrHDJIzk7qvtVg3dGQ5QMJ1BQYJ1/MoQDJwhYTCoyWlc7nxjtwBP/JdcRD85Vsm+XnSI8gRzD4CwbktZdtQVlJVQnyf4UNKJl3nwdfGpyCm//dS+2EPgkwKooK8YUa/mf/ew9ePHFg5YLM1YSmQmhppbptLpdmD3zJAusUfSxDiEuTGN7TdBI7JPFMElQEbnD5MAoPEVF+OTdm3EpSKDdtRdtH34bx7sLUDuvBZVsr3tpdDXufqqEKs9CLUmeYNhi1imkIqVts50vfSOXS0r+THi7RFRsdb/vRcT4hYPSmnaawBhKSngyBIvLr9yI9gcu4kc3zbJddq22m6dYI9g/uArOsVNwtl5L3uDWbhIpUx89dprNFRvx+M3rNW645ksPs4mJba/P78bLJELvzLGpQlvI33scOt6NlujT9Ez8UlZtkQpJkpcWiyX9nUWAYfgwe5W+//NDeLnDS3fK9jkJzkikqNTG9PSAzSlLddhHQU32n8CpP95EPLfjwLtCnASI6baUrirHoQkIEvubsHbHPdrE0EoVb1i4wvhObs5NV/fZ/3OMqDqrE0lDgre0Fo5KCZPJG44cxfF3TiDIVNeWq9bhxGt7UURBbS5rQPx3v0Z873OQfEw7W1kf/vbPruzcLJWPGeYZikoxQSIm2SJxk9pxwguEn1w80aleZeN9b+BEZDka2piJsuKOMIs4U0HpVzY2L39FsT0s7+doQmUtq3DmPMvwHkOgNDIXQqLkjm/FmcObYv9tXfM8zeH7+PUWyf9pAoKZkn/8/G50HDuE2+++l+8zFcX7JOnB9Ce/R/Oupq+2bt2ILZvbcceNW1hB9uEvj38PLX5mlOgQxAxCsVm8OTOJG557hTXGVaLUumARphYvup7HI0+cR3GZiyYgobmht6Khwx0dWLdrFT7y4F72KzXRPEzJS/FLQnNmpIvLK9g0MaEuVjmC9CrlGTI1/BPffEcr3OJO9et2mkzTIMfODgwvrv7iq/yyQ4lBSMt9yOovv7AD+07n8Pn/fj82bdlsaDFXffjUNBbNr8GO2x7C1nVt2LZ+AbZvWKTxQjG/gFXIAEt6feTaUCKGEQr5aCyCW69fjWZ2en31m+9pgQBic1MAq3feiKJyiQ+KdG3iIRLRCB78/gElYseHmJmuZrHVCq5EgPrlLeUINmpQmQm2LKUS4YqpyrO3fJExO36oXESuES2QTK2LAFFQyNy8Sty0EQnN/P6dUdRsfQrf++H/xK233YDkeCdZ2RQunzqHjTc/QHBj0OFlnE7/nBe2i2q2lNGadGXGidaRRBRD3Phfo6bdRjjH8qX1eOC+D+JPe1/T0//tU3swEiL3mOnVSFIAUN4XXOk+14t3uqf57TNyBfYDa4+P+jfD6FRL+DdJ7OGGSOslA5T/jJZFzEmT70gMk2PJPNTPLDYfTlMt4XI4UHhOegL4VRf9oX1T2vH4DG7+/Ot4/i/Pw88FLWmrwkrnKIbYs7/sIz+9MoVpqTc9hzowBdfV3Y01pdU4TZVPkUe8TvVPc74RttRUV5aimN8yKy4OYD5zED967Al84t672CzZjJ4X99DueYraYwT09VzCn0+5cd0NdzLrzqqS5BSF11upLdmobM7NxorOd/9Gs1yAZhKjPAjKeibHBvlttVpNl4m3mldLVyrHIN+flYeYwO1fffmKf1RyQuIwePYIHv7Bw0ydixtKY+rww+iaXYzWG7+j1DLfnKSBnLV560jww02b8fZUEBsqGrDytjswzp4CJ1W1kUkPKbIoQ6Om9Q+N4aEv3o+K8gA37mCmeD7rBmxjYZ1AJJyl+i7f8UHs2LkLDTUBlFdWalVnfEy6RQnO8o0Q/lw+9To3L4VPFkOEqguwy97UWvVbwgqYdjLRdpqp4AbjQJMqElVP2hmaSlAj7pBa0XH0r+ztOaWakI1HmCQtpZT/Fxpk8bxJ8+9WT55MYv1XF3MnT95JVVtfXo1gOIRf/+YJawGyGkN/JPkip9nEhu286Qz1HEb18ut5vHuZ7JzQrPDocBT/cP96/L/XCYIfehSOCqa/uc4AvYWQs7df/TU7TLaR1JQxt1BN/OC3VXXXlinz2hKuw1gLI0MKVj0HEyPsEjOSkQZEYYFCOYX2Ht33O0wEe1R6EkO7+NUzu0OEJXanCEIBvKfyeUHIWE/+9KdopneI8mQkGT1GADOuKl8sNToiLTj6FV1LIPJe+/WfZigdwcjL0j9ImixEbMbDtYk1pDE6U44DjxThYz/hN1H5veQM51i1/ZNkjQ79HqKdezDu7704RL72V1LBqpIUbPnpCKPaajZeCBH6/x3w0sDRP3MBAAAAAElFTkSuQmCC"
    window = sg.Window('merge lora', layout,icon=icon_path,keep_on_top=True)

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break
        elif event=="EXIT":
            break
        elif event=="RUN":
            c1=not(values["ckpt1"]=="" and values["ckpt2"]=="" and values["ckpt3"]=="" and values["ckpt4"]=="")
            if c1:
                if values["out"]=="":
                    names=[]
                    for i in range(4):
                        if values["ckpt"+str(i+1)]!="":
                            names.append(values["ckpt"+str(i+1)])
                    out_path=""
                    for line in names:
                        if out_path=="":
                            out_path=os.path.dirname(line)+"/"+os.path.basename(line).split(".")[0]
                        else:
                            out_path=out_path+"_"+os.path.basename(line).split(".")[0]
                    out_path=out_path+".safetensors"
                    values["out"]=out_path
                else:
                    out_path=values["out"]
                ok = sg.popup_ok_cancel(out_path,title='output file',keep_on_top=True)
                if ok=="OK":
                    thread1 = threading.Thread(target=get_function,args=(values,window))
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
        elif "clear" in event:
            try:
                if event=="clear_out":
                    key="out"
                else:
                    key="ckpt"+event.replace("clear","")
                    key2="id"+event.replace("clear","")
                    window[key2].update("")
                window[key].update("")
            except:
                pass

    window.close()
    
