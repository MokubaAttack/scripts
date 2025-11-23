import itertools
import json
import os,shutil
import re
import torch
from safetensors.torch import load_file, save_file

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

def load_state_dict(file_name, dtype):
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")

    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor:
            sd[key] = sd[key].to(dtype)

    return sd

def save_to_file(file_name, state_dict):
    if os.path.splitext(file_name)[1] == ".safetensors":
        save_file(state_dict, file_name)
    else:
        torch.save(state_dict, file_name)

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

def merge_lora_models(models, ratios, lbws, new_rank, new_conv_rank, device, merge_dtype,win,mem_limit):
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
    for model, ratio, lbw in itertools.zip_longest(models, ratios, lbws):
        lora_sd = load_state_dict(model, merge_dtype)

        if lbw:
            lbw_weights = [1] * 26
            for index, value in zip(LBW_TARGET_IDX, lbw):
                lbw_weights[index] = value
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
                    
        # merge
        key_sum=0
        for m in range(len(keys)):
            key_sum=key_sum+len(keys[m])
        key_count=0
        if win==None:
            print(os.path.basename(model))
        for m in range(len(keys)):
            if not(os.path.exists("temp/"+str(m)+".safetensors")):
                merged_sd={}
            else:
                merged_sd=load_file("temp/"+str(m)+".safetensors")
            for key in keys[m]:
                key_count=key_count+1
                if win!=None:
                    win["info"].update(os.path.basename(model)+" : "+str(key_count)+"/"+str(key_sum))
                else:
                    print("\r"+str(key_count)+"/"+str(key_sum),end="")
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
            save_file(merged_sd,"temp/"+str(m)+"_temp.safetensors")
            del merged_sd
            if os.path.exists("temp/"+str(m)+".safetensors"):
                os.remove("temp/"+str(m)+".safetensors")
            os.rename("temp/"+str(m)+"_temp.safetensors","temp/"+str(m)+".safetensors")
        if win==None:
            print("\n")
    del lora_sd,n,all_keys

    merged_lora_sd = {}
    with torch.no_grad():
        key_sum=0
        for m in range(len(keys)):
            key_sum=key_sum+len(keys[m])
        key_count=0
        if win==None:
            print("merge")
        for m in range(len(keys)):
            merged_sd=load_file("temp/"+str(m)+".safetensors")
            for lora_module_name in keys[m]:
                key_count=key_count+1
                mat=merged_sd[lora_module_name]
                if win!=None:
                    win["info"].update("merge : "+str(key_count)+"/"+str(key_sum))
                else:
                    print("\r"+str(key_count)+"/"+str(key_sum),end="")
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
        print("\n")
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
    mem_limit=200
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
        assert str(type(win))=="<class 'FreeSimpleGUI.window.Window'>"

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
    state_dict = merge_lora_models(
        loras, weights, lbws, new_rank, new_conv_rank, device, merge_dtype,win,mem_limit
    )

    # cast to save_dtype before calculating hashes
    for key in list(state_dict.keys()):
        value = state_dict[key]
        if type(value) == torch.Tensor and value.dtype.is_floating_point and value.dtype != save_dtype:
            state_dict[key] = value.to(save_dtype)

    save_to_file(save_to, state_dict)
    if win==None:
        print("fin")
