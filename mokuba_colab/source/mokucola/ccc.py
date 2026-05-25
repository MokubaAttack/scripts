import gc
import torch

def flush():
	gc.collect()
	torch.xpu.empty_cache()
