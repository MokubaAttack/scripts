import gc
import torch

def flush():
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	if torch.backends.mps.is_available():
		torch.mps.empty_cache()
	if torch.xpu.is_available():
		torch.xpu.empty_cache()
