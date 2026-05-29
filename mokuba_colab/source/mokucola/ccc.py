import gc
import torch
import safetensors

def flush():
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	if torch.backends.mps.is_available():
		torch.mps.empty_cache()
	if torch.xpu.is_available():
		torch.xpu.empty_cache()

def getid(path,w=None):
	try:
		f=safetensors.safe_open(path, framework="pt", device="cpu")
		meta_dict=f.metadata()
	except:
		meta_dict={}

	if "id" in meta_dict:
		meta_id=meta_dict["id"]
		if "," in meta_id:
			meta_id=meta_id.split(",")
			for i in range(len(meta_id)):
				try:
					meta_id[i]=int(meta_id[i])
				except:
					meta_id[i]=""
		else:
			try:
				meta_id=[int(meta_id)]
			except:
				meta_id=[""]
	else:
		meta_id=[""]

	if w==None:
		if type(meta_id)==list:
			meta_id=str(meta_id[0])
		return meta_id

	if "weight" in meta_dict:
		meta_weight=meta_dict["weight"]
		if "," in meta_weight:
			meta_weight=meta_weight.split(",")
			for i in range(len(meta_weight)):
				try:
					meta_weight[i]=float(meta_weight[i])*w
				except:
					meta_weight[i]=w
		else:
			try:
				meta_weight=[float(meta_weight)*w]
			except:
				meta_weight=[w]
	else:
		meta_weight=[w]

	while True:
		if len(meta_weight)>=len(meta_id):
			break
		meta_weight.appned(w)

	meta_id2=[]
	meta_weight2=[]
	for i in range(len(meta_id)):
		if meta_id[i]!="":
			meta_id2.append(meta_id[i])
			meta_weight2.append(meta_weight[i])

	return meta_id2,meta_weight2
