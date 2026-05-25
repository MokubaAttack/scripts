import os

path=__file__.replace("\\","/")
path=path.replace("mokucola/__init__.py","diffusers_anima")

if not(os.path.exists(path)):
	import requests
	import subprocess

	cmd=["pip","install","git+https://github.com/hdae/diffusers-anima.git"]
	subprocess.run(cmd)

	path1=path+"/loaders/lora_pipeline.py"
	url1="https://raw.githubusercontent.com/MokubaAttack/scripts_anima/refs/heads/main/lora_pipeline.py"
	response = requests.get(url1)
	with open(path1, 'wb') as f:
		f.write(response.content)

	path2=path+"/pipelines/anima/loading.py"
	url2="https://raw.githubusercontent.com/MokubaAttack/scripts_anima/refs/heads/main/loading.py"
	response = requests.get(url2)
	with open(path2, 'wb') as f:
		f.write(response.content)

try:
	import transformers
	import re

	v=str(transformers.__version__)
	match=re.match(r"([0-9]+)\.([0-9]+)\.([0-9]+)",v)
	t1=str(match.group(1)).zfill(5)
	t2=str(match.group(2)).zfill(5)
	t3=str(match.group(3)).zfill(5)
	v=t1+"."+t2+"."+t3
except:
	v="00001.00000.00000"
if v<"00005.00005.00004":
	import subprocess
	
	cmd=["pip","install","transformers==5.5.4"]
	subprocess.run(cmd)

import sys
import importlib
for k in list(sys.modules.keys()):
	if k.startswith("transformers") or k.startswith("PIL"):
		try:
			importlib.reload(sys.modules[k])
		except:
			pass

try:
	from .workflow import (
		mokucola,
		mokuup,
		mokuani
	)
	from .dl import (
		dlc,
		dlk
	)

	path=__file__.replace("\\","/")
	f=open(path,"w")
	f.write("from .workflow import (\n")
	f.write("	mokucola,\n")
	f.write("	mokuup,\n")
	f.write("	mokuani\n")
	f.write(")\n")
	f.write("from .dl import (\n")
	f.write("	dlc,\n")
	f.write("	dlk\n")
	f.write(")\n")
	f.close()

except:
	path=__file__.replace("\\","/")
	path=path.replace("mokucola/__init__.py","basicsr/data/degradations.py")
	f=open(path,"r")
	data=[]
	for line in f:
		if "from torchvision.transforms.functional_tensor import rgb_to_grayscale" in line:
			line=line.replace(
				"from torchvision.transforms.functional_tensor import rgb_to_grayscale",
				"from torchvision.transforms.functional import rgb_to_grayscale"
				)
		data+=[line]
	f.close()
	f=open(path,"w")
	for line in data:
		f.write(line)
	f.close()

	from .workflow import (
		mokucola,
		mokuup,
		mokuani
	)
	from .dl import (
		dlc,
		dlk
	)

	path=__file__.replace("\\","/")
	f=open(path,"w")
	f.write("from .workflow import (\n")
	f.write("	mokucola,\n")
	f.write("	mokuup,\n")
	f.write("	mokuani\n")
	f.write(")\n")
	f.write("from .dl import (\n")
	f.write("	dlc,\n")
	f.write("	dlk\n")
	f.write(")\n")
	f.close()
