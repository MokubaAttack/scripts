import subprocess

cmd=["pip","install","transformers==5.5.4"]
subprocess.run(cmd)

import sys
import importlib
for k in list(sys.modules.keys()):
	if k.startswith("transformers"):
		try:
			importlib.reload(sys.modules[k])
		except:
			pass

try:
	from .workflow import (
		mokucola,
		mokuup
	)
	from .dl import (
		dlc,
		dlk
	)

	path=__file__.replace("\\","/")
	f=open(path,"w")
	f.write("from .workflow import (\n")
	f.write("	mokucola,\n")
	f.write("	mokuup\n")
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
		mokuup
	)
	from .dl import (
		dlc,
		dlk
	)

	path=__file__.replace("\\","/")
	f=open(path,"w")
	f.write("from .workflow import (\n")
	f.write("	mokucola,\n")
	f.write("	mokuup\n")
	f.write(")\n")
	f.write("from .dl import (\n")
	f.write("	dlc,\n")
	f.write("	dlk\n")
	f.write(")\n")
	f.close()
