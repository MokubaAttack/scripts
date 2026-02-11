try:
    from .workflow import (
        mokucola,
        mokuup
    )
    from .dl import dlc
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
    from .dl import dlc
