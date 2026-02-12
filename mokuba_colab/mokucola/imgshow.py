import math
from diffusers.utils import make_image_grid
from PIL import Image
from IPython.display import display

def imgshow(imgs):
    r=math.ceil(len(imgs)/2)
    if r*2!=len(imgs):
        simgs=imgs+[Image.new('RGB', imgs[-1].size, (0, 0, 0))]
    else:
        simgs=imgs
    display(make_image_grid(simgs, rows=r, cols=2).resize((600,400*r)))
    