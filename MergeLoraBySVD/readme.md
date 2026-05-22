# MergeLoraBySVD
I make [svd_merge_lora.py](https://github.com/kohya-ss/sd-scripts/blob/main/networks/svd_merge_lora.py) of kohya-ss/sd-scripts into a module. I make it to run by one file.
## requirements
python modules
```
pip install torch safetensors packaging numpy
```
## explanations
MergeLoraBySVD.merge(

- loras=[],  
  list of lora file
- weights=[],  
  list of lora weight
- lbws = [],  
  lbw for each lora
- precision="float",  
  precision of calculation ( "float", "fp16", "bf16" )
- save_precision="fp16",  
  precision of output file ( "float", "fp16", "bf16" )
- new_rank=16,  
  dim of LoRA
- new_conv_rank=None,  
  dim of Conv2d 3x3 LoRA ( When it is None, it is same to new_rank )
- device=None,  
  When you input "cuda", it calculates by GPU. 
- save_to=None,  
  filename of output
- win=None,  
  window of FreeSimpleGUI
- mem_limit=None,  
  the degree of using memory 
- meta_dict=None  
  the dict of metadata
  
)
## Credits
[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)
