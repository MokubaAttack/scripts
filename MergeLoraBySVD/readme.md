# MergeLoraBySVD
I make [svd_merge_lora.py](https://github.com/kohya-ss/sd-scripts/blob/main/networks/svd_merge_lora.py) of kohya-ss/sd-scripts into a module. <mark>I make it to run by one file, and metadata to be baked in output file.</mark>
## requirements
python modules
```
pip install opencv-contrib-python torch diffusers torchvision accelerate toml transformers einops imagesize
```
~~github repository  
[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)~~
## setup
1. App path of sd-scripts directory to system path.
2. Save MergeLoraBySVD to sd-scripts directory or current directory.
## explanations
**MergeLoraBySVD.merge(**  
&nbsp;&nbsp;&nbsp;&nbsp;
#list of lora file  
&nbsp;&nbsp;&nbsp;&nbsp;
**loras=[],**  
&nbsp;&nbsp;&nbsp;&nbsp;
#list of lora weight  
&nbsp;&nbsp;&nbsp;&nbsp;
**weights=[],**  
&nbsp;&nbsp;&nbsp;&nbsp;
#lbw for each lora   
&nbsp;&nbsp;&nbsp;&nbsp;
**lbws = [],**  
&nbsp;&nbsp;&nbsp;&nbsp;
#precision of calculation ( "float", "fp16", "bf16" )  
&nbsp;&nbsp;&nbsp;&nbsp;
**precision="float",**  
&nbsp;&nbsp;&nbsp;&nbsp;
#precision of output file ( "float", "fp16", "bf16" )  
&nbsp;&nbsp;&nbsp;&nbsp;
**save_precision="fp16",**  
&nbsp;&nbsp;&nbsp;&nbsp;
#dim of LoRA  
&nbsp;&nbsp;&nbsp;&nbsp;
**new_rank=16,**  
&nbsp;&nbsp;&nbsp;&nbsp;
#dim of Conv2d 3x3 LoRA ( When it is None, it is same to new_rank )  
&nbsp;&nbsp;&nbsp;&nbsp;
**new_conv_rank=None,**  
&nbsp;&nbsp;&nbsp;&nbsp;
#When you input "cuda", it calculates by GPU.  
&nbsp;&nbsp;&nbsp;&nbsp;
**device=None,**  
&nbsp;&nbsp;&nbsp;&nbsp;
~~#When you input True, it do not save sai modelspec metadata. ( minimum ss_metadata for LoRA is saved. )~~  
&nbsp;&nbsp;&nbsp;&nbsp;
**~~no_metadata=True,~~**  
&nbsp;&nbsp;&nbsp;&nbsp;
#filename of output  
&nbsp;&nbsp;&nbsp;&nbsp;
**save_to=None**  
**)**
