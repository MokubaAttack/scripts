# differense_between_ckpts
I make [extract_lora_from_models.py](https://github.com/kohya-ss/sd-scripts/blob/main/networks/extract_lora_from_models.py) of kohya-ss/sd-scripts into a module. But this script supports SDXL models only. I make it to run by one file.
## requirements
python modules
```
pip install torch diffusers transformers accelerate
```
## explanations
differense_between_ckpts.diff_ckpt(

-	paths=[],  
  list of ckpt file
-	out_path=None,  
  filename of output
-	dim=16,  
  dim of LoRA
-	unet_out=True,  
  whether you output unet difference
-	text1_out=True,  
  whether you output text_encoder difference
-	text2_out=True,  
  whether you output text_encoder2 difference
-	win=None  
  window of FreeSimpleGUI

)
## Credits
- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)
