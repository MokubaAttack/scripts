# scripts
Sorry, I am not good at English.  
There are scripts that I use when I create images by diffusers or make models. Scripts that run to open gui and Scripts that you need to rewrite about the input path, the output path etc are mixed together.  
## [get_vae](https://github.com/MokubaAttack/scripts/tree/main/get_vae)
It is a script that extracts a vae safetensors from a checkpoint safetensors. I'm successful in sdxl model.
## [mokuba_colab](https://github.com/MokubaAttack/scripts/tree/main/mokuba_colab)
I make My Own Modules from the workflow that I use whenever I make images. My workflow is based on Hires.fix. That works on Google Colab and Kaggle by torch.float16. And that doesn't work for Flux.
![flow image](https://github.com/MokubaAttack/scripts/blob/main/mokuba_colab/flow_image.jpg)
## [lora_mod](https://github.com/MokubaAttack/scripts/tree/main/lora_mod)
When the message "ValueError: Checkpoint not supported because layer lora_unet_label_emb_0_0.alpha not supported." appears,try to run this program. I'm successful in lora of Illustrious model.
## [make_embedding](https://github.com/MokubaAttack/scripts/tree/main/make_embedding)
It is a script that make a embedding from text.
## [make_safetensors](https://github.com/MokubaAttack/scripts/tree/main/make_safetensors)
It is a script that burns a vae and loras in a checkpoint. This script is based on [convert_diffusers_to_original_sdxl.py](https://github.com/huggingface/diffusers/blob/main/scripts/convert_diffusers_to_original_sdxl.py) of huggingface/diffusers.
## [merge_ckpt](https://github.com/MokubaAttack/scripts/tree/main/merge_ckpt)
It is a script that merge checkpoints.
## [MergeLoraBySVD](https://github.com/MokubaAttack/scripts/tree/main/MergeLoraBySVD)
I make [svd_merge_lora.py](https://github.com/kohya-ss/sd-scripts/blob/main/networks/svd_merge_lora.py) of kohya-ss/sd-scripts into a module.
## [plus_metadata](https://github.com/MokubaAttack/scripts/tree/main/plus_metadata)
It is a script that write metadata to PNG file. That metadata is recognized in CivitAi.
