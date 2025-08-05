# scripts
Sorry, I am not good at English.<br>
There are scripts that I use when I create images by diffusers. Scripts that run to open gui and Scripts that you need to rewrite about the input path, the output path etc.<br>
## [get_vae](https://github.com/MokubaAttack/scripts/tree/main/get_vae)
It is a script that extracts a vae safetensors from a checkpoint safetensors. I'm successful in sdxl model.
## [mokuba_colab](https://github.com/MokubaAttack/scripts/tree/main/mokuba_colab)
I make My Own Modules from the workflow that I use whenever I make images. My workflow is based on Hires.fix. That works on Google Colab and Kaggle by torch.float16. And that doesn't work for Flux.
## [lora_mod](https://github.com/MokubaAttack/scripts/tree/main/lora_mod)
When the message "ValueError: Checkpoint not supported because layer lora_unet_label_emb_0_0.alpha not supported." appears,try to run this program. I'm successful in lora of Illustrious model.
