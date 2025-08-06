# I want to make Embedding.
from diffusers import StableDiffusionXLPipeline
import torch

ckpt_path=
pipe=StableDiffusionXLPipeline.from_single_file(ckpt_path,torch_dtype=torch.float16)

ex_prompt="8k, RAW photo, best quality, masterpiece"
ex_prompt_list=ex_prompt.split(",")
ex_prompt_embs=[]
ex_prompt_embs2=[]

for ex_prompt_elem in ex_prompt_list:
    ex_prompt_elem_ids = pipe.tokenizer(ex_prompt_elem,return_tensors="pt",truncation=False).input_ids
    ex_prompt_elem_tensor = pipe.text_encoder(ex_prompt_elem_ids)[0]
    
    ex_prompt_elem_ids2 = pipe.tokenizer_2(ex_prompt_elem,return_tensors="pt",truncation=False).input_ids
    ex_prompt_elem_tensor2 = pipe.text_encoder_2(ex_prompt_elem_ids2)[0]

    #how to use Nan tensor( Because I don't understand, I postulate that it is a zero tensor. 
    if ex_prompt_elem_tensor.dim()>2:
        ex_prompt_elem_tensor=torch.zeros(1, 768).to(torch.float16)

    #how to use Nan tensor( Because I don't understand, I postulate that it is a zero tensor. 
    if ex_prompt_elem_tensor2.dim()>2:
        ex_prompt_elem_tensor2=torch.zeros(1, 1280).to(torch.float16)
        
    ex_prompt_embs.append(ex_prompt_elem_tensor)
    ex_prompt_embs2.append(ex_prompt_elem_tensor2)
    
clip_l_tensors=torch.cat(ex_prompt_embs, dim=0)
clip_g_tensors=torch.cat(ex_prompt_embs2, dim=0)
