from diffusers import StableDiffusionXLPipeline
import torch

ckpt_path=
pipe=StableDiffusionXLPipeline.from_single_file(ckpt_path,torch_dtype=torch.float16)

prompt=
#list of tokenids
prompt_ids = pipe.tokenizer(prompt).input_ids
,return_tensors="pt"
#get tokenid from token
tokenid = pipe.tokenizer.convert_tokens_to_ids(token)

#reverse
for tokenid in tokenids:
  token = pipe.tokenizer.convert_ids_to_tokens(tokenid)
pipe.text_encoder
