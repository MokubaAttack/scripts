# mokubamerge  
DARA merge include randomness. It doesn't feel right. I thought about new method.  
## algorithm  
![algorithm](https://github.com/MokubaAttack/scripts/blob/main/merge_ckpt/mokubamerge/algorithm.jpg)  
## test  
### merge setting  
- **ckpt1** : [Ikastrious Classic v2.0](https://civitai.com/models/1728694?modelVersionId=2056128) (illustrious)  
- **ckpt2** : [Pure Florine v1.0 (No VAE)](https://civitai.com/models/1678041?modelVersionId=1899272) (Pony)  
- **Dropout probability** : 0.5  
- **merge seed** : random  
- **weights of ckpt2**  
  - **case 1**  
base : 0.0, other : 0.1  
  - **case 3**  
base : 0.0, other : 0.3  
  - **case 5**  
base : 0.0, other : 0.5  
  - **case 7**  
base : 0.0, other : 0.7  
  - **case 9**  
base : 0.0, other : 0.9  
### txt2img setting
- **workflow** : mokuba_colab
- **prompt**  
1girl, solo girl, long hair, black hair, hair over one eye, green eyes, upper body, dress, smile, blush, grin, waving, night, masterpiece, high resolution, Stable_Yogis_Anatomy_Positives_V1  
- **negative_prompt**  
bad fingers, extra fingers, liquid finger, missing fingers, nipples, Stable_Yogis_Anatomy_Negatives_V1  
- **guidance_scale** : 6.0  
- **num_inference_steps** : 30  
- **clip_skip** : 2
- **scheduler** : DPM++ 2M beta  
- **interpolation** : BILINEAR  
- **Hires steps** : 25  
- **denoising_strength** : 0.4  
- **pag_scale** : 3.0  
- **size** : ( 600, 800 ) to ( 1200, 1600 )  
- **Embedding**  
[Stable Yogis PDXL Negatives Anatomy_Negatives_v1.0](https://civitai.com/models/1331758?modelVersionId=2044068)  
[Stable Yogis PDXL Positives Anatomy_Positives_v1.0](https://civitai.com/models/1331980?modelVersionId=2044573)  
### result
![result](https://github.com/MokubaAttack/scripts/blob/main/merge_ckpt/mokubamerge/mokubamerge.jpg)
