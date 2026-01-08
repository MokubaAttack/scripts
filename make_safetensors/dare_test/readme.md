# DARA TEST
I examined the difference when I changed weights.
## DARA merge setting
- **ckpt1** : [Ikastrious Classic v2.0](https://civitai.com/models/1728694?modelVersionId=2056128) (illustrious)  
- **ckpt2** : [Pure Florine v1.0 (No VAE)](https://civitai.com/models/1678041?modelVersionId=1899272) (Pony)  
- **Dropout probability** : 0.5  
- **merge seed** : random  
- **weights of ckpt2**  
  - **case 1**  
base : 0.0, other : 0.1  
  - **case 2**  
base : 0.0, other : 0.2  
  - **case 3**  
base : 0.0, other : 0.3  
  - **case 4**  
base : 0.0, other : 0.4  
  - **case 5**  
base : 0.0, other : 0.5  
## txt2img setting
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
