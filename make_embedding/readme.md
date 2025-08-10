# make_embedding
It is a script that makes a embedding from text.
## requirements
python modules
```
pip install diffusers==0.34.0
pip install torch==2.7.1
pip install FreeSimpleGUI==5.2.0.post1
pip install transformers==4.55.0
pip install accelerate==1.10.0
```
## How to use
1. Run this script.
2. Select the checkpoint file ( .safetensors file ).
3. Input prompt that you want to make a embedding of.  
   ( Text separated by commas are one concept. For example, "1girl, solo , bleu dress, upper body, looking at viewer" is a embedding that has 5 concepts. )
4. Click run button.
5. A file that ".safetensor" of the checkpoint file turn into "_emb.safetensors" is made.
## the efficacy of prompt strength
seed = 854796723  
prompt = "1girl,solo,blue dress,waving"  
model : Mokubapoint_v4.0  
prompt that I made to embedding  
"bad fingers, extra fingers, liquid finger, missing fingers"

**No Embedding**  
![no embedding](https://github.com/MokubaAttack/scripts/blob/main/make_embedding/noemb.png)

**strength = 0.25**
![strength = 1.0](https://github.com/MokubaAttack/scripts/blob/main/make_embedding/strength025.png)

**strength = 0.5**
![strength = 1.5](https://github.com/MokubaAttack/scripts/blob/main/make_embedding/strength05.png)

**strength = 1.0**
![strength = 2.0](https://github.com/MokubaAttack/scripts/blob/main/make_embedding/strength10.png)
