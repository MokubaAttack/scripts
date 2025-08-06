# make_embedding
It is a script that make a embedding from text.
## requirements
python modules
```
pip install diffusers==0.34.0
pip install torch==2.7.1
pip install FreeSimpleGUI==5.2.0.post1
pip install transformers==4.55.0
```
## How to use
writing
## When you don't kwon a lot about python
Please download make_embedding.exe from next link.  
[releases page](https://github.com/MokubaAttack/scripts/releases/tag/make_embedding)
## the efficacy of prompt strength
seed = 854796723  
prompt = "1girl,solo,blue dress,waving"  
model : Mokubapoint_v4.0  
prompt that I made to embedding  
"bad fingers, extra fingers, liquid finger, missing fingers"

**No Embedding**  
![no embedding](https://github.com/MokubaAttack/scripts/blob/main/make_embedding/noemb.png)

**strength = 1.0**
![strength = 1.0](https://github.com/MokubaAttack/scripts/blob/main/make_embedding/strength10.png)

**strength = 1.5**
![strength = 1.5](https://github.com/MokubaAttack/scripts/blob/main/make_embedding/strength20.png)

**strength = 2.0**
![strength = 2.0](https://github.com/MokubaAttack/scripts/blob/main/make_embedding/strength15.png)
