# accuracy
It is a script that changes the accuracy of the safetensors file.
## requirements
python modules
```
pip install torch numpy safetensors FreeSimpleGUI
```
## how to use
1. Run this script.
2. Enter path of the safetensors file and select the accuracy theat you want.
3. Click run button.
4. After a while, the output file is generated.  
   output file name
   In case of float16, it is input file name + "_fp16.safetensors".
   In case of bfloat16, it is input file name + "_bf16.safetensors".
