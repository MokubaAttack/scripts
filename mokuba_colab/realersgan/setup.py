import py_real_esrgan,os,requests
path=os.path.dirname(py_real_esrgan.__file__)+"/model.py"
url="https://raw.githubusercontent.com/MokubaAttack/scripts/refs/heads/main/mokuba_colab/realersgan/model_mod.py"

urlData = requests.get(url).content
with open(path ,mode='wb') as f:
    f.write(urlData)