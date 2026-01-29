import basicsr,os
path=os.path.dirname(basicsr.__file__)+"/data/degradations.py"
url="https://raw.githubusercontent.com/MokubaAttack/scripts/refs/heads/main/mokuba_colab/realersgan/degradations_mod.py"
urlData = requests.get(url).content
with open(path ,mode='wb') as f:
    f.write(urlData)
