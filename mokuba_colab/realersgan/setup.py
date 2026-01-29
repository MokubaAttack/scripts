import os
import torch
path=os.path.dirname(torch.__file__).replace("/torch","/basicsr/data/degradations.py")
url="https://raw.githubusercontent.com/MokubaAttack/scripts/refs/heads/main/mokuba_colab/realersgan/degradations_mod.py"
urlData = requests.get(url).content
with open(path ,mode='wb') as f:
    f.write(urlData)
